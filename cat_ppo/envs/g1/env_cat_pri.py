# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union
import jax
import jaxlie
import jax.numpy as jp
from jax.scipy.ndimage import map_coordinates
from jax import lax
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import cat_ppo
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.envs.g1 import constants as consts

ENABLE_RANDOMIZE = False


def g1_loco_task_config() -> config_dict.ConfigDict:
    from cat_ppo.envs.g1.randomize import domain_randomize

    env_config = config_dict.create(
        task_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=15,
        num_obs=175,
        num_pri=209,
        num_act=12,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        gait_config=config_dict.create(
            gait_bound=0.6,
            freq_range=[1.3, 1.5],
            foot_height_range=[0.05, 0.05],
        ),
        dm_rand_config=config_dict.create(
            enable_pd=True,
            kp_range=[0.75, 1.25],
            kd_range=[0.75, 1.25],
            enable_rfi=True,
            rfi_lim=0.1,
            rfi_lim_range=[0.5, 1.5],
            enable_ctrl_delay=False,
            ctrl_delay_range=[0, 2],
        ),
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # behavior reward
                tracking_orientation=2.0,
                tracking_root_field=1.0,
                body_motion=-0.5,
                body_rotation=1.0,
                feet_rotation=1.0,
                foot_contact=-1.0,
                foot_clearance=-15.0, 
                foot_slip=-0.5, 
                foot_balance=-10,
                straight_knee = -30,
                # energy reward
                smoothness_joint=-1e-6,
                smoothness_action=-1e-3,
                joint_limits=-1.0,
                joint_torque=-1e-4,
                # field
                headgf=0.0,
                handsgf=0.0,
                feetgf=0.0,
                headdf=0.0,
                handsdf=0.0,
                feetdf=0.0,
                kneesdf=0.0,
                shldsdf=0.0,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        term_collision_threshold=0.04,
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        command_config=config_dict.create(
            resampling_time=10.0, 
            stop_prob=0.2,
        ),
        lin_vel_x=[-0.5, 0.5],
        lin_vel_y=[-0.3, 0.3],
        ang_vel_yaw=[-0.5, 0.5],
        torso_height=[0.5, consts.DEFAULT_CHEST_Z],
        pf_config=config_dict.create(
            path='data/assets/TypiObs/empty',
            dx=0.04,
            origin=np.array([-0.5, -1.0, 0.0], dtype=np.float32),
        ),
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        # environment wrapper
        num_envs=32768,  # 8192(256*32), 16384(512*32), 32768(1024*32)
        episode_length=1000,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize if ENABLE_RANDOMIZE else None,
        # ppo params
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=1024,  
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        # eval
        num_evals=6,
        eval_env=None,
        num_eval_envs=0,
        deterministic_eval=False,
        # training metrics
        log_training_metrics=True,
        training_metrics_steps=int(1e6),  # 1M
        # callbacks
        progress_fn=lambda *args: None,
        # policy_params_fn=lambda *args: None,
        # checkpointing
        save_checkpoint_path=None,
        restore_checkpoint_path=None, 
        restore_params=None,
        restore_value_fn=False,
    )

    # vel: move_flag[0|1], x[m], y[m], yaw[rad]
    eval_config = config_dict.create(
        duration=50.0,
        command_waypoints=np.array(
            [
                [0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )
    return config

cat_ppo.registry.register("G1CatPri", "config")(g1_loco_task_config())


def base2navi_transform(base2world: jax.Array) -> jax.Array:
    x = base2world[:, 0]
    x_proj = x.at[2].set(0.0)
    x_proj /= jp.linalg.norm(x_proj)
    z_axis = jp.array([0.0, 0.0, 1.0])
    y_axis = jp.cross(z_axis, x_proj)
    y_axis /= jp.linalg.norm(y_axis)
    x_axis = jp.cross(y_axis, z_axis)
    return jp.column_stack((x_axis, y_axis, z_axis))


def torque_step(
        rng: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        qpos_des: jax.Array,
        kps: jax.Array,
        kds: jax.Array,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
        rfi_lim_scale: jax.Array,
        torque_limit: jax.Array,
        n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data]:
    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # pd control
        pos_err = qpos_des - data.qpos[7:]
        vel_err = -data.qvel[6:]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        # rfi noise
        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0)
        torque += rfi_noise

        # clip
        torque = jp.clip(torque, -torque_limit, torque_limit)

        # apply torque
        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]


@cat_ppo.registry.register("G1CatPri", "train_env_class")
class G1CatPriEnv(G1LocoEnv):
    """Track a joystick command."""

    def __init__(
            self,
            task_type: str = "flat_terrain",
            config: config_dict.ConfigDict = None,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        pf_path = config.pf_config.path
        self.dx = config.pf_config.dx
        self.sdf = jp.array(np.load(f"{pf_path}/sdf.npy"))[...,None]   # (Nx,Ny,Nz)
        self.bf  = jp.array(np.load(f"{pf_path}/bf.npy"))    # (Nx,Ny,Nz,3)
        self.gf  = jp.array(np.load(f"{pf_path}/gf.npy"))    # (Nx,Ny,Nz,3)
        self.pf_origin = jp.array(np.array(config.pf_config.origin, dtype=np.float32), dtype=jp.float32)
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape
        self._head_site_id = self._mj_model.site("head").id
        self._knees_site_id = np.array([self._mj_model.site(name).id for name in consts.KNEE_SITES])
        self._shlds_site_id = np.array([self._mj_model.site(name).id for name in consts.SHOULDER_SITES])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q.copy()
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = qpos.at[2].set(0.8)  # 0.8 [m]

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-np.pi / 2, maxval=np.pi / 2)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # qpos[7:]=*U(0.5, 1.5)
        rng, key = jax.random.split(rng)
        rand_qpos = qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
        rand_qpos = jp.clip(rand_qpos, self._soft_lowers, self._soft_uppers)
        qpos = qpos.at[7:].set(rand_qpos)

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        rtf = self.sample_field(self.gf, data.site_xpos[self._pelvis_imu_site_id].reshape(1, -1)).reshape(-1)
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = jp.zeros_like(head_pos)
        headgf = self.sample_field(self.gf, head_pos.reshape(1, -1))
        headbf = self.sample_field(self.bf, head_pos.reshape(1, -1))
        headdf = self.sample_field(self.sdf, head_pos.reshape(1, -1))
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = jp.zeros_like(feet_pos)
        feetgf = self.sample_field(self.gf, feet_pos)
        feetbf = self.sample_field(self.bf, feet_pos)
        feetdf = self.sample_field(self.sdf, feet_pos)
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = jp.zeros_like(hands_pos)
        handsgf = self.sample_field(self.gf, hands_pos)
        handsbf = self.sample_field(self.bf, hands_pos)
        handsdf = self.sample_field(self.sdf, hands_pos)
        knees_pos = data.site_xpos[self._knees_site_id]
        kneesbf = self.sample_field(self.bf, knees_pos)
        kneesdf = self.sample_field(self.sdf, knees_pos)
        shlds_pos = data.site_xpos[self._shlds_site_id]
        shldsbf = self.sample_field(self.bf, shlds_pos)
        shldsdf = self.sample_field(self.sdf, shlds_pos)
        command = self.compute_cmd_from_rtf(rtf, jp.concat([headgf, feetgf, handsgf], axis=0), jp.concat([headbf, feetbf, handsbf], axis=0))

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        # gait
        # Phase, freq=U(1.0, 1.5)
        rng, gait_freq_rng, foot_height_rng = jax.random.split(rng, 3)
        gait_freq = jax.random.uniform(
            gait_freq_rng,
            minval=self._config.gait_config.freq_range[0],
            maxval=self._config.gait_config.freq_range[1],
        )
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        rng, phase_rng = jax.random.split(rng)
        cond_phase = jax.random.bernoulli(phase_rng)
        phase = jp.where(cond_phase, self._init_phase_l, self._init_phase_r)
        # phase = self._init_phase_l.copy()
        foot_height = jax.random.uniform(
            foot_height_rng,
            minval=self._config.gait_config.foot_height_range[0],
            maxval=self._config.gait_config.foot_height_range[1],
        )

        # randomize torque
        rng, key_kp, key_kd, key_rfi, key_delay = jax.random.split(rng, 5)
        kp_scale = jax.random.uniform(
            key_kp,
            minval=self._config.dm_rand_config.kp_range[0],
            maxval=self._config.dm_rand_config.kp_range[1],
        )
        kp_scale = jp.where(self._config.dm_rand_config.enable_pd, kp_scale, jp.ones_like(kp_scale))

        kd_scale = jax.random.uniform(
            key_kd,
            minval=self._config.dm_rand_config.kd_range[0],
            maxval=self._config.dm_rand_config.kd_range[1],
        )
        kd_scale = jp.where(self._config.dm_rand_config.enable_pd, kd_scale, jp.ones_like(kd_scale))

        rfi_lim_noise_scale = jax.random.uniform(
            key_rfi,
            self.torque_limit.shape,
            minval=self._config.dm_rand_config.rfi_lim_range[0],
            maxval=self._config.dm_rand_config.rfi_lim_range[1],
        )
        rfi_lim_scale = self._config.dm_rand_config.rfi_lim * rfi_lim_noise_scale * self.torque_limit
        rfi_lim_scale = jp.where(self._config.dm_rand_config.enable_rfi, rfi_lim_scale, jp.zeros_like(rfi_lim_scale))

        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            # history
            "last_command": jp.zeros(4),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "last_feet_vel": jp.zeros(2),
            "last_joint_vel": np.zeros(self.num_joints),
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # state
            "motor_targets": self._default_qpos.copy(),
            "local_lin_vel": jp.zeros(3),
            "global_lin_vel": jp.zeros(3),
            "global_ang_vel": jp.zeros(3),
            "navi2world_rot": jp.eye(3),
            "navi2world_pose": jp.eye(4),
            "navi_torso_rpy": jp.zeros(3),
            "navi_torso_lin_vel": jp.zeros(3),
            "navi_torso_ang_vel": jp.zeros(3),
            "navi_pelvis_rpy": jp.zeros(3),
            "navi_pelvis_lin_vel": jp.zeros(3),
            "navi_pelvis_ang_vel": jp.zeros(3),
            # Phase related.
            "stop_timestep": 100,
            "phase": phase,
            "phase_dt": phase_dt,
            "gait_mask": jp.zeros(2),
            "gait_freq": gait_freq,
            "foot_height": foot_height,
            # domain randomization
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "rfi_lim_scale": rfi_lim_scale,
            "rtf": rtf.copy(),
            "headgf": headgf.copy(),
            "headbf": headbf.copy(),
            "headdf": headdf.copy(),
            "feetgf": feetgf.copy(),
            "feetbf": feetbf.copy(),
            "feetdf": feetdf.copy(),
            "handsgf": handsgf.copy(),
            "handsbf": handsbf.copy(),
            "handsdf": handsdf.copy(),
            "kneesbf": kneesbf.copy(),
            "kneesdf": kneesdf.copy(),
            "shldsbf": shldsbf.copy(),
            "shldsdf": shldsdf.copy(),
            "head_pos": head_pos.copy(),
            "head_vel": head_vel.copy(),
            "feet_pos": feet_pos.copy(),
            "feet_vel": feet_vel.copy(),
            "hands_pos": hands_pos.copy(),
            "hands_vel": hands_vel.copy(),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push_signal = jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= push_signal
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + push * push_magnitude)
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # set motor target
        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)

        state.info["rng"], data = torque_step(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=state.info["rfi_lim_scale"],
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        # collect info
        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        state.info["motor_targets"] = motor_targets
        state.info["local_lin_vel"] = self.get_local_linvel(data, "pelvis")
        state.info["global_lin_vel"] = self.get_global_linvel(data, "pelvis")
        state.info["global_ang_vel"] = self.get_global_angvel(data, "pelvis")

        # navi frame
        pelvis2world_rot = data.site_xmat[self._pelvis_imu_site_id]
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:3, :3].set(navi2world_rot)
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[:2, 3].set(data.site_xpos[self._pelvis_imu_site_id][:2])
        )
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[2, 3].set(self._config.reward_config.base_height_target)
        )

        # pelvis projection
        pelvis2navi_rot = navi2world_rot.T @ pelvis2world_rot
        state.info["navi2world_rot"] = navi2world_rot
        state.info["navi_pelvis_rpy"] = jp.array(jaxlie.SO3.from_matrix(pelvis2navi_rot).as_rpy_radians())
        state.info["navi_pelvis_lin_vel"] = pelvis2navi_rot @ self.get_local_linvel(data, "pelvis")
        state.info["navi_pelvis_ang_vel"] = pelvis2navi_rot @ self.get_gyro(data, "pelvis")
        # torso projection
        torso2world_rot = data.site_xmat[self._torso_imu_site_id]
        torso2navi_rot = navi2world_rot.T @ torso2world_rot
        state.info["navi_torso_rpy"] = jp.array(jaxlie.SO3.from_matrix(torso2navi_rot).as_rpy_radians())
        state.info["navi_torso_lin_vel"] = torso2navi_rot @ self.get_local_linvel(data, "torso")
        state.info["navi_torso_ang_vel"] = torso2navi_rot @ self.get_gyro(data, "torso")

        state.info["last_command"] = state.info["command"].copy()
        rtf = self.sample_field(self.gf, data.site_xpos[self._pelvis_imu_site_id].reshape(1, -1)).reshape(-1)
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = (head_pos - state.info["head_pos"]) / self.dt
        headgf = self.sample_field(self.gf, head_pos.reshape(1, -1))
        headbf = self.sample_field(self.bf, head_pos.reshape(1, -1))
        headdf = self.sample_field(self.sdf, head_pos.reshape(1, -1))
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = (feet_pos - state.info["feet_pos"]) / self.dt
        feetgf = self.sample_field(self.gf, feet_pos)
        feetbf = self.sample_field(self.bf, feet_pos)
        feetdf = self.sample_field(self.sdf, feet_pos)
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = (hands_pos - state.info["hands_pos"]) / self.dt
        handsgf = self.sample_field(self.gf, hands_pos)
        handsbf = self.sample_field(self.bf, hands_pos)
        handsdf = self.sample_field(self.sdf, hands_pos)
        knees_pos = data.site_xpos[self._knees_site_id]
        kneesbf = self.sample_field(self.bf, knees_pos)
        kneesdf = self.sample_field(self.sdf, knees_pos)
        shlds_pos = data.site_xpos[self._shlds_site_id]
        shldsbf = self.sample_field(self.bf, shlds_pos)
        shldsdf = self.sample_field(self.sdf, shlds_pos)

        command = self.compute_cmd_from_rtf(rtf, jp.concat([headgf,feetgf,handsgf], axis=0), jp.concat([headbf,feetbf,handsbf], axis=0))
        state.info["rtf"] = rtf.copy()
        state.info["headgf"] = headgf.copy()
        state.info["headbf"] = headbf.copy()
        state.info["headdf"] = headdf.copy()
        state.info["feetgf"] = feetgf.copy()
        state.info["feetbf"] = feetbf.copy()
        state.info["feetdf"] = feetdf.copy()
        state.info["handsgf"] = handsgf.copy()
        state.info["handsbf"] = handsbf.copy()
        state.info["handsdf"] = handsdf.copy()
        state.info["kneesbf"] = kneesbf.copy()
        state.info["kneesdf"] = kneesdf.copy()
        state.info["shldsbf"] = shldsbf.copy()
        state.info["shldsdf"] = shldsdf.copy()

        state.info["head_pos"] = head_pos.copy()
        state.info["head_vel"] = head_vel.copy()
        state.info["feet_pos"] = feet_pos.copy()
        state.info["feet_vel"] = feet_vel.copy()
        state.info["hands_pos"] = hands_pos.copy()
        state.info["hands_vel"] = hands_vel.copy()
        state.info["command"] = command
        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1

        # update gait
        self._update_phase(state)

        # update history
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)


        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])

        state.info["motor_targets"] = jp.where(
            done, self._default_qpos, state.info["motor_targets"]
        )
        # ransom
        state.info["rng"], episode_rng = jax.random.split(state.info["rng"])

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_joint_vel"] = data.qvel[6:].copy()
        state.info["last_feet_vel"] = data.sensordata[self._foot_linvel_sensor_adr][..., 2]
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _update_phase(self, state):
        task_mask = state.info["command"][0]
        last_task_mask = state.info["last_command"][0]

        stop_timestep = state.info["stop_timestep"]
        before_stop = stop_timestep > 50
        during_stop = (~before_stop) & (stop_timestep > 0)
        after_stop = (~before_stop) & (~during_stop)
        move2stop = (last_task_mask == 1.0) & (task_mask == 0.0) & before_stop
        stop_timestep = jp.where(move2stop, 50, stop_timestep)
        stop_timestep = jp.where(during_stop, stop_timestep-1, stop_timestep)
        state.info["stop_timestep"] = stop_timestep
        command = jp.where(before_stop, state.info["command"], self._stop_cmd)
        command=command.at[0].set(jp.where(after_stop, 0.0, 1.0))
        state.info["command"] = command

        phase = state.info["phase"] + state.info["phase_dt"]
        phase = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
        phase = jp.where(after_stop, self._stance_phase, phase)
        state.info["phase"] = phase

        # gait flag
        gait_cycle = jp.cos(phase)
        gait_mask = jp.where(gait_cycle > self._gait_bound, 1, 0)
        gait_mask = jp.where(gait_cycle < -self._gait_bound, -1, gait_mask)
        state.info["gait_mask"] = jp.float32(gait_mask)

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= info["head_pos"][2] < 0.7
        contact_termination = collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_foot_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._left_foot_geom_id,
            self._right_shin_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_shin_geom_id,
        )
        contact_termination |= jp.any(info['headdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['feetdf'] < -self._config.term_collision_threshold)
        contact_termination |= jp.any(info['handsdf'] < -self._config.term_collision_threshold)
        contact_termination &= (info["step"] >= 100)
        return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | timeout

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        # body pose
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        # joint
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        headgf = info["headgf"].copy()
        headbf = info["headbf"].copy()
        headdf = info["headdf"].copy()
        feetgf = info["feetgf"].copy()
        feetbf = info["feetbf"].copy()
        feetdf = info["feetdf"].copy()
        handsgf= info["handsgf"].copy()
        handsbf= info["handsbf"].copy()
        handsdf= info["handsdf"].copy()
        kneesbf = info["kneesbf"].copy()
        kneesdf = info["kneesdf"].copy()
        shldsbf = info["shldsbf"].copy()
        shldsdf = info["shldsdf"].copy()
        head_pos = info["head_pos"].copy()
        head_vel = info["head_vel"].copy()
        feet_pos = info["feet_pos"].copy()
        feet_vel = info["feet_vel"].copy()
        hands_pos = info["hands_pos"].copy()
        hands_vel = info["hands_vel"].copy()
        command = info["command"].copy()

        privileged_state = jp.hstack(
            [
                # noiseless state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                (joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                joint_vel[self.obs_joint_ids],  # 23
                info["last_act"],  # num_actions
                info["motor_targets"][self.action_joint_ids],  # num_actions
                command,  # 4
                info["foot_height"],  # 1
                gait_phase,  # (num_foot * 2)
                # hint state
                linvel_pelvis,  # 3
                info["rtf"],
                headgf.reshape(-1),
                headbf.reshape(-1),
                headdf.reshape(-1),
                feetgf.reshape(-1),
                feetbf.reshape(-1),
                feetdf.reshape(-1),
                handsgf.reshape(-1),
                handsbf.reshape(-1),
                handsdf.reshape(-1),
                kneesbf.reshape(-1),
                kneesdf.reshape(-1),
                shldsbf.reshape(-1),
                shldsdf.reshape(-1),
                head_pos.reshape(-1),
                head_vel.reshape(-1),
                feet_pos.reshape(-1),
                feet_vel.reshape(-1),
                hands_pos.reshape(-1),
                hands_vel.reshape(-1),
                info["navi_torso_rpy"][:2],
                info["gait_mask"],
                feet_contact,  # num_foot
                # domain randomization
                info["kp_scale"],
                info["kd_scale"],
                info["rfi_lim_scale"],
            ]
        )
        state = jp.hstack(
            [
                # noiseless state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                (joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                joint_vel[self.obs_joint_ids],  # 23
                info["last_act"],  # num_actions
                info["motor_targets"][self.action_joint_ids],  # num_actions
                command,  # 4
                info["foot_height"],  # 1
                gait_phase,  # (num_foot * 2)
                # hint state
                linvel_pelvis,  # 3
                headgf.reshape(-1),
                headbf.reshape(-1),
                headdf.reshape(-1),
                feetgf.reshape(-1),
                feetbf.reshape(-1),
                feetdf.reshape(-1),
                handsgf.reshape(-1),
                handsbf.reshape(-1),
                handsdf.reshape(-1),
                kneesbf.reshape(-1),
                kneesdf.reshape(-1),
                shldsbf.reshape(-1),
                shldsdf.reshape(-1),
                head_pos.reshape(-1),
                head_vel.reshape(-1),
                feet_pos.reshape(-1),
                feet_vel.reshape(-1),
                hands_pos.reshape(-1),
                hands_vel.reshape(-1),
                info["navi_torso_rpy"][:2],
                info["gait_mask"],
                feet_contact,  # num_foot
                # domain randomization
                # info["kp_scale"],
                # info["kd_scale"],
                # info["rfi_lim_scale"],
                # info["delay_steps"],
            ]
        )

        # Nan to 0
        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)

        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        move_flag = info["command"][0]
        cmd_vel = info["command"][1:].copy()  # [x, y, yaw]

        reward_dict = {
            # behavior reward
            "tracking_orientation": self._reward_orientation(
                info["navi_pelvis_rpy"], info["navi_torso_rpy"], info["head_pos"][2] > (self._config.torso_height[1] + 0.1)
            ),
            "tracking_root_field": self._reward_tracking_root_field(cmd_vel, info["global_lin_vel"]),
            "body_motion": self._cost_body_motion(info["global_lin_vel"], info["navi_torso_ang_vel"], cmd_vel), # TODO
            "body_rotation": self._reward_body_rotation(data, cmd_vel, info["navi2world_rot"]),
            "feet_rotation": self._reward_feet_rotation(data, info["navi2world_rot"]),
            "foot_contact": self._cost_foot_contact(data, feet_contact, info["gait_mask"], move_flag),
            "foot_clearance": self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], move_flag),
            "foot_slip": self._cost_foot_slip(data, info["gait_mask"]),
            "foot_balance": self._cost_foot_balance(data, info["navi2world_pose"], move_flag),
            "straight_knee": self._cost_straight_knee(data.qpos[jp.array(self._knee_indices) + 7]),
            # energy reward
            "joint_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "joint_torque": self._cost_torque(data.actuator_force),
            "smoothness_joint": self._cost_smoothness_joint(data, info["last_joint_vel"]),
            "smoothness_action": self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
            # field
            "headgf": self._re_gf0(info["headgf"], info["head_vel"], info["headdf"], move_flag[None]<0.5, tau=0.5),
            "feetgf": self._re_gf0(info["feetgf"], info["feet_vel"], info["feetdf"], (move_flag[None]<0.5) | (info["gait_mask"] == 1), tau=0.3),
            "handsgf": self._re_gf0(info["handsgf"], info["hands_vel"], info["handsdf"], move_flag[None]<0.5, tau=0.5),
            "headdf": self._re_sdf(info["headdf"]),
            "feetdf": self._re_sdf(info["feetdf"]),
            "handsdf": self._re_sdf(info["handsdf"],), 
            "kneesdf": self._re_sdf(info["kneesdf"]),
            "shldsdf": self._re_sdf(info["shldsdf"]),
        }
        for k, v in reward_dict.items():
            # replace NaN with 0
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)

        return reward_dict

    def _re_gf0(self, gf_vel: jax.Array, lin_vel: jax.Array, sdf: jax.Array, crossed: jax.Array, tau = 0.3) -> jax.Array:
        eps = 1e-6

        tau         = tau 
        k_window    = 40.0 
        alpha_align = 5.0   

        g_norm = gf_vel / (jp.linalg.norm(gf_vel, axis=-1, keepdims=True) + eps)
        v_norm = lin_vel / (jp.linalg.norm(lin_vel, axis=-1, keepdims=True) + eps)
        cos_align = jp.sum(g_norm * v_norm, axis=-1) 

        sdf_flat = sdf.reshape(-1)
        window = jax.nn.sigmoid(k_window * (tau - sdf_flat)) 

        reward_near = window * (alpha_align * cos_align) 
        reward_near = jp.where(crossed, alpha_align*0.6, reward_near)

        return jp.mean(reward_near) 

    def _re_sdf(self, sdf: jax.Array, sdf_safe = 0.05) -> jax.Array:
        beta_inside = 0.02  
        pen_inside_scale   = 20.0   

        sdf_flat = sdf.reshape(-1)
        pen_inside = jax.nn.softplus((sdf_safe - sdf_flat)/ beta_inside)

        penalty = (pen_inside_scale * pen_inside) 

        re_gf = - penalty

        return jp.mean(re_gf)  # 

    def _reward_tracking_root_field(self, cmd_vel: jax.Array, local_lin_vel: jax.Array) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(cmd_vel[:2] - local_lin_vel[:2]))
        return jp.exp(-4.0 * lin_vel_error)
    
    def _cost_body_motion(
        self, local_lin_vel, local_ang_vel: jax.Array, cmd_vel: jax.Array
    ) -> jax.Array:
        cmd_xy = cmd_vel[:2]
        cmd_norm = jp.linalg.norm(cmd_xy)
        is_zero_cmd = jp.isclose(cmd_norm, 0.0)
        cmd_dir = jp.where(is_zero_cmd, jp.zeros_like(cmd_xy), cmd_xy / cmd_norm)

        lin_xy = local_lin_vel[:2]
        lin_xy_orth = lin_xy - jp.dot(lin_xy, cmd_dir) * cmd_dir
        cost_lin_xy_orth = jp.where(is_zero_cmd, 0.0, jp.sum(jp.square(lin_xy_orth)))

        cost = (
            1.2 * cost_lin_xy_orth
            + 0.4 * jp.abs(local_ang_vel[0])
            + 0.4 * jp.abs(local_ang_vel[1])
        )
        return cost

    def _reward_feet_rotation(
        self, data: mjx.Data, navi2world_rot: jax.Array
    ) -> jax.Array:
        knees2world_rot = jp.concat(
            [
                data.xmat[self.body_id_knee_l][None],
                data.xmat[self.body_id_knee_r][None],
            ]
        )
        knees2navi_rot = navi2world_rot.T[None] @ knees2world_rot
        ankles2world_rot = jp.concat(
            [
                data.xmat[self.body_id_ankle_l][None],
                data.xmat[self.body_id_ankle_r][None],
            ]
        )
        ankles2navi_rot = navi2world_rot.T[None] @ ankles2world_rot

        knees_roll_err = jp.sum(jp.abs(knees2navi_rot[:, 2, 1]))
        knees_yaw_err = jp.sum(jp.abs(knees2navi_rot[:, 0, 1]))
        ankles_roll_err = jp.sum(jp.abs(ankles2navi_rot[:, 1, 2]))
        ankles_pitch_err = jp.sum(jp.abs(ankles2navi_rot[:, 0, 2]))
        ankles_yaw_err = jp.sum(jp.square(ankles2navi_rot[:, 0, 1]))

        axis_rew = jp.exp(
            -1.0
            * (
                knees_roll_err
                + knees_yaw_err
                + ankles_roll_err
                + ankles_pitch_err
                + ankles_yaw_err
            )
        )
        return axis_rew

    def _reward_orientation(
        self, pelvis_rpy: jax.Array, torso_rpy: jax.Array, idle_mask: jax.Array
    ) -> jax.Array:
        err_roll = jp.abs(pelvis_rpy[0]) + jp.abs(torso_rpy[0])
        err_pitch_dire = jp.abs(jp.clip(torso_rpy[1], -np.pi, 0.0))
        err_pitch_idle = idle_mask * jp.abs(torso_rpy[1])
        err_ori = err_roll + err_pitch_dire + err_pitch_idle
        rew = jp.exp(-0.5 * err_ori) - err_pitch_dire
        return rew
    
    def _reward_facing(
        self, cmd_vel: jax.Array, local_lin_vel: jax.Array, world_rpy: jax.Array, ideal_yaw: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(cmd_vel[:2] - local_lin_vel[:2]))
        err_yaw = jp.abs(world_rpy[2] - ideal_yaw)
        rew = jp.exp(-0.5 * err_yaw) * (0.0 + 1.0 * jp.exp(-4.0 * lin_vel_error))
        return rew
    
    def _cost_foot_far(self, data: mjx.Data) -> jax.Array:
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_distance = jp.linalg.norm(foot_pos[0] - foot_pos[1])
        foot_spread_penalty = jp.where(
            foot_distance < 0.35,
            (0.35 - foot_distance),
            0.0
        )
        return foot_spread_penalty

    def _cost_straight_knee(self, knee_pos) -> jax.Array:
        penalty = jp.clip(0.1 - knee_pos, min = 0.0) # NOTE
        cost = jp.sum(penalty)
        return cost

    def _cost_foot_balance(
        self, data: mjx.Data, navi2world_pose: jax.Array, task_mask: jax.Array
    ):
        stance_mask = 1 - task_mask
        sup2world_pos_h = jp.ones((3, 4))
        sup2world_pos_h = sup2world_pos_h.at[0, :3].set(
            data.subtree_com[self.body_id_pelvis]
        )
        sup2world_pos_h = sup2world_pos_h.at[1, :3].set(
            data.site_xpos[self._feet_site_id[0]]
        )
        sup2world_pos_h = sup2world_pos_h.at[2, :3].set(
            data.site_xpos[self._feet_site_id[1]]
        )
        sup2navi_pos = (jp.linalg.inv(navi2world_pose) @ sup2world_pos_h.T).T[:, :3]

        foot2com_err = sup2navi_pos[1:] - sup2navi_pos[0]
        foot_center = foot2com_err[0, :2] + foot2com_err[1, :2]  # ignore z-axis
        cost_support = jp.sum(jp.square(foot_center))
        cost_support *= stance_mask
        return cost_support
    
    def _cost_smoothness_action(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        smooth_0th = jp.square(act)
        smooth_1st = jp.square(act - last_act)
        smooth_2nd = jp.square(act - 2 * last_act + last_last_act)
        cost = jp.sum(smooth_0th + smooth_1st + smooth_2nd)
        return cost

    def _reward_body_rotation(self, data: mjx.Data, cmd_vel: jax.Array, navi2world_rot: jax.Array) -> jax.Array:
        cmd_max = jp.abs(self._config.ang_vel_yaw[1]) + 1e-6
        cmd_decay = jp.clip((cmd_max - jp.abs(cmd_vel[2])) / cmd_max, 0.0, 1.0) ** 2
        legs2world_rot = jp.concat([data.xmat[self.body_ids_left_leg], data.xmat[self.body_ids_right_leg]])
        legs2navi_rot = navi2world_rot.T[None] @ legs2world_rot  # (N, 3, 3)
        axis_roll_err = jp.mean(jp.abs(legs2navi_rot[:, 2, 1]))
        axis_rew = jp.exp(-5.0 * axis_roll_err)
        return axis_rew

    def world_to_grid(self, pos):
        """ 世界坐标 -> voxel index (浮点) """
        rel = pos - self.pf_origin
        idx = rel / self.dx
        return idx

    def sample_field(self, field, pos):
        idx = self.world_to_grid(pos)                  # (N,3)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]     # (N,)

        x = jp.clip(x, 0, self.Nx - 2)
        y = jp.clip(y, 0, self.Ny - 2)
        z = jp.clip(z, 0, self.Nz - 2)

        xi = jp.floor(x).astype(jp.int32)             # (N,)
        yi = jp.floor(y).astype(jp.int32)
        zi = jp.floor(z).astype(jp.int32)
        xd = x - xi                                    # (N,)
        yd = y - yi
        zd = z - zi

        offsets = jp.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1]
        ], dtype=jp.int32)                             # (8,3)

        base = jp.stack([xi, yi, zi], axis=1)         # (N,3)
        corners = base[:, None, :] + offsets[None, :, :]     # (N,8,3)

        vals = field[corners[..., 0], corners[..., 1], corners[..., 2], :]  # (N,8,C)

        wx = jp.stack([1.0 - xd, xd], axis=1)         # (N,2)
        wy = jp.stack([1.0 - yd, yd], axis=1)         # (N,2)
        wz = jp.stack([1.0 - zd, zd], axis=1)         # (N,2)

        w = (wx[:, :, None, None] *
            wy[:, None, :, None] *
            wz[:, None, None, :]).reshape(-1, 8)      # (N,8)

        out = jp.einsum('ne,nec->nc', w, vals)        # (N,C)
        return out

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):
        # reuse command in velocity-control locomotion for our HumanoidPF, can be seen as a single iteration of field projection

        v = rtf[:2] * 0.6 

        bnorm = jp.linalg.norm(cbf[:, :2], axis=-1, keepdims=True) + 1e-9
        b_hat = cbf[:, :2] / bnorm  # (M,2)

        Ls = jp.sum(b_hat * cgf[:, :2], axis=-1)  # (M,)

        bv = jp.sum(b_hat * v, axis=-1)           # (M,)

        diff = (Ls - bv)[:, None] / (jp.sum(b_hat * b_hat, axis=-1, keepdims=True) + 1e-9)
        delta = diff * b_hat  # (M,2)

        mask = (Ls > bv)[:, None]  # (M,1)
        delta = jp.where(mask, delta, 0.0)

        v_new = v + jp.mean(delta, axis=0)

        command = jp.hstack([1.0, v_new[0], v_new[1], 0.0])

        small_cond = jp.linalg.norm(command[1:4]) < 0.2
        command = jp.where(small_cond, self._stop_cmd, command)
        return command
    
@cat_ppo.registry.register("G1CatPri", "command_to_reference_fn")
def command_to_reference(env_config: config_dict.ConfigDict, command: jax.Array):
    command_vel = command[1:]
    base_height = env_config.reward_config.base_height_target
    base_gvec = np.array([0.0, 0.0, 1.0])
    base_lin_vel = np.array([command_vel[0], command_vel[1], 0.0])
    base_ang_vel = np.array([0.0, 0.0, command_vel[2]])

    return {
        "base_height": base_height,
        "base_gvec": base_gvec,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
    }
