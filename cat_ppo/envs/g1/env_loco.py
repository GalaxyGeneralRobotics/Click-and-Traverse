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
from jax import lax
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import cat_ppo
from cat_ppo.envs.g1 import base as g1_base
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
        num_obs=85,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        gait_config=config_dict.create(
            gait_bound=0.6,  # soft constraint ratio 0~1
            freq_range=[1.0, 1.25],
            foot_height_range=[0.05, 0.1],
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
                base_orientation=3.0,
                base_height=1.0,
                # Tracking related rewards.
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.75,
                # behavior reward
                body_motion=-1.0,
                body_rotation=0.5,
                foot_contact=-0.5,
                foot_clearance=-15.0,
                foot_slip=-0.1,
                foot_balance=-10,
                foot_far=-3.0,
                # energy reward
                smoothness_joint=-1e-6,
                smoothness_action=-0.01,
                joint_limits=-1.0,
                joint_torque=-1e-4,
                # joint_nominal=-0.01,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        command_config=config_dict.create(
            resampling_time=10.0,  # command changed time [s]
            stop_prob=0.2,
        ),
        lin_vel_x=[-0.5, 0.5],
        lin_vel_y=[-0.3, 0.3],
        ang_vel_yaw=[-0.5, 0.5],
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        # high-level control flow
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
        batch_size=1024,  # 256, 512, 1024
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
        restore_value_fn=True,
    )

    # vel: move_flag[0|1], x[m], y[m], yaw[rad]
    eval_config = config_dict.create(
        duration=50.0,
        command_waypoints=np.array(
            [
                [0, 0.0, 0.0, 0.0],
                [1, 0.5, 0.0, 0.0],
                [1, 1.0, 0.0, 0.0],
                [0, 0.0, 0.0, 0.0],
                [1, -1.0, 0.0, 1.0],
                [1, 0.0, 0.4, -0.7],
                [0, 0.0, 0.0, 0.0],
                [1, 0.5, -0.5, 0.5],
                [0, 0.0, 0.0, 0.0],
                [1, 0.6, -0.4, 0.5],
                [1, 0.0, 0.0, 1.0],
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


cat_ppo.registry.register("G1Loco", "config")(g1_loco_task_config())


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


@cat_ppo.registry.register("G1Loco", "train_env_class")
@cat_ppo.registry.register("G1LocoDis", "train_env_class")
class G1LocoEnv(g1_base.G1Env):
    """Track a joystick command."""

    @property
    def action_size(self) -> int:
        return len(self.action_joint_names)

    def __init__(
            self,
            task_type: str = "flat_terrain",
            config: config_dict.ConfigDict = None,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task_type).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self.num_joints = self.mjx_model.nq - 7
        self.episode_length = self._config.episode_length

        self.action_joint_names = consts.ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = []
        for j_name in self.action_joint_names:
            self.action_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.action_joint_ids = jp.array(self.action_joint_ids)

        self.obs_joint_names = consts.OBS_JOINT_NAMES.copy()
        self.obs_joint_ids = []
        for j_name in self.obs_joint_names:
            self.obs_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.obs_joint_ids = jp.array(self.obs_joint_ids)

        self._up_vec = jp.array([0.0, 0.0, 1.0])
        # self._up_vec_torso = jp.array([0.073, 0.0, 1.0])
        self._left_vec = jp.array([0.0, 1.0, 0.0])
        self._gait_bound = self._config.gait_config.gait_bound
        self._init_phase_l = jp.array([0, np.pi])  # swing left first
        self._init_phase_r = jp.array([np.pi, 0])  # swing right first
        self._stance_phase = jp.array([0.0, 0.0])
        self._stop_cmd = jp.array([0.0, 0.0, 0.0, 0.0])
        self._init_q = jp.array(consts.DEFAULT_QPOS)
        self._default_qpos = jp.array(consts.DEFAULT_QPOS[7:])

        # Note: First joint is freejoint.
        self._kps = jp.array(consts.KPs)
        self._kds = jp.array(consts.KDs)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        waist_indices = []
        waist_joint_names = ["waist_yaw", "waist_roll", "waist_pitch"]
        for joint_name in waist_joint_names:
            waist_indices.append(self._mj_model.joint(f"{joint_name}_joint").qposadr - 7)
        self._waist_indices = jp.array(waist_indices)

        arm_indices = []
        arm_joint_names = ["shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        for side in ["left", "right"]:
            for joint_name in arm_joint_names:
                arm_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._arm_indices = jp.array(arm_indices)

        hip_indices = []
        hip_joint_names = ["hip_roll", "hip_yaw"]
        for side in ["left", "right"]:
            for joint_name in hip_joint_names:
                hip_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._hip_indices = jp.array(hip_indices)

        hip_pitch_indices = []
        hip_pitch_joint_names = ["hip_pitch"]
        for side in ["left", "right"]:
            for joint_name in hip_pitch_joint_names:
                hip_pitch_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._hip_pitch_indices = jp.array(hip_pitch_indices)

        knee_indices = []
        knee_joint_names = ["knee"]
        for side in ["left", "right"]:
            for joint_name in knee_joint_names:
                knee_indices.append(self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7)
        self._knee_indices = jp.array(knee_indices)

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._torso_imu_site_id = self._mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

        self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        self._hands_site_id = np.array([self._mj_model.site(name).id for name in consts.HAND_SITES])
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._cmd_resample_steps = int(self._config.command_config.resampling_time / self.dt)
        self._cmd_stop_prob = self._config.command_config.stop_prob

        self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
        self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
        self._left_foot_geom_id = self._mj_model.geom("left_foot").id
        self._right_foot_geom_id = self._mj_model.geom("right_foot").id
        self._left_shin_geom_id = self._mj_model.geom("left_shin").id
        self._right_shin_geom_id = self._mj_model.geom("right_shin").id
        self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
        self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id

        # bodies
        self.body_id_pelvis = self.mj_model.body("pelvis").id
        self.body_id_torso = self.mj_model.body("torso_link").id
        self.body_names_left_leg = ["left_knee_link", "left_ankle_roll_link"]
        self.body_ids_left_leg = jp.array([self.mj_model.body(n).id for n in self.body_names_left_leg])
        self.body_names_right_leg = ["right_knee_link", "right_ankle_roll_link"]
        self.body_ids_right_leg = jp.array([self.mj_model.body(n).id for n in self.body_names_right_leg])
        self.body_id_knee_l = self.mj_model.body("left_knee_link").id
        self.body_id_knee_r = self.mj_model.body("right_knee_link").id
        self.body_id_ankle_l = self.mj_model.body("left_ankle_roll_link").id
        self.body_id_ankle_r = self.mj_model.body("right_ankle_roll_link").id

        
        self.torque_limit = jp.array(consts.TORQUE_LIMIT)

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

        rng, cmd_rng = jax.random.split(rng)
        command = self.sample_command(cmd_rng)

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
        phase = self._init_phase_l.copy()
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

        # control delay
        delay_steps = jax.random.randint(
            key_delay,
            shape=(),
            minval=self._config.dm_rand_config.ctrl_delay_range[0],
            maxval=self._config.dm_rand_config.ctrl_delay_range[1] + 1,
        )
        delay_steps = jp.where(
            self._config.dm_rand_config.enable_ctrl_delay, delay_steps, jp.zeros_like(delay_steps, dtype=jp.int32)
        )

        motor_targets_history = jp.repeat(
            self._default_qpos.reshape(1, -1), self._config.dm_rand_config.ctrl_delay_range[1] + 1, axis=0
        )
        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            # history
            "last_command": command.copy(),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "last_feet_vel": jp.zeros(2),
            "last_joint_vel": np.zeros(self.num_joints),
            # "obs_history": jp.zeros((self._config.history_len, self._config.num_obs)),
            # push
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # state
            "motor_targets": self._default_qpos.copy(),
            "motor_targets_history": motor_targets_history,
            "local_lin_vel": jp.zeros(3),
            "global_lin_vel": jp.zeros(3),
            "global_ang_vel": jp.zeros(3),
            "left_foot_force": jp.zeros(3),
            "right_foot_force": jp.zeros(3),
            "navi2world_rot": jp.eye(3),
            "navi2world_pose": jp.eye(4),
            "navi_torso_rpy": jp.zeros(3),
            "navi_torso_lin_vel": jp.zeros(3),
            "navi_torso_ang_vel": jp.zeros(3),
            "navi_pelvis_rpy": jp.zeros(3),
            "navi_pelvis_lin_vel": jp.zeros(3),
            "navi_pelvis_ang_vel": jp.zeros(3),
            # Phase related.
            "phase": phase,
            "phase_dt": phase_dt,
            "gait_mask": jp.zeros(2),
            "gait_freq": gait_freq,
            "foot_height": foot_height,
            # domain randomization
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "rfi_lim_scale": rfi_lim_scale,
            "delay_steps": delay_steps,
        }
        # update gait state

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
        lower_motor_targets = self._default_qpos[self.action_joint_ids] + action * self._config.action_scale
        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)
        _motor_targets_history = jp.roll(state.info["motor_targets_history"], 1, axis=0).at[0].set(motor_targets)
        state.info["motor_targets_history"] = _motor_targets_history
        delay_motor_targets = state.info["motor_targets_history"][state.info["delay_steps"]]

        # data = mjx_env.step(self.mjx_model, state.data, delay_motor_targets, self.n_substeps)
        state.info["rng"], data = torque_step(
            state.info["rng"],
            self.mjx_model,
            state.data,
            delay_motor_targets,
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
        state.info["left_foot_force"] = mjx_env.get_sensor_data(self.mj_model, data, "left_foot_force")
        state.info["right_foot_force"] = mjx_env.get_sensor_data(self.mj_model, data, "right_foot_force")

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

        # state.info["feet_contact"] = feet_contact
        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])

        state.info["last_command"] = state.info["command"].copy()
        state.info["command"] = jp.where(
            state.info["step"] % self._cmd_resample_steps == 0,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1

        # update gait
        self._update_phase(state)

        # update history
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["last_joint_vel"] = data.qvel[6:].copy()

        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])
        # state.info["obs_history"] = jp.where(done, 0, state.info["obs_history"])
        state.info["motor_targets_history"] = jp.where(done, self._default_qpos, state.info["motor_targets_history"])
        # ransom
        state.info["rng"], episode_rng = jax.random.split(state.info["rng"])
        _is_resample = jp.where(
            done,
            self.resample_domain_random_param(episode_rng, state),
            False,
        )

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_feet_vel"] = data.sensordata[self._foot_linvel_sensor_adr][..., 2]

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _update_phase(self, state):
        task_mask = state.info["command"][0]
        last_task_mask = state.info["last_command"][0]
        state.info["rng"], rng = jax.random.split(state.info["rng"])
        cond_phase = jax.random.bernoulli(rng)
        init_phase = jp.where(cond_phase, self._init_phase_l, self._init_phase_r)

        phase = state.info["phase"] + state.info["phase_dt"]
        phase = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
        phase = jp.where(task_mask == 1.0, phase, self._stance_phase)
        phase = jp.where(
            (last_task_mask == 0.0) & (task_mask == 1.0),
            init_phase,
            phase,
        )
        state.info["phase"] = phase

        # gait flag
        gait_cycle = jp.cos(phase)
        gait_mask = jp.where(gait_cycle > self._gait_bound, 1, 0)
        gait_mask = jp.where(gait_cycle < -self._gait_bound, -1, gait_mask)
        state.info["gait_mask"] = jp.float32(gait_mask)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= data.site_xpos[self._pelvis_imu_site_id][2] < 0.4
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
        return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        # body pose
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        acc_pelvis = self.get_accelerometer(data, "pelvis")
        gyro_torso = self.get_gyro(data, "torso")
        gvec_torso = data.site_xmat[self._torso_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_torso = self.get_local_linvel(data, "torso")
        acc_torso = self.get_accelerometer(data, "torso")

        # joint
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]
        joint_acc = data.qacc[6:]
        joint_force = data.actuator_force

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro_pelvis = (
                gyro_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gyro_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gyro
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gvec_pelvis = (
                gvec_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gvec_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gravity
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
                joint_angles
                + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_pos
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
                joint_vel
                + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_vel
        )

        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        state = jp.hstack(
            [
                # pose state
                noisy_gyro_pelvis,  # 3
                noisy_gvec_pelvis,  # 3
                # joint state
                (noisy_joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                noisy_joint_vel[self.obs_joint_ids],  # 23
                info["last_act"],  # num_actions
                # [0,0],
                # commands
                info["command"],  # 4
                info["foot_height"],  # 1
                gait_phase,  # (num_foot * 2)
            ]
        )
        privileged_state = jp.hstack(
            [
                # noiseless state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                linvel_pelvis,  # 3
                (joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                joint_vel[self.obs_joint_ids],  # 23
                info["last_act"],  # num_actions
                # [0,0],
                info["command"],  # 4
                info["foot_height"],  # 1
                gait_phase,  # (num_foot * 2)
                # hint state
                info["navi_torso_rpy"][:2],
                info["gait_mask"],
                feet_contact,  # num_foot
                # domain randomization
                info["kp_scale"],
                info["kd_scale"],
                info["rfi_lim_scale"],
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
        cmd_vel = info["command"][1:]  # [x, y, yaw]

        reward_dict = {
            # Tracking rewards.
            "base_orientation": self._reward_orientation(info["navi_torso_rpy"]),
            "base_height": self._reward_base_height(data.qpos[2], move_flag),
            "body_motion": self._cost_body_motion(info["navi_torso_lin_vel"], info["navi_torso_ang_vel"], cmd_vel),
            "tracking_lin_vel": self._reward_tracking_lin_vel(cmd_vel, info["navi_pelvis_lin_vel"]),
            "tracking_ang_vel": self._reward_tracking_ang_vel(cmd_vel, info["navi_pelvis_ang_vel"]),
            # behavior reward
            "body_rotation": self._reward_body_rotation(data, cmd_vel, info["navi2world_rot"]),
            "foot_contact": self._cost_foot_contact(data, feet_contact, info["gait_mask"], move_flag),
            "foot_clearance": self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], move_flag),
            "foot_slip": self._cost_foot_slip(data, info["gait_mask"]),
            "foot_balance": self._cost_foot_balance(data, info["navi2world_pose"], move_flag),
            "foot_far": self._cost_foot_far(data),
            # energy reward
            "joint_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "joint_torque": self._cost_torque(data.actuator_force),
            "smoothness_joint": self._cost_smoothness_joint(data, info["last_joint_vel"]),
            "smoothness_action": self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
        }
        for k, v in reward_dict.items():
            # replace NaN with 0
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)

        return reward_dict

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _reward_tracking_lin_vel(self, cmd_vel: jax.Array, local_lin_vel: jax.Array) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(cmd_vel[:2] - local_lin_vel[:2]))
        return jp.exp(-4.0 * lin_vel_error)

    def _reward_tracking_ang_vel(self, cmd_vel: jax.Array, local_ang_vel: jax.Array) -> jax.Array:
        angvel_error = jp.square(cmd_vel[2] - local_ang_vel[2])
        return jp.exp(-4.0 * angvel_error)

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
            1.2 * jp.square(local_lin_vel[2])
            + 1.2 * cost_lin_xy_orth
            + 0.4 * jp.abs(local_ang_vel[0])
            + 0.4 * jp.abs(local_ang_vel[1])
        )
        return cost

    def _reward_orientation(self, pelvis_rpy: jax.Array) -> jax.Array:
        err_roll = jp.abs(pelvis_rpy[0])
        err_pitch = jp.abs(pelvis_rpy[1])
        err_ori = err_roll + err_pitch
        rew = jp.exp(-err_ori)
        return rew

    def _reward_base_height(self, root_height, move_flag):
        base_height_error = jp.abs(root_height - self._config.reward_config.base_height_target)
        # rew = jp.exp(-base_height_error)
        rew = jp.exp(-3 * base_height_error)
        rew = jp.where((root_height * (1-move_flag)) > self._config.reward_config.base_height_target, 0.5, rew)
        return rew

    def _cost_foot_contact(
            self, data: mjx.Data, feet_contact: jax.Array, gait_flag: jax.Array, move_flag: jax.Array
    ) -> jax.Array:
        stance = feet_contact != 0
        swing = feet_contact == 0
        stance_des = jp.float32(gait_flag == 1)
        swing_des = jp.float32(gait_flag == -1)
        is_constrained = jp.float32(gait_flag != 0)
        cost_stance = jp.sum(jp.abs(stance - stance_des) * is_constrained)
        cost_swing = jp.sum(jp.abs(swing - swing_des) * is_constrained)
        cost = cost_stance + cost_swing
        cost *= move_flag
        return cost

    def _cost_foot_clearance(
            self, data: mjx.Data, tar_foot_height: jax.Array, gait_flag: jax.Array, move_flag: jax.Array
    ) -> jax.Array:
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        swing_des = jp.float32(gait_flag == -1)
        foot_z_tar = self._config.reward_config.foot_height_stance + tar_foot_height
        cost = jp.sum(swing_des * jp.square(foot_z - foot_z_tar))
        cost *= move_flag
        return cost

    def _cost_foot_slip(self, data: mjx.Data, gait_flag: jax.Array) -> jax.Array:
        stance_des = jp.float32(gait_flag == 1)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        feet_vel = jp.linalg.norm(feet_vel, axis=-1)
        cost = jp.sum(jp.square(feet_vel) * stance_des)
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

        foot2com_err = sup2navi_pos[0] - sup2navi_pos[1:]
        cost_support = jp.sum(jp.square(foot2com_err[0, :2] + foot2com_err[1, :2]))
        cost_support *= stance_mask
        return cost_support

    def _cost_foot_far(self, data: mjx.Data) -> jax.Array:
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_distance = jp.linalg.norm(foot_pos[0] - foot_pos[1])
        foot_spread_penalty = jp.where(
            foot_distance < 0.35,
            (0.35 - foot_distance),
            0.0
        )
        return foot_spread_penalty

    def _cost_torque(self, torques: jax.Array) -> jax.Array:
        cost_energy = jp.sum(jp.square(torques))
        return cost_energy

    def _cost_smoothness_action(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        smooth_1st = jp.square(act - last_act)
        smooth_2nd = jp.square(act - 2 * last_act + last_last_act)
        cost = jp.sum(smooth_1st + smooth_2nd)
        return cost

    def _cost_smoothness_joint(self, data: mjx.Data, last_joint_vel):
        qvel = data.qvel[6:][self.obs_joint_ids]
        qacc = (last_joint_vel[self.obs_joint_ids] - qvel) / self.dt
        cost = jp.sum(0.01 * jp.square(qvel) + jp.square(qacc))
        return cost

    def _reward_body_rotation(self, data: mjx.Data, cmd_vel: jax.Array, navi2world_rot: jax.Array) -> jax.Array:
        cmd_max = jp.abs(self._config.ang_vel_yaw[1]) + 1e-6
        cmd_decay = jp.clip((cmd_max - jp.abs(cmd_vel[2])) / cmd_max, 0.0, 1.0) ** 2
        legs2world_rot = jp.concat([data.xmat[self.body_ids_left_leg], data.xmat[self.body_ids_right_leg]])
        legs2navi_rot = navi2world_rot.T[None] @ legs2world_rot  # (N, 3, 3)
        axis_roll_err = jp.mean(jp.abs(legs2navi_rot[:, 2, 1]))
        axis_yaw_err = jp.mean(cmd_decay * jp.abs(legs2navi_rot[:, 0, 1]))
        axis_rew = jp.exp(-5.0 * (axis_roll_err + axis_yaw_err))
        return axis_rew

    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        lin_vel_x = jax.random.uniform(rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng3, minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])

        command = jp.hstack([1.0, lin_vel_x, lin_vel_y, ang_vel_yaw])

        # set small commands to zero
        small_cond = jp.linalg.norm(command[1:4]) < 0.2
        command = jp.where(small_cond, self._stop_cmd, command)

        # stop: stance with zero vel
        stop_cond = jax.random.bernoulli(rng4, p=self._cmd_stop_prob)
        command = jp.where(stop_cond, self._stop_cmd, command)

        return command

    def resample_domain_random_param(self, rng: jax.Array, state: mjx_env.State):
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


        state.info["kp_scale"] = kp_scale
        state.info["kd_scale"] = kd_scale
        state.info["rfi_lim_scale"] = rfi_lim_scale

        return True


@cat_ppo.registry.register("G1Loco", "command_to_reference_fn")
@cat_ppo.registry.register("G1LocoDis", "command_to_reference_fn")
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
