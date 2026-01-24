import time
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

import cat_ppo
from cat_ppo.envs.g1 import constants as consts
from scipy.spatial.transform import Rotation as R

EPS=1e-6
def base2navi_transform(base2world: np.ndarray) -> np.ndarray:
    x_proj = base2world[:, 0]
    x_proj /= np.linalg.norm(x_proj)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_proj)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))

def world_to_navi_vel(navi2world_pose: np.ndarray, vel: np.ndarray) -> np.ndarray:
    world2navi = np.linalg.inv(navi2world_pose)
    R = world2navi[:3, :3]
    return (R @ vel.T).T

def quat_conj(q):
    # q: (..., 4)  wxyz
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)

def quat_mul(q1, q2):
    # q1, q2: (..., 4)  wxyz
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

def quat_rotate(q, v):
    # q: (..., 4)  wxyz
    # v: (..., 3)
    zeros = np.zeros_like(v[..., :1])
    q_v = np.concatenate([zeros, v], axis=-1)
    return quat_mul(quat_mul(q, q_v), quat_conj(q))[..., 1:]

def delay_body_pos(p_gt, q_gt, p_odom, q_odom, body_pos):
    body_pos_local = quat_rotate(quat_conj(q_gt), body_pos - p_gt)
    return (p_odom + quat_rotate(q_odom, body_pos_local)).reshape(-1,3)

def normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def noisy_rootpose(qpos_root):
    dxyz = (np.random.rand(3) * 2 - 1) * 0.05  # (3,)

    angle = (np.random.rand() * 2 - 1) * np.deg2rad(10.0)  
    half = angle / 2.0

    q_dr = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz

    q_gt = qpos_root[3:7]  # wxyz
    q_new = normalize(quat_mul(q_dr, q_gt))

    return np.concatenate([qpos_root[:3] + dxyz, q_new])  # (7,)



def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
    return dist, normal


def geoms_colliding(state: mujoco.MjData, geom1: int, geom2: int):
    """Return True if the two geoms are colliding."""
    if len(state.contact) == 0:
        return 0
    return get_collision_info(state.contact, geom1, geom2)[0] < 0

from pathlib import Path
import xml.etree.ElementTree as ET
import os
def set_scene_for_xml(xml_path, scene_path):
        xml_dir = Path(xml_path).resolve().parent

        xml_text = Path(xml_path).read_text(encoding="utf-8")

        root = ET.fromstring(xml_text)

        scene_mesh = None
        for mesh in root.iter("mesh"):
            if mesh.get("name") == "scene_mesh":
                scene_mesh = mesh
                break
        if scene_mesh is None:
            raise RuntimeError('Cannot find <mesh name="scene_mesh" ...> in the XML.')

        obs_root = Path(scene_path).resolve()       # e.g., .../data/assets/TypiObs/narrow1
        obs_obj_abs = obs_root / "obs.obj"
        if not obs_obj_abs.exists():
            raise FileNotFoundError(f"Obstacle mesh not found: {obs_obj_abs}")

        obs_obj_rel = obs_obj_abs.relative_to(xml_dir) if obs_obj_abs.is_relative_to(xml_dir) \
            else Path("../") / obs_obj_abs.relative_to(xml_dir.parent) 

        try:
            obs_obj_rel = obs_obj_abs.relative_to(xml_dir)
        except ValueError:
            obs_obj_rel = Path(os.path.relpath(obs_obj_abs, xml_dir))

        scene_mesh.set("file", obs_obj_rel.as_posix())

        tmp_xml_path = xml_dir / (xml_path.stem + "._tmp_scene.xml")
        tmp_xml_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")

        return str(tmp_xml_path)

class BaseEnv:
    def __init__(self, task_type, config=None):
        xml_path = consts.task_to_xml(task_type)
        tmp_xml = set_scene_for_xml(xml_path, config.pf_config.path)
        self.mj_model = mujoco.MjModel.from_xml_path(tmp_xml)
        self.mj_data = mujoco.MjData(self.mj_model)

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = self.mj_model.sensor(sensor_name).id
        sensor_adr = self.mj_model.sensor_adr[sensor_id]
        sensor_dim = self.mj_model.sensor_dim[sensor_id]
        return self.mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]

    # Sensor readings.
    def get_gravity(self, frame: str) -> np.ndarray:
        """Return the gravity vector in the world frame."""
        return self.get_sensor_data(f"{consts.GRAVITY_SENSOR}_{frame}")

    def get_global_linvel(self, frame: str) -> np.ndarray:
        """Return the linear velocity of the robot in the world frame."""
        return self.get_sensor_data(f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}")

    def get_global_angvel(self, frame: str) -> np.ndarray:
        """Return the angular velocity of the robot in the world frame."""
        return self.get_sensor_data(f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}")

    def get_local_linvel(self, frame: str) -> np.ndarray:
        """Return the linear velocity of the robot in the local frame."""
        return self.get_sensor_data(f"{consts.LOCAL_LINVEL_SENSOR}_{frame}")

    def get_accelerometer(self, frame: str) -> np.ndarray:
        """Return the accelerometer readings in the local frame."""
        return self.get_sensor_data(f"{consts.ACCELEROMETER_SENSOR}_{frame}")

    def get_gyro(self, frame: str) -> np.ndarray:
        """Return the gyroscope readings in the local frame."""
        return self.get_sensor_data(f"{consts.GYRO_SENSOR}_{frame}")


def to_pose(pos: np.ndarray, rot: np.ndarray):
    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = rot
    return pose

@dataclass
class State:
    info: dict
    obs: dict


@cat_ppo.registry.register("G1Cat", "play_env_class")
@cat_ppo.registry.register("G1CatPri", "play_env_class")
class PlayG1CatEnv(BaseEnv):
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData

    def __init__(
        self,
        task_type="mesh",
        fix_body=False,
        config=None,
        dt=0.02,
        sim_dt=0.002,
        headless=False,
    ):
        super().__init__("mesh", config)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.fix_body = fix_body
        self._config = config
        self._post_init()
        self.dt = dt
        self.sim_dt = sim_dt
        self.pri = False

        pf_path = config.pf_config.path
        self.dx = config.pf_config.dx
        self.sdf = np.load(f"{pf_path}/sdf.npy")[...,None]   # (Nx,Ny,Nz)
        self.bf  = np.load(f"{pf_path}/bf.npy")    # (Nx,Ny,Nz,3)
        self.gf  = np.load(f"{pf_path}/gf.npy")    # (Nx,Ny,Nz,3)
        self.pf_origin = np.array(config.pf_config.origin, dtype=np.float32)
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape
        self.current_goal_global = np.array([2.0, 0.0, 0.7])
        self.gait_freq = 1.5
        self.foot_height = 0.07
        
    @property
    def action_size(self) -> int:
        return len(consts.ACTION_JOINT_NAMES)

    def _post_init(self):
        self._init_command = np.zeros(3)
        self._init_phase = np.array([0.0, np.pi])
        self._stance_phase = np.array([0, 0])
        self._num_joints = len(self.mj_data.qpos[7:])
        self._default_qpos = np.array(consts.DEFAULT_QPOS[7:])
        self.action_joint_names = consts.ACTION_JOINT_NAMES.copy()
        self.action_joint_ids = []
        for j_name in self.action_joint_names:
            self.action_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.action_joint_ids = np.array(self.action_joint_ids)

        self.obs_joint_names = consts.OBS_JOINT_NAMES.copy()
        self.obs_joint_ids = []
        for j_name in self.obs_joint_names:
            self.obs_joint_ids.append(self.mj_model.actuator(j_name).id)
        self.obs_joint_ids = np.array(self.obs_joint_ids)

        self._floor_geom_id = self.mj_model.geom("floor").id
        self._torso_imu_site_id = self.mj_model.site("imu_in_torso").id
        self._pelvis_imu_site_id = self.mj_model.site("imu_in_pelvis").id
        self._feet_geom_id = np.array(
            [self.mj_model.geom(name).id for name in consts.FEET_GEOMS]
        )
        self._feet_site_id = np.array(
            [self.mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self._hands_site_id = np.array(
            [self.mj_model.site(name).id for name in consts.HAND_SITES]
        )
        self._knees_site_id = np.array(
            [self.mj_model.site(name).id for name in consts.KNEE_SITES]
        )
        self._shlds_site_id = np.array(
            [self.mj_model.site(name).id for name in consts.SHOULDER_SITES]
        )

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self.mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self.mj_model.sensor_adr[sensor_id]
            sensor_dim = self.mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr)

        self.body_id_pelvis = self.mj_model.body("pelvis").id
        self.body_id_torso = self.mj_model.body("torso_link").id
        self.body_names_left_leg = ["left_knee_link", "left_ankle_roll_link"]
        self.body_ids_left_leg = [
            self.mj_model.body(n).id for n in self.body_names_left_leg
        ]
        self.body_names_right_leg = ["right_knee_link", "right_ankle_roll_link"]
        self.body_ids_right_leg = [
            self.mj_model.body(n).id for n in self.body_names_right_leg
        ]
        self._head_site_id = self.mj_model.site("head").id

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    def reset(self):
        self.mj_data.qpos[:7] = consts.DEFAULT_QPOS[:7]
        self.mj_data.qpos[7:] = self._default_qpos

        mujoco.mj_forward(self.mj_model, self.mj_data)
        if not self.headless:
            self.viewer.sync()
        phase_dt = 2 * np.pi * self.dt * self.gait_freq

        head_pos = self.mj_data.site_xpos[self._head_site_id]
        head_vel = np.zeros_like(head_pos)
        feet_pos = self.mj_data.site_xpos[self._feet_site_id]
        feet_vel = np.zeros_like(feet_pos)
        hands_pos = self.mj_data.site_xpos[self._hands_site_id]
        hands_vel = np.zeros_like(hands_pos)
        knees_pos = self.mj_data.site_xpos[self._knees_site_id]
        shlds_pos = self.mj_data.site_xpos[self._shlds_site_id]
        pelv_pos = self.mj_data.site_xpos[self._pelvis_imu_site_id].reshape(1, -1)
        tors_pos = self.mj_data.site_xpos[self._torso_imu_site_id].reshape(1, -1)
        all_poses = np.concatenate([
            head_pos.reshape(1, -1),
            pelv_pos.reshape(1, -1),
            tors_pos.reshape(1, -1),
            feet_pos,
            hands_pos,
            knees_pos,
            shlds_pos,
        ], axis=0)
        all_gf = self.sample_field(self.gf, all_poses)
        all_bf = self.sample_field(self.bf, all_poses)
        all_df = self.sample_field(self.sdf, all_poses)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1,2,3,5,7,9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1,2,3,5,7,9], axis=0)

        command = self.compute_cmd_from_rtf(pelvgf.reshape(-1), np.concat([headgf, feetgf, handsgf]), np.concat([headbf, feetbf, handsbf]))

        info = {
            "step": 0,
            "command": command.copy(),
            "last_command": command.copy(),
            "flags": np.zeros(2),
            "last_flags": np.zeros(2),
            "last_act": np.zeros(12),
            "phase_dt": phase_dt,
            "phase": self._init_phase.copy(),
            "foot_height": self.foot_height,
            "motor_targets": self._default_qpos.copy(),# NOTE
            "timestamp_move2stop": 100,
            "gait_mask": np.zeros(2),
            "odom_delay": self.mj_data.qpos[:7],
            "headgf": headgf.copy(),
            "headbf": headbf.copy(),
            "headdf": headdf.copy(),
            "head_pos": head_pos.copy(),
            "head_vel": head_vel.copy(),
            "feetgf": feetgf.copy(),
            "feetbf": feetbf.copy(),
            "feetdf": feetdf.copy(),
            "feet_pos": feet_pos.copy(),
            "feet_vel": feet_vel.copy(),
            "handsgf": handsgf.copy(),
            "handsbf": handsbf.copy(),
            "handsdf": handsdf.copy(),
            "hands_pos": hands_pos.copy(),
            "hands_vel": hands_vel.copy(),
            "kneesgf": kneesgf.copy(),
            "kneesbf": kneesbf.copy(),
            "kneesdf": kneesdf.copy(),
            "knees_pos": knees_pos.copy(),
            "shldsgf": shldsgf.copy(),
            "shldsbf": shldsbf.copy(),
            "shldsdf": shldsdf.copy(),
            "shlds_pos": shlds_pos.copy(),
            "pelvgf": pelvgf.copy(),
            "pelvbf": pelvbf.copy(),
            "pelvdf": pelvdf.copy(),
            "pelv_pos": pelv_pos.copy(),
            "torsgf": torsgf.copy(),
            "torsbf": torsbf.copy(),
            "torsdf": torsdf.copy(),
            "tors_pos": tors_pos.copy(),
        }
        # breakpoint()
        obs = self.get_obs(info)
        return State(info, obs)

    def step(self, state: State, action: np.ndarray):
        if self.fix_body:
            self.mj_data.qvel[:6] = 0

        # Action Space 1: delta position from last motor targets
        lower_motor_targets = np.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )

        motor_targets = self._default_qpos.copy()
        motor_targets[self.action_joint_ids] = lower_motor_targets
        state.info["motor_targets"] = motor_targets.copy()

        for _ in range(int(self.dt / self.sim_dt)):
            torques = consts.KPs * (
                motor_targets - self.mj_data.qpos[7:]
            ) + consts.KDs * (-self.mj_data.qvel[6:])
            self.mj_data.ctrl[:] = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

        if not self.headless:
            self.viewer.sync()
            time.sleep(self.dt)
        head_pos = self.mj_data.site_xpos[self._head_site_id]
        head_vel = (head_pos - state.info["head_pos"]) / (1./50.)
        feet_pos = self.mj_data.site_xpos[self._feet_site_id]
        feet_vel = (feet_pos - state.info["feet_pos"]) / (1./50.)
        hands_pos = self.mj_data.site_xpos[self._hands_site_id]
        hands_vel = (hands_pos - state.info["hands_pos"]) / (1./50.)
        knees_pos = self.mj_data.site_xpos[self._knees_site_id]
        shlds_pos = self.mj_data.site_xpos[self._shlds_site_id]
        pelv_pos = self.mj_data.site_xpos[self._pelvis_imu_site_id]
        tors_pos = self.mj_data.site_xpos[self._torso_imu_site_id]
        all_poses = np.concatenate([
            head_pos.reshape(1, -1),
            pelv_pos.reshape(1, -1),
            tors_pos.reshape(1, -1),
            feet_pos,
            hands_pos,
            knees_pos,
            shlds_pos,
        ], axis=0)
        update_pf = (state.info["step"] % 1) == 0 # NOTE simulate delay
        odo_noisy = noisy_rootpose(self.mj_data.qpos[:7])
        odom_delay = np.where(update_pf, self.mj_data.qpos[:7].copy(), state.info["odom_delay"])
        state.info["odom_delay"]=odom_delay.copy()
        p_gt = self.mj_data.qpos[:3]
        q_gt = self.mj_data.qpos[3:7]
        p_odom = odom_delay[:3]
        q_odom = odom_delay[3:7]
        all_poses_delay = delay_body_pos(p_gt, q_gt, p_odom, q_odom, all_poses)
        all_gf = self.sample_field(self.gf, all_poses_delay)
        all_bf = self.sample_field(self.bf, all_poses_delay)
        all_df = self.sample_field(self.sdf, all_poses_delay)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1,2,3,5,7,9], axis=0)
        headdf, pelvdf, torsdf, feetdf, handsdf, kneesdf, shldsdf = np.split(all_df, [1,2,3,5,7,9], axis=0)
        command = self.compute_cmd_from_rtf(pelvgf.reshape(-1), np.concat([headgf, feetgf, handsgf]), np.concat([headbf, feetbf, handsbf]))
        state.info["command"] = command.copy()
        self._update_phase(state)
        move_flag = state.info["last_flags"][1]
        all_gf = all_gf * (move_flag[None] > 0.5) / (np.linalg.norm(all_gf, axis=-1, keepdims=True) + EPS)
        all_bf = all_bf / (np.linalg.norm(all_bf, axis=-1, keepdims=True) + EPS)
        headgf, pelvgf, torsgf, feetgf, handsgf, kneesgf, shldsgf = np.split(all_gf, [1,2,3,5,7,9], axis=0)
        headbf, pelvbf, torsbf, feetbf, handsbf, kneesbf, shldsbf = np.split(all_bf, [1,2,3,5,7,9], axis=0)
        
        state.info["headgf"] = headgf.copy()
        state.info["headbf"] = headbf.copy()
        state.info["headdf"] = headdf.copy()
        state.info["head_pos"] = head_pos.copy()
        state.info["head_vel"] = head_vel.copy()
        state.info["feetgf"] = feetgf.copy()
        state.info["feetbf"] = feetbf.copy()
        state.info["feetdf"] = feetdf.copy()
        state.info["feet_vel"] = feet_vel.copy()
        state.info["feet_pos"] = feet_pos.copy()
        state.info["handsgf"] = handsgf.copy()
        state.info["handsbf"] = handsbf.copy()
        state.info["handsdf"] = handsdf.copy()
        state.info["hands_vel"] = hands_vel.copy()
        state.info["hands_pos"] = hands_pos.copy()
        state.info["pelvgf"] = pelvgf.copy()
        state.info["pelvbf"] = pelvbf.copy()
        state.info["pelvdf"] = pelvdf.copy()
        state.info["torsgf"] = torsgf.copy()
        state.info["torsbf"] = torsbf.copy()
        state.info["torsdf"] = torsdf.copy()
        state.info["kneesgf"] = kneesgf.copy()
        state.info["kneesbf"] = kneesbf.copy()
        state.info["kneesdf"] = kneesdf.copy()
        state.info["shldsgf"] = shldsgf.copy()
        state.info["shldsbf"] = shldsbf.copy()
        state.info["shldsdf"] = shldsdf.copy()
        state.info["step"] += 1
        state.info["last_act"] = action.copy()

        obs = self.get_obs(state.info)
        return State(state.info, obs)

    def _update_phase(self, state):
        step = state.info["step"]
        command = state.info["command"]
        last_flags = state.info["last_flags"]
        phase = state.info["phase"]
        phase_dt = state.info["phase_dt"]
        timestamp = state.info["timestamp_move2stop"]

        has_vel = np.linalg.norm(command) > 0.2
        had_vel = last_flags[0]

        # Transitions
        move2stop = (had_vel == 1.0) & (has_vel == 0.0)
        stop2move = (had_vel == 0.0) & (has_vel == 1.0)

        # Timestamp update via np.where
        state.info["timestamp_move2stop"] = np.where(move2stop, step + 50, timestamp)

        after_delay = step > state.info["timestamp_move2stop"]
        moving = np.where((has_vel == 0.0) & after_delay, 0.0, 1.0)

        # Phase update
        new_phase = (phase + phase_dt + np.pi) % (2 * np.pi) - np.pi
        phase = np.where(moving, new_phase, self._stance_phase)
        phase = np.where(stop2move, self._init_phase, phase)

        # Update state
        state.info["phase"] = phase
        state.info["last_command"] = command.copy()
        state.info["last_flags"] = [has_vel, moving]
        gait_cycle = np.cos(phase)
        gait_mask = np.where(gait_cycle > 0.6, 1, 0)
        gait_mask = np.where(gait_cycle < -0.6, -1, gait_mask)
        state.info["gait_mask"] = np.float32(gait_mask)

    def get_obs(self, info):
        # pose
        gyro_pelvis = self.get_gyro("pelvis")
        gvec_pelvis = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(
            3, 3
        ).T @ np.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel("pelvis")
        # joint
        joint_angles = self.mj_data.qpos[7:]
        joint_vel = self.mj_data.qvel[6:]
        move_flag = info["last_flags"][1]
        pelvis2world_rot = self.mj_data.site_xmat[self._pelvis_imu_site_id].reshape(
            3, 3
        )
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        navi2world_pose = np.eye(4)
        navi2world_pose[:3, :3]=navi2world_rot
        navi2world_pose[:2, 3] = self.mj_data.site_xpos[self._pelvis_imu_site_id][:2]
        # print(self.mj_data.site_xpos[self._pelvis_imu_site_id][:2])
        navi2world_pose[2, 3] = 0.75
        # navi2world_pose = navi2world_pose.reshape(-1)
        torso2world_rot = self.mj_data.site_xmat[self._torso_imu_site_id].reshape(
            3, 3
        )
        torso2navi_rot = navi2world_rot.T @ torso2world_rot
        navi_torso_rpy = R.from_matrix(torso2navi_rot).as_euler('xyz', degrees=False)
        # print(navi_torso_rpy[2])
        gait_phase = np.hstack([np.cos(info["phase"]), np.sin(info["phase"])])
        floor_contact = [geoms_colliding(self.mj_data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id]

        headgf = info["headgf"].copy()
        headbf = info["headbf"].copy()
        headdf = info["headdf"].copy()
        pelvgf = info["pelvgf"].copy()
        pelvbf = info["pelvbf"].copy()
        pelvdf = info["pelvdf"].copy()
        torsgf = info["torsgf"].copy()
        torsbf = info["torsbf"].copy()
        torsdf = info["torsdf"].copy()
        feetgf = info["feetgf"].copy()
        feetbf = info["feetbf"].copy()
        feetdf = info["feetdf"].copy()
        handsgf= info["handsgf"].copy()
        handsbf= info["handsbf"].copy()
        handsdf= info["handsdf"].copy()
        kneesgf = info["kneesgf"].copy()
        kneesbf= info["kneesbf"].copy()
        kneesdf= info["kneesdf"].copy()
        shldsgf= info["shldsgf"].copy()
        shldsbf= info["shldsbf"].copy()
        shldsdf= info["shldsdf"].copy()
        head_pos = info["head_pos"].copy()
        head_vel = info["head_vel"].copy()
        feet_pos = info["feet_pos"].copy()
        feet_vel = info["feet_vel"].copy()
        hands_pos = info["hands_pos"].copy()
        hands_vel = info["hands_vel"].copy()
        command = info["command"].copy()
        if self.pri:
            state = np.hstack(
                [
                    # pose state
                    gyro_pelvis,  # 3
                    gvec_pelvis,  # 3
                    # joint state
                    (joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                    joint_vel[self.obs_joint_ids],  # 23
                    info["last_act"], # 12
                    info["motor_targets"][self.action_joint_ids],  # num_actions
                    # commands
                    [move_flag],
                    command,  # 4
                    info["foot_height"], # 1
                    gait_phase,  # 4
                    linvel_pelvis,
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
                    navi_torso_rpy[:2],
                    info["gait_mask"],
                    floor_contact,  # num_foot
                ]
            )
        else:
            headgf = world_to_navi_vel(navi2world_pose, headgf.reshape(-1, 3))
            headbf = world_to_navi_vel(navi2world_pose, headbf.reshape(-1, 3))
            pelvgf = world_to_navi_vel(navi2world_pose, pelvgf.reshape(-1, 3))
            pelvbf = world_to_navi_vel(navi2world_pose, pelvbf.reshape(-1, 3))
            torsgf = world_to_navi_vel(navi2world_pose, torsgf.reshape(-1, 3))
            torsbf = world_to_navi_vel(navi2world_pose, torsbf.reshape(-1, 3))
            feetgf = world_to_navi_vel(navi2world_pose, feetgf.reshape(-1, 3))
            feetbf = world_to_navi_vel(navi2world_pose, feetbf.reshape(-1, 3))
            handsgf = world_to_navi_vel(navi2world_pose, handsgf.reshape(-1, 3))
            handsbf = world_to_navi_vel(navi2world_pose, handsbf.reshape(-1, 3))
            kneesgf = world_to_navi_vel(navi2world_pose, kneesgf.reshape(-1, 3))
            kneesbf = world_to_navi_vel(navi2world_pose, kneesbf.reshape(-1, 3))
            shldsgf = world_to_navi_vel(navi2world_pose, shldsgf.reshape(-1, 3))
            shldsbf = world_to_navi_vel(navi2world_pose, shldsbf.reshape(-1, 3))
            command = world_to_navi_vel(navi2world_pose, command.reshape(-1, 3)).reshape(3)
            command[-1] = 0
            headbf = headbf * (headdf < 0.5)
            headdf = np.clip(headdf, -1.0, 0.5)
            pelvbf = pelvbf * (pelvdf < 0.5)
            pelvdf = np.clip(pelvdf, -1.0, 0.5)
            torsbf = torsbf * (torsdf < 0.5)
            torsdf = np.clip(torsdf, -1.0, 0.5)
            feetbf = feetbf * (feetdf < 0.5)
            feetdf = np.clip(feetdf, -1.0, 0.5)
            handsbf = handsbf * (handsdf < 0.5)
            handsdf = np.clip(handsdf, -1.0, 0.5)
            kneesbf = kneesbf * (kneesdf < 0.5)
            kneesdf = np.clip(kneesdf, -1.0, 0.5)
            shldsbf = shldsbf * (shldsdf < 0.5)
            shldsdf = np.clip(shldsdf, -1.0, 0.5)
            state = np.hstack(
                [
                    gyro_pelvis,  # 3
                    gvec_pelvis,  # 3
                    # joint state
                    (joint_angles - self._default_qpos)[self.obs_joint_ids],  # 23
                    joint_vel[self.obs_joint_ids],  # 23
                    info["last_act"], # 12
                    info["motor_targets"][self.action_joint_ids],  # num_actions
                    # commands
                    [move_flag],
                    command,  # 4
                    info["foot_height"], # 1
                    gait_phase,  # 4
                    headgf.reshape(-1),
                    headbf.reshape(-1),
                    headdf.reshape(-1),
                    pelvgf.reshape(-1),
                    pelvbf.reshape(-1),
                    pelvdf.reshape(-1),
                    torsgf.reshape(-1),
                    torsbf.reshape(-1),
                    torsdf.reshape(-1),
                    feetgf.reshape(-1),
                    feetbf.reshape(-1),
                    feetdf.reshape(-1),
                    handsgf.reshape(-1),
                    handsbf.reshape(-1),
                    handsdf.reshape(-1),
                    kneesgf.reshape(-1),
                    kneesbf.reshape(-1),
                    kneesdf.reshape(-1),
                    shldsgf.reshape(-1),
                    shldsbf.reshape(-1),
                    shldsdf.reshape(-1),
                ]
            )
        self.mj_data.mocap_pos[0] = self.mj_data.xpos[self.body_ids_left_leg[1]]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        return {
            "state": state,
            "privileged_state": None,
        }
    
    def world_to_grid(self, pos):
        """ 世界坐标 -> voxel index (浮点) """
        rel = pos - self.pf_origin
        idx = rel / self.dx
        return idx

    def sample_field(self, field, pos):
        idx = self.world_to_grid(pos)                         # (N,3)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]            # (N,)

        x = np.clip(x, 0, self.Nx - 2)
        y = np.clip(y, 0, self.Ny - 2)
        z = np.clip(z, 0, self.Nz - 2)

        xi = np.floor(x).astype(np.int32)                    # (N,)
        yi = np.floor(y).astype(np.int32)
        zi = np.floor(z).astype(np.int32)
        xd = x - xi                                          # (N,)
        yd = y - yi
        zd = z - zi

        offsets = np.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1]
        ], dtype=np.int32)                                   # (8,3)

        base = np.stack([xi, yi, zi], axis=1)                # (N,3)
        corners = base[:, None, :] + offsets[None, :, :]     # (N,8,3)

        vals = field[corners[..., 0], corners[..., 1], corners[..., 2], :]  # (N,8,C)
        wx = np.stack([1.0 - xd, xd], axis=1)                # (N,2)
        wy = np.stack([1.0 - yd, yd], axis=1)                # (N,2)
        wz = np.stack([1.0 - zd, zd], axis=1)                # (N,2)

        w = (wx[:, :, None, None] *
            wy[:, None, :, None] *
            wz[:, None, None, :]).reshape(-1, 8)            # (N,8)

        out = np.einsum('ne,nec->nc', w, vals)               # (N,C)
        return out

    
    def get_goal(self, navi2world_pose):
        navi2world_rot = navi2world_pose[:3, :3]
        root_pos = navi2world_pose[:3, 3]

        goal2world = self.current_goal_global - root_pos
        world2navi_rot = navi2world_rot.T
        goal2navi = world2navi_rot @ goal2world

        current_goal_yaw = np.arctan2(goal2navi[1], goal2navi[0])

        command = np.stack([goal2navi[0], goal2navi[1], current_goal_yaw])
        self.done = np.linalg.norm(goal2navi[:2]) < 0.2 
        command[self.done] = 0.0 # assuming self._stop_cmd = [1, 0, 0, 0]
        return command

    def compute_cmd_from_rtf(self, rtf, cgf, cbf):

        v = rtf[:2]* 0.7  

        bnorm = np.linalg.norm(cbf[:, :2], axis=-1, keepdims=True) + 1e-9
        b_hat = cbf[:, :2] / bnorm  # (M,2)

        Ls = np.sum(b_hat * cgf[:, :2], axis=-1)  # (M,)

        bv = np.sum(b_hat * v, axis=-1)           # (M,)

        diff = (Ls - bv)[:, None] / (np.sum(b_hat * b_hat, axis=-1, keepdims=True) + 1e-9)
        delta = diff * b_hat  # (M,2)

        mask = (Ls > bv)[:, None]  # (M,1)
        delta = np.where(mask, delta, 0.0)

        v_new = v + np.mean(delta, axis=0)

        command = np.hstack([v_new[0], v_new[1], 0.0]) * 0.75

        small_cond = np.linalg.norm(command) < 0.2
        command = np.where(small_cond, self._init_command, command)
        return command

