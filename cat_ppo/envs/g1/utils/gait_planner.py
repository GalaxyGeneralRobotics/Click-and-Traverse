from typing import Dict, Any

import jax.numpy as jnp
from jax.scipy.stats import norm

STANCE_IDX = 0.25


def smooth_fn(x, kappa, stance):
    r = jnp.remainder(x, 1.0)
    cdf = lambda v: norm.cdf(v, loc=0, scale=kappa)
    return cdf(r) * (1 - cdf(r - stance)) + cdf(r - 1) * (1 - cdf(r - stance - 1))


class GaitPlanner:
    def __init__(
        self,
        dt,
        init_foot_width,
        init_foot_height,
        max_foot_height,
        num_feet=2,
        freq=1.5,
        offset=0.5,
        stance=0.6,
        kappa=0.06,
    ):
        self.num_feet = num_feet
        self.dt = dt
        self.freq = freq
        self.offset = offset
        self.stance = stance
        self.kappa = kappa
        self.init_foot_height = init_foot_height
        self.max_foot_height = max_foot_height

        # Pre-compute constants
        self.swing_scale = 0.5 / (1 - stance)
        self.stance_scale = 0.5 / stance

        # Constant nominal foot position for a single instance
        self._foot_nom = jnp.array(
            [[0.0, init_foot_width / 2, 0.0], [0.0, -init_foot_width / 2, 0.0]],
            dtype=jnp.float32,
        )

    def init_state(self):
        return {
            "phase": jnp.array(STANCE_IDX, dtype=jnp.float32),
            "idx": jnp.zeros((self.num_feet,), dtype=jnp.float32),
            "contact": jnp.zeros((self.num_feet,), dtype=jnp.float32),
            "contact_bin": jnp.zeros((self.num_feet,), dtype=jnp.float32),
            "clock": jnp.zeros((self.num_feet,), dtype=jnp.float32),
        }

    def update(self, state: Dict[str, Any], mode: int):
        """

        Parameters
        ----------
        state
        mode: 1: move, 0: stop

        Returns
        -------

        """
        phase, idx = state["phase"], state["idx"]
        dt, freq, offset, stance = self.dt, self.freq, self.offset, self.stance
        stance_scale, swing_scale = self.stance_scale, self.swing_scale

        # Update gait phase (0-1 range)
        phase = (phase + dt * freq) % 1.0

        # Apply stop condition (freeze at STANCE_IDX and disable offset)
        phase = jnp.where(mode, phase, STANCE_IDX)
        offset_active = offset * mode  # Zero offset when stopped

        # Update indices
        idx = idx.at[0].set((phase + offset_active) % 1.0)
        idx = idx.at[1].set(phase)

        # Convert to normalized swing/stance coordinates
        stance_mask = idx < stance
        swing_vals = 0.5 + (idx - stance) * swing_scale
        stance_vals = idx * stance_scale
        norm_idx = jnp.where(stance_mask, stance_vals, swing_vals)

        contact = jnp.stack(
            [
                smooth_fn(norm_idx[0], self.kappa, self.stance),
                smooth_fn(norm_idx[1], self.kappa, self.stance),
            ]
        )

        contact_bin = stance_mask.astype(jnp.float32)
        clock = jnp.sin(2 * jnp.pi * norm_idx)

        return {
            "phase": phase,
            "idx": norm_idx,
            "contact": contact,
            "contact_bin": contact_bin,
            "clock": clock,
        }

    def compute_foot_pos(self, x_vel, y_vel, state):
        phase = jnp.abs(1.0 - (state["idx"] * 2.0)) - 0.5
        vel = jnp.array([[x_vel, y_vel]])
        offset = phase[:, jnp.newaxis] * vel * (0.5 / self.freq)
        local_xy = self._foot_nom[:, :2] + offset
        world_z = self.max_foot_height * (1 - state["contact"]) + self.init_foot_height
        return local_xy, world_z

    # def compute_foot_pos(self, x_vel, state):
    #     phase = jnp.abs(1.0 - (state["idx"] * 2.0)) - 0.5
    #     offset = phase[:, jnp.newaxis] * x_vel * (0.5 / self.freq)
    #     local_x = self._foot_nom[:, 0] + offset
    #     world_z = self.max_foot_height * (1 - state["contact"]) + self.init_foot_height
    #     return local_x, world_z


def demo_planner():
    import matplotlib.pyplot as plt
    import numpy as np

    # 初始化 gait planner（单个环境）
    planner = GaitPlanner(
        dt=0.02,
        init_foot_width=0.4,
        init_foot_height=0.0,
        max_foot_height=0.1,
    )

    steps = 200
    dt = planner.dt
    command = jnp.array([1.0, -0.5, 0.0])  # x vel, y vel, yaw (ignored)

    # 初始化状态
    state = planner.init_state()

    # 日志缓存
    gait_phases = []
    contacts = []
    contacts_bin = []
    foot_heights = []
    foot_traj = []

    for i in range(steps):
        if 100 < i < 150:
            state = planner.update(state, 0)
        else:
            state = planner.update(state, 1)

        local_xy, world_z = planner.compute_foot_pos(command[0], command[1], state)

        gait_phases.append(state["idx"][None, ...])
        contacts.append(state["contact"][None, ...])
        contacts_bin.append(state["contact_bin"][None, ...])
        foot_heights.append(world_z[None, ...])
        foot_traj.append(local_xy[None, ...])

    # 整理为 numpy
    gait_phases = np.concatenate(gait_phases, axis=0).reshape(steps, 1, 2)
    contacts = np.concatenate(contacts, axis=0).reshape(steps, 1, 2)
    contacts_bin = np.concatenate(contacts_bin, axis=0).reshape(steps, 1, 2)
    foot_heights = np.concatenate(foot_heights, axis=0).reshape(steps, 1, 2)
    foot_traj = np.concatenate(foot_traj, axis=0).reshape(steps, 1, 2, 2)

    time = np.arange(steps) * dt

    # 可视化
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(time, gait_phases[:, 0, 0], label="Left Phase", color="blue")
    axs[0].plot(time, gait_phases[:, 0, 1], label="Right Phase", color="orange")
    axs[0].set_ylabel("Gait Phase")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time, contacts[:, 0, 0], label="Left Contact", color="blue")
    axs[1].plot(time, contacts[:, 0, 1], label="Right Contact", color="orange")
    axs[1].set_ylabel("Contact")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time, foot_heights[:, 0, 0], label="Left Z", color="blue")
    axs[2].plot(time, foot_heights[:, 0, 1], label="Right Z", color="orange")
    axs[2].set_ylabel("Foot Height")
    axs[2].legend()
    axs[2].grid(True)

    left_x = foot_traj[:, 0, 0, 0]
    left_y = foot_traj[:, 0, 0, 1]
    right_x = foot_traj[:, 0, 1, 0]
    right_y = foot_traj[:, 0, 1, 1]

    axs[3].plot(time, left_x, label="Left X", color="blue")
    axs[3].plot(time, right_x, label="Right X", color="orange")
    axs[3].plot(time, left_y, "--", label="Left Y", color="blue")
    axs[3].plot(time, right_y, "--", label="Right Y", color="orange")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("XY Pos")
    axs[3].set_title("Foot XY Position")
    axs[3].legend()
    axs[3].grid(True)

    plt.suptitle("Bipedal Gait Planner Demo")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_planner()
