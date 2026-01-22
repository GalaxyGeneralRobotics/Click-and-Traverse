from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tree
from brax.envs.wrappers import training as brax_training
from cat_ppo.utils.logger import LOGGER  # noqa: F401
from mujoco import mjx
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
from ml_collections import config_dict

import cat_ppo.envs.g1  # noqa: F401


def _split_any(keys):
    """
    Split PRNG keys into (main, sample).

    keys shape  (2,)   -> returns (2,), (2,)
          (B,2) -> returns (B,2), (B,2)
    """
    if keys.ndim == 1:  # scalar key
        k1, k2 = jax.random.split(keys)
        return k1, k2
    elif keys.ndim == 2:  # batched keys
        split = jax.vmap(jax.random.split)(keys)  # (B,2,2)
        return split[:, 0], split[:, 1]  # two (B,2) arrays
    else:
        raise ValueError(f"PRNG key must be shape (2,) or (B,2); got {keys.shape}")


def _randint_any(keys, lo, hi):
    """
    Uniform ints using scalar or batched keys.

    keys (2,)   -> scalar int
    keys (B,2)  -> (B,) ints
    """
    if keys.ndim == 1:
        return jax.random.randint(keys, (), lo, hi)
    elif keys.ndim == 2:
        return jax.vmap(lambda k: jax.random.randint(k, (), lo, hi))(keys)
    else:
        raise ValueError(f"PRNG key must be shape (2,) or (B,2); got {keys.shape}")


def _take_cache(cache, idx):
    """cache: {name: [T,...]}, idx: scalar or (B,) -> {name: [...]} or {name: [B,...]}"""
    return {k: jnp.take(v, idx, axis=0, mode="clip") for k, v in cache.items()}


def _to_batch(x, mask):
    """Broadcast scalar x to (B,...) if mask is (B,)."""
    if x.ndim == 0 and mask.ndim == 1:
        return jnp.broadcast_to(x, mask.shape)
    return x


class SamplePFWrapper(wrapper.Wrapper):
    """
    Loads mocap trajectories from npz files with keys:
      - qpos: [T, 7+J]
      - qvel: [T, 6+J]
      - kpt_npose: [T, K, 4, 4]
      - kpt_cvel: [T, K, 6]

    Caches data to device and resamples per-episode reference when episode ends.
    """

    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def _batch_size(state):
        try:
            return jax.tree_util.tree_leaves(state.obs)[0].shape[0]
        except Exception:
            return state.done.shape[0] if state.done.ndim else 1

    def reset(self, rng) -> mjx_env.State:
        state = self.env.reset(rng)
        return state

    def step(self, state: mjx_env.State, action) -> mjx_env.State:
        state = self.env.step(state, action)

        done = state.done
        if done.ndim == 0:
            done = done[None]

        rng = state.info["rng"]
        state_reset = self.reset(rng)
        done_exp = done[:, None]
        a_obs = jnp.where(done_exp, state_reset.obs['state'], state.obs['state'])
        c_obs = jnp.where(done_exp, state_reset.obs['privileged_state'], state.obs['privileged_state'])
        state.obs.update(
            {
                "state": a_obs,
                "privileged_state": c_obs
            }
        )
        command = jnp.where(done_exp, state_reset.info["command"], state.info["command"])
        last_command = jnp.where(done_exp, state_reset.info["last_command"], state.info["last_command"])
        last_act = jnp.where(done_exp, state_reset.info["last_act"], state.info["last_act"])
        motor_targets = jnp.where(done_exp, state_reset.info["motor_targets"], state.info["motor_targets"])
        stop_timestep = jnp.where(done, state_reset.info["stop_timestep"], state.info["stop_timestep"])
        phase = jnp.where(done_exp, state_reset.info["phase"], state.info["phase"])
        phase_dt = jnp.where(done, state_reset.info["phase_dt"], state.info["phase_dt"])
        gait_freq = jnp.where(done, state_reset.info["gait_freq"], state.info["gait_freq"])
        foot_height = jnp.where(done, state_reset.info["foot_height"], state.info["foot_height"])
        state.info.update(
            {
                "rng": state_reset.info["rng"],
                "command": command,
                "last_command": last_command,
                "last_act": last_act,
                "motor_targets": motor_targets,
                "stop_timestep": stop_timestep,
                "phase": phase,
                "phase_dt": phase_dt,
                "gait_freq": gait_freq,
                "foot_height": foot_height,
            }
        )
        qpos = jnp.where(done_exp, state_reset.data.qpos, state.data.qpos)
        qvel = jnp.where(done_exp, state_reset.data.qvel, state.data.qvel)
        state = state.replace(
            data=state.data.replace(qpos=qpos, qvel=qvel),
        )
        reward = jnp.where(done, state_reset.reward, state.reward)
        state = state.replace(reward=reward)
        return state


def wrap_for_brax_training_reset(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]] | None = None,
) -> wrapper.Wrapper:
    if vision:
        env = wrapper.MadronaWrapper(env, num_vision_envs, randomization_fn)
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env)
    env = SamplePFWrapper(env)
    return env

