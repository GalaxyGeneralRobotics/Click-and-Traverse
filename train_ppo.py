import swanlab

swanlab.sync_wandb()

import inspect
import functools
import time
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from absl import logging
import tqdm
import tyro
import wandb
import numpy as np
import jax

from mujoco_playground import wrapper
from brax.training.agents.ppo.networks import make_ppo_networks

import cat_ppo
from cat_ppo import update_file_handler
from cat_ppo.constant import PATH_LOG
from cat_ppo.learning.policy.ppo import train as ppo # brax.training.agents.ppo
from cat_ppo.learning.train.pf_utils import wrap_for_brax_training_reset

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

@dataclass
class Args:
    task: str
    exp_name: str = "debug"
    num_timesteps: int = 400_000_000
    seed: int = 42
    convert_onnx: bool = True
    restore_name: str = "none"
    ground: float = 0
    lateral: float = 0
    overhead: float = 0
    term_collision_threshold: float = 0.04
    obs_name: str = 'empty'
    def generate_exp_name(self):
        exp_name_parts = [self.exp_name]

        if self.ground != 0:
            exp_name_parts.append('G'+str(self.ground).replace('.', ''))

        if self.lateral != 0:
            exp_name_parts.append('L'+str(self.lateral).replace('.', ''))

        if self.overhead != 0:
            exp_name_parts.append('O'+str(self.overhead).replace('.', ''))

        exp_name_parts.append(f"T{str(self.term_collision_threshold).replace('.', '')}")

        if self.obs_name:
            exp_name_parts.append(self.obs_name.replace('_', ''))

        return "x".join(exp_name_parts)

def _prepare_exp_name(task: str, exp_name: str) -> str:
    timestamp = datetime.now().strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"

def _validate_exp_name_format(exp_name: str, debug_mode: bool):
    if not debug_mode and len(exp_name.split("_")) != 4:
        raise ValueError(
            f"exp_name should be in the format <task>_<tag>_<version>, got {exp_name}"
        )


def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = Path(PATH_LOG) / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path


def _log_checkpoint_path(ckpt_path: Path):
    logging.info(f"Checkpoint path: {ckpt_path}")


def _apply_args_to_config(args: Args, policy_cfg, env_config, debug: bool):
    policy_cfg.num_timesteps = args.num_timesteps
    if debug:
        policy_cfg.training_metrics_steps = 1000
        policy_cfg.num_evals = 5
        policy_cfg.batch_size = 8
        policy_cfg.num_minibatches = 2
        policy_cfg.num_envs = policy_cfg.batch_size * policy_cfg.num_minibatches
        policy_cfg.episode_length = 200
        policy_cfg.unroll_length = 10
        policy_cfg.num_updates_per_batch = 1
        policy_cfg.action_repeat = 1
        policy_cfg.num_timesteps = 100_000
        policy_cfg.num_resets_per_eval = 1
        policy_cfg.num_eval_envs = 128
    # cfg.restore_checkpoint_path = Path(args.restore_checkpoint_path)
    if args.restore_name != "none":
        from cat_ppo.constant import get_latest_ckpt
        policy_cfg.restore_checkpoint_path = str(get_latest_ckpt(args.restore_name))
    env_config.reward_config.scales.feetgf = args.ground
    env_config.reward_config.scales.feetdf = args.ground
    env_config.reward_config.scales.headgf = args.overhead
    env_config.reward_config.scales.headdf = args.overhead
    env_config.reward_config.scales.handsgf = args.lateral
    env_config.reward_config.scales.handsdf = args.lateral
    env_config.reward_config.scales.kneesdf = args.lateral
    env_config.reward_config.scales.shldsdf = args.lateral
    env_config.term_collision_threshold = args.term_collision_threshold
    env_config.pf_config.path = f'data/assets/RandObs/{args.obs_name}' if args.obs_name.startswith('D') else f'data/assets/TypiObs/{args.obs_name}'

def _prepare_training_params(cfg, ckpt_path: Path):
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_for_brax_training_reset #wrapper.wrap_for_brax_training
    network_fn = make_ppo_networks
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory)
        if hasattr(cfg, "network_factory")
        else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params


def _init_wandb(
    args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"
):
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "num_timesteps": args.num_timesteps,
            "task": args.task,
        },
        dir=PATH_LOG,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _progress(num_steps, metrics, times, total_steps, debug_mode, exp_name):
    now = time.monotonic()
    times.append(now)
    if metrics and not debug_mode:
        try:
            wandb.log(metrics, step=num_steps)
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

    if len(times) < 2 or num_steps == 0:
        return
    step_times = np.diff(times)
    median_step_time = np.median(step_times)
    if median_step_time <= 0:
        return
    steps_logged = num_steps / len(step_times)
    est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
    logging.info(f"NumSteps {num_steps} - EstTimeLeft {est_seconds_left:.1f}[s]")
    logging.info(exp_name)


def _report_training_time(times):
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def train(args: Args):
    env_class = cat_ppo.registry.get(args.task, "train_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config
    eval_config = task_cfg.eval_config

    exp_name = _prepare_exp_name(args.task, args.generate_exp_name())
    debug_mode = "debug" in exp_name

    _validate_exp_name_format(exp_name, debug_mode)

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    _apply_args_to_config(args, policy_cfg, env_cfg, debug_mode)
    task_cfg.env_config = env_cfg
    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    if not debug_mode:
        _init_wandb(args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(ppo.train, **policy_params)
    times = [time.monotonic()]

    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    _eval_env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    # process
    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
        pass

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda s, m: _progress(s, m, times, policy_cfg.num_timesteps, debug_mode, exp_name),
        eval_env=_eval_env,
        policy_params_fn=policy_params_fn,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    logging.info(f"Run {exp_name} Train done.")

    if args.convert_onnx:
        try:
            from cat_ppo.eval.brax2onnx import convert_jax2onnx, get_latest_ckpt

            ckpt_dir = get_latest_ckpt(ckpt_path)
            obs_size = {
                'privileged_state': (env_cfg.num_pri,),
                'state': (env_cfg.num_obs,),
            }
            act_size = env_cfg.num_act
            policy_obs_key = policy_cfg.network_factory.policy_obs_key
            convert_jax2onnx(
                ckpt_dir=ckpt_dir,
                output_path=f"{ckpt_dir}/policy.onnx",
                inference_fn=inference_fn,
                hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                obs_size=obs_size,
                action_size=act_size,
                policy_obs_key=policy_obs_key,
                jax_params=params,
                activation="swish",
            )
        except ImportError:
            logging.warning(
                "TensorFlow is not installed. Please install TensorFlow to use ONNX conversion."
            )


if __name__ == "__main__":
    train(tyro.cli(Args))
