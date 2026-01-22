import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass

import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo


@dataclass
class Args:
    task: str
    exp_name: str = None
    seed: int = 42
    onnx_path: str = None
    pri: bool = False
    pf_name: str = 'empty'


@dataclass
class State:
    info: dict
    obs: dict
    

def play(args: Args):
    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = f'data/assets/RandObs/{args.pf_name}' if args.pf_name.startswith('D') else f'data/assets/TypiObs/{args.pf_name}'
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    env.pri = args.pri
    if args.onnx_path is not None:
        onnx_path = args.onnx_path
    else:
        ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
        onnx_path = ckpt_path / "policy.onnx"
    output_names = ["continuous_actions"]
    policy = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    state = env.reset()
    _ctr = 0
    # from plot import ActionPlotter
    #
    # plotter = ActionPlotter(num_dim=5)

    while True:
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        
        onnx_input = {"obs": obs}
        action = policy.run(output_names, onnx_input)[0]
        
        action = action[0]
        state = env.step(state, action)
        # a = np.concatenate([state.info["feetgf"].reshape(-1), state.info["feet_vel"].reshape(-1), feet_r.reshape(-1)])
        # plotter.add_action(a,idx=[2,5,8,11,12])  # 添加当前action

        _ctr += 1


if __name__ == "__main__":
    args = tyro.cli(Args)
    play(args)
