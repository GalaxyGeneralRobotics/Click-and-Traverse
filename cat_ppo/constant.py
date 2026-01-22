import os
from pathlib import Path
from absl import logging

PATH_GLI_STR = os.environ.get("GLI_PATH")
if PATH_GLI_STR is None:
    raise ValueError("GLI_PATH environment variable not set.")

PATH_GLI = Path(PATH_GLI_STR)
if not PATH_GLI.exists():
    raise ValueError("GLI_PATH does not exist.")

PATH_STORAGE = PATH_GLI.parent / "data"
PATH_ASSET = PATH_STORAGE / "assets"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
PATH_LOG = PATH_STORAGE / "logs" / WANDB_PROJECT


def get_path_log(tag):
    return PATH_LOG / tag


def get_latest_ckpt(tag):
    ckpt_dir = PATH_LOG / tag / "checkpoints"
    ckpts = [
        ckpt for ckpt in Path(ckpt_dir).glob("*") if not ckpt.name.endswith(".json")
    ]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None

def get_latest_ckpt_rl(tag):
    ckpt_dir = PATH_LOG / tag / "checkpoints"
    ckpts = [
        ckpt for ckpt in Path(ckpt_dir).glob("*") if not ckpt.name.endswith(".json")
    ]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None