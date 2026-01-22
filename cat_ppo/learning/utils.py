from datetime import datetime
from pathlib import Path

from cat_ppo import update_file_handler
from cat_ppo.constant import PATH_LOG


def prepare_exp_name(task, exp_tag: str) -> str:
    timestamp = datetime.now().strftime("%m%d%H%M")
    return f"{task}_{timestamp}_{exp_tag}"


def validate_exp_name_format(exp_name: str, debug_mode: bool):
    if not debug_mode and len(exp_name.split("_")) != 3:
        raise ValueError(
            f"exp_name should be in the format <task>_<tag>_<version>, got {exp_name}"
        )


def setup_paths(exp_name: str, mkdir: bool = True) -> tuple[Path, Path]:
    logdir = Path(PATH_LOG) / exp_name
    ckpt_path = logdir / "checkpoints"
    if mkdir:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        logdir.mkdir(parents=True, exist_ok=True)
        update_file_handler(filename=f"{logdir}/info.log")
    return logdir, ckpt_path
