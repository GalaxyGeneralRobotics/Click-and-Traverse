import importlib.util
from pathlib import Path
import shutil
from cat_ppo.constant import PATH_ASSET


def create_mujoco_menagerie_soft_link():
    spec = importlib.util.find_spec("mujoco_playground").origin
    mj_play_dir = Path(spec).parent
    tgt_path = Path(mj_play_dir) / "external_deps" / "mujoco_menagerie"

    if tgt_path.exists() and len(list(tgt_path.iterdir())) > 1:
        print(f"{tgt_path} already exists")
        return
    src_path = PATH_ASSET / "mujoco_menagerie"
    print(src_path)
    if not src_path.exists():
        raise FileExistsError(src_path, "source path does not exist")

    # remove the target dir if it exists
    if tgt_path.exists():
        # remove a softlink
        if tgt_path.is_symlink():
            tgt_path.unlink()
        else:
            shutil.rmtree(tgt_path)
    tgt_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_path.symlink_to(src_path, target_is_directory=True)
    print(f"Link {src_path} to {tgt_path}")


if __name__ == "__main__":
    create_mujoco_menagerie_soft_link()
