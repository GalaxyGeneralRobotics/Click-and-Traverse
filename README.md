# Collision-Free Humanoid Traversal in Cluttered Indoor Scenes

This repository contains the official implementation for the paper "Collision-Free Humanoid Traversal in Cluttered Indoor Scenes". The work addresses the challenge of enabling humanoid robots to navigate complex indoor environments by learning collision-free traversal skills through reinforcement learning. Key contributions include the Humanoid Potential Field (HumanoidPF) representation for encoding obstacle relationships and a hybrid scene generation method combining realistic 3D indoor crops with procedurally synthesized obstacles. The codebase supports training specialist policies for specific scenes and distillation to generalist policies, with successful sim-to-real transfer demonstrated on the Unitree G1 robot.

## TODOs

- [x] Code for procedurally generated obstacles and HumanoidPF construction, generating random obstacle scenes and corresponding HumanoidPF
- [x] Code for training specialist policies
- [x] Open-source pre-trained specialist policy models and corresponding scene data
- [ ] Code for specialist-to-generalist policy distillation
- [ ] Open-source pre-trained generalist policy models and corresponding scene data
- [ ] Open-source more scene data, including crops of realistic 3D indoor scenes and procedurally generated obstacles
- [ ] Code for sim-to-real deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Axian12138/Click-and-Traverse.git
cd Click-and-Traverse
```

2. Create a virtual environment and install dependencies:
```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH  # Adjust to your CUDA path, CUDA 12.5 recommended
uv sync -i https://pypi.org/simple
```

3. Customize the `.env` file in the root directory. This file stores configurations such as working directory, WandB account, etc.

4. Initialize the MuJoCo environment (create softlinks for MuJoCo assets):
```bash
source .venv/bin/activate
source .env
python -m cat_ppo.utils.mj_playground_init
```

## Repository Structure

Download pre-trained checkpoints and scene data from [Google Drive](https://drive.google.com/drive/folders/1q57nJJ6uC26RmmCuxYjv6q1zE1gnVFvr?usp=sharing). This includes obstacle scenes and corresponding specialist policy models, placed under the `data` directory.

- `cat_ppo/`: Core package containing environments, learning algorithms (PPO), evaluation tools, and utilities for training and deploying policies.
- `data/`: Directory for storing assets (obstacle scenes, HumanoidPF data), logs (model checkpoints).
- `procedural_obstacle_generation/`: Scripts for generating hybrid obstacle scenes (typical and random) and constructing corresponding HumanoidPF representations.

```
Click-and-Traverse/
|-- LICENSE
|-- pyproject.toml
|-- README.md
|-- train_batch.py
|-- train_ppo.py
|-- .env
|-- .gitignore
|-- cat_ppo/
|   |-- __init__.py
|   |-- constant.py
|   |-- envs/
|   |   |-- ...
|   |-- eval/
|   |   |-- ...
|   |-- learning/
|   |   |-- ...
|   |-- utils/
|   |   |-- ...
|-- data/
|   |-- assets/
|   |   |-- ...
|   |-- logs/
|   |   |-- ...
|-- procedural_obstacle_generation/
|   |-- main.py
|   |-- pf_modular.py
|   |-- random_obstacle.py
|   |-- typical_obstacle.py
|   |-- utills.py
|   |-- __pycache__/
```

## Hybrid Obstacle Generation & HumanoidPF Construction

We provide two types of obstacle scenarios:
- **typical_obstacle**: Manually designed obstacle scenes with a limited number.
- **random_obstacle**: Randomly generated obstacle scenes as mentioned in the paper.

### Generating Typical Obstacles

1. Activate the environment:
```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
cd procedural_obstacle_generation
```

2. Generate typical obstacles:
Call `generate_typical_obstacle('ceil0')` in `main.py`. 'ceil0' is the obstacle scene name; refer to comments in `main.py` for details.

### Generating Random Obstacles

1. Activate the environment (same as above).

2. Generate random obstacles:
Call `generate_random_obstacle(difficulty, seed, dL, dG, dO)` in `main.py`.
- `difficulty`: Overall difficulty.
- `seed`: Random seed.
- `dL`: Difficulty of lateral obstacles.
- `dG`: Difficulty of ground obstacles.
- `dO`: Difficulty of overhead obstacles.

Generated data will be saved in `data/assets/TypiObs` and `data/assets/RandObs/`, with HumanoidPF data generated synchronously in the same directories.

## Traversal Skill Learning with HumanoidPF

### Training the Model

1. Activate the environment:
```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
```

2. Train the model:
```bash
python train_batch.py
```
See `train_batch.py` for argument details. We support two tasks: `G1Cat` (for quick sim2real deployment) and `G1CatPri` (with prior knowledge, for distillation).

### Evaluating the Model

MuJoCo visualization is tied to the XML files, so modify the obstacle name in the XML to `empty` to use the eval module.

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1CatPri --pri --exp_name 01202236_G1Cat_debug_xT00xempty --obs_name empty
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name 12151522_G1LocoPFR10_SlowV2OdonoiseV2_xP0xMxK00xnarrow1 --obs_name narrow1
```

If you change the observation space, manually convert the model checkpoint to ONNX format:
```bash
python -m cat_ppo.eval.brax2onnx --task G1Cat --exp_name 01202236_G1Cat_debug_xT00xempty
```

## License

@misc{xue2026collisionfreehumanoidtraversalcluttered,
      title={Collision-Free Humanoid Traversal in Cluttered Indoor Scenes}, 
      author={Han Xue and Sikai Liang and Zhikai Zhang and Zicheng Zeng and Yun Liu and Yunrui Lian and Jilong Wang and Qingtao Liu and Xuesong Shi and Li Yi},
      year={2026},
      eprint={2601.16035},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.16035}, 
}

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.