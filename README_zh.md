<div align="center">
  <h1 align="center"><img src="assets/icon.png" width="40" style="vertical-align: middle;">  Click and Traverse </h1>
  <h3 align="center"> 清华 · 银河 </h3>
[中文](README_zh.md) | [English](README.md)

:page_with_curl:[论文](https://arxiv.org/abs/2601.16035) | :house:[项目主页](https://axian12138.github.io/CAT/) | :film_projector:[视频](https://www.youtube.com/watch?v=blek__Qf0Vc)

</div>

本仓库**官方实现**了论文：

> **Collision-Free Humanoid Traversal in Cluttered Indoor Scenes**
>  *Han Xue et al* 
>  arXiv 预印本: [arXiv:2601.16035](https://arxiv.org/abs/2601.16035)<br>
> 项目主页: [https://axian12138.github.io/CAT/](https://axian12138.github.io/CAT/).


本项目研究如何使类人机器人在**杂乱的室内场景**中安全穿行。我们将**杂乱的室内场景**定义为同时具有：

- **全空间约束**：地面、侧向以及头顶方向的障碍同时存在，限制了类人机器人在三维空间中的移动。
- **复杂几何形状**：障碍物具有复杂、非规则的形状，而非简单的矩形或规则多面体等原始几何体。

<p align="center">
  <img src="assets/teaser.png" width="40%">
  <img src="assets/comparison.png" width="50%">
</p>

本仓库包含：

- **Humanoid Potential Field（HumanoidPF）**：一种结构化表示，用以编码机器人身体与周围障碍物之间的空间关系；
- **混合场景生成**：将真实室内场景裁剪与程序化合成障碍物相结合，生成训练场景；
- **基于强化学习的 specialist 与 generalist 策略**，分别在特定场景上训练专家策略，并蒸馏为通用策略。

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

## 目录

- [项目状态](#项目状态)
- [安装](#安装)
- [仓库结构](#仓库结构)
- [混合障碍生成与 HumanoidPF](#混合障碍生成与-humanoidpf)
- [穿行技能学习](#穿行技能学习)
- [引用](#引用)
- [许可证](#许可证)
- [贡献](#贡献)
- [致谢](#致谢)
---

## 项目状态

- [x] 🧩 程序化障碍生成与 HumanoidPF 构建
- [x] 🧩 专家策略训练代码
- [x] 🗂️ 预训练专家模型与场景数据
- [ ] 🧩 专家到通用策略的蒸馏代码
- [ ] 🗂️ 预训练通用模型
- [ ] 🗂️ 扩展的场景数据集
- [ ] 🚀 Sim-to-Real 部署工具

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Axian12138/Click-and-Traverse.git
cd Click-and-Traverse
```

### 2. 环境配置

推荐使用 CUDA 12.5。

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH  # 根据需要调整
uv sync -i https://pypi.org/simple
```

### 3. 配置

在仓库根目录创建并自定义 `.env` 文件。该文件定义运行时配置，例如：

- 工作目录路径
- 日志（例如 WandB 账户）
- 实验标识

### 4. 初始化 MuJoCo 资源

```bash
source .venv/bin/activate
source .env
python -m cat_ppo.utils.mj_playground_init
```

---

## 仓库结构

可下载的预训练检查点与场景资源（即将提供）：

- **Google Drive**: https://drive.google.com/drive/folders/1q57nJJ6uC26RmmCuxYjv6q1zE1gnVFvr

将下载的数据放置于 `data/` 目录下。

```
Click-and-Traverse/
├── LICENSE
├── README.md
├── README_zh.md
├── pyproject.toml
├── train_batch.py
├── train_ppo.py
├── .env
├── cat_ppo/                        # 核心 RL 框架
│   ├── envs/
│   ├── learning/
│   ├── eval/
│   └── utils/
├── data/                           # 资源、日志（检查点）
│   ├── assets/
│   └── logs/
└── procedural_obstacle_generation/ # 障碍生成
    ├── main.py
    ├── pf_modular.py               # HumanoidPF 构建
    ├── random_obstacle.py
    ├── typical_obstacle.py
    └── utils.py
```

---

## 混合障碍生成与 HumanoidPF

支持两类障碍场景：

- **典型障碍（Typical obstacles）**：手工设计、语义明确的场景配置；
- **随机障碍（Random obstacles）**：可控难度的程序化生成场景。

HumanoidPF 表示会与场景同步生成。

输出保存在：

- `data/assets/TypiObs/`
- `data/assets/RandObs/`

### 生成典型障碍

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
cd procedural_obstacle_generation
```

编辑 `main.py` 并调用：

```python
generate_typical_obstacle(obs_name)
```

参数：
- `obs_name`：障碍配置名称（详见 `main.py` 注释）

### 生成随机障碍

在 `main.py` 中调用：

```python
generate_random_obstacle(difficulty, seed, dL, dG, dO)
```

参数：
- `difficulty`：全局难度等级
- `seed`：随机种子
- `dL`：侧向障碍难度
- `dG`：地面障碍难度
- `dO`：头顶障碍难度

---

## 穿行技能学习

### 训练

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
python train_batch.py
```

支持的任务：

- `G1Cat`：默认任务（便于直接上真机）
- `G1CatPri`：带特权观测的任务（对蒸馏到通用策略有更有帮助）

详见 `train_batch.py` 中的参数说明。

`train_batch.py` 会自动将 checkpoints 转换为 ONNX 格式；如果你更改了策略结构，可能需要手动转换：

```bash
python -m cat_ppo.eval.brax2onnx \
  --task G1Cat \
  --exp_name exp_name
```

### 评估

评估已训练策略时，确保 MuJoCo XML（`data/assets/unitree_g1/scene_mjx_feetonly_mesh.xml`）中的障碍 `file` 路径指向目标场景。例如，如果障碍名是 `narrow1`，则替换为：

```xml
<mesh name="scene_mesh" file="../TypiObs/narrow1/obs.obj"/>
```

然后运行：

```bash
python -m cat_ppo.eval.mj_onnx_play \
  --task G1Cat \
  --exp_name 12151522_G1LocoPFR10_SlowV2OdonoiseV2_xP0xMxK00xnarrow1 \
  --obs_name narrow1
```

若使用带特权观测的模型进行评估：

```bash
python -m cat_ppo.eval.mj_onnx_play \
  --task G1CatPri --pri \
  --exp_name 01202236_G1Cat_debug_xT00xempty \
  --obs_name empty
```

---

## 引用

若本工作对你有帮助，请引用：

```bibtex
@misc{xue2026collisionfreehumanoidtraversalcluttered,
  title        = {Collision-Free Humanoid Traversal in Cluttered Indoor Scenes},
  author       = {Xue, Han and Liang, Sikai and Zhang, Zhikai and Zeng, Zicheng and Liu, Yun and Lian, Yunrui and Wang, Jilong and Liu, Qingtao and Shi, Xuesong and Li, Yi},
  year         = {2026},
  eprint       = {2601.16035},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2601.16035}
}
```

---

## 许可证

本项目根据仓库中的 LICENSE 文件发布。

---

## 贡献

欢迎贡献。请先开 issue 讨论重大修改，或直接提交 pull request。

---

## 致谢

感谢 MuJoCo Playground 提供了便利的仿真框架。

---

# 联系方式

如有讨论意向，可发送邮件至 xue-h21@mails.tsinghua.edu.cn 或添加微信：xh15158435129。