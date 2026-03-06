# rev2fwd-il: Reverse-to-Forward Imitation Learning (Isaac Lab)

> Core idea: Run an easy reverse policy B (goal → table), collect reverse trajectories,
> then time-reverse them to generate forward data for training policy A (table → goal).

---

## 1. Installation

```bash
conda create -n rev2fwd_il python=3.11
conda activate rev2fwd_il
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
pip install wandb==0.24.0 
pip install lerobot==0.4.2
# pip install numpy==1.26.0
pip install -e ./isaaclab_tasks
pip install -e .
```

### (Optional) Piper SDK for Real-Robot Experiments

If you plan to run the real Piper arm scripts under [scripts/scripts_piper_local](scripts/scripts_piper_local), install the Piper SDK and its dependencies:

```bash
pip install -e ./piper_sdk
pip install piper_sdk>=0.1.9
pip install python-can>=4.3.1
pip install opencv-python==4.9.0.80
pip install scipy
pip install keyboard
# pip install numpy==1.26.0
pip install pygame==2.6.1
pip install pydualsense 
sudo apt-get install -y libhidapi-hidraw0 libhidapi-dev
pip install hidapi
```

### (Optional) DiT Flow Policy for Real-Robot Experiments

To use DiT Flow (Diffusion Transformer + Flow Matching) as an alternative to Diffusion Policy, install the following:

```bash
# 1. LeRobot (fork with third-party plugin support)
git clone https://github.com/danielsanjosepro/lerobot.git
pip install -e ./lerobot

# 2. DiT Flow Policy plugin (included as a git submodule, pinned at commit fc8db68)
git submodule update --init --recursive
pip install -e ./lerobot_policy_ditflow
```

> **Note**: `lerobot_policy_ditflow` is included as a git submodule (pinned to commit `fc8db68`) with local patches for lerobot API compatibility. It registers itself automatically via LeRobot's `register_third_party_plugins()`. See [Script 7](scripts/scripts_piper_local/README.md#12-script-7-train-dit-flow-policy) for training instructions.

### Pinned Environment (snapshot 2026-02-09)

The `lerobot_policy_ditflow` library does **not** use strict version tags.
Below are the exact commits and package versions used for the current working
environment (training run `ditflow_piper_teleop_B_0206`, 200k steps).
Use these to reproduce an identical setup.

```bash
# ── Core ──
conda create -n rev2fwd_il python=3.11.14
conda activate rev2fwd_il
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── LeRobot (PyPI release, NOT the danielsanjosepro fork) ──
pip install lerobot==0.4.2

# ── DiT Flow Policy plugin (pinned to exact commit) ──
#   ⚠️  This library has NO version tags — the commit hash is the only
#       reliable identifier.  Different commits change the model architecture
#       (e.g., adding nn.Sequential wrapping), which breaks checkpoint loading.
# lerobot_policy_ditflow is included as a git submodule (pinned at fc8db68)
git submodule update --init --recursive
pip install -e ./lerobot_policy_ditflow

# ── Other key packages (pinned) ──
pip install numpy==1.26.0
pip install scipy==1.15.3
pip install safetensors==0.7.0
pip install diffusers==0.35.2
pip install accelerate==1.12.0
pip install einops==0.8.2
pip install draccus==0.10.0
pip install av==15.1.0
pip install torchcodec==0.5
pip install wandb==0.21.4
pip install opencv-python==4.9.0.80
pip install pygame==2.6.1
```

**Quick reference of pinned identifiers:**

| Package | Version / Commit | Notes |
|---------|-----------------|-------|
| `lerobot` | `0.4.2` (PyPI) | Standard release, no fork needed |
| `lerobot_policy_ditflow` | commit [`fc8db68`](https://github.com/danielsanjosepro/lerobot_policy_ditflow/commit/fc8db684e933c883550df406b797499b3819a644) | **No version tags** — must pin by commit |
| `torch` | `2.7.0+cu128` | CUDA 12.8 |
| `torchvision` | `0.22.0+cu128` | |
| `numpy` | `1.26.0` | |
| `safetensors` | `0.7.0` | Checkpoint I/O |
| `diffusers` | `0.35.2` | Used by lerobot internals |

> **Why commit pinning matters for `lerobot_policy_ditflow`**: The library's
> `_DiTDecoder` class changed between commits — newer versions wrap `linear1`/
> `linear2` inside an `nn.Sequential` called `mlp`, which creates extra
> state_dict keys (`mlp.0.*`, `mlp.3.*`).  Checkpoints trained with an older
> commit will fail to load on a newer commit with `strict=True`.  Script 8 now
> handles this gracefully with `strict=False`, but pinning the commit avoids
> the mismatch entirely.

---

## 2. Workflow

### 2.1 Individual Scripts

The core pipeline consists of these Python scripts (under `scripts/scripts_pick_place/`):

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `1_collect_data_pick_place.py` | Collect reverse trajectory data | - | `data/B_*.npz` |
| `2_inspect_data.py` | Visualize and inspect data | NPZ file | PNG/MP4/JSON |
| `3_make_forward_data.py` | Time-reverse to generate forward data | `data/B_*.npz` | `data/A_*.npz` |
| `4_train_diffusion.py` | Train Diffusion Policy | `data/A_*.npz` | `runs/*/checkpoints` |
| `5_eval_diffusion.py` | Evaluate and visualize | checkpoint | `runs/*/videos` |
| `7_finetune_with_rollout.py` | Prepare data + checkpoint for finetuning | lerobot dataset + rollout npz | merged dataset |
| `9_eval_with_recovery.py` | Evaluate A-B cycles with failure recovery | two checkpoints | rollout npz + stats json |

### 2.2 Iterative Training Pipeline (Pick & Place, Isaac Lab)

Two bash scripts orchestrate the full iterative training loop. Both are **auto-resumable** — on crash or interrupt, simply re-run the same command to pick up where it left off.

| Script | Mode | Description |
|--------|------|-------------|
| `run_pipeline.sh` | **DAgger** | Evaluate → aggregate successful rollout data → finetune → repeat |
| `run_ablation.sh` | **Baseline** | Evaluate → finetune on original data only (no rollout aggregation) → repeat |

**Experiment directory structure:**

All results are stored under `data/pick_place_isaac_lab_simulation/exp{N}/` with auto-incrementing experiment numbers:

```
data/pick_place_isaac_lab_simulation/
  exp1/                              # First experiment
    config.json                      # Mode, hyperparams, source policies
    record.json                      # Per-iteration success rate metrics
    success_rate_curve.png           # Generated plot
    pipeline.log  or  ablation.log   # Full console output
    iter1_eval_A.npz                 # Rollout data (Task A, iteration 1)
    iter1_eval_A.stats.json          # Evaluation statistics
    iter1_eval_B.npz                 # Rollout data (Task B, iteration 1)
    ...
    .done_iter1_eval                 # Phase markers (for auto-resume)
    .done_iter1_train_A
    .done_iter1_train_B
    .complete                        # Set when all iterations finish
  exp2/                              # Next experiment (auto-numbered)
    ...
```

Working checkpoint directories are scoped per-experiment in `runs/`:
```
runs/
  exp1_PP_A_temp/                    # Policy A working directory for exp1
  exp1_PP_B_temp/                    # Policy B working directory for exp1
```

**Usage:**

```bash
# Run the DAgger pipeline (aggregates rollout data)
bash scripts/scripts_pick_place/run_pipeline.sh

# Run the ablation baseline (no rollout aggregation, isolates training-step effect)
bash scripts/scripts_pick_place/run_ablation.sh

# On crash, just re-run the same command — resumes automatically
bash scripts/scripts_pick_place/run_pipeline.sh
```

**Auto-resume mechanism:**

Each iteration has 3 phases (evaluate, train_A, train_B). After each phase completes, a marker file (`.done_iter{N}_{phase}`) is written. On restart, completed phases are skipped. Directory state is also recovered (e.g., if a crash occurs mid-rotation between `last → temp`).

---

## 3. Usage Examples (Individual Scripts)

### Step 1: Collect Reverse Trajectory Data

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/1_collect_data_pick_place.py \
    --headless --num_episodes 500 --num_envs 256 \
    --out data/B_2images_goal.npz
```

### Step 2: Visualize and Inspect Data

```bash
python scripts/scripts_pick_place/2_inspect_data.py \
    --dataset data/B_2images_goal.npz \
    --episode 0 --enable_xyz_viz
```

### Step 3: Generate Forward Training Data

```bash
python scripts/scripts_pick_place/3_make_forward_data.py \
    --input data/B_2images_goal.npz \
    --out data/A_2images_goal.npz
```

### Step 4: Train Diffusion Policy

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/A_2images_goal.npz \
    --out runs/diffusion_A_goal \
    --batch_size 64 --steps 50000 \
    --include_obj_pose --wandb

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/A_2images_goal.npz \
    --out runs/diffusion_A_goal \
    --batch_size 64 --steps 50000 \
    --include_obj_pose --wandb
```

### Step 5: Evaluate and Visualize

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/5_eval_diffusion.py \
    --checkpoint runs/diffusion_A_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_goal/videos \
    --num_episodes 5 --visualize_xyz --headless
```

---

## 4. Data Format

### Reverse Data (B_*.npz)

```
obs:           (T, 36)      State observations
images:        (T, H, W, 3) Table camera RGB images
wrist_images:  (T, H, W, 3) Wrist camera RGB images
ee_pose:       (T, 7)       End-effector pose [x,y,z,qw,qx,qy,qz]
obj_pose:      (T, 7)       Object pose
action:        (T, 8)       Goal action [ee_pose, gripper]
gripper:       (T,)         Gripper action (+1=open, -1=close)
fsm_state:     (T,)         FSM state
```

### Forward Data (A_*.npz)

Same format, but trajectory direction is reversed (from random table position to goal).

---

## 5. Nut Threading Task (FORGE Environment)

### 5.1 PhysX Collision Volume 问题（2026-03-05）

**现象**：夹爪和螺母明明没有接触，但仍然被检测为碰撞（contact penalty 被触发）。

**根因**：PhysX 的 `contact_offset` 参数决定了接触检测的"提前量"——两个物体之间的间距小于 `contact_offset_A + contact_offset_B` 时就会生成接触对。Isaac Lab Factory 任务中，所有碰撞体的默认 `contact_offset = 0.005m (5mm)`，导致：

| 碰撞对 | 间隙阈值 | 说明 |
|---------|----------|------|
| 夹爪 ↔ 螺母 | 5mm + 5mm = **10mm** | 等于螺母整体高度 (10mm)！ |
| 螺母 ↔ 螺栓 | 5mm + 5mm = **10mm** | 5倍于螺纹螺距 (2mm) |

也就是说螺母在距离夹爪还有 10mm 时就会被算作碰撞。

**碰撞体详情**（运行时通过 `debug_collision.py` 获取）：

| 物体 | contact_offset | mesh_type | 备注 |
|------|---------------|-----------|------|
| Robot (panda_link0 ~ rightfinger) | 0.005m | convexHull | 所有12个link |
| Bolt (FixedAsset, BoltM16) | 0.005m | **SDF** | 24mm直径, 25mm高 |
| Nut (HeldAsset, NutM16) | 0.005m | **SDF** | 24mm直径, 10mm高 |

> SDF (Signed Distance Field) 碰撞体精度高但计算开销大；convexHull 是凸包近似。

**修复**（需修改 site-packages 中的实际加载文件，见下文 §5.2 模块遮蔽问题）：

| 物体 | 修改前 | 修改后 | 间隙阈值变化 |
|------|--------|--------|-------------|
| Robot (所有link) | 5mm | **1mm** | 夹爪↔螺母: 10mm → **1.5mm** |
| Bolt (FixedAsset) | 5mm | **0.5mm** | 螺母↔螺栓: 10mm → **1mm** |
| Nut (HeldAsset) | 5mm | **0.5mm** | |

修改的文件（**两处都要改**，见 §5.2）：

```
# factory_env_cfg.py (robot contact_offset)
collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)

# factory_tasks_cfg.py (NutThread 的 fixed_asset 和 held_asset)
collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)
```

**调试工具**：`scripts/scripts_nut/debug_collision.py` 可以在运行时检查所有碰撞体的实际属性：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_nut/debug_collision.py --headless --disable_fabric 1
```

输出包括每个碰撞体的 `contact_offset`、`rest_offset`、`mesh_type`、几何尺寸等信息。

### 5.2 Isaac Lab Task 模块遮蔽问题

**现象**：修改了 workspace 中的 `isaaclab_tasks/` 配置文件后，运行时完全不生效。

**根因**：`isaaclab_tasks` 包存在两份拷贝，Python import 加载的是 isaaclab 主包内嵌的副本，而不是 workspace 中 `pip install -e ./isaaclab_tasks` 安装的版本：

```
# ✅ workspace 中的 editable install（我们修改的）
/mnt/.../workspace/rev2fwd-il/isaaclab_tasks/isaaclab_tasks/direct/factory/

# ❌ 实际被 import 的（isaaclab 包自带的副本）
$CONDA_PREFIX/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/
```

**原因**：`isaaclab` 主包在安装时把 `isaaclab_tasks` 的源码也打包进了自己的 `source/` 目录，并在 `sys.path` 中注册了该路径。由于 site-packages 中的路径优先级高于 editable install 的 `.pth` 文件，所以 workspace 中的修改永远不会被加载。

**解决方案**：

1. **修改实际加载的文件**（推荐，当前采用的方案）：

```bash
# 找到实际加载路径
python -c "import isaaclab_tasks.direct.factory.factory_tasks_cfg as m; print(m.__file__)"

# 直接修改该文件
# $CONDA_PREFIX/lib/python3.11/site-packages/isaaclab/source/isaaclab_tasks/...
```

2. **同步修改两处**：workspace 中的修改用于版本管理（git），site-packages 中的修改用于运行时生效。

3. **验证模块来源**：任何涉及 `isaaclab_tasks` 的修改都应验证实际加载路径：

```python
import isaaclab_tasks.direct.factory.factory_tasks_cfg as m
print(m.__file__)  # 检查是否来自预期路径
```

> ⚠️ **注意**：`simulation_context.py` 的 Camera 渲染 patch（见 §5.3）同样受此问题影响，需要修改 site-packages 中的文件。

### 5.3 Camera Rendering Fix (2026-03-03)

在 Isaac Lab 2.3.0 中使用 FORGE 环境 + `Camera` sensor 采集图像时，发现**相机画面完全不会随机械臂运动更新**。经过深入排查，发现了以下问题：

**根因**：Isaac Lab 的 `Camera` sensor 要求 `--disable_fabric 1`，但关闭 Fabric 后，`SimulationContext.forward()` 方法会完全跳过 `update_articulations_kinematic()` 调用（因为 `_fabric_iface is None`），导致 PhysX 中的关节体 mesh 永远不会同步到 USD 渲染器。相机拍到的始终是初始帧。

**修复方案**（两处改动）：

1. **Patch Isaac Lab 源码**（安全兜底）：修改 `simulation_context.py` 的 `forward()` 方法，使 `update_articulations_kinematic()` 不再依赖 `_fabric_iface` 是否存在，始终执行同步。
   - 文件位置：`$CONDA_PREFIX/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py`
   - 备份：`simulation_context.py.bak`

2. **数据采集脚本切换为 `TiledCameraCfg`**（主要修复）：`TiledCamera` 通过 Hydra 渲染管线读取图像，可在 Fabric 开启时正常工作。`forward()` 此时会正确执行 `update_articulations_kinematic()` + `_update_fabric()`，确保 PhysX→Fabric→Hydra 全链路同步。
   - 修改文件：`scripts/scripts_nut/1_collect_data_nut_thread.py`、`scripts/scripts_nut/1_1_collect_data_nut_unthread.py`
   - `CameraCfg` → `TiledCameraCfg`
   - 采集时**不再需要** `--disable_fabric 1`

**验证**：通过连通分量分析（connected component analysis）证实，机械臂从 home 位置到 arm_up 位置（末端执行器移动 40+ cm），相机画面中最大连通变化区域为 461 像素的连贯形状，而随机噪声基线仅 3 像素。

### 5.4 采集 Nut Threading 数据

```bash
# 基本用法（不需要 --disable_fabric）
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 --out data/nut_thread.npz

# 自定义图像尺寸
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 \
    --image_width 128 --image_height 128 --out data/nut_thread_128.npz
```

### 5.5 NPZ 大文件 CRC-32 校验错误（2026-03-06）

**现象**：`np.load('data/nut_thread_0305.npz', allow_pickle=True)` 时抛出：

```
zipfile.BadZipFile: Bad CRC-32 for file 'episodes.npy'
```

**根因**：`np.savez_compressed` 将所有 episode 序列化为一个 pickle 对象数组（`episodes.npy`），使用 zlib 压缩后写入 ZIP 容器。当未压缩数据超过 **4GB**（此处为 5.93GB / 100 episodes × 600 steps × 128×128 images），ZIP 内部的 CRC-32 校验和会因为 32-bit 整数溢出而产生错误值。文件内容本身是完整的。

**修复方法**：

```python
import zipfile
# Monkey-patch 跳过 CRC 校验
zipfile.ZipExtFile._update_crc = lambda self, newdata: None

import numpy as np
data = np.load('data/nut_thread_0305.npz', allow_pickle=True)
episodes = data['episodes']  # 正常读取 100 episodes

# 用不压缩的 np.savez 重新保存（避免 CRC 问题）
np.savez('data/nut_thread_0305_resaved.npz', episodes=episodes)
```

**预防**：对于大数据集（>100 episodes + images），建议：
1. 使用 `np.savez`（不压缩）代替 `np.savez_compressed`，避免 CRC 溢出
2. 或将 episodes 拆分为多个小文件保存（如每 20 个 episode 一个文件）
3. 或改用 HDF5 格式（`h5py`），天然支持大文件和分块读取

### 5.6 可视化 Nut Threading 数据

```bash
python scripts/scripts_nut/2_inspect_nut_data.py \
    --dataset data/nut_thread.npz \
    --episode 0 --enable_force_plot --enable_trajectory_plot
```

---

## 6. Archived Scripts

Legacy scripts have been moved to `scripts_archive/`, including:
- MLP BC training/evaluation
- Early data collection scripts
- Debug utilities
