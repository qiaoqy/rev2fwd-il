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

The complete pipeline consists of 5 scripts:

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `1_collect_data.py` | Collect reverse trajectory data | - | `data/B_*.npz` |
| `2_inspect_data.py` | Visualize and inspect data | NPZ file | PNG/MP4/JSON |
| `3_make_forward_data.py` | Time-reverse to generate forward data | `data/B_*.npz` | `data/A_*.npz` |
| `4_train_diffusion.py` | Train Diffusion Policy | `data/A_*.npz` | `runs/*/checkpoints` |
| `5_eval_diffusion.py` | Evaluate and visualize | checkpoint | `runs/*/videos` |

---

## 3. Usage Examples

### Step 1: Collect Reverse Trajectory Data

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/1_collect_data.py \
    --headless --num_episodes 500 --num_envs 256 \
    --out data/B_2images_goal.npz
```

### Step 2: Visualize and Inspect Data

```bash
python scripts/2_inspect_data.py \
    --dataset data/B_2images_goal.npz \
    --episode 0 --enable_xyz_viz
```

### Step 3: Generate Forward Training Data

```bash
python scripts/3_make_forward_data.py \
    --input data/B_2images_goal.npz \
    --out data/A_2images_goal.npz
```

### Step 4: Train Diffusion Policy

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 python scripts/4_train_diffusion.py \
    --dataset data/A_2images_goal.npz \
    --out runs/diffusion_A_goal \
    --batch_size 64 --steps 50000 \
    --include_obj_pose --wandb

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/4_train_diffusion.py \
    --dataset data/A_2images_goal.npz \
    --out runs/diffusion_A_goal \
    --batch_size 64 --steps 50000 \
    --include_obj_pose --wandb
```

### Step 5: Evaluate and Visualize

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/5_eval_diffusion.py \
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

## 5. Archived Scripts

Legacy scripts have been moved to `scripts_archive/`, including:
- MLP BC training/evaluation
- Early data collection scripts
- Debug utilities
