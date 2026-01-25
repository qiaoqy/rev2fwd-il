# Rev2Fwd Imitation Learning Pipeline

> **Purpose**: This document serves as a specification to implement the remaining scripts (`6_test_alternating.py`, `7_finetune_with_rollout.py`, `run_iterative_training.sh`).

---

## Overview

This project implements a **Reverse-to-Forward (Rev2Fwd)** imitation learning approach for robotic Pick-and-Place tasks.

**Core Idea**:
1. Collect expert data for the **easier** Task B using a state machine
2. **Time-reverse** Task B trajectories to automatically obtain Task A data
3. Train separate Diffusion Policies for Task A and Task B
4. Iteratively improve policies through test-collect-retrain loops

---

## Method

### Task Definitions

| Task | Description | Initial State | Difficulty |
|------|-------------|---------------|------------|
| **Task A** | Pick object from **arbitrary position** on table → Place at **fixed target** (e.g., ring center) | Object anywhere on table | **Hard** (large state space) |
| **Task B** | Pick object from **fixed target** → Place at **arbitrary position** on table | Object at fixed target | **Easy** (small state space) |

### Key Insight: Time Reversibility

Pick-and-Place trajectories are **temporally reversible**:
- **Forward**: approach → descend → grasp → lift → move → descend → release
- **Reverse**: the exact inverse of the above

**Therefore**:
1. Implement a simple state machine to collect Task B data (starts from fixed position, easy to implement)
2. Time-reverse Task B trajectories → automatically get Task A data (starts from arbitrary position)

This way, we only need ONE simple expert to generate training data for BOTH tasks.

### Data Format

Each trajectory (NPZ file) contains:

| Field | Shape | Description |
|-------|-------|-------------|
| `images` | (T, H, W, 3) | Table camera RGB images (uint8) |
| `wrist_images` | (T, H, W, 3) | Wrist camera RGB images (uint8) |
| `ee_pose` | (T, 7) | End-effector pose [x, y, z, qw, qx, qy, qz] |
| `obj_pose` | (T, 7) | Object pose [x, y, z, qw, qx, qy, qz] |
| `action` | (T, 8) | Target action [ee_pose (7), gripper (1)] |

---

## Code Architecture

```
scripts/scripts_pick_place/
├── PIPELINE_README.md              # This document (specification)
├── 1_collect_data_pick_place.py    # Data collection: state machine expert for Task B
├── 2_inspect_data.py               # Data inspection: visualize and validate data
├── 3_make_forward_data.py          # Data processing: time-reverse B→A, format conversion
├── 4_train_diffusion.py            # Model training: Diffusion Policy with LeRobot
├── 5_eval_diffusion.py             # Model evaluation: single-task evaluation
├── 6_test_alternating.py           # ✅ Alternating test: A→B→A→B... loop with rollout collection
├── 7_finetune_with_rollout.py      # ✅ Incremental finetuning with rollout data
└── run_iterative_training.sh       # ✅ Automated iterative training loop
```

---

## Pipeline Details

### Steps 1-3: Data Collection & Preprocessing (DONE)

```
State Machine Expert      Time Reversal + Format
      (Task B)                Conversion
        ↓                        ↓
  B_circle.npz  ──────────→  A_circle.npz
```

### Step 4: Policy Training (DONE)

```bash
# Train Task A policy (pick from anywhere → place at target)
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29500 --rdzv_id=job1 \
    scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/A_circle.npz --out runs/diffusion_A_circle \
    --batch_size 32 --steps 8000 --include_obj_pose --wandb

# Train Task B policy (pick from target → place anywhere)
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 --master_port=29501 --rdzv_id=job2 \
    scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/B_circle.npz --out runs/diffusion_B_circle \
    --batch_size 32 --steps 8000 --include_obj_pose --wandb
```

---

## Scripts To Implement

> **Note**: All scripts below have been implemented and tested.

### Step 6: `6_test_alternating.py` - Alternating Test

**Purpose**: Execute A→B→A→B... alternating loop on a **single robot arm** in **one Isaac environment** until failure. Collect rollout data for both tasks.

**Features**:
- Single environment execution (no reset between tasks)
- Visual markers: Red marker (Task B target, updates after each Task A), Green marker (fixed goal)
- Automatic rollout data collection for both tasks
- Configurable success thresholds

#### Usage Examples

```bash
# Basic usage (headless mode)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_circle_iter1.npz \
    --out_B data/rollout_B_circle_iter1.npz \
    --max_cycles 50 --headless

# With custom thresholds
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_iter1.npz \
    --out_B data/rollout_B_iter1.npz \
    --max_cycles 100 \
    --horizon 400 \
    --height_threshold 0.15 \
    --distance_threshold 0.05 \
    --headless

# With visualization (non-headless, for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_debug.npz \
    --out_B data/rollout_B_debug.npz \
    --max_cycles 10
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy_A` | (required) | Path to Task A checkpoint |
| `--policy_B` | (required) | Path to Task B checkpoint |
| `--out_A` | `data/rollout_A_iter.npz` | Output path for Task A rollout data |
| `--out_B` | `data/rollout_B_iter.npz` | Output path for Task B rollout data |
| `--max_cycles` | 50 | Maximum A→B cycles to attempt |
| `--horizon` | 400 | Maximum steps per task |
| `--height_threshold` | 0.15 | Min z-position for "lifted" |
| `--distance_threshold` | 0.05 | Max distance for "at target" |
| `--goal_xy` | 0.5 0.0 | Goal XY position (green marker) |
| `--headless` | False | Run without visualization |

---

### Step 7: `7_finetune_with_rollout.py` - Incremental Finetuning

**Purpose**: Merge original training data with newly collected rollout data, then continue training from existing checkpoint.

#### Usage Examples

```bash
# Finetune Task A policy (single GPU)
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/A_circle.npz \
    --rollout_data data/rollout_A_circle_iter1.npz \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A_circle \
    --steps 5000 \
    --include_obj_pose

# Finetune Task B policy (single GPU)
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/B_circle.npz \
    --rollout_data data/rollout_B_circle_iter1.npz \
    --checkpoint runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_B_circle \
    --steps 5000 \
    --include_obj_pose

# Multi-GPU finetuning (recommended for faster training)
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \
    scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/A_circle.npz \
    --rollout_data data/rollout_A_circle_iter1.npz \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A_circle \
    --steps 5000 \
    --include_obj_pose
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--original_data` | (required) | Path to original training data NPZ |
| `--rollout_data` | (required) | Path to rollout data from script 6 |
| `--checkpoint` | (required) | Path to checkpoint to resume from |
| `--out` | (required) | Output directory for continued training |
| `--steps` | 5000 | Additional training steps |
| `--batch_size` | 32 | Training batch size |
| `--include_obj_pose` | False | Include object pose in observations |
| `--wandb` | False | Enable W&B logging |

---

### Bash Script: `run_iterative_training.sh`

**Purpose**: Automate the complete test→collect→finetune loop for iterative policy improvement.

#### Usage

```bash
# Make executable (first time only)
chmod +x scripts/scripts_pick_place/run_iterative_training.sh

# Run with default configuration
CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_pick_place/run_iterative_training.sh

# Or set GPU in the script and run directly
bash scripts/scripts_pick_place/run_iterative_training.sh
```

#### Configuration Variables

Edit the script header to customize:

```bash
# =============================================================================
# Configuration (edit these variables)
# =============================================================================
MAX_ITERATIONS=10           # Maximum number of test-finetune iterations
STEPS_PER_ITER=5000         # Training steps per finetuning iteration
MAX_CYCLES=50               # Maximum A→B cycles per alternating test
HORIZON=400                 # Maximum steps per task attempt
BATCH_SIZE=32               # Training batch size

# Thresholds for success detection
HEIGHT_THRESHOLD=0.15       # Minimum object z-position to consider lifted
DISTANCE_THRESHOLD=0.05     # Maximum distance from target for success

# Policy directories
POLICY_A_DIR="runs/diffusion_A_circle"
POLICY_B_DIR="runs/diffusion_B_circle"

# Original training data
DATA_A="data/A_circle.npz"
DATA_B="data/B_circle.npz"
```

#### What the Script Does

For each iteration:

1. **Alternating Test** (`6_test_alternating.py`)
   - Loads latest checkpoints for Policy A and B
   - Runs A→B→A→B... cycles until failure
   - Saves rollout data: `data/rollout_A_circle_iter{N}.npz`, `data/rollout_B_circle_iter{N}.npz`

2. **Finetune Policy A** (`7_finetune_with_rollout.py`)
   - Merges original A data + rollout A data
   - Continues training from checkpoint for `STEPS_PER_ITER` steps

3. **Finetune Policy B** (`7_finetune_with_rollout.py`)
   - Merges original B data + rollout B data
   - Continues training from checkpoint for `STEPS_PER_ITER` steps

4. **Repeat** until `MAX_ITERATIONS` reached or no data collected

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `consecutive_success` | Number of successful A→B cycles before first failure |
| `A_success_rate` | Task A success rate (successful / attempted) |
| `B_success_rate` | Task B success rate (successful / attempted) |

---

## Quick Start

```bash
# 1. Activate environment
conda activate rev2fwd_il

# 2. Initial training (DONE - run if starting fresh)
# Train Task A policy
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29500 --rdzv_id=job1 \
    scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/A_circle.npz --out runs/diffusion_A_circle \
    --batch_size 32 --steps 20000 --include_obj_pose --wandb

# Train Task B policy
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 --master_port=29501 --rdzv_id=job2 \
    scripts/scripts_pick_place/4_train_diffusion.py \
    --dataset data/B_circle.npz --out runs/diffusion_B_circle \
    --batch_size 32 --steps 20000 --include_obj_pose --wandb

# 3. Test trained policies with alternating test (single run)
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_circle_iter1.npz \
    --out_B data/rollout_B_circle_iter1.npz \
    --max_cycles 50 --headless

# 4. Run full iterative training loop (automated)
chmod +x scripts/scripts_pick_place/run_iterative_training.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_pick_place/run_iterative_training.sh
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Policy checkpoint not found` | Ensure initial training completed. Check path: `runs/diffusion_X_circle/checkpoints/checkpoints/last/pretrained_model` |
| `No rollout data collected` | Policy failed immediately. Try reducing `distance_threshold` or check policy training |
| `CUDA out of memory` | Reduce `--batch_size` or use fewer GPUs |
| `Isaac Lab hangs` | Use `--headless` mode, ensure `--enable_cameras` is set |

### Verifying Checkpoints

```bash
# Check if checkpoint exists
ls -la runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model/

# Should contain:
# - config.json
# - model.safetensors
# - preprocessors/
```

## References

- [LeRobot](https://github.com/huggingface/lerobot): Diffusion Policy implementation
- [DAgger](https://arxiv.org/abs/1011.0686): Dataset Aggregation algorithm
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/): Simulation environment
