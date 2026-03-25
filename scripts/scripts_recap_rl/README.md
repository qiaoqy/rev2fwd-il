# RECAP RL — Advantage-Conditioned Diffusion Policy

> **RECAP** (**Re**verse-**C**urriculum **A**dvantage **P**olicy) improves a pretrained Diffusion Policy by conditioning it on per-frame advantage scores — telling the policy at inference time to reproduce only its historically-successful behaviours.

---

## Table of Contents

1. [Background](#1-background)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Directory Structure](#3-directory-structure)
4. [Data Format](#4-data-format)
5. [Step-by-Step Usage](#5-step-by-step-usage)
6. [Pipeline Script](#6-pipeline-script)
7. [Key Hyperparameters](#7-key-hyperparameters)
8. [Architecture Details and Design Rationale](#8-architecture-details-and-design-rationale)
9. [Relationship to Baseline](#9-relationship-to-baseline)
10. [Using exp17 (DAgger) Checkpoints and Data](#10-using-exp17-dagger-checkpoints-and-data)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Background

The baseline system (see `scripts/scripts_pick_place_simulator/`) trains a Diffusion Policy with the **Rev2Fwd** curriculum: demos are reversed so the policy first learns to approach the goal and gradually extends to the full trajectory.

After baseline training converges, the policy has a non-trivial success rate but still fails on a fraction of episodes. RECAP identifies *why* it fails and fine-tunes the policy to avoid those failure modes.

Key insight: we do **not** run RL in the standard sense (no policy gradient, no reward shaping). Instead we:
1. Collect rollout trajectories (both successful and failed).
2. Learn a value function that predicts how good a state-action frame is.
3. Label each frame with a **binary advantage indicator** (good=1 / bad=0).
4. Fine-tune the policy with **Classifier-Free Guidance** (CFG) conditioning on the indicator.
5. At inference, always condition on indicator=1 → the policy reproduces only good frames.

---

## 2. Algorithm Overview

```
Pretrained policy π₀
        │
        ▼
┌────────────────────┐
│  1. Collect ALL    │   Run π₀ for N episodes, save every episode
│     rollouts       │   (success + failure), record success flag
└────────┬───────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│  2. Train distributional value function V_φ        │
│     • Architecture: frozen_rgb_encoder + ValueHead │
│     • Input: single-frame observation (image+state)│
│     • Output: distribution over 32 return bins     │
│     • Loss: cross-entropy on normalised returns    │
│     • Returns: 0 (success at t=0) → -1 (full fail)│
└────────┬───────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│  3. Compute per-frame advantages                   │
│     • Advantage(t) = R(t) - V_φ(s_t)              │
│     • Indicator(t) = 1 if advantage > threshold   │
│     • Threshold = 30th percentile of all advantages│
└────────┬───────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│  4. Fine-tune policy with CFG conditioning         │
│     • Expand state dim 15 → 16 (append indicator) │
│     • null_prob=0.2: randomly replace with 0.5    │
│       (CFG null token = "unknown advantage")       │
│     • Migrate checkpoint weights (zero-pad UNet)   │
│     • Fine-tune for 15k steps on demo+rollout data │
└────────┬───────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│  5. Evaluate RECAP policy                          │
│     • Always inject indicator=1.0 at inference     │
│     • Policy learns: "given indicator=1, act well" │
└────────────────────────────────────────────────────┘
```

### Return Normalisation

For a trajectory of length T:
- **Success**: $R(t) = -(t / T_{\max})$, so $R \in [-1, 0]$; early success → higher return
- **Failure**: $R(t) = -1$ for all frames (terminal penalty of $C_{fail} = 1200$)

Where $T_{\max} = 1200$ steps. This creates a strong signal distinguishing good from bad behaviour.

### Advantage Formula

$$A(t) = R(t) - \hat{V}_\phi(s_t)$$

Frames where the policy did *better than expected* ($A > \theta_{30}$) receive indicator=1.

---

## 3. Directory Structure

```
scripts/scripts_recap_rl/
├── utils.py                  # Shared utilities (value function, checkpointing, migration)
├── 1_collect_rollouts.py     # Collect rollout data (all episodes) from the simulator
├── 2_train_value_function.py # Train distributional value function offline
├── 3_compute_advantages.py   # Compute per-frame advantages, write indicators to NPZ
├── 4_retrain_with_recap.py   # Fine-tune DP with advantage conditioning (state dim 15→16)
├── 5_eval_fair.py            # Fair evaluation with indicator=1.0 at inference
└── run_recap_pipeline.sh     # End-to-end pipeline (auto-resumable via .done_* files)
```

Key dependencies (reused from `scripts_pick_place_simulator/`):
- `6_test_alternating.py` → `AlternatingTester`, `load_diffusion_policy`, `load_policy_config`
- `4_train_diffusion.py` → `train_with_lerobot_api`, `convert_npz_to_lerobot_format`

---

## 4. Data Format

### Input NPZ (from `1_collect_rollouts.py`)

Each NPZ file contains a list of episodes under `episodes` key. Per episode dict:

| Key | Shape | Description |
|-----|-------|-------------|
| `images` | `[T, H, W, 3]` | Table camera RGB frames |
| `wrist_images` | `[T, H, W, 3]` | (Optional) wrist camera RGB frames |
| `ee_pose` | `[T, 7]` | End-effector pose (pos3 + quat4) |
| `obj_pose` | `[T, 7]` | Object pose (pos3 + quat4) |
| `action` | `[T, 8]` | Actions taken (pos3 + quat4 + gripper1) |
| `success` | `bool` | Whether this episode succeeded |
| `success_step` | `int/None` | Step at which success detected |
| `place_pose` | `[7]` | (Optional) target placement pose |
| `goal_pose` | `[7]` | (Optional) goal position pose |

> **Note:** Demo data has an explicit `gripper` key; DAgger/rollout data encodes gripper as `action[:, 7]`. All RECAP scripts handle both formats automatically.

### Augmented NPZ (after `3_compute_advantages.py`)

Same as above but with:

| Key | Shape | Description |
|-----|-------|-------------|
| `indicators` | `[T]` | int8, 1=positive advantage, 0=negative |
| `returns` | `[T]` | float32, normalised return at each timestep |
| `advantages` | `[T]` | float32, raw advantage value |

### State dimension

| Stage | `observation.state` dims | Contents |
|-------|--------------------------|---------|
| Baseline (15-dim) | 15 | ee_pose(7) + obj_pose(7) + gripper(1) |
| RECAP (16-dim)   | 16 | ee_pose(7) + obj_pose(7) + gripper(1) + **adv_indicator(1)** |

---

## 5. Step-by-Step Usage

### Prerequisites

```bash
# Activate environment
conda activate rev2fwd_il

# Verify Isaac Lab is accessible
python -c "import isaaclab; print('ok')"

# Set paths (adjust to your setup)
export POLICY_A="data/pick_place.../last/pretrained_model"
export POLICY_B="data/pick_place.../last/pretrained_model"
export DEMO_A="data/pick_place_isaac_lab_simulation/exp_new/task_A_reversed_100.npz"
export DEMO_B="data/pick_place_isaac_lab_simulation/exp_new/task_B_100.npz"
```

---

### Step 1: Collect Rollouts (Parallel Alternating A→B)

Runs both policies in alternating A→B cycles in the simulator, saving **all** episodes (success and failure). Designed for multi-GPU parallel collection.

```bash
# Single GPU: 10 A→B cycles = 10 Task A + 10 Task B episodes
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_recap_rl/1_collect_rollouts.py \
    --policy_A "$POLICY_A" \
    --policy_B "$POLICY_B" \
    --out_A data/recap/iter1/rollouts_gpu0_A.npz \
    --out_B data/recap/iter1/rollouts_gpu0_B.npz \
    --num_cycles 10 \
    --horizon 1200 \
    --seed 0 --headless

# 10-GPU parallel (100 episodes/task total):
for gpu in $(seq 0 9); do
    CUDA_VISIBLE_DEVICES=$gpu python scripts/scripts_recap_rl/1_collect_rollouts.py \
        --policy_A "$POLICY_A" --policy_B "$POLICY_B" \
        --out_A data/recap/iter1/rollouts_gpu${gpu}_A.npz \
        --out_B data/recap/iter1/rollouts_gpu${gpu}_B.npz \
        --num_cycles 10 --seed $gpu --headless &
done
wait

# Merge shards (pipeline does this automatically):
python3 -c "
import numpy as np
all_A = []
for g in range(10):
    all_A.extend(np.load(f'rollouts_gpu{g}_A.npz', allow_pickle=True)['episodes'])
np.savez_compressed('rollouts_A.npz', episodes=np.array(all_A, dtype=object))
"
```

> **Why alternating A→B?** This mirrors the cyclic evaluation pattern from `6_eval_cyclic.py`, ensuring realistic sim state transitions between tasks. Each GPU runs independent cycles with a unique seed for diverse coverage.

> **Why collect all episodes?** Unlike evaluation scripts that discard failures, RECAP *needs* failed episodes to learn what to avoid. The success rate of the rollouts doesn't need to be high — even 20% success provides a strong advantage signal.

---

### Step 2: Train Value Function

Trains a distributional MLP value head on top of a *frozen* copy of the policy's own visual encoder. No simulator needed.

```bash
# Task A value function
python scripts/scripts_recap_rl/2_train_value_function.py \
    --policy "$POLICY_A" \
    --npz_paths "$DEMO_A" data/recap/iter1/rollouts_A_200.npz \
    --out data/recap/iter1/vf_A.pt \
    --epochs 300 \
    --batch_size 512 \
    --lr 1e-3 \
    --num_bins 32 \
    --c_fail 1200 \
    --max_ep_len 1200

# Task B value function
python scripts/scripts_recap_rl/2_train_value_function.py \
    --policy "$POLICY_B" \
    --npz_paths "$DEMO_B" data/recap/iter1/rollouts_B_200.npz \
    --out data/recap/iter1/vf_B.pt \
    --epochs 300 --batch_size 512 --lr 1e-3
```

**Training efficiency:** The visual encoder processes all frames once (pre-computation), then only the tiny `ValueHead` MLP (≈64K parameters) is trained. Full training on 30k frames takes ~5 minutes on CPU.

---

### Step 3: Compute Advantages

Loads the value function, evaluates every frame in demo+rollout data, computes advantages, and writes binary indicators to a new NPZ.

```bash
# Task A
python scripts/scripts_recap_rl/3_compute_advantages.py \
    --policy "$POLICY_A" \
    --vf_ckpt data/recap/iter1/vf_A.pt \
    --npz_paths "$DEMO_A" data/recap/iter1/rollouts_A_200.npz \
    --out data/recap/iter1/advantages_A.npz \
    --stats_out data/recap/iter1/stats_A.json \
    --percentile 30

# Task B
python scripts/scripts_recap_rl/3_compute_advantages.py \
    --policy "$POLICY_B" \
    --vf_ckpt data/recap/iter1/vf_B.pt \
    --npz_paths "$DEMO_B" data/recap/iter1/rollouts_B_200.npz \
    --out data/recap/iter1/advantages_B.npz \
    --stats_out data/recap/iter1/stats_B.json \
    --percentile 30
```

Check `stats_A.json` to verify the indicator quality:

```json
{
  "threshold": -0.12,
  "positive_ratio": 0.70,
  "success_ep_positive_ratio": 0.92,
  "failure_ep_positive_ratio": 0.45
}
```

A good value function produces a *large gap* between `success_ep_positive_ratio` and `failure_ep_positive_ratio`.

---

### Step 4: RECAP Fine-tuning

Fine-tunes the policy with advantage conditioning. The state is extended from 15-dim to 16-dim by appending the indicator, and the UNet's global conditioning layer is zero-padded to accommodate the extra dimension.

```bash
# Single GPU
python scripts/scripts_recap_rl/4_retrain_with_recap.py \
    --npz_path data/recap/iter1/advantages_A.npz \
    --policy "$POLICY_A" \
    --out data/recap/iter1/recap_A \
    --steps 15000 \
    --batch_size 64 \
    --lr 5e-5 \
    --null_prob 0.2 \
    --alpha 1.0

# Multi-GPU (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/scripts_recap_rl/4_retrain_with_recap.py \
    --npz_path data/recap/iter1/advantages_B.npz \
    --policy "$POLICY_B" \
    --out data/recap/iter1/recap_B \
    --steps 15000 --batch_size 64 --lr 5e-5
```

**CFG null probability:** During training, `null_prob=0.2` of frames have their indicator replaced with 0.5 (the "null token"). This teaches the policy to generate actions unconditionally when no advantage information is provided. At inference, indicator=1.0 conditions on good behaviours only.

---

### Step 5: Fair Evaluation

Evaluates the RECAP policy with indicator=1.0 forced at every inference step.

```bash
python scripts/scripts_recap_rl/5_eval_fair.py \
    --policy data/recap/iter1/recap_A/checkpoints/checkpoints/last/pretrained_model \
    --task A \
    --num_episodes 100 \
    --out data/recap/iter1/eval_A.json \
    --headless

python scripts/scripts_recap_rl/5_eval_fair.py \
    --policy data/recap/iter1/recap_B/checkpoints/checkpoints/last/pretrained_model \
    --task B \
    --num_episodes 100 \
    --out data/recap/iter1/eval_B.json \
    --headless
```

**CFG guidance (optional):** Pass `--cfg_beta 1.5` to apply guidance strength β > 1:

$$a_{\text{guided}} = a_{\text{uncond}} + \beta \cdot (a_{\text{positive}} - a_{\text{uncond}})$$

Default β=1.0 uses the conditioned prediction directly (no guidance scaling).

---

### Iterative Refinement

Use the RECAP policy as the new `POLICY_A/B` for the next iteration:

```bash
EXP_NAME=recap_iter2 \
POLICY_A=data/recap/recap_iter1/recap_A/checkpoints/checkpoints/last/pretrained_model \
POLICY_B=data/recap/recap_iter1/recap_B/checkpoints/checkpoints/last/pretrained_model \
bash scripts/scripts_recap_rl/run_recap_pipeline.sh
```

---

## 6. Pipeline Script

The `run_recap_pipeline.sh` script runs all 5 steps automatically with **auto-resume**:

```bash
# Full run
EXP_NAME=recap_iter1 \
POLICY_A=<path_A> POLICY_B=<path_B> \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/scripts_recap_rl/run_recap_pipeline.sh
```

If the pipeline is interrupted, re-running picks up from where it stopped (`.done_*` marker files track completed steps). To force re-run a specific step:

```bash
rm data/recap/recap_iter1/.done_step2a_vf_A
bash scripts/scripts_recap_rl/run_recap_pipeline.sh
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXP_NAME` | `recap_iter1` | Output directory name under `data/recap/` |
| `POLICY_A` | — | **Required.** Path to Task A pretrained checkpoint |
| `POLICY_B` | — | **Required.** Path to Task B pretrained checkpoint |
| `DEMO_A` | `data/.../task_A_reversed_100.npz` | Demo NPZ for Task A |
| `DEMO_B` | `data/.../task_B_100.npz` | Demo NPZ for Task B |
| `CYCLES_PER_GPU` | `10` | A→B cycles per GPU (total episodes/task = GPUs × cycles) |
| `RECAP_STEPS` | `15000` | Fine-tuning steps |
| `NULL_PROB` | `0.2` | CFG null probability |
| `EVAL_EPISODES` | `100` | Episodes for final evaluation |
| `WANDB_ENABLED` | unset | Set to any value to enable W&B logging |
| `WANDB_PROJECT` | `rev2fwd-recap-<name>` | W&B project name |

---

## 7. Key Hyperparameters

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `C_FAIL` | 1200 | utils.py | Penalty for failed episode (= max horizon) |
| `MAX_EP_LEN` | 1200 | utils.py | Maximum episode length in steps |
| `percentile` | 30 | 3_compute_advantages.py | Advantage threshold: top 70% of frames get indicator=1 |
| `num_bins` | 32 | utils.py | Return distribution bins in ValueHead |
| `null_prob` | 0.2 | 4_retrain_with_recap.py | CFG dropout rate |
| `alpha` | 1.0 | 4_retrain_with_recap.py | Loss weight for RECAP frames vs. demo frames |
| `cfg_beta` | 1.0 | 5_eval_fair.py | CFG guidance strength (1.0 = off) |
| `ADV_POSITIVE` | 1.0 | 5_eval_fair.py | Indicator value injected at inference |
| `ADV_NULL` | 0.5 | 4_retrain_with_recap.py | Indicator value for CFG null token |
| `ADV_NEGATIVE` | 0.0 | utils.py | Indicator value for negative frames |
| VF `lr` | 1e-3 | 2_train_value_function.py | Adam LR for ValueHead training |
| VF `epochs` | 300 | 2_train_value_function.py | Training epochs |
| RECAP `lr` | 5e-5 | 4_retrain_with_recap.py | Fine-tuning LR (lower than baseline 1e-4) |
| RECAP `steps` | 15000 | 4_retrain_with_recap.py | Fine-tuning steps |

---

## 8. Architecture Details and Design Rationale

### Design Choice 1 — Why State dim 15 → 16?

The Diffusion Policy denoises actions conditioned on a **global conditioning vector** built from:

```
global_cond = concat(visual_features, robot_state)   # shape: [B, visual_dim + state_dim]
```

Currently `robot_state = [ee_pose(7) + obj_pose(7) + gripper_width(1)] = 15 dims`. This vector is fed into the UNet's `cond_encoder` (a small MLP) at every denoising step.

To inject the advantage indicator we need to add **one extra scalar** to this vector. The simplest compatible approach: append it as a 16th dimension. This touches exactly one weight matrix — the first linear layer of `cond_encoder`, whose input width grows from `n_obs_steps × (visual_dim + 15)` to `n_obs_steps × (visual_dim + 16)`. Every other part of the architecture — the UNet, the visual encoder, the action decoder — is completely unchanged.

The policy then learns: *"when the 16th input is 1.0, reproduce the high-advantage action distribution; when it is 0.0 or 0.5, produce a baseline/unconditional distribution."* At inference, we always feed 1.0, steering the policy toward its own best historical behaviour.

The pretrained 15-dim weights are preserved by **zero-padding** the new column in the expanded weight matrix. Zero-initialising the extra column means the indicator starts as a no-op at the beginning of fine-tuning — it contributes zero signal initially and the model gradually learns to use it, preserving the pretrained performance as a floor.

---

### Design Choice 2 — Why CFG Null Token = 0.5?

Classifier-Free Guidance (CFG) requires a **null / unconditional conditioning value**. During training, a random fraction (`null_prob=0.2`) of frames have their real indicator replaced by this null value. At inference, you can generate two predictions:
- $a_\text{cond}$: with indicator=1.0 (positive)
- $a_\text{uncond}$: with indicator=null

And amplify: $a = a_\text{uncond} + \beta \cdot (a_\text{cond} - a_\text{uncond})$ with $\beta \geq 1$.

The null token **must be distinguishable** from both the positive label (1.0) and the negative label (0.0). If null were 0.0, the model would see "unknown advantage" and "bad frame" as the same thing, conflating the two. If null were 1.0, it would conflate "unknown" with "good frame". Setting null to **0.5** (the midpoint of [0, 1]) avoids both collisions and has no accidental semantic meaning — it lies strictly between the two real labels, giving the model a clean three-way distinction:

| Value | Meaning | When used |
|-------|---------|----------|
| `1.0` | Positive advantage | Real indicator=1 (not null-dropped) |
| `0.0` | Negative advantage | Real indicator=0 (not null-dropped) |
| `0.5` | Unknown / unconditional | null_prob dropout OR inference unconditional branch |

---

### Design Choice 3 — Why Feature Pre-computation?

The value function is `V_φ = frozen_encoder + trainable_ValueHead`. The **frozen encoder** (a ResNet deepcopy of the pretrained policy's visual backbone) does not change during ValueHead training.

Without pre-computation, each training gradient step would encode all mini-batch frames through the full ResNet (expensive), then compute the ValueHead output and loss (cheap).

With pre-computation:
1. Run the ResNet **once** on all frames, cache feature vectors to RAM (done before training begins)
2. At each training step, load cached features directly — the ResNet is never touched again

With 300 epochs × 30k frames, the ResNet would otherwise run 9 million forward passes. Pre-computing reduces this to exactly **1 pass per frame**. The ValueHead MLP is ~64K parameters and runs in microseconds per batch. Total training time goes from hours to ~5 minutes on CPU.

---

### Design Choice 4 — Why `.done_stepN` Auto-resume Markers?

The pipeline spans ~10 steps, takes several hours, and involves an Isaac Lab simulator (which is prone to crashes on timeout or OOM). Without checkpointing, a crash at step 9 would require re-running everything from scratch.

Marker files are the simplest reliable checkpointing mechanism for shell scripts:
- Before running step N: `if [ -f .done_stepN ]; then skip; fi`
- After step N completes successfully: `touch .done_stepN`

This is more robust than checking output file timestamps (files can be partially written) or encoding state in environment variables (not persistent across restarts). To force a step to re-run, simply `rm .done_stepN`.

---

### Value Function Architecture

```
Observation
├── image [B, 1, C, H, W]
│   └── policy.rgb_encoder (frozen deepcopy of pretrained ResNet)
│       └── [B, visual_dim]  (≈1024 for ResNet-50)
└── state [B, 15]  (ee_pose + obj_pose + gripper)
    └── normalised via policy.normalize_inputs

Concatenated features [B, visual_dim + 15]
    └── ValueHead: Linear → ReLU → Linear → ReLU → Linear
        └── logits [B, 32]  →  softmax → bin probabilities
            └── MSE/CE loss on target bin(normalised_return)
```

Return normalisation maps $R \in [-1, 0]$ to bin index via `torch.bucketize` on 32 uniformly-spaced thresholds.

**Inference:** Expected return = $\sum_k p_k \cdot c_k$ where $c_k$ is the bin centre.

### Checkpoint Migration (15-dim → 16-dim)

The UNet global conditioning input changes from `n_obs_steps * (visual_dim + 15)` to `n_obs_steps * (visual_dim + 16)`. The `cond_encoder` first layer weight tensor grows by `n_obs_steps` columns.

Migration strategy (in `utils.migrate_checkpoint_for_recap`):
1. Instantiate a fresh `DiffusionPolicy` with `state_dim=16`.
2. Compare every parameter in `new_policy.state_dict()` vs old checkpoint.
3. For 2D weight tensors where `new.shape[1] > old.shape[1]`: copy old weights to the first `old.shape[1]` columns, zero-pad the remaining columns.
4. Load with `strict=False` to ignore shape mismatches (all other params are loaded normally).

This is robust: no hardcoded layer names, works regardless of LeRobot version changes.

### Diffusion Policy State Encoding

At inference time, `5_eval_fair.py` subclasses `AlternatingTester` and overrides `_build_policy_inputs()`:

```python
# Original (15-dim state)
observation.state = [ee_pose(7), obj_pose(7), gripper(1)]

# RECAP (16-dim state)
observation.state = [ee_pose(7), obj_pose(7), gripper(1), 1.0]
#                                                          ^^^
#                                               ADV_POSITIVE indicator
```

The policy learned to associate indicator=1.0 with high-quality actions during fine-tuning.

---

## 9. Relationship to Baseline

| Component | Baseline (`scripts_pick_place_simulator/`) | RECAP |
|-----------|---------------------------------------------|-------|
| Data collection | Successes only (`6_eval_cyclic.py`) | All episodes, alternating A→B (`1_collect_rollouts.py`) |
| Policy training | From scratch / iterated fine-tuning | Fine-tunes baseline weights |
| State dim | 15 | 16 (+ advantage indicator) |
| Checkpoints | Direct LeRobot format | Migrated via `migrate_checkpoint_for_recap` |
| Evaluation | `7_eval_fair.py` | `5_eval_fair.py` (same structure, forces indicator=1) |
| Shell pipeline | `run_pipeline.sh` | `run_recap_pipeline.sh` |

The RECAP scripts reuse these functions from the baseline:
- `AlternatingTester` → role rotation between tasks
- `load_diffusion_policy` → loads LeRobot DiffusionPolicy from checkpoint dir
- `load_policy_config` → reads `pretrained_cfg.json` from checkpoint
- `train_with_lerobot_api` → runs the LeRobot training loop
- `convert_npz_to_lerobot_format` → base helper for dataset construction

---

## 10. Using exp17 (DAgger) Checkpoints and Data

exp17 is the ongoing DAgger baseline experiment. Its data and checkpoints are **100% compatible** with the RECAP scripts — the NPZ field names (`images`, `wrist_images`, `ee_pose`, `obj_pose`, `gripper`, `action`, `success`) are identical to what `utils.py` and `4_retrain_with_recap.py` expect.

### Available exp17 artifacts

```
data/pick_place_isaac_lab_simulation/exp17/
├── task_A_reversed_100.npz          # 100 demo episodes for Task A (use as DEMO_A)
├── task_B_100.npz                   # 100 demo episodes for Task B (use as DEMO_B)
├── iter1_collect_A.npz              # DAgger iter 1 successful Task A rollouts
├── iter2_collect_A.npz              # DAgger iter 2 successful Task A rollouts
├── iter3_collect_A.npz              # DAgger iter 3 successful Task A rollouts
├── iter{1,2,3}_collect_B.npz        # Same for Task B
├── iter3_ckpt_A/                    # Latest Task A checkpoint (pretrained_model format)
└── iter3_ckpt_B/                    # Latest Task B checkpoint
```

iter3 performance (prior to RECAP): Task A 38%, Task B 100%.

### Starting RECAP from exp17

```bash
EXP_NAME=recap_exp17_iter1 \
POLICY_A=data/pick_place_isaac_lab_simulation/exp17/iter3_ckpt_A \
POLICY_B=data/pick_place_isaac_lab_simulation/exp17/iter3_ckpt_B \
DEMO_A=data/pick_place_isaac_lab_simulation/exp17/task_A_reversed_100.npz \
DEMO_B=data/pick_place_isaac_lab_simulation/exp17/task_B_100.npz \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
bash scripts/scripts_recap_rl/run_recap_pipeline.sh
```

### Reusing DAgger rollout data for richer VF training

The DAgger `iter*_collect_A/B.npz` files contain **successful episodes only** (from `6_eval_cyclic.py`). They can supplement the demo data for value function training, providing more diverse coverage of good frames. Pass them alongside DEMO to `2_train_value_function.py` and `3_compute_advantages.py`:

```bash
python scripts/scripts_recap_rl/2_train_value_function.py \
    --policy data/pick_place_isaac_lab_simulation/exp17/iter3_ckpt_A \
    --npz_paths \
        data/pick_place_isaac_lab_simulation/exp17/task_A_reversed_100.npz \
        data/pick_place_isaac_lab_simulation/exp17/iter1_collect_A.npz \
        data/pick_place_isaac_lab_simulation/exp17/iter2_collect_A.npz \
        data/pick_place_isaac_lab_simulation/exp17/iter3_collect_A.npz \
        data/recap/recap_exp17_iter1/rollouts_A_200.npz \
    --out data/recap/recap_exp17_iter1/vf_A.pt \
    --epochs 300
```

> **Note:** DAgger rollouts are successes-only, so all their frames receive `success=True` returns during advantage computation. This is correct — they represent good behaviour and act as extra positive examples for the value function.

### Why you still need to run step 1 (collect new rollouts)

The key ingredient RECAP needs that DAgger data does **not** provide: **failed episodes**. The value function learns to distinguish good frames (low time-to-success) from bad frames (episode ends in failure) by contrasting their normalised returns. Without failures in the training set, the VF cannot learn this contrast and will produce a poor advantage signal. Run `1_collect_rollouts.py` with the iter3 policy to collect ~200 mixed-success episodes before proceeding.

### Data format compatibility

Both demo/DAgger data and new rollouts use the same field names:

| Field | exp17 demo | exp17 DAgger | RECAP rollout |
|-------|------------|-------------|---------------|
| `images` | ✅ | ✅ | ✅ (from `AlternatingTester`) |
| `wrist_images` | ✅ | ✅ | ✅ |
| `ee_pose`, `obj_pose` | ✅ | ✅ | ✅ |
| `gripper` (explicit key) | ✅ | ❌ (use `action[:,7]`) | ❌ (use `action[:,7]`) |
| `action` (8-dim) | ✅ | ✅ | ✅ |
| `success` flag | ✅ (all True) | ✅ (all True) | ✅ (mixed) |

All RECAP scripts handle both `gripper` formats automatically (explicit key or `action[:, 7]` fallback).

---

## 11. Troubleshooting

### "state_dict mismatch on load"

The RECAP fine-tuned policy has `state_dim=16`. When loading it with `load_diffusion_policy`, you must pass `override_state_dim=16` (handled automatically by `load_recap_policy()` in `5_eval_fair.py`). Do not try to load RECAP checkpoints with standard `load_diffusion_policy`.

### "Value function produces poor advantage signal"

Check `stats_A.json`:
- If `success_ep_positive_ratio ≈ failure_ep_positive_ratio`: the VF is not discriminating → train longer or increase `--epochs`
- If `positive_ratio < 0.3`: threshold too strict → lower `--percentile`
- If `positive_ratio > 0.9`: threshold too loose → raise `--percentile`

Good target: `success_ep_positive_ratio > 0.85`, `failure_ep_positive_ratio < 0.60`.

### "RECAP policy ignores conditioning"

Check that fine-tuning used consistent `null_prob`. If `null_prob` was too high (e.g., 0.5), the policy never sees enough conditioned examples to learn the conditioning. Default 0.2 is recommended.

After fine-tuning, verify conditioning works:
```python
# Evaluate both conditioning values and compare action distributions
# indicator=0.0 → policy should be more erratic / lower quality
# indicator=1.0 → policy should be more precise / goal-directed
```

### "CUDA OOM during feature pre-computation"

Reduce `--batch_size` in `2_train_value_function.py` (pre-computation batch, default 64). The `precompute_features()` function in `utils.py` uses `batch_size` frames per encoder forward pass. Lowering to 16 or 8 reduces peak VRAM.

### "Rollout collection is slow"

Use `--headless` flag to disable rendering. Also verify `--horizon` matches your policy's expected episode length. Setting it shorter than necessary may cause the policy to always fail.

### "Policy A and B reward signals are imbalanced"

RECAP trains value functions independently per task. If one task has very different success rates, the advantage percentile threshold will also differ. This is expected and correct — the threshold adapts to each task's difficulty.

---

## Quick Reference

```bash
# Starting from exp17 (the running DAgger experiment)
EXP_NAME=recap_exp17_iter1 \
POLICY_A=data/pick_place_isaac_lab_simulation/exp17/iter3_ckpt_A \
POLICY_B=data/pick_place_isaac_lab_simulation/exp17/iter3_ckpt_B \
DEMO_A=data/pick_place_isaac_lab_simulation/exp17/task_A_reversed_100.npz \
DEMO_B=data/pick_place_isaac_lab_simulation/exp17/task_B_100.npz \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
bash scripts/scripts_recap_rl/run_recap_pipeline.sh

# Check advantage quality (gap between success and failure = healthy VF)
cat data/recap/recap_exp17_iter1/advantage_stats_A.json

# Check eval results
cat data/recap/recap_exp17_iter1/eval_recap_A.stats.json
cat data/recap/recap_exp17_iter1/eval_recap_B.stats.json

# Force re-run a single step (e.g., redo VF training)
rm data/recap/recap_exp17_iter1/.done_step2a_vf_A
bash scripts/scripts_recap_rl/run_recap_pipeline.sh
```

---

## 12. Experiment Progress — exp19 (2026-03-24)

### Overview

| Item | Value |
|------|-------|
| **Experiment** | exp19 — RECAP RL iteration 1 |
| **Base policy A** | exp17 iter2 checkpoint (Task A 58%) |
| **Base policy B** | exp17 iter3 checkpoint (Task B 100%) |
| **GPUs** | 10 × RTX 3090 (dongxu-g5) |
| **Directory** | `data/pick_place_isaac_lab_simulation/exp19/` |
| **Pipeline** | `data/pick_place_isaac_lab_simulation/exp19/run_exp19.sh` |

### Step Status

| Step | Status | Marker file | Duration | Notes |
|------|--------|-------------|----------|-------|
| 1. Collect rollouts | ✅ Done | `.done_step1_collect` (06:18) | ~4h | 10 GPUs × 10 A→B cycles |
| 2a. Train VF A | ✅ Done | `.done_step2_vf_A` (06:38) | ~20min | vf_A.pt (295K) |
| 2b. Train VF B | ✅ Done | `.done_step2_vf_B` (06:52) | ~14min | vf_B.pt (295K) |
| 3a. Compute advantages A | ✅ Done | `.done_step3_adv_A` (07:10) | ~18min | advantages_A.npz (5.4G) |
| 3b. Compute advantages B | ✅ Done | `.done_step3_adv_B` (07:30) | ~20min | advantages_B.npz (5.5G) |
| 4a. RECAP fine-tune A | ❌ **FAILED** | No marker | — | Failed 4 times, see below |
| 4b. RECAP fine-tune B | ⬜ Not started | — | — | Blocked by 4a |
| 5a. Evaluate A | ⬜ Not started | — | — | |
| 5b. Evaluate B | ⬜ Not started | — | — | |

### Collection Results (Step 1)

| Task | Episodes | Success rate | File size |
|------|----------|-------------|-----------|
| A | 100 | **34/100 = 34%** | rollouts_A.npz (2.8G) |
| B | 100 | **96/100 = 96%** | rollouts_B.npz (2.2G) |

### Advantage Statistics (Step 3)

| Metric | Task A | Task B |
|--------|--------|--------|
| Total episodes (demo + DAgger + rollouts) | 278 | 325 |
| Total frames | 212K | 218K |
| Advantage threshold (30th percentile) | -0.0068 | -0.0031 |
| Positive ratio | 70% | 70% |
| **Success ep positive ratio** | **81.4%** | **71.1%** |
| **Failure ep positive ratio** | **49.9%** | **31.8%** |
| Gap (success - failure) | 31.5pp | 39.3pp |

Both tasks have a healthy gap between success and failure positive ratios, indicating the value function learned meaningful discrimination.

### Step 4 Failure History

Step 4 was attempted 4 times, each failing for a **different** bug (all fixed in code):

| Attempt | Time | Root cause | Fix applied |
|---------|------|-----------|-------------|
| 1 | 07:30 | NCCL timeout — rank 0 converting data inside `init_process_group` while other ranks waited | Moved NPZ→LeRobot conversion to BEFORE `init_process_group`; non-main ranks poll for `.conversion_meta.json` |
| 2 | 07:55 | `AttributeError: 'LeRobotDataset' has no attribute 'consolidate'` | Changed to `dataset.finalize()` (LeRobot 0.4.3 API) |
| 3 | 08:37 | `ProcessorMigrationError` — `_recap_pretrained/` missing processor files | Added `_copy_and_adapt_processor_files()` to copy & expand processor stats from 15→16 dims |
| 4 | 12:40 | **SIGHUP** — user disconnected; `nohup` doesn't protect `torchrun` child processes | Need `setsid` instead of `nohup` |

### Current State of Partial Artifacts

- `recap_A/lerobot_dataset/` — **partial**: 53/278 episodes converted (from attempt 4). **Must be deleted** before re-running.
- No `.conversion_meta.json` or `_recap_pretrained/` exists (cleaned up on crash).

### Bugs Fixed During This Session (2026-03-24)

Code changes made across these files:

1. **`1_collect_rollouts.py`** — Added `sys.exit(1)` after traceback to prevent silent failures
2. **`4_retrain_with_recap.py`** — Three fixes:
   - `consolidate()` → `finalize()`
   - Dataset conversion moved before `init_process_group`; non-main ranks poll for `.conversion_meta.json`
   - Gripper field handling (`_gripper_fn` lambda)
3. **`utils.py`** — Fixed encoder access path: `policy.rgb_encoder` → `policy.diffusion.rgb_encoder` (affects `infer_vf_feat_dim`, `build_vf_model`, `FrozenEncoderValueFunction`, `precompute_features`)
4. **`../scripts_pick_place/4_train_diffusion.py`** — Added `_copy_and_adapt_processor_files()`, `_pad_state_stats()`, `_update_state_shape_in_config()` for RECAP checkpoint migration with processor files
5. **`run_exp19.sh`** — Added `PYTHONUNBUFFERED=1`; merge script validates non-empty data

### How to Resume

```bash
cd /mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il

# 1. Clean partial artifacts from failed step 4 attempts
rm -rf data/pick_place_isaac_lab_simulation/exp19/recap_A/

# 2. Launch with setsid (survives SSH disconnection, unlike nohup+torchrun)
setsid bash data/pick_place_isaac_lab_simulation/exp19/run_exp19.sh \
    > data/pick_place_isaac_lab_simulation/exp19/logs/recap_pipeline.log 2>&1 &

# 3. Monitor
tail -f data/pick_place_isaac_lab_simulation/exp19/logs/recap_pipeline.log
# or
bash data/pick_place_isaac_lab_simulation/exp19/monitor.sh
```

Steps 1–3 will be skipped automatically (`.done_*` markers exist). Pipeline resumes at Step 4a.

**Expected remaining work:**
- Step 4a: RECAP fine-tune Policy A (15000 steps, 10 GPUs DDP)
- Step 4b: RECAP fine-tune Policy B (15000 steps, 10 GPUs DDP)
- Step 5a: Evaluate RECAP Policy A (100 episodes)
- Step 5b: Evaluate RECAP Policy B (100 episodes)

### Known Issue: nohup + torchrun

`nohup` does NOT protect `torchrun` child processes from SIGHUP. When the user disconnects, `torchrun`'s elastic agent receives SIGHUP and kills all workers. **Use `setsid` instead** to create a new session that is immune to terminal hangup signals:

```bash
# ❌ WRONG — children still get SIGHUP
nohup bash run_exp19.sh > log 2>&1 &

# ✅ CORRECT — new session, immune to SIGHUP
setsid bash run_exp19.sh > log 2>&1 &
```
