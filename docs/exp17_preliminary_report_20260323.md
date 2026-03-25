# Exp17 — Rev2Fwd Pick-and-Place: Preliminary Progress Report

> **Status**: In progress — 2 of 10 planned iterations completed (as of 2026-03-23)
> **Experiment dir**: `data/pick_place_isaac_lab_simulation/exp17/`
> **Visualization**: `data/pick_place_isaac_lab_simulation/exp17/progress_plot.png`

---

## 1. Experiment Setup

### Goal
Validate the Rev2Fwd (reverse-to-forward) imitation learning pipeline on the
Pick-and-Place task using iterative DAgger-style data collection, with 6 GPUs
and larger batch/LR compared to earlier experiments.

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Training GPUs | 2, 3, 4, 5, 6, 7 (6 GPUs) |
| Batch size | 256 (↑ from 128 in exp16) |
| Learning rate | 2e-4 (scaled up for 6-GPU effective batch) |
| Planned iterations | 10 |
| Steps per iter (finetune) | 5000 |
| Eval horizon | 1500 steps/episode |
| Distance threshold (Task A) | 0.03 m |
| Policy (train / eval action steps) | 16 / 8 |

### Task Description

| Task | Start state | Goal | Difficulty |
|------|-------------|------|------------|
| **Task A** (hard) | Cube in red rectangle (random XY) | Place near green goal `(0.5, -0.2)` | Random initial position |
| **Task B** (easy) | Cube at green goal | Place inside red rectangle `center=(0.5, 0.2)` | Fixed start, random target |

**Red rectangle**: `center=(0.5, 0.2)`, `size=0.3×0.3 m`

Training data source:
- **Task B**: FSM expert forward collection (100 episodes)
- **Task A**: Time-reversed Task B trajectories (automatic, no expert needed)

---

## 2. Results Summary (2 iterations)

### 2.1 Fair Test — 50 Independent Episodes per Iteration

| Iteration | Task A | Task B |
|-----------|--------|--------|
| **1** | **50%** (25/50) | **66%** (33/50) |
| **2** | **58%** (29/50) | **100%** (50/50) ✓ |

**Task B saturated to 100% in just 2 iterations.**  
Task A improved from 50% → 58% (+8 pp) and is still climbing.

### 2.2 Cyclic A→B Collection — 50 Cycles per Iteration

| Iteration | Task A | Task B |
|-----------|--------|--------|
| **1** | 62% (31/50) | 60% (30/50) |
| **2** | 66% (33/50) | 94% (47/50) |

Cyclic rates are measured on the policy used for *data collection* (before
the fine-tuned checkpoint is evaluated in the fair test), so they slightly
lead the fair test by reflecting on-policy behavior.

### 2.3 Checkpoint Info

| Iteration | Policy A steps | Policy B steps |
|-----------|---------------|---------------|
| 1 | 13,389 | 13,421 |
| 2 | 18,389 | 18,421 |

Initial training (phase 2): ~13k steps from scratch.  
Each finetune round adds 5,000 steps on the merged dataset.

---

## 3. Analysis

### 3.1 Task B (Goal → Red Zone): Very Fast Convergence
Task B reached **100% in fair testing after only 2 DAgger iterations**.  
The cyclic rate also jumped from 60% → 94% between iterations 1 and 2.

This confirms that:
- The forward-collected FSM data + iterative on-policy fine-tuning is a highly
  efficient bootstrapping approach for Task B.
- The Red Rectangle is a forgiving target (30×30 cm), helping success rates.

### 3.2 Task A (Red Zone → Goal): Steady Improvement
Task A is harder because the cube starts at a **random position** inside the
red rectangle, requiring robust grasping from diverse initial states.

- Iteration 1 fair test: 50% (chance is near 0 without learning)
- Iteration 2 fair test: 58% (+8 pp)

The upward trend is clear; further iterations should push this higher.

**Cyclic collection quality**: The per-cycle rolling average (see visualization,
bottom row) shows both tasks gradually improving over the 50 cycles within each
iteration, suggesting the policy is learning from its own newly collected data.

### 3.3 Time per Iteration
| Iteration | Elapsed (collection) |
|-----------|----------------------|
| 1 | ~4.2 hours (15,066 s) |
| 2 | ~3.1 hours (11,115 s) |

Iteration 2 was faster, likely because the policies are more competent (fewer
wasted steps per episode).

### 3.4 Comparison with Prior Experiments

| Exp | Method | Task A (final) | Task B (final) |
|-----|--------|----------------|----------------|
| exp13 | Baseline — both tasks forward (FSM) | 74% | 78% |
| exp14 | Rev2Fwd (reversed Task A) | ~17% (early) | ~88% (early) |
| **exp17** | Rev2Fwd + iterative DAgger, iter 2 | **58%** | **100%** |

Exp17 Task B already surpasses exp13 forward baseline. Task A is catching up
with more iterations remaining.

---

## 4. Visualization

The plot (`progress_plot.png`) contains four panels:

| Panel | Content |
|-------|---------|
| Top-left | Fair test success rate (Task A & B) over iterations |
| Top-right | Cyclic collection success rate (Task A & B) over iterations |
| Bottom-left | Iter 1 — per-cycle rolling average (window=10) |
| Bottom-right | Iter 2 — per-cycle rolling average (window=10) |

To regenerate the plot after more iterations complete:

```bash
python scripts/scripts_pick_place_simulator/plot_exp_progress.py \
    --exp_dir data/pick_place_isaac_lab_simulation/exp17 \
    --out data/pick_place_isaac_lab_simulation/exp17/progress_plot.png
```

---

## 5. Next Steps

The experiment is currently running on the remote server (iterations 3–10 in
progress). Expected outcomes:

- **Task B**: Likely to stay near 100% as it has already saturated.
- **Task A**: Expected to continue improving with more on-policy data; target
  is to match or exceed the exp13 forward baseline of 74%.
- Failure analysis videos (`failure_videos_A_gpu1/`, `failure_videos_B_gpu2/`)
  are available for qualitative inspection of remaining failure modes.

---

*Generated: 2026-03-23 | Script: `scripts/scripts_pick_place_simulator/plot_exp_progress.py`*
