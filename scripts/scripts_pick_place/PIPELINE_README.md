# Rev2Fwd Imitation Learning Pipeline

This project implements a **Reverse-to-Forward (Rev2Fwd)** imitation learning approach for robotic Pick-and-Place tasks.

---

## 1. Method Overview

### Core Idea

Pick-and-Place can be decomposed into two inverse sub-tasks:

| Task | Description | Difficulty |
|------|-------------|------------|
| **Task A** | Pick from **arbitrary position** → Place at **fixed target** | **Hard** (large initial state space) |
| **Task B** | Pick from **fixed target** → Place at **arbitrary position** | **Easy** (small initial state space) |

**Key Insight**: Pick-and-Place trajectories are **temporally reversible**. Therefore:
1. Implement a simple state-machine expert for the easier Task B
2. **Time-reverse** Task B trajectories to automatically obtain Task A training data

This way, **one simple expert** generates training data for **both tasks**.

### Iterative Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     Iteration N                              │
├─────────────────────────────────────────────────────────────┤
│  1. Rollout: Run A→B→A→B... cycles, collect successful traj │
│  2. Aggregate: Original data + newly collected rollout data  │
│  3. Finetune: Continue training from previous checkpoint     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        Iteration N+1
```

---

## 2. Data Flow Design

### Directory Structure

Iterative training uses three directories to manage data and checkpoints:

| Directory | Role | Description |
|-----------|------|-------------|
| `origin` | Original data (read-only) | Initial checkpoint and LeRobot dataset |
| `temp` | Previous iteration | Base for accumulating new data |
| `last` | Current training | Output location for this iteration |

### Per-Iteration Data Flow

```
                    Iteration N
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌────────┐         ┌──────────┐         ┌──────────┐
│ origin │         │   temp   │         │   last   │
│(r/o)   │         │(prev)    │         │(current) │
└────────┘         └──────────┘         └──────────┘
    │                    │                    │
    │                    │                    │
    │     ┌──────────────┘                    │
    │     │                                   │
    │     ▼                                   │
    │  ┌─────────────────┐                    │
    │  │ temp/lerobot +  │ ──aggregate──▶  last/lerobot
    │  │ rollout_data    │                    │
    │  └─────────────────┘                    │
    │                                         │
    │     ┌───────────────────────────────────┘
    │     │
    │     ▼
    │  ┌─────────────────┐
    │  │ temp/checkpoint │ ──resume──▶  last/checkpoint
    │  └─────────────────┘
    │
    └──────────────────────────────────────────────────────┐
                                                           │
                      After iteration                      │
                         │                                 │
                         ▼                                 │
                ┌─────────────────┐                        │
                │ delete temp     │                        │
                │ last → temp     │ ◀── prepare for next ──┘
                └─────────────────┘
```

### Data Aggregation Strategy

Training data **accumulates** across iterations:

```
Iter 1: original_data + rollout_iter1
Iter 2: original_data + rollout_iter1 + rollout_iter2
Iter 3: original_data + rollout_iter1 + rollout_iter2 + rollout_iter3
...
```

This **DAgger-style** aggregation allows the model to learn from its own mistakes and progressively improve.

---

## 3. Baseline Experiment Design

To validate the effectiveness of data aggregation, we design a **baseline experiment**.

### Baseline vs Full Method

| Aspect | Full Method (`run_iterative_training.sh`) | Baseline (`run_iterative_training_baseline.sh`) |
|--------|-------------------------------------------|------------------------------------------------|
| Rollout testing | ✅ Yes | ✅ Yes |
| Data collection | ✅ Collect | ✅ Collect (but not used) |
| Data aggregation | ✅ Original + rollout | ❌ Original only |
| Training data | Grows each iteration | Fixed |

### Purpose of Baseline

The baseline experiment aims to demonstrate:

1. **Necessity of data aggregation**: If baseline performance stagnates, it proves original data alone cannot sustain improvement
2. **Value of rollout data**: Comparing performance curves quantifies the contribution of rollout data

### Baseline Data Flow

```
                    Iteration N (Baseline)
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌────────┐         ┌──────────┐         ┌──────────┐
│ origin │         │   temp   │         │   last   │
│(r/o)   │         │(prev)    │         │(current) │
└────────┘         └──────────┘         └──────────┘
    │                    │                    │
    │                    │                    │
    │                    │                    │
    ▼                    │                    │
 ┌─────────┐             │                    │
 │ origin/ │ ──copy directly─────────────▶  last/lerobot
 │ lerobot │   (no rollout aggregation)      │
 └─────────┘                                  │
                                              │
              ┌───────────────────────────────┘
              │
              ▼
           ┌─────────────────┐
           │ temp/checkpoint │ ──resume──▶  last/checkpoint
           └─────────────────┘
```

**Key Difference**: In baseline, each iteration trains on a **copy of the original dataset** without any rollout data.

---

## 4. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `consecutive_successes` | Number of successful A→B cycles before first failure |
| `task_A_success_rate` | Task A success rate |
| `task_B_success_rate` | Task B success rate |
| `avg_success_step` | Average steps for successful episodes (lower is better) |

---

## 5. Quick Start

```bash
# Activate environment
conda activate rev2fwd_il

# Run full iterative training (with data aggregation)
bash scripts/scripts_pick_place/run_iterative_training.sh

# Run baseline experiment (without data aggregation)
bash scripts/scripts_pick_place/run_iterative_training_baseline.sh

# Analyze and compare results
python scripts/scripts_pick_place/8_analyze_rollout_record.py
```

---

## 6. References

- [LeRobot](https://github.com/huggingface/lerobot): Diffusion Policy implementation
- [DAgger](https://arxiv.org/abs/1011.0686): Dataset Aggregation algorithm
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/): Simulation environment
