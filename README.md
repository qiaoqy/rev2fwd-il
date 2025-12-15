# rev2fwd-il: Reverse-to-Forward Imitation Learning (Isaac Lab)

This repo is a minimal simulator validation for the idea:

> To learn a forward pick-and-place policy A (table random -> goal),
> we first run an easier reverse policy B (goal -> table random),
> collect ~100 reverse rollouts, reverse them in time, and train A by behavior cloning.

We intentionally use **IK-Abs (end-effector pose setpoints)** so that "time reversal" can be implemented as:
- reverse the observation/state sequence
- reconstruct action labels as **next-step end-effector pose + next-step gripper state**

This is the easiest way to get a working sanity-check experiment.

---

## 1. Requirements

### Hardware / OS
- Linux recommended
- NVIDIA GPU recommended (Isaac Sim / Isaac Lab)

### Simulator
We target **Isaac Lab** with the built-in Franka lift task:
- `Isaac-Lift-Cube-Franka-IK-Abs-v0`[https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#:~:text=Isaac%2DLift%2DCube%2DFranka%2DIK%2DAbs%2Dv0%3A%20Franka%20arm%20with%20absolute%20IK%20control]

Isaac Lab provides state-machine examples that can pick a cube and move it to a desired pose.
We reuse that logic as the "prebuilt policy B".

> If you installed Isaac Lab from pip packages:
> note that Isaac Lab pip packages do NOT ship the standalone example scripts,
> so this repo provides its own runner scripts.

---

## 2. Installation

### Option A (recommended): Layer this repo on top of an existing Isaac Lab env

1) Create an env and install Isaac Lab (pick versions that match your CUDA/toolchain):
```bash
conda create -n rev2fwd_il python=3.11
conda activate rev2fwd_il
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
pip install wandb
pip install -e ./isaaclab_tasks
```

2) Install this repo (editable) and any optional extras:
```bash
pip install -e .
# optional, recommended utilities used by later scripts
pip install -e ".[extras]"
```

### Option B: Use the provided lightweight conda env (no Isaac Lab inside)
This just creates a minimal Python 3.11 environment with common utilities.
You still need to install Isaac Lab separately.

```bash
conda env create -f environment.yml
conda activate rev2fwd_il
pip install -e .
pip install -e ".[extras]"
```

---

## 3. Quickstart

### 3.1 Sanity check: can we run the environment?

This script launches Isaac Sim via Isaac Lab's `AppLauncher`, creates a gymnasium environment,
prints action/observation spaces + key scene entities, and then steps the simulation.

```bash
# Minimal headless smoke test (recommended)
python scripts/00_sanity_check_env.py --headless --steps 50
```

More examples:

```bash
# GUI mode (creates a window)
python scripts/00_sanity_check_env.py --steps 200

# Random actions (useful to verify actuation is alive)
python scripts/00_sanity_check_env.py --headless --steps 50 --random_action 1

# Deterministic micro-perturbation on action dim-0 (useful for IK tasks)
python scripts/00_sanity_check_env.py --headless --steps 50 --nudge_action 0.01
```

#### CLI arguments (script-specific)

| Argument | Description |
|----------|-------------|
| `--task` | Gym task id to create via `gym.make(...)`. |
| `--num_envs` | Number of parallel environments (vectorized) inside the same simulation. |
| `--rl_device` | Alias for the AppLauncher `--device` argument (e.g. `cuda`, `cuda:0`, `cpu`). |
| `--disable_fabric` | If set to `1`, disables Fabric in the parsed env cfg. |
| `--steps` | How many `env.step(...)` calls to run. |
| `--seed` | RNG seed for python/numpy/torch. |
| `--random_action` | If `1`, sample actions from `env.action_space` each step. |
| `--nudge_action` | If non-zero, use a stable zero action and add a small deterministic value to action dimension 0. |
| `--print_every` | Print debug diagnostics every N steps (default: 1). |
| `--step_timeout_s` | If a single `env.step()` is slower than this threshold, print a warning. |

#### AppLauncher arguments (Isaac Lab / Isaac Sim)

The script also supports standard launcher flags injected by
`isaaclab.app.AppLauncher.add_app_launcher_args(parser)`.
Common ones include:

| Argument | Description |
|----------|-------------|
| `--headless` | Run Isaac Sim without opening a window. Use `--headless` or `--headless 1` for headless mode, `--headless 0` for GUI. |
| `--device` | Simulation device for Isaac Lab tensors (e.g. `cuda:0`). |

Run `python scripts/00_sanity_check_env.py --help` to see the full list provided by your Isaac Lab version.

---

### 3.2 Debug Expert B (Reverse Task)

This script tests the Expert B finite state machine, which performs the **reverse** pick-and-place task:
- Cube starts at the GOAL position (plate center)
- Expert B picks it up and places it at a random table position
- Robot returns to REST pose

```bash
# Run with GUI visualization (recommended for first-time debugging)
python scripts/01_debug_expert_B_one_episode.py

# Run in headless mode (faster, no rendering)
python scripts/01_debug_expert_B_one_episode.py --headless

# Custom parameters
python scripts/01_debug_expert_B_one_episode.py --headless --seed 123 --steps 600 --goal_xy 0.5 0.1
```

#### Expert B State Machine

The expert follows this finite state machine sequence:

```
REST → GO_ABOVE_OBJ → GO_TO_OBJ → CLOSE → LIFT → GO_ABOVE_PLACE → GO_TO_PLACE → OPEN → RETURN_REST → DONE
```

| State | Description | Gripper |
|-------|-------------|---------|
| `REST` | Initial resting position | Open |
| `GO_ABOVE_OBJ` | Move EE above the object | Open |
| `GO_TO_OBJ` | Lower EE to grasp position | Open |
| `CLOSE` | Close gripper to grasp object | Close |
| `LIFT` | Lift object to hover height | Close |
| `GO_ABOVE_PLACE` | Move to above place position | Close |
| `GO_TO_PLACE` | Lower to place position | Close |
| `OPEN` | Open gripper to release object | Open |
| `RETURN_REST` | Return to initial REST pose | Open |
| `DONE` | Episode complete | Open |

#### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Gym task ID |
| `--num_envs` | `1` | Number of parallel environments |
| `--steps` | `500` | Maximum simulation steps per episode |
| `--seed` | `42` | Random seed for reproducibility |
| `--goal_xy` | `0.5 0.0` | Goal (x, y) position for initial cube placement |
| `--disable_fabric` | `0` | If `1`, disables Fabric backend |
| `--headless` | - | Run without GUI (from AppLauncher) |
| `--device` | `cuda:0` | Compute device (from AppLauncher) |

#### Example Output

```
Initial EE pose: [0.463, 0.0, 0.385]
Object teleported to: [0.500, 0.0, 0.021]
Target place position: (0.582, -0.031)

============================================================
Starting Expert B episode
============================================================
Step    0 | State: GO_ABOVE_OBJ     | EE: [0.375, -0.001, 0.525] | Obj: [0.500, -0.000, 0.021] | Gripper: +1.0
Step   80 | State: GO_TO_OBJ        | EE: [0.494, -0.000, 0.254] | Obj: [0.500, -0.000, 0.021] | Gripper: +1.0
Step  120 | State: CLOSE            | EE: [0.499, -0.000, 0.027] | Obj: [0.500, -0.000, 0.021] | Gripper: -1.0
Step  140 | State: LIFT             | EE: [0.487, -0.000, 0.194] | Obj: [0.488, 0.000, 0.188] | Gripper: -1.0
Step  200 | State: GO_TO_PLACE      | EE: [0.575, -0.030, 0.115] | Obj: [0.576, -0.030, 0.109] | Gripper: -1.0
Step  240 | State: RETURN_REST      | EE: [0.463, 0.000, 0.385] | Obj: [0.580, -0.031, 0.055] | Gripper: +1.0

Expert finished at step 279

============================================================
Episode finished
============================================================
Final object position: [0.580, -0.031, 0.055]
Target place position: [0.582, -0.031]
XY distance to target: 0.0023
Success (within 0.03m): True
EE distance to rest: 0.0046
```

---

### 3.3 Step 1: Collect reverse rollouts (policy B)

Reverse task:

- cube starts at the GOAL region ("in the plate")
- B picks and places the cube to a random table location

```bash
python scripts/10_collect_B_reverse_rollouts.py \
  --task Isaac-Lift-Cube-Franka-IK-Abs-v0 \
  --num_episodes 100 \
  --horizon 250 \
  --seed 0 \
  --out data/B_reverse_100eps.npz \
  --headless
```

Notes:
- We only keep SUCCESS episodes (or keep all and filter later).
- After placing, we add settle_steps so the cube comes to rest on the table.

### 3.4 Step 2: Reverse trajectories -> build forward BC dataset for A

Forward task we want A to learn:
- cube starts at random table location
- A moves it into GOAL region

We DO NOT simply reverse B actions. Instead we:
- reverse (obs, ee_pose, gripper) sequences
- build action labels as the next-step (ee_pose, gripper)

```bash
python scripts/20_make_A_forward_dataset.py \
  --in data/B_reverse_100eps.npz \
  --out data/A_forward_from_reverse.npz
```

### 3.5 Step 3: Train policy A with Behavior Cloning

```bash
python scripts/30_train_A_bc.py \
  --dataset data/A_forward_from_reverse.npz \
  --out runs/bc_A \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-3 \
  --seed 0
```

Outputs:
- `runs/bc_A/model.pt`
- `runs/bc_A/norm.json`
- `runs/bc_A/config.yaml`

### 3.6 Step 4: Evaluate policy A on the forward task

```bash
python scripts/40_eval_A.py \
  --task Isaac-Lift-Cube-Franka-IK-Abs-v0 \
  --checkpoint runs/bc_A/model.pt \
  --norm runs/bc_A/norm.json \
  --num_rollouts 50 \
  --horizon 250 \
  --headless
```

Report:
- success rate
- mean final distance to goal
- optional: save videos

---

## 4. What "success" means in this repo

A rollout is SUCCESS if:
- cube XY distance to goal center < `success_radius`
- (optional) cube Z close to table height
- (optional) gripper is open at the end

---

## 5. Code structure

```text
rev2fwd-il/
├── README.md
├── environment.yml
├── pyproject.toml
├── src/
│   └── rev2fwd_il/
│       ├── __init__.py
│       ├── sim/
│       │   ├── make_env.py          # Create Isaac Lab env, unify reset/step
│       │   ├── scene_api.py         # Get/set EE pose, object pose helpers
│       │   └── task_spec.py         # Task definition: goal, table bounds, success criteria
│       ├── experts/
│       │   ├── __init__.py
│       │   └── pickplace_expert_b.py  # Expert B: FSM for reverse pick-and-place
│       ├── data/
│       │   ├── episode.py           # Episode data structure
│       │   ├── recorder.py          # Rollout recorder
│       │   ├── io_npz.py            # Save/load npz datasets
│       │   ├── reverse_time.py      # Time reversal + action reconstruction
│       │   └── normalize.py         # Obs/action normalization
│       ├── models/
│       │   ├── mlp_policy.py        # Policy A: MLP outputs ee pose + gripper
│       │   └── losses.py            # Pose/quat/gripper losses
│       ├── train/
│       │   └── bc_trainer.py        # PyTorch BC training loop
│       ├── eval/
│       │   ├── rollout.py           # Run policy and compute metrics
│       │   └── metrics.py           # Success rate / distance curves
│       └── utils/
│           ├── seed.py              # Random seed utilities
│           ├── config.py            # Configuration utilities
│           └── logging.py           # Logging utilities
└── scripts/
    ├── 00_sanity_check_env.py       # Environment sanity check
    ├── 01_debug_expert_B_one_episode.py  # Debug Expert B FSM
    ├── 10_collect_B_reverse_rollouts.py  # Collect reverse rollouts
    ├── 20_make_A_forward_dataset.py      # Build forward dataset
    ├── 30_train_A_bc.py                  # Train policy A
    └── 40_eval_A.py                      # Evaluate policy A
```

### Key Modules

| Module | Description |
|--------|-------------|
| `sim/make_env.py` | Factory function to create Isaac Lab gymnasium environment |
| `sim/scene_api.py` | Helper functions to read/write scene state (EE pose, object pose, teleport) |
| `sim/task_spec.py` | Task parameters (goal position, table bounds, success radius) |
| `experts/pickplace_expert_b.py` | Finite state machine expert for reverse task |

---

## 6. Reproducibility checklist

- [ ] Fix seeds (python/numpy/torch)
- [ ] Log simulator step dt and action decimation
- [ ] Save dataset checksum + config
- [ ] Report how many SUCCESS reverse episodes were used to build the forward dataset

---

## 7. Common pitfalls

| Issue | Solution |
|-------|----------|
| Reversing raw actions breaks grasp/release timing | Use "next-step setpoint" reconstruction for actions |
| Cube still moving when B finishes | Add settle steps after placing |
| IK-Rel is harder to reverse | Use IK-Abs for this sanity check |
| Expert B sometimes fails | Filter to SUCCESS-only episodes |

---

## 8. Shell command notes

When running scripts, you may see commands like:

```bash
python script.py 2>&1 | head -100
```

**Explanation of `2>&1`:**
- `2` = stderr (standard error stream)
- `1` = stdout (standard output stream)
- `>&` = redirect one stream to another
- `2>&1` = redirect stderr to stdout (merge error messages with normal output)

This is useful when you want to capture both normal output and error messages together,
especially when piping to tools like `head`, `grep`, or `tee`.