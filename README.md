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
REST → GO_ABOVE_OBJ → GO_TO_OBJ → CLOSE → LIFT → GO_ABOVE_PLACE → GO_TO_PLACE → LOWER_TO_RELEASE → OPEN → RETURN_REST → DONE
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
| `LOWER_TO_RELEASE` | Lower further for stable release | Close |
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

This script collects demonstration data from Expert B performing the **reverse** pick-and-place task.

**Reverse task definition:**
- Cube starts at the GOAL position (plate center at ~[0.5, 0.0])
- Expert B picks it up and places it at a RANDOM position on the table
- Robot returns to REST pose

```bash
# Basic usage (headless mode, 100 episodes)
python scripts/10_collect_B_reverse_rollouts.py --headless --num_episodes 100

# Custom output path and seed
python scripts/10_collect_B_reverse_rollouts.py --headless --num_episodes 50 \
    --out data/B_reverse_50eps.npz --seed 42

# With longer horizon (if expert doesn't complete FSM)
python scripts/10_collect_B_reverse_rollouts.py --headless --num_episodes 100 \
    --horizon 500 --settle_steps 50
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Isaac Lab Gym task ID |
| `--num_episodes` | `100` | Number of episodes to collect |
| `--horizon` | `400` | Max steps per episode (Expert B needs ~300-400 steps) |
| `--settle_steps` | `40` | Extra steps after expert finishes to let cube settle |
| `--seed` | `0` | Random seed for reproducibility |
| `--out` | `data/B_reverse_100eps.npz` | Output NPZ file path |
| `--disable_fabric` | `0` | If `1`, disables Fabric backend |
| `--headless` | - | Run without GUI |

#### Output Data Format

The script saves an NPZ file with the following structure for each episode `i`:

| Key | Shape | Description |
|-----|-------|-------------|
| `obs_i` | `(T, 36)` | Policy observation sequence |
| `ee_pose_i` | `(T, 7)` | End-effector pose `[x, y, z, qw, qx, qy, qz]` |
| `obj_pose_i` | `(T, 7)` | Object (cube) pose `[x, y, z, qw, qx, qy, qz]` |
| `gripper_i` | `(T,)` | Gripper action (`+1`=open, `-1`=close) |
| `place_pose_i` | `(7,)` | Target place position (random table position) |
| `goal_pose_i` | `(7,)` | Goal position (plate center, fixed) |
| `success_i` | `bool` | Whether cube ended up near target position |

**Notes:**
- Only episodes where Expert B completes the full FSM (reaches DONE state) are saved
- The `success` flag indicates whether the cube is within `success_radius` of the target
- Episode length `T` is typically 330-400 steps

#### Example Output

```
============================================================
Collecting 30 reverse rollouts
Settings:
  - horizon: 400
  - settle_steps: 40
  - success_radius: 0.08m
  - release_z_offset: -0.015m
  - Only saving episodes with completed FSM (DONE state)
============================================================

Attempt    1 | Saved: 1/30 | Completed: 1 (100.0%) | Success: 0 | Rate: 0.25 ep/s | Length: 394
Attempt   10 | Saved: 10/30 | Completed: 10 (100.0%) | Success: 1 | Rate: 0.31 ep/s | Length: 330
Attempt   20 | Saved: 20/30 | Completed: 20 (100.0%) | Success: 2 | Rate: 0.33 ep/s | Length: 394
Attempt   30 | Saved: 30/30 | Completed: 30 (100.0%) | Success: 3 | Rate: 0.33 ep/s | Length: 392

============================================================
Collection finished in 92.3s
Total attempts: 30
Completed FSM: 30 (100.0%)
Saved episodes: 30
Success (cube at target): 3 (10.0%)
============================================================

Saved 30 episodes to data/B_reverse_30eps.npz
```

---

### 3.4 Replay B Episode (Video Recording)

This script replays a single episode from the collected B dataset and saves it as an MP4 video.

```bash
# Replay episode 0 with GUI
python scripts/11_replay_B_episode.py

# Replay episode 5 in headless mode
python scripts/11_replay_B_episode.py --episode 5 --headless

# Custom dataset and output
python scripts/11_replay_B_episode.py --dataset data/B_reverse_100eps.npz --episode 10 --out data/B_ep10.mp4
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/B_reverse_latest.npz` | Path to B dataset NPZ file |
| `--episode` | `0` | Episode index to replay |
| `--out` | `data/B_ep{episode}.mp4` | Output video path |
| `--playback_speed` | `1.0` | Playback speed multiplier |

---

### 3.5 Collect Reverse Rollouts with Images

This script extends script 10 by adding a camera sensor to capture RGB images at each step.
The camera is dynamically added to the environment configuration without modifying isaaclab_tasks.

```bash
# Basic usage (headless mode, 100 episodes)
python scripts/12_collect_B_with_images.py --headless --num_episodes 100

# Custom resolution
python scripts/12_collect_B_with_images.py --headless --num_episodes 50 \
    --image_width 256 --image_height 256 --out data/B_images_256.npz

# With GUI visualization
python scripts/12_collect_B_with_images.py --num_episodes 10
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Isaac Lab Gym task ID |
| `--num_episodes` | `100` | Number of episodes to collect |
| `--num_envs` | `4` | Parallel environments (fewer for images due to memory) |
| `--image_width` | `128` | Width of captured images |
| `--image_height` | `128` | Height of captured images |
| `--horizon` | `400` | Max steps per episode |
| `--settle_steps` | `40` | Extra steps after expert finishes |
| `--seed` | `0` | Random seed |
| `--out` | `data/B_images_latest.npz` | Output NPZ file path |

#### Output Data Format

| Key | Shape | Description |
|-----|-------|-------------|
| `obs` | `(T, 36)` | Policy observation sequence |
| `images` | `(T, H, W, 3)` | RGB images (uint8, 0-255) |
| `ee_pose` | `(T, 7)` | End-effector pose |
| `obj_pose` | `(T, 7)` | Object pose |
| `gripper` | `(T,)` | Gripper action |
| `place_pose` | `(7,)` | Target place position |
| `goal_pose` | `(7,)` | Goal position |
| `success` | `bool` | Success flag |

**Note:** The `--headless` mode requires `enable_cameras=True` internally for image rendering.

---

### 3.6 Inspect Image Data

This script inspects the image dataset collected by script 12, without requiring Isaac Lab:
- Extracts a single frame as PNG + JSON for inspection
- Generates an MP4 video from one episode's camera sequence

```bash
# Basic usage - inspect episode 0, frame 0
python scripts/13_inspect_B_images.py

# Specify dataset and episode
python scripts/13_inspect_B_images.py --dataset data/B_images_latest.npz --episode 5

# Specify which frame to extract
python scripts/13_inspect_B_images.py --episode 3 --frame 100

# Custom output folder name
python scripts/13_inspect_B_images.py --name my_inspection
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/B_images_latest.npz` | Path to image dataset |
| `--episode` | `0` | Episode index to inspect |
| `--frame` | `0` | Frame index for PNG/JSON extraction |
| `--name` | `""` | Optional suffix for output folder |
| `--fps` | `30` | FPS for output video |

#### Output Files

```
data/inspect_YYYYMMDD_HHMMSS_<name>/
├── episode_info.json       # Episode metadata and dataset statistics
├── frame_<N>.png           # Single frame image
├── frame_<N>_data.json     # Frame data (obs breakdown, poses, gripper)
└── episode_<M>_video.mp4   # Full episode video (H.264 encoded)
```

#### Observation Vector Breakdown

The `obs` field is a 36-dimensional vector:

| Index | Dim | Field | Description |
|-------|-----|-------|-------------|
| 0-8 | 9 | `joint_pos_rel` | Joint positions relative to default |
| 9-17 | 9 | `joint_vel_rel` | Joint velocities |
| 18-20 | 3 | `object_position` | Object XYZ in robot root frame |
| 21-27 | 7 | `target_object_position` | Target pose (pos + quat) |
| 28-35 | 8 | `last_action` | Previous action |

**Note:** At frame 0, most values are 0 because the robot is at default position and stationary.

---

### 3.8 Step 2: Reverse trajectories -> build forward BC dataset for A

This script converts REVERSE rollouts from Expert B into FORWARD training data for Policy A.

**Core idea:** A reverse trajectory (goal → table) when played backwards becomes a forward trajectory (table → goal).

**Key challenge:** We cannot simply reverse the actions because:
1. The gripper open/close timing would be wrong
2. IK-Abs actions are absolute poses, not deltas

**Solution:** We reconstruct action labels using heuristics:
- **EE pose action:** Use the NEXT timestep's EE pose (where we want to go)
- **Gripper action:** Infer from EE-object distance and object-goal proximity

```bash
# Basic usage (use all episodes)
python scripts/20_make_A_forward_dataset.py \
    --input data/B_reverse_100eps.npz \
    --out data/A_forward_from_reverse.npz \
    --success_only 0

# Only use successful episodes (stricter filtering)
python scripts/20_make_A_forward_dataset.py \
    --input data/B_reverse_100eps.npz \
    --out data/A_forward_from_reverse.npz \
    --success_only 1
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data/B_reverse_100eps.npz` | Input NPZ file from script 10 |
| `--out` | `data/A_forward_from_reverse.npz` | Output NPZ file for BC training |
| `--success_only` | `1` | If `1`, only use successful episodes |

#### Gripper Heuristic

For the forward task (table → goal), the gripper action is inferred as:

| Condition | Gripper | Reasoning |
|-----------|---------|-----------|
| EE close to object AND object far from goal | **CLOSE** (-1) | Grasp object to transport it |
| EE far from object OR object close to goal | **OPEN** (+1) | Approaching or finished placing |

A healthy gripper CLOSE ratio is typically **30-50%** (grasping/transporting phase).

#### Output Data Format

The script outputs a flat NPZ file for BC training:

| Key | Shape | Description |
|-----|-------|-------------|
| `obs` | `(N, 36)` | Observations in FORWARD time order |
| `act` | `(N, 8)` | Actions: `[ee_pose(7), gripper(1)]` |
| `ep_id` | `(N,)` | Episode index (for debugging/analysis) |

Where `N = sum(T_i - 1)` for all episodes (we lose one step per episode because `action[t] = next_ee_pose[t+1]`).

#### Example Output

```
Loading episodes from data/B_reverse_30eps.npz
Loaded 30 episodes from data/B_reverse_30eps.npz

============================================================
Processing 30 episodes
============================================================

Episode    1 | Length:  329 | CLOSE ratio: 35.0%
Episode   20 | Length:  393 | CLOSE ratio: 34.1%

============================================================
Dataset Statistics
============================================================
Total episodes: 30
Total steps: 10170
Average episode length: 339.0
Observation dim: 36
Action dim: 8
Gripper CLOSE ratio: 35.0%
============================================================

Saved forward BC dataset to data/A_forward_from_reverse.npz
  obs shape: (10170, 36)
  act shape: (10170, 8)
  ep_id shape: (10170,)
```

---

### 3.9 Step 3: Train Policy A with Behavior Cloning

This script trains an MLP policy using Behavior Cloning (BC) on the forward dataset generated in Step 2.

**Behavior Cloning Algorithm:**
- Supervised learning approach: treat imitation as regression
- Input: observation `obs` (36-dim)
- Output: action `act` (8-dim) = [ee_pose(7), gripper(1)]
- Loss: weighted combination of position MSE, quaternion loss, and gripper MSE

**Network Architecture:**
- MLP with configurable hidden layers (default: 256-256)
- Output constraints:
  - Position (3D): direct output, no constraint
  - Quaternion (4D): normalized to unit length
  - Gripper (1D): tanh activation, maps to [-1, 1]

```bash
# Standard training (recommended settings)
python scripts/30_train_A_bc.py \
    --dataset data/A_forward_from_reverse.npz \
    --out runs/bc_A \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --seed 0

# Quick test run
python scripts/30_train_A_bc.py \
    --dataset data/A_forward_from_reverse.npz \
    --out runs/bc_A_test \
    --epochs 10 \
    --batch_size 64

# Custom network architecture (deeper network)
python scripts/30_train_A_bc.py \
    --dataset data/A_forward_from_reverse.npz \
    --out runs/bc_A_large \
    --hidden 512 512 256 \
    --epochs 300

# Adjust loss weights (emphasize position accuracy)
python scripts/30_train_A_bc.py \
    --dataset data/A_forward_from_reverse.npz \
    --out runs/bc_A_pos \
    --pos_weight 2.0 \
    --quat_weight 1.0 \
    --gripper_weight 0.5
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `data/A_forward_from_reverse.npz` | Input NPZ file with `obs` and `act` arrays |
| `--out` | `runs/bc_A` | Output directory for model and config |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `256` | Mini-batch size for SGD |
| `--lr` | `1e-3` | Learning rate for Adam optimizer |
| `--seed` | `0` | Random seed for reproducibility |
| `--hidden` | `256 256` | Hidden layer sizes (space-separated) |
| `--pos_weight` | `1.0` | Weight for position MSE loss |
| `--quat_weight` | `1.0` | Weight for quaternion loss |
| `--gripper_weight` | `1.0` | Weight for gripper MSE loss |
| `--print_every` | `10` | Print loss every N epochs |
| `--device` | auto | Torch device (`cuda` or `cpu`) |

#### Output Files

The script saves three files to the `--out` directory:

| File | Description |
|------|-------------|
| `model.pt` | PyTorch checkpoint with model weights and architecture info |
| `norm.json` | Observation normalization statistics (mean, std) |
| `config.yaml` | Training configuration and final metrics |

#### Loss Functions

| Component | Formula | Description |
|-----------|---------|-------------|
| Position | MSE(pos_pred, pos_gt) | Mean squared error on XYZ |
| Quaternion | 1 - \|dot(q_pred, q_gt)\| | Handles q ≡ -q equivalence |
| Gripper | MSE(grip_pred, grip_gt) | Works with -1/+1 labels |

#### Example Output

```
============================================================
Behavior Cloning Training
============================================================
Dataset: data/A_forward_from_reverse.npz
Output: runs/bc_A
Device: cuda
Seed: 0
============================================================

Loading dataset...
  Samples: 37121
  Obs dim: 36
  Act dim: 8

Computing observation normalization...
  Mean range: [-0.2673, 1.0000]
  Std range: [0.0000, 1.1189]

DataLoader created: 146 batches per epoch

Model created: 77,320 parameters

============================================================
Starting training...
============================================================

Epoch    1/200 | Loss: 0.086595 (pos: 0.006664, quat: 0.009868, grip: 0.070064) | Time: 0.5s
Epoch   10/200 | Loss: 0.000286 (pos: 0.000129, quat: 0.000104, grip: 0.000053) | Time: 2.6s
...
Epoch  200/200 | Loss: 0.000036 (pos: 0.000030, quat: 0.000005, grip: 0.000000) | Time: 46.3s

Training completed in 46.3s

============================================================
Training Summary
============================================================
Final loss: 0.000036
  Position: 0.000030
  Quaternion: 0.000005
  Gripper: 0.000000
============================================================

Saved model to runs/bc_A/model.pt
Saved normalization to runs/bc_A/norm.json
Saved config to runs/bc_A/config.yaml
```

---

### 3.10 Step 4: Evaluate Policy A on the Forward Task

This script evaluates the trained BC policy on the actual forward pick-and-place task in Isaac Lab.

**Forward Task Definition:**
- Cube starts at a RANDOM position on the table (default env.reset behavior)
- Policy A should pick it up and place it at the GOAL (plate center)
- No teleportation of the cube is performed

**Success Criterion:**
- Cube XY distance to goal < `success_radius` (default 3cm)
- Note: Gripper state is NOT considered for success determination

```bash
# Standard evaluation (headless mode, 50 rollouts)
python scripts/40_eval_A.py \
    --checkpoint runs/bc_A/model.pt \
    --norm runs/bc_A/norm.json \
    --num_rollouts 50 \
    --horizon 450 \
    --headless

# Quick validation (fewer rollouts)
python scripts/40_eval_A.py \
    --checkpoint runs/bc_A/model.pt \
    --norm runs/bc_A/norm.json \
    --num_rollouts 5 \
    --horizon 450 \
    --headless

# With GUI visualization (for debugging)
python scripts/40_eval_A.py \
    --checkpoint runs/bc_A/model.pt \
    --norm runs/bc_A/norm.json \
    --num_rollouts 10 \
    --horizon 450
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `runs/bc_A/model.pt` | Path to trained model checkpoint |
| `--norm` | `runs/bc_A/norm.json` | Path to normalization statistics |
| `--task` | `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Isaac Lab Gym task ID |
| `--num_envs` | `1` | Number of parallel environments |
| `--num_rollouts` | `50` | Number of evaluation episodes |
| `--horizon` | `250` | Maximum steps per episode |
| `--seed` | `42` | Random seed for reproducibility |
| `--disable_fabric` | `0` | If `1`, disables Fabric backend |
| `--headless` | - | Run without GUI (from AppLauncher) |

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | Percentage of rollouts where cube reaches goal |
| Avg Final Dist | Mean XY distance from cube to goal at episode end |
| Min Final Dist | Best performance across all rollouts |
| Max Final Dist | Worst performance across all rollouts |

#### Example Output

```
============================================================
Evaluating Policy A on Forward Task
============================================================
Goal XY: [0.5 0. ]
Success radius: 0.03
Horizon: 450
Num rollouts: 10
============================================================

Rollout   1/10 | Steps: 250 | Final dist: 0.2173 | Gripper: +1.00 | Success: False
Rollout   2/10 | Steps: 250 | Final dist: 0.1680 | Gripper: +1.00 | Success: False
Rollout   3/10 | Steps: 250 | Final dist: 0.1718 | Gripper: +0.99 | Success: False
Rollout   4/10 | Steps: 250 | Final dist: 0.0782 | Gripper: +0.99 | Success: False
Rollout   5/10 | Steps: 250 | Final dist: 0.1818 | Gripper: +1.00 | Success: False
Rollout   6/10 | Steps: 250 | Final dist: 0.0514 | Gripper: +1.00 | Success: False
Rollout   7/10 | Steps: 250 | Final dist: 0.0225 | Gripper: -0.93 | Success: True
Rollout   8/10 | Steps: 250 | Final dist: 0.0673 | Gripper: +1.00 | Success: False
Rollout   9/10 | Steps: 250 | Final dist: 0.2377 | Gripper: +0.99 | Success: False
Rollout  10/10 | Steps: 250 | Final dist: 0.0780 | Gripper: -1.00 | Success: False

============================================================
Evaluation Results
============================================================
Success rate: 10.0% (1/10)
Average final distance: 0.1274m
Min final distance: 0.0225m
Max final distance: 0.2377m
============================================================

============================================================
FINAL EVALUATION SUMMARY
============================================================
Checkpoint: runs/bc_A/model.pt
Task: Isaac-Lift-Cube-Franka-IK-Abs-v0
Num rollouts: 10
Horizon: 450
------------------------------------------------------------
SUCCESS RATE: 10.0%
AVG FINAL DIST: 0.1274m
MIN FINAL DIST: 0.0225m
MAX FINAL DIST: 0.2377m
============================================================
```

#### Notes on Evaluation

- **Episode Length:** The default environment timeout is 250 steps. Use `--horizon` to set the maximum, but the episode may terminate earlier.
- **Random Initialization:** Each rollout starts with the cube at a different random position on the table.
- **Sanity Check:** Even with low success rate, `avg_final_dist` should be smaller than random policy (~0.2-0.3m).

---

## 4. Data Formats Reference

### 4.1 Episode Data Structure (from script 10)

Each episode contains the following fields:

| Field | Shape | Physical Meaning |
|-------|-------|------------------|
| `obs` | `(T, 36)` | Policy observation: joint_pos(9), joint_vel(9), object_pos(3), target_pos(7), prev_actions(8) |
| `ee_pose` | `(T, 7)` | End-effector pose `[x, y, z, qw, qx, qy, qz]` in world frame |
| `obj_pose` | `(T, 7)` | Object (cube) pose `[x, y, z, qw, qx, qy, qz]` in world frame |
| `gripper` | `(T,)` | Gripper action: `+1.0` = open, `-1.0` = close |
| `place_pose` | `(7,)` | Target place pose (Expert B's random table target) |
| `goal_pose` | `(7,)` | Fixed goal pose (plate center for forward task) |
| `success` | `bool` | Whether cube ended up near target position |

### 4.2 BC Training Dataset (from script 20)

| Field | Shape | Physical Meaning |
|-------|-------|------------------|
| `obs` | `(N, 36)` | Observations in FORWARD time order |
| `act` | `(N, 8)` | Actions: `[ee_pose(7), gripper(1)]` |
| `ep_id` | `(N,)` | Episode index (for debugging/analysis) |

### 4.3 Coordinate System

- **World frame:** Z-up, origin at robot base
- **Table height:** ~0.0m (Z coordinate)
- **Cube size:** ~5cm cube, center at ~0.025m when on table
- **Goal position:** Plate center at approximately `[0.5, 0.0, 0.055]`
- **Table bounds:** X ∈ [0.35, 0.65], Y ∈ [-0.3, 0.3]

---

## 5. What "success" means in this repo

A rollout is SUCCESS if:
- cube XY distance to goal center < `success_radius`
- (optional) cube Z close to table height
- (optional) gripper is open at the end

---

## 6. Code structure

```text
rev2fwd-il/
├── README.md
├── environment.yml
├── pyproject.toml
├── data/                              # Collected datasets
│   ├── B_reverse_*.npz               # Reverse rollouts from Expert B
│   └── A_forward_from_reverse.npz    # Forward BC dataset for Policy A
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
│       │   ├── recorder.py          # Rollout recorder for Expert B
│       │   ├── io_npz.py            # Save/load NPZ datasets
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
    ├── 00_sanity_check_env.py           # Environment sanity check
    ├── 01_debug_expert_B_one_episode.py # Debug Expert B FSM
    ├── 10_collect_B_reverse_rollouts.py # Step 1: Collect reverse rollouts (state-only)
    ├── 11_replay_B_episode.py           # Replay & record B episode as video
    ├── 12_collect_B_with_images.py      # Collect reverse rollouts with camera images
    ├── 13_inspect_B_images.py           # Inspect image data (PNG, JSON, MP4)
    ├── 20_make_A_forward_dataset.py     # Step 2: Build forward dataset
    ├── 21_replay_A_episode.py           # Replay forward dataset episode
    ├── 30_train_A_bc.py                 # Step 3: Train policy A
    └── 40_eval_A.py                     # Step 4: Evaluate policy A
```

### Key Modules

| Module | Description |
|--------|-------------|
| `sim/make_env.py` | Factory function to create Isaac Lab gymnasium environment |
| `sim/scene_api.py` | Helper functions to read/write scene state (EE pose, object pose, teleport) |
| `sim/task_spec.py` | Task parameters (goal position, table bounds, success radius) |
| `experts/pickplace_expert_b.py` | Finite state machine expert for reverse task |
| `data/recorder.py` | Rollout recorder that runs Expert B and saves trajectories |
| `data/reverse_time.py` | Time reversal and gripper heuristic for action reconstruction |
| `data/io_npz.py` | NPZ file I/O for episodes and datasets |

---

## 7. Reproducibility checklist

- [ ] Fix seeds (python/numpy/torch)
- [ ] Log simulator step dt and action decimation
- [ ] Save dataset checksum + config
- [ ] Report how many SUCCESS reverse episodes were used to build the forward dataset

---

## 8. Common pitfalls

| Issue | Solution |
|-------|----------|
| Reversing raw actions breaks grasp/release timing | Use "next-step setpoint" reconstruction for actions |
| Cube still moving when B finishes | Add settle steps after placing |
| IK-Rel is harder to reverse | Use IK-Abs for this sanity check |
| Expert B sometimes fails | Filter to SUCCESS-only episodes |
| Expert B doesn't reach DONE state | Increase `--horizon` (default 400 steps) |
| Gripper CLOSE ratio is 0% or 100% | Check gripper heuristic thresholds in `reverse_time.py` |

---

## 9. Shell command notes

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