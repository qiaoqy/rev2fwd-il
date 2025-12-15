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

- `--task`:
  Gym task id to create via `gym.make(...)`.

- `--num_envs`:
  Number of parallel environments (vectorized) inside the same simulation.

- `--rl_device`:
  Alias for the AppLauncher `--device` argument (e.g. `cuda`, `cuda:0`, `cpu`).
  Use this to avoid argparse name conflicts.

- `--disable_fabric`:
  If set to `1`, disables Fabric in the parsed env cfg.

- `--steps`:
  How many `env.step(...)` calls to run.

- `--seed`:
  RNG seed for python/numpy/torch.

- `--random_action`:
  If `1`, sample actions from `env.action_space` each step.

- `--nudge_action`:
  If non-zero, use a stable zero action and add a small deterministic value to action dimension 0.
  Helpful for debugging IK control paths.

- `--print_every`:
  Print debug diagnostics every N steps (default: 1).

- `--step_timeout_s`:
  If a single `env.step()` is slower than this threshold, print a warning.

#### AppLauncher arguments (Isaac Lab / Isaac Sim)

The script also supports standard launcher flags injected by
`isaaclab.app.AppLauncher.add_app_launcher_args(parser)`.
Common ones include:

- `--headless`:
  Run Isaac Sim without opening a window.

- `--device`:
  Simulation device for Isaac Lab tensors (e.g. `cuda:0`).

Run `python scripts/00_sanity_check_env.py --help` to see the full list provided by your Isaac Lab version.

### 3.2 Step 1: Collect reverse rollouts (policy B)

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
  --headless 1
```
Notes:

- We only keep SUCCESS episodes (or keep all and filter later).
- After placing, we add settle_steps so the cube comes to rest on the table.

### 3.3 Step 2: Reverse trajectories -> build forward BC dataset for A
Forward task we want A to learn:

- cube starts at random table location
- A moves it into GOAL region
We DO NOT simply reverse B actions.
Instead we:

- reverse (obs, ee_pose, gripper) sequences
- build action labels as the next-step (ee_pose, gripper)

```bash
python scripts/20_make_A_forward_dataset.py \
  --in data/B_reverse_100eps.npz \
  --out data/A_forward_from_reverse.npz
```

### 3.4 Step 3: Train policy A with Behavior Cloning


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

- runs/bc_A/model.pt
- runs/bc_A/norm.json
- runs/bc_A/config.yaml

### 3.5 Step 4: Evaluate policy A on the forward task

```bash
python scripts/40_eval_A.py \
  --task Isaac-Lift-Cube-Franka-IK-Abs-v0 \
  --checkpoint runs/bc_A/model.pt \
  --norm runs/bc_A/norm.json \
  --num_rollouts 50 \
  --horizon 250 \
  --headless 1
```
Report:

- success rate
- mean final distance to goal
- optional: save videos

## 4. What "success" means in this repo
A rollout is SUCCESS if:

- cube XY distance to goal center < success_radius
- (optional) cube Z close to table height
- (optional) gripper is open at the end

## 5. Code structure
src/rev2fwd_il/sim/

make_env.py: create Isaac Lab env, unify reset/step, flatten obs["policy"]
scene_api.py: get ee pose, object pose, set object pose, etc.
task_spec.py: table bounds, goal pose, sampling, success predicate

src/rev2fwd_il/experts/

pickplace_sm.py: scripted expert policy B (state machine / motion primitive)
wrappers.py: common policy interface

src/rev2fwd_il/data/

episode.py: Episode dataclass
recorder.py: rollout and record episodes
reverse_time.py: reverse episodes and reconstruct action labels for BC
io_npz.py: save/load datasets
normalize.py: obs/action normalization

src/rev2fwd_il/models/

mlp_policy.py: MLP policy A output (ee pose + gripper)
losses.py: position + quaternion + gripper losses

src/rev2fwd_il/train/

bc_trainer.py: PyTorch BC training loop

src/rev2fwd_il/eval/

rollout.py: run policy and compute metrics

```text
rev2fwd-il/
├── README.md
├── environment.yml
├── pyproject.toml                  # 可选：做成可 pip install -e 的包
├── src/
│   └── rev2fwd_il/
│       ├── __init__.py
│       ├── sim/
│       │   ├── make_env.py          # 创建 Isaac Lab env、reset/step 的统一封装
│       │   ├── scene_api.py         # 读取 ee/object pose、设置 object pose 的 helper
│       │   └── task_spec.py         # 任务定义：目标点、采样桌面点、成功判据
│       ├── experts/
│       │   ├── pickplace_sm.py      # “现成 policy B”：复用/移植 IsaacLab lift_cube_sm state machine
│       │   └── wrappers.py          # ExpertPolicy 接口封装（act/reset）
│       ├── data/
│       │   ├── episode.py           # Episode 数据结构（obs, ee_pose, gripper, obj_pose, etc.）
│       │   ├── recorder.py          # rollout 录制器
│       │   ├── io_npz.py            # 保存/加载 npz
│       │   ├── reverse_time.py      # 倒放 + 重建动作标签（关键）
│       │   └── normalize.py         # obs/action 归一化统计与应用
│       ├── models/
│       │   ├── mlp_policy.py        # policy A：MLP 输出 ee pose + gripper
│       │   └── losses.py            # pose loss / quat loss / gripper loss
│       ├── train/
│       │   └── bc_trainer.py        # 纯 pytorch BC 训练循环
│       ├── eval/
│       │   ├── rollout.py           # 用 policy A 跑 env 并算成功率
│       │   └── metrics.py           # success rate / distance 曲线
│       └── utils/
│           ├── seed.py
│           ├── config.py            # argparse/hydra 都行；最简就 argparse
│           └── logging.py
└── scripts/
    ├── 00_sanity_check_env.py
    ├── 10_collect_B_reverse_rollouts.py
    ├── 20_make_A_forward_dataset.py
    ├── 30_train_A_bc.py
    └── 40_eval_A.py
```

## 6. Reproducibility checklist
Fix seeds (python/numpy/torch)
Log simulator step dt and action decimation
Save dataset checksum + config
Report how many SUCCESS reverse episodes were used to build the forward dataset

## 7. Common pitfalls
If you reverse raw actions, grasp/release timing will break.
Use "next-step setpoint" reconstruction for actions.
If the cube is still moving when B finishes, A will start from unstable states.
Add settle steps after placing.
IK-Rel is harder to reverse; use IK-Abs for this sanity check.
If the expert B sometimes fails, filter to SUCCESS-only episodes.