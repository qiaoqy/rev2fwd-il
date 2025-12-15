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
- `Isaac-Lift-Cube-Franka-IK-Abs-v0`

Isaac Lab provides state-machine examples that can pick a cube and move it to a desired pose.
We reuse that logic as the "prebuilt policy B".

> If you installed Isaac Lab from pip packages:
> note that Isaac Lab pip packages do NOT ship the standalone example scripts,
> so this repo provides its own runner scripts.

---

## 2. Installation

### Option A (recommended): Use an existing Isaac Lab installation
1) Install Isaac Lab (from source or pip) following official docs
2) Create a python env and install this repo:
```bash
conda env create -f environment.yml
conda activate rev2fwd_il
pip install -e .
```

### Option B: Install Isaac Lab via pip (advanced users)
The official docs suggest pip packages for Isaac Lab 2.x (Isaac Sim 5.x requires Python 3.11).
Example (adjust CUDA / torch to your machine):

```bash
pip install isaaclab[isaacsim,all]==2.3.0 --extra-index-url https://pypi.nvidia.com
```
## 3. Quickstart
### 3.1 Sanity check: can we run the environment?
```bash
python scripts/00_sanity_check_env.py \
  --task Isaac-Lift-Cube-Franka-IK-Abs-v0 \
  --headless 0
```
Expected:
- Franka + cube on a table
- the scripted expert should move towards the cube and perform a pick motion

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