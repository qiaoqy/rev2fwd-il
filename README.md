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