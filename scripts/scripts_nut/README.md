# Nut Threading Data Collection (FORGE Environment)

This directory contains data collection scripts for the nut threading task based on the Isaac Lab FORGE environment.

## Directory Structure

| Script | Description |
|--------|-------------|
| `1_collect_data_nut_thread.py` | **Forward** data collection: starts with nut suspended, executes threading |
| `1_1_collect_data_nut_unthread.py` | **Reverse** data collection (rev2fwd-il): starts from threaded state, unthreads |
| `2_inspect_nut_data.py` | Data visualization: frame extraction, video generation, force curves, 3D trajectories |

---

## Environment Overview

### FORGE Nut Threading Task

- **Task ID**: `Isaac-Forge-NutThread-Direct-v0`
- **Robot**: Franka Panda (7-DOF arm + 2-DOF parallel gripper)
- **Task**: Thread a nut onto a bolt
- **Control frequency**: 15 Hz (120 Hz sim / 8 decimation)
- **Force sensor**: 6-DOF F/T sensor (force_sensor link), EMA smoothed (α=0.25)

### Action Space (7D)

| Dim | Meaning | Range | Notes |
|-----|---------|-------|-------|
| `action[0:3]` | Position target (xyz) | [-1, 1] | Offset relative to bolt top, scaled by `pos_action_bounds` |
| `action[3:5]` | Roll/Pitch | [-1, 1] | **Forced to zero** — only yaw rotation allowed |
| `action[5]` | Yaw rotation target | [-1, 1] | Maps to [-180°, +90°] absolute yaw angle |
| `action[6]` | Gripper control | [-1, 1] | **-1 = fully closed (0.0m), +1 = fully open (0.04m)** |

> **Important change**: `action[6]` was originally a success prediction signal (for RL training).
> We modified `forge_env.py` to use it as gripper control instead.
> Modified file: `isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py`, method `_apply_action()`.

### Observation Space

| Observation | Dim | Description |
|-------------|-----|-------------|
| `fingertip_pos_rel_fixed` | 3 | Fingertip position relative to bolt |
| `fingertip_quat` | 4 | Fingertip quaternion |
| `ee_linvel` | 3 | End-effector linear velocity |
| `ee_angvel` | 3 | End-effector angular velocity |
| `ft_force` | 3 | Force sensor readings (Fx, Fy, Fz) |
| `force_threshold` | 1 | Contact force penalty threshold |

---

## 1. Forward Data Collection

### Script: `1_collect_data_nut_thread.py`

Starts from the nut-in-air state and executes threading using a force-feedback state machine expert policy.

#### Basic Usage

```bash
# Collect 100 episodes in a single environment (headless mode)
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 \
    --out data/nut_thread.npz

# Specify image resolution
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 \
    --image_width 128 --image_height 128 \
    --out data/nut_thread_128.npz
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `Isaac-Forge-NutThread-Direct-v0` | Isaac Lab task ID |
| `--num_envs` | `1` | Number of parallel environments (use 1 for image collection) |
| `--num_episodes` | `100` | Number of episodes to collect |
| `--horizon` | `600` | Maximum steps per episode |
| `--image_width` | `128` | Image width |
| `--image_height` | `128` | Image height |
| `--rotation_speed` | `0.5` | Threading angular velocity (rad/s) |
| `--downward_force` | `5.0` | Target downward force during threading (N) |
| `--seed` | `0` | Random seed |
| `--out` | `data/nut_thread.npz` | Output NPZ file path |
| `--disable_fabric` | `0` | Whether to disable Fabric backend (PhysX GPU) |
| `--headless` | - | Run in headless mode (Isaac Lab argument) |

#### Expert Policy State Machine

Data collection uses the `NutThreadingExpert` force-feedback state machine with the following phases:

```
APPROACH → SEARCH → ENGAGE → THREAD → DONE
                                ↓
                      (triggered for multi-turn)
                    RELEASE → REPOSITION → REGRASP → THREAD
```

| Phase | ID | Description | Gripper |
|-------|----|-------------|---------|
| **APPROACH** | 0 | Move down until contact force detected | Closed |
| **SEARCH** | 1 | Spiral search to find thread alignment | Closed |
| **ENGAGE** | 2 | Reverse-forward rotation + downward pressure to catch threads | Closed |
| **THREAD** | 3 | Continuous rotation + adaptive downward force | Closed |
| **DONE** | 4 | Completed (torque too high or timeout) | Closed |
| **RELEASE** | 5 | Lift up + open gripper to release nut | **Open** |
| **REPOSITION** | 6 | Stay lifted, rotate yaw back to initial position | **Open** |
| **REGRASP** | 7 | Lower + re-grasp nut | Open then Close |

> **Multi-turn threading**: FORGE's yaw range is only [-180°, +90°] = 270°, less than one full turn.
> When yaw approaches the upper limit (0.9), RELEASE→REPOSITION→REGRASP is triggered automatically.
> Supports up to 15 re-grasps (~15 × 270° = 4050° ≈ 11 turns).

---

## 2. Reverse Data Collection (rev2fwd-il)

### Script: `1_1_collect_data_nut_unthread.py`

Starts from the nut-already-threaded state and unthreads (counterclockwise rotation + lift). Collected trajectories can be time-reversed to get forward threading demonstrations.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_1_collect_data_nut_unthread.py \
    --headless --num_episodes 500 \
    --out data/B_nut_unthread.npz
```

**rev2fwd-il approach**:
1. Initialize with nut already threaded onto bolt (goal state)
2. Execute simple "unthread" policy (counterclockwise rotation + lift)
3. Time-reverse collected trajectories to get forward threading demonstrations

Advantage: Unthreading is simpler (gravity assists contact) and the initial goal state can be set directly.

---

## 3. Data Inspection & Visualization

### Script: `2_inspect_nut_data.py`

```bash
# Basic inspection (print stats + generate frame images + video)
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz

# Specify episode, generate force curves and 3D trajectory
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz \
    --episode 5 --enable_force_plot --enable_trajectory_plot
```

Output directory: `data/inspect_<dataset_name>_<timestamp>/`

Generated files:
- `frame_N_table.png` — Table camera single frame
- `frame_N_wrist_cam.png` — Wrist camera single frame
- `frame_N_data.json` — Frame metadata (ee_pose, action, force, etc.)
- `episode_X_video.mp4` — Multi-camera side-by-side video
- `episode_X_with_force.mp4` — Video with force/torque curve overlay
- `episode_X_force_plot.png` — Force sensor time-series plot (requires `--enable_force_plot`)
- `episode_X_trajectory.png` — 3D trajectory plot (requires `--enable_trajectory_plot`)

---

## Output Data Format

All scripts output NPZ files containing an `episodes` array (list of dicts). Each episode dict contains:

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `obs` | (T, obs_dim) | float32 | Policy observation |
| `state` | (T, state_dim) | float32 | Full state observation (privileged) |
| `images` | (T, H, W, 3) | uint8 | Table camera RGB |
| `wrist_wrist_cam` | (T, H, W, 3) | uint8 | Wrist camera RGB |
| `ee_pose` | (T, 7) | float32 | End-effector pose [x, y, z, qw, qx, qy, qz] |
| `nut_pose` | (T, 7) | float32 | Nut pose |
| `bolt_pose` | (T, 7) | float32 | Bolt pose |
| `action` | (T, 7) | float32 | Action [pos(3), rot(3), gripper(1)] |
| `ft_force` | (T, 3) | float32 | Force sensor (Fx, Fy, Fz) |
| `ft_force_raw` | (T, 6) | float32 | Raw force/torque (Fx, Fy, Fz, Tx, Ty, Tz) |
| `joint_pos` | (T, 7) | float32 | Robot joint positions |
| `phase` | (T,) | int32 | State machine phase ID |
| `episode_length` | scalar | int | Episode total steps |
| `success` | scalar | bool | Whether episode succeeded |
| `success_threshold` | scalar | float | Success threshold value |
| `wrist_cam_names` | list | str | Names of wrist cameras |

### Reading Example

```python
import numpy as np

data = np.load("data/nut_thread.npz", allow_pickle=True)
episodes = data["episodes"]

ep = episodes[0]
print(f"Episode length: {ep['episode_length']}")
print(f"Success: {ep['success']}")
print(f"Images shape: {ep['images'].shape}")      # (T, 128, 128, 3)
print(f"Action shape: {ep['action'].shape}")       # (T, 7)
print(f"Force shape: {ep['ft_force'].shape}")      # (T, 3)
print(f"Gripper action range: [{ep['action'][:, 6].min():.1f}, {ep['action'][:, 6].max():.1f}]")
```

---

## FORGE Environment Files

FORGE environment code is located at `isaaclab_tasks/isaaclab_tasks/direct/forge/`:

| File | Description |
|------|-------------|
| `forge_env.py` | Main environment class (extends FactoryEnv), action processing, force sensing, rewards |
| `forge_env_cfg.py` | Environment config (action bounds, controller gains, observation noise, etc.) |
| `forge_tasks_cfg.py` | Task config (NutThread, PegInsert, GearMesh) |
| `forge_utils.py` | Utility functions (force/torque frame transforms, etc.) |
| `forge_events.py` | Domain randomization events (dead zone, etc.) |

---

## Required Patches (Reproduction Guide)

This project requires **two patches** to Isaac Lab's source code. Both modify files inside the installed
`site-packages` directory because Isaac Sim's runtime extension loader reads from there rather than from
workspace editable installs. **These patches must be re-applied after every reinstallation of Isaac Lab.**

### Patch 1: Gripper Control in `forge_env.py`

**File**: `isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py` (workspace copy is the source of truth)

**Problem**: The original FORGE environment has **two issues** preventing gripper open/close:

1. **Hardcoded closed gripper**: `_apply_action()` calls `generate_ctrl_signals(ctrl_target_gripper_dof_pos=0.0)`,
   locking the gripper closed. `action[:, 6]` was used as a success prediction signal, not gripper control.

2. **EMA smoothing makes gripper sluggish**: The parent `FactoryEnv._pre_physics_step()` applies EMA smoothing
   to **all** action dimensions:
   ```python
   # factory_env.py _pre_physics_step()
   self.actions = ema_factor * action + (1 - ema_factor) * self.actions
   ```
   With `ema_factor ∈ [0.025, 0.1]`, a -1→+1 gripper open command takes 30+ steps to converge,
   making the gripper appear stuck.

**Fix** — three modifications in `forge_env.py`:

**Change 1** — Override `_pre_physics_step()` to capture raw gripper action before EMA:
```python
def _pre_physics_step(self, action):
    """Override to capture raw gripper action before EMA smoothing."""
    self._raw_gripper_action = action[:, 6].clone()
    super()._pre_physics_step(action)  # parent EMA still applies to arm joints
```

**Change 2** — In `_apply_action()`, use the raw (non-EMA) gripper command:
```python
# End of _apply_action() — bypass EMA, use raw value directly
if hasattr(self, '_raw_gripper_action'):
    gripper_action = (self._raw_gripper_action + 1.0) / 2.0 * 0.04  # [-1,1] → [0, 0.04]m
else:
    gripper_action = (self.actions[:, 6] + 1.0) / 2.0 * 0.04  # fallback

self.generate_ctrl_signals(
    ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
    ctrl_target_gripper_dof_pos=gripper_action,  # replaces the original 0.0
)
```

**Change 3** — In `_reset_idx()`, initialize `_raw_gripper_action` to -1.0 (closed):
```python
if not hasattr(self, '_raw_gripper_action'):
    self._raw_gripper_action = torch.full((self.num_envs,), -1.0, device=self.device)
else:
    self._raw_gripper_action[:] = -1.0
```

Franka gripper DOF range: `[0.0, 0.04]` meters (0 = fully closed, 0.04 = fully open).

> **Note**: This modification affects `_get_rewards()` which originally reads `action[:, 6]` as a success
> prediction. For data collection only this is harmless; for RL training, the reward function must be
> updated accordingly.

**Deployment** — Isaac Sim loads `isaaclab_tasks` from site-packages at runtime, **not** from the
workspace editable install. After editing the workspace copy, sync it to site-packages:

```bash
# 1. Find the site-packages path
SITE_PKG=$(python -c "import isaaclab_tasks; import os; print(os.path.dirname(isaaclab_tasks.__file__))")

# 2. Copy the modified file
cp isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py "$SITE_PKG/direct/forge/forge_env.py"

# 3. Clear .pyc cache
find "$SITE_PKG/direct/forge/" -name "*.pyc" -delete
find "$SITE_PKG/direct/forge/__pycache__" -type f -delete 2>/dev/null
```

**Verification**: Add this to your collection script to confirm the correct file is loaded at runtime:
```python
import inspect
print(inspect.getfile(type(env.unwrapped)))  # should point to site-packages
```

---

### Patch 2: Camera Rendering Fix in `simulation_context.py`

**File**: `<site-packages>/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py`

This patch is applied **directly to the Isaac Lab library**. The workspace does not contain a copy of this file.

#### Problem A: Empty Camera Images — Missing `libGLU`

Isaac Sim's RTX rendering pipeline requires `libGLU.so.1`. If missing from the conda environment,
all RTX shaders fail to load (log shows 29 "Cannot load shader" errors), causing camera sensors to
return empty images with shape `(T, 0)` instead of `(T, H, W, 3)`.

**Diagnosis**: Check if camera data shape is `(num_envs, 0)` instead of `(num_envs, H, W, 3)`.

**Fix**:
```bash
# Option 1: Copy from conda package cache
cp /path/to/conda/pkgs/libglu-*/lib/libGLU.so* $CONDA_PREFIX/lib/

# Option 2: conda install
conda install -c conda-forge libglu

# Verify
ls $CONDA_PREFIX/lib/libGLU.so.1
```

> **Note**: This issue affects all environments (DirectRLEnv and ManagerBasedRLEnv alike) —
> it is a system-level rendering pipeline problem, not task-specific.

#### Problem B: Camera Images Exist but Don't Update — PhysX Fabric Sync Missing

**Symptom**: Camera image shape is correct `(T, 128, 128, 3)`, but all frames show the same static scene.
Frame-to-frame mean pixel difference < 1.0 (only render noise), even though the end-effector moves 5+ cm.

**Root cause**: When `--disable_fabric 1` is used (required for camera sensors), Isaac Lab's
`SimulationContext.forward()` method is responsible for syncing PhysX simulation data to the renderer
before each `_app.update()`. The **original** `forward()` looks like:

```python
# ORIGINAL (broken with disable_fabric=1)
def forward(self) -> None:
    if self._fabric_iface is not None:           # <-- gates EVERYTHING on fabric
        if self.physics_sim_view is not None and self.is_playing():
            self.physics_sim_view.update_articulations_kinematic()
        self._update_fabric(0.0, 0.0)
```

When fabric is disabled, `self._fabric_iface = None`, so **nothing** executes — no articulation kinematic
update and no data sync to the Hydra renderer. The rendered scene stays frozen at its initial state.

Isaac Lab overrides `render()` and does **not** call `super().render()`. The base class
(`isaacsim.core.api.SimulationContext.render()`) contains fallback logic that lazily loads
`physx_fabric_interface` and calls `force_update()` even when fabric is "disabled" — this is the
mechanism that syncs PhysX data to the renderer. Isaac Lab's override skips this entirely.

**Fix**: Replace the `forward()` method with a patched version that:
1. Always calls `update_articulations_kinematic()` (regardless of fabric state)
2. Falls back to raw `physx_fabric_interface.force_update()` when Isaac Lab's fabric interface is disabled

#### How to Apply Patch 2

**Step 1**: Locate the file:
```bash
SIMCTX=$(python -c "
import isaaclab.sim.simulation_context as m
import os
print(os.path.abspath(m.__file__))
")
echo "File to patch: $SIMCTX"
```

This is typically at:
```
<conda_env>/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py
```

**Step 2**: Find the `forward()` method (around line 466). The **original** code looks like:
```python
    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        if self._fabric_iface is not None:
            if self.physics_sim_view is not None and self.is_playing():
                # Update the articulations' link's poses before rendering
                self.physics_sim_view.update_articulations_kinematic()
            self._update_fabric(0.0, 0.0)
```

**Step 3**: Replace it with the following patched version:
```python
    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        # [PATCHED] Always sync articulation link poses to renderer, even without fabric.
        # Original code gated this on _fabric_iface, causing stale camera images
        # when disable_fabric=1 (Camera sensor requirement).
        if self.physics_sim_view is not None and self.is_playing():
            self.physics_sim_view.update_articulations_kinematic()
        if self._fabric_iface is not None:
            self._update_fabric(0.0, 0.0)
        else:
            # [PATCHED] When Isaac Lab's fabric is disabled (disable_fabric=1), we still
            # need to sync PhysX data to the renderer via the physx fabric interface.
            # The base class SimulationContext.render() does this lazily, but since Isaac Lab
            # overrides render() and calls forward() instead of super().render(), we handle it here.
            if not hasattr(self, '_physx_fabric_fallback'):
                self._physx_fabric_fallback = None
                try:
                    if self._extension_manager.is_extension_enabled("omni.physx.fabric"):
                        from omni.physxfabric import get_physx_fabric_interface
                        self._physx_fabric_fallback = get_physx_fabric_interface()
                except Exception:
                    pass
            if self._physx_fabric_fallback is not None:
                self._physx_fabric_fallback.force_update(0.0, 0.0)
```

**Step 4**: Clear the `.pyc` cache so Python recompiles the module:
```bash
find "$(dirname "$SIMCTX")/__pycache__" -name "simulation_context*" -delete
```

**Step 5** (optional): Back up the original file before patching:
```bash
cp "$SIMCTX" "${SIMCTX}.bak"
```

#### Automated Patch Script

For convenience, here is a one-shot Python script that applies Patch 2 automatically:

```python
"""Apply camera rendering patch to Isaac Lab's simulation_context.py.
Run with the target conda environment activated."""

import isaaclab.sim.simulation_context as m
import os

simctx_path = os.path.abspath(m.__file__)
print(f"Patching: {simctx_path}")

with open(simctx_path, 'r') as f:
    content = f.read()

# The original forward() — match both possible indentation styles
ORIGINAL = '''    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        if self._fabric_iface is not None:
            if self.physics_sim_view is not None and self.is_playing():
                # Update the articulations' link's poses before rendering
                self.physics_sim_view.update_articulations_kinematic()
            self._update_fabric(0.0, 0.0)'''

PATCHED = '''    def forward(self) -> None:
        """Updates articulation kinematics and fabric for rendering."""
        # [PATCHED] Always sync articulation link poses to renderer, even without fabric.
        # Original code gated this on _fabric_iface, causing stale camera images
        # when disable_fabric=1 (Camera sensor requirement).
        if self.physics_sim_view is not None and self.is_playing():
            self.physics_sim_view.update_articulations_kinematic()
        if self._fabric_iface is not None:
            self._update_fabric(0.0, 0.0)
        else:
            # [PATCHED] When Isaac Lab's fabric is disabled (disable_fabric=1), we still
            # need to sync PhysX data to the renderer via the physx fabric interface.
            if not hasattr(self, '_physx_fabric_fallback'):
                self._physx_fabric_fallback = None
                try:
                    if self._extension_manager.is_extension_enabled("omni.physx.fabric"):
                        from omni.physxfabric import get_physx_fabric_interface
                        self._physx_fabric_fallback = get_physx_fabric_interface()
                except Exception:
                    pass
            if self._physx_fabric_fallback is not None:
                self._physx_fabric_fallback.force_update(0.0, 0.0)'''

if '[PATCHED]' in content:
    print("Already patched — skipping.")
elif ORIGINAL in content:
    content = content.replace(ORIGINAL, PATCHED, 1)
    # Back up original
    with open(simctx_path + '.bak', 'w') as f_bak:
        f_bak.write(content)
    with open(simctx_path, 'w') as f:
        f.write(content)
    print("Patched successfully.")
    # Clear .pyc
    cache_dir = os.path.join(os.path.dirname(simctx_path), '__pycache__')
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith('simulation_context'):
                os.remove(os.path.join(cache_dir, fname))
                print(f"  Removed cache: {fname}")
    print("Done.")
else:
    print("ERROR: Could not find the original forward() code to patch.")
    print("The file may have been modified or the Isaac Lab version is different.")
    print("Please apply the patch manually — see README for the exact code.")
```

**Usage**:
```bash
conda activate <your_env>
python scripts/scripts_nut/patch_simulation_context.py
```

#### Verification

After applying both patches, run a short test and check that camera images actually update:

```python
import numpy as np
data = np.load("data/test.npz", allow_pickle=True)
imgs = data["episodes"][0]["images"]
# Frame-to-frame diff should be >> 1.0 (typically 2-10)
diff = np.abs(imgs[1].astype(float) - imgs[0].astype(float)).mean()
print(f"Frame 0→1 mean pixel diff: {diff:.2f}")  # Should be > 1.5, not ~0.5
# Large gap diff should show significant change
diff_large = np.abs(imgs[-1].astype(float) - imgs[0].astype(float)).mean()
print(f"Frame 0→{len(imgs)-1} mean pixel diff: {diff_large:.2f}")  # Should be > 5.0
```

---

### About `--disable_fabric`

When collecting data with camera sensors, the Fabric backend must be disabled to ensure USD stage
synchronization:

```bash
python scripts/scripts_nut/1_collect_data_nut_thread.py --headless --disable_fabric 1 ...
```

Without `--disable_fabric 1`, cameras may still capture images in some configurations, but scene data
may be out of sync. **Always use this flag when collecting image data.**

---

## Quick-Start Reproduction Checklist

On a fresh Isaac Lab installation, apply patches in this order:

```bash
# 0. Activate environment
conda activate rev2fwd_il

# 1. Apply Patch 1: Gripper control (copy workspace forge_env.py → site-packages)
SITE_PKG=$(python -c "import isaaclab_tasks; import os; print(os.path.dirname(isaaclab_tasks.__file__))")
cp isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py "$SITE_PKG/direct/forge/forge_env.py"
find "$SITE_PKG/direct/forge/__pycache__" -name "*.pyc" -delete 2>/dev/null

# 2. Apply Patch 2: Camera rendering (run automated script or patch manually)
python scripts/scripts_nut/patch_simulation_context.py

# 3. (If needed) Install libGLU for RTX shader loading
conda install -c conda-forge libglu

# 4. Verify with a short collection
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 1 --horizon 50 --disable_fabric 1 \
    --out /tmp/patch_test.npz

# 5. Check image updates
python -c "
import numpy as np
d = np.load('/tmp/patch_test.npz', allow_pickle=True)
imgs = d['episodes'][0]['images']
diff = np.abs(imgs[-1].astype(float) - imgs[0].astype(float)).mean()
print(f'First-to-last frame diff: {diff:.2f} (should be > 5.0)')
act = d['episodes'][0]['action']
print(f'Gripper action range: [{act[:,6].min():.1f}, {act[:,6].max():.1f}] (should be [-1.0, 1.0])')
"
```
