# Piper Local Real-Robot Experiments

This directory contains scripts for validating the Rev2Fwd imitation learning method on a **real Piper robotic arm**. The algorithm architecture follows [PIPELINE_README.md](../scripts_pick_place/PIPELINE_README.md).

The goal is to perform a **Pick & Place** task with local data collection and fine-tuning on an **NVIDIA RTX 4090** GPU.

---

## Table of Contents

1. [Hardware Setup](#1-hardware-setup)
2. [Software Dependencies](#2-software-dependencies)
3. [Coordinate System & Workspace](#3-coordinate-system--workspace)
4. [Data Format Specification](#4-data-format-specification)
5. [Script 0: System Test](#5-script-0-system-test)
6. [Script 1: PS5 Teleoperation](#6-script-1-ps5-teleoperation)
7. [Script 2: Data Visualization](#7-script-2-data-visualization)
8. [Script 3: Time Reversal](#8-script-3-time-reversal)
9. [Script 4: Train Diffusion Policy](#9-script-4-train-diffusion-policy)
10. [Script 5: Model Evaluation](#10-script-5-model-evaluation)
11. [Script 6: FSM Data Collection](#11-script-6-fsm-data-collection)
12. [Script 7: Train DiT Flow Policy](#12-script-7-train-dit-flow-policy)
13. [Safety Guidelines](#13-safety-guidelines)
14. [Debugging Guide](#14-debugging-guide--known-issues)
15. [TODO / Open Questions](#15-todo--open-questions)

---

## 1. Hardware Setup

### 1.1 Robotic Arm

| Component | Specification |
|-----------|---------------|
| Robot | Piper 6-DoF Robotic Arm (AgileX/Songlin) |
| Interface | CAN bus (single arm) |
| Control Mode | End-effector Pose Control (`EndPoseCtrl`) |
| Gripper | Integrated parallel gripper |

**CAN Configuration**: Refer to [piper_sdk/asserts/can_config.MD](../../../piper_sdk/asserts/can_config.MD) for CAN interface setup.

### 1.2 Cameras

| Camera | Model | Interface | Resolution | Purpose |
|--------|-------|-----------|------------|---------|
| Fixed Camera | Orbbec Gemini 335L | USB | 640 × 480 | Third-person view of workspace |
| Wrist Camera | Piper Official Binocular | USB | 640 × 480 | Eye-in-hand view |

**Note**: Only RGB images are used; depth data is not required.

### 1.3 Workspace Setup

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TABLE (Top View)                              │
│                                                                      │
│                    ┌───────────────────┐                             │
│                    │  Random Place     │                             │
│                    │  Region           │                             │
│                    │  x: [-0.1, +0.1]  │                             │
│                    │  y: [+0.05, +0.25]│                             │
│                    │       ┌─────┐     │                             │
│                    │       │Plate│ ← Center (0, 0.15, 0)            │
│                    │       │ +Obj│                                   │
│                    │       └─────┘                                   │
│                    └───────────────────┘                             │
│                                                                      │
│      [Fixed Camera]                            [Piper Arm Base]      │
│           📷                                        🦾               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Software Dependencies

### 2.1 Python Environment

```bash
# Required packages
pip install piper_sdk>=0.1.9
pip install python-can>=4.3.1
pip install opencv-python
pip install numpy
pip install scipy
pip install keyboard  # For keyboard control
```

### 2.2 Piper SDK Reference Files

| File | Purpose |
|------|---------|
| [piper_ctrl_end_pose.py](../../../piper_sdk/piper_sdk/demo/V2/piper_ctrl_end_pose.py) | End-effector pose control |
| [piper_read_tcp_pose.py](../../../piper_sdk/piper_sdk/demo/V2/piper_read_tcp_pose.py) | Read TCP (Tool Center Point) pose |
| [piper_read_gripper_status.py](../../../piper_sdk/piper_sdk/demo/V2/piper_read_gripper_status.py) | Read gripper state |
| [piper_ctrl_go_zero.py](../../../piper_sdk/piper_sdk/demo/V2/piper_ctrl_go_zero.py) | Return to home/zero position |
| [piper_ctrl_gripper.py](../../../piper_sdk/piper_sdk/demo/V2/piper_ctrl_gripper.py) | Gripper open/close control |

---

## 3. Coordinate System & Workspace

### 3.1 Key Positions (in Robot Base Frame)

| Position | X (m) | Y (m) | Z (m) | Description |
|----------|-------|-------|-------|-------------|
| **Plate Center** | 0.0 | 0.15 | 0.0 | Fixed pick location (object initial position) |
| **Grasp Height** | - | - | 0.15 | Height for grasping objects |
| **Hover Height** | - | - | 0.25 | Safe height for horizontal movement |
| **Home Position** | TBD | TBD | TBD | Read from `piper_ctrl_go_zero.py` at startup |

### 3.2 Random Place Region (Task B)

The random placement position is sampled uniformly:

```python
place_x = plate_center_x + random.uniform(-0.1, +0.1)  # Range: [-0.1, +0.1]
place_y = plate_center_y + random.uniform(-0.1, +0.1)  # Range: [+0.05, +0.25]
place_z = table_height  # Same as plate center z
```

**Note**: Random positions have **no visual markers** on the table. The FSM uses pre-specified coordinates, but the learned policy must generalize without visual cues.

### 3.3 Orientation Convention

| SDK Format | Policy Format |
|------------|---------------|
| Euler XYZ (RX, RY, RZ) in 0.001° units | Quaternion (qw, qx, qy, qz) |

**Conversion**: Use `scipy.spatial.transform.Rotation` to convert between formats.

```python
from scipy.spatial.transform import Rotation as R

def euler_to_quat(rx_deg, ry_deg, rz_deg):
    """Convert Euler XYZ (degrees) to quaternion (w, x, y, z)."""
    rot = R.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # (w, x, y, z)

def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to Euler XYZ (degrees)."""
    rot = R.from_quat([qx, qy, qz, qw])  # scipy uses [x, y, z, w]
    return rot.as_euler('xyz', degrees=True)  # [rx, ry, rz]
```

---

## 4. Data Format Specification

### 4.1 Observation Space

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `fixed_camera_rgb` | (H, W, 3) | uint8 | RGB image from fixed Orbbec camera |
| `wrist_camera_rgb` | (H, W, 3) | uint8 | RGB image from wrist camera |
| `ee_pose` | (7,) | float32 | End-effector pose [x, y, z, qw, qx, qy, qz] |
| `gripper_state` | (1,) | float32 | Gripper position (0=closed, 1=open) |

### 4.2 Action Space

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `target_ee_pose` | (7,) | float32 | Target end-effector pose [x, y, z, qw, qx, qy, qz] |
| `gripper_action` | (1,) | float32 | Gripper command (0=close, 1=open) |

### 4.3 Episode Data Structure (NPZ Format)

```python
{
    # Per-timestep data (T = episode length)
    "fixed_images": (T, 480, 640, 3),   # Fixed camera RGB (uint8)
    "wrist_images": (T, 480, 640, 3),   # Wrist camera RGB (uint8)
    "ee_pose": (T, 7),                   # EE pose [x, y, z, qw, qx, qy, qz]
    "gripper_state": (T,),               # Gripper position
    "action": (T, 8),                    # Target [ee_pose(7), gripper(1)]
    "fsm_state": (T,),                   # FSM state index (int)
    "timestamp": (T,),                   # Unix timestamp (float)
    
    # Episode metadata
    "place_pose": (7,),                  # Target place position (random)
    "goal_pose": (7,),                   # Plate center position (fixed)
    "success": bool,                     # Episode success flag
    "episode_id": int,                   # Episode index
}
```

### 4.4 Image Storage

- **Format**: PNG (lossless compression)
- **Directory Structure**:
  ```
  data/
  └── piper_pick_place/
      ├── episode_0000/
      │   ├── fixed_cam/
      │   │   ├── 000000.png
      │   │   ├── 000001.png
      │   │   └── ...
      │   ├── wrist_cam/
      │   │   ├── 000000.png
      │   │   └── ...
      │   └── episode_data.npz
      ├── episode_0001/
      └── ...
  ```

---

## 5. Script 0: System Test

### 5.1 Overview

`0_system_test.py` 是首次运行前的**系统测试与校准工具**，用于验证所有硬件和软件组件是否正常工作。

**功能列表**：
1. CAN 接口连接测试
2. 机械臂通信与使能
3. TCP 位姿 & 关节角度读取
4. 相机连接测试与预览
5. 夹爪开合测试
6. 运动控制测试 (Home 位置)
7. 多点位运动测试
8. 全自动测试流程

### 5.2 键盘控制

| 按键 | 功能 |
|------|------|
| `1` | 测试 CAN 接口 |
| `2` | 连接并使能机械臂 |
| `3` | 读取当前位姿 (TCP + 关节) |
| `4` | 测试相机连接 |
| `5` | 显示相机预览 (按 Q 退出预览) |
| `6` | 测试夹爪 (开→闭→开) |
| `7` | 移动到 Home 位置 |
| `8` | 多点位运动测试 |
| `9` | 运行全部测试 |
| `R` | 重置/重新使能 |
| `D` | 失能机械臂 |
| `Space` | **紧急停止** |
| `Q` | 退出程序 |

### 5.3 Usage

```bash
# 基本使用 (默认配置)
python 0_system_test.py

# 指定 CAN 接口
python 0_system_test.py --can can0

# 指定相机 ID
python 0_system_test.py --front-cam 0 --wrist-cam 2

# 设置运动速度 (测试时建议用低速)
python 0_system_test.py --speed 15

# 完整示例
python 0_system_test.py --can can0 --front-cam 0 --wrist-cam 2 --speed 20
```

### 5.4 测试流程建议

1. **首先运行测试 1 (CAN 接口)**：确保 CAN 硬件连接正常
2. **运行测试 2 (连接机械臂)**：验证 SDK 能正常通信
3. **运行测试 3 (读取位姿)**：确认位姿数据格式正确
4. **运行测试 4 (相机)**：检查所有相机可用
5. **运行测试 6 (夹爪)**：确认夹爪响应正常
6. **运行测试 7 (Home 位置)**：⚠️ **确保周围安全后再运行**

### 5.5 紧急停止

按下 **空格键** 会立即：
- 停止当前运动
- 失能机械臂
- 进入紧急停止状态

要恢复，请按 **R** 键重置。

---

## 6. Script 1: PS5 Teleoperation

### 6.1 Overview

`1_teleop_ps5_controller.py` enables manual teleoperation of the Piper arm using a PS5 DualSense controller. This is the primary method for collecting demonstration data.

**Features**:
- Real-time end-effector position/orientation control via joysticks
- Gripper control via triggers
- Gyro mode for intuitive orientation control
- Built-in data recording with episode management
- Force estimation and display

### 6.2 PS5 Controller Mapping

| Control           | Action                                              |
|-------------------|-----------------------------------------------------|
| Left Stick        | Move EE in X/Y axis                                 |
| Right Stick       | Move EE in Z axis / Rotate around Z (yaw)           |
| L2/R2 Triggers    | Close/Open gripper                                  |
| L1/R1 Bumpers     | Decrease/Increase motion speed                      |
| D-pad             | Pitch/Roll end-effector                             |
| Cross (✕)         | Go to home position                                 |
| Circle (⭕) HOLD  | Gyro mode: controller tilt controls EE orientation  |
| Triangle (△)      | Toggle gripper (open/close)                         |
| Square (▢)        | Toggle recording / Print pose                       |
| Share             | Emergency stop / Resume                             |
| Options           | Re-enable arm                                       |
| PS Button         | Quit program                                        |

### 6.3 Usage

```bash
# Basic teleoperation with data recording (default)
python 1_teleop_ps5_controller.py

# Specify CAN interface and speed
python 1_teleop_ps5_controller.py --can_interface can0 --speed 30

# Custom output directory
python 1_teleop_ps5_controller.py --out_dir data/teleop_data

# With camera display
python 1_teleop_ps5_controller.py --show_camera
```

---

## 7. Script 2: Data Visualization

### 7.1 Overview

`2_visualize_collected_data.py` creates comprehensive visualization videos of collected data, showing:
- Side-by-side camera views (front + wrist camera)
- XYZ trajectory curves for observation and action
- FSM state indicator (for scripted data) or teleop mode indicator

### 7.2 Usage

```bash
# Visualize a specific episode (tar.gz archive)
python 2_visualize_collected_data.py --episode data/teleop_data/episode_0000.tar.gz

# Visualize a specific episode (directory format)
python 2_visualize_collected_data.py --episode data/piper_pick_place/episode_0000

# Custom output path and fps
python 2_visualize_collected_data.py --episode data/teleop_data/episode_0000.tar.gz --output viz.mp4 --fps 15
```

---

## 8. Script 3: Time Reversal

### 8.1 Overview

`3_make_forward_data.py` creates **forward training data** by time-reversing teleop trajectories. This is the core of the **Rev2Fwd** method:

- **Input**: Task B trajectories (e.g., place → pick) from Script 1
- **Output**: Task A trajectories (e.g., pick → place) for policy training

### 8.2 Core Algorithm

```
Original trajectory (Task B):
  t=0      t=1      t=2      ...      t=T-1
  pose[0] → pose[1] → pose[2] → ... → pose[T-1]
  
Reversed trajectory (Task A):
  t=0          t=1          t=2          ...      t=T-1
  pose[T-1] → pose[T-2] → pose[T-1] → ... → pose[0]
```

For **relative delta actions**, the reversal process:

1. **Reverse all sequences** in time: `ee_pose`, `gripper_state`, images, etc.
2. **Recompute relative actions** from the reversed `ee_pose`:
   ```python
   # Position delta
   action_rev[t][:3] = ee_pose_rev[t+1][:3] - ee_pose_rev[t][:3]
   
   # Quaternion delta: rotation from quat[t] to quat[t+1]
   action_rev[t][3:7] = quat_inverse(quat[t]) * quat[t+1]
   
   # Gripper target: the gripper state we're moving TO
   action_rev[t][7] = gripper_state_rev[t+1]
   ```
3. **Verify** that integrating actions recovers the trajectory (error should be ~0)

### 8.3 Usage

```bash
# Basic usage: reverse all episodes
python 3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A

# Process only successful episodes
python 3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A \
    --success_only

# Verbose mode with detailed verification
python 3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A \
    --verbose

# Skip verification for faster processing
python 3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A \
    --no_verify
```

### 8.4 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | (required) | Input directory with teleop episodes (tar.gz) |
| `--output`, `-o` | (required) | Output directory for reversed episodes |
| `--success_only` | False | Only process episodes marked as successful |
| `--verbose`, `-v` | False | Print detailed debug info for each episode |
| `--verify` | True | Verify each reversal is correct |
| `--no_verify` | False | Disable verification (faster) |

### 8.5 Output Format

The output maintains the **same format** as the input:
- `episode_*.tar.gz` archives with identical structure
- Updated `metadata.json` with:
  - `collection_type`: `"reversed_teleop"`
  - `reversed_from`: source directory path
  - `reversal_timestamp`: when reversal was performed

### 8.6 Verification

The script automatically verifies each reversal:

1. **Position check**: Original start == Reversed end (and vice versa)
2. **Action integration**: Summing delta actions should reconstruct the trajectory
3. **Gripper check**: Gripper states correctly reversed

Expected integration error: **< 1e-6 meters** (numerical precision)

---

## 9. Script 4: Train Diffusion Policy

### 9.1 Overview

`4_train_diffusion.py` trains a vision-based **Diffusion Policy** using LeRobot on the teleoperation data collected by Script 1 (or time-reversed data from Script 3).

**Key Features**:
- Uses **relative delta actions** (position + quaternion + gripper)
- Supports dual cameras (fixed + wrist) with automatic resolution alignment
- LeRobot v3.0 dataset format with video encoding
- Multi-GPU training support via `torchrun`
- WandB logging integration

### 9.2 Data Requirements

The script expects data in the format produced by Script 1:

```
data/pick_place_piper/
├── episode_0000.tar.gz
├── episode_0001.tar.gz
├── ...
└── metadata.json
```

Each `episode_*.tar.gz` contains:
```
episode_XXXX/
├── episode_data.npz    # Numeric data (ee_pose, action, gripper_state, etc.)
├── fixed_cam/          # Front camera images (000000.png, ...)
└── wrist_cam/          # Wrist camera images (000000.png, ...)
```

### 9.3 Action Format

The script uses **8D relative delta actions**:

| Index | Field | Description |
|-------|-------|-------------|
| 0-2 | `delta_xyz` | Position change in meters |
| 3-6 | `delta_quat` | Quaternion change (qw, qx, qy, qz) |
| 7 | `gripper` | Target gripper state [0-1] |

### 9.4 Observation Format

| Feature | Shape | Description |
|---------|-------|-------------|
| `observation.image` | (3, H, W) | Fixed camera RGB |
| `observation.wrist_image` | (3, H, W) | Wrist camera RGB (optional) |
| `observation.state` | (7,) or (8,) | EE pose (+ gripper if `--include_gripper`) |

**Note**: If cameras have different resolutions, all images are resized to a common size (default: 240×320) during conversion.

### 9.5 Usage Examples

```bash
# Basic training (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --batch_size 64 --steps 50000 --wandb

# With gripper state in observation (state_dim=8)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --include_gripper \
    --batch_size 64 --steps 50000 --wandb

# Custom image size (both cameras resized)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --image_size 240 320 \
    --batch_size 64 --steps 50000 --wandb

# Data conversion only (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --convert_only

# Multi-GPU training (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --batch_size 32 --steps 100000 --wandb

# Resume training from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_teleop \
    --resume --steps 100000

# Overfit mode (1 episode for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/4_train_diffusion.py \
    --dataset data/pick_place_piper \
    --out runs/diffusion_piper_overfit \
    --overfit --steps 1000
```

### 9.6 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `data/pick_place_piper` | Path to episode tar.gz files |
| `--out` | `runs/diffusion_piper_teleop` | Output directory for checkpoints |
| `--steps` | `100000` | Number of training steps |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--image_size` | `240 320` | Target image size (H, W) |
| `--include_gripper` | `False` | Include gripper state in observation |
| `--num_episodes` | `-1` | Number of episodes to use (-1 = all) |
| `--convert_only` | `False` | Only convert data, don't train |
| `--skip_convert` | `False` | Skip conversion (use existing dataset) |
| `--force_convert` | `False` | Force re-conversion |
| `--resume` | `False` | Resume from checkpoint |
| `--overfit` | `False` | Overfit mode (1 episode) |
| `--wandb` | `False` | Enable WandB logging |
| `--wandb_project` | `piper-diffusion-teleop` | WandB project name |

### 9.7 Diffusion Policy Architecture Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n_obs_steps` | `2` | Number of observation steps |
| `--horizon` | `16` | Diffusion horizon (action sequence length) |
| `--n_action_steps` | `8` | Number of action steps to execute |
| `--vision_backbone` | `resnet18` | Vision backbone architecture |
| `--crop_shape` | `128 128` | Image crop shape |
| `--num_train_timesteps` | `100` | Diffusion timesteps |

### 9.8 Output Structure

```
runs/diffusion_piper_teleop/
├── lerobot_dataset/           # Converted LeRobot dataset
│   ├── data/
│   ├── meta/
│   └── videos/
├── checkpoints/               # Training checkpoints
│   └── checkpoints/
│       ├── 020000/
│       │   ├── model.safetensors
│       │   └── pretrained_model/
│       ├── 040000/
│       └── last/              # Latest checkpoint
└── .conversion_meta.json      # Metadata for distributed training
```

### 9.9 Training Tips

1. **GPU Memory**: With ResNet18 backbone, batch_size=64 requires ~20GB VRAM
2. **Training Time**: 50,000 steps ≈ 2-3 hours on RTX 4090
3. **Checkpoints**: Saved every 20,000 steps by default (use `--save_freq` to change)
4. **Validation Split**: Use `--val_split 0.1` for 10% validation set
5. **XYZ Visualization**: Use `--enable_xyz_viz` to generate trajectory plots during training

---

## 10. Script 5: Model Evaluation

### 10.1 Overview

`5_eval_diffusion_piper.py` evaluates a trained diffusion policy on the real Piper arm:
- **Task A**: Pick from **random table position** → Place at **plate center**

### 10.2 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Inference Loop (30 Hz)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│   │ Capture Images  │───▶│ Preprocess &    │───▶│ Diffusion      │  │
│   │ + Read EE Pose  │    │ Build Obs Dict  │    │ Policy Model   │  │
│   └─────────────────┘    └─────────────────┘    └────────────────┘  │
│                                                          │           │
│                                                          ▼           │
│   ┌─────────────────┐    ┌─────────────────┐    ┌────────────────┐  │
│   │ Execute Action  │◀───│ Post-process    │◀───│ Predict Action │  │
│   │ on Piper Arm    │    │ (clip, smooth)  │    │ Chunk (8 steps)│  │
│   └─────────────────┘    └─────────────────┘    └────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.3 Action Chunking

The diffusion policy predicts **8-step action chunks**. Execution strategy:

```python
# Inference every N steps (default N=8)
if step % n_action_steps == 0:
    action_chunk = policy.predict(observation)  # Shape: (8, 8)

# Execute current action from chunk
current_action = action_chunk[step % n_action_steps]
send_to_robot(current_action)
```

### 10.4 Success Criteria

An episode is considered **successful** if:
1. The gripper closes on the object (object detected in grasp)
2. The object is placed within **5cm** of the plate center
3. No emergency stop was triggered
4. Episode completed within **max_steps** (default: 500)

### 10.5 Command Line Interface

```bash
# Basic evaluation
python 5_eval_diffusion_piper.py \
    --checkpoint runs/piper_diffusion/checkpoints/last/pretrained_model \
    --num_episodes 10

# Full options
python 5_eval_diffusion_piper.py \
    --checkpoint runs/piper_diffusion/checkpoints/last/pretrained_model \
    --can_interface can0 \
    --num_episodes 20 \
    --max_steps 500 \
    --n_action_steps 8 \
    --control_freq 30 \
    --out_dir runs/piper_diffusion/eval_videos \
    --record_video \
    --seed 42
```

### 10.6 Output

```
runs/piper_diffusion/eval_videos/
├── episode_00_success.mp4
├── episode_01_fail.mp4
├── ...
└── eval_results.json
```

**eval_results.json**:
```json
{
    "num_episodes": 20,
    "success_rate": 0.75,
    "avg_steps": 342,
    "episodes": [
        {"id": 0, "success": true, "steps": 312, "final_dist": 0.023},
        {"id": 1, "success": false, "steps": 500, "final_dist": 0.156}
    ]
}
```

---

## 11. Script 6: FSM Data Collection

### 11.1 Overview

`6_collect_data_piper.py` collects **Task B** trajectories using a finite state machine (FSM) expert:
- **Task B**: Pick from **plate center** → Place at **random table position**

The collected trajectories will later be **time-reversed** to generate Task A training data.

### 11.2 FSM State Diagram

```
IDLE → GO_TO_HOME → HOVER_PLATE → LOWER_GRASP → CLOSE_GRIP → LIFT_OBJECT 
     → HOVER_PLACE → LOWER_PLACE → OPEN_GRIP → LIFT_RETREAT → RETURN_HOME → DONE → IDLE
```

### 11.3 Usage

```bash
# Basic usage with default cameras
python 6_collect_data_piper.py --num_episodes 50 --out_dir data/piper_pick_place

# Specify cameras by device name
python 6_collect_data_piper.py -f Orbbec_Gemini_335L -w Dabai_DC1

# Custom workspace parameters
python 6_collect_data_piper.py \
    --can_interface can0 \
    --num_episodes 100 \
    --out_dir data/piper_pick_place \
    --plate_x 0.25 --plate_y 0.0 \
    --speed 20
```

### 11.4 Keyboard Controls

| Input         | Action                              |
|---------------|-------------------------------------|
| ENTER / start | Start new episode                   |
| e / esc       | Emergency stop (freeze arm)         |
| q / quit      | Quit and save all data              |
| r / reset     | Reset to home position              |
| s / skip      | Skip current episode (discard)      |

---

## 12. Script 7: Train DiT Flow Policy

### 12.1 Overview

`7_train_ditflow.py` trains a vision-based **DiT Flow Policy** (Diffusion Transformer + Flow Matching) using the `lerobot_policy_ditflow` plugin on the teleoperation data collected by Script 1 (or time-reversed data from Script 3).

This script provides the **same functionality as Script 4** (Diffusion Policy) but uses a different underlying architecture:

| Feature | Script 4 (Diffusion) | Script 7 (DiT Flow) |
|---------|----------------------|----------------------|
| Architecture | U-Net (CNN) | DiT (Transformer) |
| Noise Model | DDPM (diffusion) | Flow Matching (ODE) |
| Default crop_shape | 128 × 128 | 84 × 84 |
| Inference | DDPM reverse process | Euler ODE integration |
| Parameters | ~61M | ~53M |
| Config class | `DiffusionConfig` | `DiTFlowConfig` |

**Key Features** (same as Script 4):
- Uses **relative delta actions** (position + quaternion + gripper)
- Supports dual cameras (fixed + wrist) with automatic resolution alignment
- LeRobot v3.0 dataset format with video encoding
- Multi-GPU training via `torchrun`
- WandB logging integration
- XYZ trajectory visualization during training

### 12.2 Dependencies

DiT Flow requires two additional packages (not included in the base installation):

```bash
# 1. LeRobot (danielsanjosepro's fork with plugin support)
git clone https://github.com/danielsanjosepro/lerobot.git
pip install -e ./lerobot

# 2. DiT Flow Policy plugin
git clone https://github.com/danielsanjosepro/lerobot_policy_ditflow.git
pip install -e ./lerobot_policy_ditflow
```

Verify installation:
```bash
python -c "from lerobot.utils.import_utils import register_third_party_plugins; \
           register_third_party_plugins(); \
           from lerobot.configs.policies import PreTrainedConfig; \
           print('ditflow' in PreTrainedConfig.get_known_choices())"
# Should print: True
```

### 12.3 Usage Examples

```bash
# Basic training (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --batch_size 64 --steps 50000 --wandb

# With gripper state in observation (state_dim=8)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --include_gripper \
    --batch_size 64 --steps 50000 --wandb

# With XYZ visualization during training
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --include_gripper --enable_xyz_viz --viz_save_freq 5000 --wandb

# Custom DiT architecture (smaller model)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --hidden_dim 256 --num_blocks 4 --num_heads 8 --dim_feedforward 2048 \
    --batch_size 64 --steps 50000 --wandb

# Beta noise scheduling (from Pi0 paper)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --training_noise_sampling beta \
    --batch_size 64 --steps 50000 --wandb

# Multi-GPU training (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --batch_size 32 --steps 100000 --wandb

# Data conversion only (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --convert_only

# Overfit mode (1 episode for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_overfit \
    --overfit --steps 1000 --include_gripper

# Resume training from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper \
    --out runs/ditflow_piper_teleop \
    --resume --steps 200000
```

### 12.4 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `data/pick_place_piper` | Path to episode tar.gz files |
| `--out` | `runs/ditflow_piper_teleop` | Output directory for checkpoints |
| `--steps` | `100000` | Number of training steps |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `1e-4` | Learning rate |
| `--image_size` | `240 320` | Target image size (H, W) |
| `--include_gripper` | `False` | Include gripper state in observation |
| `--num_episodes` | `-1` | Number of episodes to use (-1 = all) |
| `--convert_only` | `False` | Only convert data, don't train |
| `--skip_convert` | `False` | Skip conversion (use existing dataset) |
| `--force_convert` | `False` | Force re-conversion |
| `--resume` | `False` | Resume from checkpoint |
| `--overfit` | `False` | Overfit mode (1 episode) |
| `--wandb` | `False` | Enable WandB logging |
| `--wandb_project` | `piper-ditflow-teleop` | WandB project name |
| `--enable_xyz_viz` | `False` | Enable XYZ trajectory plots during training |
| `--viz_save_freq` | `5000` | Steps between visualization saves |
| `--clip_sample` | `True` | Clip denoised actions to a fixed range |
| `--clip_sample_range` | `1.0` | Clipping range |

### 12.5 DiT Flow Architecture Options

| Option | Default | Description |
|--------|---------|-------------|
| `--n_obs_steps` | `2` | Number of observation steps |
| `--horizon` | `16` | Action sequence length |
| `--n_action_steps` | `8` | Number of action steps to execute |
| `--vision_backbone` | `resnet18` | Vision backbone architecture |
| `--crop_shape` | `84 84` | Image crop shape (DiT default) |
| `--hidden_dim` | `512` | Transformer hidden dimension |
| `--num_blocks` | `6` | Number of DiT transformer blocks |
| `--num_heads` | `16` | Number of attention heads |
| `--dim_feedforward` | `4096` | Feedforward dimension |
| `--dropout` | `0.1` | Dropout rate |
| `--num_inference_steps` | `100` | Euler steps for ODE sampling |
| `--training_noise_sampling` | `uniform` | Noise schedule: `uniform` or `beta` |

### 12.6 Output Structure

Same as Script 4:

```
runs/ditflow_piper_teleop/
├── lerobot_dataset/           # Converted LeRobot dataset
│   ├── data/
│   ├── meta/
│   └── videos/
├── checkpoints/               # Training checkpoints
│   └── checkpoints/
│       ├── 020000/
│       │   ├── model.safetensors
│       │   └── pretrained_model/
│       ├── 040000/
│       └── last/              # Latest checkpoint
├── xyz_visualizations/        # (if --enable_xyz_viz)
│   ├── xyz_viz_step_005000.png
│   └── xyz_viz_step_010000.png
└── .conversion_meta.json      # Metadata for distributed training
```

### 12.7 Training Tips

1. **GPU Memory**: With ResNet18 backbone + DiT, batch_size=64 requires ~18GB VRAM
2. **DiT vs Diffusion**: DiT Flow tends to converge faster but may need tuning of `num_inference_steps`
3. **Noise Scheduling**: `beta` scheduling (from Pi0) may improve sample quality at the cost of slightly slower convergence
4. **Crop Shape**: Default (84, 84) is smaller than Diffusion's (128, 128); adjust if needed

---

## 13. Safety Guidelines

### 12.1 Pre-Flight Checklist

- [ ] CAN interface is properly configured and connected
- [ ] Arm is powered on and in slave mode
- [ ] Workspace is clear of obstacles
- [ ] Emergency stop button is accessible
- [ ] Cameras are connected and recognized
- [ ] Object is placed on the plate

### 12.2 Emergency Stop Procedures

1. **Software E-Stop**: Press `ESC` key → Arm freezes in place
2. **Hardware E-Stop**: Press physical emergency stop button
3. **Recovery**: 
   ```bash
   # After emergency stop, run:
   python -c "from piper_sdk import *; p=C_PiperInterface_V2(); p.ConnectPort(); p.DisablePiper()"
   # Then re-enable:
   python -c "from piper_sdk import *; p=C_PiperInterface_V2(); p.ConnectPort(); p.EnablePiper()"
   ```

### 12.3 Workspace Limits (Soft Limits)

```python
# Enforce these limits in software to prevent collisions
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.3,
    "y_min": 0.0,   "y_max": 0.4,
    "z_min": 0.05,  "z_max": 0.4,
}
```

---

## 14. Debugging Guide & Known Issues

### 14.1 Critical Parameters to Calibrate

| Parameter | Location | Issue | How to Calibrate |
|-----------|----------|-------|------------------|
| `DEFAULT_ORIENTATION_EULER` | Both scripts | `(180, 0, 0)` may not match your gripper pointing-down pose | Read actual pose with `piper.GetArmEndPoseMsgs()` when gripper is pointing down |
| `PLATE_CENTER` | Both scripts | `(0.0, 0.15, 0.0)` is placeholder | Manually jog arm to plate center, read position |
| `GRASP_HEIGHT` | Both scripts | `0.15m` may be too high/low | Adjust based on object height |
| `GRIPPER_OPEN_POS` | Both scripts | `70000` (70mm) may not match your gripper | Check SDK docs or test with `piper.GetArmGripperMsgs()` |

### 14.2 Known Issues & Bugs

#### Script 6: FSM Data Collection

1. **Keyboard module requires root on Linux**
   ```bash
   # Solution: run with sudo or use alternative input method
   sudo python 6_collect_data_piper.py ...
   ```

2. **Camera ID may change on reconnect**
   - Use `v4l2-ctl --list-devices` to find correct IDs
   - Or use camera serial number for consistent identification

3. **CAN interface name on Windows**
   - Default `can0` is Linux-specific
   - On Windows with USB-CAN adapter, check piper_sdk docs for correct interface name

4. **Home position orientation may differ from `DEFAULT_ORIENTATION_EULER`**
   - The `go_to_home()` uses joint control, then switches to end-pose control
   - The actual home orientation may not be `(180, 0, 0)`

5. **NPZ does not save images**
   - Images are saved as PNG files, NPZ only contains metadata
   - Need to load both for training data conversion

#### Script 5: Model Evaluation

1. **LeRobot version compatibility**
   - `make_pre_post_processors` API may differ between versions
   - Tested with lerobot >= 0.4.0

2. **Action chunk timing**
   - If `n_action_steps > 1`, actions are executed open-loop between inferences
   - May cause drift if control frequency is inconsistent

3. **Success condition is simplistic**
   - Only checks XY distance to goal, not actual object placement
   - No object detection/tracking implemented

4. **Image normalization mismatch**
   - Manual `/ 255.0` normalization in `build_observation()` may conflict with preprocessor
   - Check if preprocessor already handles image normalization

### 14.3 Pre-Run Checklist
# 1. Test CAN connection
python -c "from piper_sdk import *; p=C_PiperInterface_V2('can0'); p.ConnectPort(); print('OK')"

# 2. Test camera IDs
python -c "import cv2; print('cam0:', cv2.VideoCapture(0).isOpened()); print('cam1:', cv2.VideoCapture(1).isOpened())"

# 3. Read current arm pose (to calibrate positions)
python -c "
from piper_sdk import *
import time
p = C_PiperInterface_V2('can0')
p.ConnectPort()
while not p.EnablePiper(): time.sleep(0.01)
pose = p.GetArmEndPoseMsgs().end_pose
print(f'X={pose.X_axis/1e6:.3f}m Y={pose.Y_axis/1e6:.3f}m Z={pose.Z_axis/1e6:.3f}m')
print(f'RX={pose.RX_axis/1e3:.1f}° RY={pose.RY_axis/1e3:.1f}° RZ={pose.RZ_axis/1e3:.1f}°')
"

# 4. Test gripper
python -c "
from piper_sdk import *
import time
p = C_PiperInterface_V2('can0')
p.ConnectPort()
while not p.EnablePiper(): time.sleep(0.01)
p.GripperCtrl(70000, 1000, 0x01, 0)  # Open
time.sleep(1)
p.GripperCtrl(0, 1000, 0x01, 0)  # Close
"
```

### 14.4 Recommended Debugging Steps

1. **First run with arm disabled** (comment out `piper.connect()`)
   - Verify cameras work
   - Verify keyboard input works
   - Check data saving format

2. **Test arm motion manually first**
   ```python
   # Safe test script
   from piper_sdk import C_PiperInterface_V2
   import time
   
   piper = C_PiperInterface_V2('can0')
   piper.ConnectPort()
   while not piper.EnablePiper(): time.sleep(0.01)
   
   # Move slowly to a known safe position
   piper.MotionCtrl_2(0x01, 0x02, 10, 0x00)  # 10% speed
   piper.EndPoseCtrl(0, 150000, 250000, 180000, 0, 0)  # x=0, y=0.15m, z=0.25m
   ```

3. **Verify coordinate convention**
   - Piper SDK uses **0.001mm** for position, **0.001°** for angles
   - Policy uses **meters** and **quaternion (w,x,y,z)**

---

## 15. TODO / Open Questions

- [ ] Determine exact home position by reading from `piper_ctrl_go_zero.py`
- [ ] Calibrate camera intrinsics if needed for future visual servoing
- [ ] Tune gripper force parameters for reliable grasping
- [x] ~~Add support for LeRobot dataset format conversion~~ (Done: Script 4)
- [ ] Add object detection for better success criteria
- [ ] Test on Windows (CAN interface compatibility)
- [ ] Add teleoperation mode for manual data collection fallback

---

## References

- [Rev2Fwd Pipeline Documentation](../scripts_pick_place/PIPELINE_README.md)
- [Piper SDK V2 Demo](../../../piper_sdk/piper_sdk/demo/V2/README.MD)
- [Piper SDK CAN Configuration](../../../piper_sdk/asserts/can_config.MD)
