# Nut Threading Data Collection (FORGE Environment)

本目录包含基于 Isaac Lab FORGE 环境的螺母拧紧（nut threading）任务数据收集脚本。

## 目录结构

| 脚本 | 说明 |
|------|------|
| `1_collect_data_nut_thread.py` | **正向**数据收集：从螺母悬空状态开始，执行拧紧动作 |
| `1_1_collect_data_nut_unthread.py` | **逆向**数据收集（rev2fwd-il）：从拧紧状态开始，执行松螺母动作 |
| `2_inspect_nut_data.py` | 数据可视化：提取帧、生成视频、力传感器曲线、3D 轨迹 |

---

## 环境概述

### FORGE Nut Threading Task

- **Task ID**: `Isaac-Forge-NutThread-Direct-v0`
- **机器人**: Franka Panda (7-DOF 机械臂 + 2-DOF 平行夹爪)
- **任务**: 将螺母拧到螺栓上
- **控制频率**: 15 Hz (120 Hz 仿真 / 8 decimation)
- **力传感器**: 6-DOF F/T 传感器 (force_sensor link)，EMA 平滑 (α=0.25)

### Action Space (7D)

| 维度 | 含义 | 范围 | 说明 |
|------|------|------|------|
| `action[0:3]` | 位置目标 (xyz) | [-1, 1] | 相对于螺栓顶部的位置偏移，乘以 `pos_action_bounds` |
| `action[3:5]` | Roll/Pitch | [-1, 1] | **被强制归零**，即只允许 yaw 旋转 |
| `action[5]` | Yaw 旋转目标 | [-1, 1] | 映射到 [-180°, +90°] 的绝对 yaw 角度 |
| `action[6]` | 夹爪控制 | [-1, 1] | **-1 = 全闭合 (0.0m)，+1 = 全张开 (0.04m)** |

> **重要变更**: `action[6]` 原本是 success prediction 信号（用于 RL 训练），我们已修改
> `forge_env.py` 将其改为夹爪控制。修改位置：
> `isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py` 的 `_apply_action()` 方法。

### Observation Space

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `fingertip_pos_rel_fixed` | 3 | 指尖相对螺栓的位置 |
| `fingertip_quat` | 4 | 指尖四元数 |
| `ee_linvel` | 3 | 末端执行器线速度 |
| `ee_angvel` | 3 | 末端执行器角速度 |
| `ft_force` | 3 | 力传感器读数 (Fx, Fy, Fz) |
| `force_threshold` | 1 | 接触力惩罚阈值 |

---

## 1. 正向数据收集

### 脚本: `1_collect_data_nut_thread.py`

从螺母悬空状态开始，使用力反馈状态机控制策略执行拧紧。

#### 基本用法

```bash
# 单环境收集 100 个 episode（headless 模式）
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 \
    --out data/nut_thread.npz

# 指定图像尺寸
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 100 \
    --image_width 128 --image_height 128 \
    --out data/nut_thread_128.npz
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `Isaac-Forge-NutThread-Direct-v0` | Isaac Lab 任务 ID |
| `--num_envs` | `1` | 并行环境数量（图像采集建议 1） |
| `--num_episodes` | `100` | 收集 episode 数量 |
| `--horizon` | `600` | 每个 episode 最大步数 |
| `--image_width` | `128` | 图像宽度 |
| `--image_height` | `128` | 图像高度 |
| `--rotation_speed` | `0.5` | 拧紧角速度 (rad/s) |
| `--downward_force` | `5.0` | 拧紧时目标下压力 (N) |
| `--seed` | `0` | 随机种子 |
| `--out` | `data/nut_thread.npz` | 输出 NPZ 文件路径 |
| `--disable_fabric` | `0` | 是否禁用 Fabric 后端 (PhysX GPU) |
| `--headless` | - | 无头模式运行（Isaac Lab 参数） |

#### 专家策略状态机

数据采集使用 `NutThreadingExpert` 力反馈状态机，包含以下阶段：

```
APPROACH → SEARCH → ENGAGE → THREAD → DONE
                                ↓
                      (需要多圈时触发)
                    RELEASE → REPOSITION → REGRASP → THREAD
```

| 阶段 | ID | 说明 | 夹爪 |
|------|----|------|------|
| **APPROACH** | 0 | 向下移动直到检测到接触力 | 闭合 |
| **SEARCH** | 1 | 螺旋搜索找到螺纹对齐点 | 闭合 |
| **ENGAGE** | 2 | 反转-正转旋转 + 下压，捕捉螺纹 | 闭合 |
| **THREAD** | 3 | 持续旋转 + 自适应下压力 | 闭合 |
| **DONE** | 4 | 完成（扭矩过大或超时） | 闭合 |
| **RELEASE** | 5 | 抬起 + 张开夹爪释放螺母 | **张开** |
| **REPOSITION** | 6 | 保持抬起，yaw 旋转回初始位置 | **张开** |
| **REGRASP** | 7 | 下降 + 重新抓取螺母 | 先张开后闭合 |

> **多圈拧紧**: FORGE 的 yaw 范围只有 [-180°, +90°] = 270°，不到一整圈。
> 当 yaw 接近上限（0.9）时，自动触发 RELEASE→REPOSITION→REGRASP 循环，
> 最多支持 15 次重抓（约 15 × 270° = 4050° ≈ 11 圈）。

---

## 2. 逆向数据收集（rev2fwd-il）

### 脚本: `1_1_collect_data_nut_unthread.py`

从螺母已拧紧的状态开始，执行松螺母（逆时针旋转 + 上提）。收集到的轨迹可以通过时间反转得到正向拧紧数据。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_nut/1_1_collect_data_nut_unthread.py \
    --headless --num_episodes 500 \
    --out data/B_nut_unthread.npz
```

**rev2fwd-il 思路**:
1. 初始化为螺母已拧紧在螺栓上（目标状态）
2. 执行简单的"松螺母"策略（逆时针旋转 + 上提）
3. 对采集的轨迹做时间反转，得到正向拧紧 demonstration

优势：松螺母更简单（重力辅助保持接触），且初始状态可以直接设置。

---

## 3. 数据检查与可视化

### 脚本: `2_inspect_nut_data.py`

```bash
# 基本检查（打印统计信息 + 生成帧图片 + 视频）
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz

# 指定 episode，生成力传感器曲线和 3D 轨迹
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz \
    --episode 5 --enable_force_plot --enable_trajectory_plot
```

输出目录: `data/inspect_<dataset_name>_<timestamp>/`

生成内容：
- `frame_N_table.png` — 桌面相机单帧图片
- `frame_N_wrist_cam.png` — 腕部相机单帧图片
- `frame_N_data.json` — 帧元数据 (ee_pose, action, force 等)
- `episode_X_video.mp4` — 多相机拼接视频
- `episode_X_with_force.mp4` — 带力/扭矩曲线 overlay 的视频
- `episode_X_force_plot.png` — 力传感器时序曲线（需 `--enable_force_plot`）
- `episode_X_trajectory.png` — 3D 轨迹图（需 `--enable_trajectory_plot`）

---

## 输出数据格式

所有脚本输出 NPZ 文件，内含 `episodes` 数组（list of dict）。每个 episode dict 包含：

| 字段 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `obs` | (T, obs_dim) | float32 | 策略观测 |
| `state` | (T, state_dim) | float32 | 完整状态观测（privileged） |
| `images` | (T, H, W, 3) | uint8 | 桌面相机 RGB |
| `wrist_wrist_cam` | (T, H, W, 3) | uint8 | 腕部相机 RGB |
| `ee_pose` | (T, 7) | float32 | 末端位姿 [x, y, z, qw, qx, qy, qz] |
| `nut_pose` | (T, 7) | float32 | 螺母位姿 |
| `bolt_pose` | (T, 7) | float32 | 螺栓位姿 |
| `action` | (T, 7) | float32 | 动作 [pos(3), rot(3), gripper(1)] |
| `ft_force` | (T, 3) | float32 | 力传感器 (Fx, Fy, Fz) |
| `ft_force_raw` | (T, 6) | float32 | 原始力/扭矩 (Fx, Fy, Fz, Tx, Ty, Tz) |
| `joint_pos` | (T, 7) | float32 | 机器人关节位置 |
| `phase` | (T,) | int32 | 状态机阶段 ID |
| `episode_length` | scalar | int | Episode 总步数 |
| `success` | scalar | bool | 是否成功 |
| `success_threshold` | scalar | float | 成功判定阈值 |
| `wrist_cam_names` | list | str | 腕部相机名称列表 |

### 读取示例

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

## FORGE 环境相关文件

FORGE 环境代码位于 `isaaclab_tasks/isaaclab_tasks/direct/forge/`：

| 文件 | 说明 |
|------|------|
| `forge_env.py` | 主环境类（继承自 FactoryEnv），action 处理、力传感、奖励 |
| `forge_env_cfg.py` | 环境配置（action bounds、控制器增益、观测噪声等） |
| `forge_tasks_cfg.py` | 任务配置（NutThread、PegInsert、GearMesh） |
| `forge_utils.py` | 工具函数（力/扭矩坐标变换等） |
| `forge_events.py` | 域随机化事件（dead zone 等） |

### 关于夹爪控制的修改

原始 FORGE 环境**硬编码夹爪始终闭合** (`ctrl_target_gripper_dof_pos=0.0`)，
因为 RL 训练中螺母拧紧任务不需要松手。

我们在 `forge_env.py` 的 `_apply_action()` 中做了修改，让 `action[:, 6]` 控制夹爪：

```python
# forge_env.py _apply_action() 末尾
gripper_action = (self.actions[:, 6] + 1.0) / 2.0 * 0.04  # [-1,1] → [0, 0.04]
self.generate_ctrl_signals(
    ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
    ctrl_target_gripper_dof_pos=gripper_action,  # 替代原来的 0.0
)
```

Franka 夹爪 DOF 范围: `[0.0, 0.04]` 米 (0 = 全闭, 0.04 = 全开)。

> **注意**: 此修改会影响 `_get_rewards()` 中的 success prediction reward 计算
> （原来读取 `action[:, 6]` 作为 success prediction）。若仅用于数据采集则无影响；
> 若需要 RL 训练，需同步修改 reward 函数。
