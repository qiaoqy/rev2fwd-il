# Pick-and-Place Simulator Scripts (Simplified Pipeline)

> **Purpose**: 一套精简的 Rev2Fwd 仿真实验脚本, 消除旧版 `scripts_pick_place/` 中的冗余.

---

## 0. 与旧版脚本的对应关系

| 新脚本 | 旧版参考 | 改动要点 |
|--------|---------|---------|
| `1_collect_task_B.py` | `scripts_pick_place/1_collect_data_pick_place.py` | 仅保留 Mode 3 (红色矩形), 删除 legacy/circle 分支; wrist camera 强制开启 |
| `2_reverse_to_task_A.py` | `scripts_pick_place/3_1_make_forward_data.py` | 逻辑不变, 整理参数名 |
| `3_inspect_data.py` | `scripts_pick_place/2_inspect_data.py` | 不变, 直接复用 |
| `4_train.py` | `scripts_pick_place/4_train_diffusion_ddim.py` | **DDIM 内置** (不再需要 wrapper, 见 §1.3 说明); `n_action_steps` 训练时传 16 |
| `5_finetune.py` | `scripts_pick_place/7_finetune_with_rollout.py` | 不变, 整理参数名 |
| `6_eval_cyclic.py` | `scripts_pick_place/9_eval_with_recovery.py` | 仅保留 Mode 3; `n_action_steps=8`; 用于**迭代数据收集** |
| `7_eval_fair.py` | `scripts_pick_place/10_eval_independent.py` | 仅保留 Mode 3; `n_action_steps=8`; 每个 episode 独立硬重置 |
| `8_eval_failure_analysis.py` | `scripts_pick_place/12_eval_failure_analysis.py` + `13_eval_failure_analysis_A.py` | 合并为单脚本, 通过 `--task A/B` 切换; 失败 episode 渲染标注视频 |
| `run_pipeline.sh` | `data/pick_place_isaac_lab_simulation/exp14/run_exp14_iterative.sh` | 统一编排脚本 |

---

## 1. 方法概要

### 1.1 核心思路

Pick-and-Place 可分解为两个互逆子任务:

| 任务 | 描述 | 难度 |
|------|------|------|
| **Task A** | 从红色矩形区域拾取 → 放到绿色目标 (固定) | 难 (初始位置随机) |
| **Task B** | 从绿色目标 (固定) 拾取 → 放到红色矩形区域内 | 易 (初始位置固定) |

1. 用 FSM 专家收集 Task B 数据 (正向收集, 简单)
2. **时间反转** Task B 轨迹 → 自动得到 Task A 训练数据
3. 分别训练 Policy A 和 Policy B
4. 迭代: Cyclic A→B 测试 → 收集成功轨迹 → finetune → 重复

### 1.2 Action 约定

所有数据统一使用 **下一帧 ee_pose** 作为 action:

```
action[t] = [ee_pose[t+1][:7], gripper[t]]    # shape: (8,)
```

- `ee_pose[:7]` = `[x, y, z, qw, qx, qy, qz]` (末端执行器位姿)
- `gripper` = `+1` (张开) / `-1` (闭合)
- 最后一帧: `action[T-1][:7] = ee_pose[T-1]` (保持不动)

### 1.3 Diffusion Policy + DDIM

训练和推理统一使用 **DDIM 去噪器**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `noise_scheduler_type` | `DDIM` | 使用 DDIM 调度器 (旧版默认 DDPM) |
| `num_inference_steps` | `10` | DDIM 去噪步数 (DDPM=100) |
| `num_train_timesteps` | `100` | 训练时的扩散步数 |

#### 旧版 wrapper 架构 vs 新版内置

**旧版** 有两个独立脚本:
- `4_train_diffusion.py` — 主训练脚本, 默认使用 DDPM (100 步去噪)
- `4_train_diffusion_ddim.py` — **wrapper 脚本**, 通过 `importlib` 动态加载
  `4_train_diffusion.py` 模块, 然后 monkey-patch `DiffusionConfig` 类, 将
  `noise_scheduler_type` 强制覆盖为 `"DDIM"`, `num_inference_steps` 覆盖为 `10`.
  本质上 wrapper 不包含任何训练逻辑, 只是一个 DiffusionConfig 的参数注入层.

**新版** (`4_train.py`) 不再需要 wrapper, 因为:
- 我们的 pipeline **统一使用 DDIM**, 不需要在 DDPM/DDIM 之间切换
- 直接在 `4_train.py` 的 `main()` 中设置 `DiffusionConfig` 的默认值即可
- 实现方式: 在创建 `DiffusionConfig` 之前, patch 该类使其默认
  `noise_scheduler_type="DDIM"`, `num_inference_steps=10`
- 仍保留 `--noise_scheduler_type` 和 `--num_inference_steps` CLI 参数,
  但默认值已是 DDIM, 正常使用无需指定

### 1.4 `n_action_steps`: 训练 vs 推理

| 场景 | `n_action_steps` | 说明 |
|------|-----------------|------|
| **训练** (`4_train.py`) | **16** | diffusion horizon=16, 全部作为 action chunk 训练 |
| **推理** (`6_eval_cyclic.py`, `7_eval_fair.py`, `8_eval_failure_analysis.py`) | **8** | 预测 16 步, 仅执行前 8 步后重新推理, 提高响应性 |

注意: `n_action_steps` 是 LeRobot DiffusionPolicy 的配置参数, 会被保存到
checkpoint 中. **训练时写入 16, 推理时通过 `--n_action_steps 8` 覆盖**.
参考 `6_test_alternating.py` 中 `load_diffusion_policy()` 的 `n_action_steps`
参数, 它会在加载后覆盖 policy config 的值.

### 1.5 Wrist Camera (手腕相机)

**所有脚本强制开启 wrist camera**, 不提供关闭选项:
- 数据收集: `wrist_images` 字段必须存在
- 训练: 自动检测到 `wrist_images` 并作为第二个图像输入
- 推理: 必须提供 wrist camera 图像

参考: `4_train_diffusion.py` 中的 `convert_npz_to_lerobot_format()` 会自动
检测 `has_wrist = "wrist_images" in episode`, 并据此创建 LeRobot 数据集.

### 1.6 仿真时序参数

| 参数 | 值 |
|------|------|
| Physics frequency | 90 Hz (`sim.dt = 1/90`) |
| Decimation | 3 |
| Control frequency | 30 Hz (90 / 3) |
| Camera render frequency | 30 Hz |

---

## 2. 环境配置 (红色矩形区域 — Mode 3)

**所有脚本统一使用以下区域参数**, 不再支持 legacy 或 circle 模式:

| 参数 | 值 | 说明 |
|------|------|------|
| `goal_xy` | `(0.50, -0.20)` | 绿色目标 (固定, 圆形标记) |
| `red_region_center_xy` | `(0.50, 0.20)` | 红色矩形区域中心 |
| `red_region_size_xy` | `(0.30, 0.30)` | 矩形大小 (30cm × 30cm) |
| `red_marker_shape` | `rectangle` | 矩形标记 |
| `fix_red_marker_pose` | `1` | 标记始终可见 (不随机) |
| `distance_threshold` | `0.03` | Task A 成功判定距离 (3cm) |

### 2.1 采样逻辑

Task B 放置目标在红色矩形区域内随机采样, 并内缩 `cube_half_size=0.02m` 以确保
方块整体在矩形内:

```
有效采样范围 X: [0.50 - 0.15 + 0.02, 0.50 + 0.15 - 0.02] = [0.37, 0.63]
有效采样范围 Y: [0.20 - 0.15 + 0.02, 0.20 + 0.15 - 0.02] = [0.07, 0.33]
```

参考实现: `scripts_pick_place/1_collect_data_pick_place.py`, L837–L862.

### 2.2 成功判定

**Task A 成功条件** (全部满足):
1. `obj_z < 0.15m` — 物体已放回桌面
2. `gripper > 0.5` — 夹爪已张开 (物体已释放)
3. `dist(obj_xy, goal_xy) < distance_threshold` — 物体在绿色目标附近

**Task B 成功条件** (全部满足):
1. `obj_z < 0.15m`
2. `gripper > 0.5`
3. `obj_xy` 在红色矩形区域内 (内缩 `cube_half_size`)

参考实现: `scripts_pick_place/6_test_alternating.py`, `check_task_A_success` / `check_task_B_success`.

### 2.3 视觉标记

| 标记 | 形状 | 颜色 | 位置 |
|------|------|------|------|
| Goal | 圆平面 (radius=0.05m) | 绿色 | `(0.50, -0.20)`, 固定 |
| Region | 矩形薄片 | 红色 | `(0.50, 0.20)`, 固定 |

参考实现: `scripts_pick_place/1_collect_data_pick_place.py`, L308–L333 (红色矩形创建).

---

## 3. NPZ 数据格式

每个 `.npz` 文件包含 `episodes` 数组, 每个 episode 是一个 dict:

```python
{
    "obs":          np.float32,  # (T, 36)  — 策略特征 (未在 diffusion 训练中使用)
    "images":       np.uint8,    # (T, 128, 128, 3) — 桌面相机 RGB
    "wrist_images": np.uint8,    # (T, 128, 128, 3) — 手腕相机 RGB (必须)
    "ee_pose":      np.float32,  # (T, 7)  — [x, y, z, qw, qx, qy, qz]
    "obj_pose":     np.float32,  # (T, 7)  — 物体位姿
    "action":       np.float32,  # (T, 8)  — [ee_pose[t+1], gripper[t]]
    "gripper":      np.float32,  # (T,)    — +1=open, -1=close
    "fsm_state":    np.int32,    # (T,)    — FSM 状态 (仅收集时记录)
    "place_pose":   np.float32,  # (7,)    — 放置目标位姿
    "goal_pose":    np.float32,  # (7,)    — 绿色目标位姿
    "success":      bool,        # 是否成功
}
```

---

## 4. 脚本详细说明

### 4.1 `1_collect_task_B.py` — 正向收集 Task B 数据

**参考**: `scripts_pick_place/1_collect_data_pick_place.py` (仅保留 Mode 3 分支)

**功能**: 使用 FSM 专家 (`PickPlaceExpertB`) 收集 Task B 轨迹.
专家从绿色目标拾取方块, 放到红色矩形区域内的随机位置.

**CLI 参数**:

```
--out              输出 NPZ 路径 (必须)
--num_episodes     收集的 episode 数 (默认: 100)
--horizon          每个 episode 最大步数 (默认: 2000)
--image_width      相机图像宽度 (默认: 128)
--image_height     相机图像高度 (默认: 128)
--settle_steps     稳定帧数 (默认: 30)
--goal_xy          绿色目标 XY (默认: 0.5 -0.2)
--red_region_center_xy  红色矩形中心 (默认: 0.5 0.2)
--red_region_size_xy    红色矩形大小 (默认: 0.3 0.3)
--headless         无头模式
--seed             随机种子 (默认: 42)
```

**关键逻辑**:
1. 创建环境, 放置绿色目标和红色矩形标记
2. 每个 episode: 方块从绿色目标出发 → FSM 专家执行 pick-place → 放到红色区域内随机位置
3. Action 使用下一帧 ee_pose 约定: `action[t][:7] = ee_pose[t+1]`
4. 保存所有成功 episode 到 NPZ

**参考核心函数**: `rollout_expert_B_with_next_frame_actions()` in `1_collect_data_pick_place.py`

**输出**: `task_B_100.npz`

### 4.2 `2_reverse_to_task_A.py` — 时间反转得到 Task A 数据

**参考**: `scripts_pick_place/3_1_make_forward_data.py` (逻辑完全相同)

**功能**: 将 Task B 轨迹在时间上反转, 得到 Task A 训练数据.

**CLI 参数**:

```
--input            输入 Task B NPZ 路径 (必须)
--out              输出 Task A NPZ 路径 (必须)
--success_only     仅保留成功 episode (默认: 1)
--verify           验证 action[t][:7] == ee_pose[t+1] (默认: False)
```

**反转逻辑** (参考 `3_1_make_forward_data.py`, `reverse_episode()`):

```python
# 1. 所有序列在时间轴上翻转
obs_rev    = obs[::-1]
ee_rev     = ee_pose[::-1]
gripper_rev = gripper[::-1]

# 2. 重新计算 action (下一帧 ee_pose 约定)
action_rev[t][:7] = ee_rev[t+1]    # t = 0..T-2
action_rev[T-1][:7] = ee_rev[T-1]  # 最后一帧保持
action_rev[t][7] = gripper_rev[t]

# 3. 丢弃最后一帧 (无有效 next-frame target)
# 输出长度 = T - 1
```

**验证**: `--verify` 会检查 `max|action[t][:7] - ee_pose[t+1]| < 1e-6`.

**输出**: `task_A_reversed_100.npz`

### 4.3 `3_inspect_data.py` — 数据可视化检查

**参考**: `scripts_pick_place/2_inspect_data.py`

**功能**: 可视化检查 NPZ 数据集 (图像序列, ee_pose 轨迹, action 分布等).
可直接复用旧版脚本, 或简单 wrapper 调用.

### 4.4 `4_train.py` — 训练 Diffusion Policy (DDIM)

**参考**: `scripts_pick_place/4_train_diffusion_ddim.py` + `scripts_pick_place/4_train_diffusion.py`

**功能**: NPZ → LeRobot 数据集转换 + Diffusion Policy 训练. **DDIM 为默认调度器**.

**关键改动** (相对旧版):
- 将 `4_train_diffusion_ddim.py` 的 monkey-patch 机制内置 (见 §1.3 说明)
- `--noise_scheduler_type` 默认值改为 `DDIM`
- `--num_inference_steps` 默认值 `10`
- `--n_action_steps` 默认值改为 `16` (训练时全 chunk)
- `--include_obj_pose` 和 `--include_gripper` 硬编码开启, 不再作为 CLI 参数
- Wrist camera 强制开启 (由数据集自动检测, 不需要参数)
- 其余逻辑与 `4_train_diffusion.py` 完全一致

**CLI 参数**:

```
--dataset              输入 NPZ 路径 (必须)
--out                  输出目录 (checkpoint 保存位置)
--lerobot_dataset_dir  LeRobot 数据集路径 (默认: <out>/lerobot_dataset)
--batch_size           批量大小 (默认: 32)
--steps                总训练步数 (必须)
--lr                   学习率 (默认: 1e-4)
--n_obs_steps          观测回看步数 (默认: 2)
--horizon              扩散预测 horizon / action chunk 长度 (默认: 16)
--n_action_steps       训练时 action 执行步数 (默认: 16, 即全 chunk)
--vision_backbone      视觉骨干 (默认: resnet18)
--noise_scheduler_type 调度器类型: DDIM / DDPM (默认: DDIM)
--num_inference_steps  DDIM 去噪步数 (默认: 10)
--num_episodes         限制使用的 episode 数 (默认: -1, 即全部)
--skip_convert         跳过 NPZ→LeRobot 转换 (已存在时)
--force_convert        强制重新转换
--convert_only         仅转换不训练
--resume               从 checkpoint 恢复训练
--finetune             加载权重但重置步数
--wandb                开启 WandB 日志
--wandb_project        WandB 项目名
--seed                 随机种子 (默认: 42)
--save_freq            checkpoint 保存间隔步数
--sample_weights       采样权重 JSON 路径 (用于加权采样)
```

**Observation State 维度** (固定为 15):

`state = ee_pose(7) + obj_pose(7) + gripper(1)` → **state_dim = 15**

旧版通过 `--include_obj_pose` / `--include_gripper` 控制, 新版硬编码全部开启.

**NPZ → LeRobot 格式转换**:
- 参考 `4_train_diffusion.py` 中的 `convert_npz_to_lerobot_format()` 函数
- 图像编码为视频 (imageio_ffmpeg)
- 生成 LeRobot v3.0 格式数据集

**多 GPU 训练**:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/scripts_pick_place_simulator/4_train.py \
    --dataset data/task_A_reversed_100.npz \
    --out weights/PP_A \
    --steps 31072 --batch_size 128 \
    --wandb
```

注意: `--n_action_steps 16`, `include_obj_pose`, `include_gripper` 已是内置默认值,
无需显式指定.

**步数计算公式**:

$$\text{steps} = \left\lceil \frac{\text{total\_frames} \times \text{epochs}}{\text{batch\_size} \times \text{num\_gpus}} \right\rceil$$

### 4.5 `5_finetune.py` — 迭代微调 (数据聚合)

**参考**: `scripts_pick_place/7_finetune_with_rollout.py`

**功能**: 将新收集的 rollout 数据与原始数据合并, 准备 finetune 数据集.

**CLI 参数**:

```
--original_lerobot    原始 LeRobot 数据集路径
--rollout_data        新收集的 rollout NPZ
--checkpoint          当前 checkpoint 路径
--out                 输出目录
--prepare_only        仅准备数据不训练
--include_obj_pose    同训练脚本
--include_gripper     同训练脚本
--n_action_steps      同训练脚本
```

**数据聚合策略** (DAgger 风格):

```
Iter 1: original_data + rollout_iter1
Iter 2: original_data + rollout_iter1 + rollout_iter2
...
```

**加权采样**: 新旧数据 1:1 等权采样. 参考旧版 README §2 的"Weighted Sampling"说明.
生成 `sampling_weights.json`, 被 `4_train.py --sample_weights` 消费.

### 4.6 `6_eval_cyclic.py` — 循环 A→B 评估 (用于迭代数据收集)

**参考**: `scripts_pick_place/9_eval_with_recovery.py`

**功能**: 运行 N 轮 A→B 循环测试. 每轮:
1. **Task A**: 硬重置 → 方块出现在红色矩形内随机位置 → 运行 Policy A → 拾取并放到绿色目标
2. **Task B**: 硬重置 → 方块出现在绿色目标位置 → 运行 Policy B → 拾取并放到红色矩形内

成功的 episode 会被保存, 用于后续 finetune.

**CLI 参数**:

```
--policy_A             Policy A checkpoint 路径 (必须)
--policy_B             Policy B checkpoint 路径 (必须)
--out_A                Task A 成功 rollout 输出 NPZ (必须)
--out_B                Task B 成功 rollout 输出 NPZ (必须)
--num_cycles           循环次数 (默认: 50)
--horizon              每个 task 最大步数 (默认: 1500)
--distance_threshold   Task A 成功距离 (默认: 0.03)
--n_action_steps       策略推理的执行步数 (默认: 8, 见 §1.4)
--goal_xy              绿色目标 XY (默认: 0.5 -0.2)
--red_region_center_xy 红色矩形中心 (默认: 0.5 0.2)
--red_region_size_xy   红色矩形大小 (默认: 0.3 0.3)
--save_video           保存视频 (默认: False)
--headless             无头模式
```

**硬重置机制**:

| 前一个 Task | 下一个 Task | 重置动作 |
|-------------|-------------|---------|
| (开始) / Task B | Task A | `env.reset()` → 方块传送到红色矩形内随机位置 → settle 10步 |
| Task A | Task B | `env.reset()` → 方块传送到绿色目标位置 → 采样新放置目标 → settle 10步 |

**Policy 推理循环** (参考 `6_test_alternating.py`, L1124–L1145):

```python
for t in range(horizon):
    if t % n_action_steps == 0:        # 每 n_action_steps 步重新推理
        action = policy.select_action(preprocessed_obs)
        action = postprocessor(action)  # 反归一化
    env.step(action)
```

**输出**:
- `out_A.npz` — Task A 成功 rollout
- `out_B.npz` — Task B 成功 rollout
- `out_A.stats.json` — 每轮结果和成功率统计

### 4.7 `7_eval_fair.py` — 独立公平测试 (手动重置)

**参考**: `scripts_pick_place/10_eval_independent.py`

**功能**: 对单个 task 进行独立公平测试. 每个 episode 之间完全硬重置, 消除上一轮的影响.

**CLI 参数**:

```
--policy               Policy checkpoint 路径 (必须)
--task                 测试的任务: A 或 B (必须)
--out                  统计结果 JSON 路径 (必须)
--num_episodes         测试 episode 数 (默认: 50)
--horizon              每个 episode 最大步数 (默认: 1500)
--distance_threshold   Task A 成功距离 (默认: 0.03)
--n_action_steps       策略推理的执行步数 (默认: 8, 见 §1.4)
--goal_xy              绿色目标 XY (默认: 0.5 -0.2)
--red_region_center_xy 红色矩形中心 (默认: 0.5 0.2)
--red_region_size_xy   红色矩形大小 (默认: 0.3 0.3)
--headless             无头模式
```

**与 `6_eval_cyclic.py` 的区别**:
- `6_eval_cyclic.py`: A→B 交替执行, 收集成功轨迹用于 finetune
- `7_eval_fair.py`: 单任务独立评估, **仅测试不收集数据**, 输出成功率统计

**输出** (`stats.json`):

```json
{
    "task": "A",
    "success_rate": 0.86,
    "num_success": 43,
    "num_total": 50,
    "per_episode": [
        {"episode": 0, "success": true, "steps": 312},
        {"episode": 1, "success": false, "steps": 2000},
        ...
    ],
    "elapsed_sec": 245.3,
    "timestamp": "2026-03-20T15:30:45"
}
```

### 4.8 `8_eval_failure_analysis.py` — 失效分析评估

**参考**: `scripts_pick_place/12_eval_failure_analysis.py` (Task B) + `scripts_pick_place/13_eval_failure_analysis_A.py` (Task A)

**功能**: 运行 N 个独立 episode, 保存所有 rollout 数据, 并为**失败 episode 渲染标注 MP4 视频**
以便于人工失效分析. 旧版分为两个脚本 (12 针对 Task B, 13 针对 Task A), 新版合并为
单脚本, 通过 `--task A/B` 切换.

**CLI 参数**:

```
--policy               Policy checkpoint 路径 (必须)
--task                 测试的任务: A 或 B (必须)
--num_episodes         episode 数 (默认: 100)
--run_id               运行标识, 用于并行执行区分输出 (默认: run0)
--out_dir              输出目录 (必须)
--horizon              每个 episode 最大步数 (默认: 1500)
--distance_threshold   Task A 成功距离 (默认: 0.03)
--n_action_steps       策略推理的执行步数 (默认: 8, 见 §1.4)
--video_fps            视频帧率 (默认: 30)
--render_success_videos  同时渲染成功 episode 的视频 (默认: 仅渲染失败)
--seed                 随机种子 (默认: 从 run_id 哈希)
--headless             无头模式
```

**每个 episode 的流程**:
1. `env.reset()` + `pre_position_gripper_down()`
2. Task A: 方块传送到红色矩形内随机位置; Task B: 方块传送到绿色目标位置
3. settle 10 步 → 运行 policy → 记录所有帧数据
4. 判定成功/失败
5. 如果失败 (或指定 `--render_success_videos`): 渲染标注视频

**视频标注内容** (参考 `render_episode_video()` in `12_eval_failure_analysis.py`):
- Episode 编号、步数计数器
- 物体 XY 位置、Z 高度
- 到目标的距离 (Task A) 或区域内外状态 (Task B)
- 夹爪状态 (OPEN/CLOSED)
- 进度条 + 最终结果 (SUCCESS/FAILED)

**输出**:

```
<out_dir>/
├── failure_analysis_<task>_<run_id>.json    # 每个 episode 的详细统计
├── failure_analysis_<task>_<run_id>.npz     # 所有 episode 的 rollout 数据
└── failure_videos_<task>_<run_id>/
    ├── fail_ep000.mp4                       # 失败 episode 标注视频
    ├── fail_ep003.mp4
    └── ...
```

**JSON 统计格式**:

```json
{
    "experiment": "failure_analysis_task_A",
    "timestamp": "2026-03-20T15:30:45",
    "run_id": "gpu0",
    "config": { "policy": "...", "num_episodes": 100, ... },
    "summary": {
        "total_episodes": 100,
        "success_count": 86,
        "fail_count": 14,
        "success_rate": 0.86
    },
    "episodes": [
        {
            "episode_index": 0,
            "success": true,
            "success_step": 312,
            "total_steps": 312,
            "final_dist_to_goal": 0.021
        },
        ...
    ]
}
```

**并行执行** (多 GPU 分摊):

```bash
# GPU 0: 100 episodes
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \
    --policy weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --task A --num_episodes 100 --run_id gpu0 \
    --out_dir data/exp_new --headless

# GPU 1: 100 episodes (并行)
CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \
    --policy weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --task A --num_episodes 100 --run_id gpu1 \
    --out_dir data/exp_new --headless
```

### 4.9 `run_pipeline.sh` — 统一编排脚本

**参考**: `data/pick_place_isaac_lab_simulation/exp14/run_exp14_iterative.sh`

**功能**: 端到端实验编排.

**Pipeline 流程**:

```
Phase 1: 数据准备
  ├── Step 1: 收集 Task B 数据 (1_collect_task_B.py)
  ├── Step 2: 时间反转得到 Task A (2_reverse_to_task_A.py)
  └── Step 3: NPZ → LeRobot 格式转换 (4_train.py --convert_only)

Phase 2: 初始训练 (并行)
  ├── GPU_A: 训练 Policy A (4_train.py, 使用反转后的 Task A 数据)
  └── GPU_B: 训练 Policy B (4_train.py, 使用原始 Task B 数据)

Phase 3: 迭代改进 (N 轮, 每轮 3 个评估并行)
  ├── Step 1 (并行, 3 GPU):
  │   ├── GPU_1: Cyclic 评估 + 数据收集 (6_eval_cyclic.py)
  │   ├── GPU_2: Fair test Task A (7_eval_fair.py --task A)
  │   └── GPU_3: Fair test Task B (7_eval_fair.py --task B)
  ├── Step 2: 数据聚合 (5_finetune.py --prepare_only)
  ├── Step 3: 恢复训练 A (4_train.py --resume)
  ├── Step 4: 恢复训练 B (4_train.py --resume)
  └── Step 5: 记录指标 (cyclic + fair test), 旋转 checkpoint 目录

Phase 4: (已合并到 Phase 3 的每轮评估中)

Phase 5: 失效分析 (按需手动执行)
  └── 8_eval_failure_analysis.py --task A/B
```

**配置区** (脚本开头的 bash 变量):

```bash
# 目录
BASE_DIR="data/pick_place_isaac_lab_simulation/expXX"
LOG_DIR="${BASE_DIR}/logs"

# 区域参数 (Mode 3, 固定不变)
GOAL_X="0.5"
GOAL_Y="-0.2"
DISTANCE_THRESHOLD="0.03"
RED_REGION_CENTER_X="0.5"
RED_REGION_CENTER_Y="0.2"
RED_REGION_SIZE_X="0.3"
RED_REGION_SIZE_Y="0.3"

# 训练参数
BATCH_SIZE=128
TRAIN_N_ACTION_STEPS=16      # 训练时 action chunk 全执行
EVAL_N_ACTION_STEPS=8        # 推理时只执行前 8 步
TARGET_EPOCHS=500            # 初始训练 epoch 数
STEPS_PER_ITER=5000          # 每轮 finetune 增加的步数

# 迭代参数
ITER_ROUNDS=10
NUM_CYCLES=50                # 每轮 cyclic 测试的循环数
HORIZON=1500                 # 每个 task 最大步数

# GPU 分配
TRAIN_GPUS="0,1"
COLLECT_GPU="0"

# DDIM
NOISE_SCHEDULER_TYPE="DDIM"
NUM_INFERENCE_STEPS=10

# WandB
WANDB_PROJECT="rev2fwd-expXX"
```

**Helper 函数** (从 `run_exp14_iterative.sh` 复用):

| 函数 | 功能 |
|------|------|
| `log()` | 带时间戳的日志 |
| `gpu_count()` | 计算 GPU 数量 |
| `random_port()` | 随机分配 torchrun master port |
| `get_ckpt()` | 标准化 checkpoint 路径: `<dir>/checkpoints/checkpoints/last/pretrained_model` |
| `get_step()` | 读取当前训练步数 |
| `calc_steps_from_npz()` | 根据 NPZ 帧数计算训练步数 |
| `append_record()` | 追加迭代指标到 `record.json` |
| `run_resume_train()` | 恢复训练的统一封装 |

**目录结构**:

```
data/pick_place_isaac_lab_simulation/expXX/
├── config.json                        # 实验配置 (区域参数等)
├── task_B_100.npz                     # Step 1: Task B 原始数据
├── task_A_reversed_100.npz            # Step 2: 时间反转的 Task A 数据
├── lerobot/
│   ├── task_A/                        # LeRobot 格式 Task A
│   └── task_B/                        # LeRobot 格式 Task B
├── weights/
│   ├── PP_A/                          # Policy A checkpoint
│   └── PP_B/                          # Policy B checkpoint
├── work/
│   ├── PP_A_temp/                     # 迭代训练工作目录 (上一轮)
│   ├── PP_A_last/                     # 迭代训练工作目录 (当前轮)
│   ├── PP_B_temp/
│   └── PP_B_last/
├── iter1_collect_A.npz                # 第1轮收集的 Task A rollout
├── iter1_collect_B.npz
├── iter1_collect_A.stats.json
├── iter1_fair_A.stats.json            # 第1轮 Fair test A 结果
├── iter1_fair_B.stats.json            # 第1轮 Fair test B 结果
├── fair_test_A.stats.json             # 最新 Fair test A (从最后一轮复制)
├── fair_test_B.stats.json             # 最新 Fair test B (从最后一轮复制)
├── record.json                        # 迭代指标记录 (含 cyclic + fair test)
├── collection_curve.png               # 成功率曲线图
├── logs/
│   ├── pipeline.log
│   ├── collect_B.log
│   ├── reverse.log
│   ├── train_A.log
│   ├── train_B.log
│   ├── iter1_collect.log
│   ├── iter1_fair_A.log               # 第1轮 Fair test A 日志
│   ├── iter1_fair_B.log               # 第1轮 Fair test B 日志
│   ├── iter1_train_A.log
│   ├── fair_test_A.log
│   └── fair_test_B.log
├── failure_analysis_A_gpu0.json        # 失效分析 (按需)
├── failure_analysis_A_gpu0.npz
├── failure_videos_A_gpu0/
│   ├── fail_ep000.mp4
│   └── ...
└── run_pipeline.sh → (symlink or copy)
```

---

## 5. 共享模块依赖 (src/rev2fwd_il)

新脚本应直接 import 以下已有模块 (不要重复实现):

| 模块路径 | 用途 |
|----------|------|
| `src/rev2fwd_il/sim/make_env.py` | Isaac Lab 环境创建 |
| `src/rev2fwd_il/sim/scene_api.py` | 场景 API (物体传送, 标记放置等) |
| `src/rev2fwd_il/sim/task_spec.py` | 任务参数定义 (`TaskSpec`) |
| `src/rev2fwd_il/sim/obs_utils.py` | 观测处理工具 |
| `src/rev2fwd_il/experts/pickplace_expert_b.py` | Task B FSM 专家 |
| `src/rev2fwd_il/data/io_npz.py` | NPZ 读写工具 |
| `src/rev2fwd_il/data/reverse_time.py` | 时间反转工具 |
| `src/rev2fwd_il/data/recorder.py` | 数据记录器 |
| `src/rev2fwd_il/train/lerobot_train_with_viz.py` | LeRobot 训练封装 (含加权采样) |

---

## 6. 实现检查清单

给下一个 AI agent 的指引:

- [ ] **1_collect_task_B.py**: 从 `1_collect_data_pick_place.py` 提取 Mode 3 分支.
      删除所有 `if target_mode == "legacy"` / `"green_region"` / circle 相关逻辑.
      保留 `rollout_expert_B_with_next_frame_actions()` 核心函数.
      硬编码 `red_marker_shape="rectangle"`, `fix_red_marker_pose=1`.
      强制开启 wrist camera, `wrist_images` 字段必须存在于输出 NPZ.

- [ ] **2_reverse_to_task_A.py**: 基本照搬 `3_1_make_forward_data.py`,
      整理 CLI 参数名使之与本 pipeline 一致.

- [ ] **3_inspect_data.py**: 简单 wrapper, 调用 `2_inspect_data.py` 的核心逻辑.

- [ ] **4_train.py**: 合并 `4_train_diffusion.py` + `4_train_diffusion_ddim.py`.
      将 DDIM monkey-patch 逻辑内置: 在 `main()` 中直接将 `DiffusionConfig` 的
      默认 `noise_scheduler_type` 设为 `"DDIM"`, `num_inference_steps` 设为 `10`.
      `--n_action_steps` 默认值改为 **16** (训练时全 chunk).
      硬编码 `include_obj_pose=True`, `include_gripper=True` (不再接受 CLI 参数).
      其余逻辑 (数据转换, 多GPU, WandB) 全部复用.

- [ ] **5_finetune.py**: 照搬 `7_finetune_with_rollout.py`,
      整理 CLI 参数名.

- [ ] **6_eval_cyclic.py**: 从 `9_eval_with_recovery.py` 提取,
      删除 non-Mode3 的代码路径. `--n_action_steps` 默认值改为 **8**.
      `--horizon` 默认值改为 **1500**.
      确保 DDIM 模型加载正确
      (参考 `load_policy_auto()` in `6_test_alternating.py`).

- [ ] **7_eval_fair.py**: 从 `10_eval_independent.py` 提取,
      简化为单 task 接口 (`--task A` 或 `--task B`). 删除 non-Mode3 代码.
      `--n_action_steps` 默认值改为 **8**. `--horizon` 默认 **1500**.

- [ ] **8_eval_failure_analysis.py**: 合并旧版 `12_eval_failure_analysis.py` (Task B)
      和 `13_eval_failure_analysis_A.py` (Task A) 为单脚本.
      通过 `--task A/B` 切换. 复用 `render_episode_video()` 和
      `add_multi_line_overlay()` 等标注渲染函数.
      Task A: 标注 dist_to_goal; Task B: 标注区域内外状态 (dX/dY).
      `--n_action_steps` 默认 **8**. `--horizon` 默认 **1500**.
      参考旧版的 `AlternatingTester` + dummy policy slot 模式来加载单个 policy.

- [ ] **run_pipeline.sh**: 从 `run_exp14_iterative.sh` 精简,
      添加 Phase 1 数据准备 (收集 + 反转), 传递 DDIM 参数.
      训练时传 `--n_action_steps 16`, 评估时传 `--n_action_steps 8`.
      初始训练 `TARGET_EPOCHS=500`. 测试 `HORIZON=1500`.

---

## 7. Quick Start

```bash
conda activate rev2fwd_il
export CUDA_VISIBLE_DEVICES=0,1

# 1. 收集 Task B 数据
python scripts/scripts_pick_place_simulator/1_collect_task_B.py \
    --out data/exp_new/task_B_100.npz \
    --num_episodes 100 --headless

# 2. 时间反转得到 Task A
python scripts/scripts_pick_place_simulator/2_reverse_to_task_A.py \
    --input data/exp_new/task_B_100.npz \
    --out data/exp_new/task_A_reversed_100.npz \
    --verify

# 3. 训练 Policy A (DDIM)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/scripts_pick_place_simulator/4_train.py \
    --dataset data/exp_new/task_A_reversed_100.npz \
    --out data/exp_new/weights/PP_A \
    --steps 31072 --batch_size 128 --wandb

# 4. 训练 Policy B (DDIM)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/scripts_pick_place_simulator/4_train.py \
    --dataset data/exp_new/task_B_100.npz \
    --out data/exp_new/weights/PP_B \
    --steps 31072 --batch_size 128 --wandb

# 5. 循环评估 + 数据收集 (n_action_steps=8 推理)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/6_eval_cyclic.py \
    --policy_A data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --policy_B data/exp_new/weights/PP_B/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/exp_new/iter1_collect_A.npz \
    --out_B data/exp_new/iter1_collect_B.npz \
    --num_cycles 50 --headless

# 6. 公平测试
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/7_eval_fair.py \
    --policy data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --task A --num_episodes 50 \
    --out data/exp_new/fair_test_A.stats.json --headless

# 7. 失效分析 (手动按需执行)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \
    --policy data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
    --task A --num_episodes 100 --run_id gpu0 \
    --out_dir data/exp_new --headless

# 或一键执行完整 pipeline
nohup bash scripts/scripts_pick_place_simulator/run_pipeline.sh \
    > data/exp_new/logs/pipeline.log 2>&1 &
```


