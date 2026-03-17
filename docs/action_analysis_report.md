# 时间反转数据集 Action 分析报告

## 1. 问题描述

在 rev2fwd-il 项目中，脚本 1 (`1_collect_data_pick_place.py`) 采集反向轨迹 (Task B: 从目标位置取物放到桌面)，
脚本 3 (`3_make_forward_data.py`) 通过时间反转生成正向轨迹 (Task A: 从桌面取物放到目标位置)。

实际测试发现：**倒放的数据集在机械臂上执行时，得到的轨迹和正向收集的轨迹差距很大。** 本报告分析其根本原因。

---

## 2. Action 字段的真实含义

### 2.1 你期望的 action

你原本希望 `action[t]` 是 **下一帧的 ee_pose**，即：
```
action[t] ≈ ee_pose[t+1]    (期望)
```

### 2.2 实际记录的 action

代码中 `action[t]` 记录的是 `get_fsm_goal_action()` 的输出，即 **FSM 状态机的目标 waypoint**：

```python
# 1_collect_data_pick_place.py 第 900 行
goal_action = get_fsm_goal_action(expert, object_pose)   # <-- 记录的是这个
...
action_lists[i].append(goal_action_np[i])

# 真正发给 env.step() 的是:
action = expert.act(ee_pose, object_pose)                 # <-- 但实际执行的是这个
obs_dict, reward, terminated, truncated, info = env.step(action)
```

虽然 `expert.act()` 和 `get_fsm_goal_action()` 在位置上输出相同的 waypoint，
但关键问题是：**这个 waypoint 是一个远程的恒定目标，不是下一帧位置。**

### 2.3 量化证据 (exp9 数据集)

| 对比项 | B 数据集 | A 数据集 |
|--------|---------|---------|
| `action[t]` vs `ee_pose[t]` 平均距离 | 0.074 m | 0.136 m |
| `action[t]` vs `ee_pose[t+1]` 平均距离 | 0.070 m | 0.133 m |
| `ee_pose[t+1]` vs `ee_pose[t]` 平均步长 | 0.005 m | 0.005 m |
| **action 偏离 next-frame 倍数** | **13.5 倍** | **25.8 倍** |

**action 离下一帧 ee_pose 的距离是实际步长的 13~26 倍，说明 action 绝不是 next-frame ee_pose。**

### 2.4 Action 在 FSM 段内的行为

在同一个 FSM 状态段内，action **保持恒定**，而 ee_pose 逐渐逼近：

```
B 数据集 Episode 0:

State  1 (GO_ABOVE_OBJ), t=0~80:
  action 恒定 = (0.503, 0.001, 0.259)    ← FSM 目标 waypoint
  ee_pose 从 (0.375, -0.001, 0.525)      ← 实际起点
          到 (0.503, 0.000, 0.257)        ← 经过 80 步逐渐逼近

State  2 (GO_TO_OBJ), t=80~117:
  action 恒定 = (0.500, 0.000, 0.021)    ← FSM 目标 (物体表面)
  ee_pose 从 (0.503, 0.000, 0.257)       ← 从上方
          到 (0.499, -0.000, 0.028)       ← 经过 37 步降到物体表面
```

每段内的 XYZ range = [0.000, 0.000, 0.000]，**action 完全不变。**

---

## 3. 时间反转的问题

### 3.1 ee_pose 反转是正确的

```
A.ee_pose vs reverse(B.ee_pose)[:-1]: max=0.00000000, mean=0.00000000
```

脚本 3 对 ee_pose 的反转完全正确，A 的 ee_pose 就是 B 的 ee_pose 倒序。

### 3.2 Action 反转逻辑详解

```
A.action vs reverse(B.action)[:-1]: max=0.313, mean=0.140
```

脚本 3 并没有简单地对 action 做时间反转，而是通过 `compute_forward_goal_actions()` **重新计算**了正向的 goal action。
下面详细拆解该函数的算法逻辑，并用 exp9 episode 0 的真实数据做完整追踪。

#### 3.2.1 算法步骤

`compute_forward_goal_actions(fsm_state_rev, action_rev)` 的核心逻辑（见 `3_make_forward_data.py` 第 105-155 行）：

```python
# 1. 在反转后的 fsm_state 序列中，找到状态转换点
transitions = np.where(np.diff(fsm_state_rev) != 0)[0]
boundaries = np.concatenate([[0], transitions + 1, [T]])

# 2. 对每个 FSM 段 [start, end):
for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i + 1]
    
    if end < T:
        # 非最后一段: 取下一段第一帧的 action 值
        goal_action = action_rev[end].copy()
    else:
        # 最后一段: 取本段末帧的 action 值
        goal_action = action_rev[end - 1].copy()
    
    new_actions[start:end] = goal_action
```

**用一句话概括**: 对于反转后的每个 FSM 段，用 **下一段第一帧的 action** 作为当前段的新 goal action；最后一段则沿用本段的 action。

#### 3.2.2 直觉解释

为什么要这么做？作者的意图是：

- B 数据集中，每个 FSM 段的 action 是**该段的目标 waypoint**（即机械臂正在趋近的位置）
- 时间反转后，机械臂运动的方向反了——原来是"从 A 到 B"，现在变成"从 B 到 A"
- 因此，反转后每段应该"往下一段的位置走"
- 下一段的 action 正好记录了那个位置的 waypoint → 用它作为当前段的目标

#### 3.2.3 用 exp9 ep0 数据完整追踪

**Step 1: B 原始 FSM 段（正序，共 11 段）**

| 段 | FSM State | 名称 | t 范围 | action_xyz | ee 起点 → 终点 |
|----|-----------|------|--------|------------|---------------|
| 0 | 1 | GO_ABOVE_OBJ | [0, 80) | (0.503, 0.001, 0.259) | (0.375, -0.001, 0.525) → (0.497, 0.000, 0.263) |
| 1 | 2 | GO_TO_OBJ | [80, 117) | (0.500, 0.000, 0.021) | (0.503, 0.000, 0.257) → (0.499, 0.000, 0.028) |
| 2 | 3 | CLOSE | [117, 126) | (0.500, 0.000, 0.021) | (0.499, 0.000, 0.028) → (0.500, 0.000, 0.027) |
| 3 | 4 | LIFT | [126, 161) | (0.492, 0.001, 0.259) | (0.489, 0.000, 0.031) → (0.500, 0.004, 0.256) |
| 4 | 5 | GO_ABOVE_PLACE | [161, 190) | (0.451, 0.145, 0.259) | (0.496, 0.006, 0.255) → (0.452, 0.142, 0.260) |
| 5 | 6 | GO_TO_PLACE | [190, 225) | (0.450, 0.150, 0.055) | (0.457, 0.144, 0.255) → (0.449, 0.149, 0.061) |
| 6 | 7 | LOWER_TO_RELEASE | [225, 246) | (0.450, 0.150, 0.015) | (0.450, 0.150, 0.059) → (0.450, 0.151, 0.026) |
| 7 | 8 | OPEN | [246, 255) | (0.450, 0.150, 0.015) | (0.450, 0.151, 0.025) → (0.450, 0.150, 0.022) |
| 8 | 9 | LIFT_AFTER_RELEASE | [255, 289) | (0.451, 0.145, 0.259) | (0.438, 0.147, 0.025) → (0.452, 0.146, 0.257) |
| 9 | 10 | RETURN_REST | [289, 341) | (0.375, -0.001, 0.525) | (0.455, 0.138, 0.262) → (0.379, 0.001, 0.528) |
| 10 | 11 | DONE | [341, 388) | (0.375, -0.001, 0.525) | (0.378, 0.001, 0.528) → (0.371, -0.001, 0.525) |

**Step 2: 时间反转后的 FSM 段（倒序排列）**

所有数组做 `[::-1]` 反转后，FSM 段序变为 11→10→9→8→7→6→5→4→3→2→1：

| 段 | FSM State | t 范围 | act_rev_xyz（反转后的原始 action）|
|----|-----------|--------|----------------------------------|
| 0 | 11 (DONE) | [0, 47) | (0.375, -0.001, 0.525) |
| 1 | 10 (RETURN_REST) | [47, 99) | (0.375, -0.001, 0.525) |
| 2 | 9 (LIFT_AFTER_REL) | [99, 133) | (0.451, 0.145, 0.259) |
| 3 | 8 (OPEN) | [133, 142) | (0.450, 0.150, 0.015) |
| 4 | 7 (LOWER_RELEASE) | [142, 163) | (0.450, 0.150, 0.015) |
| 5 | 6 (GO_TO_PLACE) | [163, 198) | (0.450, 0.150, 0.055) |
| 6 | 5 (GO_ABOVE_PLACE) | [198, 227) | (0.451, 0.145, 0.259) |
| 7 | 4 (LIFT) | [227, 262) | (0.503, 0.005, 0.259) |
| 8 | 3 (CLOSE) | [262, 271) | (0.500, 0.000, 0.021) |
| 9 | 2 (GO_TO_OBJ) | [271, 308) | (0.500, 0.000, 0.021) |
| 10 | 1 (GO_ABOVE_OBJ) | [308, 388) | (0.503, 0.001, 0.259) |

**Step 3: `compute_forward_goal_actions` 的赋值过程**

算法遍历上表每一行，为每段赋值新的 goal action：

| 段 | FSM | t 范围 | 新 action ← 来源 | 新 action_xyz |
|----|-----|--------|-------------------|---------------|
| 0 | 11 | [0, 47) | act_rev[47]（段1首帧, state=10）| (0.375, -0.001, 0.525) |
| 1 | 10 | [47, 99) | act_rev[99]（段2首帧, state=9）| (0.451, 0.145, 0.259) |
| 2 | 9 | [99, 133) | act_rev[133]（段3首帧, state=8）| (0.450, 0.150, 0.015) |
| 3 | 8 | [133, 142) | act_rev[142]（段4首帧, state=7）| (0.450, 0.150, 0.015) |
| 4 | 7 | [142, 163) | act_rev[163]（段5首帧, state=6）| (0.450, 0.150, 0.055) |
| 5 | 6 | [163, 198) | act_rev[198]（段6首帧, state=5）| (0.451, 0.145, 0.259) |
| 6 | 5 | [198, 227) | act_rev[227]（段7首帧, state=4）| (0.503, 0.005, 0.259) |
| 7 | 4 | [227, 262) | act_rev[262]（段8首帧, state=3）| (0.500, 0.000, 0.021) |
| 8 | 3 | [262, 271) | act_rev[271]（段9首帧, state=2）| (0.500, 0.000, 0.021) |
| 9 | 2 | [271, 308) | act_rev[308]（段10首帧, state=1）| (0.503, 0.001, 0.259) |
| 10 | 1 | [308, 388) | act_rev[387]（本段末帧，最后一段特殊处理）| (0.503, 0.001, 0.259) |

验证结果：**以上计算结果与 A 数据集中的 action 完全吻合（max diff = 0.000, mean diff = 0.000）。**

#### 3.2.4 语义分析：每段新 action 指向了正确的位置吗？

将每段的新 action 与该段 ee_pose 的实际运动方向做对比：

| 段 | FSM State | ee 运动方向 | 新 action 指向 | action 含义是否合理 |
|----|-----------|------------|---------------|-------------------|
| 0 | 11 DONE | 静止不动 | ≈ 当前位置 (rest) | ✅ 机械臂静止，目标就是当前位 |
| 1 | 10 RETURN_REST | rest → place 上方 | place 上方 (0.451, 0.145, 0.259) | ❌ **方向错误**: 反转后应该从 rest 往 place 走，但 action 直接指向 place 上方，跳过了中间段。实际是 B 的 LIFT_AFTER_REL 的 waypoint |
| 2 | 9 LIFT_AFTER_REL | place 上方 → place 低处 | place 表面 (0.450, 0.150, 0.015) | ⚠️ **跳步**: 反转后应向 place 低处移动，action 直接指向极低位置，跳过了 GO_TO_PLACE 级别(z=0.055) |
| 3 | 8 OPEN | 静止（开爪） | 同上 (0.450, 0.150, 0.015) | ✅ 开爪不移动，action 影响不大 |
| 4 | 7 LOWER_RELEASE | 上升（反转后变下降） | place 表面 (0.450, 0.150, 0.055) | ⚠️ **方向大致对**，但 z=0.055 vs 段终 z=0.059，差距非常小 |
| 5 | 6 GO_TO_PLACE | 上升至 hover 高度 | place 上方 (0.451, 0.145, 0.259) | ✅ 方向正确，向上走向 hover |
| 6 | 5 GO_ABOVE_PLACE | 水平移向 obj 上方 | obj 上方 (0.503, 0.005, 0.259) | ✅ 方向正确，从 place 上方水平移到 obj 上方 |
| 7 | 4 LIFT | 下降（反转后变下降） | obj 表面 (0.500, 0.000, 0.021) | ✅ 方向正确，从高处下降到物体表面 |
| 8 | 3 CLOSE | 静止（关爪） | 同上 (0.500, 0.000, 0.021) | ✅ 关爪不移动，action 合理 |
| 9 | 2 GO_TO_OBJ | 上升至 hover 高度 | obj 上方 (0.503, 0.001, 0.259) | ✅ 方向正确，往上走 |
| 10 | 1 GO_ABOVE_OBJ | 水平移向 rest | obj 上方 (0.503, 0.001, 0.259) | ❌ **方向错误**: 反转后应从 obj 上方移向 rest 位置 (0.375, -0.001, 0.525)，但 action 指向的却是 obj 上方本身。这是因为最后一段特殊处理取了 `act_rev[end-1]` |

#### 3.2.5 逐段对比：A 新 action vs B 中同一 FSM state 的原始 action

更直接地看 `compute_forward_goal_actions` 的效果——对于同一个 FSM state，A 数据集中该段的 action 是否和 B 数据集中该段的 action 一致？

| A 段 | FSM State | A 的新 action_xyz | B 中该 state 的原始 action_xyz | diff | 一致？ |
|------|-----------|-------------------|-------------------------------|------|--------|
| 0 | 11 | (0.375, -0.001, 0.525) | (0.375, -0.001, 0.525) | 0.000 | ✅ |
| 1 | 10 | (0.451, 0.145, 0.259) | (0.375, -0.001, 0.525) | **0.313** | ❌ |
| 2 | 9 | (0.450, 0.150, 0.015) | (0.451, 0.145, 0.259) | **0.244** | ❌ |
| 3 | 8 | (0.450, 0.150, 0.015) | (0.450, 0.150, 0.015) | 0.000 | ✅ |
| 4 | 7 | (0.450, 0.150, 0.055) | (0.450, 0.150, 0.015) | **0.040** | ❌ |
| 5 | 6 | (0.451, 0.145, 0.259) | (0.450, 0.150, 0.055) | **0.204** | ❌ |
| 6 | 5 | (0.503, 0.005, 0.259) | (0.451, 0.145, 0.259) | **0.150** | ❌ |
| 7 | 4 | (0.500, 0.000, 0.021) | (0.492, 0.001, 0.259) | **0.238** | ❌ |
| 8 | 3 | (0.500, 0.000, 0.021) | (0.500, 0.000, 0.021) | 0.000 | ✅ |
| 9 | 2 | (0.503, 0.001, 0.259) | (0.500, 0.000, 0.021) | **0.238** | ❌ |
| 10 | 1 | (0.503, 0.001, 0.259) | (0.503, 0.001, 0.259) | 0.000 | ✅ |

**11 段中只有 4 段一致（state 11, 8, 3, 1），其余 7 段的 action 完全不同。**

#### 3.2.6 规律总结

通过以上追踪可以看出 `compute_forward_goal_actions` 的系统性偏差：

1. **"取下一段"的本质是取了 B 中"上一段"的 waypoint**：
   反转后段序是 11→10→9→...→1，取"下一段的 action"等价于取了 B 中前一个 FSM state 的 waypoint。
   例如 A 段 1 (state=10) 拿到的是 B state=9 的 waypoint，而不是 B state=10 自己的 waypoint。

2. **静止段一致，运动段不一致**：
   一致的都是不需要运动的段（DONE, OPEN, CLOSE）或恰好相邻段目标相同的段（GO_ABOVE_OBJ 的首尾段）。
   真正需要精确运动方向的段几乎全部错误。

3. **根本问题——正向和反向的 waypoint 语义不同**：
   B 任务 state=10 (RETURN_REST) 的 waypoint 是 rest 位置 (0.375, -0.001, 0.525)；
   反转后 A 中 state=10 段的运动方向是 **离开 rest → 去 place**，
   但 `compute_forward_goal_actions` 给出的 action 是 place 上方 (0.451, 0.145, 0.259)——
   这虽然方向大致对（离开 rest），但值来自 B 的 LIFT_AFTER_RELEASE 的 waypoint，语义完全不同。

### 3.3 Replay 实测结果

使用 `3_replay_ab_actions_fixed_start_goal.py` (command_source=ee_pose, 默认值) 测试：

| 回放 | 成功? | 最终距离 | 回放轨迹 vs 原始轨迹 |
|------|-------|---------|-------------------|
| Task B (原始) | ✅ 成功 | 0.001 m | mean=0.063, max=0.187 |
| Task A (反转) | ❌ 失败 | 0.158 m | mean=0.050, max=0.145 |

**关键发现：即使是 B 本身的 ee_pose 回放，轨迹也偏离了原始轨迹 (mean=0.063m)！**

---

## 4. 根本原因分析

### 4.1 原因一：Action 是段级 waypoint，不是帧级目标

```
原始采集:                          回放:
                                   
t=0:  action=W₁  →  ee逐渐移向W₁   t=0:  发送ee[0]作为目标 → IK求解
t=1:  action=W₁  →  ee继续移向W₁   t=1:  发送ee[1]作为目标 → IK求解
...                                ...
t=80: action=W₂  →  ee跳转向W₂     t=80: 发送ee[80]作为目标 → IK求解
```

原始采集中，IK 控制器持续收到 **同一个远程 waypoint** 作为目标，产生的是一个
由 IK 求解器产生的 **自然过渡轨迹**。

回放时，无论用 action (waypoint) 还是 ee_pose，IK 控制器面对的 **初始关节状态不同**，
产生的关节轨迹也不同，导致最终笛卡尔空间轨迹偏移。

### 4.2 原因二：IK 控制器的路径依赖性

Isaac Lab 使用 `DifferentialIKController` (command_type="pose", use_relative_mode=False)。

该控制器在每一步：
1. 计算目标 pose 与当前 pose 的差异
2. 通过雅可比矩阵求解关节增量
3. 应用到当前关节位置

这意味着：**即使目标相同，不同的起始关节配置会产生不同的运动路径。**
特别是对于冗余机器人 (Franka 7-DOF)，同一个笛卡尔 pose 可能有多个 IK 解。

### 4.3 原因三：物理仿真的不可逆性

- **重力**: 抬起物体需要力，放下物体受重力辅助，两个方向的动力学不对称
- **夹爪状态**: 反向播放时夹爪开关时序需要精确匹配，但 IK 轨迹偏移会导致时序错位
- **接触力学**: 抓取/释放时的接触状态在反向播放时无法完美复现

### 4.4 原因四：ee_pose 作为目标的滞后问题

当使用 `command_source=ee_pose` 回放时：
- 发送 `ee_pose[t]` 作为 t 时刻的目标
- 但 `ee_pose[t]` 是当前帧的位置，不是下一帧的目标
- IK 控制器看到 "目标≈当前位置" → 产生微小运动
- 实际上应该发送 `ee_pose[t+1]` 才能近似复现原始运动

不过从 replay 脚本的代码看，回放是将整个序列 ee_pose 作为逐帧 IK 绝对目标发送的。
由于 ee_pose 帧间变化小 (平均 0.005m)，IK 控制器每步只需做微调，所以 B 回放还是比较接近的。
但 A 回放由于反向运动的路径依赖性导致误差累积更快。

---

## 5. 为什么 B 回放成功但 A 回放失败?

根本原因是 **任务方向的不对称性**:

**B 回放 (goal → table)**: 
- 物体从 goal 位置出发
- 机械臂先 hover，再下降抓取，运动简单直接
- 即使轨迹略有偏移，抓取/放置的容错性较好

**A 回放 (table → goal)**: 
- 物体从 table 随机位置出发
- 反转后的轨迹在关键时刻 (抓取、释放) 的精度要求更高
- 轨迹偏移导致抓取失误 → 后续全部错位
- 最终物体没到目标位置 (dist=0.158m)

---

## 6. 修改建议

### 方案 A：将 action 改为 next-frame ee_pose（推荐）

**最直接的解决方案**：在脚本 1 中，将 action 记录为 `ee_pose[t+1]` 而不是 FSM waypoint。

```python
# 修改 1_collect_data_pick_place.py 的记录逻辑:

# 当前: 记录 FSM waypoint
goal_action = get_fsm_goal_action(expert, object_pose)
action_lists[i].append(goal_action_np[i])

# 改为: 先执行 step, 再记录下一帧的 ee_pose 作为 action
action = expert.act(ee_pose, object_pose)
obs_dict, reward, terminated, truncated, info = env.step(action)
next_ee_pose = get_ee_pose_w(env)
# action = [next_ee_pose, gripper_from_expert]
```

**优点**: 
- action[t] 就是下一帧的真实 ee_pose，回放时可以精确复现轨迹
- 时间反转后 action 的语义也是正确的（反转后 next-frame ee_pose 仍然是有效的运动目标）

**缺点**:
- 需要重新采集数据

### 方案 B：回放时使用 ee_pose[t+1] 而非 ee_pose[t]

如果不想重新采集，可以修改回放逻辑：

```python
# 当前回放逻辑:
for t in range(T):
    act_env = ee_pose_sequence[t]  # 发送当前帧 pose 作为目标
    env.step(act_env)

# 改为:
for t in range(T - 1):
    act_env = ee_pose_sequence[t + 1]  # 发送下一帧 pose 作为目标
    env.step(act_env)
```

这等价于让 IK 控制器 "往前看一步"，跟踪能力会好很多。

### 方案 C：使用轨迹跟踪控制器代替逐帧 IK

```python
# 用整条轨迹做样条插值，然后用更密集的时间步跟踪
from scipy.interpolate import CubicSpline
cs = CubicSpline(time_orig, ee_pose_trajectory)
for t_dense in np.linspace(0, T, T * substeps):
    target = cs(t_dense)
    env.step(target)
```

**优点**: 更平滑的轨迹跟踪，减少 IK 振荡
**缺点**: 实现复杂度高

### 方案 D：重采集时保存 expert.act() 的输出 + next-frame ee_pose

最稳妥的方案——同时保存两种 action：

1. `action_waypoint`: FSM 目标 waypoint（当前已有）
2. `action_next_ee`: 下一帧的实际 ee_pose（新增）

这样训练策略时可以选择用哪种标签，回放验证也有正确的基准。

---

## 7. 关于让正向和反向 action replay 轨迹相等

### 7.1 理论上的不可能性

严格来说，**在物理仿真中正向和反向 action replay 轨迹不可能完全相等**，因为：
- IK 求解的路径依赖性
- 物理仿真的不可逆性（重力、摩擦）
- 数值积分的非对称误差

### 7.2 实践上的最佳近似

能做到的最好结果是：**让回放轨迹尽可能接近记录的 ee_pose 轨迹。**

为此需要：

1. **Action = next-frame ee_pose** (方案 A)
   - 这样 replay 时 IK 控制器会精确跟踪记录的轨迹
   - 时间反转后，反转的 next-frame ee_pose 仍然是有效的帧级目标

2. **确保回放的初始状态一致**
   - 物体初始位置必须与记录时完全一致
   - 机械臂初始关节角必须一致（不仅是末端位姿）

3. **考虑添加关节角记录**
   - 同时记录 `joint_pos[t]`，回放时可用关节位置控制代替 IK
   - 这完全消除 IK 的路径依赖问题

**最终推荐方案**: 方案 A + 关节角记录。在脚本 1 中同时保存 `next_ee_pose` 和
`joint_positions`，脚本 3 时间反转后使用 next_ee_pose 作为 action label，
如需精确回放则使用关节角直接控制。

---

## 8. 总结

| 项目 | 当前状态 | 问题 |
|------|---------|------|
| action 含义 | FSM 段级 waypoint | 不是 next-frame ee_pose |
| B replay | 成功但轨迹偏移 (mean=0.063m) | IK 路径依赖 |
| A replay | 失败 (dist=0.158m) | 反向运动误差累积 |
| 反转 action | compute_forward_goal_actions 重算 | 语义不完全正确 |
| ee_pose 反转 | 完全正确 (diff=0.0) | — |

**核心问题**: action 不是 next-frame ee_pose，而是远程 FSM waypoint，
导致回放和时间反转都无法精确复现原始轨迹。

**解决方案**: 将 action 改为记录 next-frame ee_pose（方案 A），
可选添加关节角记录以实现精确回放。
