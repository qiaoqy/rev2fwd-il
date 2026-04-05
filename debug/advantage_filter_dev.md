# GAE Advantage 估计器 — 倒放 B 数据帧级过滤方案

## 背景与动机

### 数据假设

只取 B rollout 中**成功**的 episode 做倒放。倒放后的数据被视为可完成 Task A 的有效示范，但质量参差不齐 — 某些时间段的 action 可能低效（如静止、来回晃动）。目标是**在每条 episode 内部**过滤掉低质量时间段，保留高质量帧用于训练 Policy A。

### 为什么单纯用 value 过滤效果不好？

1. **Value 是绝对量，缺乏相对判断**：一条轨迹开头 V=0.3 可能只是因为离目标远（正常），而不是因为轨迹质量差。Value 只编码"离成功多远"，不编码"正在靠近还是远离成功"。
2. **分布偏移**：倒放 B 数据与 A rollout 分布不同（未经 filter/speed-adjust/z-fix），critic 在 OOD 区域可能给出漂移的绝对 value，但 **value 的变化趋势** 仍可能有意义。
3. **阈值敏感**：用 V > threshold 做硬过滤，阈值难定。高阈值丢失太多数据；低阈值保留噪声数据。

### 为什么 Advantage 更适合做帧级过滤？

**Advantage $A_t = Q(s_t, a_t) - V(s_t)$** 衡量的是"在状态 $s_t$ 执行动作 $a_t$ **比平均水平好多少**"：

- $A_t > 0$: 当前 action 使 value 增长超过预期 → 轨迹在**积极推进**
- $A_t < 0$: 当前 action 使 value 增长不及预期 → 轨迹在**浪费时间或倒退**
- $A_t \approx 0$: action 与预期持平

相比绝对 value 的优势：
1. **对分布偏移更鲁棒**：TD 差分 $\delta_t = \gamma V(t+1) - V(t)$ 消除了 baseline bias，只看**相邻帧的 value 变化**
2. **自动归一化**：advantage 天然以 0 为中心，无需手动设阈值
3. **时间步级别信号**：可以精确到"哪些时间段的 action 质量好/差"，天然支持帧级过滤
4. **与 RL literature 对齐**：AWR / MARWIL 等 offline RL 方法都用 advantage 做 policy 加权

---

## 核心思想

我们有一个训练好的 **state-action value critic** $V_\theta(o_t, a_{t:t+H})$，可以对任意 (obs, action) 对估计 value。

对每条倒放 B episode（仅成功 B），逐帧调用 critic 获取 $\hat{V}(t)$，然后：

1. 计算 **TD residual**: $\delta_t = \underbrace{r_t}_{=0} + \gamma \hat{V}(t+1) - \hat{V}(t)$
2. 计算 **GAE advantage**: $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$
3. **滑窗平滑** advantage → 识别连续低 advantage 段 → 标记为 drop
4. 输出 per-frame keep/drop mask，过滤掉低质量帧段

> **注意**：倒放 B 数据没有 GT reward 信号。我们设 $r_t = 0$（稀疏 reward 假设：非终止步 reward 为零）。此时 $\delta_t = \gamma V(t+1) - V(t)$，advantage 完全由 **value 变化率** 驱动。

### 为什么做帧级而非 episode 级过滤？

因为所有 episode 都来自成功 B rollout 的倒放，整条 episode 的最终目标是可达的。问题在于 episode 中间可能有低效时段。帧级过滤可以：
- 保留同一 episode 中高效推进的片段
- 去除静止、晃动、走弯路的片段
- 最大化利用数据（不因少量坏帧丢弃整条 episode）

### 两阶段过滤策略

直接做单阶段 advantage 过滤（v1）的问题：V(t) 到达约 0.99 后的 episode 尾部会先 plateau 再 crash，这部分数据明显无用，应直接截断而不是依赖 advantage 去过滤。

**两阶段方案**（v2）：

1. **Stage 1: Value 截断** — 在每条 episode 中找到 V(t) 首次 ≥ 0.99 的时间步，截断后面所有帧。这直接去掉"任务已完成"的尾部（~53% 帧）。
2. **Stage 2: Advantage 过滤** — 对截断后的数据计算 GAE advantage，用滑窗平滑 + 阈值过滤，保留截断数据中约 50% 的高 advantage 帧。

### 防止过滤太细碎的策略

直接用逐帧 advantage 做阈值会导致 keep/drop mask 频繁交替（锯齿状）。解决方案：

1. **滑窗平滑**：对 advantage 做 centered moving average（window=31），抹平高频噪声
2. **最小 drop 长度**（min_drop_length=30）：短于此的 drop 段自动转回 keep
3. **最小 keep 长度**（min_keep_length=30）：被 drop 段包夹的短 keep 段自动转为 drop

注：v2 的截断后 episode 长度约 300 帧，因此 min 参数从 v1 的 50 降至 30。

这确保过滤结果是"大块连续的 keep 区域 + 大块连续的 drop 区域"。

---

## 实现

### 已实现的文件

#### `src/rev2fwd_il/data/advantage_estimation.py`（已完成）

纯 numpy 后处理模块，不依赖 PyTorch，不改动 critic 模型。

**核心函数**：

| 函数 | 功能 |
|------|------|
| `compute_gae_from_values(values, γ, λ)` | 从 V(t) 序列计算 GAE advantage，支持 terminal bootstrapping |
| `compute_frame_filter_mask(advantages, ...)` | advantage → 滑窗平滑 → 阈值 → 去除短 run → per-frame keep/drop mask |
| `filter_episode_frames(episode, keep_mask)` | 应用 mask 截取帧，重新计算 action 一致性 |

**`compute_frame_filter_mask` 参数**（v2b 最终值）：

| 参数 | v1 默认 | v2b 最终 | 说明 |
|------|---------|---------|------|
| `smooth_window` | 51 | 31 | 滑窗平均窗口大小（截断后 episode 更短，窗口随之缩小） |
| `drop_threshold` | 0.0 | 0.02 | 平滑 advantage < 此值的帧为 drop 候选（需 >0 以在正 advantage 区域做区分） |
| `min_drop_length` | 50 | 30 | 短于此的 drop 段恢复为 keep |
| `min_keep_length` | 50 | 30 | 被 drop 包围的短 keep 段转为 drop |

**`filter_episode_frames` 输出**：

过滤后的 episode dict，所有时间索引的数组只保留 keep 帧。额外字段：
- `_original_length`: 原始帧数
- `_kept_length`: 保留帧数
- `_keep_ratio`: 保留比例

action 在过滤后重新计算 `action[t][:7] = ee_pose[t+1]` 以保持一致性。

#### `debug/filter_reversed_B_by_advantage.py`（已完成）

完整 pipeline 脚本，复用已有的 `eval_critic_visual.py` 中的模型加载和推理函数。

**流程（两阶段）**：
```
成功 B rollout (已倒放或现场倒放)
  → 加载 critic model
  → 逐 episode:
      Stage 1: predict_episode_values() → V(t)
               找到 V(t) ≥ 0.99 的首个时间步 → 截断 episode
      Stage 2: 对截断后数据计算 GAE advantage A(t)
               → 滑窗平滑 → keep/drop mask
               → filter_episode_frames() → 过滤后 episode
  → 保存 filtered_episodes.npz + stats JSON
  → 可视化 (视频 + 静态图)
```

**可视化内容**：

1. **Per-episode 视频** (`ep{i}/video.mp4`):
   - 双相机画面（table + wrist），被过滤的帧显示**红色边框**，保留帧显示**绿色边框**
   - 顶部状态栏：绿色=KEEP，红色=DROP
   - 下方双行曲线图（含红色竖直 cursor 标示当前帧位置，精确对齐数据区域，见 D7）：
     - Row 1: Value V(t) 曲线，背景按 keep/drop 着色
     - Row 2: 平滑后 Advantage A(t) 曲线，背景按 keep/drop 着色

2. **Per-episode 静态图** (`ep{i}/overview.png`) — 两阶段 overview：
   - Row 1: 完整 V(t) 曲线 + 截断位置（紫色虚线）+ 灰色截断区域
   - Row 2: 截断后 V(t) + keep/drop 绿红背景带
   - Row 3: 截断后 Advantage A(t) + keep/drop 绿红背景带
   - Row 4: 6 张均匀采样帧，标注 KEEP/DROP + V 和 A 值

3. **全局汇总图** (`aggregate_summary.png`)：
   - 各 episode 帧数变化条形图（原始 → 截断 → 保留）
   - Keep ratio 分布直方图（truncated vs original）
   - 截断点 vs 保留帧数散点图

**用法（v2b 推荐参数）**:
```bash
# 完整过滤 pipeline
CUDA_VISIBLE_DEVICES=0 python debug/filter_reversed_B_by_advantage.py \
    --reversed_B_path debug/data/iter3_rollout_B_reversed.npz \
    --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
    --value_truncate 0.99 \
    --gamma 0.995 --lam 0.95 \
    --smooth_window 31 --drop_threshold 0.02 \
    --min_drop_length 30 --min_keep_length 30 \
    --out_dir debug/data/filtered_reversed_B_v2b \
    --device cuda:0

# 快速可视化（指定 episode）
CUDA_VISIBLE_DEVICES=0 python debug/visualize_advantage_filter.py \
    --reversed_B_path debug/data/iter3_rollout_B_reversed.npz \
    --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
    --value_truncate 0.99 \
    --smooth_window 31 --drop_threshold 0.02 \
    --min_drop_length 30 --min_keep_length 30 \
    --episode_indices 0,1,2,10,50 \
    --out_dir debug/data/filtered_reversed_B_v2b \
    --device cuda:0
```

---

## 设计决策与讨论

### D1: Terminal value 处理

GAE 递推中最后一帧的 `next_value` 默认使用 **terminal bootstrapping**（`next_value = values[-1]`），即假设 episode 结束后 value 不变。

**原因**：成功倒放 B episode 末尾对应原始 B 的开头（物体已放置在目标位置），从 A 视角看相当于任务完成，V 应接近 1.0。如果设 `next_value = 0`，会导致最后一帧 $\delta_{T-1} = 0 - V(T-1) \approx -1$（虚假的巨大负 advantage spike）。

代码中用 `terminal_bootstrap: bool = True` 控制，默认开启。

### D2: GAE λ 的选择

| λ | 效果 | 适用场景 |
|---|------|---------|
| 0.0 | 纯 1-step TD advantage，高 bias 低 variance | critic 很准确 |
| 0.95 | 标准 GAE，平衡 bias-variance | **默认推荐** |
| 1.0 | Monte Carlo advantage，低 bias 高 variance | critic 不太准确 |

推荐从 $\lambda = 0.95$ 开始。如果 critic 在 OOD 倒放数据上 value 噪声大，可降低 $\lambda$（更信赖短程 TD）。

### D3: 滑窗参数调优

| 参数 | 过小 | 过大 |
|------|------|------|
| `smooth_window` | mask 锯齿多 | 会抹平真实的 advantage 变化，丢失信息 |
| `min_drop_length` | 留下很多零散的小 drop 段 | 本该过滤的低效段被保留 |
| `min_keep_length` | 留下很多孤立的小 keep 段 | 损失有效数据 |

建议根据 episode 长度和 control frequency 调整。截断后 episode 长度约 300 帧（v2b），因此参数从 v1 的 51/50 调整为 31/30。

#### smooth_window 参数扫描（drop_threshold=0.02, min_drop/keep ≈ sw-1）

| smooth_window | min_drop/keep | kept frames | keep % (truncated) | keep % (orig) | ratio std |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 11 | 10 | 14358 | 45.4% | 21.2% | 9.4% |
| 21 | 20 | 15638 | 49.5% | 23.1% | 11.4% |
| **31** (v2b) | **30** | **15917** | **50.3%** | **23.5%** | **12.0%** |
| 51 | 50 | 16414 | 51.9% | 24.2% | 13.5% |
| 71 | 70 | 16717 | 52.9% | 24.6% | 16.0% |

**观察**：
- **sw=11**: advantage 曲线噪声大，keep/drop 交替频繁，切分碎片化。过滤最激进（45.4%）但分段质量差。
- **sw=21**: 比 sw=11 稍平滑，碎片减少，接近 50% 目标。
- **sw=31 (v2b)**: 平滑度与分辨率的良好平衡，产生连贯的 keep/drop 大块区域，std 适中。
- **sw=51**: 非常平滑，大块连续区域，部分小 drop 段被吸收进 keep。keep 率偏高（51.9%）。
- **sw=71**: 过度平滑，advantage 曲线几乎只反映最大尺度趋势。min_keep=70 约束主导分割，std 跳升至 16%，个别 episode 达到 100% keep。

**结论**: sw=31 在当前截断后 episode 长度（~300 帧）下是合适的选择。更小窗口增加噪声碎片，更大窗口丧失时间分辨率。

**数据目录**: `debug/data/filtered_reversed_B_sw{11,21,51,71}/`

### D4: drop_threshold 调优

截断后的数据全部在 value 上升阶段，advantage 基本 >0。因此 `drop_threshold=0.0` 几乎不过滤（v2: 91.4% keep）。需要正阈值：

| threshold | keep of truncated | 说明 |
|-----------|-------------------|------|
| 0.0 | 91.4% | 基本无效 |
| 0.01 | 64.1% | 偏高 |
| **0.02** | **50.3%** | ✓ 目标值 |

可从 advantage 分布的 median（≈0.012）出发估算，但 min_length 约束会使实际 keep 率高于阈值的 raw 比例。

### D5: 与 Phase 4 (Value-Weighted Training) 的关系

帧级 advantage 过滤是**前置数据清洗步骤**，与 Phase 4 的 value-weighted training 互补：

```
成功 B rollout → 倒放
    │
    ├── Step 1: 帧级 advantage 过滤（本方案）→ 去除低质量时间段
    │
    ├── Step 2 (可选): Value-weighted training（Phase 4）→ 在清洗后的数据内用 value 加权
    │
    └── → 训练 Policy A
```

后续也可以用 advantage 直接做 AWR 加权训练（替代硬过滤）：
```python
weight = torch.exp(advantage / temperature)
weight = torch.clamp(weight, max=max_weight)
loss = (mse_loss * weight).mean()
```

### D7: 视频 cursor 同步修复

`visualize_advantage_filter.py` 中 per-episode 视频的曲线图下方有一条**竖直红色 cursor 线**，标示当前帧在 episode 时间轴上的位置。

**Bug**: 旧代码按整张图片宽度线性映射：
```python
cursor_x = int(t / T * plot_w)  # ← 错误: 忽略了 matplotlib axes margins
```
matplotlib 渲染的图片中，数据区域（axes）有左右 margin（y 轴标签、padding），cursor 按全宽映射会导致**进度条与视频帧不同步**（起始偏右、终止偏左）。

**Fix**: `render_static_curves()` 使用 `ax.get_position()` 获取 axes 在 figure 中的归一化坐标，再乘以图片像素尺寸，返回数据区域左右边界的 pixel x 坐标：
```python
pos = ax.get_position()
x_left_px  = int(pos.x0 * fig_w_px)
x_right_px = int(pos.x1 * fig_w_px)
return img, x_left_px, x_right_px
```
`create_fast_filter_video()` 中 cursor 映射改为仅在数据区域内线性插值：
```python
cursor_x = x_left_px + int(t / (T_data - 1) * (x_right_px - x_left_px))
```

**验证**: 提取 ep0 视频帧 t=0 / t=136 / t=272（共 273 帧），cursor 分别位于数据区左边界 / 中心 / 右边界，与曲线 x 轴精确对齐。

### D6: 复用现有模块

| 组件 | 需要改动？ | 说明 |
|------|-----------|------|
| **CriticModel** | ❌ 不改 | 直接调用 `forward()` 获取 predicted value |
| **CriticConfig** | ❌ 不改 | GAE 的 γ 从 config 读取 |
| **predict_episode_values()** | ❌ 不改 | 从 `eval_critic_visual.py` 复用，逐帧滑窗推理 |
| **value_labeling.py** | ❌ 不改 | Bellman return 计算不变 |
| **新增 advantage_estimation.py** | ✅ 新建 | 纯 numpy 计算，不依赖 PyTorch |
| **新增 filter pipeline 脚本** | ✅ 新建 | 组合推理 + GAE + 过滤 + 保存 + 可视化 |

**对 critic UNet 零改动**。所有新增逻辑都在 critic 推理之后的 numpy 后处理层。

---

## 文件结构

```
src/rev2fwd_il/data/
├── value_labeling.py           # (已有) compute_bellman_returns
├── advantage_estimation.py     # (新建) compute_gae_from_values, compute_frame_filter_mask,
│                               #         filter_episode_frames
└── ...

debug/
├── filter_reversed_B_by_advantage.py   # (新建) 帧级过滤完整 pipeline + 可视化
├── eval_critic_visual.py               # (已有, 复用 load_critic_model, predict_episode_values)
├── eval_critic_on_reversed_B.py        # (已有, 参考倒放 + 推理流程)
└── ...
```

---

## 实施步骤

| Step | 内容 | 状态 | 产出 |
|------|------|------|------|
| 1 | 实现 `advantage_estimation.py` | ✅ 已完成 | GAE 计算 + 帧级过滤 mask + episode 帧截取 |
| 2 | 实现 `filter_reversed_B_by_advantage.py` | ✅ 已完成 | Pipeline 脚本（含两阶段过滤 + 可视化） |
| 3 | v1 实验（单阶段 advantage 过滤） | ✅ 已完成 | 40.5% keep，但 V≥0.99 之后的数据靠 advantage 硬过滤 |
| 4 | v2b 实验（两阶段：截断 + advantage） | ✅ 已完成 | 截断后 50.3% keep，总体 23.5% keep |
| 5 | (可选) AWR 加权训练替代硬过滤 | 后续 | advantage-weighted action loss |

### 实验结果

#### v1: 单阶段 advantage 过滤
- **参数**: `smooth_window=51, drop_threshold=0.0, min_drop/keep=50`
- **结果**: 99 eps, 67820 → 27588 frames (40.5% kept)
- **问题**: V(t) 到达 0.99 后的 plateau + crash 尾部被 advantage 以 DROP 处理，但这是已知无用数据，应直接截断

#### v2b: 两阶段过滤（最终方案）
- **Stage 1**: 截断 V(t) ≥ 0.99 → 67820 → 31623 frames (46.6%)
- **Stage 2**: advantage 过滤 → 31623 → 15917 frames (50.3% of truncated)
- **整体**: 67820 → 15917 frames (23.5% of original)
- **参数**: `value_truncate=0.99, smooth_window=31, drop_threshold=0.02, min_drop/keep=30`
- **keep ratio 分布**: mean=51.7%, std=12.0%, min=15.8%, max=74.7%
- **输出**: `debug/data/filtered_reversed_B_v2b/`

**语义验证**（从 overview 可视化确认）：
- Stage 1 截断正确截在 V(t) 首次到达 0.99 处，去掉了 plateau + crash 尾部
- Stage 2 在截断后数据中：**KEEP** 对应 value 快速上升段（高 advantage），**DROP** 对应 value 缓慢爬升或近似平坦段（低 advantage）
- 过滤出的数据对应机器人**积极接近和操作物体**的时间段

#### 中间实验
- **v2** (threshold=0.0): 截断后 91.4% keep — advantage 在 value 上升段几乎全正，阈值太低
- **v2a** (threshold=0.01): 截断后 64.1% keep — 仍偏高
- **v2b** (threshold=0.02): 截断后 50.3% keep ✓

### 验收标准

- [x] `compute_gae_from_values` 对已知 value 序列的 GAE 计算结果正确
- [x] 成功 B rollout 倒放后全部推理 + 帧级过滤，输出 `filtered_episodes.npz`
- [x] KEEP 帧对应 value 快速上升段，DROP 帧对应 value 缓慢/平坦段
- [x] 过滤后的帧段是连续的大块区域（不是零散碎片）
- [x] 过滤后的数据格式与原始 NPZ episode dict 一致，action 一致性已重新计算
- [x] `filtered_episodes.npz` 可直接送入训练 pipeline

---

## 附录: GAE 数学推导速查

### 当前超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| $\gamma$ | **0.995** | 折扣因子，与 critic 训练一致 |
| $\lambda$ | **0.95** | GAE 平滑因子，平衡 bias-variance |
| $r_t$ | **0**（所有 $t$） | 稀疏 reward 假设：非终止步 reward 为零 |
| terminal bootstrap | **True** | 末帧 $V(T) = V(T{-}1)$，避免虚假负 spike |

### 一般形式

**TD residual**（单步时序差分误差）:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

含义：在状态 $s_t$ 执行 action 后，实际获得 $r_t + \gamma V(s_{t+1})$，与 critic 预测 $V(s_t)$ 的差值。正值表示这一步比预期好，负值表示比预期差。

**GAE advantage**（广义优势估计，$\lambda \in [0,1]$）:
$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}$$

含义：对当前及未来所有 TD residual 做指数衰减加权求和。$\lambda$ 控制看多远——$\lambda=0$ 只看一步（纯 TD），$\lambda=1$ 看到 episode 结束（纯 MC）。

**等价递推形式**（实现中使用，从 $t=T{-}1$ 向前计算）:
$$\hat{A}_{T-1} = \delta_{T-1}$$
$$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}, \quad t = T{-}2, T{-}3, \dots, 0$$

### 我们的具体实现表达式

代入当前超参数 $\gamma = 0.995$、$\lambda = 0.95$、$r_t = 0$、terminal bootstrap（$V(T) = V(T{-}1)$）：

**TD residual 简化为 value 差分**:
$$\delta_t = 0.995 \cdot V(t{+}1) - V(t)$$

其中末帧：$\delta_{T-1} = 0.995 \cdot V(T{-}1) - V(T{-}1) = -0.005 \cdot V(T{-}1) \approx 0$

**GAE 递推**:
$$\hat{A}_{T-1} = \delta_{T-1}$$
$$\hat{A}_t = \underbrace{\bigl[0.995 \cdot V(t{+}1) - V(t)\bigr]}_{\delta_t} + \underbrace{0.995 \times 0.95}_{= 0.94525} \cdot \hat{A}_{t+1}$$

展开前几项直观理解：
$$\hat{A}_t = \delta_t + 0.945\,\delta_{t+1} + 0.894\,\delta_{t+2} + 0.845\,\delta_{t+3} + \cdots$$

每一步 TD residual 的权重按 $0.945^l$ 衰减。大约 **13 步**后权重衰减到一半（$0.945^{13} \approx 0.49$），即 GAE 主要看前后 ~13 帧的 value 变化趋势。

### 信号含义

advantage 完全由 critic 预测的 **value 变化率** 驱动（因为 $r_t = 0$）：
- value 快速上升 $\Rightarrow$ $\delta_t \gg 0$ $\Rightarrow$ 正 advantage → 轨迹在**积极推进**
- value 缓慢上升 $\Rightarrow$ $\delta_t$ 小正值 $\Rightarrow$ 低 advantage → 轨迹在**低效推进**
- value 下降 $\Rightarrow$ $\delta_t < 0$ $\Rightarrow$ 负 advantage → 轨迹在**倒退**
- value 不变 $\Rightarrow$ $\delta_t \approx 0$ $\Rightarrow$ advantage ≈ 0 → **静止/无进展**

GAE 的多步加权使得信号比单步 $\delta_t$ 更平滑，不会因为单帧 value 噪声就产生剧烈 advantage 波动。

### 末尾帧的 GAE bias 分析

反向递推意味着越靠近 episode 末尾，能利用的未来 $\delta$ 越少：

| 帧位置 | 实际利用的 $\delta$ 数量 | 等效行为 |
|--------|------------------------|---------|
| $t = T{-}1$ | 1（仅 $\delta_{T-1}$） | 纯 1-step TD |
| $t = T{-}2$ | 2 | 短程 GAE |
| $t = T{-}k$ | $k$ | 视野逐渐扩大 |
| $t \leq T{-}13$ | ≥13（超过半衰期） | 完整 GAE 精度 |

**末尾 ~13 帧的 advantage 精度低于中间帧**（可用 $\delta$ 不足以充分平均），等效于局部 $\lambda \to 0$（退化为短程 TD）。

**为什么在我们的 pipeline 中这不构成实际问题**：

1. **Stage 1 截断已经砍掉了真正的末尾**：截断发生在 $V(t)$ 首次 $\geq 0.99$ 处。截断后的"末尾"对应 value 正在快速上升的区域，$\delta$ 本身大且正，即便只看 1-2 步也会是正 advantage，不会被误判为 DROP。

2. **Terminal bootstrap 消除虚假负 spike**：设 $V(T) = V(T{-}1)$ 使得 $\delta_{T-1} = -0.005 \cdot V(T{-}1) \approx 0$，而非灾难性的 $0 - V(T{-}1) \approx -1$。末帧 advantage $\approx 0$，不产生大的负误差。

3. **滑窗平滑掩盖末尾高方差**：`smooth_window=16` 的移动平均将末尾几帧的 advantage 与前面的帧混合，稀释单帧估计噪声。

**结论**：末尾 ~13 帧的 GAE bias 方向是偏小（因为缺少后续正 $\delta$ 的累加），但在截断后的高 value 区域，这些帧大概率仍是 KEEP（value 正在快速上升），所以 bias 不导致错误过滤。无需额外处理。
