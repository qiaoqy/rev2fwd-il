# Critic Model 开发计划

## 背景

基于 lerobot 中 Diffusion Policy 的 `DiffusionConditionalUnet1d` 架构，构建了一个 critic model。该模型复用 UNet encoder-decoder 结构，但将 `final_conv` 的输出维度改为 1，使其输出 trajectory 的 value 而非 denoised action。

### 当前进度

- [x] `DiffusionConditionalUnet1d` 已改造为 critic 网络，`final_conv` 输出 `(B, T, 1)` 维度的 value
- [x] `debug/critic_debug.py` 验证维度正确：输入 `(B=4, T=16, action_dim=7)` → 输出 `(B, T, 1)`
- [x] `compute_loss` 有占位实现（简单 MSE）

### 关键文件

| 文件 | 角色 |
|------|------|
| `src/rev2fwd_il/models/critic_model.py` | critic UNet 实现（从 lerobot 复制并修改） |
| `lerobot/.../diffusion/modeling_diffusion.py` | action policy 的 diffusion model，需要接入 value 加权 |
| `lerobot/.../diffusion/configuration_diffusion.py` | `DiffusionConfig`，critic config 的参考 |
| `src/rev2fwd_il/data/episode.py` | `Episode` dataclass，含 `success: bool` |
| `scripts/scripts_pick_place/4_train_diffusion.py` | 训练脚本，含 `load_episodes_from_npz()` 和数据转换 |
| `debug/critic_debug.py` | 维度验证脚本 |

---

## Phase 1: Config 重构 + action_feature shape 修复

> 先理清 config 和 shape 传递，确保架构干净。这是后续所有工作的基础。

### 1.1 新建 `CriticConfig` 类

**文件**: `src/rev2fwd_il/models/critic_config.py`（新建）

**当前问题**:
- critic model 直接复用 `DiffusionConfig`，但 critic 不需要 noise scheduler、vision backbone 等参数
- `debug/critic_debug.py` 中手动构造 `DiffusionConfig(input_features={...})` 且 `action_feature=7` 硬编码

**设计**:

```python
from dataclasses import dataclass, field

@dataclass
class CriticConfig:
    """Critic model configuration.
    
    只保留 critic 所需的参数子集，不继承 DiffusionConfig。
    """
    # === 输入特征 ===
    action_dim: int = 7          # action 维度，UNet 输入通道数
    state_dim: int = 7           # observation.state 维度，用于 global conditioning
    
    # === UNet 结构参数（与 DiffusionConfig 对齐） ===
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    
    # === Trajectory 参数 ===
    horizon: int = 16            # action sequence 长度 T
    n_obs_steps: int = 2         # observation 窗口大小
    
    # === Critic 特有参数 ===
    gamma: float = 0.99          # Bellman return discount factor
    value_loss_type: str = "mse" # "mse" | "huber"
    
    # === 训练参数 ===
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    
    def __post_init__(self):
        # 校验 horizon 与 UNet downsampling 的兼容性
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                f"horizon ({self.horizon}) must be divisible by "
                f"downsampling_factor ({downsampling_factor}, from len(down_dims)={len(self.down_dims)})"
            )
        if self.value_loss_type not in ("mse", "huber"):
            raise ValueError(f"value_loss_type must be 'mse' or 'huber', got '{self.value_loss_type}'")
```

**从 `DiffusionConfig` 中剔除的参数**:
- noise scheduler: `noise_scheduler_type`, `num_train_timesteps`, `beta_*`, `prediction_type`, `clip_sample*`
- vision backbone: `vision_backbone`, `crop_shape`, `pretrained_backbone_weights`, `spatial_softmax_num_keypoints`
- 推理: `num_inference_steps`, `n_action_steps`
- normalization: `normalization_mapping`

### 1.2 修复 `action_feature` shape 传递

**文件**: `src/rev2fwd_il/models/critic_model.py`

**当前代码** (L592):
```python
def __init__(self, config: DiffusionConfig, global_cond_dim: int, action_feature=7):
    ...
    in_out = [(action_feature, config.down_dims[0])] + list(...)
```

**修改方案**:
```python
def __init__(self, config: CriticConfig, global_cond_dim: int):
    ...
    # 从 config 获取 action_dim，不再硬编码
    in_out = [(config.action_dim, config.down_dims[0])] + list(...)
```

**同步修改点**:
1. `__init__` 签名：移除 `action_feature=7` 参数，改为从 `config.action_dim` 读取
2. `final_conv` 输出维度保持 1（value 输出，这是正确的）
3. `DiffusionConditionalUnet1d` 的 `config` 类型注解从 `DiffusionConfig` 改为 `CriticConfig`

### 1.3 更新 debug 脚本验证

**文件**: `debug/critic_debug.py`

```python
from src.rev2fwd_il.models.critic_config import CriticConfig
from src.rev2fwd_il.models.critic_model import DiffusionConditionalUnet1d

config = CriticConfig(
    action_dim=7,
    state_dim=10,
    horizon=16,
)

net = DiffusionConditionalUnet1d(config=config, global_cond_dim=0)
action_x = torch.randn(4, 16, 7)  # (B, T, action_dim)
timestep = torch.tensor([4])
out_y = net(action_x, timestep=timestep)
assert out_y.shape == (4, 16, 1), f"Expected (4, 16, 1), got {out_y.shape}"
print("Phase 1 验证通过 ✓")
```

### 1.4 验收标准

- [ ] `CriticConfig` 可独立实例化，`__post_init__` 校验通过
- [ ] `DiffusionConditionalUnet1d` 接受 `CriticConfig`，无 `action_feature` 硬编码
- [ ] `debug/critic_debug.py` 使用新 config 运行通过，输出 shape `(B, T, 1)`

---

## Phase 2: Bellman Return 标注

> 从稀疏 reward 计算 value label，为 critic 训练准备数据。

### 2.1 理解现有数据结构

**Episode 数据源** (`src/rev2fwd_il/data/episode.py`):
```python
@dataclass
class Episode:
    obs: np.ndarray       # (T, obs_dim)
    ee_pose: np.ndarray   # (T, 7)
    obj_pose: np.ndarray  # (T, 7)
    gripper: np.ndarray   # (T,)
    place_pose: np.ndarray  # (7,)
    goal_pose: np.ndarray   # (7,)
    success: bool = False   # ← 唯一的 reward 信号
```

**NPZ 加载** (`scripts/scripts_pick_place/4_train_diffusion.py` L406):
```python
def load_episodes_from_npz(path, num_episodes=-1) -> list[dict]:
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    # episode dict 包含: images, ee_pose, obj_pose, action, gripper, ...
    # 注意：NPZ 格式的 episode 是 dict，不是 Episode dataclass
```

**关键观察**:
- NPZ episode dict 中 `success` 字段可能存在也可能不存在（取决于数据收集方式）
- `Episode` dataclass 有 `success` 字段，但 NPZ dict 不一定有
- 需要确认目标数据（rollout data）的具体 key 结构

### 2.2 实现 Bellman Return 计算

**文件**: `src/rev2fwd_il/data/value_labeling.py`（新建）

```python
import numpy as np

def compute_bellman_returns(
    episodes: list[dict],
    gamma: float = 0.99,
    success_reward: float = 1.0,
) -> list[np.ndarray]:
    """为每个 episode 计算 discounted Bellman return。
    
    Reward 定义（稀疏）:
      - 成功 episode: r(T) = success_reward, r(t<T) = 0
      - 失败 episode: r(t) = 0 for all t
    
    Bellman return（从末尾反向计算）:
      V(T) = r(T)
      V(t) = r(t) + γ * V(t+1)
    
    对于稀疏 reward，等价于:
      - 成功 episode: V(t) = γ^(T-t) * success_reward
      - 失败 episode: V(t) = 0
    
    Args:
        episodes: list of episode dicts，每个 dict 须含 "action" (T, action_dim) 和 "success" (bool)
        gamma: discount factor
        success_reward: 成功时终止步的 reward 值
    
    Returns:
        list of np.ndarray，每个 shape (T,)，与对应 episode 的 action 长度相同
    """
    bellman_values = []
    for ep in episodes:
        T = len(ep["action"])
        success = ep.get("success", False)
        
        if not success:
            bellman_values.append(np.zeros(T, dtype=np.float32))
            continue
        
        # 成功 episode：从末尾反向计算
        values = np.zeros(T, dtype=np.float32)
        values[-1] = success_reward
        for t in range(T - 2, -1, -1):
            values[t] = gamma * values[t + 1]
        
        bellman_values.append(values)
    
    return bellman_values
```

### 2.3 集成到数据管线

**修改文件**: `scripts/scripts_pick_place/4_train_diffusion.py` — `convert_npz_to_lerobot_format()`

在 LeRobot dataset 的 features 中新增 `"bellman_value"` 字段:

```python
features = {
    ...,
    "bellman_value": {
        "dtype": "float32",
        "shape": (1,),         # 每个 frame 一个标量 value
        "names": ["value"],
    },
}
```

在逐帧写入时，从预计算的 `bellman_values[ep_idx][t]` 取值:

```python
# 在 convert 前预计算
from rev2fwd_il.data.value_labeling import compute_bellman_returns
bellman_values = compute_bellman_returns(episodes, gamma=0.99)

# 逐帧写入
for t in range(T):
    frame = {
        ...,
        "bellman_value": np.array([bellman_values[ep_idx][t]], dtype=np.float32),
    }
    dataset.add_frame(frame)
```

### 2.4 Padding 处理

lerobot 的 `load_previous_and_future_frames` 会对 episode 边界做 copy-padding，生成 `action_is_pad: (B, horizon)` mask。`bellman_value` 也需要同样处理：
- padded 区域的 bellman_value 设为 0
- 在 loss 计算时用 `action_is_pad` mask 排除

### 2.5 验收标准

- [ ] `compute_bellman_returns()` 单元测试：成功 episode value 单调递增（从 0 到 1），失败 episode 全零
- [ ] 示例：T=100, γ=0.99, success → `V(0)=0.99^99≈0.370`, `V(99)=1.0`
- [ ] NPZ → LeRobot 转换后，dataset 中每个 frame 包含 `bellman_value` 字段
- [ ] 数据可视化：抽几个 episode 画 value 曲线，确认形状合理

---

## Phase 3: Critic Loss 完整实现

> 重新设计 compute_loss，使 critic 可以独立训练。

### 3.1 架构层级重构

**当前问题**: `compute_loss` 直接放在 `DiffusionConditionalUnet1d` 内部，但 UNet 应该只负责 forward。

**目标架构** (类比 action policy):

```
DiffusionPolicy  →  DiffusionModel  →  DiffusionConditionalUnet1d
                     (obs encoding,      (纯 UNet forward)
                      loss 计算)  

CriticPolicy     →  CriticModel     →  DiffusionConditionalUnet1d  (复用)
                     (obs encoding,      (final_conv 输出 1 维)
                      value loss)
```

### 3.2 新建 `CriticModel` 外层封装

**文件**: `src/rev2fwd_il/models/critic_model.py`（在现有文件中新增）

```python
class CriticModel(nn.Module):
    """外层封装，负责 obs encoding + value loss 计算。
    
    类似 lerobot 的 DiffusionModel 对 DiffusionConditionalUnet1d 的封装。
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        # global conditioning = obs_state * n_obs_steps
        global_cond_dim = config.state_dim * config.n_obs_steps
        
        # 复用 critic 版 UNet（final_conv 输出 1 维）
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
    
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch: 包含 "observation.state" (B, n_obs, state_dim) 和 "action" (B, T, action_dim)
        Returns:
            pred_value: (B, T, 1)
        """
        # 1. 准备 global conditioning
        obs_state = batch["observation.state"]          # (B, n_obs, state_dim)
        global_cond = obs_state.flatten(start_dim=1)    # (B, n_obs * state_dim)
        
        # 2. critic 不做 diffusion，timestep 固定为 0
        action = batch["action"]                        # (B, T, action_dim)
        timestep = torch.zeros(action.shape[0], device=action.device).long()
        
        # 3. UNet forward
        pred_value = self.unet(action, timestep, global_cond=global_cond)  # (B, T, 1)
        return pred_value
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        batch 须包含:
            "action":             (B, T, action_dim)
            "observation.state":  (B, n_obs, state_dim)
            "bellman_value":      (B, T)
            "action_is_pad":      (B, T)      — 可选
        """
        pred_value = self.forward(batch)                # (B, T, 1)
        target_value = batch["bellman_value"]            # (B, T)
        
        # 选择 loss 函数
        if self.config.value_loss_type == "mse":
            loss = F.mse_loss(pred_value.squeeze(-1), target_value, reduction="none")
        elif self.config.value_loss_type == "huber":
            loss = F.huber_loss(pred_value.squeeze(-1), target_value, reduction="none")
        
        # padding mask
        if "action_is_pad" in batch:
            loss = loss * (~batch["action_is_pad"]).float()
        
        return loss.mean()
```

### 3.3 UNet 内部 `compute_loss` 的处理

将 `DiffusionConditionalUnet1d.compute_loss` 标记为 deprecated 或直接删除，所有 loss 计算移到 `CriticModel` 层。

### 3.4 Timestep 设计决策

Critic 不做 denoising，但 UNet 结构中有 `diffusion_step_encoder`，需要决策如何处理 timestep 输入：

| 方案 | 描述 | 优缺点 |
|------|------|---------|
| A. 固定 timestep=0 | 所有输入都视为 clean action | 简单，但浪费了 timestep embedding 的表达能力 |
| B. 随机采样 timestep | 训练时随机，推理时用 0 | 可做数据增强，但语义不明确 |
| C. 移除 timestep | 将 diffusion_step_encoder 替换为固定 embedding | 架构最干净，但需额外改动 |

**建议**：Phase 3 先用方案 A（最简），后续根据实验效果决定是否改为 C。

### 3.5 验收标准

- [ ] `CriticModel` 可独立实例化，`forward` 和 `compute_loss` 维度正确
- [ ] `CriticModel.compute_loss` 梯度回传正常（backward 无报错）
- [ ] 在小批量数据上 overfit 测试：loss 可收敛到接近 0
- [ ] `debug/critic_debug.py` 更新为测试 `CriticModel`（而非直接测 UNet）

---

## Phase 4: Value 加权接入 Action Policy Loss

> 将 critic 输出的 value 作为权重接入 diffusion policy 的 action loss，实现 value-guided 训练。

### 4.1 修改位置

**文件**: `/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py`
— `DiffusionModel.compute_loss` (约 L302-370)

**已有 TODO 标记**:
```python
# L315: #TODO : "bellman_value": (B, 1)
# L369: #TODO : element wise product the value to the action when action loss is calculated.
```

### 4.2 实现方案

**step 1**: `DiffusionConfig` 新增 flag

```python
# configuration_diffusion.py
use_value_weighting: bool = False    # 是否启用 value 加权
value_weight_epsilon: float = 0.01   # 防止 weight=0 的最小值
```

**step 2**: `DiffusionModel.compute_loss` 中新增加权逻辑

```python
# modeling_diffusion.py — DiffusionModel.compute_loss
loss = F.mse_loss(pred, target, reduction="none")  # (B, horizon, action_dim)

# 1. 已有的 padding mask
if self.config.do_mask_loss_for_padding:
    in_episode_bound = ~batch["action_is_pad"]
    loss = loss * in_episode_bound.unsqueeze(-1)

# 2. 新增：value 加权
if self.config.use_value_weighting and "bellman_value" in batch:
    value_weight = batch["bellman_value"]         # (B, horizon)
    # 加 epsilon 防止 weight=0 导致梯度完全消失
    value_weight = value_weight + self.config.value_weight_epsilon
    loss = loss * value_weight.unsqueeze(-1)      # broadcast → (B, horizon, action_dim)

return loss.mean()
```

### 4.3 batch 中 `bellman_value` 的来源

有两种方式，需要选择：

| 方式 | 描述 | 优缺点 |
|------|------|---------|
| A. 预计算存入 dataset | Phase 2 中已在 LeRobot dataset 中存入 `bellman_value` | 简单，但 γ 固定，改 γ 需重新转换数据 |
| B. 训练时 critic 在线推理 | 每个 batch 用 frozen critic model 推理 value | 灵活，但训练时多一次 forward pass |

**建议**：Phase 4 先用方式 A（预计算），后续如需在线更新 critic 再切换到方式 B。

### 4.4 语义分析

value 加权的效果：
- **成功 episode 的末尾步** (V≈1): action loss 权重最大 → 强化学习最终成功的 action
- **成功 episode 的开头步** (V≈γ^T): action loss 权重较小 → 对起始 action 的约束较弱
- **失败 episode** (V=0+ε): action loss 权重 ≈ ε → 几乎忽略失败数据

这意味着：
1. 如果训练数据全部是成功 demo（纯 IL），value 加权只改变时间步权重，不改变 episode 权重
2. value 加权主要在**混合数据**（成功+失败 rollout）场景下发挥作用
3. `value_weight_epsilon` 的取值需要实验调优 — 太小则失败数据彻底被忽略，太大则加权效果不明显

### 4.5 验收标准

- [ ] `use_value_weighting=False` 时行为与原始代码完全一致（不改变现有训练）
- [ ] `use_value_weighting=True` + 全成功数据：loss 正常，训练效果与不加权接近
- [ ] `use_value_weighting=True` + 混合数据：失败 episode 的 loss 贡献显著降低
- [ ] 对比实验：相同混合数据，有/无 value 加权的 policy 成功率

---

## 文件结构规划

```
src/rev2fwd_il/models/
├── critic_model.py           # CriticModel 外层 + DiffusionConditionalUnet1d (critic 版)
├── critic_config.py          # (新建) CriticConfig dataclass
└── ...

src/rev2fwd_il/data/
├── episode.py                # Episode dataclass (已有)
├── value_labeling.py         # (新建) compute_bellman_returns()
└── ...

lerobot/.../diffusion/
├── configuration_diffusion.py  # (修改) 新增 use_value_weighting, value_weight_epsilon
└── modeling_diffusion.py       # (修改) compute_loss 中接入 value 加权

debug/
├── critic_debug.py           # (更新) 使用 CriticConfig + CriticModel
└── critic_model_dev.md       # 本文档
```

## 开发顺序总结

| Phase | 内容 | 依赖 | 预期产出 |
|-------|------|------|---------|
| **1** | Config 重构 + action_feature fix | 无 | `CriticConfig`, UNet shape 修复 |
| **2** | Bellman return 标注 | Phase 1 (需要 γ 从 config 读取) | `value_labeling.py`, dataset 含 `bellman_value` |
| **3** | Critic loss 完整实现 | Phase 1 + 2 | `CriticModel` 可独立训练 |
| **4** | Value 加权接入 action loss | Phase 2 (需要 bellman_value 数据) | action policy value-guided 训练 |