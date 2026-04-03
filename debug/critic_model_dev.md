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
- critic model 直接复用 `DiffusionConfig`，但 critic 不需要 noise scheduler 等参数
- `debug/critic_debug.py` 中手动构造 `DiffusionConfig(input_features={...})` 且 `action_feature=7` 硬编码
- critic 需要接受图像输入（手腕相机 + 固定相机），需保留 vision backbone 参数

**设计**:

```python
from dataclasses import dataclass, field
from lerobot.configs.types import FeatureType, PolicyFeature

@dataclass
class CriticConfig:
    """Critic model configuration.
    
    使用 input_features dict 描述观测输入（与 DiffusionConfig/PreTrainedConfig 相同模式），
    可直接从 action model 的 config 复制 input_features，无需手动指定 num_cameras / image_shape。
    Properties robot_state_feature / image_features 从 input_features 自动导出。
    
    理想情况下复用 action model 已训练好的 visual encoder 权重（冻结），
    仅训练 critic UNet。备选方案：从头训练一个独立的 vision backbone。
    """
    # === 输入特征（与 DiffusionConfig 对齐，可直接复制 action_config.input_features） ===
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    action_dim: int = 7          # action 维度，UNet 输入通道数（critic 特有，不在 input_features 中）
    
    # === Vision backbone 参数（与 DiffusionConfig 对齐，便于权重复用） ===
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (128, 128)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # === Vision encoder 权重初始化策略 ===
    action_model_checkpoint: str | None = None  # action model checkpoint 路径
                                                 # 非 None 时从中加载 rgb_encoder 权重作为初始化
                                                 # 加载后不冻结，与 UNet 一起端到端训练
    
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
    
    # --- 自动导出的 properties（与 PreTrainedConfig 行为一致） ---
    
    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """从 input_features 中提取 STATE 类型的 observation.state。"""
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == "observation.state":
                return ft
        return None
    
    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """从 input_features 中提取所有 VISUAL 类型的特征。
        DiffusionRgbEncoder 内部读取 config.image_features 来获取图像 shape，
        因此此 property 使 CriticConfig 与 DiffusionRgbEncoder 直接兼容。
        """
        if not self.input_features:
            return {}
        return {k: ft for k, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}
    
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
- 推理: `num_inference_steps`, `n_action_steps`
- normalization: `normalization_mapping`

**保留的 vision backbone 参数**（与 `DiffusionConfig` 字段名完全一致，方便权重对齐）:
- `vision_backbone`, `crop_shape`, `crop_is_random`, `pretrained_backbone_weights`
- `use_group_norm`, `spatial_softmax_num_keypoints`, `use_separate_rgb_encoder_per_camera`

**新增/变更的参数**:
- `input_features`: 替代原来的 `state_dim` + `num_cameras` + `image_shape`，与 `DiffusionConfig` 使用相同的 dict 模式，可直接从 action config 复制
- `action_dim`: critic UNet 输入通道数（action 是 critic 的输入，不在 `input_features` 中）
- `action_model_checkpoint`: action model 权重路径（用于初始化 rgb_encoder，加载后不冻结，端到端训练）

**兼容性说明**: `DiffusionRgbEncoder.__init__` 内部读取 `config.image_features`（一个 dict property）来获取图像 shape。由于 `CriticConfig` 使用与 `PreTrainedConfig` 相同的 `input_features` dict + `image_features` / `robot_state_feature` properties，`DiffusionRgbEncoder(config)` 可以直接传入 `CriticConfig` 实例，**无需任何 shim 或适配代码**。

使用时只需从 action model config 复制 `input_features`：
```python
critic_config = CriticConfig(
    input_features=action_config.input_features,  # 直接复制，含 STATE + VISUAL entries
    action_dim=action_config.action_feature.shape[0],
    ...
)
```

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

需要测试两种模式：纯 UNet（无图像）、完整 CriticModel（带图像输入）。

```python
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from src.rev2fwd_il.models.critic_config import CriticConfig
from src.rev2fwd_il.models.critic_model import CriticModel

B, T = 4, 16

# ========== Test 1: UNet 维度验证（无图像，与当前测试对齐） ==========
from src.rev2fwd_il.models.critic_model import DiffusionConditionalUnet1d
config_simple = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    },
    action_dim=7,
    horizon=16,
)
net = DiffusionConditionalUnet1d(config=config_simple, global_cond_dim=0)
action_x = torch.randn(B, T, 7)
timestep = torch.tensor([4])
out_y = net(action_x, timestep=timestep)
assert out_y.shape == (B, T, 1), f"Expected ({B}, {T}, 1), got {out_y.shape}"
print("Test 1 (UNet only) 通过 ✓")

# ========== Test 2: CriticModel 带双相机图像输入 ==========
config_full = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    action_dim=7,
    horizon=16,
    n_obs_steps=2,
    action_model_checkpoint=None,  # debug 时无预训练权重，随机初始化
)
critic = CriticModel(config_full)

batch = {
    "action":            torch.randn(B, T, 7),               # (B, T, action_dim)
    "observation.state":  torch.randn(B, 2, 7),               # (B, n_obs, state_dim)
    "observation.images": torch.rand(B, 2, 2, 3, 96, 96),    # (B, n_obs, num_cameras, C, H, W)
    "bellman_value":      torch.rand(B, T),                   # (B, T)
}

pred = critic(batch)
assert pred.shape == (B, T, 1), f"Expected ({B}, {T}, 1), got {pred.shape}"
print("Test 2 (CriticModel + images) 通过 ✓")

# ========== Test 3: compute_loss 梯度验证 ==========
loss = critic.compute_loss(batch)
loss.backward()
print(f"Test 3 (loss backward) 通过 ✓, loss={loss.item():.4f}")
```

### 1.4 验收标准

- [x] `CriticConfig` 可独立实例化，`__post_init__` 校验通过
- [x] `DiffusionConditionalUnet1d` 接受 `CriticConfig`，无 `action_feature` 硬编码
- [x] debug Test 1: UNet 无图像模式输出 shape `(B, T, 1)`
- [ ] debug Test 2: CriticModel 带双相机输入输出 shape `(B, T, 1)`
- [ ] debug Test 3: `compute_loss` + `backward()` 无报错

---

## Phase 2: Bellman Return 标注

> 从稀疏 reward 计算 value label，为 critic 训练准备数据。

### 2.1 理解现有数据结构

#### 数据产生

Rollout 数据由 `scripts/scripts_pick_place_simulator/6_eval_cyclic.py` 调用 `AlternatingTester`（定义在 `scripts/scripts_pick_place/6_test_alternating.py`）收集。每个 cycle 先跑 Task A（pick→place at goal），再跑 Task B（pick from goal→place on table），分别记录 rollout episode。

每个 episode 在 rollout 过程中逐帧记录 images、ee_pose、obj_pose、action（模型输出的 goal position），最终构建为 dict（参见 `6_test_alternating.py` L1493–L1512）：

```python
episode_data = {
    "images":      np.array(images_list, dtype=np.uint8),      # (T, H, W, 3)
    "ee_pose":     np.array(ee_pose_list, dtype=np.float32),   # (T, 7) — [x, y, z, qw, qx, qy, qz]
    "obj_pose":    np.array(obj_pose_list, dtype=np.float32),  # (T, 7)
    "action":      np.array(action_list, dtype=np.float32),    # (T, 8) — [goal_x..z, qw..qz, gripper]
    "success":     success,                                     # bool
    "success_step": success_step,                               # int, 首次达成成功条件的时间步
}
# 可选字段（取决于 wrist camera 是否存在）
if wrist_images_list:
    episode_data["wrist_images"] = np.array(wrist_images_list, dtype=np.uint8)  # (T, H, W, 3)
# 可选字段（取决于 caller 是否传入）
if place_pose is not None:
    episode_data["place_pose"] = place_pose   # (7,) — Task B 的放置目标
if goal_pose is not None:
    episode_data["goal_pose"] = goal_pose     # (7,) — Task A 的放置目标
```

#### 数据保存

`AlternatingTester.save_data(out_A, out_B)` 只保存**成功** episode（参见 `6_test_alternating.py` L1962–L1970）：

```python
success_A = [ep for ep in self.episodes_A if ep["success"]]
success_B = [ep for ep in self.episodes_B if ep["success"]]
np.savez_compressed(out_A, episodes=np.array(success_A, dtype=object))
np.savez_compressed(out_B, episodes=np.array(success_B, dtype=object))
```

并行收集时每个 GPU 产生一个分片（如 `iter2_collect_A_p4.npz`），再合并为 `iter2_collect_A.npz`。

#### 数据读取

`load_episodes_from_npz()`（定义在 `scripts/scripts_pick_place/4_train_diffusion.py` L406）：

```python
def load_episodes_from_npz(path, num_episodes=-1) -> list[dict]:
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    # 校验 "action" 字段存在
    return episodes
```

`convert_npz_to_lerobot_format()` 逐帧写入 LeRobot dataset，其中 state 由 `ee_pose` (+可选 `obj_pose`/gripper) 拼接，action 直接取 `ep["action"][t]`（8 维 goal position）。

#### 实际数据示例

`debug/example_rollout.npz`（从 `exp27/iter2_collect_A_p4.npz` 复制）：

```
Top-level keys: ["episodes"]
Number of episodes: 4（全部成功，失败 episode 已在 save_data 时过滤）
Type: list[dict]

Episode 0 keys + shapes:
  images:       (858, 128, 128, 3)  uint8    — table camera RGB
  wrist_images: (858, 128, 128, 3)  uint8    — wrist camera RGB
  ee_pose:      (858, 7)            float32  — [x, y, z, qw, qx, qy, qz]
  obj_pose:     (858, 7)            float32  — 物体位姿
  action:       (858, 8)            float32  — [goal_x..z, qw..qz, gripper]
  success:      True                bool     — 是否成功
  success_step: 808                 int      — 首次满足成功条件的时间步
  place_pose:   (7,)                float32  — 放置目标位姿
  goal_pose:    (7,)                float32  — 目标位姿

其他 episode 长度: T=792, 1250, 920（变长，因 horizon 内不同时刻达成成功）
```

#### 与 `Episode` dataclass 的区别

`src/rev2fwd_il/data/episode.py` 中的 `Episode` dataclass 是早期实物机器人数据格式，字段不同（含 `obs`、`gripper` 等）。当前 simulator pipeline 全部使用 **NPZ episode dict** 格式，不使用 `Episode` dataclass。Critic 数据管线应基于 NPZ dict 格式设计。

#### Critic 需要用到的字段

| 字段 | 用途 |
|------|------|
| `action` (T, 8) | UNet 输入（critic 评估的 trajectory） |
| `ee_pose` (T, 7) | observation.state（robot proprio） |
| `images` (T, H, W, 3) | observation.image（table camera） |
| `wrist_images` (T, H, W, 3) | observation.wrist_image（wrist camera） |
| `success` (bool) | 计算 Bellman return 的 reward 信号 |
| `success_step` (int) | 精确定位成功时刻，用于稀疏 reward 赋值 |

### 2.2 实现 Bellman Return 计算

**文件**: `src/rev2fwd_il/data/value_labeling.py`（新建）

```python
import numpy as np

def compute_bellman_returns(
    episodes: list[dict],
    gamma: float = 0.995,
    success_reward: float = 1.0,
) -> list[np.ndarray]:
    """为每个 episode 计算 discounted Bellman return。
    
    Reward 定义（稀疏）:
      - 成功 episode: r(success_step) = success_reward, r(t != success_step) = 0
      - 失败 episode: r(t) = 0 for all t
    
    Value 赋值（成功 episode，success_step=S）:
      - t >= S: V(t) = success_reward（已成功，full value）
      - t <  S: V(t) = γ^(S-t) * success_reward（从成功点向前折扣）
    
    失败 episode: V(t) = 0 for all t
    
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
        
        success_step = ep.get("success_step", T - 1)
        success_step = min(success_step, T - 1)
        
        values = np.zeros(T, dtype=np.float32)
        # 成功后所有帧 value=1
        values[success_step:] = success_reward
        # 成功前从 success_step 向前折扣
        for t in range(success_step - 1, -1, -1):
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

- [x] `compute_bellman_returns()` 单元测试：成功 episode value 单调递增（从 0 到 1），成功后 value=1，失败 episode 全零
- [x] 示例：T=100, γ=0.995, success_step=80 → `V(0)=0.995^80≈0.670`, `V(80..99)=1.0`
- [x] NPZ → LeRobot 转换后，dataset 中每个 frame 包含 `bellman_value` 字段
- [x] 数据可视化：抽几个 episode 画 value 曲线，确认形状合理

---

## Phase 3: Critic Loss 完整实现

> 重新设计 compute_loss，使 critic 可以独立训练。

### 3.1 架构层级重构

**当前问题**: `compute_loss` 直接放在 `DiffusionConditionalUnet1d` 内部，但 UNet 应该只负责 forward。

**目标架构** (类比 action policy):

```
Action Policy:
  DiffusionPolicy  →  DiffusionModel       →  DiffusionConditionalUnet1d
                       (rgb_encoder,            (纯 UNet forward)
                        obs encoding,
                        loss 计算)

Critic (对称结构):
  CriticPolicy     →  CriticModel          →  DiffusionConditionalUnet1d  (复用)
                       (rgb_encoder 从 action    (final_conv 输出 1 维)
                        model 初始化后端到端训练,
                        obs+img encoding,
                        value loss)
```

### 3.2 已完成的改动

#### 3.2.1 `CriticModel` 外层封装（`src/rev2fwd_il/models/critic_model.py`）

在 `DiffusionConditionalUnet1d.forward` 之后、`DiffusionConditionalResidualBlock1d` 之前新增了 `CriticModel` 类（约 160 行），包含以下组件：

**`__init__`**:
- 从 `CriticConfig.input_features` 自动导出 `robot_state_feature` 和 `image_features`
- 构建 `DiffusionRgbEncoder`（共享 encoder 或 per-camera encoder，取决于 `use_separate_rgb_encoder_per_camera`）
- 计算 `global_cond_dim = (state_dim + visual_feature_dim) * n_obs_steps`
- 可选从 action model checkpoint 加载 rgb_encoder 权重（`config.action_model_checkpoint`）
- 构建 `DiffusionConditionalUnet1d`（critic 版，final_conv 输出 1 维）

**`_load_vision_weights_from_action_model(checkpoint_path)`**:
- 加载 action model state_dict，提取 `diffusion.rgb_encoder.*` 前缀的 key
- strip 前缀后 `load_state_dict(strict=False)` 到 `self.rgb_encoder`
- 加载后**不冻结**，与 UNet 一起端到端训练
- 打印 matched/missing/unexpected keys 供诊断

**`_prepare_global_conditioning(batch)`**:
- 与 `DiffusionModel._prepare_global_conditioning` 逻辑完全对称
- 编码 `observation.state` (B, n_obs, state_dim) + `observation.images` (B, n_obs, num_cameras, C, H, W)
- 输出 (B, global_cond_dim) 拼接向量

**`forward(batch)`**:
- 调用 `_prepare_global_conditioning` 获取 global_cond
- **固定 timestep=0**（方案 A）：critic 不做 denoising，所有输入视为 clean action
- 调用 `self.unet(action, timestep=0, global_cond=global_cond)` 返回 `(B, T, 1)`

**`compute_loss(batch)`**:
- 调用 `self.forward(batch)` 获取 `pred_value (B, T, 1)`
- 与 `batch["bellman_value"] (B, T)` 计算 per-step loss（支持 MSE / Huber）
- 可选用 `batch["action_is_pad"]` mask 排除 padded 区域
- 返回 `loss.mean()`

#### 3.2.2 删除 UNet 内部的 `compute_loss`

原来 `DiffusionConditionalUnet1d` 中有一个占位的 `compute_loss` 方法（简单 MSE），已被删除。所有 loss 计算移到 `CriticModel` 层。

#### 3.2.3 Import 路径修复

- `critic_model.py` 中 `from src.rev2fwd_il.models.critic_config import CriticConfig` → `from rev2fwd_il.models.critic_config import CriticConfig`（匹配 editable install 的包名）
- `critic_debug.py` 中同步修复

### 3.3 Timestep 设计决策

采用**方案 A：固定 timestep=0**。

Critic 不做 denoising，但 UNet 结构中保留了 `diffusion_step_encoder`。`forward` 中每次都传入 `timestep = torch.zeros(B, dtype=torch.long)`，sinusoidal embedding 输出固定不变，等效于一个可学习的 bias。

| 方案 | 描述 | 优缺点 |
|------|------|---------|
| **A. 固定 timestep=0** ✅ 已采用 | 所有输入都视为 clean action | 简单，但浪费了 timestep embedding 的表达能力 |
| B. 移除 timestep | 将 diffusion_step_encoder 替换为固定 embedding | 架构最干净，但需额外改动 |

后续根据实验效果决定是否改为方案 B。

### 3.4 已通过的测试（`debug/critic_debug.py`）

| Test | 描述 | 状态 |
|------|------|------|
| Test 1 | UNet 无图像模式输出 shape `(4, 16, 1)` | ✅ |
| Test 2 | `CriticModel` + 双相机 (96×96) 输入→输出 `(4, 16, 1)` | ✅ |
| Test 3 | `compute_loss` + `backward()` 无报错，loss=0.9223 | ✅ |
| Test 4 | 所有 266,680,609 参数 `requires_grad=True` | ✅ |
| Test 5 | 梯度流通：`rgb_encoder` 和 `unet` 均有非零 grad | ✅ |

测试配置：
```python
config_full = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    action_dim=7, horizon=16, n_obs_steps=2, crop_shape=(84, 84),
    action_model_checkpoint=None,  # 随机初始化
)
```

### 3.5 尚未完成的测试

| 测试项 | 描述 | 为什么还不能做 |
|--------|------|---------------|
| 真实权重加载 | 从实际训练好的 action model checkpoint 加载 `rgb_encoder`，验证 `load_state_dict` 无 missing key | 需要一个可用的 action model checkpoint 路径，且 CriticConfig 的 vision 参数需与 action model 的 DiffusionConfig 一致 |
| 小批量 overfit | 在小批量数据上反复训练，验证 loss 可收敛到接近 0 | 需要 Phase 2 的 `bellman_value` 标注数据，或手动构造合理的 target value |
| action_is_pad mask | 验证 padded 区域的 loss 确实被 mask 掉（pad 区域改变 target 不影响 loss） | 简单测试，可随时补充 |
| Huber loss 路径 | `value_loss_type="huber"` 下 `compute_loss` 正常工作 | 简单测试，可随时补充 |

### 3.6 验收标准

- [x] `CriticModel` 可独立实例化（含双相机 visual encoder），`forward` 和 `compute_loss` 维度正确
- [x] `CriticModel.compute_loss` 梯度回传正常（backward 无报错），rgb_encoder 和 UNet 参数均有梯度
- [ ] **权重加载**：从真实 action model checkpoint 加载 `rgb_encoder`，`load_state_dict` 无 missing key
- [x] 加载后所有参数 `requires_grad == True`（包括 rgb_encoder），优化器更新全部参数
- [ ] **在小批量数据上 overfit 测试**（含图像输入）：loss 可收敛
- [x] `debug/critic_debug.py` 更新为测试 `CriticModel`（含双相机图像输入 + 梯度流验证）

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
├── critic_model.py           # CriticModel (含 visual encoder + UNet) + DiffusionConditionalUnet1d (critic 版)
├── critic_config.py          # (新建) CriticConfig dataclass（含 input_features + vision backbone 参数）
└── ...

src/rev2fwd_il/data/
├── episode.py                # Episode dataclass (已有)
├── value_labeling.py         # (新建) compute_bellman_returns()
└── ...

lerobot/.../diffusion/
├── configuration_diffusion.py  # (修改) 新增 use_value_weighting, value_weight_epsilon
└── modeling_diffusion.py       # (修改) compute_loss 中接入 value 加权

debug/
├── critic_debug.py           # (更新) 测试 CriticModel + 双相机图像输入 + 权重加载
└── critic_model_dev.md       # 本文档
```

## 开发顺序总结

| Phase | 内容 | 依赖 | 预期产出 |
|-------|------|------|---------|
| **1** | Config 重构（含 input_features + vision）+ action_feature fix | 无 | `CriticConfig`（含 input_features + vision backbone 参数）, UNet shape 修复 |
| **2** | Bellman return 标注 | Phase 1 (需要 γ 从 config 读取) | `value_labeling.py`, dataset 含 `bellman_value` |
| **3** | Critic loss + visual encoder 完整实现 | Phase 1 + 2 | `CriticModel`（含 rgb_encoder 从 action model 初始化后端到端训练）可独立训练 |
| **4** | Value 加权接入 action loss | Phase 2 (需要 bellman_value 数据) | action policy value-guided 训练 |

## Phase 5: Critic 训练实验

> 只使用 Task A rollout 数据训练 critic model，不使用 Task B 数据。

### 5.0 Critic Model 输入输出说明

Critic model 是一个 **Q-function 形式的 value estimator**，输入同时包含 **observation** 和 **action**，输出每个时间步的 value。

#### 输入

| 字段 | Shape | 作用 | 处理方式 |
|------|-------|------|---------|
| `observation.state` | `(B, n_obs, 15)` | 机器人状态（ee_pose 7 + obj_pose 7 + gripper 1） | 拼入 global conditioning |
| `observation.images` | `(B, n_obs, 2, 3, 128, 128)` | 双相机图像（table + wrist） | 经 ResNet18 → SpatialSoftmax → 拼入 global conditioning |
| **`action`** | **`(B, T, 8)`** | **action trajectory（goal_xyz, quat, gripper）** | **作为 UNet 的主输入序列** |

**是的，action 是 critic 的核心输入之一。** 这是 critic 与 action policy 的关键区别：

- **Action policy** (DiffusionModel): UNet 输入是 noisy action，输出是 denoised action（或 noise prediction）
- **Critic model** (CriticModel): UNet 输入是 clean action trajectory，输出是每个时间步的 value

数据流：
```
observation.state ─────────────┐
                                ├─→ global_cond (B, global_cond_dim)
observation.images → ResNet18 ─┘         │
                                          ↓
action (B, T, 8) ──→ UNet1d(x=action, cond=global_cond, timestep=0) ──→ pred_value (B, T, 1)
```

- `observation` 经编码后作为 UNet 的 **global conditioning**（通过 FiLM 调制 ResBlock）
- `action` 是 UNet 的 **主序列输入**（1D 卷积沿时间维度 T 滑动）
- `timestep` 固定为 0（critic 不做 denoising，sinusoidal embedding 退化为可学习 bias）

#### 输出

| 字段 | Shape | 含义 |
|------|-------|------|
| `pred_value` | `(B, T, 1)` | 每个时间步的 predicted Bellman return |

#### 训练 target

| 字段 | Shape | 含义 |
|------|-------|------|
| `bellman_value` | `(B, T)` | Ground-truth discounted return：成功 episode 中 $t \geq S$ 时为 1.0，$t < S$ 时为 $\gamma^{(S - t)}$；失败 episode 为 0 |
| `action_is_pad` | `(B, T)` | Padding mask，padded 位置的 loss 被置零 |

#### 语义

给定当前观测 $o_t$ 和未来 action 序列 $a_{t:t+T}$，critic 预测"执行这些 action 后最终成功的折扣概率"：
- $V = 1$: 已经成功（t >= success_step），或 (obs, action) 组合大概率成功
- $V \in (0, 1)$: 接近但尚未成功，value 随距离 success_step 的远近而折扣
- $V = 0$: 失败 episode

### 5.1 DDP (DistributedDataParallel) 说明

DDP 是 PyTorch 的多 GPU 分布式训练方案（`torch.nn.parallel.DistributedDataParallel`）。核心机制：
- 每个 GPU 维护模型的完整副本
- 数据通过 `DistributedSampler` 按 GPU 数量分片，每个 GPU 只看到 1/N 的数据
- forward pass 各 GPU 独立进行
- backward pass 结束时，DDP hook 自动 all-reduce 梯度，确保所有 GPU 的模型参数保持同步
- 相比 `DataParallel`，DDP 每个 GPU 一个进程，无 GIL 瓶颈，通信效率更高

**DDP bug 修复**: 之前训练脚本中调用 `model.module.compute_loss()` 绕过了 DDP wrapper，导致 backward 时梯度同步 hook 不触发。修复为通过 `model(batch)` 走 DDP 的 `forward()`，在外部计算 loss。

### 5.2 数据概况

仅使用 Task A rollout 数据（`iter3_rollout_A_p*.npz` → `critic_A_train/test.npz`）。

| 集合 | Episodes | 成功/失败 | 总帧数 | 平均 episode 长度 | Bellman 均值 |
|------|----------|-----------|--------|-------------------|-------------|
| Train | 79 | 58✓ / 21✗ | 134,593 | 1,704 | 0.200 |
| Test | 21 | 15✓ / 6✗ | 35,868 | 1,708 | 0.207 |

### 5.3 当前训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 \
torchrun --nproc_per_node=10 --master_port=29500 \
debug/train_critic.py \
  --task A \
  --batch_size 80 \
  --num_steps 50000 \
  --lr 1e-4 \
  --grad_clip 1.0 \
  --log_freq 100 \
  --eval_freq 2000 \
  --save_freq 10000 \
  --num_workers 4 \
  --output_dir debug/data/critic_A_ddp_v1
```

### 5.4 训练配置与 Epoch 计算

| 参数 | 值 | 说明 |
|------|-----|------|
| GPUs | 10 (cuda:0-9) | DDP，每卡一个进程 |
| Global batch size | 80 | 所有 GPU 合计 |
| Per-GPU batch | 8 | 80 / 10 |
| Train samples | 134,593 | sliding window stride=1 |
| Per-GPU samples/epoch | ~13,459 | 134,593 / 10 (DistributedSampler 分片) |
| Steps/epoch | ~1,682 | 13,459 / 8 |
| Total steps | 50,000 | — |
| **Total epochs** | **~29.7** | 50,000 / 1,682 |
| LR schedule | cosine annealing | 1e-4 → 0 |
| Vision encoder init | iter_10/policy_A | 65 keys matched, 不冻结 |
| Model params | 267,142,433 | 全部可训练 |
| 训练速度 | ~3.1 it/s | — |
| 预计总时间 | ~270 min | — |

### 5.5 训练进度 (critic_A_ddp_v1)

| Step | Train Loss | Eval Loss | LR |
|------|-----------|-----------|-----|
| 100 | 0.1244 | — | 1.00e-4 |
| 500 | 0.0046 | — | 1.00e-4 |
| 1000 | 0.0041 | — | 9.99e-5 |
| 2000 | 0.0024 | **0.0255** | 9.96e-5 |
| 3000 | 0.0028 | — | 9.91e-5 |

输出目录: `debug/data/critic_A_ddp_v1/`
Tensorboard: `tensorboard --logdir debug/data/critic_A_ddp_v1/tb`

### 5.5b 训练进度 (critic_A_ddp_v2) — γ=0.995 + success tail value=1

**数据变更**（相对 v1）：
- γ: 0.99 → 0.995（更高折扣，value 向前传播更远）
- 成功后帧: V=0 → V=1（success_step 之后所有帧 value 设为 1.0）
- Bellman 均值: 0.081 → 0.200

| Step | Train Loss | Eval Loss | LR | 备注 |
|------|-----------|-----------|-----|------|
| 100 | 0.1839 | — | 1.00e-4 | |
| 500 | 0.0075 | — | 1.00e-4 | |
| 1000 | 0.0057 | — | 9.99e-5 | |
| 2000 | 0.0037 | **0.0585** | 9.96e-5 | |
| 4000 | 0.0012 | 0.0564 | 9.84e-5 | |
| 6000 | 0.0005 | 0.0560 | 9.64e-5 | |
| 10000 | 0.0003 | 0.0570 | 9.05e-5 | ckpt saved |
| 20000 | 0.0001 | 0.0576 | 5.95e-5 | ckpt saved |
| 30000 | 0.00005 | 0.0575 | 3.05e-5 | ckpt saved |
| 40000 | 0.00002 | 0.0581 | 9.55e-6 | ckpt saved |
| **50000** | **0.000009** | **0.0576** | 0 | **final** |

**观察**:
- Train loss 持续下降至 ~1e-5，说明模型容量足够拟合训练集
- Eval loss 在 step ~6000 后稳定在 ~0.057，未再改善 → **train set 已过拟合**
- 最佳 eval 出现在 step ~6000 (0.0560)，后续 eval 在 0.056~0.058 范围震荡
- 相比 v1 (eval=0.0255)，v2 eval loss 更高是因为 target value 分布变了（均值从 0.081 → 0.200）

输出目录: `debug/data/critic_A_ddp_v2/`
Tensorboard: `tensorboard --logdir debug/data/critic_A_ddp_v2/tb`

### 5.6 Critic v2 测试集评估

在测试集 (`critic_A_test.npz`, 21 episodes) 上选取 5 条数据（4 成功 + 1 失败）进行推理，可视化 predicted value vs GT bellman value。

**评估脚本**: `debug/eval_critic_visual.py`
**Checkpoint**: `debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt` (step=50000)
**输出目录**: `debug/data/eval_critic_A_v2/`

#### 定量结果

| Episode | T | Success | MAE | MSE |
|---------|------|---------|---------|---------|
| 1 | 917 | ✅ | 0.0347 | 0.0024 |
| 3 | 1400 | ✅ | 0.0269 | 0.0028 |
| 4 | 1098 | ✅ | 0.0247 | 0.0011 |
| 5 | 887 | ✅ | 0.0344 | 0.0022 |
| 0 | 3000 | ❌ | 0.1198 | 0.0692 |

#### 定性分析

**成功 episode (Ep 1, 3, 4, 5)**:
- 预测曲线整体趋势正确：从 0 单调上升至 1.0，接近 success_step 后快速上升
- 预测比 GT 偏高（尤其中段 t=300~600），MAE ~0.03
- 在接近 success_step 及之后区域（V≈1.0），预测非常准确
- 预测曲线存在噪声/锯齿，GT 是平滑指数曲线

**失败 episode (Ep 0)**:
- GT 全程为 0，但**模型在 t=500~1300 区间预测出了较高 value（最高 ~0.85）**
- 这说明该失败 episode 的 observation-action 序列与某些成功 episode 相似，critic 被"欺骗"
- t=1300 后 value 骤降并稳定在 ~0（机器人停止有意义动作后，critic 正确给出低 value）
- 这是当前模型最大的问题：**对"看起来像成功但最终失败"的 episode 区分能力不足**

#### 生成的可视化文件

每个 episode 目录下包含：
- `value_curve.png`: 3 子图（predicted vs GT 曲线、prediction error、EE xyz 轨迹）
- `overview.png`: 曲线 + 6 个均匀采样帧（标注 GT/Pred value）
- `video.mp4`: 逐帧实时视频（画面 + value bar）

汇总图：`debug/data/eval_critic_A_v2/aggregate_pred_vs_gt.png`

### 5.7 之前的实验记录

| 实验 | 目录 | 说明 | 状态 |
|------|------|------|------|
| Overfit test | `critic_A_overfit_test/` | 4 episodes, 单卡, 2000 steps | ✅ eval loss → 4.4e-5 |
| Task A full v1 | `critic_A_full_v1/` | 单卡, batch=16, 50K steps | ⚠️ 已停止 (DDP bug) |
| Task B full v1 | `critic_B_full_v1/` | 单卡, batch=16, 50K steps | ❌ 已停止 (不再训练 B) |
| **Task A DDP v1** | **`critic_A_ddp_v1/`** | **10卡 DDP, batch=80, 50K steps, γ=0.99** | **✅ 完成** |
| **Task A DDP v2** | **`critic_A_ddp_v2/`** | **10卡 DDP, batch=80, 50K steps, γ=0.995, success tail=1** | **✅ 完成, final eval=0.0576** |

## Phase 6: B Rollout 倒放 → Critic Value 标记实验

> 将 B rollout 数据直接倒放（不做任何过滤和调速处理），用训练好的 Task A critic model 对倒放后的数据标记 predicted value，可视化评估 critic 对反转 B 数据的判断能力。

### 6.1 动机

Critic model (v2) 在 Task A 测试集上表现良好（成功 episode MAE~0.03），但这些数据都是原始 A rollout。实际应用中，Policy A 的训练数据来源是 **B rollout 倒放**，而非直接的 A rollout。因此需要验证：

1. **Critic 对倒放 B 数据的泛化能力**：倒放 B 数据与原始 A rollout 在分布上有多大差异？Critic 能否正确评估？
2. **为 value-weighted training 做数据探索**：如果 critic 能合理区分好/坏的倒放轨迹，就可以用 predicted value 过滤或加权这些数据来训练 Policy A。
3. **观察未过滤/未调速数据的特征**：之前 exp27 pipeline 对 B rollout 做了 filter → speed-adjust → z-fix → reverse，本实验只做 reverse，观察 critic 对"原始质量"倒放数据的响应。

### 6.2 数据来源

| 数据 | 路径 | 说明 |
|------|------|------|
| B rollout 分片 | `debug/data/iter3_rollout_B_p{0..9}.npz` | 10 个 GPU 收集的 B rollout（每分片 10 eps） |
| 合并后 | `debug/data/iter3_rollout_B_all.npz` | 合并 100 episodes（99 成功 + 1 失败） |
| 倒放后 | `debug/data/iter3_rollout_B_reversed.npz` | 仅倒放，不过滤不调速 |
| Critic checkpoint | `debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt` | Task A critic v2 (γ=0.995, step=50000) |

### 6.3 处理流程

```
iter3_rollout_B_p{0..9}.npz
       │
       ├── 合并 → iter3_rollout_B_all.npz (100 eps)
       │
       └── 倒放 (2_reverse_to_task_A.py, success_only=0)
                  │
                  └── iter3_rollout_B_reversed.npz
                             │
                             └── Critic 推理 (eval_critic_visual.py 风格)
                                        │
                                        └── 可视化 5 条 episode (value 曲线 + overview + video)
```

**注意**：倒放时使用 `success_only=0`，保留失败 episode，观察 critic 对失败数据的响应。

### 6.4 与 exp27 pipeline 的区别

| 步骤 | Exp27 Pipeline | 本实验 |
|------|---------------|--------|
| Filter (静止帧) | ✅ | ❌ 不做 |
| Speed-adjust (加速) | ✅ (interp_factor=1) | ❌ 不做 |
| Z-offset fix | ✅ (z -= 0.02) | ❌ 不做 |
| Reverse | ✅ | ✅ |

本实验只做最后一步 reverse，保留数据原始质量，观察 critic 的判断。

### 6.5 脚本

**合并 B 分片 + 倒放 + Critic 评估可视化**: `debug/eval_critic_on_reversed_B.py`

```bash
# 合并 B 分片
python debug/eval_critic_on_reversed_B.py \
    --data_dir debug/data \
    --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
    --num_episodes 5 \
    --out_dir debug/data/eval_critic_reversed_B \
    --device cuda:0
```

### 6.6 关注点

在可视化结果中重点观察：
1. **成功 B episode 倒放后**：critic predicted value 是否呈现合理的从低到高递增趋势？
2. **失败 B episode 倒放后**：critic 是否给出低 value？
3. **与 A rollout 对比**：倒放 B 的 value 曲线形状是否与原始 A rollout 类似？
4. **静止帧/慢速帧的影响**：未过滤的数据中，静止段是否导致 critic value 出现异常平坦或跳变？

### 6.7 验收标准

- [ ] B 分片合并为 `iter3_rollout_B_all.npz` (100 eps)
- [ ] 倒放为 `iter3_rollout_B_reversed.npz` (100 eps, success_only=0)
- [ ] Critic 对 5 条倒放 episode 生成 value curve + overview + video 可视化
- [ ] 结果保存到 `debug/data/eval_critic_reversed_B/`

---

## 附录: Global Conditioning 维度计算

```
global_cond_dim = (robot_state_feature.shape[0] + visual_feature_dim) * n_obs_steps

示例（默认参数）:
  robot_state_feature.shape[0] = 7 (ee_pose, 从 input_features 自动导出)
  visual_feature_dim = spatial_softmax_num_keypoints * 2 * len(image_features)
                     = 32 * 2 * 2 = 128  (2 cameras, 从 input_features 自动导出)
  n_obs_steps = 2
  
  global_cond_dim = (7 + 128) * 2 = 270
```