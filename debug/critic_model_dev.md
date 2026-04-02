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
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    
    # === Vision encoder 权重策略 ===
    freeze_vision_encoder: bool = True   # True: 冻结 visual encoder（仅训练 UNet）
                                          # False: 从头训练或微调
    action_model_checkpoint: str | None = None  # action model checkpoint 路径
                                                 # 非 None 时从中加载 rgb_encoder 权重
    
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
        if self.freeze_vision_encoder and self.action_model_checkpoint is None:
            import warnings
            warnings.warn(
                "freeze_vision_encoder=True but action_model_checkpoint is None. "
                "Vision encoder will be randomly initialized and frozen (likely not useful)."
            )
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
- `freeze_vision_encoder`: 是否冻结 visual encoder
- `action_model_checkpoint`: action model 权重路径

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
    freeze_vision_encoder=False,  # debug 时不冻结（无预训练权重）
    action_model_checkpoint=None,
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
- [x] `CriticConfig(freeze_vision_encoder=True, action_model_checkpoint=None)` 触发 warning
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
Action Policy:
  DiffusionPolicy  →  DiffusionModel       →  DiffusionConditionalUnet1d
                       (rgb_encoder,            (纯 UNet forward)
                        obs encoding,
                        loss 计算)

Critic (对称结构):
  CriticPolicy     →  CriticModel          →  DiffusionConditionalUnet1d  (复用)
                       (rgb_encoder 复用/冻结,   (final_conv 输出 1 维)
                        obs+img encoding,
                        value loss)
```

### 3.2 新建 `CriticModel` 外层封装

**文件**: `src/rev2fwd_il/models/critic_model.py`（在现有文件中新增）

CriticModel 需要处理双相机图像输入 + proprio state，通过 visual encoder 提取图像特征后拼接为 global_cond 传入 UNet。

```python
class CriticModel(nn.Module):
    """外层封装，负责视觉编码 + obs encoding + value loss 计算。
    
    类似 lerobot 的 DiffusionModel 对 DiffusionConditionalUnet1d 的封装。
    支持从 action model checkpoint 加载并冻结 visual encoder 权重。
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        # ====== 构建 visual encoder（从 input_features 自动导出相机数量） ======
        global_cond_dim = config.robot_state_feature.shape[0]  # proprio（从 input_features 自动导出）
        
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                self.visual_feature_dim = encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                self.visual_feature_dim = self.rgb_encoder.feature_dim * num_images
            global_cond_dim += self.visual_feature_dim
        else:
            self.rgb_encoder = None
            self.visual_feature_dim = 0
        
        # ====== 加载 action model 的 visual encoder 权重 ======
        if config.action_model_checkpoint is not None and self.rgb_encoder is not None:
            self._load_vision_weights_from_action_model(config.action_model_checkpoint)
        
        # ====== 冻结 visual encoder ======
        if config.freeze_vision_encoder and self.rgb_encoder is not None:
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False
            print(f"[CriticModel] Visual encoder frozen "
                  f"({sum(p.numel() for p in self.rgb_encoder.parameters())} params)")
        
        # ====== UNet (value head) ======
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=global_cond_dim * config.n_obs_steps,
        )
    
    def _load_vision_weights_from_action_model(self, checkpoint_path: str):
        """从 action model checkpoint 中提取 rgb_encoder 权重并加载。
        
        Action model 的权重结构 (DiffusionPolicy state_dict):
            diffusion.rgb_encoder.backbone.0.weight
            diffusion.rgb_encoder.pool...
            diffusion.rgb_encoder.out.weight
            ...
        
        Critic 中 rgb_encoder 的结构与 action model 完全一致
        （使用相同的 DiffusionRgbEncoder 类 + 相同的 config 参数），
        因此可以直接匹配 key name。
        """
        import re
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # 处理可能的 wrapper key（如 "model." 前缀）
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # 提取 rgb_encoder 相关的 key
        # action model: "diffusion.rgb_encoder.{...}" → critic: "rgb_encoder.{...}"
        prefix = "diffusion.rgb_encoder."
        vision_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]  # strip prefix
                vision_sd[new_key] = v
        
        if len(vision_sd) == 0:
            raise ValueError(
                f"No rgb_encoder weights found in checkpoint: {checkpoint_path}. "
                f"Available keys (first 10): {list(state_dict.keys())[:10]}"
            )
        
        # 加载权重（strict=False 允许 encoder list vs single encoder 的差异）
        missing, unexpected = self.rgb_encoder.load_state_dict(vision_sd, strict=False)
        print(f"[CriticModel] Loaded vision encoder from: {checkpoint_path}")
        print(f"  matched keys: {len(vision_sd) - len(unexpected)}")
        if missing:
            print(f"  missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  unexpected keys: {unexpected[:5]}...")
    
    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """编码图像特征 + proprio，拼接为 global conditioning。
        
        与 DiffusionModel._prepare_global_conditioning 逻辑一致。
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]  # (B, n_obs, state_dim)
        
        if self.rgb_encoder is not None and "observation.images" in batch:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch["observation.images"], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat([
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ])
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)",
                    b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)",
                    b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)
        
        # (B, n_obs, state_dim + visual_feat_dim) → (B, n_obs * total_dim)
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
    
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Args:
            batch:
                "observation.state":  (B, n_obs, state_dim)
                "observation.images": (B, n_obs, num_cameras, C, H, W)  — 可选
                "action":             (B, T, action_dim)
        Returns:
            pred_value: (B, T, 1)
        """
        global_cond = self._prepare_global_conditioning(batch)
        
        action = batch["action"]
        timestep = torch.zeros(action.shape[0], device=action.device).long()
        
        pred_value = self.unet(action, timestep, global_cond=global_cond)
        return pred_value
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        batch 须包含:
            "action":              (B, T, action_dim)
            "observation.state":   (B, n_obs, state_dim)
            "observation.images":  (B, n_obs, num_cameras, C, H, W)  — 可选
            "bellman_value":       (B, T)
            "action_is_pad":       (B, T)      — 可选
        """
        pred_value = self.forward(batch)
        target_value = batch["bellman_value"]
        
        if self.config.value_loss_type == "mse":
            loss = F.mse_loss(pred_value.squeeze(-1), target_value, reduction="none")
        elif self.config.value_loss_type == "huber":
            loss = F.huber_loss(pred_value.squeeze(-1), target_value, reduction="none")
        
        if "action_is_pad" in batch:
            loss = loss * (~batch["action_is_pad"]).float()
        
        return loss.mean()
    
    def get_trainable_parameters(self):
        """返回需要更新的参数（排除冻结的 visual encoder）。"""
        if self.config.freeze_vision_encoder and self.rgb_encoder is not None:
            # 只返回 UNet 参数
            return self.unet.parameters()
        else:
            return self.parameters()
```

### 3.2.1 Visual Encoder 权重复用方案详解

**方案 A（首选）: 从 action model checkpoint 加载并冻结**

```
Action model checkpoint (DiffusionPolicy state_dict)
    │
    ├── diffusion.rgb_encoder.backbone.0.weight    ──┐
    ├── diffusion.rgb_encoder.backbone.0.bias       │
    ├── diffusion.rgb_encoder.pool.*                 ├──→  CriticModel.rgb_encoder
    ├── diffusion.rgb_encoder.out.weight             │     (冻结, requires_grad=False)
    ├── diffusion.rgb_encoder.out.bias              ──┘
    │
    ├── diffusion.unet.*                            ✗ 不加载
    └── diffusion.noise_scheduler.*                 ✗ 不加载
```

**关键前提**: CriticConfig 的 vision backbone 参数（`vision_backbone`, `crop_shape`, `use_group_norm`, `spatial_softmax_num_keypoints` 等）必须与 action model 训练时的 `DiffusionConfig` 一致，否则模型结构不匹配，`load_state_dict` 会失败。

**使用方式**:
```python
# 直接从 action model 的 config 复制 input_features + vision 参数
config = CriticConfig(
    input_features=action_config.input_features,  # 含 STATE + VISUAL entries
    action_dim=action_config.action_feature.shape[0],
    # vision backbone 参数也需与 action model 一致
    vision_backbone=action_config.vision_backbone,     # "resnet18"
    crop_shape=action_config.crop_shape,               # (84, 84)
    use_group_norm=action_config.use_group_norm,       # True
    spatial_softmax_num_keypoints=action_config.spatial_softmax_num_keypoints,  # 32
    use_separate_rgb_encoder_per_camera=action_config.use_separate_rgb_encoder_per_camera,
    # critic 特有参数
    freeze_vision_encoder=True,
    action_model_checkpoint="path/to/action_model/pretrained_model/",
)
critic = CriticModel(config)  # 自动加载 + 冻结
```

**方案 B（备选）: 独立训练新的 vision backbone**

```python
config = CriticConfig(
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    action_dim=7,
    freeze_vision_encoder=False,
    action_model_checkpoint=None,
    # 可选择 ImageNet 预训练权重作为起点
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
)
critic = CriticModel(config)  # 从 ImageNet 或随机初始化，全参数训练
```

**方案选择依据**:
| 场景 | 推荐方案 |
|------|----------|
| action model 已收敛，visual encoder 质量好 | A（冻结复用） |
| action model 和 critic 需要关注不同图像特征 | B（独立训练） |
| 显存不足，需减少训练参数量 | A（冻结复用） |
| 从零开始训练全系统 | B + ImageNet 预训练 |

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

- [ ] `CriticModel` 可独立实例化（含双相机 visual encoder），`forward` 和 `compute_loss` 维度正确
- [ ] `CriticModel.compute_loss` 梯度回传正常（backward 无报错）
- [ ] 冻结模式：验证 `rgb_encoder` 参数的 `requires_grad == False`，优化器只更新 UNet
- [ ] 权重加载：从真实 action model checkpoint 加载 `rgb_encoder`，`load_state_dict` 无 missing key
- [ ] `get_trainable_parameters()` 返回的参数数量 < `self.parameters()` 的总参数数量（冻结模式下）
- [ ] 在小批量数据上 overfit 测试（含图像输入）：loss 可收敛
- [ ] `debug/critic_debug.py` 更新为测试 `CriticModel`（含双相机图像输入）

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
├── critic_config.py          # (新建) CriticConfig dataclass（含 input_features + vision backbone + 冻结策略参数）
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
| **3** | Critic loss + visual encoder 完整实现 | Phase 1 + 2 | `CriticModel`（含 rgb_encoder 冻结/复用）可独立训练 |
| **4** | Value 加权接入 action loss | Phase 2 (需要 bellman_value 数据) | action policy value-guided 训练 |

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