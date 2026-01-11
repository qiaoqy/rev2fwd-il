# 图像归一化分析：训练 vs 推理

本文档详细分析了在训练和推理过程中，图像数据的归一化和反归一化处理流程。

## 目录
1. [训练时的图像处理](#训练时的图像处理)
2. [推理时的图像处理](#推理时的图像处理)
3. [为什么XYZ可视化中图像经过归一化](#为什么xyz可视化中图像经过归一化)
4. [Loss计算中是否包含图像Loss](#loss计算中是否包含图像loss)
5. [总结](#总结)

---

## 训练时的图像处理

### 1. 数据转换阶段 (NPZ → LeRobot Dataset)

**位置**: `scripts/31_train_A_diffusion.py` → `convert_npz_to_lerobot_format()`

```python
# 原始图像: uint8 [0, 255], shape (H, W, 3)
img = images[t]  # (H, W, 3) uint8

frame = {
    "observation.image": img,  # 直接存储 uint8 图像
    "observation.state": state.astype(np.float32),
    "action": action.astype(np.float32),
}

dataset.add_frame(frame)  # LeRobot会将图像编码为视频
```

**关键点**:
- 原始图像以 **uint8 [0, 255]** 格式存储
- LeRobot Dataset 将图像编码为 **视频文件** (MP4)
- 此时 **没有进行归一化**

### 2. 训练数据加载阶段

**位置**: LeRobot 内部的 `LeRobotDataset.__getitem__()`

当从 LeRobot Dataset 加载数据时:

```python
# LeRobot 内部处理 (伪代码)
# 1. 从视频文件解码图像
image = decode_video_frame(...)  # 返回 uint8 [0, 255]

# 2. 转换为 float32 并归一化到 [0, 1]
image = image.astype(np.float32) / 255.0  # [0, 1]

# 3. 转换为 CHW 格式
image = np.transpose(image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)

# 4. 转换为 Tensor
image = torch.from_numpy(image)  # (3, H, W) float32 [0, 1]
```

**关键点**:
- LeRobot Dataset 在加载时 **自动将图像归一化到 [0, 1]**
- 格式转换: `(H, W, 3) uint8 [0, 255]` → `(3, H, W) float32 [0, 1]`

### 3. Preprocessor 处理

**位置**: `src/rev2fwd_il/train/lerobot_train_with_viz.py` → `train_with_xyz_visualization()`

```python
# 创建 preprocessor 时配置归一化
preprocessor_overrides = {
    "normalizer_processor": {
        "stats": dataset.meta.stats,
        "features": {**policy.config.input_features, **policy.config.output_features},
        "norm_map": policy.config.normalization_mapping,
    },
}

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg.policy,
    pretrained_path=cfg.policy.pretrained_path,
    preprocessor_overrides=preprocessor_overrides,
    ...
)
```

**Normalization Mapping** (从训练日志):
```
[DEBUG] TRAINING Normalization Settings
  policy_cfg.normalization_mapping:
    FeatureType.STATE: NormalizationMode.MEAN_STD
    FeatureType.ACTION: NormalizationMode.MEAN_STD
    FeatureType.VISUAL: NormalizationMode.NONE  # 图像不再归一化!
```

**关键点**:
- **图像 (VISUAL) 的归一化模式是 `NONE`**
- 这是因为图像已经在 Dataset 加载时归一化到 [0, 1]
- Preprocessor **不会对图像进行额外的归一化**
- 只对 `observation.state` 和 `action` 进行 mean-std 归一化

### 4. 训练时的完整流程

```python
# 1. Dataset 加载 (LeRobot 内部)
batch = dataset[idx]
# batch["observation.image"]: (B, n_obs_steps, 3, H, W) float32 [0, 1]
# batch["observation.state"]: (B, n_obs_steps, state_dim) float32 (原始值)
# batch["action"]: (B, horizon, action_dim) float32 (原始值)

# 2. Preprocessor 归一化
processed_batch = preprocessor(batch)
# processed_batch["observation.image"]: (B, n_obs_steps, 3, H, W) float32 [0, 1] (不变)
# processed_batch["observation.state"]: (B, n_obs_steps, state_dim) float32 (归一化后)
# processed_batch["action"]: (B, horizon, action_dim) float32 (归一化后)

# 3. Policy forward (计算 loss)
loss, output_dict = policy.forward(processed_batch)
# loss 只计算 action 的 MSE loss，不包含图像 loss
```

---

## 推理时的图像处理

### 1. 从环境获取图像

**位置**: `scripts/41_test_A_diffusion_visualize.py` → `run_episode()`

```python
# 从 Isaac Lab 相机获取图像
table_rgb = table_camera.data.output["rgb"]  # (num_envs, H, W, 4) uint8
if table_rgb.shape[-1] > 3:
    table_rgb = table_rgb[..., :3]  # 去掉 alpha 通道
table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)  # (1, H, W, 3) uint8
table_rgb_frame = table_rgb_np[0]  # (H, W, 3) uint8 [0, 255]

# 转换为 float32 [0, 1] 并转为 BCHW 格式
table_rgb_chw = torch.from_numpy(table_rgb_frame).float() / 255.0  # uint8 -> float [0,1]
table_rgb_chw = table_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
```

**关键点**:
- 从相机获取的是 **uint8 [0, 255]** 图像
- **手动归一化到 [0, 1]**: `/ 255.0`
- 格式转换: `(H, W, 3)` → `(1, 3, H, W)`

### 2. Preprocessor 处理

```python
policy_inputs = {
    "observation.image": table_rgb_chw,  # (1, 3, H, W) float32 [0, 1]
    "observation.state": state,  # (1, state_dim) float32 (原始值)
}

# Preprocessor 归一化
if preprocessor is not None:
    policy_inputs = preprocessor(policy_inputs)
# policy_inputs["observation.image"]: (1, 3, H, W) float32 [0, 1] (不变)
# policy_inputs["observation.state"]: (1, state_dim) float32 (归一化后)
```

**关键点**:
- 推理时的 preprocessor 配置与训练时 **完全相同**
- 图像归一化模式仍然是 `NONE`
- 图像保持在 [0, 1] 范围，**不会进行额外归一化**

### 3. 推理时的完整流程

```python
# 1. 获取原始图像并归一化
table_rgb_chw = torch.from_numpy(table_rgb_frame).float() / 255.0  # [0, 1]

# 2. Preprocessor 处理
policy_inputs = preprocessor(policy_inputs)
# observation.image: [0, 1] (不变)
# observation.state: 归一化后

# 3. Policy 推理
with torch.no_grad():
    action = policy.select_action(policy_inputs)  # 归一化的 action

# 4. Postprocessor 反归一化
action = postprocessor(action)  # 反归一化到原始范围
```

---

## 为什么XYZ可视化中图像经过归一化

### 训练时的XYZ可视化

**位置**: `src/rev2fwd_il/train/lerobot_train_with_viz.py` → `extract_xyz_visualization_data()`

```python
def extract_xyz_visualization_data(
    raw_batch: dict[str, torch.Tensor],
    processed_batch: dict[str, torch.Tensor],
    ...
) -> dict:
    # 从 processed_batch 提取图像 (已经归一化到 [0, 1])
    if "observation.image" in processed_batch:
        img = processed_batch["observation.image"]  # (B, n_obs_steps, C, H, W) [0, 1]
        if img.dim() == 5:
            img_np = img[0, -1].detach().cpu().numpy()  # (C, H, W) [0, 1]
        else:
            img_np = img[0].detach().cpu().numpy()  # (C, H, W) [0, 1]
        
        # 转换为 HWC 格式并反归一化到 [0, 255]
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # [0, 1] -> [0, 255]
        viz_data["table_image"] = img_np
```

**关键点**:
- XYZ 可视化使用的是 **`processed_batch`** 中的图像
- `processed_batch` 是经过 Dataset 加载后的数据，图像已经归一化到 [0, 1]
- 为了显示，需要 **反归一化回 [0, 255]**: `* 255`

### 推理时的XYZ可视化

**位置**: `scripts/41_test_A_diffusion_visualize.py` → `run_episode()`

```python
# 推理时直接使用原始的 uint8 图像
table_rgb_frame = table_rgb_np[0]  # (H, W, 3) uint8 [0, 255]

if xyz_visualizer is not None:
    xyz_visualizer.add_frame(
        ...
        table_image=table_rgb_frame,  # 直接使用 uint8 [0, 255]
        ...
    )
```

**关键点**:
- 推理时的 XYZ 可视化使用的是 **原始的 uint8 图像**
- **不需要反归一化**，因为从相机获取的就是 uint8 格式

### 为什么训练时的可视化图像"经过归一化"

**答案**: 这是一个 **术语混淆**

- 训练时的可视化图像确实来自 **归一化后的数据** ([0, 1])
- 但在显示前会 **反归一化回 [0, 255]**
- 所以最终显示的图像和推理时的图像 **在视觉上是一样的**

**真正的区别**:
- 训练时: `Dataset [0, 1]` → `反归一化 [0, 255]` → 显示
- 推理时: `相机 [0, 255]` → 直接显示

---

## Loss计算中是否包含图像Loss

### Diffusion Policy 的 Loss 计算

**位置**: LeRobot 的 `DiffusionPolicy.forward()`

```python
def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """
    计算 Diffusion Policy 的训练 loss
    
    Args:
        batch: 包含 observation.image, observation.state, action
        
    Returns:
        loss: MSE loss between predicted and ground truth actions
        output_dict: 包含中间结果的字典
    """
    # 1. 提取输入
    obs_image = batch["observation.image"]  # (B, n_obs_steps, C, H, W) [0, 1]
    obs_state = batch["observation.state"]  # (B, n_obs_steps, state_dim) 归一化后
    action_gt = batch["action"]  # (B, horizon, action_dim) 归一化后
    
    # 2. 编码观测 (图像 + 状态)
    obs_features = self.encode_observation(obs_image, obs_state)
    
    # 3. 添加噪声到 ground truth action (diffusion forward process)
    noise = torch.randn_like(action_gt)
    timesteps = torch.randint(0, self.num_train_timesteps, (B,))
    noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
    
    # 4. 预测噪声 (diffusion model)
    noise_pred = self.unet(
        sample=noisy_action,
        timestep=timesteps,
        encoder_hidden_states=obs_features,
    )
    
    # 5. 计算 MSE loss (只针对 action)
    loss = F.mse_loss(noise_pred, noise)
    
    return loss, {"noise_pred": noise_pred, "noise_gt": noise}
```

**关键点**:
1. **图像只作为输入特征**，通过 Vision Backbone (ResNet18) 编码
2. **Loss 只计算 action 的预测误差**，不包含图像重建 loss
3. 这是一个 **条件生成模型**: 给定 (image, state)，预测 action
4. 图像通过 Vision Backbone 提取特征，这些特征用于指导 action 生成

### 为什么不计算图像 Loss?

**原因**:
1. **任务目标**: 学习从观测 (image + state) 到动作 (action) 的映射
2. **不是图像生成任务**: 不需要重建或生成图像
3. **图像是条件**: 图像作为条件输入，帮助预测更准确的 action
4. **计算效率**: 图像 loss 计算成本高，且对任务无益

### Observation Image GT 是否经过归一化?

**答案**: **是的**

```python
# 训练时的 ground truth
batch["observation.image"]  # (B, n_obs_steps, C, H, W) float32 [0, 1]
batch["observation.state"]  # (B, n_obs_steps, state_dim) 归一化后 (mean-std)
batch["action"]  # (B, horizon, action_dim) 归一化后 (mean-std)
```

**归一化流程**:
1. **图像**: Dataset 加载时归一化到 [0, 1]
2. **状态**: Preprocessor 使用 mean-std 归一化
3. **动作**: Preprocessor 使用 mean-std 归一化

---

## 🐛 问题诊断与修复

### 问题根源

通过诊断脚本 `scripts/debug_image_normalization.py` 发现：

1. **训练配置错误**:
   ```json
   "normalization_mapping": {
       "VISUAL": "MEAN_STD",  // ← 使用了 ImageNet 归一化！
       "STATE": "MIN_MAX",
       "ACTION": "MIN_MAX"
   }
   ```

2. **ImageNet 归一化参数**:
   - mean: `[0.485, 0.456, 0.406]` (RGB)
   - std: `[0.229, 0.224, 0.225]` (RGB)

3. **可视化代码问题**:
   - 从 `processed_batch` 提取图像（已经 ImageNet 归一化）
   - 直接 `* 255` 转换为 uint8，**没有反归一化**
   - 导致图像过曝和偏色

### 修复方案

**修复位置 1**: `src/rev2fwd_il/train/lerobot_train_with_viz.py`

在可视化前添加反归一化：

```python
# IMPORTANT: Reverse ImageNet normalization before visualization
# Images are normalized with: (img - mean) / std
# We need to reverse: img = normalized * std + mean
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_np = img_np * imagenet_std + imagenet_mean  # Reverse normalization

# Then convert to uint8
img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
```

**修复位置 2**: `scripts/41_test_A_diffusion_visualize.py`

添加 debug 输出验证归一化：

```python
# DEBUG: Print image stats before/after preprocessing
if t == 0:
    print(f"[DEBUG] Step 0: Image BEFORE preprocessing:")
    print(f"  Range: [{table_rgb_chw.min():.4f}, {table_rgb_chw.max():.4f}]")
    print(f"  Mean per channel: R={...}, G={...}, B={...}")
    
    # After preprocessing
    print(f"[DEBUG] Step 0: After preprocessing (ImageNet normalized):")
    print(f"  Range: [{policy_inputs['observation.image'].min():.4f}, ...]")
    print(f"  Expected after ImageNet norm: mean~0, std~1 per channel")
```

### 验证结果

运行 `scripts/test_image_normalization_fix.py`:

```
1. Original image (mid-gray):
   Mean per channel: R=0.5000, G=0.5000, B=0.5000

2. After ImageNet normalization:
   Mean per channel: R=0.0655, G=0.1964, B=0.4178

3. OLD (WRONG) visualization (normalized * 255):
   Mean per channel: R=16.0, G=50.0, B=106.0
   ⚠️  This gives wrong colors! (should be ~127 for mid-gray)

4. NEW (CORRECT) visualization (reverse norm, then * 255):
   Mean per channel: R=127.0, G=127.0, B=127.0
   ✓ Correct! Mid-gray should be ~127

5. Verification:
   ✓ PASS: Visualization is correct!
```

---

## 总结

### 图像归一化对比表（更新）

| 阶段 | 训练 | 推理 |
|------|------|------|
| **原始格式** | uint8 [0, 255] (NPZ) | uint8 [0, 255] (相机) |
| **Dataset 加载** | float32 [0, 1] (自动) | N/A |
| **手动归一化** | N/A | float32 [0, 1] (`/ 255.0`) |
| **Preprocessor** | **ImageNet MEAN_STD** ⚠️ | **ImageNet MEAN_STD** ⚠️ |
| **输入 Policy** | ImageNet 归一化 (mean~0, std~1) | ImageNet 归一化 (mean~0, std~1) |
| **XYZ 可视化** | **反 ImageNet 归一化** → [0, 1] → [0, 255] | 直接使用 [0, 255] |

### 关键结论（更新）

1. **训练和推理都使用 ImageNet 归一化** ⚠️
   - Dataset 加载: uint8 → float32 [0, 1]
   - Preprocessor: [0, 1] → ImageNet 归一化 (mean~0, std~1)
   - 归一化方式: `(img - mean) / std`
   - mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`

2. **可视化问题已修复** ✅
   - **旧代码**: 直接 `normalized_img * 255` → 过曝和偏色
   - **新代码**: 先反归一化 `img = normalized * std + mean`，再 `* 255`
   - 修复位置: `lerobot_train_with_viz.py` (4处)

3. **Loss 计算不包含图像 Loss**
   - 只计算 action 的 MSE loss
   - 图像通过 Vision Backbone 提取特征，作为条件输入

4. **Observation Image GT 经过 ImageNet 归一化**
   - 图像: ImageNet MEAN_STD 归一化
   - 状态: MIN_MAX 归一化
   - 动作: MIN_MAX 归一化

5. **推理时的归一化流程**
   - 相机 → uint8 [0, 255]
   - 手动归一化 → float32 [0, 1]
   - Preprocessor → ImageNet 归一化 (mean~0, std~1)
   - 与训练时完全一致 ✅

### 问题诊断与修复

#### 问题表现
- XYZ 可视化图像过曝
- 颜色偏移（偏蓝色）
- 对比度异常

#### 根本原因
```python
# 错误的可视化代码
img_np = processed_batch["observation.image"][0, -1]  # ImageNet 归一化后
img_np = (img_np * 255).astype(np.uint8)  # ❌ 直接 * 255，没有反归一化
```

对于 mid-gray (0.5, 0.5, 0.5):
- ImageNet 归一化后: (0.066, 0.196, 0.418)
- 错误可视化: (16, 50, 106) ← 偏蓝色，过暗
- 正确应该是: (127, 127, 127)

#### 修复方法
```python
# 正确的可视化代码
img_np = processed_batch["observation.image"][0, -1]  # ImageNet 归一化后

# ✅ 先反归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_np = img_np * imagenet_std + imagenet_mean  # 反归一化到 [0, 1]

# 然后转换为 uint8
img_np = np.transpose(img_np, (1, 2, 0))
img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
```

### 常见误解澄清（更新）

❌ **误解**: 图像归一化模式是 NONE
✅ **事实**: 图像归一化模式是 **MEAN_STD (ImageNet)**，在 config.json 中明确定义

❌ **误解**: 推理时没有归一化图像
✅ **事实**: 推理时 Preprocessor 会应用 ImageNet 归一化，与训练时一致

❌ **误解**: 可视化图像过曝是数据问题
✅ **事实**: 是可视化代码问题，没有反归一化就直接 * 255

❌ **误解**: 训练时计算图像 loss
✅ **事实**: 只计算 action 的 MSE loss，图像只作为输入特征

---

## 代码位置索引

### 训练相关
- 数据转换: `scripts/31_train_A_diffusion.py:convert_npz_to_lerobot_format()`
- Preprocessor 配置: `src/rev2fwd_il/train/lerobot_train_with_viz.py:train_with_xyz_visualization()`
- XYZ 可视化: `src/rev2fwd_il/train/lerobot_train_with_viz.py:extract_xyz_visualization_data()`

### 推理相关
- 图像获取和归一化: `scripts/41_test_A_diffusion_visualize.py:run_episode()` line 638
- Preprocessor 加载: `scripts/41_test_A_diffusion_visualize.py:load_diffusion_policy()`
- XYZ 可视化: `scripts/41_test_A_diffusion_visualize.py:run_episode()` (直接使用原始图像)

### LeRobot 内部
- Dataset 加载: LeRobot 的 `LeRobotDataset.__getitem__()`
- Policy forward: LeRobot 的 `DiffusionPolicy.forward()`
- Preprocessor: LeRobot 的 `make_pre_post_processors()`
