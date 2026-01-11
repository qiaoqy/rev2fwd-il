# 图像归一化问题修复总结

## 🐛 问题描述

在 `runs/diffusion_A_2cam_3/xyz_viz/` 文件夹下的可视化图像出现：
- **过曝** (over-exposure)
- **偏色** (color shift，偏蓝色)
- **对比度异常**

## 🔍 问题诊断

### 1. 运行诊断脚本

```bash
conda activate rev2fwd_il
python scripts/debug_image_normalization.py runs/diffusion_A_2cam_3/checkpoints/checkpoints/last/pretrained_model
```

### 2. 发现的问题

**配置文件** (`config.json`):
```json
"normalization_mapping": {
    "VISUAL": "MEAN_STD",  // ← 使用了 ImageNet 归一化！
    "STATE": "MIN_MAX",
    "ACTION": "MIN_MAX"
}
```

**ImageNet 归一化参数**:
- mean (RGB): `[0.485, 0.456, 0.406]`
- std (RGB): `[0.229, 0.224, 0.225]`

**可视化代码问题**:
```python
# 错误的代码 (旧版本)
img_np = processed_batch["observation.image"][0, -1]  # 已经 ImageNet 归一化
img_np = (img_np * 255).astype(np.uint8)  # ❌ 直接 * 255，没有反归一化
```

### 3. 问题影响

对于 mid-gray 像素 (0.5, 0.5, 0.5):

| 步骤 | R | G | B | 说明 |
|------|---|---|---|------|
| 原始值 | 0.500 | 0.500 | 0.500 | 中灰色 |
| ImageNet 归一化后 | 0.066 | 0.196 | 0.418 | (img - mean) / std |
| **错误可视化** | **16** | **50** | **106** | ❌ 直接 * 255 |
| **正确可视化** | **127** | **127** | **127** | ✅ 反归一化后 * 255 |

错误可视化导致：
- 整体偏暗（16, 50, 106 vs 127, 127, 127）
- 偏蓝色（B=106 >> R=16）
- 对比度失真

## ✅ 修复方案

### 修复位置 1: 训练可视化代码

**文件**: `src/rev2fwd_il/train/lerobot_train_with_viz.py`

**修复内容**: 在 4 个位置添加反归一化
1. `extract_xyz_visualization_data()` - table camera
2. `extract_xyz_visualization_data()` - wrist camera  
3. `extract_action_chunk_data()` - table camera
4. `extract_action_chunk_data()` - wrist camera

**修复代码**:
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

### 修复位置 2: 推理代码 (Debug 输出)

**文件**: `scripts/41_test_A_diffusion_visualize.py`

**修复内容**: 添加 debug 输出验证归一化

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

## 🧪 验证修复

### 运行测试脚本

```bash
conda activate rev2fwd_il
python scripts/test_image_normalization_fix.py
```

### 测试结果

```
================================================================================
Testing ImageNet Normalization Fix
================================================================================

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
================================================================================
```

### 可视化对比

测试脚本生成了对比图: `docs/image_normalization_comparison.png`

显示了：
- 原始图像 (Ground Truth)
- 错误可视化 (OLD - 偏蓝色、过暗)
- 正确可视化 (NEW - 与原始一致)

## 📋 后续步骤

### 1. 重新生成训练可视化

```bash
# 使用修复后的代码重新训练（生成正确的可视化）
CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_2images.npz \
    --out runs/diffusion_A_2cam_test \
    --steps 200 \
    --enable_xyz_viz
```

检查 `runs/diffusion_A_2cam_test/xyz_viz/` 中的图像是否正常。

### 2. 验证推理时的归一化

```bash
# 运行推理并查看 debug 输出
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A_2cam_3/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_2cam_3/videos_test \
    --num_episodes 1
```

检查 debug 输出，确认：
- 图像在 preprocessing 前: [0, 1]
- 图像在 preprocessing 后: mean~0, std~1 (ImageNet 归一化)

### 3. 对比新旧可视化

```bash
# 对比修复前后的可视化
ls runs/diffusion_A_2cam_3/xyz_viz/          # 旧的（有问题）
ls runs/diffusion_A_2cam_test/xyz_viz/       # 新的（修复后）
```

## 📊 技术细节

### ImageNet 归一化公式

**前向 (训练/推理)**:
```python
normalized = (img - mean) / std
```

**反向 (可视化)**:
```python
img = normalized * std + mean
```

### 完整的可视化流程

```python
# 1. 从 processed_batch 获取图像 (已 ImageNet 归一化)
img_np = processed_batch["observation.image"][0, -1].cpu().numpy()  # (C, H, W)

# 2. 反归一化到 [0, 1]
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img_np = img_np * imagenet_std + imagenet_mean  # (C, H, W) [0, 1]

# 3. 转换格式并缩放到 [0, 255]
img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # uint8 [0, 255]
```

## 🎯 关键要点

1. **LeRobot 默认对图像使用 ImageNet 归一化**
   - 这是为了配合预训练的 Vision Backbone (ResNet18)
   - 归一化模式在 `config.json` 中定义

2. **可视化时必须反归一化**
   - 从 `processed_batch` 提取的图像已经归一化
   - 必须先反归一化再转换为 uint8

3. **训练和推理的归一化必须一致**
   - 两者都使用相同的 ImageNet mean/std
   - Preprocessor 自动处理，无需手动干预

4. **修复不影响模型性能**
   - 只修复了可视化代码
   - 训练和推理的数据流程没有改变
   - 模型看到的数据仍然是正确的

## 📚 相关文件

- **诊断脚本**: `scripts/debug_image_normalization.py`
- **测试脚本**: `scripts/test_image_normalization_fix.py`
- **修复代码**: `src/rev2fwd_il/train/lerobot_train_with_viz.py`
- **推理代码**: `scripts/41_test_A_diffusion_visualize.py`
- **详细分析**: `docs/image_normalization_analysis.md`
- **对比图**: `docs/image_normalization_comparison.png`

## ✨ 总结

问题已成功修复！可视化图像现在应该显示正确的颜色和亮度，不再有过曝和偏色问题。修复只影响可视化代码，不影响模型训练和推理的正确性。
