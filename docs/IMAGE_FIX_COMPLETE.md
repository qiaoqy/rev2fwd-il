# ✅ 图像归一化问题修复完成

## 📋 问题回顾

**症状**: `runs/diffusion_A_2cam_3/xyz_viz/` 中的可视化图像出现过曝和偏色

**根本原因**: 
- 训练时使用了 **ImageNet MEAN_STD 归一化**
- 可视化代码从 `processed_batch` 提取图像（已归一化）
- 直接 `* 255` 转换为 uint8，**没有反归一化**

## 🔧 修复内容

### 修复的文件和位置

**文件**: `src/rev2fwd_il/train/lerobot_train_with_viz.py`

修复了 **4 个位置**的图像处理：

| 位置 | 函数 | 相机 | 行号 |
|------|------|------|------|
| 1 | `extract_xyz_visualization_data()` | Table Camera | ~226 |
| 2 | `extract_xyz_visualization_data()` | Wrist Camera | ~243 |
| 3 | `extract_action_chunk_data()` | Table Camera | ~340 |
| 4 | `extract_action_chunk_data()` | Wrist Camera | ~357 |

### 修复代码

在每个 `* 255` 转换之前添加：

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

## ✅ 验证结果

运行验证脚本：
```bash
python scripts/verify_image_fix.py
```

结果：
```
✓ ALL FIXES APPLIED!

Total image conversions (* 255): 4
Fixed (with ImageNet reversal): 4
Unfixed (missing reversal): 0
```

## 🎯 影响的可视化

修复后，以下可视化将显示正确的图像：

1. **XYZ Curve 可视化** (`train_xyz_curves_step*.mp4`)
   - Table camera: ✅ 修复
   - Wrist camera: ✅ 修复

2. **Action Chunk 可视化** (`train_action_chunk_step*.mp4`)
   - Table camera: ✅ 修复
   - Wrist camera: ✅ 修复

## 🚀 重新生成可视化

运行以下命令生成修复后的可视化：

```bash
conda activate rev2fwd_il

# 重新训练（会生成正确的可视化）
CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_2images.npz \
    --out runs/diffusion_A_2cam_3 \
    --steps 200 \
    --enable_xyz_viz \
    --include_obj_pose
```

检查新生成的可视化：
```bash
ls runs/diffusion_A_2cam_3/xyz_viz/
# 应该看到:
# - train_xyz_curves_step0.mp4
# - train_action_chunk_step0.mp4
# - train_xyz_curves_step200.mp4
# - train_action_chunk_step200.mp4
# 等等...
```

## 📊 修复前后对比

### 修复前（错误）
```python
# 直接 * 255，没有反归一化
img_np = processed_batch["observation.image"][0, -1]  # ImageNet 归一化后
img_np = (img_np * 255).astype(np.uint8)  # ❌ 错误
```

**效果**: 
- Mid-gray (0.5, 0.5, 0.5) → (16, 50, 106) 
- 偏蓝色、过暗

### 修复后（正确）
```python
# 先反归一化，再 * 255
img_np = processed_batch["observation.image"][0, -1]  # ImageNet 归一化后
img_np = img_np * imagenet_std + imagenet_mean  # ✅ 反归一化
img_np = (img_np * 255).astype(np.uint8)  # ✅ 正确
```

**效果**:
- Mid-gray (0.5, 0.5, 0.5) → (127, 127, 127)
- 颜色正确

## 🔍 技术细节

### ImageNet 归一化参数
```python
mean (RGB): [0.485, 0.456, 0.406]
std  (RGB): [0.229, 0.224, 0.225]
```

### 归一化流程

**训练/推理时**:
```
uint8 [0,255] → /255 → float32 [0,1] → ImageNet norm → mean~0, std~1
```

**可视化时**:
```
mean~0, std~1 → 反ImageNet norm → float32 [0,1] → *255 → uint8 [0,255]
```

### 为什么使用 ImageNet 归一化？

LeRobot 的 Diffusion Policy 默认配置：
```json
"normalization_mapping": {
    "VISUAL": "MEAN_STD",  // 使用 ImageNet 归一化
    "STATE": "MIN_MAX",
    "ACTION": "MIN_MAX"
}
```

这是为了配合预训练的 Vision Backbone (ResNet18)，使其在迁移学习时效果更好。

## 📚 相关文档

- `docs/image_normalization_fix_summary.md` - 完整修复说明
- `docs/image_normalization_analysis.md` - 深入技术分析
- `docs/QUICK_FIX_REFERENCE.md` - 快速参考
- `scripts/debug_image_normalization.py` - 诊断工具
- `scripts/test_image_normalization_fix.py` - 测试工具
- `scripts/verify_image_fix.py` - 验证工具

## ✨ 总结

所有图像归一化问题已修复！重新运行训练后，`xyz_viz/` 文件夹中的所有可视化视频（包括 `train_action_chunk_step*.mp4`）都将显示正确的颜色和亮度，不再有过曝和偏色问题。
