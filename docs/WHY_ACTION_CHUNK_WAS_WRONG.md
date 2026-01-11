# 为什么 train_action_chunk_step*.mp4 的图像没有正确反归一化？

## 问题定位

你发现：
- ✅ `train_xyz_curves_step*.mp4` 的图像已经修复
- ✗ `train_action_chunk_step*.mp4` 的 table camera 还是有问题

## 原因分析

这两种可视化由**不同的函数**生成：

### 1. XYZ Curves 可视化
**函数**: `extract_xyz_visualization_data()`  
**位置**: `lerobot_train_with_viz.py` line ~210-245  
**状态**: ✅ 已修复（第一次修复时就改了）

### 2. Action Chunk 可视化  
**函数**: `extract_action_chunk_data()`  
**位置**: `lerobot_train_with_viz.py` line ~324-360  
**状态**: ✅ 现在已修复（刚刚补充修复）

## 修复历史

### 第一次修复（之前）
只修复了 `extract_xyz_visualization_data()` 中的图像处理：
- Line ~220: Table camera ✅
- Line ~237: Wrist camera ✅

**遗漏**: `extract_action_chunk_data()` 中的图像处理

### 第二次修复（刚刚）
补充修复了 `extract_action_chunk_data()` 中的图像处理：
- Line ~332: Table camera ✅ **← 这就是你发现的问题位置**
- Line ~349: Wrist camera ✅

## 代码对比

### 修复前（错误）
```python
# extract_action_chunk_data() - Table camera
if "observation.image" in processed_batch:
    img = processed_batch["observation.image"]
    img_np = img[0, -1].detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # ❌ 没有反归一化
    viz_data["table_image"] = img_np
```

### 修复后（正确）
```python
# extract_action_chunk_data() - Table camera
if "observation.image" in processed_batch:
    img = processed_batch["observation.image"]
    img_np = img[0, -1].detach().cpu().numpy()
    
    # ✅ 添加了反归一化
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_np * imagenet_std + imagenet_mean  # Reverse normalization
    
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)  # ✅ 正确
    viz_data["table_image"] = img_np
```

## 为什么会遗漏？

两个函数的代码结构相似，但在不同位置：
- `extract_xyz_visualization_data()`: 生成 XYZ 曲线可视化
- `extract_action_chunk_data()`: 生成 Action Chunk 可视化

第一次修复时只改了第一个函数，遗漏了第二个函数。

## 验证修复

运行验证脚本确认所有位置都已修复：
```bash
python scripts/verify_image_fix.py
```

输出：
```
✓ ALL FIXES APPLIED!

Total image conversions (* 255): 4
Fixed (with ImageNet reversal): 4
Unfixed (missing reversal): 0
  Line  226: ✓ FIXED  ← extract_xyz_visualization_data() table
  Line  243: ✓ FIXED  ← extract_xyz_visualization_data() wrist
  Line  340: ✓ FIXED  ← extract_action_chunk_data() table (刚修复)
  Line  357: ✓ FIXED  ← extract_action_chunk_data() wrist (刚修复)
```

## 重新生成可视化

现在重新运行训练，所有可视化都会正确：

```bash
conda activate rev2fwd_il

CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_2images.npz \
    --out runs/diffusion_A_2cam_3 \
    --steps 200 \
    --enable_xyz_viz \
    --include_obj_pose
```

检查生成的视频：
```bash
ls runs/diffusion_A_2cam_3/xyz_viz/

# 应该看到（都已修复）:
# train_xyz_curves_step0.mp4      ✅ 图像正确
# train_action_chunk_step0.mp4    ✅ 图像正确（刚修复）
# train_xyz_curves_step200.mp4    ✅ 图像正确
# train_action_chunk_step200.mp4  ✅ 图像正确（刚修复）
```

## 总结

问题已完全解决！所有可视化函数中的图像处理都已正确添加 ImageNet 反归一化，不会再有过曝和偏色问题。
