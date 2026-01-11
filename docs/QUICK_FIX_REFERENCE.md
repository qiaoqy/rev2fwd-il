# 🚀 图像归一化问题 - 快速修复参考

## 问题症状
- ✗ XYZ 可视化图像过曝
- ✗ 颜色偏蓝色
- ✗ 对比度异常

## 根本原因
```python
# ❌ 错误: 直接 * 255，没有反归一化
img = processed_batch["observation.image"]  # ImageNet 归一化后
img = (img * 255).astype(np.uint8)  # 错误！
```

## 修复方法
```python
# ✅ 正确: 先反归一化，再 * 255
img = processed_batch["observation.image"]  # ImageNet 归一化后

# 反归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
img = img * imagenet_std + imagenet_mean  # 反归一化到 [0, 1]

# 转换为 uint8
img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
img = (img * 255).clip(0, 255).astype(np.uint8)
```

## 已修复的文件
- ✅ `src/rev2fwd_il/train/lerobot_train_with_viz.py` (4处)
- ✅ `scripts/41_test_A_diffusion_visualize.py` (debug 输出)

## 验证修复
```bash
# 1. 运行测试
conda activate rev2fwd_il
python scripts/test_image_normalization_fix.py

# 2. 重新生成可视化
CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_2images.npz \
    --out runs/test_fix \
    --steps 200 --enable_xyz_viz

# 3. 检查图像
ls runs/test_fix/xyz_viz/  # 应该看起来正常
```

## 技术要点
- **ImageNet mean**: [0.485, 0.456, 0.406]
- **ImageNet std**: [0.229, 0.224, 0.225]
- **公式**: `img = normalized * std + mean`
- **位置**: 在 `* 255` 之前反归一化

## 详细文档
- 📄 `docs/image_normalization_fix_summary.md` - 完整修复说明
- 📄 `docs/image_normalization_analysis.md` - 深入技术分析
- 🔧 `scripts/debug_image_normalization.py` - 诊断工具
- 🧪 `scripts/test_image_normalization_fix.py` - 验证工具
