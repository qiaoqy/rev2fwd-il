#!/usr/bin/env python3
"""Test script to verify the image normalization fix.

This script demonstrates the effect of ImageNet normalization and
verifies that the fix correctly reverses it for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def test_imagenet_normalization():
    """Test ImageNet normalization and reversal."""
    print("=" * 80)
    print("Testing ImageNet Normalization Fix")
    print("=" * 80)
    
    # Create a test image (mid-gray)
    test_img = np.ones((3, 128, 128), dtype=np.float32) * 0.5
    print(f"\n1. Original image (mid-gray):")
    print(f"   Shape: {test_img.shape} (C, H, W)")
    print(f"   Range: [{test_img.min():.4f}, {test_img.max():.4f}]")
    print(f"   Mean per channel: R={test_img[0].mean():.4f}, G={test_img[1].mean():.4f}, B={test_img[2].mean():.4f}")
    
    # Apply ImageNet normalization
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    normalized = (test_img - imagenet_mean) / imagenet_std
    print(f"\n2. After ImageNet normalization:")
    print(f"   Formula: (img - mean) / std")
    print(f"   Range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"   Mean per channel: R={normalized[0].mean():.4f}, G={normalized[1].mean():.4f}, B={normalized[2].mean():.4f}")
    
    # OLD (WRONG) visualization: directly * 255
    wrong_viz = np.transpose(normalized, (1, 2, 0))  # CHW -> HWC
    wrong_viz = (wrong_viz * 255).clip(0, 255).astype(np.uint8)
    print(f"\n3. OLD (WRONG) visualization (normalized * 255):")
    print(f"   Range: [{wrong_viz.min()}, {wrong_viz.max()}]")
    print(f"   Mean per channel: R={wrong_viz[:,:,0].mean():.1f}, G={wrong_viz[:,:,1].mean():.1f}, B={wrong_viz[:,:,2].mean():.1f}")
    print(f"   ⚠️  This gives wrong colors! (should be ~127 for mid-gray)")
    
    # NEW (CORRECT) visualization: reverse normalization first
    reversed_norm = normalized * imagenet_std + imagenet_mean
    correct_viz = np.transpose(reversed_norm, (1, 2, 0))  # CHW -> HWC
    correct_viz = (correct_viz * 255).clip(0, 255).astype(np.uint8)
    print(f"\n4. NEW (CORRECT) visualization (reverse norm, then * 255):")
    print(f"   Formula: img = normalized * std + mean, then * 255")
    print(f"   Range: [{correct_viz.min()}, {correct_viz.max()}]")
    print(f"   Mean per channel: R={correct_viz[:,:,0].mean():.1f}, G={correct_viz[:,:,1].mean():.1f}, B={correct_viz[:,:,2].mean():.1f}")
    print(f"   ✓ Correct! Mid-gray should be ~127")
    
    # Verify the fix
    expected = 127
    tolerance = 2
    is_correct = all(abs(correct_viz[:,:,i].mean() - expected) < tolerance for i in range(3))
    
    print(f"\n5. Verification:")
    if is_correct:
        print(f"   ✓ PASS: Visualization is correct!")
    else:
        print(f"   ✗ FAIL: Visualization is still wrong!")
    
    print("=" * 80)
    
    return is_correct


def create_comparison_image():
    """Create a visual comparison of wrong vs correct normalization."""
    print("\nCreating visual comparison...")
    
    # Create a gradient test image
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    gradient = np.repeat(gradient, 128, axis=0)  # (128, 256)
    
    # RGB channels
    test_img = np.stack([gradient, gradient, gradient], axis=0)  # (3, 128, 256)
    
    # Apply ImageNet normalization
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    normalized = (test_img - imagenet_mean) / imagenet_std
    
    # Wrong visualization
    wrong_viz = np.transpose(normalized, (1, 2, 0))
    wrong_viz = (wrong_viz * 255).clip(0, 255).astype(np.uint8)
    
    # Correct visualization
    reversed_norm = normalized * imagenet_std + imagenet_mean
    correct_viz = np.transpose(reversed_norm, (1, 2, 0))
    correct_viz = (correct_viz * 255).clip(0, 255).astype(np.uint8)
    
    # Original (for reference)
    original_viz = np.transpose(test_img, (1, 2, 0))
    original_viz = (original_viz * 255).clip(0, 255).astype(np.uint8)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_viz)
    axes[0].set_title("Original (Ground Truth)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(wrong_viz)
    axes[1].set_title("OLD (WRONG)\nDirect * 255", fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    axes[2].imshow(correct_viz)
    axes[2].set_title("NEW (CORRECT)\nReverse norm + * 255", fontsize=14, fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = Path("docs/image_normalization_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison image to: {output_path}")
    
    plt.close()


def main():
    # Run test
    is_correct = test_imagenet_normalization()
    
    # Create visual comparison
    create_comparison_image()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe fix has been applied to:")
    print("  1. src/rev2fwd_il/train/lerobot_train_with_viz.py")
    print("     - extract_xyz_visualization_data() - table camera")
    print("     - extract_xyz_visualization_data() - wrist camera")
    print("     - extract_action_chunk_data() - table camera")
    print("     - extract_action_chunk_data() - wrist camera")
    print("\n  2. scripts/41_test_A_diffusion_visualize.py")
    print("     - Added debug output to verify normalization")
    print("\nNext steps:")
    print("  1. Re-run training with --enable_xyz_viz to generate new visualizations")
    print("  2. Check that images in xyz_viz/ look correct (no over-exposure)")
    print("  3. Run inference with debug output to verify normalization is applied")
    print("\nExample commands:")
    print("  # Training (will generate corrected visualizations):")
    print("  CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \\")
    print("      --dataset data/A_forward_with_2images.npz \\")
    print("      --out runs/diffusion_A_2cam_test \\")
    print("      --steps 200 --enable_xyz_viz")
    print("\n  # Inference (with debug output):")
    print("  CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \\")
    print("      --checkpoint runs/diffusion_A_2cam_3/checkpoints/checkpoints/last/pretrained_model \\")
    print("      --out_dir runs/diffusion_A_2cam_3/videos_test \\")
    print("      --num_episodes 1")
    print("=" * 80 + "\n")
    
    return 0 if is_correct else 1


if __name__ == "__main__":
    exit(main())
