#!/usr/bin/env python3
"""Debug script to check image normalization in training vs inference.

This script helps diagnose the image over-exposure and color shift issue
by checking the normalization applied during training and inference.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def check_training_normalization(checkpoint_dir: str):
    """Check normalization settings used during training."""
    checkpoint_path = Path(checkpoint_dir)
    
    print("=" * 80)
    print("TRAINING NORMALIZATION CHECK")
    print("=" * 80)
    
    # 1. Check config.json
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        print("\n1. Policy Config (config.json):")
        print(f"   normalization_mapping: {config.get('normalization_mapping', {})}")
        
        visual_norm = config.get('normalization_mapping', {}).get('VISUAL', 'NONE')
        print(f"\n   ⚠️  VISUAL normalization mode: {visual_norm}")
        if visual_norm == "MEAN_STD":
            print("   ⚠️  WARNING: Images are normalized with MEAN_STD!")
            print("   ⚠️  This likely uses ImageNet mean/std which causes over-exposure!")
    
    # 2. Check normalizer files
    normalizer_files = list(checkpoint_path.glob("*normalizer*.safetensors"))
    print(f"\n2. Normalizer files found: {len(normalizer_files)}")
    for nf in normalizer_files:
        print(f"   - {nf.name}")
        
        # Load and inspect
        try:
            data = load_file(str(nf))
            print(f"     Keys: {list(data.keys())}")
            for key, tensor in data.items():
                print(f"     {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                if 'observation.image' in key:
                    print(f"       Values: {tensor.cpu().numpy()}")
        except Exception as e:
            print(f"     Error loading: {e}")
    
    # 3. Check dataset stats
    dataset_dir = checkpoint_path.parent.parent.parent / "lerobot_dataset"
    stats_path = dataset_dir / "meta" / "stats.json"
    
    if stats_path.exists():
        print(f"\n3. Dataset stats (stats.json):")
        with open(stats_path) as f:
            stats = json.load(f)
        
        if "observation.image" in stats:
            img_stats = stats["observation.image"]
            print(f"   observation.image stats:")
            print(f"     mean: {img_stats.get('mean', 'N/A')}")
            print(f"     std: {img_stats.get('std', 'N/A')}")
            print(f"     min: {img_stats.get('min', 'N/A')}")
            print(f"     max: {img_stats.get('max', 'N/A')}")
        else:
            print("   observation.image: NOT FOUND in stats")
    else:
        print(f"\n3. Dataset stats not found at: {stats_path}")
    
    print("\n" + "=" * 80)


def check_imagenet_normalization():
    """Show ImageNet normalization values."""
    print("\n" + "=" * 80)
    print("IMAGENET NORMALIZATION VALUES")
    print("=" * 80)
    print("\nImageNet mean (RGB): [0.485, 0.456, 0.406]")
    print("ImageNet std  (RGB): [0.229, 0.224, 0.225]")
    print("\nEffect on [0, 1] normalized images:")
    print("  normalized_img = (img - mean) / std")
    print("  For img=0.5 (mid-gray):")
    
    img = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized = (img - mean) / std
    print(f"    R: ({img[0]} - {mean[0]}) / {std[0]} = {normalized[0]:.3f}")
    print(f"    G: ({img[1]} - {mean[1]}) / {std[1]} = {normalized[1]:.3f}")
    print(f"    B: ({img[2]} - {mean[2]}) / {std[2]} = {normalized[2]:.3f}")
    
    print("\n  To reverse (for visualization):")
    print("    img = normalized * std + mean")
    reversed_img = normalized * std + mean
    print(f"    Result: [{reversed_img[0]:.3f}, {reversed_img[1]:.3f}, {reversed_img[2]:.3f}]")
    
    print("\n  ⚠️  If visualization code does: img * 255")
    print("      without reversing normalization first,")
    print("      you get wrong colors!")
    
    print("\n" + "=" * 80)


def check_visualization_code():
    """Check how images are processed in visualization code."""
    print("\n" + "=" * 80)
    print("VISUALIZATION CODE CHECK")
    print("=" * 80)
    
    viz_file = Path("src/rev2fwd_il/train/lerobot_train_with_viz.py")
    if viz_file.exists():
        print(f"\nChecking: {viz_file}")
        with open(viz_file) as f:
            content = f.read()
        
        # Find the image extraction code
        if "img_np * 255" in content:
            print("\n  ✓ Found: img_np * 255")
            print("    This converts [0, 1] to [0, 255]")
        
        if "processed_batch[\"observation.image\"]" in content:
            print("\n  ✓ Found: processed_batch['observation.image']")
            print("    ⚠️  WARNING: This is AFTER normalization!")
            print("    ⚠️  If VISUAL=MEAN_STD, images are ImageNet-normalized!")
            print("    ⚠️  Need to REVERSE normalization before * 255!")
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_image_normalization.py <checkpoint_dir>")
        print("\nExample:")
        print("  python debug_image_normalization.py runs/diffusion_A_2cam_3/checkpoints/checkpoints/last/pretrained_model")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
    check_training_normalization(checkpoint_dir)
    check_imagenet_normalization()
    check_visualization_code()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("\nLikely issue:")
    print("  1. Training uses VISUAL=MEAN_STD (ImageNet normalization)")
    print("  2. Visualization extracts from processed_batch (already normalized)")
    print("  3. Visualization does: img * 255 without reversing normalization")
    print("  4. Result: Over-exposed, color-shifted images")
    print("\nSolution:")
    print("  Option A: Change VISUAL normalization to NONE (retrain)")
    print("  Option B: Reverse ImageNet normalization before visualization")
    print("  Option C: Extract from raw_batch instead of processed_batch")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
