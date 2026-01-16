#!/usr/bin/env python3
"""Verify that all image normalization fixes are in place."""

import re
from pathlib import Path


def check_file_for_fixes(filepath: Path) -> dict:
    """Check if a file has the ImageNet normalization fix."""
    with open(filepath) as f:
        content = f.read()
    
    # Find all occurrences of "* 255"
    pattern = r'(img_np|wrist_img_np)\s*\*\s*255'
    matches = list(re.finditer(pattern, content))
    
    results = {
        "file": str(filepath),
        "total_conversions": len(matches),
        "fixed": 0,
        "unfixed": 0,
        "details": []
    }
    
    # For each match, check if there's a reverse normalization before it
    for match in matches:
        line_num = content[:match.start()].count('\n') + 1
        
        # Get context (500 chars before the match)
        start = max(0, match.start() - 500)
        context = content[start:match.start()]
        
        # Check if ImageNet normalization reversal is present
        has_reverse = "imagenet_std + imagenet_mean" in context.lower()
        
        if has_reverse:
            results["fixed"] += 1
            status = "✓ FIXED"
        else:
            results["unfixed"] += 1
            status = "✗ MISSING FIX"
        
        results["details"].append({
            "line": line_num,
            "status": status,
            "has_reverse": has_reverse
        })
    
    return results


def main():
    print("=" * 80)
    print("Verifying Image Normalization Fixes")
    print("=" * 80)
    
    files_to_check = [
        "src/rev2fwd_il/train/lerobot_train_with_viz.py",
    ]
    
    all_fixed = True
    
    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            print(f"\n✗ File not found: {filepath}")
            continue
        
        print(f"\nChecking: {filepath}")
        print("-" * 80)
        
        results = check_file_for_fixes(path)
        
        print(f"  Total image conversions (* 255): {results['total_conversions']}")
        print(f"  Fixed (with ImageNet reversal): {results['fixed']}")
        print(f"  Unfixed (missing reversal): {results['unfixed']}")
        
        for detail in results["details"]:
            print(f"    Line {detail['line']:4d}: {detail['status']}")
        
        if results["unfixed"] > 0:
            all_fixed = False
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if all_fixed:
        print("✓ ALL FIXES APPLIED!")
        print("\nAll image conversions now properly reverse ImageNet normalization.")
        print("You can re-run training to generate corrected visualizations:")
        print("\n  CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \\")
        print("      --dataset data/A_forward_with_2images.npz \\")
        print("      --out runs/diffusion_A_2cam_3 \\")
        print("      --steps 200 --enable_xyz_viz --include_obj_pose")
    else:
        print("✗ SOME FIXES MISSING!")
        print("\nPlease check the unfixed locations above.")
    
    print("=" * 80 + "\n")
    
    return 0 if all_fixed else 1


if __name__ == "__main__":
    exit(main())
