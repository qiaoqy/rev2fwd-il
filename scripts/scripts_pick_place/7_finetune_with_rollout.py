#!/usr/bin/env python3
"""Step 7: Finetune policy with rollout data.

This script merges the original training data with newly collected rollout data
and continues training from an existing checkpoint.

=============================================================================
OVERVIEW
=============================================================================
The workflow is:
1. Load original training data (e.g., data/A_circle.npz)
2. Load rollout data from alternating test (e.g., data/rollout_A_circle_iter1.npz)
3. Merge the datasets by concatenating episode lists
4. Save merged dataset to a temporary file
5. Continue training from checkpoint with merged data using 4_train_diffusion.py

=============================================================================
USAGE EXAMPLES
=============================================================================
# Finetune Task A policy
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/A_circle.npz \
    --rollout_data data/rollout_A_circle_iter1.npz \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A_circle \
    --steps 5000 \
    --include_obj_pose

# Finetune Task B policy
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/B_circle.npz \
    --rollout_data data/rollout_B_circle_iter1.npz \
    --checkpoint runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_B_circle \
    --steps 5000 \
    --include_obj_pose

# Multi-GPU finetuning
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \
    scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_data data/A_circle.npz \
    --rollout_data data/rollout_A_circle_iter1.npz \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A_circle \
    --steps 5000 \
    --include_obj_pose

=============================================================================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge rollout data with original data and finetune from checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--original_data",
        type=str,
        required=True,
        help="Path to original training data NPZ file (e.g., data/A_circle.npz).",
    )
    parser.add_argument(
        "--rollout_data",
        type=str,
        required=True,
        help="Path to rollout data NPZ file from script 6 (e.g., data/rollout_A_circle_iter1.npz).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from (e.g., runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for continued training (same as original training dir).",
    )

    # Training parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of additional training steps. Default: 5000.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for training. Default: 32.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Default: 1e-4.",
    )

    # Model settings
    parser.add_argument(
        "--include_obj_pose",
        action="store_true",
        help="Include object pose in observation.state.",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rev2fwd-diffusion-finetune",
        help="WandB project name.",
    )

    # Data handling
    parser.add_argument(
        "--keep_merged",
        action="store_true",
        help="Keep the merged dataset file after training (default: delete).",
    )
    parser.add_argument(
        "--merged_output",
        type=str,
        default=None,
        help="Custom path for merged dataset. If not specified, uses a temp file.",
    )

    return parser.parse_args()


def load_episodes_from_npz(path: str) -> list[dict]:
    """Load episodes from NPZ file.
    
    Args:
        path: Path to the NPZ file.
        
    Returns:
        List of episode dictionaries.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    
    print(f"Loading episodes from {path}...")
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    
    print(f"  Loaded {len(episodes)} episodes")
    
    # Print info about first episode
    if episodes:
        ep0 = episodes[0]
        print(f"  Episode 0 fields: {list(ep0.keys())}")
        if "images" in ep0:
            print(f"  Episode 0 images shape: {ep0['images'].shape}")
        if "action" in ep0:
            print(f"  Episode 0 action shape: {ep0['action'].shape}")
    
    return episodes


def merge_datasets(
    original_npz: str,
    rollout_npz: str,
    output_npz: str,
) -> str:
    """Merge original training data with rollout data.
    
    Args:
        original_npz: Path to original training data.
        rollout_npz: Path to rollout data from script 6.
        output_npz: Path to save merged dataset.
        
    Returns:
        Path to merged dataset.
    """
    print(f"\n{'='*60}")
    print("Merging Datasets")
    print(f"{'='*60}")
    
    # Load both datasets
    original_episodes = load_episodes_from_npz(original_npz)
    rollout_episodes = load_episodes_from_npz(rollout_npz)
    
    # Merge episodes
    merged_episodes = list(original_episodes) + list(rollout_episodes)
    
    print(f"\nMerged: {len(original_episodes)} + {len(rollout_episodes)} = {len(merged_episodes)} episodes")
    
    # Save merged dataset
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, episodes=np.array(merged_episodes, dtype=object))
    print(f"Saved merged dataset to {output_path}")
    
    return str(output_path)


def finetune_from_checkpoint(
    merged_data: str,
    checkpoint_path: str,
    output_dir: str,
    steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    include_obj_pose: bool = True,
    wandb: bool = False,
    wandb_project: str = "rev2fwd-diffusion-finetune",
) -> int:
    """Continue training from checkpoint with merged data.
    
    This function invokes 4_train_diffusion.py with appropriate flags.
    
    Args:
        merged_data: Path to merged data.
        checkpoint_path: Path to checkpoint to resume from.
        output_dir: Output directory.
        steps: Number of training steps.
        batch_size: Batch size.
        lr: Learning rate.
        include_obj_pose: Include object pose in observation.
        wandb: Enable WandB logging.
        wandb_project: WandB project name.
        
    Returns:
        Return code from training script.
    """
    print(f"\n{'='*60}")
    print("Starting Finetuning")
    print(f"{'='*60}")
    print(f"  Merged data: {merged_data}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Include obj pose: {include_obj_pose}")
    print(f"  WandB: {wandb}")
    
    # Build command for 4_train_diffusion.py
    script_path = Path(__file__).parent / "4_train_diffusion.py"
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        "--dataset", merged_data,
        "--out", output_dir,
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--resume",  # Resume from checkpoint
        "--force_convert",  # Re-convert data since it changed
    ]
    
    if include_obj_pose:
        cmd.append("--include_obj_pose")
    
    if wandb:
        cmd.extend(["--wandb", "--wandb_project", wandb_project])
    
    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    # Run training
    result = subprocess.run(cmd)
    
    return result.returncode


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    
    # Validate input files exist
    if not Path(args.original_data).exists():
        print(f"ERROR: Original data not found: {args.original_data}")
        sys.exit(1)
    
    if not Path(args.rollout_data).exists():
        print(f"ERROR: Rollout data not found: {args.rollout_data}")
        sys.exit(1)
    
    # Determine merged data path
    if args.merged_output:
        merged_data_path = args.merged_output
    else:
        # Create temp file for merged data
        # Use the same directory as original data for consistency
        original_dir = Path(args.original_data).parent
        original_stem = Path(args.original_data).stem
        merged_data_path = str(original_dir / f"{original_stem}_merged_temp.npz")
    
    try:
        # Step 1: Merge datasets
        merged_data_path = merge_datasets(
            original_npz=args.original_data,
            rollout_npz=args.rollout_data,
            output_npz=merged_data_path,
        )
        
        # Step 2: Finetune from checkpoint
        return_code = finetune_from_checkpoint(
            merged_data=merged_data_path,
            checkpoint_path=args.checkpoint,
            output_dir=args.out,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            include_obj_pose=args.include_obj_pose,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
        
        if return_code != 0:
            print(f"\nERROR: Training failed with return code {return_code}")
            sys.exit(return_code)
        
        print(f"\n{'='*60}")
        print("Finetuning Complete!")
        print(f"{'='*60}")
        
    finally:
        # Cleanup merged data if not keeping
        if not args.keep_merged and not args.merged_output:
            merged_path = Path(merged_data_path)
            if merged_path.exists():
                print(f"Cleaning up temporary merged file: {merged_data_path}")
                merged_path.unlink()


if __name__ == "__main__":
    main()
