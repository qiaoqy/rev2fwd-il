#!/usr/bin/env python3
"""Step 7: Finetune policy with rollout data.

This script merges rollout data directly into an existing LeRobot dataset
and continues training from an existing checkpoint.

=============================================================================
OVERVIEW (OPTIMIZED)
=============================================================================
The optimized workflow is:
1. Copy existing LeRobot dataset to output directory (avoids re-converting original data)
2. Load rollout data NPZ and add episodes incrementally to the copied LeRobot dataset
3. Continue training from checkpoint with the merged LeRobot dataset

This is much faster than the old workflow which:
- Loaded original NPZ + rollout NPZ
- Merged into new NPZ
- Re-converted everything to LeRobot format

=============================================================================
USAGE EXAMPLES
=============================================================================
# Finetune Task A policy
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_lerobot runs/PP_A_circle/lerobot_dataset \
    --rollout_data data/rollout_A_circle_iter1.npz \
    --checkpoint runs/PP_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/PP_A_circle_finetune \
    --steps 5000 \
    --include_obj_pose

# Finetune Task B policy  
python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_lerobot runs/PP_B_circle/lerobot_dataset \
    --rollout_data data/rollout_B_circle_iter1.npz \
    --checkpoint runs/PP_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out runs/PP_B_circle_finetune \
    --steps 5000 \
    --include_obj_pose

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def get_rank() -> int:
    """Get distributed training rank. Returns 0 if not in distributed mode."""
    # torchrun sets these environment variables
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return int(rank)


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge rollout data into LeRobot dataset and finetune from checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--original_lerobot",
        type=str,
        required=True,
        help="Path to original LeRobot dataset directory (e.g., runs/PP_A_circle/lerobot_dataset).",
    )
    parser.add_argument(
        "--rollout_data",
        type=str,
        default=None,
        help="Path to rollout data NPZ file from script 6 (e.g., data/rollout_A_circle_iter1.npz). "
             "If not provided, will finetune using only the original data.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from (e.g., runs/PP_A_circle/checkpoints/checkpoints/last/pretrained_model).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for continued training.",
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
    parser.add_argument(
        "--include_gripper",
        action="store_true",
        help="Include gripper state in observation.state.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="Number of action steps to execute per inference. Default: 16.",
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
    
    # Workflow control
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare data (copy dataset, add episodes), don't train. "
             "Use this when running distributed training separately with torchrun.",
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


def copy_lerobot_dataset(src_dir: Path, dst_dir: Path) -> None:
    """Copy LeRobot dataset to new location.
    
    Args:
        src_dir: Source LeRobot dataset directory.
        dst_dir: Destination directory.
    """
    print(f"\n{'='*60}")
    print("Copying LeRobot Dataset")
    print(f"{'='*60}")
    print(f"  From: {src_dir}")
    print(f"  To: {dst_dir}")
    
    # Check if source and destination are the same (resolved paths)
    if src_dir.resolve() == dst_dir.resolve():
        print(f"  Source and destination are the same, skipping copy.")
        return
    
    if dst_dir.exists():
        print(f"  Removing existing: {dst_dir}")
        shutil.rmtree(dst_dir)
    
    start_time = time.time()
    shutil.copytree(src_dir, dst_dir)
    elapsed = time.time() - start_time
    print(f"  Copied in {elapsed:.1f}s")


def add_episodes_to_lerobot_dataset(
    lerobot_dir: Path,
    episodes: list[dict],
    include_obj_pose: bool = None,  # Auto-detect from dataset if None
    include_gripper: bool = None,   # Auto-detect from dataset if None
) -> None:
    """Add episodes from NPZ to existing LeRobot dataset.
    
    Args:
        lerobot_dir: Path to LeRobot dataset directory.
        episodes: List of episode dictionaries from NPZ.
        include_obj_pose: Whether to include object pose in state. Auto-detected from dataset if None.
        include_gripper: Whether to include gripper state. Auto-detected from dataset if None.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"\n{'='*60}")
    print("Adding Episodes to LeRobot Dataset")
    print(f"{'='*60}")
    print(f"  Dataset: {lerobot_dir}")
    print(f"  Episodes to add: {len(episodes)}")
    
    # Load existing dataset info to get configuration
    info_path = lerobot_dir / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    
    fps = info["fps"]
    repo_id = info.get("repo_id", "local/merged_dataset")
    features = info["features"]
    has_wrist = "observation.wrist_image" in features
    
    # Auto-detect state dimension from dataset
    state_shape = features.get("observation.state", {}).get("shape", [7])
    state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape
    
    # Determine include_obj_pose and include_gripper from state_dim
    # state_dim meanings:
    #   7  = ee_pose only
    #   8  = ee_pose + gripper
    #   14 = ee_pose + obj_pose
    #   15 = ee_pose + obj_pose + gripper
    if include_obj_pose is None:
        include_obj_pose = state_dim >= 14
    if include_gripper is None:
        include_gripper = state_dim in [8, 15]
    
    # Verify state dimension matches
    expected_dim = 7
    if include_obj_pose:
        expected_dim += 7
    if include_gripper:
        expected_dim += 1
    
    print(f"  Original state_dim: {state_dim}")
    print(f"  Auto-detected: include_obj_pose={include_obj_pose}, include_gripper={include_gripper}")
    print(f"  Expected state_dim: {expected_dim}")
    
    if expected_dim != state_dim:
        print(f"  WARNING: State dimension mismatch! Expected {expected_dim}, dataset has {state_dim}")
        print(f"           Will use dataset's state_dim={state_dim}")
    
    print(f"  FPS: {fps}")
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Original episodes: {info['total_episodes']}")
    print(f"  Original frames: {info['total_frames']}")
    
    # Open dataset for appending (LeRobot v3.0 API)
    # We need to use the resume functionality
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(lerobot_dir),
    )
    
    # Process each episode
    total_frames = 0
    start_time = time.time()
    num_episodes = len(episodes)
    
    print(f"\nProcessing {num_episodes} episodes...")
    
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        
        # Print progress
        elapsed = time.time() - start_time
        rate = (ep_idx / elapsed) if elapsed > 0 and ep_idx > 0 else 0
        eta = (num_episodes - ep_idx) / rate if rate > 0 else 0
        print(f"  [{ep_idx + 1}/{num_episodes}] Processing episode, "
              f"{T} frames | {rate:.1f} ep/s | ETA: {eta:.0f}s")
        
        # Extract data
        images = ep["images"]  # (T, H, W, 3) uint8
        ee_pose = ep["ee_pose"]  # (T, 7)
        obj_pose = ep["obj_pose"]  # (T, 7)
        actions = ep["action"]  # (T, 8)
        wrist_images = ep.get("wrist_images", None)  # (T, H, W, 3) uint8 or None
        
        # Get gripper states from action
        gripper_states = actions[:, 7]  # (T,)
        
        for t in range(T):
            # Current observation
            img = images[t]  # (H, W, 3)
            
            # Build state
            state_parts = [ee_pose[t]]  # Always include ee_pose (7,)
            if include_obj_pose:
                state_parts.append(obj_pose[t])  # Add obj_pose (7,)
            if include_gripper:
                gripper_state = np.array([gripper_states[t]], dtype=np.float32)  # (1,)
                state_parts.append(gripper_state)
            state = np.concatenate(state_parts)
            
            # Action
            action = actions[t]  # (8,)
            
            frame = {
                "observation.image": img,
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": "reverse_pick_and_place",
            }
            
            # Add wrist camera image if available
            if wrist_images is not None and has_wrist:
                frame["observation.wrist_image"] = wrist_images[t]
            
            dataset.add_frame(frame)
        
        # Save episode
        dataset.save_episode()
        total_frames += T
    
    # Finalize dataset
    print("\nFinalizing dataset (encoding videos)...")
    dataset.finalize()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Episode Addition Complete!")
    print(f"{'='*60}")
    print(f"  Added episodes: {num_episodes}")
    print(f"  Added frames: {total_frames}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")


def finetune_from_checkpoint(
    lerobot_dataset_dir: str,
    checkpoint_path: str,
    output_dir: str,
    steps: int = 5000,
    batch_size: int = 32,
    lr: float = 1e-4,
    n_action_steps: int = 16,
    include_obj_pose: bool = True,
    wandb: bool = False,
    wandb_project: str = "rev2fwd-diffusion-finetune",
) -> int:
    """Continue training from checkpoint with merged LeRobot dataset.
    
    Args:
        lerobot_dataset_dir: Path to merged LeRobot dataset.
        checkpoint_path: Path to checkpoint to resume from.
        output_dir: Output directory.
        steps: Number of training steps.
        batch_size: Batch size.
        lr: Learning rate.
        n_action_steps: Number of action steps to execute per inference.
        include_obj_pose: Include object pose in observation.
        wandb: Enable WandB logging.
        wandb_project: WandB project name.
        
    Returns:
        Return code from training script.
    """
    print(f"\n{'='*60}")
    print("Starting Finetuning")
    print(f"{'='*60}")
    print(f"  LeRobot dataset: {lerobot_dataset_dir}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  N action steps: {n_action_steps}")
    print(f"  Include obj pose: {include_obj_pose}")
    print(f"  WandB: {wandb}")
    
    # Prepare the output directory for resuming from external checkpoint
    output_path = Path(output_dir)
    checkpoint_src = Path(checkpoint_path)
    
    # LeRobot expects checkpoint structure:
    # {output_dir}/checkpoints/checkpoints/{step_number}/pretrained_model
    # {output_dir}/checkpoints/checkpoints/{step_number}/training_state
    # {output_dir}/checkpoints/checkpoints/last -> {step_number}  (symlink)
    checkpoints_parent = output_path / "checkpoints" / "checkpoints"
    
    # Check if source checkpoint is external (not in output directory)
    if not str(checkpoint_src.resolve()).startswith(str(output_path.resolve())):
        print(f"\n  Source checkpoint is external, copying to output directory...")
        print(f"    From: {checkpoint_src}")
        
        # Determine the source checkpoint root (parent of pretrained_model)
        if checkpoint_src.name == "pretrained_model":
            checkpoint_root = checkpoint_src.parent
        else:
            checkpoint_root = checkpoint_src
        
        # Read source step number from training_state
        src_training_state = checkpoint_root / "training_state"
        src_step = 0
        if src_training_state.exists():
            training_step_file = src_training_state / "training_step.json"
            if training_step_file.exists():
                import json
                with open(training_step_file, "r") as f:
                    data = json.load(f)
                    src_step = data.get("step", 0)
        
        # Create step-numbered directory (e.g., 010000)
        step_dir_name = f"{src_step:06d}"
        step_checkpoint_dir = checkpoints_parent / step_dir_name
        print(f"    To: {step_checkpoint_dir} (step {src_step})")
        
        # Create target directory
        step_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy pretrained_model directory
        target_pretrained = step_checkpoint_dir / "pretrained_model"
        if target_pretrained.exists():
            print(f"    Removing existing: {target_pretrained}")
            shutil.rmtree(target_pretrained)
        
        src_pretrained = checkpoint_root / "pretrained_model" if (checkpoint_root / "pretrained_model").exists() else checkpoint_src
        shutil.copytree(src_pretrained, target_pretrained)
        print(f"    Copied pretrained_model to {target_pretrained}")
        
        # Copy training_state directory if it exists (preserves original step count)
        src_training_state = checkpoint_root / "training_state"
        if src_training_state.exists():
            target_training_state = step_checkpoint_dir / "training_state"
            if target_training_state.exists():
                print(f"    Removing existing: {target_training_state}")
                shutil.rmtree(target_training_state)
            shutil.copytree(src_training_state, target_training_state)
            print(f"    Copied training_state to {target_training_state}")
        else:
            print(f"    Warning: training_state not found at {src_training_state}")
        
        # NOTE: Do NOT copy wandb directory - it contains run IDs that are tied to the
        # original training run. Copying would cause resume='must' errors because the
        # run ID doesn't exist for this new training context. Let wandb create a fresh run.
        src_wandb = checkpoint_root.parent.parent / "wandb"
        if src_wandb.exists():
            print(f"    Note: Not copying wandb directory (will create new wandb run)")
        else:
            print(f"    Note: wandb directory not found at {src_wandb} (will create new run)")
        
        # Create 'last' symlink pointing to the step directory
        last_symlink = checkpoints_parent / "last"
        if last_symlink.exists() or last_symlink.is_symlink():
            if last_symlink.is_symlink():
                last_symlink.unlink()
            else:
                shutil.rmtree(last_symlink)
        last_symlink.symlink_to(step_dir_name)
        print(f"    Created symlink: last -> {step_dir_name}")
    
    # Build command for 4_train_diffusion.py
    # Use --lerobot_dataset_dir and --skip_convert to use pre-converted dataset
    script_path = Path(__file__).parent / "4_train_diffusion.py"
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(script_path),
        "--dataset", "dummy.npz",  # Not used when --skip_convert is set
        "--lerobot_dataset_dir", str(lerobot_dataset_dir),
        "--out", output_dir,
        "--steps", str(steps),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--n_action_steps", str(n_action_steps),
        "--skip_convert",  # Skip conversion, use existing LeRobot dataset
        "--resume",  # Resume from checkpoint
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
    """Main entry point.
    
    Two workflow modes:
    
    1. --prepare_only mode (recommended for multi-GPU):
       - Run with single process: python 7_finetune_with_rollout.py --prepare_only ...
       - Only prepares data (copy dataset, add episodes, copy checkpoint)
       - Then run training separately: torchrun 4_train_diffusion.py --skip_convert --resume ...
    
    2. Default mode (single GPU only):
       - Does data preparation AND training in one command
       - Uses subprocess to call 4_train_diffusion.py
       - NOT suitable for torchrun (each process would spawn its own training)
    """
    args = _parse_args()
    
    # Validate input paths
    original_lerobot = Path(args.original_lerobot)
    if not original_lerobot.exists():
        print(f"ERROR: Original LeRobot dataset not found: {original_lerobot}")
        sys.exit(1)
    
    # Check if rollout data is provided and exists
    has_rollout_data = args.rollout_data is not None and Path(args.rollout_data).exists()
    
    if args.rollout_data is not None and not Path(args.rollout_data).exists():
        print(f"WARNING: Rollout data not found: {args.rollout_data}")
        print("         Will finetune using only the original data.")
    
    # Output LeRobot dataset path
    output_dir = Path(args.out)
    merged_lerobot_dir = output_dir / "lerobot_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Copy original LeRobot dataset to output directory
        copy_lerobot_dataset(original_lerobot, merged_lerobot_dir)
        
        # Step 2: Add rollout episodes if provided
        if has_rollout_data:
            rollout_episodes = load_episodes_from_npz(args.rollout_data)
            # Pass None to auto-detect from dataset, or use args if explicitly provided
            add_episodes_to_lerobot_dataset(
                lerobot_dir=merged_lerobot_dir,
                episodes=rollout_episodes,
                include_obj_pose=None,  # Auto-detect from dataset
                include_gripper=None,   # Auto-detect from dataset
            )
        else:
            print(f"\nNo rollout data provided. Finetuning with original data only.")
        
        # Step 3: Copy checkpoint to expected location for --resume
        checkpoint_src = Path(args.checkpoint)
        expected_checkpoint_dir = output_dir / "checkpoints" / "checkpoints" / "last"
        expected_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine the source checkpoint root (parent of pretrained_model)
        if checkpoint_src.name == "pretrained_model":
            checkpoint_root = checkpoint_src.parent
        else:
            checkpoint_root = checkpoint_src
        
        # Copy pretrained_model directory
        target_pretrained = expected_checkpoint_dir / "pretrained_model"
        if target_pretrained.exists():
            print(f"Removing existing checkpoint: {target_pretrained}")
            shutil.rmtree(target_pretrained)
        
        print(f"\nCopying checkpoint...")
        print(f"  From: {checkpoint_root}")
        print(f"  To: {expected_checkpoint_dir}")
        
        src_pretrained = checkpoint_root / "pretrained_model" if (checkpoint_root / "pretrained_model").exists() else checkpoint_src
        shutil.copytree(src_pretrained, target_pretrained)
        print(f"  Copied pretrained_model successfully.")
        
        # Copy training_state directory if it exists
        src_training_state = checkpoint_root / "training_state"
        if src_training_state.exists():
            target_training_state = expected_checkpoint_dir / "training_state"
            if target_training_state.exists():
                print(f"Removing existing training_state: {target_training_state}")
                shutil.rmtree(target_training_state)
            shutil.copytree(src_training_state, target_training_state)
            print(f"  Copied training_state successfully.")
        else:
            print(f"  Warning: training_state not found at {src_training_state}")
        
        print(f"\n{'='*60}")
        print("Data Preparation Complete!")
        print(f"{'='*60}")
        print(f"  Merged dataset: {merged_lerobot_dir}")
        print(f"  Checkpoint: {target_pretrained}")
        
        # If --prepare_only, stop here
        if args.prepare_only:
            print(f"\n--prepare_only mode: Skipping training.")
            print(f"\nTo train with multi-GPU, run:")
            print(f"  CUDA_VISIBLE_DEVICES=5,6,7,8,9 torchrun --nproc_per_node=5 \\")
            print(f"      scripts/scripts_pick_place/4_train_diffusion.py \\")
            print(f"      --dataset dummy.npz \\")
            print(f"      --lerobot_dataset_dir {merged_lerobot_dir} \\")
            print(f"      --out {args.out} \\")
            print(f"      --steps {args.steps} \\")
            print(f"      --batch_size {args.batch_size} \\")
            print(f"      --n_action_steps {args.n_action_steps} \\")
            print(f"      --skip_convert --resume \\")
            if args.include_obj_pose:
                print(f"      --include_obj_pose \\")
            if args.wandb:
                print(f"      --wandb --wandb_project {args.wandb_project}")
            return
        
        # Step 4: Finetune from checkpoint (single process only!)
        print(f"\nStarting training (single-process mode)...")
        return_code = finetune_from_checkpoint(
            lerobot_dataset_dir=str(merged_lerobot_dir),
            checkpoint_path=args.checkpoint,
            output_dir=args.out,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            n_action_steps=args.n_action_steps,
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
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
