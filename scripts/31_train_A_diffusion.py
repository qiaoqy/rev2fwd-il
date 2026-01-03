#!/usr/bin/env python3
"""Train Policy A with Diffusion Policy on the forward dataset with images.

This script uses the LeRobot library (v0.4.2) to train a Diffusion Policy for
imitation learning on the forward dataset created by script 22.

=============================================================================
OVERVIEW
=============================================================================
This script implements training a vision-based diffusion policy that learns from:
- RGB images from camera (observation.image)
- End-effector pose (observation.state): position (3D) + quaternion (4D) = 7D
- Gripper action (part of action)

The policy outputs:
- Action: End-effector target pose (7D) + gripper command (1D) = 8D

=============================================================================
INPUT DATA FORMAT (from script 22_make_A_forward_dataset_with_images.py)
=============================================================================
NPZ file with episodes list, each dict containing:
    - obs:        (T, 36)  State observations
    - images:     (T, H, W, 3)  RGB images (uint8)
    - ee_pose:    (T, 7)   EE poses [x, y, z, qw, qx, qy, qz]
    - obj_pose:   (T, 7)   Object poses
    - gripper:    (T,)     Gripper actions (+1=open, -1=close)
    - place_pose: (7,)     Place position
    - goal_pose:  (7,)     Goal position

=============================================================================
LEROBOT DATASET CONVERSION
=============================================================================
The script converts the NPZ data to LeRobot v3.0 format:
    - observation.image: (C, H, W) float32 images (channel-first, normalized)
    - observation.state: (7,) float32 end-effector pose
    - action: (8,) float32 action [ee_pose(7), gripper(1)]

=============================================================================
USAGE EXAMPLES
=============================================================================
CUDA_VISIBLE_DEVICES=1 

# Step 1: Convert data only (recommended for first run)
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --convert_only

# Step 2: Train with converted data
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --steps 1000 \
    --batch_size 8

# Quick test run
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A_test \
    --steps 1000 \
    --batch_size 8

# Resume training
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --resume

Multi-GPU training
CUDA_VISIBLE_DEVICES=1,2,3,4,5 torchrun --nproc_per_node=5 scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --batch_size 32

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
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Policy A with Diffusion Policy using LeRobot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Input/Output Arguments
    # =========================================================================
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/A_forward_with_images.npz",
        help="Path to the forward BC dataset NPZ file with images from script 22.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/diffusion_A",
        help="Output directory for saving model checkpoints and logs.",
    )
    parser.add_argument(
        "--lerobot_dataset_dir",
        type=str,
        default=None,
        help="Directory to save the converted LeRobot dataset. "
             "If not specified, uses {out}/lerobot_dataset.",
    )

    # =========================================================================
    # Mode Selection
    # =========================================================================
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert data to LeRobot format, don't train.",
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="Skip data conversion (assume already converted).",
    )
    parser.add_argument(
        "--force_convert",
        action="store_true",
        help="Force re-conversion even if dataset exists.",
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps. Default: 100000.",
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
        help="Learning rate for Adam optimizer. Default: 1e-4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Default: 0.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Default: 4.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second of the dataset. Default: 20.",
    )

    # =========================================================================
    # Diffusion Policy Architecture
    # =========================================================================
    parser.add_argument(
        "--n_obs_steps",
        type=int,
        default=2,
        help="Number of observation steps. Default: 2.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Diffusion horizon (action sequence length). Default: 16.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="Number of action steps to execute. Default: 8.",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="resnet18",
        help="Vision backbone architecture. Default: resnet18.",
    )
    parser.add_argument(
        "--crop_shape",
        type=int,
        nargs=2,
        default=[84, 84],
        help="Image crop shape (H, W). Default: 84 84.",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps. Default: 100.",
    )

    # =========================================================================
    # Logging and Checkpointing
    # =========================================================================
    parser.add_argument(
        "--log_freq",
        type=int,
        default=100,
        help="Log metrics every N steps. Default: 100.",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps. Default: 10000.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    # =========================================================================
    # Device Selection
    # =========================================================================
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device ('cuda' or 'cpu'). Auto-detect if not specified.",
    )

    # =========================================================================
    # WandB Logging
    # =========================================================================
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rev2fwd-diffusion",
        help="WandB project name. Default: rev2fwd-diffusion.",
    )

    return parser.parse_args()


def load_episodes_from_npz(path: str) -> list[dict]:
    """Load episodes from NPZ file created by script 22.
    
    Args:
        path: Path to the NPZ file.
        
    Returns:
        List of episode dictionaries.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


def convert_npz_to_lerobot_format(
    npz_path: str,
    output_dir: str,
    fps: int = 20,
    repo_id: str = "local/rev2fwd_diffusion",
    force: bool = False,
) -> tuple[int, int]:
    """Convert NPZ dataset to LeRobot v3.0 format.
    
    Args:
        npz_path: Path to input NPZ file.
        output_dir: Directory to save LeRobot dataset.
        fps: Frames per second (should match data collection).
        repo_id: Repository ID for the dataset.
        force: Force re-conversion even if dataset exists.
        
    Returns:
        Tuple of (image_height, image_width).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    output_dir = Path(output_dir)
    
    # Check if dataset already exists
    if output_dir.exists() and not force:
        print(f"LeRobot dataset already exists at {output_dir}")
        # Load episodes to get image shape
        episodes = load_episodes_from_npz(npz_path)
        image_shape = episodes[0]["images"].shape[1:]  # (H, W, 3)
        return image_shape[0], image_shape[1]
    
    # Remove existing dataset if force conversion
    if output_dir.exists() and force:
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)
    
    print(f"\n{'='*60}")
    print("Converting NPZ to LeRobot format")
    print(f"{'='*60}")
    print(f"  Input: {npz_path}")
    print(f"  Output: {output_dir}")
    print(f"  FPS: {fps}")
    
    # Load episodes
    episodes = load_episodes_from_npz(npz_path)
    
    if len(episodes) == 0:
        raise ValueError("No episodes found in NPZ file!")
    
    # Get data dimensions from first episode
    ep0 = episodes[0]
    image_shape = ep0["images"].shape[1:]  # (H, W, 3)
    state_dim = 7  # ee_pose: [x, y, z, qw, qx, qy, qz]
    action_dim = 8  # ee_pose (7) + gripper (1)
    
    print(f"  Image shape: {image_shape} (H, W, C)")
    print(f"  State dim: {state_dim} (ee_pose)")
    print(f"  Action dim: {action_dim} (ee_pose + gripper)")
    print(f"{'='*60}\n")
    
    # Define features for LeRobot dataset
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, image_shape[0], image_shape[1]),  # (C, H, W)
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "gripper"],
        },
    }
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="franka",
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Process each episode
    total_frames = 0
    start_time = time.time()
    
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        
        # Extract data
        images = ep["images"]  # (T, H, W, 3) uint8
        ee_pose = ep["ee_pose"]  # (T, 7)
        gripper = ep["gripper"]  # (T,)
        
        # Create action labels: next ee_pose + gripper
        # For the last frame, we repeat the last action
        for t in range(T - 1):
            # Current observation
            img = images[t]  # (H, W, 3)
            state = ee_pose[t]  # (7,)
            
            # Action: target ee_pose (from next frame) + gripper
            next_ee_pose = ee_pose[t + 1]  # (7,)
            action = np.concatenate([next_ee_pose, [gripper[t]]])  # (8,)
            
            frame = {
                "observation.image": img,  # Will be converted to video
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": "pick_and_place",
            }
            dataset.add_frame(frame)
        
        # Save episode
        dataset.save_episode()
        total_frames += T - 1
        
        if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
            elapsed = time.time() - start_time
            rate = (ep_idx + 1) / elapsed
            print(f"  Processed {ep_idx + 1}/{len(episodes)} episodes "
                  f"({rate:.1f} ep/s, {total_frames} frames)")
    
    # Finalize dataset
    dataset.finalize()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Dataset saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return image_shape[0], image_shape[1]


def create_train_config_json(
    args: argparse.Namespace,
    lerobot_dataset_dir: Path,
    image_height: int,
    image_width: int,
) -> Path:
    """Create a JSON configuration file for lerobot training.
    
    Args:
        args: Parsed command-line arguments.
        lerobot_dataset_dir: Path to LeRobot dataset.
        image_height: Image height.
        image_width: Image width.
        
    Returns:
        Path to the created config file.
    """
    out_dir = Path(args.out)
    config_path = out_dir / "train_config.json"
    
    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = {
        "dataset": {
            "repo_id": "local/rev2fwd_diffusion",
            "root": str(lerobot_dataset_dir),
        },
        "policy": {
            "type": "diffusion",
            "n_obs_steps": args.n_obs_steps,
            "horizon": args.horizon,
            "n_action_steps": args.n_action_steps,
            "input_features": {
                "observation.image": {
                    "type": "VISUAL",
                    "shape": [3, image_height, image_width],
                },
                "observation.state": {
                    "type": "STATE",
                    "shape": [7],
                },
            },
            "output_features": {
                "action": {
                    "type": "ACTION",
                    "shape": [8],
                },
            },
            "vision_backbone": args.vision_backbone,
            "crop_shape": args.crop_shape,
            "num_train_timesteps": args.num_train_timesteps,
            "device": device,
            "push_to_hub": False,
            "optimizer_lr": args.lr,
        },
        "output_dir": str(out_dir / "checkpoints"),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "log_freq": args.log_freq,
        "save_freq": args.save_freq,
        "save_checkpoint": True,
        "eval_freq": 0,
        "wandb": {
            "enable": args.wandb,
            "project": args.wandb_project,
        },
    }
    
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created training config: {config_path}")
    return config_path


def train_with_lerobot_api(
    args: argparse.Namespace,
    lerobot_dataset_dir: Path,
    image_height: int,
    image_width: int,
) -> dict:
    """Train using LeRobot's Python API.
    
    Args:
        args: Parsed command-line arguments.
        lerobot_dataset_dir: Path to LeRobot dataset.
        image_height: Image height.
        image_width: Image width.
        
    Returns:
        Dictionary with training results.
    """
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.scripts.lerobot_train import train
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure input/output features for the policy
    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_height, image_width),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(7,),
        ),
    }
    
    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(8,),
        ),
    }
    
    # Create Diffusion Policy configuration
    policy_cfg = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        vision_backbone=args.vision_backbone,
        crop_shape=tuple(args.crop_shape),
        num_train_timesteps=args.num_train_timesteps,
        device=device,
        push_to_hub=False,
        optimizer_lr=args.lr,
    )
    
    # Create dataset configuration
    dataset_cfg = DatasetConfig(
        repo_id="local/rev2fwd_diffusion",
        root=str(lerobot_dataset_dir),
    )
    
    # Create WandB configuration
    wandb_cfg = WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
    )
    
    # Create training pipeline configuration
    checkpoint_dir = out_dir / "checkpoints"
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=checkpoint_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        save_checkpoint=True,
        wandb=wandb_cfg,
        resume=args.resume,
        eval_freq=0,  # Disable evaluation during training
    )
    
    # Print training info
    print("\n" + "=" * 60)
    print("Starting Diffusion Policy Training")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  LeRobot dataset: {lerobot_dataset_dir}")
    print(f"  Output: {checkpoint_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image shape: ({image_height}, {image_width})")
    print(f"  Crop shape: {tuple(args.crop_shape)}")
    print(f"  Horizon: {args.horizon}")
    print(f"  N obs steps: {args.n_obs_steps}")
    print(f"  N action steps: {args.n_action_steps}")
    print(f"  Vision backbone: {args.vision_backbone}")
    print(f"  Device: {device}")
    print(f"  WandB: {args.wandb}")
    print("=" * 60 + "\n")
    
    # Run training
    train(train_cfg)
    
    return {
        "output_dir": str(checkpoint_dir),
        "steps": args.steps,
    }


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup paths
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lerobot_dataset_dir = args.lerobot_dataset_dir
    if lerobot_dataset_dir is None:
        lerobot_dataset_dir = out_dir / "lerobot_dataset"
    lerobot_dataset_dir = Path(lerobot_dataset_dir)
    
    # Step 1: Convert data to LeRobot format
    if not args.skip_convert:
        image_height, image_width = convert_npz_to_lerobot_format(
            npz_path=args.dataset,
            output_dir=lerobot_dataset_dir,
            fps=args.fps,
            repo_id="local/rev2fwd_diffusion",
            force=args.force_convert,
        )
    else:
        # Load episodes to get image shape
        episodes = load_episodes_from_npz(args.dataset)
        image_shape = episodes[0]["images"].shape[1:]  # (H, W, 3)
        image_height, image_width = image_shape[0], image_shape[1]
    
    if args.convert_only:
        print("Data conversion complete. Exiting (--convert_only flag).")
        return
    
    # Step 2: Train the policy
    result = train_with_lerobot_api(
        args=args,
        lerobot_dataset_dir=lerobot_dataset_dir,
        image_height=image_height,
        image_width=image_width,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Total steps: {result['steps']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

