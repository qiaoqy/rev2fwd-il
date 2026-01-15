#!/usr/bin/env python3
"""
Train a Diffusion Policy from an existing LeRobot dataset.

This script directly uses a pre-converted LeRobot dataset (from script 31)
and trains using a different configuration inspired by train_dp_zarr.py.

=============================================================================
KEY DIFFERENCES FROM SCRIPT 31
=============================================================================
- Skips data conversion (uses existing LeRobot dataset)
- Uses different DiffusionConfig settings:
  * use_group_norm=False (instead of default True)
  * pretrained_backbone_weights can be specified
  * crop_is_random=False (deterministic cropping)
- Simplified training pipeline

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic training with existing dataset
CUDA_VISIBLE_DEVICES=0 python scripts/32_train_lerobot_dataset.py \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out runs/diffusion_A_mark_v2 \
    --steps 50000 \
    --batch_size 64 \
    --lr 1e-4

# With pretrained backbone
CUDA_VISIBLE_DEVICES=0 python scripts/32_train_lerobot_dataset.py \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out runs/diffusion_A_mark_v2 \
    --steps 50000 \
    --batch_size 64 \
    --lr 1e-4 \
    --pretrained_backbone

# Multi-GPU training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc_per_node=6 \
    scripts/32_train_lerobot_dataset.py \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out runs/diffusion_A_mark_v2 \
    --steps 10000 \
    --batch_size 64 \
    --lr 1e-4 \
    --wandb

# Resume training
CUDA_VISIBLE_DEVICES=0 python scripts/32_train_lerobot_dataset.py \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out runs/diffusion_A_mark_v2 \
    --resume

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy from existing LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Input/Output Arguments
    # =========================================================================
    parser.add_argument(
        "--lerobot_dataset",
        type=str,
        required=True,
        help="Path to the existing LeRobot dataset directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/diffusion_lerobot",
        help="Output directory for saving model checkpoints and logs.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="LeRobot repo_id to use. If not specified, inferred from dataset.",
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
    parser.add_argument(
        "--pretrained_backbone",
        action="store_true",
        help="Use pretrained ImageNet weights for vision backbone.",
    )
    parser.add_argument(
        "--use_group_norm",
        action="store_true",
        help="Use group normalization instead of batch normalization.",
    )
    parser.add_argument(
        "--crop_is_random",
        action="store_true",
        help="Use random cropping during training (default: deterministic center crop).",
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
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing checkpoints directory if it exists.",
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
        default="rev2fwd-diffusion-v2",
        help="WandB project name. Default: rev2fwd-diffusion-v2.",
    )

    return parser.parse_args()


def load_dataset_info(dataset_dir: Path) -> dict:
    """Load dataset metadata from info.json."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info.json not found at {info_path}")
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    return info


def train_with_lerobot(args: argparse.Namespace) -> None:
    """Train using LeRobot's Python API with train_dp_zarr style configuration."""
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.scripts.lerobot_train import train

    dataset_dir = Path(args.lerobot_dataset)
    out_dir = Path(args.out)
    
    # Load dataset info
    info = load_dataset_info(dataset_dir)
    features = info["features"]
    
    print("\n" + "=" * 60)
    print("Loading Dataset Info")
    print("=" * 60)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Total episodes: {info['total_episodes']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    
    # Extract feature dimensions
    # Image features
    image_keys = []
    image_shapes = {}
    for key, feat in features.items():
        if key.startswith("observation.") and feat["dtype"] in ["video", "image"]:
            image_keys.append(key)
            image_shapes[key] = tuple(feat["shape"])  # (C, H, W)
            print(f"  {key}: {feat['shape']}")
    
    # State feature
    state_shape = tuple(features["observation.state"]["shape"])
    print(f"  observation.state: {state_shape}")
    
    # Action feature
    action_shape = tuple(features["action"]["shape"])
    print(f"  action: {action_shape}")
    print("=" * 60)
    
    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Build input features
    input_features = {}
    for key in image_keys:
        shape = image_shapes[key]
        input_features[key] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=shape,
        )
    
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE,
        shape=state_shape,
    )
    
    # Build output features
    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=action_shape,
        ),
    }
    
    # Determine pretrained weights
    pretrained_weights = None
    if args.pretrained_backbone:
        if args.vision_backbone == "resnet18":
            pretrained_weights = "ResNet18_Weights.IMAGENET1K_V1"
        elif args.vision_backbone == "resnet34":
            pretrained_weights = "ResNet34_Weights.IMAGENET1K_V1"
        elif args.vision_backbone == "resnet50":
            pretrained_weights = "ResNet50_Weights.IMAGENET1K_V1"
        else:
            print(f"[WARNING] No pretrained weights for backbone: {args.vision_backbone}")
    
    # Create Diffusion Policy configuration
    # Key differences from script 31:
    # - use_group_norm=False by default (unless --use_group_norm specified)
    # - crop_is_random=False by default (unless --crop_is_random specified)
    # - Can use pretrained backbone weights
    policy_cfg = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        vision_backbone=args.vision_backbone,
        use_group_norm=args.use_group_norm,
        pretrained_backbone_weights=pretrained_weights,
        crop_shape=tuple(args.crop_shape),
        crop_is_random=args.crop_is_random,
        num_train_timesteps=args.num_train_timesteps,
        device=device,
        push_to_hub=False,
        optimizer_lr=args.lr,
    )
    
    # Print policy configuration
    print("\n" + "=" * 60)
    print("Diffusion Policy Configuration")
    print("=" * 60)
    print(f"  n_obs_steps: {args.n_obs_steps}")
    print(f"  horizon: {args.horizon}")
    print(f"  n_action_steps: {args.n_action_steps}")
    print(f"  vision_backbone: {args.vision_backbone}")
    print(f"  use_group_norm: {args.use_group_norm}")
    print(f"  pretrained_backbone_weights: {pretrained_weights}")
    print(f"  crop_shape: {tuple(args.crop_shape)}")
    print(f"  crop_is_random: {args.crop_is_random}")
    print(f"  num_train_timesteps: {args.num_train_timesteps}")
    print(f"  Normalization mapping:")
    for feat_type, norm_mode in policy_cfg.normalization_mapping.items():
        print(f"    {feat_type}: {norm_mode}")
    print("=" * 60)
    
    # Determine repo_id
    repo_id = args.repo_id
    if repo_id is None:
        # Try to infer from dataset path
        repo_id = "local/rev2fwd_diffusion"
    
    # Create dataset configuration
    dataset_cfg = DatasetConfig(
        repo_id=repo_id,
        root=str(dataset_dir),
    )
    
    # WandB configuration
    wandb_cfg = WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
    )
    
    # Output directory setup
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    
    # Check if running in distributed mode
    is_main_process = True
    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    
    # Handle overwrite
    if is_main_process and args.overwrite and checkpoint_dir.exists() and not args.resume:
        print(f"[INFO] Removing existing checkpoints at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
    
    # Sync processes if distributed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Create training pipeline configuration
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
        eval_freq=0,
    )
    
    # Print training info
    print("\n" + "=" * 60)
    print("Starting Diffusion Policy Training")
    print("=" * 60)
    print(f"  LeRobot dataset: {dataset_dir}")
    print(f"  Output: {checkpoint_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  WandB: {args.wandb}")
    print(f"  Resume: {args.resume}")
    print("=" * 60 + "\n")
    
    # Run training
    train(train_cfg)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Output directory: {checkpoint_dir}")
    print(f"  Total steps: {args.steps}")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    
    # Validate dataset path
    dataset_dir = Path(args.lerobot_dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"LeRobot dataset not found: {dataset_dir}")
    
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Invalid LeRobot dataset: missing {info_path}\n"
            "Make sure this is a valid LeRobot v3.0 dataset."
        )
    
    # Initialize distributed if needed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # Train
    train_with_lerobot(args)


if __name__ == "__main__":
    main()
