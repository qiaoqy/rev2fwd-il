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
CUDA_VISIBLE_DEVICES=2

# Step 1: Convert data only (recommended for first run)
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --convert_only

# Step 2: Train with converted data
python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A1 \
    --steps 5000 \
    --batch_size 2048 \
    --lr 0.0005

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
CUDA_VISIBLE_DEVICES=1,4,5,6 torchrun --nproc_per_node=4 scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_images.npz \
    --out runs/diffusion_A \
    --batch_size 2048 \
    --steps 5000 \
    --lr 0.0005

CUDA_VISIBLE_DEVICES=2 python scripts/31_train_A_diffusion.py --dataset data/A_forward_with_2images.npz --out runs/diffusion_A_2cam_3 --num_episodes 500 -
-batch_size 1024 --steps 3000 --lr 0.0005 --include_obj_pose 
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

# Suppress verbose output from video encoders (SVT-AV1, FFmpeg, etc.)
os.environ["AV_LOG_LEVEL"] = "quiet"
os.environ["SVT_LOG"] = "0"  # Suppress SVT-AV1 encoder logs
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


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
    parser.add_argument(
        "--include_obj_pose",
        action="store_true",
        help="Include object pose (7D) in observation.state. "
             "If enabled, state becomes 14D (ee_pose + obj_pose).",
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=-1,
        help="Number of episodes to use for training. Default: -1 (use all episodes).",
    )
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
        default=[128, 128],
        help="Image crop shape (H, W). Default: 128 128.",
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps. Default: 100.",
    )
    parser.add_argument(
        "--pretrained_backbone_weights",
        type=str,
        default= None,
        help="Pretrained weights for vision backbone (e.g., 'ResNet18_Weights.IMAGENET1K_V1'). Default: None.",
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
        help="Overwrite existing checkpoints directory if it exists. Default: True. Use --no-overwrite to disable.",
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


def load_episodes_from_npz(path: str, num_episodes: int = -1) -> list[dict]:
    """Load episodes from NPZ file created by script 22.
    
    Args:
        path: Path to the NPZ file.
        num_episodes: Number of episodes to load. -1 means load all.
        
    Returns:
        List of episode dictionaries.
    """
    path = Path(path)
    print(f"Loading NPZ file: {path} ...")
    load_start = time.time()
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    load_time = time.time() - load_start
    total_episodes = len(episodes)
    
    # Limit number of episodes if specified
    if num_episodes > 0:
        episodes = episodes[:num_episodes]
        print(f"Loaded {len(episodes)}/{total_episodes} episodes from {path} "
              f"(limited by --num_episodes) in {load_time:.1f}s")
    else:
        print(f"Loaded {len(episodes)} episodes from {path} in {load_time:.1f}s")
    
    return episodes


def convert_npz_to_lerobot_format(
    npz_path: str,
    output_dir: str,
    fps: int = 20,
    repo_id: str = "local/rev2fwd_diffusion",
    force: bool = False,
    num_episodes: int = -1,
    include_obj_pose: bool = False,
) -> tuple[int, int, bool]:
    """Convert NPZ dataset to LeRobot v3.0 format.
    
    Args:
        npz_path: Path to input NPZ file.
        output_dir: Directory to save LeRobot dataset.
        fps: Frames per second (should match data collection).
        repo_id: Repository ID for the dataset.
        force: Force re-conversion even if dataset exists.
        num_episodes: Number of episodes to use. -1 means use all.
        include_obj_pose: Whether to include object pose in observation.state.
        
    Returns:
        Tuple of (image_height, image_width, has_wrist_camera).
    """
    # Suppress verbose logging from video encoding libraries
    import logging
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    logging.getLogger("av").setLevel(logging.ERROR)
    
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    output_dir = Path(output_dir)
    
    # Check if dataset already exists BEFORE loading NPZ
    if output_dir.exists() and not force:
        print(f"LeRobot dataset already exists at {output_dir}")
        print("Skipping NPZ loading and conversion. Use --force_convert to re-convert.")
        # Read metadata from existing dataset
        meta_path = output_dir / "meta" / "info.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                info = json.load(f)
            # Extract image shape from features
            features = info.get("features", {})
            if "observation.image" in features:
                img_shape = features["observation.image"]["shape"]  # (C, H, W)
                image_height, image_width = img_shape[1], img_shape[2]
            else:
                # Fallback: try to read from video files
                image_height, image_width = 128, 128  # default
            has_wrist = "observation.wrist_image" in features
            print(f"  Loaded metadata: image=({image_height}, {image_width}), has_wrist={has_wrist}")
            return image_height, image_width, has_wrist
        else:
            # Fallback if meta file doesn't exist - need to load NPZ
            print("  Warning: meta/info.json not found, loading NPZ to get metadata...")
    
    # Load episodes (only if we need to convert or get metadata)
    episodes = load_episodes_from_npz(npz_path, num_episodes=num_episodes)
    image_shape = episodes[0]["images"].shape[1:]  # (H, W, 3)
    has_wrist = "wrist_images" in episodes[0]
    
    # If dataset exists and we already loaded NPZ just for metadata, return early
    if output_dir.exists() and not force:
        return image_shape[0], image_shape[1], has_wrist
    
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
    
    if len(episodes) == 0:
        raise ValueError("No episodes found in NPZ file!")
    
    # Get data dimensions from first episode
    ep0 = episodes[0]
    if include_obj_pose:
        state_dim = 14  # ee_pose (7) + obj_pose (7)
        state_names = [
            "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
            "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
        ]
    else:
        state_dim = 7  # ee_pose: [x, y, z, qw, qx, qy, qz]
        state_names = ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
    action_dim = 8  # ee_pose (7) + gripper (1)
    
    print(f"  Table camera image shape: {image_shape} (H, W, C)")
    if has_wrist:
        wrist_shape = ep0["wrist_images"].shape[1:]
        print(f"  Wrist camera image shape: {wrist_shape} (H, W, C)")
    else:
        print(f"  Wrist camera: not available")
    print(f"  State dim: {state_dim} (ee_pose{' + obj_pose' if include_obj_pose else ''})")
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
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "gripper"],
        },
    }
    
    # Add wrist camera feature if available
    if has_wrist:
        wrist_shape = ep0["wrist_images"].shape[1:]
        features["observation.wrist_image"] = {
            "dtype": "video",
            "shape": (3, wrist_shape[0], wrist_shape[1]),  # (C, H, W)
            "names": ["channel", "height", "width"],
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
    num_episodes = len(episodes)
    
    # Determine print frequency based on dataset size
    print_freq = max(1, num_episodes // 20)  # Print ~20 times during conversion
    
    print(f"\nProcessing {num_episodes} episodes...")
    
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        
        # Print progress at start of each episode (or at intervals for large datasets)
        if num_episodes <= 50 or (ep_idx + 1) % print_freq == 0 or ep_idx == 0:
            elapsed = time.time() - start_time
            rate = (ep_idx / elapsed) if elapsed > 0 and ep_idx > 0 else 0
            eta = (num_episodes - ep_idx) / rate if rate > 0 else 0
            print(f"  [{ep_idx + 1}/{num_episodes}] Processing episode {ep_idx}, "
                  f"{T} frames | {rate:.1f} ep/s | ETA: {eta:.0f}s")
        
        # Extract data
        images = ep["images"]  # (T, H, W, 3) uint8
        ee_pose = ep["ee_pose"]  # (T, 7)
        obj_pose = ep["obj_pose"]  # (T, 7)
        gripper = ep["gripper"]  # (T,)
        wrist_images = ep.get("wrist_images", None)  # (T, H, W, 3) uint8 or None
        
        # Create action labels: next ee_pose + gripper
        # For the last frame, we repeat the last action
        for t in range(T - 1):
            # Current observation
            img = images[t]  # (H, W, 3)
            if include_obj_pose:
                state = np.concatenate([ee_pose[t], obj_pose[t]])  # (14,)
            else:
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
            
            # Add wrist camera image if available
            if wrist_images is not None:
                frame["observation.wrist_image"] = wrist_images[t]
            
            dataset.add_frame(frame)
        
        # Save episode
        dataset.save_episode()
        total_frames += T - 1
    
    # Finalize dataset
    print("\nFinalizing dataset (encoding videos)...")
    dataset.finalize()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Dataset saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return image_shape[0], image_shape[1], has_wrist


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
    has_wrist: bool = False,
    include_obj_pose: bool = False,
) -> dict:
    """Train using LeRobot's Python API.
    
    Args:
        args: Parsed command-line arguments.
        lerobot_dataset_dir: Path to LeRobot dataset.
        image_height: Image height.
        image_width: Image width.
        has_wrist: Whether the dataset has wrist camera images.
        include_obj_pose: Whether object pose is included in observation.state.
        
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
    
    # Check if running in distributed mode
    is_main_process = True
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        is_main_process = (local_rank == 0)
    
    # Handle overwrite: remove existing checkpoints directory (only on main process)
    checkpoint_dir = out_dir / "checkpoints"
    if is_main_process and args.overwrite and checkpoint_dir.exists() and not args.resume:
        print(f"Removing existing checkpoints directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
    
    # Sync all processes before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine state dimension based on whether obj_pose is included
    state_dim = 14 if include_obj_pose else 7
    
    # Configure input/output features for the policy
    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_height, image_width),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        ),
    }
    
    # Add wrist camera feature if available
    if has_wrist:
        input_features["observation.wrist_image"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, image_height, image_width),
        )
    
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
        pretrained_backbone_weights=args.pretrained_backbone_weights,
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
    
    # When resuming, check that checkpoints exist
    if args.resume:
        checkpoints_subdir = checkpoint_dir / "checkpoints"
        if not checkpoints_subdir.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint directory {checkpoints_subdir} does not exist. "
                f"Please run training without --resume first."
            )
        checkpoint_dirs = sorted(checkpoints_subdir.iterdir())
        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"Cannot resume: no checkpoints found in {checkpoints_subdir}. "
                f"Please run training without --resume first."
            )
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    
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
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Include obj_pose: {include_obj_pose}")
    print(f"  State dim: {state_dim}")
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
    
    # Check if running in distributed mode
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (local_rank == 0)
    
    # Initialize distributed process group if needed (for barrier synchronization)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # Setup paths
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lerobot_dataset_dir = args.lerobot_dataset_dir
    if lerobot_dataset_dir is None:
        lerobot_dataset_dir = out_dir / "lerobot_dataset"
    lerobot_dataset_dir = Path(lerobot_dataset_dir)
    
    # Step 1: Convert data to LeRobot format (only on main process)
    # Other processes wait at the barrier until conversion is done
    if not args.skip_convert:
        if is_main_process:
            image_height, image_width, has_wrist = convert_npz_to_lerobot_format(
                npz_path=args.dataset,
                output_dir=lerobot_dataset_dir,
                fps=args.fps,
                repo_id="local/rev2fwd_diffusion",
                force=args.force_convert,
                num_episodes=args.num_episodes,
                include_obj_pose=args.include_obj_pose,
            )
            # Write metadata to a temp file so other processes can read it
            import json
            meta_file = out_dir / ".conversion_meta.json"
            with open(meta_file, "w") as f:
                json.dump({"image_height": image_height, "image_width": image_width, "has_wrist": has_wrist, "include_obj_pose": args.include_obj_pose}, f)
        
        # Synchronize all processes - wait for conversion to complete
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Non-main processes read metadata from file
        if not is_main_process:
            import json
            meta_file = out_dir / ".conversion_meta.json"
            with open(meta_file, "r") as f:
                meta = json.load(f)
            image_height = meta["image_height"]
            image_width = meta["image_width"]
            has_wrist = meta["has_wrist"]
            # Note: include_obj_pose comes from args, not meta file
    else:
        # Load episodes to get image shape and check for wrist camera
        episodes = load_episodes_from_npz(args.dataset, num_episodes=args.num_episodes)
        image_shape = episodes[0]["images"].shape[1:]  # (H, W, 3)
        image_height, image_width = image_shape[0], image_shape[1]
        has_wrist = "wrist_images" in episodes[0]
    
    if args.convert_only:
        if is_main_process:
            print("Data conversion complete. Exiting (--convert_only flag).")
        return
    
    # Step 2: Train the policy
    result = train_with_lerobot_api(
        args=args,
        lerobot_dataset_dir=lerobot_dataset_dir,
        image_height=image_height,
        image_width=image_width,
        has_wrist=has_wrist,
        include_obj_pose=args.include_obj_pose,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Total steps: {result['steps']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

