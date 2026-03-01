#!/usr/bin/env python3
"""Step 7: Train DiT Flow Policy using LeRobot.

This script trains a vision-based DiT Flow Policy (Diffusion Transformer with Flow Matching)
on the teleoperation data collected by script 1 (1_teleop_ps5_controller.py).

This is based on the paper: "DiT Policy: Scaling Diffusion Policy with Transformers"
The key difference from standard Diffusion Policy is:
- Uses Diffusion Transformer (DiT) instead of U-Net
- Uses Flow Matching instead of DDPM/DDIM
- Generally faster inference with similar or better quality

=============================================================================
DEPENDENCIES
=============================================================================
Requires:
  - lerobot >= 0.4.3 (installed from source):
        git clone https://github.com/danielsanjosepro/lerobot.git
        pip install -e ./lerobot

  - lerobot_policy_ditflow >= 0.1.0 (installed from source):
        git clone https://github.com/danielsanjosepro/lerobot_policy_ditflow.git
        pip install -e ./lerobot_policy_ditflow

The lerobot_policy_ditflow library registers itself as a third-party plugin
via the lerobot plugin system (register_third_party_plugins). On import, it
registers the "ditflow" policy type via @PreTrainedConfig.register_subclass.

To verify installation:
    python -c "from lerobot.utils.import_utils import register_third_party_plugins; \
               register_third_party_plugins(); \
               from lerobot.configs.policies import PreTrainedConfig; \
               print(PreTrainedConfig.get_known_choices().keys())"
    # Should include 'ditflow' in the output

=============================================================================
INPUT DATA FORMAT (from script 1)
=============================================================================
Data directory containing episode_*.tar.gz files and metadata.json.

Each episode tar.gz contains:
    episode_XXXX/
        episode_data.npz    # Numeric data
        fixed_cam/          # Front camera images (000000.png, ...)
        wrist_cam/          # Wrist camera images (000000.png, ...)

episode_data.npz structure:
    - ee_pose:       (T, 6)   EE poses [x, y, z, rx, ry, rz] in meters/radians
    - gripper_state: (T,)     Gripper state [0-1] (0=closed, 1=open)
    - action:        (T, 7)   RELATIVE actions [delta_xyz, delta_rpy, gripper_target] in meters/radians
    - timestamp:     (T,)     Timestamps
    - joint_angles:  (T, 6)   Joint angles in radians
    - joint_torques: (T, 6)   Joint torques in N·m
    - ee_force:      (T, 6)   End-effector force [Fx,Fy,Fz,Mx,My,Mz]
    - success:       bool     Whether episode was successful
    - episode_id:    str      Unique episode identifier

=============================================================================
OBSERVATION STATE DIMENSION OPTIONS
=============================================================================
The observation.state can include different combinations of features:

  Flags                      | state_dim | Components
  ---------------------------|-----------|---------------------------
  (none)                     |     6     | ee_pose(6)
  --include_gripper          |     7     | ee_pose(6) + gripper(1)

Default: state_dim=6 (ee_pose only)

=============================================================================
ACTION FORMAT (Relative Delta)
=============================================================================
The action output is 7D relative delta:
    [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper_target]

Where:
    - delta_xyz: Position change in meters
    - delta_rpy: Euler angle change (rx, ry, rz) in radians
    - gripper_target: Target gripper state [0-1]

=============================================================================
IMAGE HANDLING
=============================================================================
If fixed camera and wrist camera have different resolutions, all images are
resized to a common size (default: 240x320) during conversion to ensure
compatibility with the DiT Flow policy's multi-camera stacking.

=============================================================================
DITFLOW vs DIFFUSION POLICY
=============================================================================
DiT Flow uses:
- Transformer architecture instead of U-Net (better scaling)
- Flow Matching objective instead of DDPM (faster inference, ~100 steps)
- AdaLN-Zero conditioning for time step (more stable training)

=============================================================================
USAGE EXAMPLES
=============================================================================
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_A_ps5collected \
    --out runs/pickplace_piper_0226_A_ps5collected  \
    --save_freq 10000 --lr 5e-4\
    --batch_size 128 --steps 40000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_B_reversed \
    --out runs/pickplace_piper_0226_B_reversed  \
    --save_freq 10000 --lr 5e-4\
    --batch_size 128 --steps 40000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29503\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_B \
    --out runs/pickplace_piper_0226_B  \
    --save_freq 10000 --lr 5e-4\
    --batch_size 128 --steps 40000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29504\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_A \
    --out runs/pickplace_piper_0226_A  \
    --save_freq 10000 --lr 5e-4\
    --batch_size 128 --steps 40000 --wandb --include_gripper









# Basic training
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0210_B \
    --out runs/ditflow_piper_0210_B \
    --batch_size 128 --steps 50000 --wandb

CUDA_VISIBLE_DEVICES=3 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0210_A \
    --out runs/ditflow_piper_0210_A \
    --batch_size 128 --steps 50000 --wandb

# With gripper in observation (state_dim=8)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --batch_size 64 --steps 50000 \
    --include_gripper --wandb

# With XYZ visualization during training
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --batch_size 64 --steps 50000 \
    --include_gripper --enable_xyz_viz --viz_save_freq 5000 --wandb

# Custom DiT architecture (smaller model for faster training)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --hidden_dim 256 --num_blocks 4 --num_heads 8 --dim_feedforward 2048 \
    --batch_size 64 --steps 50000 --wandb --include_gripper

# Use beta noise scheduling (from Pi0 paper)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --training_noise_sampling beta \
    --batch_size 64 --steps 50000 --wandb --include_gripper

# Multi-GPU training
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0210_B \
    --out runs/ditflow_pickplace_piper_0210_B  \
    --batch_size 128 --steps 50000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0210_A \
    --out runs/ditflow_pickplace_piper_0210_A  \
    --batch_size 128 --steps 50000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=5,6,7,8,9 torchrun --nproc_per_node=5 --master_port=29501\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_A \
    --out runs/ditflow_pickplace_piper_0221_A  \
    --batch_size 64 --steps 100000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_A \
    --out runs/ditflow_pickplace_piper_0224_A  \
    --save_freq 20000 --lr 5e-4\
    --batch_size 64 --steps 500000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_B \
    --out runs/ditflow_pickplace_piper_0221_B  \
    --batch_size 32 --steps 100000 --wandb --include_gripper

    # A100 G1
CUDA_VISIBLE_DEVICES=1 python\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_A \
    --out runs/ditflow_pickplace_piper_0221_A  \
    --batch_size 128 --steps 100000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=1 python\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_B \
    --out runs/ditflow_pickplace_piper_0221_B_0222  \
    --batch_size 128 --steps 10000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=7 python\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_A \
    --out runs/ditflow_pickplace_piper_0221_A_0222  \
    --batch_size 128 --steps 10000 --wandb --include_gripper

    # 3090 G5
CUDA_VISIBLE_DEVICES=1 python\
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0221_B \
    --out runs/ditflow_pickplace_piper_0221_B  \
    --batch_size 16 --steps 20000 --wandb --include_gripper
    
# Data conversion only (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --convert_only --include_gripper

# Overfit mode (1 episode, useful for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_overfit \
    --overfit --steps 1000 --include_gripper

# Resume training from checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pick_place_piper_A \
    --out runs/ditflow_piper_teleop \
    --resume --steps 200000 --include_gripper

CUDA_VISIBLE_DEVICES=0 python scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0210_A \
    --out runs/ditflow_pickplace_piper_0210_A  \
    --batch_size 128 --steps 50000 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=1 python \
    scripts/scripts_piper_local/7_train_ditflow.py  \
    --dataset data/pickplace_piper_0221_B_extended_0222   \
    --out runs/ditflow_pickplace_piper_0222_B_extended_0222   \
    --batch_size 128 --steps 11333 --wandb --include_gripper

CUDA_VISIBLE_DEVICES=1 python \
    scripts/scripts_piper_local/7_train_ditflow.py  \
    --dataset data/pickplace_piper_0221_B_extended_0222   \
    --out runs/ditflow_pickplace_piper_0222_B_extended_0222   \
    --batch_size 128 --steps 11333 --wandb --include_gripper
=============================================================================


=============================================================================
TMUX BACKGROUND EXECUTION (Recommended for long training)
=============================================================================
# Create a new tmux session and run training in background
tmux new -s ditflow_pickplace_piper_0224_A
tmux new -s ditflow_pickplace_piper_0210_B
# Common tmux commands:
#   tmux ls                  # List all sessions
#   tmux a -t train          # Attach to session named "train"
#   tmux kill-session -t train  # Kill session named "train"
#   Ctrl+b d                 # Detach from current session (inside tmux)
#   Ctrl+b [                 # Enter scroll mode (q to exit)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

# Suppress verbose output from video encoders
os.environ["AV_LOG_LEVEL"] = "quiet"
os.environ["SVT_LOG"] = "0"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DiT Flow Policy on Piper teleoperation data with relative actions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Input/Output Arguments
    # =========================================================================
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/pick_place_piper_A",
        help="Path to the data directory containing episode_*.tar.gz files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/ditflow_piper_teleop",
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
        "--include_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include gripper state (1D) in observation.state. "
             "If enabled, state includes gripper open/close state. Default: True. "
             "Use --no-include_gripper to disable.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[240, 320],
        help="Target image size (H, W) for both cameras. Images will be resized "
             "to this size during conversion. Default: 240 320.",
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
        default=8,
        help="Number of dataloader workers. Default: 8.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second of the dataset. Default: 20.",
    )
    parser.add_argument(
        "--convert_workers",
        type=int,
        default=-1,
        help="Number of parallel workers for data conversion. "
             "-1 means auto (min of cpu_count, num_episodes, 16). Default: -1.",
    )

    # =========================================================================
    # DiT Flow Policy Architecture
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
        help="DiT Flow horizon (action sequence length). Default: 16.",
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
        "--crop_ratio",
        type=float,
        nargs=2,
        default=[0.95, 0.95],
        help="Crop ratio (H_ratio, W_ratio) relative to image_size. "
             "Actual crop shape = (int(H * H_ratio), int(W * W_ratio)). "
             "Default: 0.95 0.95.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="Number of flow integration steps for inference. Default: 100.",
    )
    parser.add_argument(
        "--pretrained_backbone_weights",
        type=str,
        default=None,
        help="Pretrained weights for vision backbone. Default: None.",
    )
    
    # DiT specific parameters
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the DiT transformer. Default: 512.",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=6,
        help="Number of transformer blocks in DiT. Default: 6.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="Number of attention heads in DiT. Default: 16.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in DiT. Default: 0.1.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=4096,
        help="Feedforward dimension in DiT. Default: 4096.",
    )
    parser.add_argument(
        "--training_noise_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "beta"],
        help="Noise sampling strategy for training. 'beta' is from Pi0 paper. Default: uniform.",
    )

    # =========================================================================
    # Logging and Checkpointing
    # =========================================================================
    parser.add_argument(
        "--log_freq",
        type=int,
        default=50,
        help="Log metrics every N steps. Default: 50.",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=20000,
        help="Save checkpoint every N steps. Default: 20000.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing checkpoints. Default: True. Use --no-overwrite to disable.",
    )
    parser.add_argument(
        "--viz_save_freq",
        type=int,
        default=20000,
        help="Save XYZ visualization every N steps. Default: 20000.",
    )
    parser.add_argument(
        "--enable_xyz_viz",
        action="store_true",
        help="Enable XYZ curve visualization during training.",
    )

    # =========================================================================
    # DiT Flow Sampling
    # =========================================================================
    parser.add_argument(
        "--clip_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip action samples to [-clip_sample_range, +clip_sample_range] during inference. "
             "Default: True. Use --no-clip_sample to disable.",
    )
    parser.add_argument(
        "--clip_sample_range",
        type=float,
        default=1.0,
        help="Range for clipping action samples during inference. Default: 1.0.",
    )

    # =========================================================================
    # Overfit Mode
    # =========================================================================
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Enable overfit mode: use only 1 episode. Automatically sets num_episodes=1.",
    )

    # =========================================================================
    # Data Augmentation (training-time only)
    # =========================================================================
    parser.add_argument(
        "--color_jitter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ColorJitter augmentation during training. Default: True. "
             "Use --no-color_jitter to disable.",
    )
    parser.add_argument(
        "--color_jitter_brightness",
        type=float,
        default=0.05,
        help="ColorJitter brightness range. Default: 0.05.",
    )
    parser.add_argument(
        "--color_jitter_contrast",
        type=float,
        default=0.05,
        help="ColorJitter contrast range. Default: 0.05.",
    )
    parser.add_argument(
        "--color_jitter_saturation",
        type=float,
        default=0.05,
        help="ColorJitter saturation range. Default: 0.05.",
    )
    parser.add_argument(
        "--color_jitter_hue",
        type=float,
        default=0.05,
        help="ColorJitter hue range. Default: 0.05.",
    )
    parser.add_argument(
        "--gaussian_blur",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable GaussianBlur augmentation during training. Default: False. "
             "Use --gaussian_blur to enable.",
    )
    parser.add_argument(
        "--gaussian_blur_kernel_size",
        type=int,
        default=3,
        help="GaussianBlur kernel size (must be odd). Default: 3.",
    )
    parser.add_argument(
        "--gaussian_blur_sigma",
        type=float,
        nargs=2,
        default=[0.1, 0.5],
        help="GaussianBlur sigma range (low, high). Default: 0.1 0.5.",
    )
    parser.add_argument(
        "--random_sharpness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RandomAdjustSharpness augmentation during training. Default: False. "
             "Use --random_sharpness to enable.",
    )
    parser.add_argument(
        "--random_sharpness_factor",
        type=float,
        default=1.2,
        help="RandomAdjustSharpness sharpness factor. Default: 1.2.",
    )
    parser.add_argument(
        "--random_sharpness_p",
        type=float,
        default=0.3,
        help="Probability of applying RandomAdjustSharpness. Default: 0.3.",
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
        default="piper-ditflow-teleop",
        help="WandB project name. Default: piper-ditflow-teleop.",
    )

    # =========================================================================
    # Validation Set
    # =========================================================================
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Fraction of episodes for validation (0.0 to 1.0). Default: 0.0.",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=500,
        help="Compute validation loss every N steps. Default: 500.",
    )

    return parser.parse_args()


def load_episode_from_archive(archive_path: Path) -> dict:
    """Load episode data from tar.gz archive.
    
    Args:
        archive_path: Path to tar.gz archive (e.g., episode_XXXX.tar.gz)
        
    Returns:
        Dictionary containing episode data with:
        - ee_pose: (T, 6) array [x, y, z, rx, ry, rz] in meters/radians
        - action: (T, 7) array [delta_xyz, delta_rpy, gripper] in meters/radians
        - gripper_state: (T,) array
        - images: list of (H, W, 3) arrays (fixed camera)
        - wrist_images: list of (H, W, 3) arrays or None
        - success: bool
        - episode_id: str
        - num_timesteps: int
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(temp_path)
        
        # Find the episode directory inside the extracted content
        extracted_dirs = list(temp_path.iterdir())
        if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
            episode_dir = extracted_dirs[0]
        else:
            # Assume the episode name matches the archive name
            episode_name = archive_path.stem.replace('.tar', '')
            episode_dir = temp_path / episode_name
        
        if not episode_dir.exists():
            raise FileNotFoundError(f"Could not find episode directory in archive: {archive_path}")
        
        # Load npz data
        npz_path = episode_dir / "episode_data.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Episode data not found: {npz_path}")
        
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract numeric data
        result = {
            'ee_pose': data['ee_pose'].astype(np.float32),
            'action': data['action'].astype(np.float32),
            'gripper_state': data['gripper_state'].astype(np.float32),
            'success': bool(data['success']),
            'episode_id': str(data['episode_id'].item() if data['episode_id'].ndim == 0 else data['episode_id']),
            'num_timesteps': int(data['num_timesteps']),
        }
        
        T = result['num_timesteps']
        
        # Load images from fixed_cam
        fixed_cam_dir = episode_dir / "fixed_cam"
        images = []
        for i in range(T):
            img_path = fixed_cam_dir / f"{i:06d}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                raise FileNotFoundError(f"Missing image: {img_path}")
        result['images'] = images
        
        # Load wrist camera images if available
        wrist_cam_dir = episode_dir / "wrist_cam"
        if wrist_cam_dir.exists():
            wrist_images = []
            for i in range(T):
                img_path = wrist_cam_dir / f"{i:06d}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    wrist_images.append(img)
                else:
                    # Missing wrist image, set to None
                    wrist_images.append(None)
            # Only keep if all images are present
            if all(img is not None for img in wrist_images):
                result['wrist_images'] = wrist_images
            else:
                result['wrist_images'] = None
        else:
            result['wrist_images'] = None
        
        return result


def _load_episode_worker(archive_path: Path) -> dict:
    """Worker function for parallel episode loading.
    
    This is a top-level function to be picklable for multiprocessing.
    """
    return load_episode_from_archive(archive_path)


def load_episodes_from_data_dir(
    data_dir: Path, 
    num_episodes: int = -1,
    num_workers: int = -1,
) -> list[dict]:
    """Load episodes from data directory containing tar.gz archives.
    
    Args:
        data_dir: Path to data directory (e.g., data/pick_place_piper)
        num_episodes: Number of episodes to load. -1 means load all.
        num_workers: Number of parallel workers. -1 means auto (cpu_count).
        
    Returns:
        List of episode dictionaries.
    """
    data_dir = Path(data_dir)
    
    # Find all episode archives
    episode_files = sorted(data_dir.glob("episode_*.tar.gz"))
    
    if len(episode_files) == 0:
        raise FileNotFoundError(f"No episode archives found in {data_dir}")
    
    total_episodes = len(episode_files)
    
    # Limit number of episodes if specified
    if num_episodes > 0:
        episode_files = episode_files[:num_episodes]
    
    # Determine number of workers
    if num_workers == -1:
        num_workers = min(mp.cpu_count(), len(episode_files), 16)  # Cap at 16 workers
    num_workers = max(1, num_workers)
    
    print(f"Loading {len(episode_files)}/{total_episodes} episodes from {data_dir} "
          f"using {num_workers} workers...")
    
    start_time = time.time()
    
    if num_workers == 1:
        # Serial loading (fallback)
        episodes = []
        for i, archive_path in enumerate(episode_files):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(episode_files) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i + 1}/{len(episode_files)}] Loading {archive_path.name} | "
                      f"{rate:.1f} ep/s | ETA: {eta:.0f}s")
            episode = load_episode_from_archive(archive_path)
            episodes.append(episode)
    else:
        # Parallel loading using ProcessPoolExecutor
        episodes = [None] * len(episode_files)
        completed = 0
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with their indices
            future_to_idx = {
                executor.submit(_load_episode_worker, path): idx 
                for idx, path in enumerate(episode_files)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    episodes[idx] = future.result()
                    completed += 1
                    
                    # Progress update
                    if completed % 10 == 0 or completed == 1 or completed == len(episode_files):
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(episode_files) - completed) / rate if rate > 0 else 0
                        print(f"  [{completed}/{len(episode_files)}] Loaded | "
                              f"{rate:.1f} ep/s | ETA: {eta:.0f}s")
                except Exception as e:
                    print(f"  Error loading episode {idx}: {e}")
                    raise
    
    load_time = time.time() - start_time
    total_frames = sum(ep['num_timesteps'] for ep in episodes)
    print(f"Loaded {len(episodes)} episodes ({total_frames} frames) in {load_time:.1f}s "
          f"({len(episodes)/load_time:.1f} ep/s with {num_workers} workers)")
    
    return episodes


def verify_lerobot_dataset_integrity(
    output_dir: Path,
    repo_id: str = "local/piper_ditflow",
    verify_all_episodes: bool = True,
) -> bool:
    """Verify integrity of a LeRobot dataset by attempting to decode video frames.
    
    This catches corruption from:
    - Incomplete video encoding (interrupted finalize/save_episode)
    - Silent PNG write failures leading to missing frames
    - Corrupt video concatenation
    - NFS transient I/O errors during encoding
    - AV1 codec encoding issues
    
    Args:
        output_dir: Path to the LeRobot dataset.
        repo_id: Repository ID.
        verify_all_episodes: If True, verify ALL episodes. If False, sample ~5.
        
    Returns:
        True if dataset is valid, False otherwise.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    output_dir = Path(output_dir)
    meta_path = output_dir / "meta" / "info.json"
    
    if not meta_path.exists():
        print(f"  [VERIFY] No metadata found at {meta_path}")
        return False
    
    try:
        with open(meta_path, "r") as f:
            info = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [VERIFY] Corrupted metadata: {e}")
        return False
    
    n_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    
    if n_episodes == 0 or total_frames == 0:
        print(f"  [VERIFY] Empty dataset (episodes={n_episodes}, frames={total_frames})")
        return False
    
    # Check video files exist and have non-zero size
    videos_dir = output_dir / "videos"
    if videos_dir.exists():
        for video_file in videos_dir.rglob("*.mp4"):
            if video_file.stat().st_size == 0:
                print(f"  [VERIFY] Empty video file: {video_file}")
                return False
    
    # Determine which episodes to verify
    if verify_all_episodes:
        episodes_to_check = list(range(n_episodes))
    else:
        episodes_to_check = sorted(set(
            [0, n_episodes - 1] + 
            list(np.linspace(0, n_episodes - 1, min(5, n_episodes), dtype=int))
        ))
    
    print(f"  [VERIFY] Checking {len(episodes_to_check)}/{n_episodes} episodes "
          f"({total_frames} total frames)...")
    
    failed_episodes = []
    for i, ep_idx in enumerate(episodes_to_check):
        try:
            ep_dataset = LeRobotDataset(
                repo_id=repo_id,
                root=output_dir,
                episodes=[ep_idx],
            )
            if len(ep_dataset) == 0:
                failed_episodes.append((ep_idx, "empty episode"))
                continue
            
            # Check first, middle, and last frame of each episode
            check_indices = sorted(set([0, len(ep_dataset) // 2, len(ep_dataset) - 1]))
            for frame_idx in check_indices:
                _ = ep_dataset[frame_idx]
            
            del ep_dataset
        except Exception as e:
            failed_episodes.append((ep_idx, str(e)))
        
        # Progress
        if (i + 1) % max(1, len(episodes_to_check) // 10) == 0:
            print(f"    [{i + 1}/{len(episodes_to_check)}] verified...")
    
    if failed_episodes:
        print(f"  [VERIFY] FAILED: {len(failed_episodes)} episodes have errors:")
        for ep_idx, err in failed_episodes[:10]:  # Show first 10
            print(f"    Episode {ep_idx}: {err[:200]}")
        if len(failed_episodes) > 10:
            print(f"    ... and {len(failed_episodes) - 10} more")
        return False
    
    print(f"  [VERIFY] All {len(episodes_to_check)} episodes verified successfully.")
    return True


def _clear_video_decoder_cache():
    """Clear LeRobot's module-level video decoder cache.
    
    CRITICAL: Must be called after any dataset verification/loading in the main
    process and before DataLoader forks workers. Otherwise, forked workers inherit
    stale cached file descriptors from the main process, causing 'Invalid data
    found when processing input' errors when multiple workers try to seek/read
    the same duplicated file descriptors concurrently.
    """
    try:
        from lerobot.datasets.video_utils import _default_decoder_cache
        cache_size = _default_decoder_cache.size()
        if cache_size > 0:
            _default_decoder_cache.clear()
            print(f"  [Cache] Cleared {cache_size} cached video decoders "
                  f"(prevents stale fd inheritance by DataLoader workers)")
    except (ImportError, AttributeError):
        pass


def convert_episodes_to_lerobot_format(
    data_dir: str,
    output_dir: str,
    fps: int = 20,
    repo_id: str = "local/piper_ditflow",
    force: bool = False,
    num_episodes: int = -1,
    include_gripper: bool = False,
    target_image_size: tuple[int, int] = (240, 320),
    num_workers: int = -1,
) -> tuple[int, int, bool]:
    """Convert episode data to LeRobot v3.0 format.
    
    Args:
        data_dir: Path to data directory with episode archives.
        output_dir: Directory to save LeRobot dataset.
        fps: Frames per second.
        repo_id: Repository ID for the dataset.
        force: Force re-conversion even if dataset exists.
        num_episodes: Number of episodes to use. -1 means use all.
        include_gripper: Whether to include gripper state in observation.state.
        target_image_size: Target (H, W) for all images. Images will be resized.
        num_workers: Number of parallel workers for loading episodes. -1 = auto.
        
    Returns:
        Tuple of (image_height, image_width, has_wrist_camera).
    """
    # Suppress verbose logging
    logging.getLogger("imageio").setLevel(logging.ERROR)
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    logging.getLogger("av").setLevel(logging.ERROR)
    
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    
    # Check if dataset already exists BEFORE loading
    if output_dir.exists() and not force:
        # Check if the dataset is complete (has valid metadata AND loadable data)
        meta_path = output_dir / "meta" / "info.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    info = json.load(f)
                features = info.get("features", {})
                if "observation.image" in features:
                    img_shape = features["observation.image"]["shape"]  # (C, H, W)
                    image_height, image_width = img_shape[1], img_shape[2]
                else:
                    image_height, image_width = 270, 320  # Default from metadata
                has_wrist = "observation.wrist_image" in features
                
                # Verify dataset integrity (check ALL episodes' video frames)
                print(f"Verifying existing dataset at {output_dir}...")
                try:
                    is_valid = verify_lerobot_dataset_integrity(
                        output_dir, repo_id=repo_id, verify_all_episodes=True,
                    )
                    # Clear decoder cache after verification to prevent stale
                    # file descriptors from being inherited by DataLoader workers
                    _clear_video_decoder_cache()
                    
                    if not is_valid:
                        raise RuntimeError("Dataset integrity check failed")
                    
                    # Dataset is valid
                    print(f"LeRobot dataset already exists at {output_dir}")
                    print("Skipping loading and conversion. Use --force_convert to re-convert.")
                    print(f"  Loaded metadata: image=({image_height}, {image_width}), has_wrist={has_wrist}")
                    return image_height, image_width, has_wrist
                except Exception as load_error:
                    print(f"Warning: Existing dataset has corrupted data files: {load_error}")
                    print(f"  Removing incomplete dataset at {output_dir} and re-converting...")
                    shutil.rmtree(output_dir)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Existing dataset has corrupted metadata: {e}")
                print(f"  Removing incomplete dataset at {output_dir} and re-converting...")
                shutil.rmtree(output_dir)
        else:
            # Dataset directory exists but no metadata - incomplete conversion
            print(f"Warning: Found incomplete dataset at {output_dir} (no metadata)")
            print(f"  Removing and re-converting...")
            shutil.rmtree(output_dir)
    
    # Load episodes (using parallel workers)
    episodes = load_episodes_from_data_dir(
        data_dir, 
        num_episodes=num_episodes,
        num_workers=num_workers,
    )
    
    if len(episodes) == 0:
        raise ValueError(f"No episodes loaded from {data_dir}")
    
    # Get data dimensions from first episode
    ep0 = episodes[0]
    original_image_shape = np.array(ep0['images'][0]).shape  # (H, W, 3)
    has_wrist = ep0.get('wrist_images') is not None
    
    # Use target image size for all cameras (ensures they can be stacked)
    image_height, image_width = target_image_size
    
    # Check if wrist camera has different size
    if has_wrist:
        original_wrist_shape = np.array(ep0['wrist_images'][0]).shape
        if original_wrist_shape[:2] != original_image_shape[:2]:
            print(f"  NOTE: Fixed camera ({original_image_shape[:2]}) and wrist camera ({original_wrist_shape[:2]}) "
                  f"have different sizes.")
            print(f"        All images will be resized to ({image_height}, {image_width})")
    
    # State dimension
    state_dim = 6  # ee_pose (6): [x, y, z, rx, ry, rz]
    state_names = ["ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"]
    if include_gripper:
        state_dim += 1
        state_names.append("gripper")
    
    # Action dimension: 7 for relative delta
    action_dim = 7
    
    # Remove existing dataset if force conversion
    if output_dir.exists() and force:
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)
    
    print(f"\n{'='*60}")
    print("Converting episodes to LeRobot format (Relative Action)")
    print(f"{'='*60}")
    print(f"  Input: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  FPS: {fps}")
    print(f"  Original image shape: {original_image_shape} (H, W, C)")
    print(f"  Target image size: ({image_height}, {image_width})")
    if has_wrist:
        print(f"  Original wrist camera shape: {original_wrist_shape}")
        print(f"  Wrist images will be resized to: ({image_height}, {image_width})")
    else:
        print(f"  Wrist camera: disabled")
    print(f"  State dim: {state_dim} (ee_pose" + (" + gripper" if include_gripper else "") + ")")
    print(f"  Action dim: {action_dim} (relative delta)")
    print(f"{'='*60}\n")
    
    # Define features for LeRobot dataset (using target size)
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, image_height, image_width),  # (C, H, W)
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
            "names": ["delta_x", "delta_y", "delta_z", 
                      "delta_rx", "delta_ry", "delta_rz", 
                      "gripper"],
        },
    }
    
    # Add wrist camera feature if available (using same target size)
    if has_wrist:
        features["observation.wrist_image"] = {
            "dtype": "video",
            "shape": (3, image_height, image_width),  # (C, H, W) - same as main camera
            "names": ["channel", "height", "width"],
        }
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="piper",
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Process each episode
    total_frames = 0
    start_time = time.time()
    num_episodes_total = len(episodes)
    
    # Determine print frequency
    print_freq = max(1, num_episodes_total // 20)
    
    print(f"\nProcessing {num_episodes_total} episodes...")
    
    for ep_idx, ep in enumerate(episodes):
        T = ep['num_timesteps']
        
        # Print progress
        if num_episodes_total <= 50 or (ep_idx + 1) % print_freq == 0 or ep_idx == 0:
            elapsed = time.time() - start_time
            rate = (ep_idx / elapsed) if elapsed > 0 and ep_idx > 0 else 0
            eta = (num_episodes_total - ep_idx) / rate if rate > 0 else 0
            print(f"  [{ep_idx + 1}/{num_episodes_total}] Processing episode {ep_idx}, "
                  f"{T} frames | {rate:.1f} ep/s | ETA: {eta:.0f}s")
        
        # Extract data
        images = ep['images']  # list of (H, W, 3) uint8
        ee_pose = ep['ee_pose']  # (T, 6)
        actions = ep['action']  # (T, 7) - RELATIVE delta actions
        gripper_states = ep['gripper_state']  # (T,)
        wrist_images = ep.get('wrist_images', None)
        
        for t in range(T):
            # Current observation - resize to target size
            img = np.array(images[t])  # (H, W, 3)
            if img.shape[0] != image_height or img.shape[1] != image_width:
                img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
            
            # Build state
            if include_gripper:
                gripper_state = np.array([gripper_states[t]], dtype=np.float32)
                state = np.concatenate([ee_pose[t], gripper_state])
            else:
                state = ee_pose[t]
            
            # Action: use relative delta directly from the dataset
            action = actions[t]  # (7,) [delta_xyz, delta_rpy, gripper]
            
            frame = {
                "observation.image": img,
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": "piper_ditflow",
            }
            
            # Add wrist camera image if available - resize to same target size
            if wrist_images is not None:
                wrist_img = np.array(wrist_images[t])
                if wrist_img.shape[0] != image_height or wrist_img.shape[1] != image_width:
                    wrist_img = cv2.resize(wrist_img, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
                frame["observation.wrist_image"] = wrist_img
            
            dataset.add_frame(frame)
        
        # Save episode
        dataset.save_episode()
        total_frames += T
    
    # Finalize dataset
    print("\nFinalizing dataset (encoding videos)...")
    dataset.finalize()
    
    # Post-conversion integrity verification
    # This catches: corrupt videos, missing frames, encoding errors, NFS I/O issues
    print("\nRunning post-conversion integrity verification...")
    verify_start = time.time()
    is_valid = verify_lerobot_dataset_integrity(
        output_dir, repo_id=repo_id, verify_all_episodes=True,
    )
    verify_time = time.time() - verify_start
    
    if not is_valid:
        print(f"\n{'!'*60}")
        print("ERROR: Post-conversion verification FAILED!")
        print(f"{'!'*60}")
        print("The converted dataset has corrupted video files.")
        print("Possible causes:")
        print("  1. NFS/network filesystem transient I/O errors during encoding")
        print("  2. Interrupted video encoding (OOM, signal, etc.)")
        print("  3. AV1 codec encoding issues")
        print("  4. Disk full or write permission issues")
        print(f"\nRemoving corrupted dataset at {output_dir}...")
        shutil.rmtree(output_dir)
        raise RuntimeError(
            "Post-conversion verification failed. The dataset has been removed. "
            "Please re-run the script to try again. If the problem persists, "
            "check disk space, NFS mount health, and try reducing convert_workers."
        )
    
    # Clear decoder cache after verification
    _clear_video_decoder_cache()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Action type: relative delta")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Dataset saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return image_height, image_width, has_wrist


def train_with_lerobot_api(
    args: argparse.Namespace,
    lerobot_dataset_dir: Path,
    image_height: int,
    image_width: int,
    has_wrist: bool = False,
    include_gripper: bool = False,
    train_episodes: list[int] | None = None,
    val_episodes: list[int] | None = None,
) -> dict:
    """Train DiT Flow using LeRobot's Python API.
    
    Args:
        args: Parsed command-line arguments.
        lerobot_dataset_dir: Path to LeRobot dataset.
        image_height: Image height.
        image_width: Image width.
        has_wrist: Whether the dataset has wrist camera images.
        include_gripper: Whether gripper state is included in observation.state.
        train_episodes: List of episode indices for training (None = all).
        val_episodes: List of episode indices for validation (None = no validation).
        
    Returns:
        Dictionary with training results.
    """
    # Register third-party plugins (this imports lerobot_policy_ditflow)
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    
    # Import DiTFlowConfig after registering plugins
    try:
        from lerobot_policy_ditflow import DiTFlowConfig
    except ImportError as e:
        raise ImportError(
            f"DiTFlow policy not found. Make sure lerobot_policy_ditflow is installed. "
            f"You can install it with: pip install -e /path/to/lerobot_policy_ditflow\n"
            f"Error: {e}"
        )
    
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.scripts.lerobot_train import train
    
    # Clear decoder cache before training to prevent stale file descriptors
    # from being inherited by DataLoader worker processes (fork-safety)
    _clear_video_decoder_cache()
    
    # Try to import custom training with viz
    try:
        from rev2fwd_il.train.lerobot_train_with_viz import train_with_xyz_visualization
        HAS_VIZ_TRAIN = True
    except ImportError:
        HAS_VIZ_TRAIN = False
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if running in distributed mode
    is_main_process = True
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        is_main_process = (local_rank == 0)
    
    # Handle overwrite: remove existing checkpoints directory
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
    
    # Determine state dimension
    state_dim = 6  # base: ee_pose (6)
    if include_gripper:
        state_dim += 1
    
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
            shape=(7,),  # Relative delta action
        ),
    }
    
    # Compute crop_shape from crop_ratio and image dimensions
    crop_shape = (
        int(image_height * args.crop_ratio[0]),
        int(image_width * args.crop_ratio[1]),
    )
    print(f"  Crop ratio: {tuple(args.crop_ratio)} -> crop shape: {crop_shape} "
          f"(image: {image_height}x{image_width})")
    
    # Dynamically calculate drop_n_last_frames based on user's parameters
    # Formula: horizon - n_action_steps - n_obs_steps + 1
    drop_n_last_frames = args.horizon - args.n_action_steps - args.n_obs_steps + 1
    if drop_n_last_frames < 0:
        print(f"Warning: drop_n_last_frames would be {drop_n_last_frames}, clamping to 0.")
        print(f"  Check: horizon({args.horizon}) - n_action_steps({args.n_action_steps}) "
              f"- n_obs_steps({args.n_obs_steps}) + 1 = {drop_n_last_frames}")
        drop_n_last_frames = 0
    
    # Create DiT Flow Policy configuration
    policy_cfg = DiTFlowConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        vision_backbone=args.vision_backbone,
        crop_shape=crop_shape,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        device=device,
        push_to_hub=False,
        # DiT-specific parameters
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_inference_steps=args.num_inference_steps,
        training_noise_sampling=args.training_noise_sampling,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
        # Dynamically calculated
        drop_n_last_frames=drop_n_last_frames,
        # Optimizer settings
        optimizer_lr=args.lr,
        # Data augmentation
        color_jitter_enabled=args.color_jitter,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
        gaussian_blur_enabled=args.gaussian_blur,
        gaussian_blur_kernel_size=args.gaussian_blur_kernel_size,
        gaussian_blur_sigma=tuple(args.gaussian_blur_sigma),
        random_sharpness_enabled=args.random_sharpness,
        random_sharpness_factor=args.random_sharpness_factor,
        random_sharpness_p=args.random_sharpness_p,
    )
    
    # Print normalization settings
    print("\n" + "=" * 60)
    print("[DEBUG] TRAINING Normalization Settings")
    print("=" * 60)
    print(f"  policy_cfg.normalization_mapping:")
    for feat_type, norm_mode in policy_cfg.normalization_mapping.items():
        print(f"    {feat_type}: {norm_mode}")
    print("=" * 60 + "\n")
    
    # Create dataset configuration
    dataset_cfg = DatasetConfig(
        repo_id="local/piper_ditflow",
        root=str(lerobot_dataset_dir),
        episodes=train_episodes,
    )
    
    # Create validation dataset configuration if specified
    val_dataset_cfg = None
    if val_episodes is not None and len(val_episodes) > 0:
        val_dataset_cfg = DatasetConfig(
            repo_id="local/piper_ditflow",
            root=str(lerobot_dataset_dir),
            episodes=val_episodes,
        )
    
    # Create WandB configuration
    wandb_cfg = WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
        run_id=None,
        mode=None,
    )
    
    # Create training pipeline configuration
    checkpoint_dir = out_dir / "checkpoints"
    
    # Handle resume
    resume_config_path = None
    if args.resume:
        checkpoints_subdir = checkpoint_dir / "checkpoints"
        if not checkpoints_subdir.exists():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint directory {checkpoints_subdir} does not exist."
            )
        checkpoint_dirs = [d for d in checkpoints_subdir.iterdir() if d.is_dir()]
        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"Cannot resume: no checkpoints found in {checkpoints_subdir}."
            )
        
        # Prefer 'last' checkpoint
        last_checkpoint = checkpoints_subdir / "last"
        if last_checkpoint.exists() and last_checkpoint.is_dir():
            latest_checkpoint = last_checkpoint
        else:
            # Find numeric directories and pick the largest
            numeric_dirs = [(int(d.name), d) for d in checkpoint_dirs if d.name.isdigit()]
            if numeric_dirs:
                numeric_dirs.sort(key=lambda x: x[0])
                latest_checkpoint = numeric_dirs[-1][1]
            else:
                latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: x.name)[-1]
        
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        
        resume_config_path = latest_checkpoint / "pretrained_model" / "train_config.json"
        if not resume_config_path.exists():
            raise FileNotFoundError(f"Cannot resume: train_config.json not found at {resume_config_path}")
        
        print(f"Loading config from: {resume_config_path}")
        
        train_cfg = TrainPipelineConfig.from_pretrained(str(resume_config_path.parent))
        train_cfg.resume = True
        train_cfg.steps = args.steps
        train_cfg.wandb = wandb_cfg
        train_cfg.checkpoint_path = latest_checkpoint
        train_cfg.policy.pretrained_path = latest_checkpoint / "pretrained_model"
        train_cfg.dataset.root = str(lerobot_dataset_dir)
        train_cfg.output_dir = checkpoint_dir
        
        print(f"  Will continue training to step {args.steps}")
    else:
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
            resume=False,
            eval_freq=0,
        )
    
    # Print training info
    print("\n" + "=" * 60)
    print("Starting DiT Flow Policy Training")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  LeRobot dataset: {lerobot_dataset_dir}")
    print(f"  Output: {checkpoint_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Image shape: ({image_height}, {image_width})")
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Include gripper: {include_gripper}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: 7 (relative delta)")
    print(f"  Crop ratio: {tuple(args.crop_ratio)} -> crop shape: {crop_shape}")
    print(f"  Horizon: {args.horizon}")
    print(f"  N obs steps: {args.n_obs_steps}")
    print(f"  N action steps: {args.n_action_steps}")
    print(f"  Vision backbone: {args.vision_backbone}")
    print(f"  Device: {device}")
    print(f"  WandB: {args.wandb}")
    print(f"  XYZ visualization: {args.enable_xyz_viz}")
    print("")
    print("  DiT-specific parameters:")
    print(f"    Hidden dim: {args.hidden_dim}")
    print(f"    Num blocks: {args.num_blocks}")
    print(f"    Num heads: {args.num_heads}")
    print(f"    Dropout: {args.dropout}")
    print(f"    Feedforward dim: {args.dim_feedforward}")
    print(f"    Inference steps: {args.num_inference_steps}")
    print(f"    Noise sampling: {args.training_noise_sampling}")
    print(f"    Clip sample: {args.clip_sample} (range: {args.clip_sample_range})")
    print(f"    Drop N last frames: {drop_n_last_frames}")
    print("")
    print("  Data augmentation (training-time only):")
    print(f"    ColorJitter: {args.color_jitter}"
          + (f" (b={args.color_jitter_brightness}, c={args.color_jitter_contrast}, "
             f"s={args.color_jitter_saturation}, h={args.color_jitter_hue})"
             if args.color_jitter else ""))
    print(f"    GaussianBlur: {args.gaussian_blur}"
          + (f" (kernel={args.gaussian_blur_kernel_size}, "
             f"sigma={args.gaussian_blur_sigma})"
             if args.gaussian_blur else ""))
    print(f"    RandomSharpness: {args.random_sharpness}"
          + (f" (factor={args.random_sharpness_factor}, p={args.random_sharpness_p})"
             if args.random_sharpness else ""))
    if train_episodes is not None:
        print(f"  Train episodes: {len(train_episodes)}")
    if val_episodes is not None:
        print(f"  Val episodes: {len(val_episodes)}")
    print("=" * 60 + "\n")
    
    # Run training
    original_argv = sys.argv.copy()
    if resume_config_path is not None:
        sys.argv = [sys.argv[0], f"--config_path={resume_config_path}"]
    
    try:
        if args.enable_xyz_viz and HAS_VIZ_TRAIN:
            xyz_viz_dir = out_dir / "xyz_viz"
            train_with_xyz_visualization(
                train_cfg,
                viz_save_freq=args.viz_save_freq,
                xyz_viz_dir=xyz_viz_dir,
                val_dataset_cfg=val_dataset_cfg,
                val_freq=args.val_freq,
            )
        else:
            if args.enable_xyz_viz and not HAS_VIZ_TRAIN:
                print("Warning: XYZ visualization not available, using standard training")
            train(train_cfg)
    finally:
        sys.argv = original_argv
    
    return {
        "output_dir": str(checkpoint_dir),
        "steps": args.steps,
    }


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    
    # Handle overfit mode
    if args.overfit:
        if args.num_episodes != 1:
            print(f"\n[Overfit Mode] Setting num_episodes=1 (was {args.num_episodes})")
            args.num_episodes = 1
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check if running in distributed mode
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (local_rank == 0)
    
    # Setup paths
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lerobot_dataset_dir = args.lerobot_dataset_dir
    if lerobot_dataset_dir is None:
        lerobot_dataset_dir = out_dir / "lerobot_dataset"
    lerobot_dataset_dir = Path(lerobot_dataset_dir)
    
    # Step 1: Convert data to LeRobot format (only on main process)
    # NOTE: We do NOT initialize distributed before conversion to avoid timeout issues.
    # Data conversion can take a long time (>10 min), which would cause NCCL barrier timeout.
    if not args.skip_convert:
        if is_main_process:
            # Remove stale conversion metadata from previous failed runs
            meta_file = out_dir / ".conversion_meta.json"
            if meta_file.exists():
                meta_file.unlink()
            
            image_height, image_width, has_wrist = convert_episodes_to_lerobot_format(
                data_dir=args.dataset,
                output_dir=lerobot_dataset_dir,
                fps=args.fps,
                repo_id="local/piper_ditflow",
                force=args.force_convert,
                num_episodes=args.num_episodes,
                include_gripper=args.include_gripper,
                target_image_size=tuple(args.image_size),
                num_workers=args.convert_workers,
            )
            # Write metadata to temp file for other processes
            meta_data = {
                "image_height": image_height,
                "image_width": image_width,
                "has_wrist": has_wrist,
                "include_gripper": args.include_gripper,
                "conversion_done": True,
            }
            with open(meta_file, "w") as f:
                json.dump(meta_data, f, indent=2)
        else:
            # Non-main processes wait for conversion to complete by polling the metadata file
            meta_file = out_dir / ".conversion_meta.json"
            print(f"[Rank {local_rank}] Waiting for rank 0 to finish data conversion...")
            wait_start = time.time()
            while True:
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        if meta.get("conversion_done", False):
                            break
                    except (json.JSONDecodeError, IOError):
                        pass  # File may be partially written
                time.sleep(5)  # Poll every 5 seconds
                elapsed = time.time() - wait_start
                if elapsed > 3600:  # 1 hour timeout
                    raise TimeoutError(f"Waited {elapsed:.0f}s for data conversion, giving up.")
            
            image_height = meta["image_height"]
            image_width = meta["image_width"]
            has_wrist = meta["has_wrist"]
            print(f"[Rank {local_rank}] Data conversion complete (waited {time.time() - wait_start:.1f}s)")
    else:
        # Skip conversion mode - read config from existing LeRobot dataset
        meta_path = lerobot_dataset_dir / "meta" / "info.json"
        if meta_path.exists():
            print(f"[Skip Convert] Reading config from existing LeRobot dataset: {lerobot_dataset_dir}")
            with open(meta_path, "r") as f:
                info = json.load(f)
            features = info.get("features", {})
            
            if "observation.image" in features:
                img_shape = features["observation.image"]["shape"]  # (C, H, W)
                image_height = img_shape[1]
                image_width = img_shape[2]
            else:
                raise ValueError("observation.image not found in dataset features")
            
            # Read state dimension from dataset - this is critical!
            if "observation.state" in features:
                actual_state_dim = features["observation.state"]["shape"][0]
                state_names = features["observation.state"].get("names", [])
                has_gripper_in_data = "gripper" in state_names
                args.include_gripper = has_gripper_in_data
                print(f"  State dim: {actual_state_dim} (gripper={'yes' if has_gripper_in_data else 'no'}, names={state_names})")
            else:
                raise ValueError("observation.state not found in dataset features")
            
            has_wrist = "observation.wrist_image" in features
            
            print(f"  Image size: {image_height}x{image_width}")
            print(f"  Has wrist camera: {has_wrist}")
            print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
            print(f"  Total frames: {info.get('total_frames', 'unknown')}")
        else:
            raise FileNotFoundError(
                f"LeRobot dataset not found at {lerobot_dataset_dir}. "
                f"Run without --skip_convert first."
            )

    if args.convert_only:
        if is_main_process:
            print("Data conversion complete. Exiting (--convert_only flag).")
        return
    
    # Auto-detect include_gripper from actual dataset metadata.
    # Check whether "gripper" is in the feature names rather than relying on
    # hardcoded dimension thresholds, because the base ee_pose dimension varies
    # between quaternion (7-dim) and Euler (6-dim) representations.
    ds_meta_path = lerobot_dataset_dir / "meta" / "info.json"
    if ds_meta_path.exists():
        with open(ds_meta_path, "r") as f:
            ds_info = json.load(f)
        ds_features = ds_info.get("features", {})
        if "observation.state" in ds_features:
            actual_state_dim = ds_features["observation.state"]["shape"][0]
            state_names = ds_features["observation.state"].get("names", [])
            has_gripper_in_data = "gripper" in state_names
            if has_gripper_in_data != args.include_gripper:
                old_flag = args.include_gripper
                args.include_gripper = has_gripper_in_data
                print(f"\n[AUTO-DETECT] Dataset state has gripper={has_gripper_in_data} "
                      f"(state_dim={actual_state_dim}, names={state_names}). "
                      f"Overriding --include_gripper from {old_flag} to {args.include_gripper}.")
    
    # Initialize distributed process group AFTER data conversion is complete
    # This avoids NCCL timeout issues during long-running conversions
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # Compute train/val episode split
    train_episodes = None
    val_episodes = None
    
    if args.val_split > 0.0:
        meta_path = lerobot_dataset_dir / "meta" / "info.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                info = json.load(f)
            total_episodes = info.get("total_episodes", 0)
        else:
            total_episodes = 0
        
        if total_episodes > 0:
            all_episode_indices = list(range(total_episodes))
            rng = np.random.default_rng(args.seed)
            rng.shuffle(all_episode_indices)
            
            n_val = max(1, int(total_episodes * args.val_split))
            val_episodes = sorted(all_episode_indices[:n_val])
            train_episodes = sorted(all_episode_indices[n_val:])
            
            if is_main_process:
                print(f"\n{'='*60}")
                print("Train/Val Split")
                print(f"{'='*60}")
                print(f"  Total episodes: {total_episodes}")
                print(f"  Val split: {args.val_split:.1%}")
                print(f"  Train episodes: {len(train_episodes)}")
                print(f"  Val episodes: {len(val_episodes)}")
                print(f"{'='*60}\n")
    
    # Step 2: Train the policy
    result = train_with_lerobot_api(
        args=args,
        lerobot_dataset_dir=lerobot_dataset_dir,
        image_height=image_height,
        image_width=image_width,
        has_wrist=has_wrist,
        include_gripper=args.include_gripper,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Total steps: {result['steps']}")
    print(f"  Action type: relative delta")
    print(f"  Policy type: DiT Flow (Diffusion Transformer + Flow Matching)")
    print("=" * 60)


if __name__ == "__main__":
    main()
