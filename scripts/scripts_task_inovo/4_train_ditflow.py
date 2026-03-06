#!/usr/bin/env python3
"""Script 4: Train DiT Flow Policy on Inovo (RoboKit) data using LeRobot.

This script trains a vision-based DiT Flow Policy (Diffusion Transformer +
Flow Matching) on the Inovo data that has been converted to LeRobot format
by Script 3.

The architecture and training logic follows the Piper pipeline
(scripts_piper_local/7_train_ditflow.py). The key difference is that:
  - Data comes from pre-converted LeRobot dataset (no inline conversion)
  - Configured for Inovo data specifics (camera resolution, action format)

=============================================================================
DEPENDENCIES
=============================================================================
  - lerobot >= 0.4.3
  - lerobot_policy_ditflow >= 0.1.0

=============================================================================
USAGE
=============================================================================
# Train Task A policy (reversed data) - single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_A/lerobot_dataset \
    --out runs/inovo_A \
    --batch_size 64 --steps 50000 --include_gripper --wandb

# Train Task B policy (original data)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_B/lerobot_dataset \
    --out runs/inovo_B \
    --batch_size 64 --steps 50000 --include_gripper --wandb

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_A/lerobot_dataset \
    --out runs/inovo_A \
    --batch_size 32 --steps 100000 --include_gripper --wandb

# Overfit mode (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_A/lerobot_dataset \
    --out runs/inovo_A_overfit \
    --overfit --steps 1000 --include_gripper

# Resume training
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_A/lerobot_dataset \
    --out runs/inovo_A \
    --resume --steps 200000 --include_gripper

# With beta noise scheduling (Pi0 paper style)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_task_inovo/4_train_ditflow.py \
    --lerobot_dataset_dir runs/inovo_A/lerobot_dataset \
    --out runs/inovo_A_beta \
    --training_noise_sampling beta \
    --batch_size 64 --steps 50000 --include_gripper --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Suppress verbose output from video encoders
os.environ["AV_LOG_LEVEL"] = "quiet"
os.environ["SVT_LOG"] = "0"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DiT Flow Policy on Inovo data (pre-converted LeRobot format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Input/Output
    # =========================================================================
    parser.add_argument(
        "--lerobot_dataset_dir",
        type=str,
        required=True,
        help="Path to pre-converted LeRobot dataset directory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/inovo_ditflow",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="local/inovo_ditflow",
        help="LeRobot repository ID. Default: local/inovo_ditflow.",
    )

    # =========================================================================
    # Mode Selection
    # =========================================================================
    parser.add_argument(
        "--include_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include gripper state in observation.state. Default: True.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit mode: train on 1 episode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing checkpoints. Default: True.",
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    parser.add_argument("--steps", type=int, default=100000, help="Training steps. Default: 100000.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default: 32.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Default: 1e-4.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0.")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers. Default: 8.")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS. Default: 30.")

    # =========================================================================
    # DiT Flow Architecture
    # =========================================================================
    parser.add_argument("--n_obs_steps", type=int, default=2, help="Observation steps. Default: 2.")
    parser.add_argument("--horizon", type=int, default=16, help="Action sequence length. Default: 16.")
    parser.add_argument("--n_action_steps", type=int, default=8, help="Action steps to execute. Default: 8.")
    parser.add_argument("--vision_backbone", type=str, default="resnet18", help="Vision backbone. Default: resnet18.")
    parser.add_argument(
        "--crop_ratio", type=float, nargs=2, default=[0.95, 0.95],
        help="Crop ratio (H_ratio, W_ratio). Default: 0.95 0.95.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=100, help="ODE steps. Default: 100.")
    parser.add_argument("--pretrained_backbone_weights", type=str, default=None, help="Pretrained weights.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="DiT hidden dim. Default: 512.")
    parser.add_argument("--num_blocks", type=int, default=6, help="DiT blocks. Default: 6.")
    parser.add_argument("--num_heads", type=int, default=16, help="Attention heads. Default: 16.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout. Default: 0.1.")
    parser.add_argument("--dim_feedforward", type=int, default=4096, help="FFN dim. Default: 4096.")
    parser.add_argument(
        "--training_noise_sampling", type=str, default="uniform", choices=["uniform", "beta"],
        help="Noise schedule: 'uniform' or 'beta'. Default: uniform.",
    )

    # =========================================================================
    # Logging & Checkpointing
    # =========================================================================
    parser.add_argument("--log_freq", type=int, default=50, help="Log every N steps. Default: 50.")
    parser.add_argument("--save_freq", type=int, default=20000, help="Save every N steps. Default: 20000.")
    parser.add_argument("--enable_xyz_viz", action="store_true", help="Enable XYZ visualization.")
    parser.add_argument("--viz_save_freq", type=int, default=20000, help="Viz save freq. Default: 20000.")

    # =========================================================================
    # Sampling
    # =========================================================================
    parser.add_argument(
        "--clip_sample", action=argparse.BooleanOptionalAction, default=True,
        help="Clip action samples. Default: True.",
    )
    parser.add_argument("--clip_sample_range", type=float, default=1.0, help="Clip range. Default: 1.0.")

    # =========================================================================
    # Data Augmentation
    # =========================================================================
    parser.add_argument(
        "--color_jitter", action=argparse.BooleanOptionalAction, default=True,
        help="ColorJitter augmentation. Default: True.",
    )
    parser.add_argument("--color_jitter_brightness", type=float, default=0.05)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.05)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.05)
    parser.add_argument("--color_jitter_hue", type=float, default=0.05)
    parser.add_argument("--gaussian_blur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gaussian_blur_kernel_size", type=int, default=3)
    parser.add_argument("--gaussian_blur_sigma", type=float, nargs=2, default=[0.1, 0.5])
    parser.add_argument("--random_sharpness", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--random_sharpness_factor", type=float, default=1.2)
    parser.add_argument("--random_sharpness_p", type=float, default=0.3)

    # =========================================================================
    # Device & WandB
    # =========================================================================
    parser.add_argument("--device", type=str, default=None, help="Torch device.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.")
    parser.add_argument("--wandb_project", type=str, default="inovo-ditflow", help="WandB project.")

    # =========================================================================
    # Validation
    # =========================================================================
    parser.add_argument("--val_split", type=float, default=0.0, help="Validation split. Default: 0.")
    parser.add_argument("--val_freq", type=int, default=500, help="Validation freq. Default: 500.")

    return parser.parse_args()


def _clear_video_decoder_cache():
    """Clear LeRobot's module-level video decoder cache (fork-safety)."""
    try:
        from lerobot.datasets.video_utils import _default_decoder_cache
        cache_size = _default_decoder_cache.size()
        if cache_size > 0:
            _default_decoder_cache.clear()
            print(f"  [Cache] Cleared {cache_size} cached video decoders")
    except (ImportError, AttributeError):
        pass


def train_with_lerobot_api(
    args: argparse.Namespace,
    lerobot_dataset_dir: Path,
    image_height: int,
    image_width: int,
    has_wrist: bool,
    include_gripper: bool,
    state_dim: int,
    action_dim: int,
    train_episodes: list[int] | None = None,
    val_episodes: list[int] | None = None,
) -> dict:
    """Configure and run DiT Flow training via LeRobot API."""

    # Register third-party plugins
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()

    try:
        from lerobot_policy_ditflow import DiTFlowConfig
    except ImportError as e:
        raise ImportError(
            f"lerobot_policy_ditflow not installed. Install with:\n"
            f"  pip install git+https://github.com/danielsanjosepro/lerobot_policy_ditflow.git\n"
            f"Error: {e}"
        )

    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.scripts.lerobot_train import train

    _clear_video_decoder_cache()

    # Try to import custom viz training
    try:
        from rev2fwd_il.train.lerobot_train_with_viz import train_with_xyz_visualization
        HAS_VIZ_TRAIN = True
    except ImportError:
        HAS_VIZ_TRAIN = False

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distributed check
    is_main_process = True
    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0

    # Handle overwrite
    checkpoint_dir = out_dir / "checkpoints"
    if is_main_process and args.overwrite and checkpoint_dir.exists() and not args.resume:
        print(f"Removing existing checkpoints: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Configure features
    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, image_height, image_width),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,),
        ),
    }
    if has_wrist:
        input_features["observation.wrist_image"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, image_height, image_width),
        )

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION, shape=(action_dim,),
        ),
    }

    # Crop shape from ratio
    crop_shape = (
        int(image_height * args.crop_ratio[0]),
        int(image_width * args.crop_ratio[1]),
    )
    print(f"  Crop ratio: {tuple(args.crop_ratio)} -> crop shape: {crop_shape}")

    # Drop N last frames
    drop_n_last_frames = args.horizon - args.n_action_steps - args.n_obs_steps + 1
    if drop_n_last_frames < 0:
        print(f"Warning: drop_n_last_frames={drop_n_last_frames}, clamping to 0.")
        drop_n_last_frames = 0

    # Create DiT Flow config
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
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_inference_steps=args.num_inference_steps,
        training_noise_sampling=args.training_noise_sampling,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
        drop_n_last_frames=drop_n_last_frames,
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

    # Dataset config
    dataset_cfg = DatasetConfig(
        repo_id=args.repo_id,
        root=str(lerobot_dataset_dir),
        episodes=train_episodes,
    )

    # Validation config
    val_dataset_cfg = None
    if val_episodes and len(val_episodes) > 0:
        val_dataset_cfg = DatasetConfig(
            repo_id=args.repo_id,
            root=str(lerobot_dataset_dir),
            episodes=val_episodes,
        )

    # WandB config
    wandb_cfg = WandBConfig(
        enable=args.wandb,
        project=args.wandb_project,
        run_id=None,
        mode=None,
    )

    checkpoint_dir = out_dir / "checkpoints"

    # Handle resume
    resume_config_path = None
    if args.resume:
        checkpoints_subdir = checkpoint_dir / "checkpoints"
        if not checkpoints_subdir.exists():
            raise FileNotFoundError(f"Cannot resume: {checkpoints_subdir} does not exist.")

        last_ckpt = checkpoints_subdir / "last"
        if last_ckpt.exists():
            latest_checkpoint = last_ckpt
        else:
            numeric = [(int(d.name), d) for d in checkpoints_subdir.iterdir()
                       if d.is_dir() and d.name.isdigit()]
            if not numeric:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_subdir}")
            numeric.sort()
            latest_checkpoint = numeric[-1][1]

        print(f"Resuming from: {latest_checkpoint}")
        resume_config_path = latest_checkpoint / "pretrained_model" / "train_config.json"
        if not resume_config_path.exists():
            raise FileNotFoundError(f"train_config.json not found at {resume_config_path}")

        train_cfg = TrainPipelineConfig.from_pretrained(str(resume_config_path.parent))
        train_cfg.resume = True
        train_cfg.steps = args.steps
        train_cfg.wandb = wandb_cfg
        train_cfg.checkpoint_path = latest_checkpoint
        train_cfg.policy.pretrained_path = latest_checkpoint / "pretrained_model"
        train_cfg.dataset.root = str(lerobot_dataset_dir)
        train_cfg.output_dir = checkpoint_dir
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
    print(f"\n{'='*60}")
    print("Starting DiT Flow Policy Training (Inovo)")
    print(f"{'='*60}")
    print(f"  LeRobot dataset: {lerobot_dataset_dir}")
    print(f"  Output: {checkpoint_dir}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Image: ({image_height}, {image_width})")
    print(f"  Wrist camera: {has_wrist}")
    print(f"  Include gripper: {include_gripper}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Crop shape: {crop_shape}")
    print(f"  Horizon: {args.horizon}")
    print(f"  N obs steps: {args.n_obs_steps}")
    print(f"  N action steps: {args.n_action_steps}")
    print(f"  Vision backbone: {args.vision_backbone}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num blocks: {args.num_blocks}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Noise sampling: {args.training_noise_sampling}")
    print(f"  Drop N last: {drop_n_last_frames}")
    print(f"  WandB: {args.wandb}")
    print(f"  XYZ viz: {args.enable_xyz_viz}")
    print(f"{'='*60}\n")

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
                print("Warning: XYZ visualization not available, using standard training.")
            train(train_cfg)
    finally:
        sys.argv = original_argv

    return {"output_dir": str(checkpoint_dir), "steps": args.steps}


def main():
    args = _parse_args()

    # Overfit mode
    if args.overfit:
        print("[Overfit Mode] Will train on 1 episode.")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = (local_rank == 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lerobot_dataset_dir = Path(args.lerobot_dataset_dir)

    # Read dataset metadata
    meta_path = lerobot_dataset_dir / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {lerobot_dataset_dir}. "
            f"Run 3_convert_to_lerobot.py first."
        )

    with open(meta_path) as f:
        info = json.load(f)

    features = info.get("features", {})

    # Image dimensions
    img_shape = features["observation.image"]["shape"]  # (C, H, W)
    image_height, image_width = img_shape[1], img_shape[2]
    has_wrist = "observation.wrist_image" in features

    # State dimension and gripper detection
    state_info = features.get("observation.state", {})
    state_dim = state_info.get("shape", [7])[0]
    state_names = state_info.get("names", [])
    has_gripper_in_data = "gripper_width" in state_names or "gripper" in state_names
    include_gripper = has_gripper_in_data

    if has_gripper_in_data != args.include_gripper:
        print(f"[AUTO-DETECT] Dataset has gripper={has_gripper_in_data} "
              f"(state_dim={state_dim}, names={state_names}). "
              f"Overriding --include_gripper to {has_gripper_in_data}.")
        args.include_gripper = has_gripper_in_data

    # Action dimension
    action_info = features.get("action", {})
    action_dim = action_info.get("shape", [7])[0]

    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)

    if is_main_process:
        print(f"\n{'='*60}")
        print("Dataset Info")
        print(f"{'='*60}")
        print(f"  Path: {lerobot_dataset_dir}")
        print(f"  Episodes: {total_episodes}")
        print(f"  Frames: {total_frames}")
        print(f"  Image: ({image_height}, {image_width})")
        print(f"  Wrist camera: {has_wrist}")
        print(f"  State dim: {state_dim} (names={state_names})")
        print(f"  Action dim: {action_dim}")
        print(f"{'='*60}")

    # Init distributed after metadata reading
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # Train/val split
    train_episodes = None
    val_episodes = None

    if args.val_split > 0.0 and total_episodes > 0:
        all_indices = list(range(total_episodes))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_indices)

        n_val = max(1, int(total_episodes * args.val_split))
        val_episodes = sorted(all_indices[:n_val])
        train_episodes = sorted(all_indices[n_val:])

        if is_main_process:
            print(f"  Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    if args.overfit:
        train_episodes = [0]
        val_episodes = None

    # Train
    result = train_with_lerobot_api(
        args=args,
        lerobot_dataset_dir=lerobot_dataset_dir,
        image_height=image_height,
        image_width=image_width,
        has_wrist=has_wrist,
        include_gripper=include_gripper,
        state_dim=state_dim,
        action_dim=action_dim,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
    )

    if is_main_process:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"  Output: {result['output_dir']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Policy: DiT Flow (Inovo)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
