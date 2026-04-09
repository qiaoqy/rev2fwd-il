#!/usr/bin/env python3
"""Step 4: Train Diffusion Policy (DDIM by default).

Merges the logic of the old ``4_train_diffusion.py`` (data conversion +
training) and ``4_train_diffusion_ddim.py`` (DDIM monkey-patch wrapper).
DDIM is now the built-in default — no separate wrapper script needed.

Key differences from the old scripts
-------------------------------------
* ``noise_scheduler_type`` defaults to **DDIM** (was DDPM).
* ``num_inference_steps`` defaults to **10** (was 100).
* ``n_action_steps`` defaults to **16** (training uses full action chunk).
* ``include_obj_pose`` and ``include_gripper`` are **always on**
  (state_dim = 15 = ee_pose(7) + obj_pose(7) + gripper(1)).
* Wrist camera is **always enabled** (auto-detected from NPZ data).

Everything else — data conversion, multi-GPU, WandB logging, resume /
finetune, weighted sampling — is delegated to the original
``4_train_diffusion.py`` module.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Single-GPU
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/4_train.py \\
    --dataset data/exp_new/task_A_reversed_100.npz \\
    --out data/exp_new/weights/PP_A \\
    --steps 31072 --batch_size 128 --wandb

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
    scripts/scripts_pick_place_simulator/4_train.py \\
    --dataset data/exp_new/task_A_reversed_100.npz \\
    --out data/exp_new/weights/PP_A \\
    --steps 31072 --batch_size 128 --wandb

# Convert only
python scripts/scripts_pick_place_simulator/4_train.py \\
    --dataset data/exp_new/task_B_100.npz \\
    --out data/exp_new/weights/PP_B --convert_only
=============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Suppress verbose video-encoder output
os.environ.setdefault("AV_LOG_LEVEL", "quiet")
os.environ.setdefault("SVT_LOG", "0")
os.environ.setdefault("FFMPEG_LOG_LEVEL", "quiet")

# ---------------------------------------------------------------------------
# Import the original training script (filename starts with a digit)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SCRIPT = _THIS_DIR.parent / "scripts_pick_place" / "4_train_diffusion.py"

_spec = importlib.util.spec_from_file_location("_orig_train_diffusion", _ORIG_SCRIPT)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)

convert_npz_to_lerobot_format = _orig_mod.convert_npz_to_lerobot_format
load_episodes_from_npz = _orig_mod.load_episodes_from_npz
train_with_lerobot_api = _orig_mod.train_with_lerobot_api


# ---------------------------------------------------------------------------
# Argument parser — mirrors the original but with new defaults
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Diffusion Policy (DDIM) on pick-place NPZ data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    parser.add_argument("--dataset", type=str, required=True,
                        help="Input NPZ path.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory (checkpoints will be saved here).")
    parser.add_argument("--lerobot_dataset_dir", type=str, default=None,
                        help="LeRobot dataset path.  Default: <out>/lerobot_dataset.")

    # Conversion control
    parser.add_argument("--convert_only", action="store_true",
                        help="Only convert NPZ → LeRobot, don't train.")
    parser.add_argument("--skip_convert", action="store_true",
                        help="Skip conversion (assume already done).")
    parser.add_argument("--force_convert", action="store_true",
                        help="Force re-conversion.")

    # Training hyper-parameters
    parser.add_argument("--num_episodes", type=int, default=-1,
                        help="Limit number of episodes (-1 = all).")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)

    # Diffusion architecture
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16,
                        help="Diffusion prediction horizon / action chunk length.")
    parser.add_argument("--n_action_steps", type=int, default=16,
                        help="Action steps to execute (training default=16, full chunk).")
    parser.add_argument("--vision_backbone", type=str, default="resnet18")
    parser.add_argument("--crop_shape", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--pretrained_backbone_weights", type=str, default=None)

    # DDIM (built-in defaults)
    parser.add_argument("--noise_scheduler_type", type=str, default="DDIM",
                        choices=["DDPM", "DDIM"],
                        help="Noise scheduler type (default: DDIM).")
    parser.add_argument("--num_inference_steps", type=int, default=10,
                        help="DDIM denoising steps at inference (default: 10).")

    # Logging / checkpointing
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=20000)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint.")
    parser.add_argument("--finetune", action="store_true",
                        help="Load weights but reset step counter.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--sample_weights", type=str, default=None,
                        help="Path to sampling_weights.json for weighted sampling.")
    parser.add_argument("--viz_save_freq", type=int, default=20000)
    parser.add_argument("--enable_xyz_viz", action="store_true")

    # Validation
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--val_freq", type=int, default=500)

    # Device / WandB
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="rev2fwd-pick-place-simulator")

    # State composition
    parser.add_argument("--no_obj_pose", action="store_true",
                        help="Exclude obj_pose from state (state_dim=8 instead of 15).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDIM monkey-patch — wrap the original __init__ to inject defaults
# (Avoids creating a subclass, which breaks draccus choice registry)
# ---------------------------------------------------------------------------
from lerobot.policies.diffusion.configuration_diffusion import (  # noqa: E402
    DiffusionConfig as _RealDiffusionConfig,
)

_ddim_overrides: dict = {}
_orig_diffusion_init = _RealDiffusionConfig.__init__


def _patched_diffusion_init(self, **kwargs):
    kwargs.setdefault("noise_scheduler_type",
                      _ddim_overrides.get("noise_scheduler_type", "DDIM"))
    kwargs.setdefault("num_inference_steps",
                      _ddim_overrides.get("num_inference_steps", 10))
    _orig_diffusion_init(self, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    # State flags
    args.include_obj_pose = not args.no_obj_pose
    args.include_gripper = True

    # Attributes expected by the original module but not exposed here
    args.overfit = False

    # Store DDIM overrides
    _ddim_overrides["noise_scheduler_type"] = args.noise_scheduler_type
    _ddim_overrides["num_inference_steps"] = args.num_inference_steps

    state_dim = 7 + (7 if args.include_obj_pose else 0) + 1
    print(f"\n{'='*60}")
    print(f"[DDIM MODE] noise_scheduler_type = {args.noise_scheduler_type}")
    print(f"[DDIM MODE] num_inference_steps  = {args.num_inference_steps}")
    print(f"[State]     include_obj_pose = {args.include_obj_pose}, include_gripper = True  (state_dim={state_dim})")
    print(f"{'='*60}\n")

    # Monkey-patch DiffusionConfig.__init__ to inject DDIM defaults
    _RealDiffusionConfig.__init__ = _patched_diffusion_init

    try:
        # ---- seed ----
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_main = local_rank == 0

        if world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        lerobot_dir = args.lerobot_dataset_dir
        if lerobot_dir is None:
            lerobot_dir = out_dir / "lerobot_dataset"
        lerobot_dir = Path(lerobot_dir)

        # ---- Step 1: Convert ----
        if not args.skip_convert:
            if is_main:
                image_height, image_width, has_wrist, _ = convert_npz_to_lerobot_format(
                    npz_path=args.dataset,
                    output_dir=lerobot_dir,
                    fps=args.fps,
                    repo_id="local/rev2fwd_diffusion_B",
                    force=args.force_convert,
                    num_episodes=args.num_episodes,
                    include_obj_pose=args.include_obj_pose,
                    include_gripper=True,
                )
                meta_file = out_dir / ".conversion_meta.json"
                with open(meta_file, "w") as f:
                    json.dump({
                        "image_height": image_height,
                        "image_width": image_width,
                        "has_wrist": has_wrist,
                        "include_obj_pose": args.include_obj_pose,
                        "include_gripper": True,
                    }, f, indent=2)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if not is_main:
                with open(out_dir / ".conversion_meta.json") as f:
                    meta = json.load(f)
                image_height = meta["image_height"]
                image_width = meta["image_width"]
                has_wrist = meta["has_wrist"]
        else:
            meta_path = lerobot_dir / "meta" / "info.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    info = json.load(f)
                features = info.get("features", {})
                img_shape = features["observation.image"]["shape"]
                image_height, image_width = img_shape[1], img_shape[2]
                has_wrist = "observation.wrist_image" in features
            else:
                episodes = load_episodes_from_npz(args.dataset, num_episodes=args.num_episodes)
                image_height, image_width = episodes[0]["images"].shape[1], episodes[0]["images"].shape[2]
                has_wrist = "wrist_images" in episodes[0]

        if args.convert_only:
            if is_main:
                print("Data conversion complete. Exiting (--convert_only).")
            return

        # ---- Train/val split ----
        train_episodes = None
        val_episodes = None
        if args.val_split > 0.0:
            meta_path = lerobot_dir / "meta" / "info.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    total_episodes = json.load(f).get("total_episodes", 0)
            else:
                total_episodes = 0
            if total_episodes > 0:
                all_idx = list(range(total_episodes))
                rng = np.random.default_rng(args.seed)
                rng.shuffle(all_idx)
                n_val = max(1, int(total_episodes * args.val_split))
                val_episodes = sorted(all_idx[:n_val])
                train_episodes = sorted(all_idx[n_val:])

        # ---- Step 2: Train (DiffusionConfig is already DDIM-patched) ----
        result = train_with_lerobot_api(
            args=args,
            lerobot_dataset_dir=lerobot_dir,
            image_height=image_height,
            image_width=image_width,
            has_wrist=has_wrist,
            include_obj_pose=args.include_obj_pose,
            include_gripper=True,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
        )

        print(f"\n{'='*60}")
        print("Training Complete! (DDIM)")
        print(f"{'='*60}")
        print(f"  Output:           {result['output_dir']}")
        print(f"  Steps:            {result['steps']}")
        print(f"  Scheduler:        {args.noise_scheduler_type}")
        print(f"  Inference steps:  {args.num_inference_steps}")
        print(f"  n_action_steps:   {args.n_action_steps}")
        print(f"{'='*60}")

    finally:
        _RealDiffusionConfig.__init__ = _orig_diffusion_init


if __name__ == "__main__":
    main()
