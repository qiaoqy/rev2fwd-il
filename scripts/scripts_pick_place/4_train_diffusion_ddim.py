#!/usr/bin/env python3
"""Step 4 (DDIM variant): Train Diffusion Policy with DDIM sampling.

This script is identical to 4_train_diffusion.py, except the noise scheduler
is changed from DDPM to DDIM.  Two extra CLI flags are added:

    --noise_scheduler_type   DDIM (default, was DDPM)
    --num_inference_steps    Number of DDIM denoising steps at inference
                             (default: 10; original DDPM uses 100)

Everything else — data conversion, observation features, multi-GPU support,
WandB logging, etc. — is delegated to the original script 4 without
modification.

=============================================================================
USAGE EXAMPLES
=============================================================================
# DDIM with 10 denoising steps (default)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/4_train_diffusion_ddim.py \\
    --dataset data/A_2images_goal.npz \\
    --out runs/diffusion_A_goal_ddim \\
    --batch_size 64 --steps 50000 --wandb

# DDIM with custom inference steps
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/4_train_diffusion_ddim.py \\
    --dataset data/B_circle.npz \\
    --out runs/diffusion_B_circle_ddim \\
    --num_inference_steps 20 \\
    --batch_size 64 --steps 50000 \\
    --include_obj_pose --wandb

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
    scripts/scripts_pick_place/4_train_diffusion_ddim.py \\
    --dataset data/A_circle.npz \\
    --out runs/PP_A_circle_ddim \\
    --num_inference_steps 10 \\
    --batch_size 32 --steps 50000 \\
    --include_obj_pose --include_gripper --wandb
=============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the original script 4 (filename starts with a digit, so we use
# importlib to load it as a module).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SCRIPT = _THIS_DIR / "4_train_diffusion.py"

_spec = importlib.util.spec_from_file_location("_orig_train_diffusion", _ORIG_SCRIPT)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)

_parse_args_orig = _orig_mod._parse_args
convert_npz_to_lerobot_format = _orig_mod.convert_npz_to_lerobot_format
load_episodes_from_npz = _orig_mod.load_episodes_from_npz
train_with_lerobot_api = _orig_mod.train_with_lerobot_api


def _parse_args() -> argparse.Namespace:
    """Wrap the original arg parser and add DDIM-specific flags."""
    # Inject DDIM-specific defaults via a pre-parse step so that the original
    # parser doesn't choke on unknown args.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--noise_scheduler_type",
        type=str,
        default="DDIM",
        choices=["DDPM", "DDIM"],
        help="Noise scheduler type. Default: DDIM.",
    )
    pre_parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of DDIM denoising steps at inference time. Default: 10.",
    )
    ddim_args, remaining = pre_parser.parse_known_args()

    # Let the original parser handle the rest
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        args = _parse_args_orig()
    finally:
        sys.argv = old_argv

    # Attach DDIM-specific fields
    args.noise_scheduler_type = ddim_args.noise_scheduler_type
    args.num_inference_steps = ddim_args.num_inference_steps
    return args


# ---------------------------------------------------------------------------
# Monkey-patch the DiffusionConfig construction inside train_with_lerobot_api
# so that it uses the DDIM scheduler (and num_inference_steps) while keeping
# every other behavior the same.
# ---------------------------------------------------------------------------
# Save a reference to the *real* DiffusionConfig so the patch can call it.
from lerobot.policies.diffusion.configuration_diffusion import (  # noqa: E402
    DiffusionConfig as _RealDiffusionConfig,
)

# We will store DDIM overrides in a thread-local-like global so the patched
# config factory can pick them up.
_ddim_overrides: dict = {}


class _DiffusionConfigWithDDIM(_RealDiffusionConfig):
    """Thin wrapper that injects noise_scheduler_type / num_inference_steps."""

    def __init__(self, **kwargs):
        kwargs.setdefault("noise_scheduler_type", _ddim_overrides.get("noise_scheduler_type", "DDIM"))
        kwargs.setdefault("num_inference_steps", _ddim_overrides.get("num_inference_steps", 10))
        super().__init__(**kwargs)


def main() -> None:
    """Entry point – mirrors the original main() but with DDIM overrides."""
    import json
    import os
    import shutil
    import time
    from pathlib import Path

    import numpy as np
    import torch

    args = _parse_args()

    # Store DDIM overrides so the patched config picks them up
    _ddim_overrides["noise_scheduler_type"] = args.noise_scheduler_type
    _ddim_overrides["num_inference_steps"] = args.num_inference_steps

    print(f"\n{'='*60}")
    print(f"[DDIM MODE] noise_scheduler_type = {args.noise_scheduler_type}")
    print(f"[DDIM MODE] num_inference_steps  = {args.num_inference_steps}")
    print(f"{'='*60}\n")

    # Monkey-patch: when train_with_lerobot_api does
    #   from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    # it has already been imported.  We patch the *module-level* reference that
    # the original function will use at call time.
    import lerobot.policies.diffusion.configuration_diffusion as _cfg_mod
    _original_cls = _cfg_mod.DiffusionConfig
    _cfg_mod.DiffusionConfig = _DiffusionConfigWithDDIM

    try:
        # ---- replicate the original main() logic with our args ----
        if args.overfit:
            if args.num_episodes != 1:
                print(f"\n[Overfit Mode] Setting num_episodes=1 (was {args.num_episodes})")
                args.num_episodes = 1
            out_dir_temp = Path(args.out)
            overfit_init_path = out_dir_temp / "overfit_env_init.json"
            lerobot_dataset_dir_temp = args.lerobot_dataset_dir
            if lerobot_dataset_dir_temp is None:
                lerobot_dataset_dir_temp = out_dir_temp / "lerobot_dataset"
            lerobot_dataset_dir_temp = Path(lerobot_dataset_dir_temp)

            if overfit_init_path.exists() and lerobot_dataset_dir_temp.exists():
                print("[Overfit Mode] Found existing overfit_env_init.json and dataset, skipping re-conversion")
            elif not args.force_convert:
                print("[Overfit Mode] Enabling force_convert to extract env init params")
                args.force_convert = True

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_main_process = (local_rank == 0)

        if world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        lerobot_dataset_dir = args.lerobot_dataset_dir
        if lerobot_dataset_dir is None:
            lerobot_dataset_dir = out_dir / "lerobot_dataset"
        lerobot_dataset_dir = Path(lerobot_dataset_dir)

        overfit_env_init = None

        # Step 1: Convert data
        if not args.skip_convert:
            if is_main_process:
                image_height, image_width, has_wrist, overfit_env_init = convert_npz_to_lerobot_format(
                    npz_path=args.dataset,
                    output_dir=lerobot_dataset_dir,
                    fps=args.fps,
                    repo_id="local/rev2fwd_diffusion_B",
                    force=args.force_convert,
                    num_episodes=args.num_episodes,
                    include_obj_pose=args.include_obj_pose,
                    include_gripper=args.include_gripper,
                    overfit=args.overfit,
                )
                meta_file = out_dir / ".conversion_meta.json"
                meta_data = {
                    "image_height": image_height,
                    "image_width": image_width,
                    "has_wrist": has_wrist,
                    "include_obj_pose": args.include_obj_pose,
                    "include_gripper": args.include_gripper,
                    "overfit_env_init": overfit_env_init,
                }
                with open(meta_file, "w") as f:
                    json.dump(meta_data, f, indent=2)

                if args.overfit and overfit_env_init is not None:
                    overfit_init_path = out_dir / "overfit_env_init.json"
                    with open(overfit_init_path, "w") as f:
                        json.dump(overfit_env_init, f, indent=2)
                    print(f"\n[Overfit Mode] Saved env init params to: {overfit_init_path}")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if not is_main_process:
                meta_file = out_dir / ".conversion_meta.json"
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                image_height = meta["image_height"]
                image_width = meta["image_width"]
                has_wrist = meta["has_wrist"]
                overfit_env_init = meta.get("overfit_env_init", None)
        else:
            meta_path = lerobot_dataset_dir / "meta" / "info.json"
            if meta_path.exists():
                print(f"[Skip Convert] Reading config from existing LeRobot dataset: {lerobot_dataset_dir}")
                with open(meta_path, "r") as f:
                    info = json.load(f)
                features = info.get("features", {})
                if "observation.image" in features:
                    img_shape = features["observation.image"]["shape"]
                    image_height = img_shape[1]
                    image_width = img_shape[2]
                else:
                    raise ValueError("observation.image not found in dataset features")
                has_wrist = "observation.wrist_image" in features
                print(f"  Image size: {image_height}x{image_width}")
                print(f"  Has wrist camera: {has_wrist}")
                print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
                print(f"  Total frames: {info.get('total_frames', 'unknown')}")
            else:
                print(f"[Skip Convert] LeRobot dataset not found, loading NPZ: {args.dataset}")
                episodes = load_episodes_from_npz(args.dataset, num_episodes=args.num_episodes)
                image_shape = episodes[0]["images"].shape[1:]
                image_height, image_width = image_shape[0], image_shape[1]
                has_wrist = "wrist_images" in episodes[0]

            if args.overfit:
                if "episodes" not in dir():
                    episodes = load_episodes_from_npz(args.dataset, num_episodes=args.num_episodes)
                if len(episodes) > 0:
                    ep0 = episodes[0]
                    overfit_env_init = {
                        "initial_obj_pose": ep0["obj_pose"][0].tolist(),
                        "initial_ee_pose": ep0["ee_pose"][0].tolist(),
                        "place_pose": ep0.get("place_pose", [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                        "goal_pose": ep0.get("goal_pose", [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    }
                    if isinstance(overfit_env_init["place_pose"], np.ndarray):
                        overfit_env_init["place_pose"] = overfit_env_init["place_pose"].tolist()
                    if isinstance(overfit_env_init["goal_pose"], np.ndarray):
                        overfit_env_init["goal_pose"] = overfit_env_init["goal_pose"].tolist()
                    overfit_init_path = out_dir / "overfit_env_init.json"
                    with open(overfit_init_path, "w") as f:
                        json.dump(overfit_env_init, f, indent=2)
                    print(f"\n[Overfit Mode] Saved env init params to: {overfit_init_path}")

        if args.convert_only:
            if is_main_process:
                print("Data conversion complete. Exiting (--convert_only flag).")
            return

        # State slicing logic
        meta_path = lerobot_dataset_dir / "meta" / "info.json"
        dataset_state_dim = None
        if meta_path.exists():
            with open(meta_path, "r") as f:
                info = json.load(f)
            features = info.get("features", {})
            if "observation.state" in features:
                dataset_state_dim = features["observation.state"]["shape"][0]

        policy_state_dim = 7
        if args.include_obj_pose:
            policy_state_dim += 7
        if args.include_gripper:
            policy_state_dim += 1

        state_slice_end = None
        if dataset_state_dim is not None and policy_state_dim < dataset_state_dim:
            state_slice_end = policy_state_dim
            if is_main_process:
                print(f"\n{'='*60}")
                print("[State Slicing] Training with subset of state features")
                print(f"{'='*60}")
                print(f"  Dataset state_dim: {dataset_state_dim}")
                print(f"  Policy state_dim:  {policy_state_dim}")
                print(f"  Will slice state to [:, :, :{state_slice_end}]")
                print(f"{'='*60}\n")
        elif dataset_state_dim is not None and policy_state_dim > dataset_state_dim:
            raise ValueError(
                f"Policy state_dim ({policy_state_dim}) > dataset state_dim ({dataset_state_dim}). "
                f"Please re-convert the dataset with the correct flags or adjust training args."
            )

        # Train/val split
        train_episodes = None
        val_episodes = None

        if args.val_split > 0.0:
            meta_path = lerobot_dataset_dir / "meta" / "info.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    info = json.load(f)
                total_episodes = info.get("total_episodes", 0)
            else:
                episodes_dir = lerobot_dataset_dir / "data"
                if episodes_dir.exists():
                    total_episodes = len(list(episodes_dir.glob("episode_*")))
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
            else:
                if is_main_process:
                    print("Warning: Could not determine total episodes, skipping validation split")

        # Step 2: Train (DiffusionConfig will be our DDIM-patched version)
        result = train_with_lerobot_api(
            args=args,
            lerobot_dataset_dir=lerobot_dataset_dir,
            image_height=image_height,
            image_width=image_width,
            has_wrist=has_wrist,
            include_obj_pose=args.include_obj_pose,
            include_gripper=args.include_gripper,
            overfit_env_init=overfit_env_init,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            state_slice_end=state_slice_end,
        )

        print("\n" + "=" * 60)
        print("Training Complete! (DDIM)")
        print("=" * 60)
        print(f"  Output directory: {result['output_dir']}")
        print(f"  Total steps: {result['steps']}")
        print(f"  Noise scheduler: {args.noise_scheduler_type}")
        print(f"  Inference steps: {args.num_inference_steps}")
        print(f"  Action type: FSM goal position (sharp)")
        print("=" * 60)
    finally:
        # Restore original DiffusionConfig
        _cfg_mod.DiffusionConfig = _original_cls


if __name__ == "__main__":
    main()
