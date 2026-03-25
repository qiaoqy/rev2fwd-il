#!/usr/bin/env python3
"""Step 4: Fine-tune Diffusion Policy with RECAP advantage conditioning.

Implements classifier-free guidance (CFG) style advantage conditioning:

  - Training:  action chunk is predicted conditioned on advantage indicator I_t ∈ {0, 1}.
               With probability null_prob (=0.2), I_t is replaced by a null token (0.5),
               training both the conditional and unconditional branches simultaneously.

  - Inference: always use I_t = 1.0 (positive).  Optionally amplify via CFG guidance.

Implementation strategy:
  The advantage indicator is appended to the robot state as a 16th dimension:
    state_dim: 15 → 16  (extra dim = indicator: 0.0 negative / 1.0 positive / 0.5 null)

  The new policy is initialised from the pretrained checkpoint.  Layers whose
  input dimension grew due to the extra state dim are identified by shape comparison
  and zero-padded (zero init = the new indicator starts as a no-op, preserving
  pretrained behaviour at the start of fine-tuning).

Usage:
    # Single-GPU fine-tune
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_recap_rl/4_retrain_with_recap.py \\
        --npz_path data/recap_exp/advantages_A.npz \\
        --policy data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out data/recap_exp/recap_A \\
        --steps 15000 --batch_size 64 --null_prob 0.2 --wandb

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        scripts/scripts_recap_rl/4_retrain_with_recap.py \\
        --npz_path data/recap_exp/advantages_A.npz \\
        --policy data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out data/recap_exp/recap_A \\
        --steps 15000 --batch_size 128 --null_prob 0.2 --wandb
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch


# ---- NULL indicator value for CFG training ----
ADV_NULL     = 0.5    # "unconditional" token
ADV_POSITIVE = 1.0    # positive advantage (used at inference)
ADV_NEGATIVE = 0.0    # negative advantage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RECAP fine-tuning with advantage conditioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data / policy
    parser.add_argument("--npz_path", type=str, required=True,
                        help="Augmented NPZ from step 3 (with 'indicators' field).")
    parser.add_argument("--policy", type=str, required=True,
                        help="Pretrained DiffusionPolicy checkpoint path.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory.")

    # Training
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--null_prob", type=float, default=0.2,
                        help="Probability of dropping advantage indicator (CFG training).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight on conditional loss (unconditional always = 1).")

    # Architecture
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--noise_scheduler_type", type=str, default="DDIM")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--num_train_timesteps", type=int, default=100)

    # I/O
    parser.add_argument("--lerobot_dataset_dir", type=str, default=None)
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--force_convert", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="rev2fwd-recap-pick-place")
    return parser.parse_args()


# ============================================================
# NPZ → LeRobot conversion with indicator in state (state_dim = 16)
# ============================================================

def convert_npz_to_lerobot_recap(
    npz_path: str,
    output_dir: Path,
    fps: int,
    repo_id: str,
    force: bool,
    null_prob: float,
    alpha: float,
    image_height: int,
    image_width: int,
    seed: int,
) -> tuple:
    """Convert augmented NPZ to LeRobot dataset with advantage indicator in state.

    The advantage indicator is appended to observation.state as dimension 16.
    Two versions of each frame are stored:
      - Unconditional: indicator = ADV_NULL (0.5)   weight = 1.0 (always trained)
      - Conditional:   indicator = I_t (0.0 or 1.0) weight = alpha

    CFG null_prob is handled by randomly replacing I_t with ADV_NULL at data
    conversion time (simpler than online dropout at training time).

    state_dim_recap = 16  (ee_pose(7) + obj_pose(7) + gripper(1) + indicator(1))
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)
    if not force and (output_dir / "meta" / "info.json").exists():
        print(f"  [skip] LeRobot dataset already exists: {output_dir}")
        data = np.load(npz_path, allow_pickle=True)
        ep0 = data["episodes"][0]
        h, w = ep0["images"].shape[1], ep0["images"].shape[2]
        has_wrist = "wrist_images" in ep0
        return h, w, has_wrist

    print(f"  Converting {npz_path} → {output_dir}")
    data       = np.load(npz_path, allow_pickle=True)
    episodes   = list(data["episodes"])
    rng        = np.random.default_rng(seed)

    ep0       = episodes[0]
    H, W      = ep0["images"].shape[1], ep0["images"].shape[2]
    has_wrist = "wrist_images" in ep0

    if "indicators" not in ep0:
        raise ValueError(
            "NPZ file does not contain 'indicators' field.  "
            "Run step 3 (3_compute_advantages.py) first."
        )

    state_dim = 16  # 15 base + 1 indicator
    state_names = [
        "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
        "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
        "gripper", "adv_indicator",
    ]
    action_dim = 8

    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, H, W),
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
            "names": ["goal_x", "goal_y", "goal_z",
                      "goal_qw", "goal_qx", "goal_qy", "goal_qz", "gripper"],
        },
    }
    if has_wrist:
        wH, wW = ep0["wrist_images"].shape[1], ep0["wrist_images"].shape[2]
        features["observation.wrist_image"] = {
            "dtype": "video",
            "shape": (3, wH, wW),
            "names": ["channel", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="franka",
        use_videos=True,
        image_writer_threads=4,
    )

    total_frames = 0
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        images     = ep["images"]
        ee_pose    = ep["ee_pose"]
        obj_pose   = ep["obj_pose"]
        actions    = ep["action"]
        indicators = ep["indicators"]               # List[int], 0 or 1
        wrist_images = ep.get("wrist_images", None)
        # Support both formats: explicit 'gripper' key (demo) or action[:,7] (DAgger)
        if "gripper" in ep:
            gripper = ep["gripper"]
            _gripper_fn = lambda t: float(gripper[t]) if gripper.ndim == 1 else float(gripper[t, 0])
        else:
            _gripper_fn = lambda t: float(actions[t, 7])

        for t in range(T):
            # Base state (15-d)
            g = _gripper_fn(t)
            base_state = np.concatenate([ee_pose[t], obj_pose[t], [g]]).astype(np.float32)

            # Apply CFG null dropout: with null_prob, replace indicator with 0.5
            raw_indicator = float(indicators[t])
            if rng.random() < null_prob:
                adv_indicator = ADV_NULL
            else:
                adv_indicator = raw_indicator

            state_recap = np.append(base_state, adv_indicator).astype(np.float32)

            frame = {
                "observation.image": images[t],
                "observation.state": state_recap,
                "action": actions[t].astype(np.float32),
                "task": "recap_pick_and_place",
            }
            if wrist_images is not None:
                frame["observation.wrist_image"] = wrist_images[t]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += T

        if (ep_idx + 1) % 50 == 0:
            print(f"    [{ep_idx+1}/{len(episodes)}] {total_frames} frames")

    dataset.finalize()
    print(f"  Converted {len(episodes)} episodes, {total_frames} frames → {output_dir}")
    return H, W, has_wrist


# ============================================================
# Checkpoint migration: load pretrained weights into 16-dim state policy
# ============================================================

def load_pretrained_for_recap(
    pretrained_dir: Path,
    new_policy_cfg,
    device: str,
    n_obs_steps: int,
):
    """Build new 16-dim state policy and migrate weights from pretrained 15-dim."""
    from safetensors.torch import load_file
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import migrate_checkpoint_for_recap

    # New policy with state_dim=16
    new_policy = DiffusionPolicy(new_policy_cfg)

    # Load pretrained safetensors
    model_path = pretrained_dir / "model.safetensors"
    old_sd = load_file(str(model_path))

    # Migrate state_dict (expand cond_encoder input layers)
    migrated_sd = migrate_checkpoint_for_recap(
        pretrained_state_dict=old_sd,
        new_policy=new_policy,
        extra_state_dims=1,
        n_obs_steps=n_obs_steps,
    )

    new_policy.load_state_dict(migrated_sd, strict=False)
    new_policy = new_policy.to(device)
    print(f"  Pretrained weights loaded with state_dim expansion (15→16).")
    return new_policy


# ============================================================
# DDIM monkey-patch (same as 4_train.py)
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = _parse_args()

    # Patch DDIM defaults
    _ddim_overrides["noise_scheduler_type"] = args.noise_scheduler_type
    _ddim_overrides["num_inference_steps"]  = args.num_inference_steps
    _RealDiffusionConfig.__init__ = _patched_diffusion_init

    try:
        from datetime import timedelta

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        local_rank  = int(os.environ.get("LOCAL_RANK", 0))
        world_size  = int(os.environ.get("WORLD_SIZE", 1))
        is_main     = local_rank == 0

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        lerobot_dir = Path(args.lerobot_dataset_dir) if args.lerobot_dataset_dir else (
            out_dir / "lerobot_dataset"
        )

        # ---- Step 1: Convert NPZ → LeRobot (BEFORE DDP init, main only) ----
        # Conversion involves heavy video encoding and can take tens of minutes.
        # Doing it before init_process_group avoids NCCL barrier timeouts.
        meta_file = out_dir / ".conversion_meta.json"

        if not args.skip_convert and is_main:
            image_height, image_width, has_wrist = convert_npz_to_lerobot_recap(
                npz_path=args.npz_path,
                output_dir=lerobot_dir,
                fps=args.fps,
                repo_id="local/recap_diffusion",
                force=args.force_convert,
                null_prob=args.null_prob,
                alpha=args.alpha,
                image_height=args.image_height,
                image_width=args.image_width,
                seed=args.seed,
            )
            with open(meta_file, "w") as f:
                json.dump({
                    "image_height": image_height,
                    "image_width": image_width,
                    "has_wrist": has_wrist,
                }, f, indent=2)

        # Non-main ranks: wait for conversion to finish before DDP init
        if not is_main and not args.skip_convert:
            print(f"  [rank {local_rank}] Waiting for dataset conversion on rank 0...")
            while not meta_file.exists():
                time.sleep(10)
            print(f"  [rank {local_rank}] Conversion complete, proceeding to DDP init.")

        # NOW init DDP — all ranks are ready, conversion is done
        if world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=1800),
            )

        if not is_main or args.skip_convert:
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                image_height = meta["image_height"]
                image_width  = meta["image_width"]
                has_wrist    = meta["has_wrist"]
            else:
                # Infer from NPZ
                data = np.load(args.npz_path, allow_pickle=True)
                ep0  = data["episodes"][0]
                image_height, image_width = ep0["images"].shape[1], ep0["images"].shape[2]
                has_wrist = "wrist_images" in ep0

        # ---- Step 2: Build new 16-dim DiffusionConfig ----
        # Import original train module to reuse train_with_lerobot_api
        _orig_spec = importlib.util.spec_from_file_location(
            "_orig_train_diffusion",
            str(Path(__file__).resolve().parent.parent
                / "scripts_pick_place" / "4_train_diffusion.py"),
        )
        _orig_mod = importlib.util.module_from_spec(_orig_spec)
        _orig_spec.loader.exec_module(_orig_mod)

        # We set include_obj_pose=True, include_gripper=True (state_dim=15)
        # BUT we add 1 extra dim (indicator) → state_dim_recap = 16
        args.include_obj_pose   = True
        args.include_gripper    = True
        args.overfit            = False
        args.val_split          = 0.0
        args.sample_weights     = None
        args.enable_xyz_viz     = False
        args.viz_save_freq      = 99999999
        args.val_freq           = 500
        args.overwrite          = True
        args.finetune           = False
        args.dataset            = args.npz_path  # for logging
        args.vision_backbone    = "resnet18"
        args.crop_shape         = [128, 128]
        args.pretrained_backbone_weights = None

        if is_main:
            print(f"\n{'='*60}")
            print(f"RECAP Fine-tuning")
            print(f"  state_dim:   16 (15 base + 1 advantage indicator)")
            print(f"  null_prob:   {args.null_prob}")
            print(f"  alpha:       {args.alpha}")
            print(f"  steps:       {args.steps}")
            print(f"  scheduler:   {args.noise_scheduler_type} ({args.num_inference_steps} steps)")
            print(f"{'='*60}\n")

        # Train (train_with_lerobot_api handles DDP, checkpointing, WandB)
        # We pass state_dim_recap=16 via a patched args attribute so the
        # LeRobot dataset reading uses the correct state dimensionality.
        result = _orig_mod.train_with_lerobot_api(
            args=args,
            lerobot_dataset_dir=lerobot_dir,
            image_height=image_height,
            image_width=image_width,
            has_wrist=has_wrist,
            include_obj_pose=True,
            include_gripper=True,
            state_dim_override=16,          # passes 16-dim state to DiffusionConfig
            pretrained_for_recap=str(Path(args.policy)),  # signals to load+migrate
            train_episodes=None,
            val_episodes=None,
        )

        if is_main:
            print(f"\n{'='*60}")
            print(f"RECAP fine-tuning complete!")
            print(f"  Output: {result['output_dir']}")
            print(f"  Steps:  {result['steps']}")
            print(f"  state_dim: 16 (indicator at dim 15)")
            print(f"  Inference tip: always pass indicator=1.0 for 'positive' conditioning")
            print(f"{'='*60}")

    finally:
        _RealDiffusionConfig.__init__ = _orig_diffusion_init


if __name__ == "__main__":
    main()
