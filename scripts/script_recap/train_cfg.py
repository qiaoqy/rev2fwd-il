#!/usr/bin/env python3
"""CFG DiffusionPolicy training for RECAP (Exp56).

Key differences from standard training (4_train.py):
  (a) NPZ → LeRobot conversion concats indicator to observation.state (dim 16)
  (b) Monkey-patches DiffusionPolicy.compute_loss for CFG dropout:
      30% probability → mask entire global_cond to 0
  (c) Always initializes from specified checkpoint (πpre, dim=16)

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/script_recap/train_cfg.py \
        --labeled_npz iter1_labeled.npz \
        --checkpoint work/PP_A_pre/checkpoints/checkpoints/last/pretrained_model \
        --out iter1_ckpt_A \
        --steps 5000 --batch_size 256 --lr 2e-4 \
        --cfg_dropout_prob 0.3 --seed 42 --wandb

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        scripts/script_recap/train_cfg.py \
        --labeled_npz iter1_labeled.npz \
        --checkpoint work/PP_A_pre/checkpoints/checkpoints/last/pretrained_model \
        --out iter1_ckpt_A \
        --steps 5000 --batch_size 256 --lr 2e-4 \
        --cfg_dropout_prob 0.3 --seed 42 --wandb
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
# Import the original training module
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SCRIPT = _THIS_DIR.parent / "scripts_pick_place" / "4_train_diffusion.py"

_spec = importlib.util.spec_from_file_location("_orig_train_diffusion", _ORIG_SCRIPT)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)

load_episodes_from_npz = _orig_mod.load_episodes_from_npz
train_with_lerobot_api = _orig_mod.train_with_lerobot_api


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CFG DiffusionPolicy training for RECAP (Exp56).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    parser.add_argument("--labeled_npz", type=str, required=True,
                        help="Labeled NPZ with indicator per frame.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pretrained model dir (πpre, dim=16) to initialize from.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for trained checkpoint.")
    parser.add_argument("--lerobot_dataset_dir", type=str, default=None,
                        help="LeRobot dataset path. Default: <out>/lerobot_dataset.")

    # CFG
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.3,
                        help="CFG dropout probability (0.3 = 30%% of training batch).")

    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)

    # Architecture
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--vision_backbone", type=str, default="resnet18")
    parser.add_argument("--crop_shape", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--noise_scheduler_type", type=str, default="DDIM",
                        choices=["DDPM", "DDIM"])
    parser.add_argument("--num_inference_steps", type=int, default=10)

    # Logging
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=20000)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rev2fwd-pick-place-simulator")

    # Conversion
    parser.add_argument("--force_convert", action="store_true")
    parser.add_argument("--skip_convert", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# NPZ → LeRobot conversion with indicator concat to state
# ---------------------------------------------------------------------------
def convert_labeled_npz_to_lerobot(
    npz_path: str,
    output_dir: str | Path,
    fps: int = 20,
    force: bool = False,
) -> tuple[int, int, bool]:
    """Convert labeled NPZ (with indicator) to LeRobot format.

    State = [ee_pose(7), obj_pose(7), gripper(1), indicator(1)] = dim 16.

    Args:
        npz_path: Labeled NPZ path.
        output_dir: LeRobot dataset output directory.
        fps: Frames per second.
        force: Force re-conversion.

    Returns:
        (image_height, image_width, has_wrist)
    """
    import logging
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    logging.getLogger("av").setLevel(logging.ERROR)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)

    # Check if exists
    if output_dir.exists() and not force:
        meta_path = output_dir / "meta" / "info.json"
        if meta_path.exists():
            with open(meta_path) as f:
                info = json.load(f)
            features = info.get("features", {})
            img_shape = features["observation.image"]["shape"]
            image_height, image_width = img_shape[1], img_shape[2]
            has_wrist = "observation.wrist_image" in features
            print(f"LeRobot dataset already exists at {output_dir}, skipping conversion.")
            return image_height, image_width, has_wrist

    # Remove existing
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    # Load episodes
    episodes = load_episodes_from_npz(npz_path)

    if not episodes:
        raise ValueError(f"No episodes in {npz_path}")

    ep0 = episodes[0]
    image_shape = ep0["images"].shape[1:]  # (H, W, 3)
    has_wrist = "wrist_images" in ep0

    # State dim = ee(7) + obj(7) + gripper(1) + indicator(1) = 16
    state_dim = 16
    action_dim = 8

    print(f"\n{'='*60}")
    print("Converting Labeled NPZ to LeRobot (CFG, state_dim=16)")
    print(f"{'='*60}")
    print(f"  Input: {npz_path}")
    print(f"  Output: {output_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  State dim: {state_dim} (ee+obj+gripper+indicator)")
    print(f"  Image: {image_shape}")
    print(f"  Has wrist: {has_wrist}")

    # Check indicator
    n_with_indicator = sum(1 for ep in episodes if "indicator" in ep)
    n_without = len(episodes) - n_with_indicator
    print(f"  With indicator: {n_with_indicator}, Without: {n_without}")
    if n_without > 0:
        print(f"  WARNING: {n_without} episodes missing indicator — will use +1 (demo default)")

    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, image_shape[0], image_shape[1]),
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
                "gripper", "indicator",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["goal_x", "goal_y", "goal_z", "goal_qw", "goal_qx", "goal_qy", "goal_qz", "gripper"],
        },
    }

    if has_wrist:
        wrist_shape = ep0["wrist_images"].shape[1:]
        features["observation.wrist_image"] = {
            "dtype": "video",
            "shape": (3, wrist_shape[0], wrist_shape[1]),
            "names": ["channel", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id="local/recap_cfg",
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="franka",
        use_videos=True,
        image_writer_threads=4,
    )

    total_frames = 0
    start_time = time.time()

    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        ee_pose = ep["ee_pose"]       # (T, 7)
        obj_pose = ep.get("obj_pose", np.zeros((T, 7), dtype=np.float32))
        actions = ep["action"]         # (T, 8)
        gripper_states = actions[:, 7] # (T,)
        images = ep["images"]          # (T, H, W, 3)
        wrist_images = ep.get("wrist_images", None)

        # Indicator: use episode's indicator or default to +1 (demo)
        if "indicator" in ep:
            indicator = ep["indicator"]  # (T,)
        else:
            indicator = np.ones(T, dtype=np.float32)

        if (ep_idx + 1) % max(1, len(episodes) // 20) == 0 or ep_idx == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}] T={T}")

        for t in range(T):
            state = np.concatenate([
                ee_pose[t],                                    # (7,)
                obj_pose[t],                                   # (7,)
                np.array([gripper_states[t]], dtype=np.float32),  # (1,)
                np.array([indicator[t]], dtype=np.float32),       # (1,)
            ])  # (16,)

            frame = {
                "observation.image": images[t],
                "observation.state": state.astype(np.float32),
                "action": actions[t].astype(np.float32),
                "task": "pick_and_place_cfg",
            }
            if wrist_images is not None:
                frame["observation.wrist_image"] = wrist_images[t]

            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += T

    print("\nFinalizing dataset (encoding videos)...")
    dataset.finalize()

    elapsed = time.time() - start_time
    print(f"  Converted {len(episodes)} episodes, {total_frames} frames in {elapsed:.1f}s")
    print(f"  Saved to: {output_dir}")

    return image_shape[0], image_shape[1], has_wrist


# ---------------------------------------------------------------------------
# CFG dropout monkey-patch
# ---------------------------------------------------------------------------
_cfg_dropout_prob_global = 0.0


def _install_cfg_dropout(dropout_prob: float) -> None:
    """Monkey-patch DiffusionModel.compute_loss to add CFG dropout on global_cond.

    During training, with probability `dropout_prob`, the entire global_cond
    vector is zeroed out for each sample in the batch. This is the CFG
    unconditional training signal.
    """
    global _cfg_dropout_prob_global
    _cfg_dropout_prob_global = dropout_prob

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

    _orig_compute_loss = DiffusionModel.compute_loss

    def _cfg_compute_loss(self, batch):
        """compute_loss with CFG dropout injected."""
        import torch.nn.functional as F

        # ---- Same as original up to global_cond ----
        OBS_STATE = "observation.state"
        OBS_IMAGES = "observation.images"
        OBS_ENV_STATE = "observation.environment_state"
        ACTION = "action"

        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # (B, D)

        # ---- CFG dropout: mask entire global_cond to 0 ----
        if self.training and _cfg_dropout_prob_global > 0:
            B = global_cond.shape[0]
            mask = (torch.rand(B, 1, device=global_cond.device) < _cfg_dropout_prob_global)
            global_cond = global_cond.masked_fill(mask, 0.0)

        # ---- Forward diffusion (same as original) ----
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"'do_mask_loss_for_padding' is activated (got batch with keys {set(batch)})."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    DiffusionModel.compute_loss = _cfg_compute_loss
    print(f"  [CFG] Installed CFG dropout monkey-patch (prob={dropout_prob})")


# ---------------------------------------------------------------------------
# DDIM monkey-patch (same as 4_train.py)
# ---------------------------------------------------------------------------
from lerobot.policies.diffusion.configuration_diffusion import (
    DiffusionConfig as _RealDiffusionConfig,
)

_ddim_overrides: dict = {}
_pretrained_path_override: Path | None = None
_orig_diffusion_init = _RealDiffusionConfig.__init__


def _patched_diffusion_init(self, **kwargs):
    kwargs.setdefault("noise_scheduler_type",
                      _ddim_overrides.get("noise_scheduler_type", "DDIM"))
    kwargs.setdefault("num_inference_steps",
                      _ddim_overrides.get("num_inference_steps", 10))
    _orig_diffusion_init(self, **kwargs)
    # Inject pretrained_path so lerobot loads weights from πpre
    if _pretrained_path_override is not None:
        self.pretrained_path = _pretrained_path_override


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    # Attributes expected by train_with_lerobot_api
    args.include_obj_pose = True
    args.include_gripper = True
    args.overfit = False
    args.finetune = False
    args.resume = False
    args.overwrite = True   # Allow overwriting output dir from failed runs
    args.enable_xyz_viz = False
    args.viz_save_freq = 0
    args.val_split = 0.0
    args.val_freq = 500
    args.sample_weights = None
    args.pretrained_backbone_weights = None
    args.dataset = args.labeled_npz  # For logging only
    args.no_obj_pose = False

    state_dim = 16  # ee(7) + obj(7) + gripper(1) + indicator(1)

    # DDIM overrides
    _ddim_overrides["noise_scheduler_type"] = args.noise_scheduler_type
    _ddim_overrides["num_inference_steps"] = args.num_inference_steps

    print(f"\n{'='*60}")
    print("[CFG-RL] RECAP Policy Training")
    print(f"{'='*60}")
    print(f"  Labeled NPZ:      {args.labeled_npz}")
    print(f"  Checkpoint (πpre): {args.checkpoint}")
    print(f"  Output:            {args.out}")
    print(f"  State dim:         {state_dim} (15 + indicator)")
    print(f"  CFG dropout:       {args.cfg_dropout_prob}")
    print(f"  Steps:             {args.steps}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  LR:                {args.lr}")
    print(f"  Scheduler:         {args.noise_scheduler_type}")
    print(f"{'='*60}\n")

    # Monkey-patch DiffusionConfig for DDIM
    _RealDiffusionConfig.__init__ = _patched_diffusion_init

    # Install CFG dropout
    _install_cfg_dropout(args.cfg_dropout_prob)

    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_main = local_rank == 0

        if world_size > 1 and not torch.distributed.is_initialized():
            from datetime import timedelta
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=180),  # dataset conversion can be very slow for large datasets
            )

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        lerobot_dir = args.lerobot_dataset_dir
        if lerobot_dir is None:
            lerobot_dir = out_dir / "lerobot_dataset"
        lerobot_dir = Path(lerobot_dir)

        # ---- Step 1: Convert labeled NPZ → LeRobot ----
        if not args.skip_convert:
            if is_main:
                image_height, image_width, has_wrist = convert_labeled_npz_to_lerobot(
                    npz_path=args.labeled_npz,
                    output_dir=lerobot_dir,
                    fps=args.fps,
                    force=args.force_convert,
                )
                meta_file = out_dir / ".conversion_meta.json"
                with open(meta_file, "w") as f:
                    json.dump({
                        "image_height": image_height,
                        "image_width": image_width,
                        "has_wrist": has_wrist,
                        "state_dim": state_dim,
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
                raise FileNotFoundError(f"Cannot skip_convert: {meta_path} not found")

        # ---- Step 2: Set up pretrained πpre for finetuning ----
        # Copy checkpoint OUTSIDE the checkpoints dir (to avoid lerobot validate conflict).
        # Pretrained weights are loaded via DiffusionConfig.pretrained_path (injected by
        # _patched_diffusion_init).
        global _pretrained_path_override
        ckpt_src = Path(args.checkpoint)
        pretrained_dir = out_dir / "_pretrained"
        if is_main:
            if pretrained_dir.exists():
                shutil.rmtree(pretrained_dir)
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            for f in ckpt_src.iterdir():
                shutil.copy2(str(f), str(pretrained_dir / f.name))
            print(f"  Copied πpre checkpoint: {ckpt_src} → {pretrained_dir}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        _pretrained_path_override = pretrained_dir

        # ---- Step 3: Train ----
        result = train_with_lerobot_api(
            args=args,
            lerobot_dataset_dir=lerobot_dir,
            image_height=image_height,
            image_width=image_width,
            has_wrist=has_wrist,
            include_obj_pose=True,
            include_gripper=True,
            state_dim_override=state_dim,
        )

        print(f"\n{'='*60}")
        print("[CFG-RL] Training Complete!")
        print(f"{'='*60}")
        print(f"  Output:      {result['output_dir']}")
        print(f"  Steps:       {result['steps']}")
        print(f"  CFG dropout: {args.cfg_dropout_prob}")
        print(f"{'='*60}")

    finally:
        _RealDiffusionConfig.__init__ = _orig_diffusion_init


if __name__ == "__main__":
    main()
