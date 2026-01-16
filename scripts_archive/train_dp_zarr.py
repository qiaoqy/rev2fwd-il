#!/usr/bin/env python3
"""
Train a vision-based Diffusion Policy from a custom Zarr dataset.

Zarr layout expected:
  diffusion_dataset.zarr/
    episode_000000/
      rgb:     (T,H,W,3) uint8
      ee_pose: (T,7) float32
      action:  (T,8) float32
      obs:     (T,36) float32 (optional)
      ...
      attrs["success"]: bool (optional)

We convert it into LeRobot dataset format and train a diffusion policy using LeRobot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import zarr


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train diffusion policy from Zarr dataset (LeRobot).")

    # input/output
    p.add_argument("--zarr", type=str, default="output/dp/diffusion_dataset.zarr", help="Path to diffusion_dataset.zarr directory.")
    p.add_argument("--out", type=str, default="runs/diffusion_from_zarr", help="Output directory.")
    p.add_argument(
        "--lerobot_dataset_dir",
        type=str,
        default=None,
        help="Where to store converted LeRobot dataset. Default: {out}/lerobot_dataset",
    )

    # convert control
    p.add_argument("--convert_only", action="store_true", help="Only convert dataset, do not train.")
    p.add_argument("--skip_convert", action="store_true", help="Assume dataset already converted.")
    p.add_argument("--force_convert", action="store_true", help="Delete and re-convert if exists.")

    # dataset settings
    p.add_argument("--fps", type=int, default=20, help="Dataset FPS meta for LeRobot.")
    p.add_argument("--max_episodes", type=int, default=-1, help="Use only first N episodes (-1=all).")
    p.add_argument(
        "--drop_last",
        action="store_true",
        help="Drop last frame per episode (sometimes used if last action is dummy).",
    )

    # training hyperparams
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)

    # diffusion policy
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--vision_backbone", type=str, default="resnet18")
    p.add_argument("--crop_shape", type=int, nargs=2, default=[256, 256])
    p.add_argument("--num_train_timesteps", type=int, default=100)

    # misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="diffusion_from_zarr")

    return p.parse_args()


# -----------------------------
# Zarr helpers
# -----------------------------
def list_episode_groups(root: zarr.Group) -> list[str]:
    # robust for zarr v2/v3: group_keys() returns immediate child groups
    keys = sorted([k for k in root.group_keys() if k.startswith("episode_")])
    return keys


def read_episode_arrays(g: zarr.Group) -> dict:
    ep = {}
    # required
    ep["rgb"] = np.asarray(g["rgb"])         # (T,H,W,3) uint8
    ep["ee_pose"] = np.asarray(g["ee_pose"]) # (T,7) float32
    ep["action"] = np.asarray(g["action"])   # (T,8) float32

    # optional
    if "obs" in g:
        ep["obs"] = np.asarray(g["obs"])
    ep["success"] = bool(g.attrs.get("success", False))
    if "goal_pose" in g:
        ep["goal_pose"] = np.asarray(g["goal_pose"])
    return ep


# -----------------------------
# Convert Zarr -> LeRobot
# -----------------------------
def convert_zarr_to_lerobot(
    zarr_path: str,
    output_dir: str,
    fps: int = 20,
    repo_id: str = "local/diffusion_from_zarr",
    force: bool = False,
    max_episodes: int = -1,
    drop_last: bool = False,
) -> tuple[int, int]:
    """
    Returns (H, W).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    zarr_path = Path(zarr_path)
    out = Path(output_dir)

    if out.exists() and force:
        shutil.rmtree(out)

    if out.exists() and not force:
        # try to infer H,W from existing conversion target OR from zarr
        root = zarr.open(str(zarr_path), mode="r")
        ep_names = list_episode_groups(root)
        if not ep_names:
            raise ValueError(f"No episode_* groups found in {zarr_path}")
        ep0 = read_episode_arrays(root[ep_names[0]])
        H, W = ep0["rgb"].shape[1], ep0["rgb"].shape[2]
        print(f"[INFO] LeRobot dataset exists at {out}, skip conversion. Inferred image: {H}x{W}")
        return H, W

    print("\n" + "=" * 60)
    print("Converting Zarr -> LeRobot format")
    print("=" * 60)
    print(f"  Zarr:   {zarr_path}")
    print(f"  Output: {out}")
    print(f"  FPS:    {fps}")
    print("=" * 60)

    root = zarr.open(str(zarr_path), mode="r")
    ep_names = list_episode_groups(root)
    print("ep_names:", ep_names)
    # exit(0)
    if not ep_names:
        raise ValueError(f"No episode_* groups found in {zarr_path}")

    if max_episodes is not None and max_episodes > 0:
        ep_names = ep_names[:max_episodes]
        print(f"[INFO] Using first {len(ep_names)} episodes due to --max_episodes")

    # print("max_episodes: ", max_episodes)
    # exit(0)

    ep0 = read_episode_arrays(root[ep_names[0]])
    rgb0 = ep0["rgb"]
    if rgb0.ndim != 4 or rgb0.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (T,H,W,3), got {rgb0.shape}")

    H, W = rgb0.shape[1], rgb0.shape[2]
    # print("H, W:", H, W)
    # exit(0)
    state_dim = 7
    action_dim = 8

    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
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

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out,
        robot_type="franka",
        use_videos=False,
    )

    start = time.time()
    total_frames = 0

    for i, ep_name in enumerate(ep_names):
        print("ep_names:", i, ep_names)
        g = root[ep_name]
        ep = read_episode_arrays(g)

        rgb = ep["rgb"]
        ee = ep["ee_pose"].astype(np.float32)
        act = ep["action"].astype(np.float32)

        T = min(len(rgb), len(ee), len(act))
        if drop_last and T > 0:
            T = T - 1

        for t in range(T):
            img = rgb[t]  # (H,W,3) uint8
            # print("img: ",img.max(), img.min())
            # img_hwc = img.astype(np.float32) / 255.0  # (3,H,W) float32: TODO: not sure 255.0???
            # print("img_chw:", img_hwc.shape)

            frame = {
                "observation.image": img,
                "observation.state": ee[t].astype(np.float32),
                "action": act[t].astype(np.float32),
                "task": "pick_and_place",
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += T

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start
            print(f"  Converted {i+1}/{len(ep_names)} episodes  | frames={total_frames}  | {((i+1)/elapsed):.2f} ep/s")
    # exit(0)
    dataset.finalize()
    elapsed = time.time() - start
    print("=" * 60)
    print(f"Conversion done. Episodes={len(ep_names)} frames={total_frames} time={elapsed:.1f}s")
    print("=" * 60 + "\n")

    return H, W


# -----------------------------
# Train via LeRobot API
# -----------------------------
def train_with_lerobot(
    args: argparse.Namespace,
    lerobot_dataset_dir: Path,
    H: int,
    W: int,
) -> None:
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.scripts.lerobot_train import train

    # device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_features = {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, H, W),
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

    policy_cfg = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        input_features=input_features,
        output_features=output_features,
        vision_backbone=args.vision_backbone,
        # pre_norm=False,
        use_group_norm=False,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        crop_shape=tuple(args.crop_shape),
        crop_is_random=False,
        num_train_timesteps=args.num_train_timesteps,
        device=device,
        push_to_hub=False,
        optimizer_lr=args.lr,
    )

    dataset_cfg = DatasetConfig(
        repo_id="local/diffusion_from_zarr",
        root=str(lerobot_dataset_dir),
    )

    wandb_cfg = WandBConfig(enable=args.wandb, project=args.wandb_project)

    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    # ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=ckpt_dir,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=100,
        save_freq=10000,
        save_checkpoint=True,
        wandb=wandb_cfg,
        resume=args.resume,
        eval_freq=0,
    )

    print("\n" + "=" * 60)
    print("Starting training (LeRobot Diffusion Policy)")
    print("=" * 60)
    print(f"  Zarr: {args.zarr}")
    print(f"  LeRobot dataset: {lerobot_dataset_dir}")
    print(f"  Output: {ckpt_dir}")
    print(f"  Steps: {args.steps}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"  Image: {H}x{W}  Crop: {tuple(args.crop_shape)}")
    print(f"  n_obs_steps: {args.n_obs_steps}  horizon: {args.horizon}  n_action_steps: {args.n_action_steps}")
    print(f"  device: {device}")
    print("=" * 60 + "\n")

    train(train_cfg)


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lerobot_dataset_dir = Path(args.lerobot_dataset_dir) if args.lerobot_dataset_dir else (out_dir / "lerobot_dataset")

    # 1) convert
    if not args.skip_convert:
        H, W = convert_zarr_to_lerobot(
            zarr_path=args.zarr,
            output_dir=str(lerobot_dataset_dir),
            fps=args.fps,
            repo_id="local/diffusion_from_zarr",
            force=args.force_convert,
            max_episodes=args.max_episodes,
            drop_last=args.drop_last,
        )
    else:
        # infer H,W from zarr
        root = zarr.open(str(args.zarr), mode="r")
        ep_names = list_episode_groups(root)
        ep0 = read_episode_arrays(root[ep_names[0]])
        H, W = ep0["rgb"].shape[1], ep0["rgb"].shape[2]
        print(f"[INFO] --skip_convert, inferred image: {H}x{W}")

    if args.convert_only:
        print("[DONE] convert_only set, exiting.")
        return

    # 2) train
    train_with_lerobot(args, lerobot_dataset_dir, H, W)


if __name__ == "__main__":
    main()
