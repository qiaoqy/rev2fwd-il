#!/usr/bin/env python3
"""Script 3: Convert Inovo (RoboKit) data to LeRobot v3.0 format.

Converts per-frame RoboKit NPZ files into the LeRobot dataset format expected
by the DiT Flow / Diffusion training scripts.

=============================================================================
FORMAT MAPPING
=============================================================================
  Source (RoboKit NPZ)            Target (LeRobot)
  ─────────────────────           ───────────────────────
  primary_rgb (480×848)       ->  observation.image (H×W resized)
  gripper_rgb (480×848)       ->  observation.wrist_image (H×W resized)
  robot_obs[0:6] (tcp pose)   ->  observation.state (6D or 7D)
  robot_obs[6] (gripper width)->  observation.state[6] (if --include_gripper)
  actions[0:6] (velocity)     ->  action[0:6] (delta pose or velocity)
  actions[6] (gripper)        ->  action[6] (binary gripper)

=============================================================================
ACTION FORMAT OPTIONS
=============================================================================
Option A: --action_format delta_pose  (DEFAULT, recommended)
    action[t] = robot_obs[t+1][0:6] - robot_obs[t][0:6]
    + Compatible with existing Piper training infrastructure
    + Reuse maximum code with DiT Flow / Diffusion training scripts

Option B: --action_format velocity
    action[t] = actions[t][0:6]  (keep native velocity commands)
    + No conversion artifacts
    - May need different action post-processing at inference

=============================================================================
USAGE
=============================================================================
# Convert Task B (original) to LeRobot with delta pose actions
python scripts/scripts_task_inovo/3_convert_to_lerobot.py \
    --input data/inovo_data/0209_tower_boby \
    --output runs/inovo_B/lerobot_dataset \
    --image_size 240 320 \
    --include_gripper \
    --action_format delta_pose

# Convert Task A (reversed) to LeRobot
python scripts/scripts_task_inovo/3_convert_to_lerobot.py \
    --input data/inovo_data/tower_boby_A \
    --output runs/inovo_A/lerobot_dataset \
    --image_size 240 320 \
    --include_gripper \
    --action_format delta_pose

# Convert with native velocity actions
python scripts/scripts_task_inovo/3_convert_to_lerobot.py \
    --input data/inovo_data/0209_tower_boby \
    --output runs/inovo_B_vel/lerobot_dataset \
    --include_gripper \
    --action_format velocity

# Center-crop to 640x480 before resize (preserve 4:3 aspect ratio)
python scripts/scripts_task_inovo/3_convert_to_lerobot.py \
    --input data/inovo_data/0209_tower_boby \
    --output runs/inovo_B/lerobot_dataset \
    --include_gripper \
    --center_crop 480 640 \
    --image_size 240 320
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Suppress verbose output from video encoders
os.environ["AV_LOG_LEVEL"] = "quiet"
os.environ["SVT_LOG"] = "0"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"


# =============================================================================
# RoboKit NPZ Decoder
# =============================================================================

def load_robokit_frame(path: str | Path) -> dict:
    """Load and decode a single RoboKit NPZ frame."""
    f = np.load(str(path), allow_pickle=True)

    primary_rgb = cv2.imdecode(
        np.frombuffer(f["primary_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if primary_rgb is not None:
        primary_rgb = cv2.cvtColor(primary_rgb, cv2.COLOR_BGR2RGB)

    gripper_rgb = cv2.imdecode(
        np.frombuffer(f["gripper_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if gripper_rgb is not None:
        gripper_rgb = cv2.cvtColor(gripper_rgb, cv2.COLOR_BGR2RGB)

    robot_obs = pickle.loads(f["robot_obs"].item())
    actions = pickle.loads(f["actions"].item())

    return {
        "primary_rgb": primary_rgb,
        "gripper_rgb": gripper_rgb,
        "robot_obs": np.asarray(robot_obs, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
    }


# =============================================================================
# Episode Discovery
# =============================================================================

def discover_episodes(data_dir: Path) -> list[dict]:
    """Discover episode directories."""
    episodes = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name in (
            "extracted", "__pycache__", "viz_videos"
        ):
            continue
        npz_files = sorted(entry.glob("*.npz"))
        if len(npz_files) == 0:
            continue
        episodes.append({
            "name": entry.name,
            "path": entry,
            "npz_files": npz_files,
        })
    return episodes


# =============================================================================
# Angle Wrapping
# =============================================================================

def wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Image Pre-processing
# =============================================================================

def preprocess_image(
    img: np.ndarray,
    center_crop: tuple[int, int] | None,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Center-crop (optional) then resize an image.

    Args:
        img: (H, W, 3) uint8 RGB
        center_crop: (crop_H, crop_W) or None
        target_size: (target_H, target_W)

    Returns:
        Preprocessed (target_H, target_W, 3) uint8 image.
    """
    if center_crop is not None:
        crop_h, crop_w = center_crop
        h, w = img.shape[:2]
        y0 = max(0, (h - crop_h) // 2)
        x0 = max(0, (w - crop_w) // 2)
        img = img[y0:y0 + crop_h, x0:x0 + crop_w]

    target_h, target_w = target_size
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return img


# =============================================================================
# Conversion Core
# =============================================================================

def convert_to_lerobot(
    data_dir: Path,
    output_dir: Path,
    fps: int = 30,
    repo_id: str = "local/inovo_ditflow",
    image_size: tuple[int, int] = (240, 320),
    center_crop: tuple[int, int] | None = None,
    include_gripper: bool = True,
    action_format: str = "delta_pose",
    num_episodes: int = -1,
    force: bool = False,
) -> tuple[int, int, bool]:
    """Convert RoboKit episodes to LeRobot v3.0 format.

    Args:
        data_dir: Directory with episode sub-dirs.
        output_dir: Output LeRobot dataset directory.
        fps: Dataset FPS.
        repo_id: LeRobot repository ID.
        image_size: (H, W) target image size.
        center_crop: (H, W) center crop before resize, or None.
        include_gripper: Include gripper width in observation.state.
        action_format: "delta_pose" or "velocity".
        num_episodes: Limit episodes (-1 = all).
        force: Force re-conversion.

    Returns:
        (image_height, image_width, has_wrist_camera)
    """
    logging.getLogger("imageio").setLevel(logging.ERROR)
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    logging.getLogger("av").setLevel(logging.ERROR)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)

    # Check existing dataset
    if output_dir.exists() and not force:
        meta_path = output_dir / "meta" / "info.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    info = json.load(f)
                n_ep = info.get("total_episodes", 0)
                n_fr = info.get("total_frames", 0)
                if n_ep > 0 and n_fr > 0:
                    features = info.get("features", {})
                    img_shape = features.get("observation.image", {}).get("shape", [3, 240, 320])
                    has_wrist = "observation.wrist_image" in features
                    print(f"LeRobot dataset already exists at {output_dir} "
                          f"({n_ep} episodes, {n_fr} frames)")
                    print("  Use --force to re-convert.")
                    return img_shape[1], img_shape[2], has_wrist
            except Exception:
                pass
        # Remove incomplete
        print(f"Removing incomplete dataset at {output_dir}")
        shutil.rmtree(output_dir)

    if output_dir.exists() and force:
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    # Discover episodes
    episodes = discover_episodes(data_dir)
    if not episodes:
        raise ValueError(f"No episodes found in {data_dir}")
    if num_episodes > 0:
        episodes = episodes[:num_episodes]

    image_height, image_width = image_size
    has_wrist = True  # Inovo always has gripper camera

    # State dimension
    state_dim = 6  # tcp_pose [x, y, z, roll, pitch, yaw]
    state_names = ["tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw"]
    if include_gripper:
        state_dim += 1
        state_names.append("gripper_width")

    # Action dimension: 7 (xyz + rpy + gripper)
    action_dim = 7
    if action_format == "delta_pose":
        action_names = ["delta_x", "delta_y", "delta_z",
                        "delta_roll", "delta_pitch", "delta_yaw", "gripper"]
    else:
        action_names = ["v_x", "v_y", "v_z",
                        "v_roll", "v_pitch", "v_yaw", "gripper"]

    print(f"\n{'='*60}")
    print("Converting Inovo (RoboKit) data to LeRobot format")
    print(f"{'='*60}")
    print(f"  Input: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  FPS: {fps}")
    print(f"  Image size: ({image_height}, {image_width})")
    if center_crop:
        print(f"  Center crop: {center_crop}")
    print(f"  State dim: {state_dim} ({'+ gripper' if include_gripper else 'pose only'})")
    print(f"  Action format: {action_format}")
    print(f"  Action dim: {action_dim}")
    print(f"{'='*60}\n")

    # Define features
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (3, image_height, image_width),
            "names": ["channel", "height", "width"],
        },
        "observation.wrist_image": {
            "dtype": "video",
            "shape": (3, image_height, image_width),
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
            "names": action_names,
        },
    }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="inovo",
        use_videos=True,
        image_writer_threads=4,
    )

    total_frames = 0
    start_time = time.time()
    print_freq = max(1, len(episodes) // 20)

    for ep_idx, ep in enumerate(episodes):
        npz_files = ep["npz_files"]
        T = len(npz_files)

        if (ep_idx + 1) % print_freq == 0 or ep_idx == 0 or len(episodes) <= 20:
            elapsed = time.time() - start_time
            rate = ep_idx / elapsed if elapsed > 0 and ep_idx > 0 else 0
            eta = (len(episodes) - ep_idx) / rate if rate > 0 else 0
            print(f"  [{ep_idx+1}/{len(episodes)}] {ep['name']} ({T} frames) | "
                  f"{rate:.1f} ep/s | ETA: {eta:.0f}s")

        # Load all frames for this episode
        frames_data = []
        for npz_path in npz_files:
            frames_data.append(load_robokit_frame(npz_path))

        # Build arrays
        tcp_poses = np.array([f["robot_obs"][:6] for f in frames_data])   # (T, 6)
        gripper_widths = np.array([f["robot_obs"][6] for f in frames_data])  # (T,)
        raw_actions = np.array([f["actions"] for f in frames_data])       # (T, 7)

        # Compute actions based on format
        if action_format == "delta_pose":
            # delta_pose[t] = tcp_pose[t+1] - tcp_pose[t]
            actions = np.zeros((T, 7), dtype=np.float32)
            for t in range(T - 1):
                delta_pos = tcp_poses[t + 1, :3] - tcp_poses[t, :3]
                delta_ori = wrap_angle(tcp_poses[t + 1, 3:6] - tcp_poses[t, 3:6])
                actions[t, :3] = delta_pos
                actions[t, 3:6] = delta_ori
                actions[t, 6] = raw_actions[t, 6]
            # Last action: zero movement, keep gripper
            actions[T - 1, :6] = 0.0
            actions[T - 1, 6] = raw_actions[T - 1, 6]
        else:
            # velocity: keep raw actions as-is
            actions = raw_actions.astype(np.float32)

        # Add frames to dataset
        for t in range(T):
            # Preprocess images
            primary = frames_data[t]["primary_rgb"]
            gripper = frames_data[t]["gripper_rgb"]

            if primary is None:
                primary = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            else:
                primary = preprocess_image(primary, center_crop, (image_height, image_width))

            if gripper is None:
                gripper = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            else:
                gripper = preprocess_image(gripper, center_crop, (image_height, image_width))

            # Build state
            state = tcp_poses[t].astype(np.float32)
            if include_gripper:
                state = np.concatenate([state, [gripper_widths[t]]]).astype(np.float32)

            frame_dict = {
                "observation.image": primary,
                "observation.wrist_image": gripper,
                "observation.state": state,
                "action": actions[t],
                "task": "inovo_tower_body",
            }
            dataset.add_frame(frame_dict)

        dataset.save_episode()
        total_frames += T

    # Finalize
    print("\nFinalizing dataset (encoding videos) ...")
    dataset.finalize()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Action format: {action_format}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Dataset saved to: {output_dir}")
    print(f"{'='*60}\n")

    return image_height, image_width, has_wrist


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Inovo (RoboKit) data to LeRobot v3.0 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing episode sub-directories.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Dataset FPS. Default: 30.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[240, 320],
        help="Target image size (H W). Default: 240 320.",
    )
    parser.add_argument(
        "--center_crop",
        type=int,
        nargs=2,
        default=None,
        help="Center crop (H W) before resize. Default: None (no crop). "
             "E.g., 480 640 to crop 848×480 to 640×480 (4:3).",
    )
    parser.add_argument(
        "--include_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include gripper width in observation.state. Default: True.",
    )
    parser.add_argument(
        "--action_format",
        type=str,
        default="delta_pose",
        choices=["delta_pose", "velocity"],
        help="Action format: 'delta_pose' (compute from obs diff) or 'velocity' (native). "
             "Default: delta_pose.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=-1,
        help="Number of episodes to convert (-1 = all). Default: -1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if dataset exists.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="local/inovo_ditflow",
        help="LeRobot repository ID. Default: local/inovo_ditflow.",
    )

    args = parser.parse_args()

    convert_to_lerobot(
        data_dir=Path(args.input),
        output_dir=Path(args.output),
        fps=args.fps,
        repo_id=args.repo_id,
        image_size=tuple(args.image_size),
        center_crop=tuple(args.center_crop) if args.center_crop else None,
        include_gripper=args.include_gripper,
        action_format=args.action_format,
        num_episodes=args.num_episodes,
        force=args.force,
    )


if __name__ == "__main__":
    main()
