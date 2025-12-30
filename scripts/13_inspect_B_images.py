#!/usr/bin/env python3
"""Inspect collected image data from Expert B rollouts.

This script provides utilities to inspect the data collected by script 12:
1. Extract a single frame and save as PNG image + JSON metadata
2. Compile an episode's image sequence into an MP4 video

The inspection data is saved in a timestamped folder under data/ for easy management.

=============================================================================
OBSERVATION VECTOR (36-dim) EXPLANATION for Isaac-Lift-Cube-Franka-IK-Abs-v0
=============================================================================
The 36-dimensional observation vector is a concatenation of several terms,
as defined in isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg.ObservationsCfg.PolicyCfg.

| Index   | Dim | Field                 | Description                            |
|---------|-----|-----------------------|----------------------------------------|
| 0-8     |  9  | joint_pos_rel         | Joint positions relative to default    |
| 9-17    |  9  | joint_vel_rel         | Joint velocities                       |
| 18-20   |  3  | object_position       | Object XYZ in robot root frame         |
| 21-27   |  7  | target_object_position| Target pose (pos_xyz + quat_wxyz)      |
| 28-35   |  8  | last_action           | Previous action (ee_pose + gripper)    |

Note: Quaternions are typically stored as [w, x, y, z].
=============================================================================

USAGE EXAMPLES
=============================================================================
# Extract frame 0 from episode 0 and create video
python scripts/13_inspect_B_images.py --dataset data/B_with_images_latest.npz

# Extract specific frame and episode
python scripts/13_inspect_B_images.py --dataset data/B_with_images_latest.npz \\
    --episode 5 --frame 50

# Custom output name and video fps
python scripts/13_inspect_B_images.py --dataset data/B_with_images_latest.npz \\
    --name my_inspection --fps 30

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect collected image data from Expert B rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/B_images_latest.npz",
        help="Path to the NPZ file containing collected episodes with images.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to inspect (default: 0).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to extract as PNG (default: 0).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for output folder. If not provided, uses timestamp.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video (default: 30).",
    )
    
    return parser.parse_args()


def load_episode_data(npz_path: str, episode_idx: int) -> dict:
    """Load data for a specific episode from NPZ file.
    
    Expects Episode format: {episodes: [ep1_dict, ep2_dict, ...]}
    Each episode contains: obs, images, ee_pose, obj_pose, gripper, place_pose, goal_pose, success
    
    This format is produced by:
    - Script 12 (reverse rollouts with images)
    - Script 22 (forward dataset with images, time-reversed from script 12)
    
    Args:
        npz_path: Path to the NPZ file.
        episode_idx: Index of the episode to load.
        
    Returns:
        Dictionary containing episode data.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Expect Episode format with 'episodes' key
    if 'episodes' not in data.files:
        raise ValueError(f"Invalid dataset format. Expected 'episodes' key but found: {data.files}")
    
    episodes = data['episodes']
    num_episodes = len(episodes)
    
    if episode_idx >= num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {num_episodes} episodes.")
    
    episode_data = episodes[episode_idx]
    
    if 'images' not in episode_data:
        raise ValueError(f"No images found for episode {episode_idx}. "
                        "Make sure the dataset was collected with script 12 or 22.")
    
    return episode_data


def save_frame_as_png(images: np.ndarray, frame_idx: int, output_path: Path) -> None:
    """Save a single frame as PNG image.
    
    Args:
        images: Array of shape (T, H, W, C) containing RGB images.
        frame_idx: Index of the frame to save.
        output_path: Path to save the PNG file.
    """
    import imageio
    
    if frame_idx >= len(images):
        raise ValueError(f"Frame {frame_idx} not found. Episode has {len(images)} frames.")
    
    frame = images[frame_idx]
    imageio.imwrite(str(output_path), frame)
    print(f"Saved frame {frame_idx} to {output_path}")


def save_frame_data_as_json(episode_data: dict, frame_idx: int, output_path: Path) -> None:
    """Save frame metadata as JSON file.
    
    Args:
        episode_data: Dictionary containing episode data.
        frame_idx: Index of the frame.
        output_path: Path to save the JSON file.
    """
    if frame_idx >= len(episode_data["obs"]):
        raise ValueError(f"Frame {frame_idx} not found. Episode has {len(episode_data['obs'])} frames.")
    
    frame_data = {
        "frame_idx": frame_idx,
        "obs": episode_data["obs"][frame_idx].tolist(),
        "ee_pose": episode_data["ee_pose"][frame_idx].tolist(),
        "obj_pose": episode_data["obj_pose"][frame_idx].tolist(),
        "gripper": float(episode_data["gripper"][frame_idx]),
        "place_pose": episode_data["place_pose"].tolist(),
        "goal_pose": episode_data["goal_pose"].tolist(),
        "success": bool(episode_data["success"]),
    }
    
    with open(output_path, "w") as f:
        json.dump(frame_data, f, indent=2)
    print(f"Saved frame {frame_idx} data to {output_path}")


def create_episode_video(images: np.ndarray, output_path: Path, fps: int) -> None:
    """Create MP4 video from episode images.
    
    Uses H.264 codec (libx264) with yuv420p pixel format for broad compatibility,
    including direct playback in VSCode.
    
    Args:
        images: Array of shape (T, H, W, C) containing RGB images.
        output_path: Path to save the MP4 file.
        fps: Frames per second for the video.
    """
    import imageio
    
    print(f"Creating video with {len(images)} frames at {fps} fps...")
    
    # Use libx264 codec with yuv420p for broad compatibility (including VSCode)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    ) as writer:
        for img in images:
            writer.append_data(img)
    
    print(f"Saved video to {output_path}")


def main() -> None:
    """Main entry point for inspection."""
    args = _parse_args()
    
    # Check if dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        output_dir = Path("data") / f"inspect_{args.name}_{timestamp}"
    else:
        output_dir = Path("data") / f"inspect_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Loading dataset: {dataset_path}")
    
    # Load episode data
    try:
        episode_data = load_episode_data(str(dataset_path), args.episode)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    images = episode_data["images"]
    num_frames = len(images)
    H, W = images.shape[1], images.shape[2]
    
    print(f"Episode {args.episode}:")
    print(f"  - Number of frames: {num_frames}")
    print(f"  - Image size: {W}x{H}")
    print(f"  - Success: {episode_data['success']}")
    
    # Save single frame as PNG
    frame_idx = min(args.frame, num_frames - 1)
    png_path = output_dir / f"frame_{frame_idx}.png"
    save_frame_as_png(images, frame_idx, png_path)
    
    # Save frame data as JSON
    json_path = output_dir / f"frame_{frame_idx}_data.json"
    save_frame_data_as_json(episode_data, frame_idx, json_path)
    
    # Create episode video
    video_path = output_dir / f"episode_{args.episode}_video.mp4"
    create_episode_video(images, video_path, args.fps)
    
    print(f"\nInspection complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()

