#!/usr/bin/env python3
"""Step 2: Inspect and visualize collected trajectory data.

This script provides utilities to inspect the data collected by script 1:
1. Extract a single frame and save as PNG image + JSON metadata
2. Compile an episode's image sequence into an MP4 video
3. Generate XYZ curve visualization video (optional)

The inspection data is saved in a timestamped folder under data/ for easy management.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic inspection (episode 0, frame 0)
python scripts/2_inspect_data.py --dataset data/B_2images_goal.npz

# Specific episode and frame
python scripts/2_inspect_data.py --dataset data/B_pick_place.npz \
    --episode 5 --frame 50

# With XYZ curve visualization
python scripts/2_inspect_data.py --dataset data/B_pick_place.npz \
    --enable_xyz_viz --episode 0

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add imports for XYZ visualization
from rev2fwd_il.data.visualize_xyz_curve import XYZCurveVisualizer


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
        default=30,
        help="Frame index to extract as PNG (default: 30).",
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
    parser.add_argument(
        "--enable_xyz_viz",
        action="store_true",
        help="Enable XYZ curve visualization video generation.",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default="runs/diffusion_A_2cam_3/lerobot_dataset/meta/stats.json",
        help="Path to stats.json file for normalization parameters.",
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


def load_normalization_stats(stats_path: str) -> dict:
    """Load normalization statistics from LeRobot dataset stats.json.
    
    Args:
        stats_path: Path to the stats.json file.
        
    Returns:
        Dictionary containing normalization stats for observation.state.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if 'observation.state' not in stats:
        raise ValueError(f"observation.state not found in stats file: {stats_path}")
    
    state_stats = stats['observation.state']
    
    # Extract mean and std for the first 3 dimensions (XYZ position)
    # Note: ee_pose is stored as [x, y, z, qw, qx, qy, qz] in the dataset
    mean = np.array(state_stats['mean'][:3])  # XYZ means
    std = np.array(state_stats['std'][:3])   # XYZ stds
    
    return {
        'mean': mean,
        'std': std,
        'min': np.array(state_stats.get('min', [0, 0, 0])[:3]),
        'max': np.array(state_stats.get('max', [1, 1, 1])[:3]),
    }


def normalize_ee_pose_xyz(ee_pose_xyz: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize EE pose XYZ using the same method as training.
    
    Args:
        ee_pose_xyz: Array of shape (T, 3) containing XYZ positions.
        stats: Normalization statistics from load_normalization_stats.
        
    Returns:
        Normalized XYZ array of shape (T, 3).
    """
    # Use mean and std normalization like in LeRobot
    mean = stats['mean']
    std = stats['std']
    
    # Normalize: (x - mean) / std
    normalized = (ee_pose_xyz - mean) / std
    return normalized


def create_xyz_visualization_video(
    episode_data: dict,
    output_dir: Path,
    episode_idx: int,
    stats: dict,
    fps: int = 30,
) -> None:
    """Create XYZ curve visualization video for the episode.
    
    Args:
        episode_data: Episode data dictionary.
        output_dir: Directory to save the video.
        episode_idx: Episode index.
        stats: Normalization statistics.
        fps: Frames per second for the video.
    """
    # Extract EE pose data (T, 7) -> XYZ is first 3 dimensions
    ee_poses = episode_data['ee_pose']  # (T, 7)
    ee_xyz_raw = ee_poses[:, :3]  # (T, 3) - raw XYZ
    
    # Normalize XYZ
    ee_xyz_norm = normalize_ee_pose_xyz(ee_xyz_raw, stats)  # (T, 3) - normalized XYZ
    
    # Extract camera images
    table_images = episode_data['images']  # (T, H, W, 3)
    wrist_images = episode_data.get('wrist_images', None)  # (T, H, W, 3) or None
    
    # Create visualizer
    visualizer = XYZCurveVisualizer(
        output_dir=output_dir,
        episode_id=episode_idx,
        fps=fps,
    )
    
    # Add frames one by one
    T = len(ee_xyz_raw)
    for t in range(T):
        # Raw EE pose XYZ
        ee_raw_xyz = ee_xyz_raw[t]  # (3,)
        
        # Normalized EE pose XYZ  
        ee_norm_xyz = ee_xyz_norm[t]  # (3,)
        
        # For output, we leave empty (no action data in inspection)
        action_raw = np.zeros(3)  # Placeholder
        action_norm = np.zeros(3)  # Placeholder
        
        # Camera images
        table_img = table_images[t]  # (H, W, 3)
        wrist_img = wrist_images[t] if wrist_images is not None else None  # (H, W, 3) or None
        
        # Add frame to visualizer
        visualizer.add_frame(
            ee_pose_raw=ee_raw_xyz,
            ee_pose_norm=ee_norm_xyz,
            action_raw=action_raw,
            action_norm=action_norm,
            action_gt=None,  # No ground truth in inspection
            action_gt_raw=None,  # No ground truth in inspection
            table_image=table_img,
            wrist_image=wrist_img,
        )
    
    # Generate video
    video_path = visualizer.generate_video(filename_prefix=f"episode_{episode_idx}_xyz")
    print(f"Created XYZ visualization video: {video_path}")


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


def create_episode_video(
    images: np.ndarray,
    output_path: Path,
    fps: int,
    wrist_images: np.ndarray | None = None,
) -> None:
    """Create MP4 video from episode images.
    
    Uses H.264 codec (libx264) with yuv420p pixel format for broad compatibility,
    including direct playback in VSCode.
    
    If wrist_images is provided, creates a side-by-side video with table camera
    on the left and wrist camera on the right.
    
    Args:
        images: Array of shape (T, H, W, C) containing RGB images from table camera.
        output_path: Path to save the MP4 file.
        fps: Frames per second for the video.
        wrist_images: Optional array of shape (T, H, W, C) from wrist camera.
                      If provided, will be concatenated side-by-side with images.
    """
    import imageio
    
    if wrist_images is not None:
        print(f"Creating side-by-side video with {len(images)} frames at {fps} fps...")
        print(f"  - Left: Table camera ({images.shape[2]}x{images.shape[1]})")
        print(f"  - Right: Wrist camera ({wrist_images.shape[2]}x{wrist_images.shape[1]})")
        
        # Concatenate images side-by-side (along width axis)
        # Both should have shape (T, H, W, C), concatenate along axis=2 (width)
        combined_images = np.concatenate([images, wrist_images], axis=2)
    else:
        print(f"Creating video with {len(images)} frames at {fps} fps...")
        combined_images = images
    
    # Use libx264 codec with yuv420p for broad compatibility (including VSCode)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    ) as writer:
        for img in combined_images:
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
    
    # Check if wrist camera images are available
    wrist_images = episode_data.get("wrist_images", None)
    has_wrist = wrist_images is not None
    
    print(f"Episode {args.episode}:")
    print(f"  - Number of frames: {num_frames}")
    print(f"  - Table camera image size: {W}x{H}")
    if has_wrist:
        wH, wW = wrist_images.shape[1], wrist_images.shape[2]
        print(f"  - Wrist camera image size: {wW}x{wH}")
    else:
        print(f"  - Wrist camera: not available")
    print(f"  - Success: {episode_data['success']}")
    
    # Save single frame as PNG (table camera)
    frame_idx = min(args.frame, num_frames - 1)
    png_path = output_dir / f"frame_{frame_idx}_table.png"
    save_frame_as_png(images, frame_idx, png_path)
    
    # Save wrist camera frame if available
    if has_wrist:
        wrist_png_path = output_dir / f"frame_{frame_idx}_wrist.png"
        save_frame_as_png(wrist_images, frame_idx, wrist_png_path)
    
    # Save frame data as JSON
    json_path = output_dir / f"frame_{frame_idx}_data.json"
    save_frame_data_as_json(episode_data, frame_idx, json_path)
    
    # Create episode video (side-by-side if wrist camera available)
    video_path = output_dir / f"episode_{args.episode}_video.mp4"
    create_episode_video(images, video_path, args.fps, wrist_images=wrist_images)
    
    # Create XYZ visualization video if enabled
    if args.enable_xyz_viz:
        print("\nGenerating XYZ curve visualization...")
        try:
            # Load normalization stats
            stats_path = Path(args.stats_json)
            if not stats_path.exists():
                print(f"Warning: Stats file not found at {stats_path}, skipping XYZ visualization")
            else:
                stats = load_normalization_stats(str(stats_path))
                create_xyz_visualization_video(
                    episode_data=episode_data,
                    output_dir=output_dir,
                    episode_idx=args.episode,
                    stats=stats,
                    fps=args.fps,
                )
        except Exception as e:
            print(f"Error creating XYZ visualization: {e}")
    
    print(f"\nInspection complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()

