#!/usr/bin/env python3
"""Step 2: Inspect and visualize nut threading trajectory data.

This script provides utilities to inspect the data collected by 1_collect_data_nut_thread.py:
1. Extract a single frame and save as PNG image + JSON metadata
2. Compile an episode's image sequence into an MP4 video
3. Plot force sensor data over time
4. Visualize EE trajectory in 3D

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic inspection (episode 0)
CUDA_VISIBLE_DEVICES=2 python scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz

# Specific episode with force plot
CUDA_VISIBLE_DEVICES=2 python scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz \
    --episode 5 --enable_force_plot

# All visualizations
CUDA_VISIBLE_DEVICES=2 python scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz \
    --episode 0 --enable_force_plot --enable_trajectory_plot

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
        description="Inspect nut threading data with force sensing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/nut_thread.npz",
        help="Path to the NPZ file containing collected episodes.",
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
        default=15,
        help="Frames per second for the output video (default: 15).",
    )
    parser.add_argument(
        "--enable_force_plot",
        action="store_true",
        help="Enable force sensor data plot generation.",
    )
    parser.add_argument(
        "--enable_trajectory_plot",
        action="store_true",
        help="Enable 3D trajectory plot generation.",
    )
    
    return parser.parse_args()


def load_episode_data(npz_path: str, episode_idx: int) -> dict:
    """Load data for a specific episode from NPZ file.
    
    Args:
        npz_path: Path to the NPZ file.
        episode_idx: Index of the episode to load.
        
    Returns:
        Dictionary containing episode data.
    """
    data = np.load(npz_path, allow_pickle=True)
    
    if 'episodes' not in data.files:
        raise ValueError(f"Invalid dataset format. Expected 'episodes' key but found: {data.files}")
    
    episodes = data['episodes']
    num_episodes = len(episodes)
    
    if episode_idx >= num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {num_episodes} episodes.")
    
    episode_data = episodes[episode_idx]
    return episode_data


def save_frame_as_png(images: np.ndarray, frame_idx: int, output_path: Path) -> None:
    """Save a single frame as PNG image."""
    import imageio
    
    if frame_idx >= len(images):
        raise ValueError(f"Frame {frame_idx} not found. Episode has {len(images)} frames.")
    
    frame = images[frame_idx]
    imageio.imwrite(str(output_path), frame)
    print(f"Saved frame {frame_idx} to {output_path}")


def save_frame_data_as_json(episode_data: dict, frame_idx: int, output_path: Path) -> None:
    """Save frame metadata as JSON file."""
    if frame_idx >= len(episode_data["obs"]):
        raise ValueError(f"Frame {frame_idx} not found.")
    
    frame_data = {
        "frame_idx": frame_idx,
        "obs": episode_data["obs"][frame_idx].tolist(),
        "ee_pose": episode_data["ee_pose"][frame_idx].tolist(),
        "nut_pose": episode_data["nut_pose"][frame_idx].tolist(),
        "bolt_pose": episode_data["bolt_pose"][frame_idx].tolist(),
        "action": episode_data["action"][frame_idx].tolist(),
        "ft_force": episode_data["ft_force"][frame_idx].tolist(),
        "ft_force_raw": episode_data["ft_force_raw"][frame_idx].tolist(),
        "joint_pos": episode_data["joint_pos"][frame_idx].tolist(),
        "success": bool(episode_data.get("success", False)),
        "episode_length": int(episode_data.get("episode_length", len(episode_data["obs"]))),
    }
    
    with open(output_path, "w") as f:
        json.dump(frame_data, f, indent=2)
    print(f"Saved frame {frame_idx} data to {output_path}")


def create_episode_video(
    images: np.ndarray,
    output_path: Path,
    fps: int,
    wrist_images_dict: dict | None = None,
) -> None:
    """Create MP4 video from episode images.
    
    Args:
        images: Table camera images (T, H, W, 3)
        output_path: Path to save video
        fps: Frames per second
        wrist_images_dict: Dict of {cam_name: images (T, H, W, 3)} for wrist cameras
    """
    import imageio
    
    # Combine all camera views horizontally
    all_views = [images]
    
    if wrist_images_dict:
        for cam_name, wrist_images in wrist_images_dict.items():
            if wrist_images is not None and not np.all(wrist_images == 0):
                all_views.append(wrist_images)
    
    if len(all_views) > 1:
        print(f"Creating multi-view video ({len(all_views)} cameras) with {len(images)} frames at {fps} fps...")
        combined_images = np.concatenate(all_views, axis=2)  # Concatenate horizontally
    else:
        print(f"Creating video with {len(images)} frames at {fps} fps...")
        combined_images = images
    
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


def create_force_plot(episode_data: dict, output_path: Path) -> None:
    """Create force sensor data plot over time."""
    import matplotlib.pyplot as plt
    
    ft_force = episode_data["ft_force"]  # (T, 3)
    ft_force_raw = episode_data["ft_force_raw"]  # (T, 6)
    T = len(ft_force)
    time_steps = np.arange(T)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot force (Fx, Fy, Fz)
    ax1 = axes[0]
    ax1.plot(time_steps, ft_force[:, 0], 'r-', label='Fx', linewidth=1.5)
    ax1.plot(time_steps, ft_force[:, 1], 'g-', label='Fy', linewidth=1.5)
    ax1.plot(time_steps, ft_force[:, 2], 'b-', label='Fz', linewidth=1.5)
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Force Sensor Readings')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot torque (Tx, Ty, Tz) from raw data
    ax2 = axes[1]
    ax2.plot(time_steps, ft_force_raw[:, 3], 'r-', label='Tx', linewidth=1.5)
    ax2.plot(time_steps, ft_force_raw[:, 4], 'g-', label='Ty', linewidth=1.5)
    ax2.plot(time_steps, ft_force_raw[:, 5], 'b-', label='Tz', linewidth=1.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title('Torque Sensor Readings')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved force plot to {output_path}")


def create_trajectory_plot(episode_data: dict, output_path: Path) -> None:
    """Create 3D trajectory plot of EE, nut, and bolt positions."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    ee_pose = episode_data["ee_pose"]  # (T, 7)
    nut_pose = episode_data["nut_pose"]  # (T, 7)
    bolt_pose = episode_data["bolt_pose"]  # (T, 7)
    
    ee_xyz = ee_pose[:, :3]
    nut_xyz = nut_pose[:, :3]
    bolt_xyz = bolt_pose[:, :3]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot EE trajectory
    ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], 'b-', 
            label='End-Effector', linewidth=2, alpha=0.8)
    ax.scatter(ee_xyz[0, 0], ee_xyz[0, 1], ee_xyz[0, 2], 
               c='blue', s=100, marker='o', label='EE Start')
    ax.scatter(ee_xyz[-1, 0], ee_xyz[-1, 1], ee_xyz[-1, 2], 
               c='blue', s=100, marker='s', label='EE End')
    
    # Plot Nut trajectory
    ax.plot(nut_xyz[:, 0], nut_xyz[:, 1], nut_xyz[:, 2], 'r-', 
            label='Nut', linewidth=2, alpha=0.8)
    ax.scatter(nut_xyz[0, 0], nut_xyz[0, 1], nut_xyz[0, 2], 
               c='red', s=100, marker='o')
    ax.scatter(nut_xyz[-1, 0], nut_xyz[-1, 1], nut_xyz[-1, 2], 
               c='red', s=100, marker='s')
    
    # Plot Bolt position (should be relatively stationary)
    bolt_mean = bolt_xyz.mean(axis=0)
    ax.scatter(bolt_mean[0], bolt_mean[1], bolt_mean[2], 
               c='green', s=200, marker='^', label='Bolt (mean)')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory: End-Effector, Nut, and Bolt')
    ax.legend(loc='upper left')
    
    # Set equal aspect ratio
    max_range = np.array([
        ee_xyz[:, 0].max() - ee_xyz[:, 0].min(),
        ee_xyz[:, 1].max() - ee_xyz[:, 1].min(),
        ee_xyz[:, 2].max() - ee_xyz[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (ee_xyz[:, 0].max() + ee_xyz[:, 0].min()) * 0.5
    mid_y = (ee_xyz[:, 1].max() + ee_xyz[:, 1].min()) * 0.5
    mid_z = (ee_xyz[:, 2].max() + ee_xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to {output_path}")


def create_video_with_overlay(
    episode_data: dict,
    output_path: Path,
    fps: int,
) -> None:
    """Create video with force/torque data overlay and phase annotation.
    
    Displays:
    - Left: Camera image with current phase annotation
    - Right top: Force plot (Fx, Fy, Fz) over time
    - Right bottom: Torque plot (Tx, Ty, Tz) over time
    - Phase colored background regions on plots
    """
    import imageio
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    images = episode_data["images"]
    ft_force = episode_data["ft_force"]  # (T, 3)
    ft_force_raw = episode_data["ft_force_raw"]  # (T, 6) - last 3 are torque
    ee_pose = episode_data["ee_pose"]
    T = len(images)
    
    # Get phase data if available
    phase_data = episode_data.get("phase", None)
    phase_names = episode_data.get("phase_names", ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE"])
    
    # Phase colors for background shading
    phase_colors = {
        0: '#FFE4E1',  # APPROACH - light pink/red
        1: '#E6E6FA',  # SEARCH - light lavender
        2: '#FFFACD',  # ENGAGE - light yellow
        3: '#98FB98',  # THREAD - light green
        4: '#D3D3D3',  # DONE - light gray
    }
    
    # Extract torque from raw data
    ft_torque = ft_force_raw[:, 3:6] if ft_force_raw.shape[1] >= 6 else np.zeros((T, 3))
    
    print(f"Creating video with overlay ({T} frames at {fps} fps)...")
    if phase_data is not None:
        print(f"  Phase data available: {len(phase_data)} timesteps")
    else:
        print(f"  Phase data NOT available (will not show phase)")
    
    frames_with_overlay = []
    
    for t in range(T):
        # Create figure with image and data (3 subplots: image, force, torque)
        fig = plt.figure(figsize=(14, 6))
        
        # Layout: [Image (left 40%)] [Force plot (right top 60%)] [Torque plot (right bottom 60%)]
        ax_img = fig.add_axes([0.02, 0.1, 0.38, 0.8])  # Left: camera image
        ax_force = fig.add_axes([0.45, 0.55, 0.52, 0.38])  # Right top: force
        ax_torque = fig.add_axes([0.45, 0.1, 0.52, 0.38])  # Right bottom: torque
        
        # Get current phase
        current_phase = int(phase_data[t]) if phase_data is not None else -1
        current_phase_name = phase_names[current_phase] if 0 <= current_phase < len(phase_names) else "UNKNOWN"
        
        # Left: Camera image with phase annotation
        ax_img.imshow(images[t])
        if phase_data is not None:
            # Add phase label on image with colored background
            phase_color = phase_colors.get(current_phase, '#FFFFFF')
            ax_img.text(0.5, 0.95, f'Phase: {current_phase_name}',
                       transform=ax_img.transAxes, fontsize=14, fontweight='bold',
                       horizontalalignment='center', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=phase_color, 
                                edgecolor='black', alpha=0.9))
        ax_img.set_title(f'Frame {t}/{T}', fontsize=12)
        ax_img.axis('off')
        
        # Add EE position text below image
        ee_pos = ee_pose[t, :3]
        ax_img.text(0.5, 0.02, f'EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})',
                   transform=ax_img.transAxes, fontsize=9, horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Helper function to add phase background shading
        def add_phase_background(ax, phase_data, T, y_min, y_max):
            if phase_data is None:
                return
            # Find phase transitions
            phase_starts = [0]
            phase_values = [int(phase_data[0])]
            for i in range(1, len(phase_data)):
                if phase_data[i] != phase_data[i-1]:
                    phase_starts.append(i)
                    phase_values.append(int(phase_data[i]))
            phase_starts.append(len(phase_data))
            
            # Draw background rectangles for each phase region
            for i in range(len(phase_values)):
                start = phase_starts[i]
                end = phase_starts[i + 1]
                phase_val = phase_values[i]
                color = phase_colors.get(phase_val, '#FFFFFF')
                ax.axvspan(start, end, alpha=0.3, color=color, zorder=0)
        
        # Right top: Force plot
        time_range = np.arange(min(t + 1, T))
        
        # Add phase background shading to force plot
        add_phase_background(ax_force, phase_data, T, ft_force.min() - 0.5, ft_force.max() + 0.5)
        
        ax_force.plot(time_range, ft_force[:t + 1, 0], 'r-', label='Fx', linewidth=1.5)
        ax_force.plot(time_range, ft_force[:t + 1, 1], 'g-', label='Fy', linewidth=1.5)
        ax_force.plot(time_range, ft_force[:t + 1, 2], 'b-', label='Fz', linewidth=1.5)
        ax_force.axvline(x=t, color='k', linestyle='--', alpha=0.7, linewidth=1)
        ax_force.set_xlim(0, T)
        ax_force.set_ylim(ft_force.min() - 0.5, ft_force.max() + 0.5)
        ax_force.set_ylabel('Force (N)', fontsize=10)
        ax_force.set_title('Force Sensor (Fx, Fy, Fz)', fontsize=11)
        ax_force.legend(loc='upper right', fontsize=8)
        ax_force.grid(True, alpha=0.3, zorder=1)
        ax_force.set_xticklabels([])  # Hide x labels since torque plot has them
        
        # Add current force values annotation
        ax_force.text(0.02, 0.95, f'Fz={ft_force[t, 2]:.2f}N',
                     transform=ax_force.transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Right bottom: Torque plot
        add_phase_background(ax_torque, phase_data, T, ft_torque.min() - 0.1, ft_torque.max() + 0.1)
        
        ax_torque.plot(time_range, ft_torque[:t + 1, 0], 'r-', label='Tx', linewidth=1.5)
        ax_torque.plot(time_range, ft_torque[:t + 1, 1], 'g-', label='Ty', linewidth=1.5)
        ax_torque.plot(time_range, ft_torque[:t + 1, 2], 'b-', label='Tz', linewidth=1.5)
        ax_torque.axvline(x=t, color='k', linestyle='--', alpha=0.7, linewidth=1)
        ax_torque.set_xlim(0, T)
        # Handle case where torque data might be all zeros
        torque_min = min(ft_torque.min(), -0.1)
        torque_max = max(ft_torque.max(), 0.1)
        ax_torque.set_ylim(torque_min - 0.1, torque_max + 0.1)
        ax_torque.set_xlabel('Time Step', fontsize=10)
        ax_torque.set_ylabel('Torque (Nm)', fontsize=10)
        ax_torque.set_title('Torque Sensor (Tx, Ty, Tz)', fontsize=11)
        ax_torque.legend(loc='upper right', fontsize=8)
        ax_torque.grid(True, alpha=0.3, zorder=1)
        
        # Add current torque values annotation
        ax_torque.text(0.02, 0.95, f'Tz={ft_torque[t, 2]:.3f}Nm',
                      transform=ax_torque.transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add phase legend at the top if phase data available
        if phase_data is not None:
            legend_text = ' | '.join([f'{name}' for name in phase_names])
            fig.text(0.72, 0.98, f'Phases: {legend_text}', fontsize=8,
                    horizontalalignment='center', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Convert figure to image array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]  # RGB only
        frames_with_overlay.append(frame)
        
        plt.close(fig)
        
        if (t + 1) % 100 == 0:
            print(f"  Processed {t + 1}/{T} frames")
    
    # Write video
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p'
    ) as writer:
        for frame in frames_with_overlay:
            writer.append_data(frame)
    
    print(f"Saved overlay video to {output_path}")


def print_episode_stats(episode_data: dict, episode_idx: int) -> None:
    """Print statistics for the episode."""
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx} Statistics")
    print(f"{'='*60}")
    
    print(f"Episode length: {episode_data.get('episode_length', len(episode_data['obs']))}")
    print(f"Success: {episode_data.get('success', 'N/A')}")
    
    # Image stats
    images = episode_data["images"]
    print(f"\nImages:")
    print(f"  - Shape: {images.shape}")
    print(f"  - Dtype: {images.dtype}")
    
    # EE pose stats
    ee_pose = episode_data["ee_pose"]
    ee_xyz = ee_pose[:, :3]
    print(f"\nEnd-Effector Position (XYZ):")
    print(f"  - Range X: [{ee_xyz[:, 0].min():.4f}, {ee_xyz[:, 0].max():.4f}]")
    print(f"  - Range Y: [{ee_xyz[:, 1].min():.4f}, {ee_xyz[:, 1].max():.4f}]")
    print(f"  - Range Z: [{ee_xyz[:, 2].min():.4f}, {ee_xyz[:, 2].max():.4f}]")
    
    # Force stats
    ft_force = episode_data["ft_force"]
    print(f"\nForce Sensor:")
    print(f"  - Shape: {ft_force.shape}")
    print(f"  - Fx: min={ft_force[:, 0].min():.3f}, max={ft_force[:, 0].max():.3f}, mean={ft_force[:, 0].mean():.3f}")
    print(f"  - Fy: min={ft_force[:, 1].min():.3f}, max={ft_force[:, 1].max():.3f}, mean={ft_force[:, 1].mean():.3f}")
    print(f"  - Fz: min={ft_force[:, 2].min():.3f}, max={ft_force[:, 2].max():.3f}, mean={ft_force[:, 2].mean():.3f}")
    
    # Action stats
    action = episode_data["action"]
    print(f"\nAction:")
    print(f"  - Shape: {action.shape}")
    print(f"  - Position action range: [{action[:, :3].min():.3f}, {action[:, :3].max():.3f}]")
    print(f"  - Rotation action range: [{action[:, 3:6].min():.3f}, {action[:, 3:6].max():.3f}]")
    
    # Phase stats (if available)
    phase_data = episode_data.get("phase", None)
    phase_names = episode_data.get("phase_names", ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE"])
    if phase_data is not None:
        print(f"\nState Machine Phase:")
        print(f"  - Shape: {phase_data.shape}")
        # Count steps in each phase
        for i, name in enumerate(phase_names):
            count = np.sum(phase_data == i)
            if count > 0:
                print(f"  - {name}: {count} steps ({100*count/len(phase_data):.1f}%)")
        # Final phase
        final_phase = int(phase_data[-1])
        final_name = phase_names[final_phase] if final_phase < len(phase_names) else "UNKNOWN"
        print(f"  - Final phase: {final_name}")
    else:
        print(f"\nState Machine Phase: NOT AVAILABLE (data collected without phase tracking)")
    
    # Torque stats
    ft_force_raw = episode_data.get("ft_force_raw", None)
    if ft_force_raw is not None and ft_force_raw.shape[1] >= 6:
        ft_torque = ft_force_raw[:, 3:6]
        print(f"\nTorque Sensor:")
        print(f"  - Tx: min={ft_torque[:, 0].min():.4f}, max={ft_torque[:, 0].max():.4f}, mean={ft_torque[:, 0].mean():.4f}")
        print(f"  - Ty: min={ft_torque[:, 1].min():.4f}, max={ft_torque[:, 1].max():.4f}, mean={ft_torque[:, 1].mean():.4f}")
        print(f"  - Tz: min={ft_torque[:, 2].min():.4f}, max={ft_torque[:, 2].max():.4f}, mean={ft_torque[:, 2].mean():.4f}")
    
    print(f"{'='*60}\n")


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
    dataset_name = dataset_path.stem
    if args.name:
        output_dir = Path("data") / f"inspect_{args.name}_{timestamp}"
    else:
        output_dir = Path("data") / f"inspect_{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Loading dataset: {dataset_path}")
    
    # Load episode data
    try:
        episode_data = load_episode_data(str(dataset_path), args.episode)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Print episode statistics
    print_episode_stats(episode_data, args.episode)
    
    images = episode_data["images"]
    num_frames = len(images)
    H, W = images.shape[1], images.shape[2]
    
    # Check for wrist cameras - support both old format (wrist_images) and new format (wrist_<cam_name>)
    wrist_images_dict = {}
    wrist_cam_names = episode_data.get("wrist_cam_names", None)
    
    # Try new multi-camera format first
    if wrist_cam_names is not None and len(wrist_cam_names) > 0:
        for cam_name in wrist_cam_names:
            key = f"wrist_{cam_name}"
            if key in episode_data:
                wrist_img = episode_data[key]
                if not np.all(wrist_img == 0):
                    wrist_images_dict[cam_name] = wrist_img
    else:
        # Fall back to old single wrist_images format
        wrist_images = episode_data.get("wrist_images", None)
        if wrist_images is not None and not np.all(wrist_images == 0):
            wrist_images_dict["wrist_cam"] = wrist_images
    
    has_wrist = len(wrist_images_dict) > 0
    
    print(f"Episode {args.episode}:")
    print(f"  - Number of frames: {num_frames}")
    print(f"  - Table camera image size: {W}x{H}")
    if has_wrist:
        print(f"  - Wrist cameras available: {list(wrist_images_dict.keys())}")
        for cam_name, wrist_img in wrist_images_dict.items():
            print(f"      {cam_name}: {wrist_img.shape}")
    else:
        print(f"  - Wrist camera: not available / all zeros")
    
    # Save single frame as PNG for table camera
    frame_idx = min(args.frame, num_frames - 1)
    png_path = output_dir / f"frame_{frame_idx}_table.png"
    save_frame_as_png(images, frame_idx, png_path)
    
    # Save single frame for each wrist camera
    if has_wrist:
        for cam_name, wrist_images in wrist_images_dict.items():
            wrist_png_path = output_dir / f"frame_{frame_idx}_{cam_name}.png"
            save_frame_as_png(wrist_images, frame_idx, wrist_png_path)
    
    # Save frame data as JSON
    json_path = output_dir / f"frame_{frame_idx}_data.json"
    save_frame_data_as_json(episode_data, frame_idx, json_path)
    
    # Create basic episode video with all cameras
    video_path = output_dir / f"episode_{args.episode}_video.mp4"
    create_episode_video(images, video_path, args.fps, 
                        wrist_images_dict=wrist_images_dict if has_wrist else None)
    
    # Also create individual videos for each wrist camera for easier comparison
    if has_wrist:
        for cam_name, wrist_images in wrist_images_dict.items():
            wrist_video_path = output_dir / f"episode_{args.episode}_{cam_name}.mp4"
            create_episode_video(wrist_images, wrist_video_path, args.fps)
    
    # Create video with force overlay
    overlay_video_path = output_dir / f"episode_{args.episode}_with_force.mp4"
    create_video_with_overlay(episode_data, overlay_video_path, args.fps)
    
    # Create force plot if enabled
    if args.enable_force_plot:
        force_plot_path = output_dir / f"episode_{args.episode}_force_plot.png"
        create_force_plot(episode_data, force_plot_path)
    
    # Create 3D trajectory plot if enabled
    if args.enable_trajectory_plot:
        trajectory_plot_path = output_dir / f"episode_{args.episode}_trajectory.png"
        create_trajectory_plot(episode_data, trajectory_plot_path)
    
    print(f"\nInspection complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
