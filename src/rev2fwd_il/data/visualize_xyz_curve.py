"""Visualize XYZ curves for debugging training and evaluation.

This module provides functions to visualize:
1. Input EE pose XYZ (raw and normalized)
2. Output action XYZ (raw and normalized, with ground truth during training)
3. Camera images (table camera and optionally wrist camera)

The visualizations are saved as video files with XYZ curves and camera images side by side.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np


class XYZCurveVisualizer:
    """Accumulates XYZ data over an episode and generates visualization videos.
    
    This class collects:
    - Input EE pose XYZ (raw and normalized)
    - Output action XYZ (raw and normalized)
    - Ground truth action XYZ (for training)
    - Camera images (table camera and optionally wrist camera)
    
    And generates a video showing curves and camera images side by side.
    """
    
    def __init__(
        self,
        output_dir: str | Path,
        episode_id: int = 0,
        fps: int = 20,
        figsize: Tuple[int, int] = (20, 10),
    ):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save the visualization video.
            episode_id: Episode identifier for naming the output file.
            fps: Frames per second for the output video.
            figsize: Figure size (width, height) in inches.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.fps = fps
        self.figsize = figsize
        
        # Data storage
        self.ee_pose_raw: List[np.ndarray] = []  # (3,) each
        self.ee_pose_norm: List[np.ndarray] = []  # (3,) each
        self.action_raw: List[np.ndarray] = []  # (3,) each - output after unnormalization
        self.action_norm: List[np.ndarray] = []  # (3,) each - output before unnormalization
        self.action_gt_norm: List[np.ndarray] = []  # (3,) each - normalized ground truth (training only)
        self.action_gt_raw: List[np.ndarray] = []  # (3,) each - raw ground truth (training only)
        
        # Camera images storage
        self.table_images: List[np.ndarray] = []  # (H, W, 3) uint8 each
        self.wrist_images: List[np.ndarray] = []  # (H, W, 3) uint8 each (optional)
        
        self._has_gt = False
        self._has_wrist = False
    
    def add_frame(
        self,
        ee_pose_raw: np.ndarray,
        ee_pose_norm: np.ndarray,
        action_raw: np.ndarray,
        action_norm: np.ndarray,
        action_gt: Optional[np.ndarray] = None,
        action_gt_raw: Optional[np.ndarray] = None,
        table_image: Optional[np.ndarray] = None,
        wrist_image: Optional[np.ndarray] = None,
    ) -> None:
        """Add a single frame of data.
        
        Args:
            ee_pose_raw: Raw EE pose XYZ (3,).
            ee_pose_norm: Normalized EE pose XYZ (3,).
            action_raw: Raw action XYZ after unnormalization (3,).
            action_norm: Normalized action XYZ before unnormalization (3,).
            action_gt: Normalized ground truth action XYZ (3,), for normalized plot.
            action_gt_raw: Raw ground truth action XYZ (3,), for unnormalized plot.
            table_image: Table camera RGB image (H, W, 3) uint8.
            wrist_image: Wrist camera RGB image (H, W, 3) uint8 (optional).
        """
        self.ee_pose_raw.append(np.asarray(ee_pose_raw).flatten()[:3])
        self.ee_pose_norm.append(np.asarray(ee_pose_norm).flatten()[:3])
        self.action_raw.append(np.asarray(action_raw).flatten()[:3])
        self.action_norm.append(np.asarray(action_norm).flatten()[:3])
        
        if action_gt is not None:
            self.action_gt_norm.append(np.asarray(action_gt).flatten()[:3])
            self._has_gt = True
        
        if action_gt_raw is not None:
            self.action_gt_raw.append(np.asarray(action_gt_raw).flatten()[:3])
            self._has_gt = True
        
        # Store camera images
        if table_image is not None:
            self.table_images.append(np.asarray(table_image).astype(np.uint8))
        if wrist_image is not None:
            self.wrist_images.append(np.asarray(wrist_image).astype(np.uint8))
            self._has_wrist = True
    
    def generate_video(self, filename_prefix: str = "xyz_curves") -> str:
        """Generate a video showing the XYZ curves and camera images over time.
        
        The video layout:
        - Left side (2/3 width): 4 XYZ curve subplots (2x2 grid)
        - Right side (1/3 width): Camera images (table camera on top, wrist camera below if available)
        
        XYZ Curve subplots:
        1. Input EE Pose XYZ (raw)
        2. Input EE Pose XYZ (normalized)
        3. Output Action XYZ (normalized, with GT if available)
        4. Output Action XYZ (raw/unnormalized, with GT if available)
        
        Args:
            filename_prefix: Prefix for the output filename.
            
        Returns:
            Path to the saved video file.
        """
        if len(self.ee_pose_raw) == 0:
            print("[XYZCurveVisualizer] No data to visualize.")
            return ""
        
        # Convert lists to arrays
        ee_raw = np.array(self.ee_pose_raw)  # (T, 3)
        ee_norm = np.array(self.ee_pose_norm)  # (T, 3)
        act_raw = np.array(self.action_raw)  # (T, 3)
        act_norm = np.array(self.action_norm)  # (T, 3)
        
        # GT arrays for normalized and raw plots
        act_gt_norm = np.array(self.action_gt_norm) if len(self.action_gt_norm) > 0 else None  # (T, 3) or None
        act_gt_raw = np.array(self.action_gt_raw) if len(self.action_gt_raw) > 0 else None  # (T, 3) or None
        
        T = len(ee_raw)
        timesteps = np.arange(T)
        
        # Check if we have camera images
        has_camera_images = len(self.table_images) > 0
        
        # Output path
        video_path = self.output_dir / f"{filename_prefix}_ep{self.episode_id}.mp4"
        
        # Try to use imageio for video writing with compatible settings
        try:
            import imageio
            # Use ffmpeg backend with H.264 codec and yuv420p pixel format for maximum compatibility
            writer = imageio.get_writer(
                str(video_path), 
                fps=self.fps, 
                codec='libx264',
                output_params=['-pix_fmt', 'yuv420p'],  # Required for QuickTime/browser compatibility
            )
        except ImportError:
            print("[XYZCurveVisualizer] imageio not available, saving static plot instead.")
            return self._save_static_plot(ee_raw, ee_norm, act_raw, act_norm, act_gt_norm, act_gt_raw, filename_prefix)
        
        print(f"[XYZCurveVisualizer] Generating video with {T} frames...")
        if has_camera_images:
            print(f"  - Including camera images: table={len(self.table_images)}, wrist={len(self.wrist_images)}")
        
        colors = {'x': 'r', 'y': 'g', 'z': 'b'}
        labels = ['X', 'Y', 'Z']
        
        for t in range(T):
            # Create figure with gridspec for flexible layout
            if has_camera_images:
                # Layout: left 2/3 for curves (2x2), right 1/3 for cameras (2x1)
                fig = plt.figure(figsize=(self.figsize[0] + 6, self.figsize[1]))
                gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
                axes = np.array([
                    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
                ])
                ax_table = fig.add_subplot(gs[0, 2])
                ax_wrist = fig.add_subplot(gs[1, 2])
            else:
                fig, axes = plt.subplots(2, 2, figsize=self.figsize)
            
            fig.suptitle(f"XYZ Curves - Episode {self.episode_id} - Frame {t}/{T}", fontsize=14)
            
            # Subplot 1: Input EE Pose XYZ (raw)
            ax = axes[0, 0]
            ax.set_title("Input: EE Pose XYZ (Raw)")
            for i, (label, color) in enumerate(zip(labels, colors.values())):
                ax.plot(timesteps[:t+1], ee_raw[:t+1, i], color=color, label=label, linewidth=1.5)
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Position (m)")
            ax.legend(loc='upper right')
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.3)
            
            # Subplot 2: Input EE Pose XYZ (normalized)
            ax = axes[0, 1]
            ax.set_title("Input: EE Pose XYZ (Normalized)")
            for i, (label, color) in enumerate(zip(labels, colors.values())):
                ax.plot(timesteps[:t+1], ee_norm[:t+1, i], color=color, label=label, linewidth=1.5)
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.legend(loc='upper right')
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.3)
            
            # Subplot 3: Output Action XYZ (normalized)
            ax = axes[1, 0]
            has_gt_norm = act_gt_norm is not None and len(act_gt_norm) > t
            ax.set_title("Output: Action XYZ (Normalized)" + (" + GT" if has_gt_norm else ""))
            for i, (label, color) in enumerate(zip(labels, colors.values())):
                ax.plot(timesteps[:t+1], act_norm[:t+1, i], color=color, label=f"{label} pred", linewidth=1.5)
                if has_gt_norm:
                    ax.plot(timesteps[:t+1], act_gt_norm[:t+1, i], color=color, label=f"{label} GT", 
                           linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.legend(loc='upper right', ncol=2 if has_gt_norm else 1, fontsize=8)
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.3)
            
            # Subplot 4: Output Action XYZ (raw/unnormalized)
            ax = axes[1, 1]
            has_gt_raw = act_gt_raw is not None and len(act_gt_raw) > t
            ax.set_title("Output: Action XYZ (Unnormalized)" + (" + GT" if has_gt_raw else ""))
            for i, (label, color) in enumerate(zip(labels, colors.values())):
                ax.plot(timesteps[:t+1], act_raw[:t+1, i], color=color, label=f"{label} pred", linewidth=1.5)
                if has_gt_raw:
                    ax.plot(timesteps[:t+1], act_gt_raw[:t+1, i], color=color, label=f"{label} GT", 
                           linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Position (m)")
            ax.legend(loc='upper right', ncol=2 if has_gt_raw else 1, fontsize=8)
            ax.set_xlim(0, T)
            ax.grid(True, alpha=0.3)
            
            # Display camera images if available
            if has_camera_images:
                # Table camera image (top right)
                ax_table.set_title("Table Camera", fontsize=10)
                if t < len(self.table_images) and self.table_images[t] is not None:
                    ax_table.imshow(self.table_images[t])
                ax_table.axis('off')
                
                # Wrist camera image (bottom right)
                ax_wrist.set_title("Wrist Camera", fontsize=10)
                if self._has_wrist and t < len(self.wrist_images) and self.wrist_images[t] is not None:
                    ax_wrist.imshow(self.wrist_images[t])
                else:
                    ax_wrist.text(0.5, 0.5, "No wrist camera", ha='center', va='center', 
                                 fontsize=12, color='gray', transform=ax_wrist.transAxes)
                ax_wrist.axis('off')
            
            plt.tight_layout()
            
            # Convert figure to image (compatible with newer matplotlib versions)
            fig.canvas.draw()
            # Use buffer_rgba() instead of deprecated tostring_rgb()
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            # Convert RGBA to RGB
            img = img[:, :, :3]
            
            writer.append_data(img)
            plt.close(fig)
        
        writer.close()
        print(f"[XYZCurveVisualizer] Saved video to: {video_path}")
        return str(video_path)
    
    def _save_static_plot(
        self,
        ee_raw: np.ndarray,
        ee_norm: np.ndarray,
        act_raw: np.ndarray,
        act_norm: np.ndarray,
        act_gt_norm: Optional[np.ndarray],
        act_gt_raw: Optional[np.ndarray],
        filename_prefix: str,
    ) -> str:
        """Save a static plot showing all curves (fallback when imageio not available)."""
        T = len(ee_raw)
        timesteps = np.arange(T)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f"XYZ Curves - Episode {self.episode_id} (Full)", fontsize=14)
        
        colors = {'x': 'r', 'y': 'g', 'z': 'b'}
        labels = ['X', 'Y', 'Z']
        
        # Subplot 1: Input EE Pose XYZ (raw)
        ax = axes[0, 0]
        ax.set_title("Input: EE Pose XYZ (Raw)")
        for i, (label, color) in enumerate(zip(labels, colors.values())):
            ax.plot(timesteps, ee_raw[:, i], color=color, label=label, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position (m)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Input EE Pose XYZ (normalized)
        ax = axes[0, 1]
        ax.set_title("Input: EE Pose XYZ (Normalized)")
        for i, (label, color) in enumerate(zip(labels, colors.values())):
            ax.plot(timesteps, ee_norm[:, i], color=color, label=label, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: Output Action XYZ (normalized)
        ax = axes[1, 0]
        ax.set_title("Output: Action XYZ (Normalized)" + (" + GT" if act_gt_norm is not None else ""))
        for i, (label, color) in enumerate(zip(labels, colors.values())):
            ax.plot(timesteps, act_norm[:, i], color=color, label=f"{label} pred", linewidth=1.5)
            if act_gt_norm is not None:
                ax.plot(timesteps, act_gt_norm[:, i], color=color, label=f"{label} GT", 
                       linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc='upper right', ncol=2 if act_gt_norm is not None else 1, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Subplot 4: Output Action XYZ (raw/unnormalized)
        ax = axes[1, 1]
        ax.set_title("Output: Action XYZ (Unnormalized)" + (" + GT" if act_gt_raw is not None else ""))
        for i, (label, color) in enumerate(zip(labels, colors.values())):
            ax.plot(timesteps, act_raw[:, i], color=color, label=f"{label} pred", linewidth=1.5)
            if act_gt_raw is not None:
                ax.plot(timesteps, act_gt_raw[:, i], color=color, label=f"{label} GT", 
                       linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position (m)")
        ax.legend(loc='upper right', ncol=2 if act_gt_raw is not None else 1, fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"{filename_prefix}_ep{self.episode_id}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[XYZCurveVisualizer] Saved static plot to: {plot_path}")
        return str(plot_path)
    
    def reset(self, episode_id: Optional[int] = None) -> None:
        """Reset the visualizer for a new episode.
        
        Args:
            episode_id: New episode ID. If None, increments current ID.
        """
        self.ee_pose_raw = []
        self.ee_pose_norm = []
        self.action_raw = []
        self.action_norm = []
        self.action_gt_norm = []
        self.action_gt_raw = []
        self.table_images = []
        self.wrist_images = []
        self._has_gt = False
        self._has_wrist = False
        
        if episode_id is not None:
            self.episode_id = episode_id
        else:
            self.episode_id += 1


def create_training_xyz_visualizer(
    output_dir: str | Path,
    batch_idx: int = 0,
    fps: int = 20,
) -> XYZCurveVisualizer:
    """Create a visualizer for training data.
    
    Args:
        output_dir: Directory to save visualizations.
        batch_idx: Batch index for naming.
        fps: Frames per second for video.
        
    Returns:
        XYZCurveVisualizer instance.
    """
    return XYZCurveVisualizer(
        output_dir=output_dir,
        episode_id=batch_idx,
        fps=fps,
    )


def create_eval_xyz_visualizer(
    output_dir: str | Path,
    episode_id: int = 0,
    fps: int = 20,
) -> XYZCurveVisualizer:
    """Create a visualizer for evaluation data.
    
    Args:
        output_dir: Directory to save visualizations.
        episode_id: Episode index for naming.
        fps: Frames per second for video.
        
    Returns:
        XYZCurveVisualizer instance.
    """
    return XYZCurveVisualizer(
        output_dir=output_dir,
        episode_id=episode_id,
        fps=fps,
    )
