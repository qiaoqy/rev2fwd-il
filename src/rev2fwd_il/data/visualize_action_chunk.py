"""Visualize action chunk predictions during training.

This module provides visualization for diffusion policy action chunk predictions:
- Shows input: EE pose XYZ (raw and normalized), camera images
- Shows output: Predicted action chunk XYZ curves (horizon steps)
- Generates a video with one frame per training sample
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np


class ActionChunkVisualizer:
    """Visualizes action chunk predictions during training.
    
    For each training sample, shows:
    - Input: EE pose XYZ (raw and normalized), camera images (static)
    - Output: Predicted action chunk XYZ curves (horizon steps on x-axis)
    
    Accumulates frames and generates a video showing predictions for multiple samples.
    """
    
    def __init__(
        self,
        output_dir: str | Path,
        step_id: int = 0,
        fps: int = 20,
        figsize: Tuple[int, int] = (20, 10),
    ):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save the visualization video.
            step_id: Training step identifier for naming the output file.
            fps: Frames per second for the output video.
            figsize: Figure size (width, height) in inches.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_id = step_id
        self.fps = fps
        self.figsize = figsize
        
        # Data storage - each element is one training sample
        self.frames: List[dict] = []
    
    def add_frame(
        self,
        ee_pose_raw: np.ndarray,
        ee_pose_norm: np.ndarray,
        action_chunk_norm: np.ndarray,
        action_chunk_raw: Optional[np.ndarray] = None,
        gt_chunk_norm: Optional[np.ndarray] = None,
        gt_chunk_raw: Optional[np.ndarray] = None,
        table_image: Optional[np.ndarray] = None,
        wrist_image: Optional[np.ndarray] = None,
    ) -> None:
        """Add a single training sample frame.
        
        Args:
            ee_pose_raw: Raw EE pose XYZ (3,).
            ee_pose_norm: Normalized EE pose XYZ (3,).
            action_chunk_norm: Predicted action chunk XYZ normalized (horizon, 3).
            action_chunk_raw: Predicted action chunk XYZ raw (horizon, 3), optional.
            gt_chunk_norm: Ground truth action chunk XYZ normalized (horizon, 3), optional.
            gt_chunk_raw: Ground truth action chunk XYZ raw (horizon, 3), optional.
            table_image: Table camera RGB image (H, W, 3) uint8.
            wrist_image: Wrist camera RGB image (H, W, 3) uint8 (optional).
        """
        frame_data = {
            "ee_pose_raw": np.asarray(ee_pose_raw).flatten()[:3].copy(),
            "ee_pose_norm": np.asarray(ee_pose_norm).flatten()[:3].copy(),
            "action_chunk_norm": np.asarray(action_chunk_norm).copy(),  # (horizon, 3)
            "action_chunk_raw": np.asarray(action_chunk_raw).copy() if action_chunk_raw is not None else None,
            "gt_chunk_norm": np.asarray(gt_chunk_norm).copy() if gt_chunk_norm is not None else None,
            "gt_chunk_raw": np.asarray(gt_chunk_raw).copy() if gt_chunk_raw is not None else None,
            "table_image": np.asarray(table_image).astype(np.uint8).copy() if table_image is not None else None,
            "wrist_image": np.asarray(wrist_image).astype(np.uint8).copy() if wrist_image is not None else None,
        }
        self.frames.append(frame_data)
    
    def generate_video(self, filename_prefix: str = "action_chunk") -> str:
        """Generate a video showing action chunk predictions for each training sample.
        
        The video layout:
        - Top row: Input EE pose XYZ (raw on left, normalized on right)
        - Middle row: Predicted action chunk XYZ curves (normalized on left, raw on right)
        - Bottom row: Camera images (table camera on left, wrist camera on right if available)
        
        Each frame shows one training sample's input and predicted action chunk.
        
        Args:
            filename_prefix: Prefix for the output filename.
            
        Returns:
            Path to the saved video file.
        """
        if len(self.frames) == 0:
            print("[ActionChunkVisualizer] No data to visualize.")
            return ""
        
        # Output path
        video_path = self.output_dir / f"{filename_prefix}_step{self.step_id}.mp4"
        
        # Try to use imageio for video writing
        try:
            import imageio
            writer = imageio.get_writer(
                str(video_path), 
                fps=self.fps, 
                codec='libx264',
                output_params=['-pix_fmt', 'yuv420p'],
            )
        except ImportError:
            print("[ActionChunkVisualizer] imageio not available, cannot generate video.")
            return ""
        
        num_frames = len(self.frames)
        print(f"[ActionChunkVisualizer] Generating video with {num_frames} frames...")
        
        colors = {'x': 'r', 'y': 'g', 'z': 'b'}
        labels = ['X', 'Y', 'Z']
        
        for frame_idx, frame_data in enumerate(self.frames):
            ee_raw = frame_data["ee_pose_raw"]  # (3,)
            ee_norm = frame_data["ee_pose_norm"]  # (3,)
            chunk_norm = frame_data["action_chunk_norm"]  # (horizon, 3)
            chunk_raw = frame_data["action_chunk_raw"]  # (horizon, 3) or None
            gt_chunk_norm = frame_data["gt_chunk_norm"]  # (horizon, 3) or None
            gt_chunk_raw = frame_data["gt_chunk_raw"]  # (horizon, 3) or None
            table_img = frame_data["table_image"]  # (H, W, 3) or None
            wrist_img = frame_data["wrist_image"]  # (H, W, 3) or None
            
            horizon = len(chunk_norm)
            chunk_steps = np.arange(horizon)
            
            has_camera = table_img is not None
            has_wrist = wrist_img is not None
            has_raw_chunk = chunk_raw is not None
            has_gt_norm = gt_chunk_norm is not None
            has_gt_raw = gt_chunk_raw is not None
            
            # Create figure layout
            if has_camera:
                # 3 rows: EE pose, action chunk, camera images
                fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] + 4))
                gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
                
                # Row 1: EE pose
                ax_ee_raw = fig.add_subplot(gs[0, 0])
                ax_ee_norm = fig.add_subplot(gs[0, 1])
                
                # Row 2: Action chunk
                ax_chunk_norm = fig.add_subplot(gs[1, 0])
                ax_chunk_raw = fig.add_subplot(gs[1, 1])
                
                # Row 3: Camera images
                ax_table = fig.add_subplot(gs[2, 0])
                ax_wrist = fig.add_subplot(gs[2, 1])
            else:
                # 2 rows: EE pose, action chunk
                fig, axes = plt.subplots(2, 2, figsize=self.figsize)
                ax_ee_raw = axes[0, 0]
                ax_ee_norm = axes[0, 1]
                ax_chunk_norm = axes[1, 0]
                ax_chunk_raw = axes[1, 1]
            
            fig.suptitle(f"Action Chunk Prediction - Step {self.step_id} - Sample {frame_idx+1}/{num_frames}", 
                        fontsize=14)
            
            # Plot 1: Input EE Pose XYZ (raw) - bar chart
            ax_ee_raw.set_title("Input: EE Pose XYZ (Raw)")
            x_pos = np.arange(3)
            bars = ax_ee_raw.bar(x_pos, ee_raw, color=[colors['x'], colors['y'], colors['z']], alpha=0.7)
            ax_ee_raw.set_xticks(x_pos)
            ax_ee_raw.set_xticklabels(labels)
            ax_ee_raw.set_ylabel("Position (m)")
            ax_ee_raw.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, ee_raw)):
                height = bar.get_height()
                ax_ee_raw.text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Input EE Pose XYZ (normalized) - bar chart
            ax_ee_norm.set_title("Input: EE Pose XYZ (Normalized)")
            bars = ax_ee_norm.bar(x_pos, ee_norm, color=[colors['x'], colors['y'], colors['z']], alpha=0.7)
            ax_ee_norm.set_xticks(x_pos)
            ax_ee_norm.set_xticklabels(labels)
            ax_ee_norm.set_ylabel("Normalized Value")
            ax_ee_norm.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, ee_norm)):
                height = bar.get_height()
                ax_ee_norm.text(bar.get_x() + bar.get_width()/2., height,
                              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 3: Predicted Action Chunk XYZ (normalized) - line plot with GT comparison
            title_suffix = " + GT" if has_gt_norm else ""
            ax_chunk_norm.set_title(f"Output: Action Chunk XYZ (Normalized, horizon={horizon}){title_suffix}")
            for i, (label, color) in enumerate(zip(labels, colors.values())):
                # Plot predicted action chunk (solid line)
                ax_chunk_norm.plot(chunk_steps, chunk_norm[:, i], color=color, label=f"{label} pred", 
                                  linewidth=2, marker='o', markersize=4)
                # Plot GT action chunk (dashed line) if available
                if has_gt_norm:
                    ax_chunk_norm.plot(chunk_steps, gt_chunk_norm[:, i], color=color, label=f"{label} GT", 
                                      linewidth=2, linestyle='--', alpha=0.7, marker='s', markersize=3)
            ax_chunk_norm.set_xlabel("Chunk Step")
            ax_chunk_norm.set_ylabel("Normalized Value")
            ax_chunk_norm.legend(loc='upper right', fontsize=8, ncol=2)
            ax_chunk_norm.set_xlim(-0.5, horizon - 0.5)
            ax_chunk_norm.grid(True, alpha=0.3)
            
            # Plot 4: Predicted Action Chunk XYZ (raw) - line plot with GT comparison
            if has_raw_chunk:
                title_suffix = " + GT" if has_gt_raw else ""
                ax_chunk_raw.set_title(f"Output: Action Chunk XYZ (Raw, horizon={horizon}){title_suffix}")
                for i, (label, color) in enumerate(zip(labels, colors.values())):
                    # Plot predicted action chunk (solid line)
                    ax_chunk_raw.plot(chunk_steps, chunk_raw[:, i], color=color, label=f"{label} pred", 
                                     linewidth=2, marker='o', markersize=4)
                    # Plot GT action chunk (dashed line) if available
                    if has_gt_raw:
                        ax_chunk_raw.plot(chunk_steps, gt_chunk_raw[:, i], color=color, label=f"{label} GT", 
                                         linewidth=2, linestyle='--', alpha=0.7, marker='s', markersize=3)
                ax_chunk_raw.set_xlabel("Chunk Step")
                ax_chunk_raw.set_ylabel("Position (m)")
                ax_chunk_raw.legend(loc='upper right', fontsize=8, ncol=2)
                ax_chunk_raw.set_xlim(-0.5, horizon - 0.5)
                ax_chunk_raw.grid(True, alpha=0.3)
            else:
                ax_chunk_raw.text(0.5, 0.5, "Raw action chunk\nnot available", 
                                ha='center', va='center', transform=ax_chunk_raw.transAxes,
                                fontsize=12, color='gray')
                ax_chunk_raw.set_xticks([])
                ax_chunk_raw.set_yticks([])
            
            # Display camera images if available
            if has_camera:
                ax_table.imshow(table_img)
                ax_table.set_title("Table Camera")
                ax_table.axis('off')
                
                if has_wrist:
                    ax_wrist.imshow(wrist_img)
                    ax_wrist.set_title("Wrist Camera")
                    ax_wrist.axis('off')
                else:
                    ax_wrist.text(0.5, 0.5, "Wrist camera\nnot available", 
                                ha='center', va='center', transform=ax_wrist.transAxes,
                                fontsize=12, color='gray')
                    ax_wrist.axis('off')
            
            # Convert plot to image
            fig.canvas.draw()
            # Use buffer_rgba() and convert to RGB
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, :3]  # Drop alpha channel
            
            # Write frame to video
            writer.append_data(img)
            
            plt.close(fig)
        
        writer.close()
        
        print(f"[ActionChunkVisualizer] Video saved to: {video_path}")
        return str(video_path)
    
    def reset(self, step_id: int) -> None:
        """Reset the visualizer for a new training interval.
        
        Args:
            step_id: New training step identifier.
        """
        self.step_id = step_id
        self.frames.clear()
