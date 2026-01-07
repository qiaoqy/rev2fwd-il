#!/usr/bin/env python3
"""Test script for action chunk visualizer."""

import numpy as np
from pathlib import Path
from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer

def main():
    """Test the action chunk visualizer with synthetic data."""
    output_dir = Path("runs/test_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizer
    viz = ActionChunkVisualizer(
        output_dir=output_dir,
        step_id=0,
        fps=10,
    )
    
    # Generate synthetic data for 50 frames
    horizon = 16
    num_frames = 50
    
    print(f"Generating {num_frames} frames with horizon={horizon}...")
    
    for i in range(num_frames):
        # Synthetic EE pose
        ee_pose_raw = np.array([0.5 + 0.1 * np.sin(i * 0.1), 
                                0.3 + 0.1 * np.cos(i * 0.1), 
                                0.2])
        ee_pose_norm = (ee_pose_raw - 0.3) / 0.2  # Simple normalization
        
        # Synthetic action chunk (horizon steps)
        t = np.linspace(0, 1, horizon)
        action_chunk_norm = np.zeros((horizon, 3))
        action_chunk_norm[:, 0] = np.sin(2 * np.pi * t + i * 0.1)  # X
        action_chunk_norm[:, 1] = np.cos(2 * np.pi * t + i * 0.1)  # Y
        action_chunk_norm[:, 2] = 0.5 * np.sin(4 * np.pi * t + i * 0.1)  # Z
        
        # Unnormalize
        action_chunk_raw = action_chunk_norm * 0.2 + 0.3
        
        # Synthetic camera images (random colored noise)
        table_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        wrist_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Add frame
        viz.add_frame(
            ee_pose_raw=ee_pose_raw,
            ee_pose_norm=ee_pose_norm,
            action_chunk_norm=action_chunk_norm,
            action_chunk_raw=action_chunk_raw,
            table_image=table_image,
            wrist_image=wrist_image,
        )
    
    # Generate video
    print("Generating video...")
    video_path = viz.generate_video(filename_prefix="test_action_chunk")
    print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()
