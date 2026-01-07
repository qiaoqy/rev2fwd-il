# Action Chunk Visualization During Training

## Overview

This document describes the new action chunk visualization feature added to the training pipeline. This visualization helps debug and understand the diffusion policy's predictions during training.

## What's New

Every 200 training steps (configurable via `--viz_save_freq`), the training script now generates **two** visualization videos:

1. **XYZ Curve Video** (existing): Shows the temporal evolution of EE pose and actions over accumulated training samples
2. **Action Chunk Video** (NEW): Shows the model's predicted action chunks for individual training samples

## Action Chunk Visualization Details

### What It Shows

For each training sample (up to 200 samples per interval), the visualization displays:

**Input (Static per frame):**
- EE pose XYZ (raw values in meters)
- EE pose XYZ (normalized values)
- Table camera image
- Wrist camera image (if available)

**Output (Model Prediction):**
- Predicted action chunk XYZ curves (normalized)
  - X-axis: Chunk step (0 to horizon-1)
  - Y-axis: Normalized XYZ values
  - Shows the full action sequence the model predicts
- Predicted action chunk XYZ curves (raw/unnormalized)
  - X-axis: Chunk step (0 to horizon-1)
  - Y-axis: Position in meters
  - Shows the actual target positions

### Video Layout

```
┌─────────────────────────────────────────────────────────┐
│  Action Chunk Prediction - Step XXX - Sample YYY/200   │
├──────────────────────────┬──────────────────────────────┤
│  Input: EE Pose (Raw)    │  Input: EE Pose (Normalized)│
│  [Bar chart: X, Y, Z]    │  [Bar chart: X, Y, Z]       │
├──────────────────────────┼──────────────────────────────┤
│  Output: Action Chunk    │  Output: Action Chunk       │
│  (Normalized)            │  (Raw)                      │
│  [Line plot: horizon     │  [Line plot: horizon        │
│   steps on x-axis]       │   steps on x-axis]          │
├──────────────────────────┼──────────────────────────────┤
│  Table Camera Image      │  Wrist Camera Image         │
│  [RGB image]             │  [RGB image]                │
└──────────────────────────┴──────────────────────────────┘
```

### Key Features

1. **Per-Sample Visualization**: Each frame shows one training sample's input and the model's predicted action chunk
2. **Full Horizon Display**: Shows the complete action sequence (e.g., 16 steps) predicted by the diffusion policy
3. **Normalized & Raw Values**: Displays both normalized (model output) and unnormalized (actual positions) predictions
4. **Camera Context**: Includes visual context from table and wrist cameras
5. **Limited Frame Count**: Collects up to 200 samples per interval to avoid memory issues

## Files Modified/Created

### New Files
- `src/rev2fwd_il/data/visualize_action_chunk.py`: Action chunk visualizer class
- `scripts/test_action_chunk_viz.py`: Test script for the visualizer
- `docs/action_chunk_visualization.md`: This documentation

### Modified Files
- `src/rev2fwd_il/train/lerobot_train_with_viz.py`:
  - Added `ActionChunkVisualizer` import
  - Added `extract_action_chunk_data()` function to get model predictions
  - Integrated action chunk visualization into training loop
  - Generates action chunk video every `viz_save_freq` steps

## Usage

The action chunk visualization is automatically enabled when using `--enable_xyz_viz` flag:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/31_train_A_diffusion.py \
    --dataset data/A_forward_with_2images.npz \
    --out runs/diffusion_A_2cam_3 \
    --num_episodes 500 \
    --batch_size 2048 \
    --steps 800 \
    --lr 0.0005 \
    --enable_xyz_viz \
    --viz_save_freq 200 \
    --include_obj_pose \
    --wandb
```

### Output Files

Videos are saved to `{output_dir}/xyz_viz/`:
- `train_xyz_step_{step}.mp4`: XYZ curve video (existing)
- `train_action_chunk_step{step}.mp4`: Action chunk video (NEW)

Example:
```
runs/diffusion_A_2cam_3/xyz_viz/
├── train_xyz_step_200.mp4
├── train_action_chunk_step200.mp4
├── train_xyz_step_400.mp4
├── train_action_chunk_step400.mp4
└── ...
```

## Implementation Details

### Data Extraction

The `extract_action_chunk_data()` function:
1. Extracts input observations (EE pose, camera images) from the batch
2. Runs model inference using `policy.select_action()` to get predicted action chunks
3. Unnormalizes predictions using dataset statistics
4. Returns data dictionary for visualization

### Visualization Generation

The `ActionChunkVisualizer` class:
1. Accumulates up to 200 training samples per interval
2. For each sample, creates a matplotlib figure with:
   - Bar charts for input EE pose (raw and normalized)
   - Line plots for predicted action chunks (normalized and raw)
   - Camera images (table and wrist)
3. Converts figures to video frames using imageio
4. Saves as MP4 with H.264 codec for compatibility

### Performance Considerations

- **Frame Limit**: Only collects 200 samples per interval to avoid memory issues
- **Inference Mode**: Temporarily switches policy to eval mode for prediction
- **Main Process Only**: Visualization only runs on the main process in distributed training
- **Error Handling**: Gracefully handles failures without stopping training

## Debugging Tips

1. **Check Video Files**: Videos should be ~4-10 MB for 200 frames
2. **Verify Predictions**: Action chunks should show smooth trajectories
3. **Compare Normalized vs Raw**: Ensure unnormalization is working correctly
4. **Camera Images**: Should match the training data distribution

## Future Improvements

Potential enhancements:
- Add ground truth action chunks for comparison
- Show prediction uncertainty (for ensemble models)
- Add temporal consistency metrics across chunks
- Support for multi-modal predictions
- Interactive HTML visualization instead of video
