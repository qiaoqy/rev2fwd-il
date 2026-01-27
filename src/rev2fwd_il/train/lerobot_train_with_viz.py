#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modified for rev2fwd-il project to add XYZ curve visualization during training.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Custom LeRobot training with XYZ curve visualization.

This module provides a modified training function that:
1. Saves checkpoints every N steps (configurable, default 200)
2. Generates XYZ curve visualization videos during training
3. Maintains full compatibility with LeRobot's training API

The visualization shows:
- Input: EE pose XYZ (raw and normalized)
- Output: Action XYZ (normalized and unnormalized) with ground truth
- Camera images (table and wrist if available)
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

from rev2fwd_il.data.visualize_xyz_curve import XYZCurveVisualizer
from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def extract_xyz_visualization_data(
    raw_batch: dict[str, torch.Tensor],
    processed_batch: dict[str, torch.Tensor],
    preprocessor,
    postprocessor,
    policy: PreTrainedPolicy,
    accelerator: Accelerator,
    dataset_stats: dict = None,
    sample_idx: int = 0,
) -> dict:
    """Extract data for XYZ curve visualization from a training batch.
    
    Extracts the specified sample from the batch and computes:
    - Raw EE pose XYZ (before normalization)
    - Normalized EE pose XYZ (after normalization)
    - Normalized action XYZ (ground truth)
    - Raw action XYZ (from raw batch)
    - Ground truth action XYZ
    - Camera images
    
    Args:
        raw_batch: Raw batch from dataloader (before preprocessing)
        processed_batch: Preprocessed batch from dataloader (already normalized)
        preprocessor: The preprocessor used for normalization
        postprocessor: The postprocessor used for unnormalization
        policy: The policy model (for getting action predictions)
        accelerator: The accelerator instance
        dataset_stats: Dataset statistics for manual unnormalization
        sample_idx: Index of the sample to extract from the batch (default: 0)
        
    Returns:
        Dictionary containing visualization data for the specified sample
    """
    viz_data = {}
    
    # Get batch size and clamp sample_idx to valid range
    batch_size = processed_batch["observation.state"].shape[0] if "observation.state" in processed_batch else 1
    sample_idx = sample_idx % batch_size  # Wrap around if needed
    
    # Extract observation.state (already normalized by preprocessor)
    # Take the last observation step for visualization
    if "observation.state" in processed_batch:
        state = processed_batch["observation.state"]
        if state.dim() == 3:  # (B, n_obs_steps, state_dim)
            state_norm = state[0, -1].detach().cpu().numpy()  # Last obs step, first batch
        else:  # (B, state_dim)
            state_norm = state[0].detach().cpu().numpy()
        viz_data["ee_pose_norm"] = state_norm[:3].copy()  # XYZ only
        
        # Unnormalize state using dataset stats if available
        if dataset_stats is not None and "observation.state" in dataset_stats:
            stats = dataset_stats["observation.state"]
            if "mean" in stats and "std" in stats:
                mean = np.array(stats["mean"])[:3]
                std = np.array(stats["std"])[:3]
                viz_data["ee_pose_raw"] = state_norm[:3] * std + mean
            elif "min" in stats and "max" in stats:
                min_val = np.array(stats["min"])[:3]
                max_val = np.array(stats["max"])[:3]
                # Unnormalize from [-1, 1] to [min, max]
                viz_data["ee_pose_raw"] = (state_norm[:3] + 1) / 2 * (max_val - min_val) + min_val
            else:
                viz_data["ee_pose_raw"] = state_norm[:3].copy()
        else:
            viz_data["ee_pose_raw"] = state_norm[:3].copy()
    
    # Extract camera images
    # observation.image: (B, n_obs_steps, C, H, W) normalized with ImageNet mean/std
    if "observation.image" in processed_batch:
        img = processed_batch["observation.image"]
        if img.dim() == 5:  # (B, n_obs_steps, C, H, W)
            img_np = img[sample_idx, -1].detach().cpu().numpy()  # Last obs step
        else:  # (B, C, H, W)
            img_np = img[sample_idx].detach().cpu().numpy()
        
        # IMPORTANT: Reverse ImageNet normalization before visualization
        # Images are normalized with: (img - mean) / std
        # We need to reverse: img = normalized * std + mean
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = img_np * imagenet_std + imagenet_mean  # Reverse normalization
        
        # Convert from CHW to HWC and [0,1] to [0,255]
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["table_image"] = img_np
    
    # Extract wrist camera if available
    if "observation.wrist_image" in processed_batch:
        wrist_img = processed_batch["observation.wrist_image"]
        if wrist_img.dim() == 5:
            wrist_img_np = wrist_img[sample_idx, -1].detach().cpu().numpy()
        else:
            wrist_img_np = wrist_img[sample_idx].detach().cpu().numpy()
        
        # IMPORTANT: Reverse ImageNet normalization before visualization
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        wrist_img_np = wrist_img_np * imagenet_std + imagenet_mean  # Reverse normalization
        
        wrist_img_np = np.transpose(wrist_img_np, (1, 2, 0))
        wrist_img_np = (wrist_img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["wrist_image"] = wrist_img_np
    
    # Extract ground truth action (raw from raw_batch, norm from processed_batch)
    if "action" in raw_batch:
        action_raw = raw_batch["action"]
        if action_raw.dim() == 3:  # (B, horizon, action_dim)
            # Take first action in horizon for the specified sample
            action_gt_raw = action_raw[sample_idx, 0].detach().cpu().numpy()
        else:  # (B, action_dim)
            action_gt_raw = action_raw[sample_idx].detach().cpu().numpy()
        viz_data["action_gt_raw"] = action_gt_raw[:3].copy()  # XYZ only
    
    if "action" in processed_batch:
        action_norm = processed_batch["action"]
        if action_norm.dim() == 3:  # (B, horizon, action_dim)
            # Take first action in horizon for the specified sample
            action_gt_norm = action_norm[sample_idx, 0].detach().cpu().numpy()
        else:  # (B, action_dim)
            action_gt_norm = action_norm[sample_idx].detach().cpu().numpy()
        viz_data["action_gt_norm"] = action_gt_norm[:3].copy()  # XYZ only
    
    # For training visualization, we use ground truth as the "predicted" action
    # since we can't easily get the diffusion model's prediction during training
    viz_data["action_pred_norm"] = viz_data.get("action_gt_norm", np.zeros(3))
    viz_data["action_pred_raw"] = viz_data.get("action_gt_raw", np.zeros(3))
    
    return viz_data


def extract_action_chunk_data(
    raw_batch: dict[str, torch.Tensor],
    processed_batch: dict[str, torch.Tensor],
    policy: PreTrainedPolicy,
    accelerator: Accelerator,
    postprocessor = None,
    dataset_stats: dict = None,
    sample_idx: int = 0,
) -> dict:
    """Extract data for action chunk visualization from a training batch.
    
    Gets the model's predicted action chunk (full horizon) and input observations.
    Also extracts GT action chunk for comparison visualization.
    
    Args:
        raw_batch: Raw batch from dataloader (before preprocessing)
        processed_batch: Preprocessed batch from dataloader (already normalized)
        policy: The policy model (for getting action predictions)
        accelerator: The accelerator instance
        postprocessor: LeRobot postprocessor for action unnormalization (preferred)
        dataset_stats: Dataset statistics for manual unnormalization (fallback)
        sample_idx: Index of the sample to extract from the batch (default: 0)
        
    Returns:
        Dictionary containing action chunk visualization data for the specified sample,
        including GT action chunk for comparison.
    """
    viz_data = {}
    
    # Get batch size and clamp sample_idx to valid range
    batch_size = processed_batch["observation.state"].shape[0] if "observation.state" in processed_batch else 1
    sample_idx = sample_idx % batch_size  # Wrap around if needed
    
    # Extract observation.state (already normalized by preprocessor)
    # Take the last observation step for visualization
    if "observation.state" in processed_batch:
        state = processed_batch["observation.state"]
        if state.dim() == 3:  # (B, n_obs_steps, state_dim)
            state_norm = state[sample_idx, -1].detach().cpu().numpy()  # Last obs step, specified sample
        else:  # (B, state_dim)
            state_norm = state[sample_idx].detach().cpu().numpy()
        viz_data["ee_pose_norm"] = state_norm[:3].copy()  # XYZ only
        
        # Unnormalize state using dataset stats if available
        if dataset_stats is not None and "observation.state" in dataset_stats:
            stats = dataset_stats["observation.state"]
            if "mean" in stats and "std" in stats:
                mean = np.array(stats["mean"])[:3]
                std = np.array(stats["std"])[:3]
                viz_data["ee_pose_raw"] = state_norm[:3] * std + mean
            elif "min" in stats and "max" in stats:
                min_val = np.array(stats["min"])[:3]
                max_val = np.array(stats["max"])[:3]
                # Unnormalize from [-1, 1] to [min, max]
                viz_data["ee_pose_raw"] = (state_norm[:3] + 1) / 2 * (max_val - min_val) + min_val
            else:
                viz_data["ee_pose_raw"] = state_norm[:3].copy()
        else:
            viz_data["ee_pose_raw"] = state_norm[:3].copy()
    
    # Extract camera images
    if "observation.image" in processed_batch:
        img = processed_batch["observation.image"]
        if img.dim() == 5:  # (B, n_obs_steps, C, H, W)
            img_np = img[sample_idx, -1].detach().cpu().numpy()  # Last obs step
        else:  # (B, C, H, W)
            img_np = img[sample_idx].detach().cpu().numpy()
        
        # IMPORTANT: Reverse ImageNet normalization before visualization
        # Images are normalized with: (img - mean) / std
        # We need to reverse: img = normalized * std + mean
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_np = img_np * imagenet_std + imagenet_mean  # Reverse normalization
        
        # Convert from CHW to HWC and [0,1] to [0,255]
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["table_image"] = img_np
    
    # Extract wrist camera if available
    if "observation.wrist_image" in processed_batch:
        wrist_img = processed_batch["observation.wrist_image"]
        if wrist_img.dim() == 5:
            wrist_img_np = wrist_img[sample_idx, -1].detach().cpu().numpy()
        else:
            wrist_img_np = wrist_img[sample_idx].detach().cpu().numpy()
        
        # IMPORTANT: Reverse ImageNet normalization before visualization
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        wrist_img_np = wrist_img_np * imagenet_std + imagenet_mean  # Reverse normalization
        
        wrist_img_np = np.transpose(wrist_img_np, (1, 2, 0))
        wrist_img_np = (wrist_img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["wrist_image"] = wrist_img_np
    
    # Helper function to unnormalize action chunk using postprocessor
    def unnormalize_action_chunk(action_chunk_norm: np.ndarray) -> np.ndarray:
        """Unnormalize action chunk using LeRobot postprocessor.
        
        The postprocessor is designed for single actions (action_dim,) or batched 
        actions (batch, action_dim). For action chunks (horizon, action_dim), we 
        process each action row individually to maintain compatibility.
        
        Args:
            action_chunk_norm: Normalized action chunk (horizon, action_dim)
            
        Returns:
            Unnormalized action chunk XYZ (horizon, 3)
        """
        if postprocessor is not None:
            try:
                # Convert to tensor
                action_tensor = torch.from_numpy(action_chunk_norm).float()  # (horizon, action_dim)
                
                # LeRobot's postprocessor expects single actions (action_dim,) 
                # Process each action in the chunk individually
                horizon = action_tensor.shape[0]
                unnorm_actions = []
                for i in range(horizon):
                    single_action = action_tensor[i]  # (action_dim,)
                    unnorm_action = postprocessor(single_action)  # Returns torch.Tensor (action_dim,)
                    unnorm_actions.append(unnorm_action)
                
                # Stack back to (horizon, action_dim)
                unnorm_chunk = torch.stack(unnorm_actions, dim=0)
                
                # Extract XYZ only (first 3 dimensions)
                return unnorm_chunk[:, :3].cpu().numpy()  # (horizon, 3)
                
            except Exception as e:
                logging.warning(f"Postprocessor failed: {e}, returning normalized action chunk")
                return action_chunk_norm[:, :3].copy()
        
        # No postprocessor available - return normalized values with warning
        logging.warning("No postprocessor available, returning normalized action chunk")
        return action_chunk_norm[:, :3].copy()
    
    # Extract GT action chunk from processed_batch (normalized) and raw_batch (raw)
    if "action" in processed_batch:
        action_gt = processed_batch["action"]
        if action_gt.dim() == 3:  # (B, horizon, action_dim)
            gt_chunk_norm = action_gt[sample_idx].detach().cpu().numpy()  # (horizon, action_dim)
        else:  # (B, action_dim)
            gt_chunk_norm = action_gt[sample_idx].detach().cpu().numpy()[np.newaxis, :]  # (1, action_dim)
        viz_data["gt_chunk_norm"] = gt_chunk_norm[:, :3].copy()  # (horizon, 3)
        # Note: gt_chunk_raw will be overwritten by raw_batch below if available
    
    # Also get raw GT from raw_batch for verification
    if "action" in raw_batch:
        action_raw = raw_batch["action"]
        if action_raw.dim() == 3:  # (B, horizon, action_dim)
            gt_chunk_raw_direct = action_raw[sample_idx].detach().cpu().numpy()  # (horizon, action_dim)
        else:  # (B, action_dim)
            gt_chunk_raw_direct = action_raw[sample_idx].detach().cpu().numpy()[np.newaxis, :]
        # Use raw_batch directly as the ground truth raw values
        viz_data["gt_chunk_raw"] = gt_chunk_raw_direct[:, :3].copy()  # (horizon, 3)
    
    # Get model's predicted action chunk
    # Run inference to get the full action chunk prediction
    # Need to unwrap policy from DDP wrapper to call generate_actions
    unwrapped_policy = accelerator.unwrap_model(policy)
    unwrapped_policy.eval()
    with torch.no_grad():
        # For diffusion policy, call diffusion.generate_actions directly
        # This expects the batch format from training: (B, n_obs_steps, ...)
        try:
            # Create a single-sample batch for inference using the specified sample_idx
            # Only include tensor values (skip float/int metadata like 'reward', 'done', etc.)
            inference_batch = {
                k: v[sample_idx:sample_idx+1] for k, v in processed_batch.items() 
                if isinstance(v, torch.Tensor)
            }
            
            # For diffusion policy, we need to:
            # 1. Stack images into observation.images key
            # 2. Call diffusion.generate_actions directly
            
            # Check if this is a diffusion policy with image features
            if hasattr(unwrapped_policy, 'diffusion') and hasattr(unwrapped_policy, 'config'):
                config = unwrapped_policy.config
                
                # Stack image features into the expected format
                if hasattr(config, 'image_features') and config.image_features:
                    # image_features contains keys like ['observation.image', 'observation.wrist_image']
                    images_list = []
                    for key in config.image_features:
                        if key in inference_batch:
                            images_list.append(inference_batch[key])
                    if images_list:
                        # Stack along camera dimension: (B, n_obs_steps, num_cams, C, H, W)
                        inference_batch['observation.images'] = torch.stack(images_list, dim=-4)
                
                # Call generate_actions directly
                # Returns (B, n_action_steps, action_dim)
                action_chunk = unwrapped_policy.diffusion.generate_actions(inference_batch)
            else:
                # Fallback to select_action for other policy types
                action_chunk = unwrapped_policy.select_action(inference_batch)
            
            if action_chunk.dim() == 3:  # (B, n_action_steps, action_dim)
                action_chunk_norm = action_chunk[0].detach().cpu().numpy()  # (n_action_steps, action_dim)
            elif action_chunk.dim() == 2:  # (n_action_steps, action_dim)
                action_chunk_norm = action_chunk.detach().cpu().numpy()
            else:  # (action_dim,) - single action
                action_chunk_norm = action_chunk.detach().cpu().numpy()[np.newaxis, :]  # (1, action_dim)
            
            # Extract XYZ only (first 3 dimensions)
            viz_data["action_chunk_norm"] = action_chunk_norm[:, :3].copy()  # (n_action_steps, 3)
            
            # Unnormalize action chunk using the postprocessor (pass full action for correct unnorm)
            viz_data["action_chunk_raw"] = unnormalize_action_chunk(action_chunk_norm)
                
        except Exception as e:
            import traceback
            logging.warning(f"Failed to get action chunk prediction: {e}")
            logging.warning(f"Traceback: {traceback.format_exc()}")
            # Fallback: use ground truth action as placeholder
            if "action" in processed_batch:
                action = processed_batch["action"]
                if action.dim() == 3:  # (B, horizon, action_dim)
                    action_chunk_norm = action[sample_idx].detach().cpu().numpy()
                else:
                    action_chunk_norm = action[sample_idx].detach().cpu().numpy()[np.newaxis, :]
                viz_data["action_chunk_norm"] = action_chunk_norm[:, :3].copy()
                viz_data["action_chunk_raw"] = unnormalize_action_chunk(action_chunk_norm)
    
    unwrapped_policy.train()
    
    return viz_data


def compute_validation_loss(
    policy: PreTrainedPolicy,
    val_dataloader,
    preprocessor,
    accelerator: Accelerator,
    max_batches: int = 50,
) -> float:
    """
    Compute validation loss over the validation dataset.
    
    Args:
        policy: The policy model.
        val_dataloader: DataLoader for validation data.
        preprocessor: The preprocessor for normalizing batches.
        accelerator: The accelerator instance.
        max_batches: Maximum number of batches to evaluate (to limit time).
        
    Returns:
        Average validation loss.
    """
    policy.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break
            
            processed_batch = preprocessor(raw_batch)
            
            with accelerator.autocast():
                loss, _ = policy.forward(processed_batch)
            
            # Gather loss across processes
            gathered_loss = accelerator.gather(loss)
            total_loss += gathered_loss.mean().item()
            num_batches += 1
    
    policy.train()
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_with_xyz_visualization(
    cfg: TrainPipelineConfig,
    accelerator: Accelerator | None = None,
    viz_save_freq: int = 200,
    xyz_viz_dir: str | Path | None = None,
    val_dataset_cfg=None,
    val_freq: int = 500,
    state_slice_end: int | None = None,
):
    """
    Train a policy with XYZ curve visualization.
    
    This is a modified version of LeRobot's train() function that adds:
    1. Checkpoint saving every viz_save_freq steps (default 200)
    2. XYZ curve visualization for debugging
    3. Validation loss computation and logging
    4. State dimension slicing for training with subset of state features
    
    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
        viz_save_freq: Frequency (in steps) to save checkpoints and generate visualization.
        xyz_viz_dir: Directory to save XYZ visualization videos. If None, uses output_dir/xyz_viz.
        val_dataset_cfg: Optional DatasetConfig for validation dataset.
        val_freq: Frequency (in steps) to compute validation loss.
        state_slice_end: If set, slice observation.state to [:state_slice_end].
                         Useful for training with subset of state without re-converting data.
    """
    cfg.validate()

    # Create Accelerator if not provided
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)

    # =========================================================================
    # State slicing: modify dataset stats and add transform if needed
    # =========================================================================
    if state_slice_end is not None:
        if is_main_process:
            logging.info(f"Applying state slicing: observation.state[:, :, :{state_slice_end}]")
        
        # Create a transform that slices the state
        def state_slice_transform(batch):
            """Slice observation.state to the first state_slice_end dimensions."""
            if "observation.state" in batch:
                # batch["observation.state"] shape: (batch, n_obs_steps, state_dim)
                batch["observation.state"] = batch["observation.state"][..., :state_slice_end]
            return batch
        
        # Store original transform if any (use getattr for compatibility with different LeRobot versions)
        original_transform = getattr(dataset, 'transform', None)
        
        # Compose transforms
        if original_transform is not None:
            def combined_transform(batch):
                batch = original_transform(batch)
                return state_slice_transform(batch)
            dataset.transform = combined_transform
        else:
            dataset.transform = state_slice_transform
        
        # Update dataset stats to reflect sliced state
        if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'stats'):
            stats = dataset.meta.stats
            if "observation.state" in stats:
                old_stats = stats["observation.state"]
                new_stats = {}
                for key, value in old_stats.items():
                    if isinstance(value, torch.Tensor):
                        new_stats[key] = value[..., :state_slice_end]
                    elif isinstance(value, np.ndarray):
                        new_stats[key] = value[..., :state_slice_end]
                    else:
                        new_stats[key] = value
                stats["observation.state"] = new_stats
                if is_main_process:
                    logging.info(f"  Updated observation.state stats shape: {new_stats.get('mean', new_stats.get('min', 'N/A'))}")
        
        # Update features in metadata
        if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'features'):
            if "observation.state" in dataset.meta.features:
                old_shape = dataset.meta.features["observation.state"].get("shape", None)
                if old_shape is not None:
                    new_shape = (state_slice_end,)
                    dataset.meta.features["observation.state"]["shape"] = new_shape
                    if is_main_process:
                        logging.info(f"  Updated observation.state feature shape: {old_shape} -> {new_shape}")

    # Create environment for evaluation if configured
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    accelerator.wait_for_everyone()

    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info(f"XYZ visualization save frequency: {viz_save_freq} steps")

    # Create dataloader - let accelerator handle distributed sampling
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,  # Drop last incomplete batch
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    # Create validation dataset and dataloader if configured
    val_dataloader = None
    if val_dataset_cfg is not None:
        if is_main_process:
            logging.info("Creating validation dataset")
        
        # Create a temporary config for validation dataset
        from copy import deepcopy
        val_cfg = deepcopy(cfg)
        val_cfg.dataset = val_dataset_cfg
        
        val_dataset = make_dataset(val_cfg)
        
        # Apply state slicing transform to validation dataset if needed
        if state_slice_end is not None:
            def val_state_slice_transform(batch):
                """Slice observation.state to the first state_slice_end dimensions."""
                if "observation.state" in batch:
                    batch["observation.state"] = batch["observation.state"][..., :state_slice_end]
                return batch
            
            # Use getattr for compatibility with different LeRobot versions
            original_val_transform = getattr(val_dataset, 'transform', None)
            if original_val_transform is not None:
                def combined_val_transform(batch):
                    batch = original_val_transform(batch)
                    return val_state_slice_transform(batch)
                val_dataset.transform = combined_val_transform
            else:
                val_dataset.transform = val_state_slice_transform
            
            if is_main_process:
                logging.info(f"Applied state slicing to validation dataset: [:, :, :{state_slice_end}]")
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=min(cfg.num_workers, 2),  # Use fewer workers for validation
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
        
        # Prepare validation dataloader with accelerator
        val_dataloader = accelerator.prepare(val_dataloader)
        
        if is_main_process:
            logging.info(f"Validation dataset: {val_dataset.num_frames} frames, {val_dataset.num_episodes} episodes")
            logging.info(f"Validation frequency: every {val_freq} steps")
        
        accelerator.wait_for_everyone()

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # Setup XYZ visualization directory
    if xyz_viz_dir is None:
        xyz_viz_dir = Path(cfg.output_dir) / "xyz_viz"
    else:
        xyz_viz_dir = Path(xyz_viz_dir)
    
    if is_main_process:
        xyz_viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"XYZ visualization will be saved to: {xyz_viz_dir}")

    # XYZ visualizer for accumulating data
    xyz_visualizer = None
    action_chunk_visualizer = None
    if is_main_process:
        xyz_visualizer = XYZCurveVisualizer(
            output_dir=xyz_viz_dir,
            episode_id=step,
            fps=20,
        )
        # Action chunk visualizer for showing model predictions
        action_chunk_visualizer = ActionChunkVisualizer(
            output_dir=xyz_viz_dir,
            step_id=step,
            fps=20,
        )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
    
    # DEBUG: Print dataloader info
    if is_main_process:
        logging.info(f"[DEBUG] Training dataloader: batch_size={cfg.batch_size}, len(dataset)={len(dataset)}")
        if val_dataloader is not None:
            logging.info(f"[DEBUG] Validation dataloader ready")
    
    # Synchronize before starting training loop
    accelerator.wait_for_everyone()
    if is_main_process:
        logging.info("[DEBUG] All processes synchronized, starting training loop...")
    
    # Get dataset stats for unnormalization in visualization
    dataset_stats = dataset.meta.stats if hasattr(dataset.meta, 'stats') else None
    
    # Counter for visualization sample index - cycles through batch to show different samples
    viz_sample_counter = 0

    for _ in range(step, cfg.steps):
        # if is_main_process and _ == step:
        #     logging.info(f"[DEBUG] Entering first iteration (step={step})")
        
        start_time = time.perf_counter()
        
        # if is_main_process and _ == step:
        #     logging.info("[DEBUG] Calling next(dl_iter)...")
        
        raw_batch = next(dl_iter)
        
        # if is_main_process and _ == step:
        #     logging.info(f"[DEBUG] Got batch, keys={list(raw_batch.keys())}")
        
        processed_batch = preprocessor(raw_batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        # Get batch size for cycling sample index
        batch_size = processed_batch["observation.state"].shape[0] if "observation.state" in processed_batch else 1
        viz_sample_idx = viz_sample_counter % batch_size

        # Extract visualization data before policy update (on main process only)
        # Limit to 1000 frames per visualization interval to avoid memory issues and slow video generation
        if is_main_process and xyz_visualizer is not None:
            if len(xyz_visualizer.ee_pose_raw) < 1000:  # Limit to 1000 frames for XYZ viz
                try:
                    viz_data = extract_xyz_visualization_data(
                        raw_batch=raw_batch,
                        processed_batch=processed_batch,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        policy=policy,
                        accelerator=accelerator,
                        dataset_stats=dataset_stats,
                        sample_idx=viz_sample_idx,  # Use cycling sample index
                    )
                    
                    # Add frame to visualizer
                    xyz_visualizer.add_frame(
                        ee_pose_raw=viz_data.get("ee_pose_raw", np.zeros(3)),
                        ee_pose_norm=viz_data.get("ee_pose_norm", np.zeros(3)),
                        action_raw=viz_data.get("action_gt_raw", np.zeros(3)),
                        action_norm=viz_data.get("action_gt_norm", np.zeros(3)),
                        action_gt=viz_data.get("action_gt_norm", None),  # Normalized GT for subplot 3
                        action_gt_raw=viz_data.get("action_gt_raw", None),  # Raw GT for subplot 4
                        table_image=viz_data.get("table_image", None),
                        wrist_image=viz_data.get("wrist_image", None),
                    )
                except Exception as e:
                    logging.debug(f"Failed to extract visualization data: {e}")
        
        # Extract action chunk visualization data (after policy update to get predictions)
        # We'll do this after the update_policy call below

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            processed_batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        
        # Extract action chunk visualization data after policy update (on main process only)
        # Limit to 200 frames per visualization interval to avoid memory issues
        if is_main_process and action_chunk_visualizer is not None:
            if len(action_chunk_visualizer.frames) < 200:  # Limit to 200 frames
                try:
                    chunk_data = extract_action_chunk_data(
                        raw_batch=raw_batch,
                        processed_batch=processed_batch,
                        policy=policy,
                        accelerator=accelerator,
                        postprocessor=postprocessor,
                        dataset_stats=dataset_stats,
                        sample_idx=viz_sample_idx,  # Use cycling sample index
                    )
                    
                    # Add frame to action chunk visualizer
                    action_chunk_visualizer.add_frame(
                        ee_pose_raw=chunk_data.get("ee_pose_raw", np.zeros(3)),
                        ee_pose_norm=chunk_data.get("ee_pose_norm", np.zeros(3)),
                        action_chunk_norm=chunk_data.get("action_chunk_norm", np.zeros((16, 3))),
                        action_chunk_raw=chunk_data.get("action_chunk_raw", None),
                        gt_chunk_norm=chunk_data.get("gt_chunk_norm", None),
                        gt_chunk_raw=chunk_data.get("gt_chunk_raw", None),
                        table_image=chunk_data.get("table_image", None),
                        wrist_image=chunk_data.get("wrist_image", None),
                    )
                except Exception as e:
                    logging.warning(f"Failed to extract action chunk data: {e}")
        
        # Increment visualization sample counter to cycle through different samples
        viz_sample_counter += 1

        step += 1
        train_tracker.step()
        
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_viz_saving_step = step % viz_save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = val_dataloader is not None and val_freq > 0 and step % val_freq == 0

        # Compute validation loss
        if is_val_step:
            val_loss = compute_validation_loss(
                policy=policy,
                val_dataloader=val_dataloader,
                preprocessor=preprocessor,
                accelerator=accelerator,
                max_batches=50,  # Limit to 50 batches for speed
            )
            if is_main_process:
                logging.info(f"Step {step}: val_loss={val_loss:.4f}")
                if wandb_logger:
                    # Note: log_dict doesn't support mode="val", just use key prefix
                    wandb_logger.log_dict({"val_loss": val_loss}, step)
                # Also save to log file
                log_file = Path(cfg.output_dir) / "training_log.txt"
                with open(log_file, "a") as f:
                    f.write(f"Step {step}: val_loss={val_loss:.4f}\n")

        if is_log_step:
            logging.info(train_tracker)
            # 保存loss等信息到文件
            log_file = Path(cfg.output_dir) / "training_log.txt"
            with open(log_file, "a") as f:
                f.write(f"Step {step}: {train_tracker}\n")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # Save checkpoint at viz_save_freq intervals
        if cfg.save_checkpoint and is_viz_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            # IMPORTANT: Synchronize BEFORE generating videos to avoid NCCL timeout
            # Video generation can take a long time with many frames
            accelerator.wait_for_everyone()
            
            # Generate visualization videos AFTER synchronization (only on main process)
            # This way other processes don't have to wait for video generation
            if is_main_process:
                # Generate XYZ visualization video
                if xyz_visualizer is not None and len(xyz_visualizer.ee_pose_raw) > 0:
                    num_frames = len(xyz_visualizer.ee_pose_raw)
                    # Warn if too many frames (will take a long time)
                    if num_frames > 10000:
                        logging.warning(f"XYZ visualizer has {num_frames} frames, video generation may take a while...")
                    try:
                        video_path = xyz_visualizer.generate_video(
                            filename_prefix=f"train_xyz_step_{step}"
                        )
                        logging.info(f"Saved XYZ visualization: {video_path}")
                    except Exception as e:
                        logging.warning(f"Failed to generate XYZ visualization: {e}")
                    
                    # Reset visualizer for next interval
                    xyz_visualizer.reset(episode_id=step)
                
                # Generate action chunk visualization video
                if action_chunk_visualizer is not None and len(action_chunk_visualizer.frames) > 0:
                    try:
                        video_path = action_chunk_visualizer.generate_video(
                            filename_prefix=f"train_action_chunk"
                        )
                        logging.info(f"Saved action chunk visualization: {video_path}")
                    except Exception as e:
                        logging.warning(f"Failed to generate action chunk visualization: {e}")
                    
                    # Reset visualizer for next interval
                    action_chunk_visualizer.reset(step_id=step)

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    """CLI entry point - parse config and run training."""
    from lerobot.configs import parser
    
    @parser.wrap()
    def _train(cfg: TrainPipelineConfig):
        train_with_xyz_visualization(cfg, viz_save_freq=200)
    
    _train()


if __name__ == "__main__":
    main()
