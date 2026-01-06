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
    batch: dict[str, torch.Tensor],
    preprocessor,
    postprocessor,
    policy: PreTrainedPolicy,
    accelerator: Accelerator,
    dataset_stats: dict = None,
) -> dict:
    """Extract data for XYZ curve visualization from a training batch.
    
    Extracts the first sample from the batch and computes:
    - Raw EE pose XYZ (before normalization)
    - Normalized EE pose XYZ (after normalization)
    - Normalized action XYZ (ground truth)
    - Raw action XYZ (after unnormalization)
    - Ground truth action XYZ
    - Camera images
    
    Args:
        batch: Preprocessed batch from dataloader (already normalized)
        preprocessor: The preprocessor used for normalization
        postprocessor: The postprocessor used for unnormalization
        policy: The policy model (for getting action predictions)
        accelerator: The accelerator instance
        dataset_stats: Dataset statistics for manual unnormalization
        
    Returns:
        Dictionary containing visualization data for first sample
    """
    viz_data = {}
    
    # Extract observation.state (already normalized by preprocessor)
    # Take the last observation step for visualization
    if "observation.state" in batch:
        state = batch["observation.state"]
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
    # observation.image: (B, n_obs_steps, C, H, W) normalized to [0, 1]
    if "observation.image" in batch:
        img = batch["observation.image"]
        if img.dim() == 5:  # (B, n_obs_steps, C, H, W)
            img_np = img[0, -1].detach().cpu().numpy()  # Last obs step
        else:  # (B, C, H, W)
            img_np = img[0].detach().cpu().numpy()
        # Convert from CHW to HWC and [0,1] to [0,255]
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["table_image"] = img_np
    
    # Extract wrist camera if available
    if "observation.wrist_image" in batch:
        wrist_img = batch["observation.wrist_image"]
        if wrist_img.dim() == 5:
            wrist_img_np = wrist_img[0, -1].detach().cpu().numpy()
        else:
            wrist_img_np = wrist_img[0].detach().cpu().numpy()
        wrist_img_np = np.transpose(wrist_img_np, (1, 2, 0))
        wrist_img_np = (wrist_img_np * 255).clip(0, 255).astype(np.uint8)
        viz_data["wrist_image"] = wrist_img_np
    
    # Extract ground truth action (normalized)
    # action: (B, horizon, action_dim) or (B, action_dim)
    if "action" in batch:
        action_gt = batch["action"]
        if action_gt.dim() == 3:  # (B, horizon, action_dim)
            # Take first action in horizon for the first sample
            action_gt_norm = action_gt[0, 0].detach().cpu().numpy()
        else:  # (B, action_dim)
            action_gt_norm = action_gt[0].detach().cpu().numpy()
        viz_data["action_gt_norm"] = action_gt_norm[:3].copy()  # XYZ only
        
        # Unnormalize action using dataset stats if available
        if dataset_stats is not None and "action" in dataset_stats:
            stats = dataset_stats["action"]
            if "mean" in stats and "std" in stats:
                mean = np.array(stats["mean"])[:3]
                std = np.array(stats["std"])[:3]
                viz_data["action_gt_raw"] = action_gt_norm[:3] * std + mean
            elif "min" in stats and "max" in stats:
                min_val = np.array(stats["min"])[:3]
                max_val = np.array(stats["max"])[:3]
                # Unnormalize from [-1, 1] to [min, max]
                viz_data["action_gt_raw"] = (action_gt_norm[:3] + 1) / 2 * (max_val - min_val) + min_val
            else:
                viz_data["action_gt_raw"] = action_gt_norm[:3].copy()
        else:
            viz_data["action_gt_raw"] = action_gt_norm[:3].copy()
    
    # For training visualization, we use ground truth as the "predicted" action
    # since we can't easily get the diffusion model's prediction during training
    viz_data["action_pred_norm"] = viz_data.get("action_gt_norm", np.zeros(3))
    viz_data["action_pred_raw"] = viz_data.get("action_gt_raw", np.zeros(3))
    
    return viz_data


def train_with_xyz_visualization(
    cfg: TrainPipelineConfig,
    accelerator: Accelerator | None = None,
    viz_save_freq: int = 200,
    xyz_viz_dir: str | Path | None = None,
):
    """
    Train a policy with XYZ curve visualization.
    
    This is a modified version of LeRobot's train() function that adds:
    1. Checkpoint saving every viz_save_freq steps (default 200)
    2. XYZ curve visualization for debugging
    
    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
        viz_save_freq: Frequency (in steps) to save checkpoints and generate visualization.
        xyz_viz_dir: Directory to save XYZ visualization videos. If None, uses output_dir/xyz_viz.
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

    # Create dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

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
    if is_main_process:
        xyz_visualizer = XYZCurveVisualizer(
            output_dir=xyz_viz_dir,
            episode_id=step,
            fps=20,
        )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
    
    # Get dataset stats for unnormalization in visualization
    dataset_stats = dataset.meta.stats if hasattr(dataset.meta, 'stats') else None

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # Extract visualization data before policy update (on main process only)
        if is_main_process and xyz_visualizer is not None:
            try:
                viz_data = extract_xyz_visualization_data(
                    batch=batch,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    policy=policy,
                    accelerator=accelerator,
                    dataset_stats=dataset_stats,
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

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_viz_saving_step = step % viz_save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
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
                
                # Generate XYZ visualization video
                if xyz_visualizer is not None and len(xyz_visualizer.ee_pose_raw) > 0:
                    try:
                        video_path = xyz_visualizer.generate_video(
                            filename_prefix=f"train_xyz_step_{step}"
                        )
                        logging.info(f"Saved XYZ visualization: {video_path}")
                    except Exception as e:
                        logging.warning(f"Failed to generate XYZ visualization: {e}")
                    
                    # Reset visualizer for next interval
                    xyz_visualizer.reset(episode_id=step)

            accelerator.wait_for_everyone()

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
