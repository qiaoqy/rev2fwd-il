#!/usr/bin/env python
"""Critic model training script.

Usage:
  # Overfit test (single GPU, 4 episodes)
  CUDA_VISIBLE_DEVICES=1 python debug/train_critic.py \
      --task A --overfit --overfit_episodes 4 --batch_size 4 --num_steps 2000

  # Full training (multi-GPU)
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 debug/train_critic.py \
      --task A --batch_size 64 --num_steps 50000

  # Full training Task B
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 debug/train_critic.py \
      --task B --batch_size 64 --num_steps 50000
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType, PolicyFeature
from rev2fwd_il.models.critic_config import CriticConfig
from rev2fwd_il.models.critic_model import CriticModel


# ============================================================
# Dataset
# ============================================================

class CriticEpisodeDataset(Dataset):
    """Per-frame dataset over NPZ episodes for critic training.

    Each sample is a single frame's (obs_state, obs_images, mc_value).
    Observations use n_obs_steps frames ending at the sample frame.
    No action window is needed (V(s) critic, not Q(s,a)).
    """

    def __init__(
        self,
        npz_path: str,
        n_obs_steps: int = 2,
        include_obj_pose: bool = True,
        include_gripper: bool = True,
        max_episodes: int = -1,
    ):
        self.n_obs_steps = n_obs_steps
        self.include_obj_pose = include_obj_pose
        self.include_gripper = include_gripper

        # Load episodes
        with np.load(npz_path, allow_pickle=True) as data:
            episodes = list(data["episodes"])
        if max_episodes > 0:
            episodes = episodes[:max_episodes]

        # Build index: (episode_idx, frame)
        self.episodes = episodes
        self.samples = []
        for ep_idx, ep in enumerate(episodes):
            T = len(ep["action"])
            for t in range(T):
                self.samples.append((ep_idx, t))

    def __len__(self):
        return len(self.samples)

    def _build_state(self, ep, t):
        """Build state vector: ee_pose (7) + obj_pose (7) + gripper (1) = 15."""
        parts = [ep["ee_pose"][t]]  # (7,)
        if self.include_obj_pose:
            parts.append(ep["obj_pose"][t])  # (7,)
        if self.include_gripper:
            parts.append(np.array([ep["action"][t, 7]], dtype=np.float32))  # (1,)
        return np.concatenate(parts).astype(np.float32)

    def _get_frame_with_padding(self, ep, t):
        """Get frame at time t, clamping to episode bounds (copy-padding)."""
        T = len(ep["action"])
        t_clamped = max(0, min(t, T - 1))
        return t_clamped

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        T = len(ep["action"])

        # ---- Observation: n_obs_steps frames ending at `t` ----
        obs_states = []
        obs_images_list = []
        for i in range(self.n_obs_steps):
            obs_t = self._get_frame_with_padding(ep, t - self.n_obs_steps + 1 + i)
            obs_states.append(self._build_state(ep, obs_t))

            # Stack table + wrist camera images: (2, 3, H, W)
            table_img = ep["images"][obs_t]       # (H, W, 3) uint8
            wrist_img = ep["wrist_images"][obs_t]  # (H, W, 3) uint8
            # HWC -> CHW, normalize to [0, 1]
            table_img = np.transpose(table_img, (2, 0, 1)).astype(np.float32) / 255.0
            wrist_img = np.transpose(wrist_img, (2, 0, 1)).astype(np.float32) / 255.0
            obs_images_list.append(np.stack([table_img, wrist_img], axis=0))  # (2, 3, H, W)

        obs_state = np.stack(obs_states, axis=0)       # (n_obs, state_dim)
        obs_images = np.stack(obs_images_list, axis=0)  # (n_obs, 2, 3, H, W)

        # ---- Scalar MC value target ----
        mc_value = ep["mc_value"][t]

        return {
            "observation.state": torch.from_numpy(obs_state),
            "observation.images": torch.from_numpy(obs_images),
            "mc_value": torch.tensor(mc_value, dtype=torch.float32),
        }


# ============================================================
# Training utilities
# ============================================================

def cosine_schedule(step, total_steps, lr_max, lr_min=0.0):
    """Cosine annealing learning rate."""
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def evaluate(model, dataloader, device, max_batches=50, value_loss_type="mse"):
    """Evaluate model on val set, return mean loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_value = model(batch)  # (B, 1)
            target_value = batch["mc_value"]  # (B,)
            if value_loss_type == "mse":
                loss = F.mse_loss(pred_value.squeeze(-1), target_value)
            elif value_loss_type == "huber":
                loss = F.huber_loss(pred_value.squeeze(-1), target_value)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank):
    return rank == 0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train critic model")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--data_dir", type=str, default="debug/data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: debug/data/critic_{task}_{timestamp})")
    parser.add_argument("--action_model_checkpoint", type=str, default=None,
                        help="Action model checkpoint dir for vision encoder init")

    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Global batch size")
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Model
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--crop_shape", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--value_loss_type", type=str, default="mse", choices=["mse", "huber"])

    # Eval / logging
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=10000)

    # Overfit mode
    parser.add_argument("--overfit", action="store_true", help="Overfit on small subset")
    parser.add_argument("--overfit_episodes", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ---- Distributed setup ----
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ---- Seeding ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- Output dir ----
    if args.output_dir is None:
        tag = "overfit" if args.overfit else "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"debug/data/critic_{args.task}_{tag}_{timestamp}"

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

    # ---- Data ----
    train_npz = Path(args.data_dir) / f"critic_{args.task}_train.npz"
    val_npz = Path(args.data_dir) / f"critic_{args.task}_val.npz"

    max_ep = args.overfit_episodes if args.overfit else -1
    train_dataset = CriticEpisodeDataset(
        str(train_npz), n_obs_steps=args.n_obs_steps,
        max_episodes=max_ep,
    )
    val_dataset = CriticEpisodeDataset(
        str(val_npz), n_obs_steps=args.n_obs_steps,
        max_episodes=max_ep if args.overfit else -1,
    )

    if is_main_process(rank):
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")

    # Batch size per GPU
    per_gpu_batch = args.batch_size // world_size

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    else:
        train_sampler = None
    # Keep validation on the full dataset so the logged val loss is comparable
    # across runs and does not depend on the rank-0 shard only.
    val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_batch, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=per_gpu_batch, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ----
    # Default to iter_10 action model checkpoint if not specified
    if args.action_model_checkpoint is None:
        default_ckpt = f"data/baseline_checkpoints/iter_10/policy_{args.task}/pretrained_model"
        if Path(default_ckpt).exists():
            args.action_model_checkpoint = default_ckpt

    config = CriticConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        n_obs_steps=args.n_obs_steps,
        crop_shape=tuple(args.crop_shape),
        value_loss_type=args.value_loss_type,
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        action_model_checkpoint=args.action_model_checkpoint,
        # Align with action model config
        vision_backbone="resnet18",
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        use_separate_rgb_encoder_per_camera=False,
        mlp_hidden_dims=(512, 512, 256, 256),
    )

    if is_main_process(rank):
        print(f"\nCriticConfig:")
        print(f"  mlp_hidden_dims={config.mlp_hidden_dims}")
        print(f"  state_dim={config.robot_state_feature.shape}")
        print(f"  image_features={list(config.image_features.keys())}")
        print(f"  crop_shape={config.crop_shape}")
        print(f"  action_model_checkpoint={config.action_model_checkpoint}")
        print(f"  value_loss_type={config.value_loss_type}")

    model = CriticModel(config).to(device)

    if is_main_process(rank):
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  total params: {n_params:,}")
        print(f"  trainable:    {n_trainable:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Tensorboard ----
    writer = None
    if is_main_process(rank):
        writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        # Save config
        config_dict = {
            "args": vars(args),
            "model_params": n_params,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"\nOutput: {output_dir}")
        print(f"Tensorboard: tensorboard --logdir {output_dir / 'tb'}")

    # ---- Training loop ----
    model.train()
    train_iter = iter(train_loader)
    step = 0
    epoch = 0
    log_losses = []
    t_start = time.time()

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Training: {args.num_steps} steps, batch_size={args.batch_size} "
              f"({'overfit' if args.overfit else 'full'})")
        print(f"{'='*60}\n")

    while step < args.num_steps:
        # Get next batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if world_size > 1:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # LR schedule
        lr = cosine_schedule(step, args.num_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        optimizer.zero_grad()
        # Call model() (not model.module) so DDP hooks fire for gradient sync
        pred_value = model(batch)  # (B, 1)
        # Compute loss outside model for DDP compatibility
        raw_config = (model.module if isinstance(model, DDP) else model).config
        target_value = batch["mc_value"]  # (B,)
        if raw_config.value_loss_type == "mse":
            loss = F.mse_loss(pred_value.squeeze(-1), target_value)
        elif raw_config.value_loss_type == "huber":
            loss = F.huber_loss(pred_value.squeeze(-1), target_value)
        else:
            raise ValueError(f"Unsupported: {raw_config.value_loss_type}")
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)

        optimizer.step()
        step += 1

        log_losses.append(loss.item())

        # ---- Logging ----
        if is_main_process(rank) and step % args.log_freq == 0:
            avg_loss = sum(log_losses) / len(log_losses)
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            eta = (args.num_steps - step) / max(steps_per_sec, 1e-6)

            print(f"step {step:6d}/{args.num_steps} | loss {avg_loss:.6f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                  f"speed {steps_per_sec:.1f} it/s | ETA {eta/60:.0f}min")

            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, step)
            log_losses = []

        # ---- Evaluation ----
        if is_main_process(rank) and step % args.eval_freq == 0:
            eval_model = model.module if isinstance(model, DDP) else model
            val_loss = evaluate(eval_model, val_loader, device,
                                 value_loss_type=args.value_loss_type)
            print(f"  [val] step {step} | val_loss {val_loss:.6f}")
            writer.add_scalar("val/loss", val_loss, step)

        # ---- Save checkpoint ----
        if is_main_process(rank) and step % args.save_freq == 0:
            ckpt_dir = output_dir / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_model = model.module if isinstance(model, DDP) else model
            torch.save({
                "step": step,
                "model_state_dict": save_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
                "loss": loss.item(),
            }, ckpt_dir / "checkpoint.pt")
            print(f"  [save] checkpoint at step {step}")

    # ---- Final save ----
    if is_main_process(rank):
        ckpt_dir = output_dir / "checkpoints" / "final"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_model = model.module if isinstance(model, DDP) else model
        torch.save({
            "step": step,
            "model_state_dict": save_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        }, ckpt_dir / "checkpoint.pt")

        # Final eval
        eval_model = model.module if isinstance(model, DDP) else model
        val_loss = evaluate(eval_model, val_loader, device,
                             value_loss_type=args.value_loss_type)
        print(f"\n{'='*60}")
        print(f"Training complete. Final val loss: {val_loss:.6f}")
        print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
        print(f"{'='*60}")

        writer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
