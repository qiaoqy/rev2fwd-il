"""Behavior cloning trainer for policy A."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import matplotlib.pyplot as plt

from rev2fwd_il.data.normalize import compute_obs_norm, save_norm, apply_norm
from rev2fwd_il.models.mlp_policy import MLPPolicy
from rev2fwd_il.models.resnet_policy import ResNetPolicy
from rev2fwd_il.models.losses import bc_loss
from rev2fwd_il.utils.seed import set_seed


def save_loss_curve(history: dict, save_path: Path) -> None:
    """Save training loss curve as PNG.

    Args:
        history: Dictionary containing loss history with keys:
            - epoch: List of epoch numbers
            - loss_total: List of total loss values
            - loss_pos: List of position loss values
            - loss_quat: List of quaternion loss values
            - loss_gripper: List of gripper loss values
        save_path: Path to save the PNG file.
    """
    plt.figure(figsize=(10, 6))
    
    epochs = history["epoch"]
    
    plt.plot(epochs, history["loss_total"], label="Total Loss", linewidth=2)
    plt.plot(epochs, history["loss_pos"], label="Position Loss", linewidth=1.5, linestyle="--")
    plt.plot(epochs, history["loss_quat"], label="Quaternion Loss", linewidth=1.5, linestyle="--")
    plt.plot(epochs, history["loss_gripper"], label="Gripper Loss", linewidth=1.5, linestyle="--")
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.yscale("log")
    plt.title("Behavior Cloning Training Loss Curve", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_bc(
    dataset_npz: str,
    out_dir: str,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: tuple[int, ...] = (256, 256),
    pos_weight: float = 1.0,
    quat_weight: float = 1.0,
    gripper_weight: float = 1.0,
    print_every: int = 10,
    device: Optional[str] = None,
    arch: str = "resnet",
    num_blocks: int = 3,
    dropout: float = 0.0,
) -> dict:
    """Train policy with behavior cloning.

    Args:
        dataset_npz: Path to the forward BC dataset NPZ file.
        out_dir: Output directory for model, norm, and config.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        seed: Random seed.
        hidden: Hidden layer sizes for MLP, or hidden_dim for ResNet (first element used).
        pos_weight: Weight for position loss.
        quat_weight: Weight for quaternion loss.
        gripper_weight: Weight for gripper loss.
        print_every: Print loss every N epochs.
        device: Torch device (auto-detected if None).
        arch: Model architecture, 'mlp' or 'resnet'.
        num_blocks: Number of residual blocks (only for resnet).
        dropout: Dropout probability (only for resnet).

    Returns:
        Dictionary with training statistics.
    """
    # Set seed
    set_seed(seed)

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("Behavior Cloning Training")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_npz}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Step 1: Load dataset
    # =========================================================================
    print("Loading dataset...")
    data = np.load(dataset_npz)
    obs_np = data["obs"].astype(np.float32)  # (N, obs_dim)
    act_np = data["act"].astype(np.float32)  # (N, 8)

    N, obs_dim = obs_np.shape
    act_dim = act_np.shape[1]

    print(f"  Samples: {N}")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Act dim: {act_dim}")

    # =========================================================================
    # Step 2: Compute and apply observation normalization
    # =========================================================================
    print("\nComputing observation normalization...")
    mean, std = compute_obs_norm(obs_np)
    obs_normed = apply_norm(obs_np, mean, std)

    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")

    # =========================================================================
    # Step 3: Create DataLoader
    # =========================================================================
    obs_tensor = torch.from_numpy(obs_normed).float()
    act_tensor = torch.from_numpy(act_np).float()

    # Decide whether to put data on GPU or use CPU with pin_memory
    # For large datasets (>1M samples), keep data on CPU to avoid slow GPU shuffle
    USE_GPU_DATA = (N < 1_000_000) and (device == "cuda")
    
    if USE_GPU_DATA:
        # Small dataset: put everything on GPU for faster training
        obs_tensor = obs_tensor.to(device)
        act_tensor = act_tensor.to(device)
        pin_memory = False
        num_workers = 0
        print(f"  Using GPU data loading (dataset < 1M samples)")
    else:
        # Large dataset: keep on CPU, use pin_memory and workers
        pin_memory = (device == "cuda")
        num_workers = 4
        print(f"  Using CPU data loading with {num_workers} workers (dataset >= 1M samples)")

    dataset = TensorDataset(obs_tensor, act_tensor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    num_batches = len(dataloader)
    print(f"\nDataLoader created: {num_batches} batches per epoch")
    print(f"  Data location: {obs_tensor.device}")

    # =========================================================================
    # Step 4: Create model and optimizer
    # =========================================================================
    if arch == "mlp":
        model = MLPPolicy(obs_dim=obs_dim, hidden=hidden, act_dim=act_dim)
    elif arch == "resnet":
        model = ResNetPolicy(
            obs_dim=obs_dim,
            hidden_dim=hidden[0],
            act_dim=act_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created: {num_params:,} parameters")

    # =========================================================================
    # Step 5: Training loop
    # =========================================================================
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    history = {
        "epoch": [],
        "loss_total": [],
        "loss_pos": [],
        "loss_quat": [],
        "loss_gripper": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = {"total": 0.0, "pos": 0.0, "quat": 0.0, "gripper": 0.0}

        for obs_batch, act_batch in dataloader:
            # Data is already on device if using CUDA, but this is a no-op if so
            if obs_batch.device.type != device:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

            # Forward pass
            act_pred = model(obs_batch)

            # Compute loss
            losses = bc_loss(
                act_pred, act_batch,
                pos_weight=pos_weight,
                quat_weight=quat_weight,
                gripper_weight=gripper_weight,
            )

            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            # Accumulate losses
            epoch_loss["total"] += losses["total"].item()
            epoch_loss["pos"] += losses["pos"].item()
            epoch_loss["quat"] += losses["quat"].item()
            epoch_loss["gripper"] += losses["gripper"].item()

        # Average over batches
        for k in epoch_loss:
            epoch_loss[k] /= num_batches

        # Record history
        history["epoch"].append(epoch)
        history["loss_total"].append(epoch_loss["total"])
        history["loss_pos"].append(epoch_loss["pos"])
        history["loss_quat"].append(epoch_loss["quat"])
        history["loss_gripper"].append(epoch_loss["gripper"])

        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {epoch_loss['total']:.6f} "
                f"(pos: {epoch_loss['pos']:.6f}, "
                f"quat: {epoch_loss['quat']:.6f}, "
                f"grip: {epoch_loss['gripper']:.6f}) | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    # =========================================================================
    # Step 6: Save outputs
    # =========================================================================
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = out_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": hidden,
        "arch": arch,
        "num_blocks": num_blocks if arch == "resnet" else None,
        "dropout": dropout if arch == "resnet" else None,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Save normalization
    norm_path = out_path / "norm.json"
    save_norm(norm_path, mean, std)
    print(f"Saved normalization to {norm_path}")

    # Save config
    config = {
        "dataset": str(dataset_npz),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "hidden": list(hidden),
        "pos_weight": pos_weight,
        "quat_weight": quat_weight,
        "gripper_weight": gripper_weight,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "num_samples": N,
        "num_params": num_params,
        "training_time_s": total_time,
        "final_loss": {
            "total": history["loss_total"][-1],
            "pos": history["loss_pos"][-1],
            "quat": history["loss_quat"][-1],
            "gripper": history["loss_gripper"][-1],
        },
    }

    config_path = out_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")

    # Save loss curve
    loss_curve_path = out_path / "loss_curve.png"
    save_loss_curve(history, loss_curve_path)
    print(f"Saved loss curve to {loss_curve_path}")

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Final loss: {history['loss_total'][-1]:.6f}")
    print(f"  Position: {history['loss_pos'][-1]:.6f}")
    print(f"  Quaternion: {history['loss_quat'][-1]:.6f}")
    print(f"  Gripper: {history['loss_gripper'][-1]:.6f}")
    print(f"{'='*60}\n")

    return {
        "history": history,
        "config": config,
        "model_path": str(model_path),
        "norm_path": str(norm_path),
    }
