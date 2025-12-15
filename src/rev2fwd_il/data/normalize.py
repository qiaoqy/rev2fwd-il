"""Observation normalization utilities for BC training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import torch


def compute_obs_norm(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for observation normalization.

    Args:
        obs: Observation array of shape (N, obs_dim).

    Returns:
        Tuple of (mean, std) arrays, each of shape (obs_dim,).
        std is clipped to minimum 1e-6 to avoid division by zero.
    """
    mean = obs.mean(axis=0).astype(np.float32)
    std = obs.std(axis=0).astype(np.float32)
    # Clip std to avoid division by zero
    std = np.clip(std, 1e-6, None)
    return mean, std


def apply_norm(
    obs: Union[np.ndarray, "torch.Tensor"],
    mean: Union[np.ndarray, "torch.Tensor"],
    std: Union[np.ndarray, "torch.Tensor"],
) -> Union[np.ndarray, "torch.Tensor"]:
    """Apply z-score normalization to observations.

    Args:
        obs: Observation array/tensor of shape (..., obs_dim).
        mean: Mean array/tensor of shape (obs_dim,).
        std: Std array/tensor of shape (obs_dim,).

    Returns:
        Normalized observations with same shape and type as input.
    """
    return (obs - mean) / std


def save_norm(path: Union[str, Path], mean: np.ndarray, std: np.ndarray) -> None:
    """Save normalization statistics to JSON file.

    Args:
        path: Output file path (should end with .json).
        mean: Mean array of shape (obs_dim,).
        std: Std array of shape (obs_dim,).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_norm(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load normalization statistics from JSON file.

    Args:
        path: Input file path.

    Returns:
        Tuple of (mean, std) arrays, each of shape (obs_dim,).
    """
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)

    return mean, std
