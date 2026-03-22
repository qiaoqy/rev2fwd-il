"""Shared utility functions for SAC training scripts.

Reuses the same interface as scripts_rl/utils.py for consistency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = τ * source + (1 - τ) * target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy parameters from source to target."""
    target.load_state_dict(source.state_dict())


def save_checkpoint(
    path: str | Path,
    step: int = 0,
    **state_dicts,
) -> None:
    """Save training checkpoint with arbitrary state dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"step": step}
    state.update(state_dicts)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    device: str = "cuda",
) -> dict:
    """Load training checkpoint, returns full state dict."""
    return torch.load(path, map_location=device, weights_only=False)


def save_metrics(path: str | Path, metrics: dict) -> None:
    """Append metrics to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def linear_schedule(start: float, end: float, current: int, total: int) -> float:
    """Linear interpolation from start to end over total steps."""
    frac = min(current / max(total, 1), 1.0)
    return start + (end - start) * frac


class RunningMeanStd:
    """Running mean/std for reward normalization (Welford's algorithm)."""

    def __init__(self, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / max(total, 1)
        new_var = m2 / max(total, 1)

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)
