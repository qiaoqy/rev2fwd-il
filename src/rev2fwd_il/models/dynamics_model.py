#!/usr/bin/env python3
"""Forward Dynamics Model (FDM) and Inverse Dynamics Model (IDM) for Exp49.

FDM: Residual prediction — obs_next_pred = obs + FDM(obs, action)
IDM: Direct prediction  — action_pred = IDM(obs, obs_next)

Both are simple MLPs with LayerNorm + ReLU between hidden layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardDynamicsModel(nn.Module):
    """Residual forward dynamics: obs_next = obs + FDM(obs, action)."""

    def __init__(
        self,
        obs_dim: int = 143,
        action_dim: int = 8,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        in_dim = obs_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, obs_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict residual Δobs.

        Args:
            obs: (B, obs_dim) normalized observation.
            action: (B, action_dim) normalized action.

        Returns:
            delta_obs: (B, obs_dim) predicted residual.
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

    def predict(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next observation: obs + Δobs."""
        return obs + self.forward(obs, action)


class InverseDynamicsModel(nn.Module):
    """Inverse dynamics: action = IDM(obs, obs_next)."""

    def __init__(
        self,
        obs_dim: int = 143,
        action_dim: int = 8,
        hidden_dims: tuple[int, ...] = (512, 512, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        in_dim = obs_dim * 2
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, obs_next: torch.Tensor) -> torch.Tensor:
        """Predict action from (obs, obs_next).

        Args:
            obs: (B, obs_dim) normalized current observation.
            obs_next: (B, obs_dim) normalized next observation.

        Returns:
            action: (B, action_dim) predicted action (in normalized space).
        """
        x = torch.cat([obs, obs_next], dim=-1)
        return self.net(x)
