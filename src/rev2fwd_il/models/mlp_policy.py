"""MLP policy network for behavior cloning."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """MLP policy that outputs end-effector pose and gripper action.

    Output structure (8-dim):
        - position: 3D (direct output, no constraint)
        - quaternion: 4D (normalized to unit quaternion)
        - gripper: 1D (tanh activation, maps to [-1, 1])

    Args:
        obs_dim: Dimension of observation input.
        hidden: Tuple of hidden layer sizes.
        act_dim: Dimension of action output (default 8).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: Tuple[int, ...] = (256, 256),
        act_dim: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build MLP layers
        layers = []
        in_dim = obs_dim
        for h_dim in hidden:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Output layer: 3 (pos) + 4 (quat) + 1 (gripper) = 8
        layers.append(nn.Linear(in_dim, act_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass with output constraints.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            Action tensor of shape (B, 8) with:
                - [:, 0:3]: position (unconstrained)
                - [:, 3:7]: quaternion (normalized)
                - [:, 7:8]: gripper (tanh, in [-1, 1])
        """
        raw = self.mlp(obs)  # (B, 8)

        # Split into components
        pos = raw[:, 0:3]  # (B, 3) - position, no constraint
        quat_raw = raw[:, 3:7]  # (B, 4) - quaternion before normalization
        gripper_raw = raw[:, 7:8]  # (B, 1) - gripper before tanh

        # Normalize quaternion to unit length
        eps = 1e-8
        quat_norm = quat_raw / (torch.norm(quat_raw, dim=-1, keepdim=True) + eps)

        # Gripper: tanh activation to constrain to [-1, 1]
        gripper = torch.tanh(gripper_raw)

        # Concatenate all components
        act = torch.cat([pos, quat_norm, gripper], dim=-1)  # (B, 8)

        return act
