"""ResNet-style MLP policy network for behavior cloning.

This architecture uses residual connections to enable training of deeper networks
with better gradient flow. It typically outperforms vanilla MLPs for BC tasks.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and skip connection.
    
    Structure: x -> Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> (+x) -> ReLU
    
    Args:
        dim: Hidden dimension (input and output are the same).
        dropout: Dropout probability (default 0.0).
    """
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.relu(x + self.net(x))


class ResNetPolicy(nn.Module):
    """ResNet-style MLP policy for behavior cloning.
    
    Architecture:
        1. Input projection: obs_dim -> hidden_dim
        2. N residual blocks at hidden_dim
        3. Output projection: hidden_dim -> act_dim
    
    Output structure (8-dim):
        - position: 3D (direct output, no constraint)
        - quaternion: 4D (normalized to unit quaternion)
        - gripper: 1D (tanh activation, maps to [-1, 1])

    Args:
        obs_dim: Dimension of observation input.
        hidden_dim: Hidden dimension for all residual blocks.
        num_blocks: Number of residual blocks.
        act_dim: Dimension of action output (default 8).
        dropout: Dropout probability in residual blocks.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        act_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, act_dim)
        
        # Initialize output layer with small weights for stable training
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

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
        # Feature extraction
        h = self.input_proj(obs)
        h = self.res_blocks(h)
        raw = self.output_proj(h)  # (B, 8)

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
