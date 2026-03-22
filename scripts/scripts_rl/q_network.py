"""Twin Q-Networks for TD-Learning Diffusion Policy (Plan A).

Architecture:
- Shared vision encoder (frozen or finetunable ResNet18 from pretrained policy)
- MLP heads that take (obs_embedding, action_flat) → Q scalar
- Twin Q-networks for clipped double-Q (TD3 style)
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMlpHead(nn.Module):
    """MLP head: (obs_embed + action) → scalar Q-value."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = (512, 512)):
        super().__init__()
        layers = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU(inplace=True))
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class VisionEncoder(nn.Module):
    """Extract visual features from table + wrist images using ResNet18.

    Weights can be initialized from a pretrained DiffusionPolicy's vision
    backbone.  During RL training the encoder can be frozen or partially
    frozen.
    """

    def __init__(
        self,
        image_channels: int = 3,
        output_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)

        # Remove final FC, use avgpool output (512-dim)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, C, H, W)
        Returns:
            (B, output_dim)
        """
        feat = self.features(image)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)


class DualVisionEncoder(nn.Module):
    """Encode table + wrist images → single embedding."""

    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.table_enc = VisionEncoder(output_dim=output_dim, pretrained=pretrained)
        self.wrist_enc = VisionEncoder(output_dim=output_dim, pretrained=pretrained)
        self.combine = nn.Linear(output_dim * 2, output_dim)

    def forward(
        self, table_img: torch.Tensor, wrist_img: torch.Tensor,
    ) -> torch.Tensor:
        t = self.table_enc(table_img)
        w = self.wrist_enc(wrist_img)
        return self.combine(torch.cat([t, w], dim=-1))


class TwinQNetwork(nn.Module):
    """Twin Q-networks for TD3-style clipped double-Q.

    Input: observation images + state + action chunk.
    Output: two scalar Q-values.
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 8,
        action_horizon: int = 16,
        vision_dim: int = 512,
        hidden_dims: list[int] = (512, 512),
        pretrained_vision: bool = True,
    ):
        super().__init__()
        self.vision_encoder = DualVisionEncoder(
            output_dim=vision_dim, pretrained=pretrained_vision,
        )

        # Action chunk is flattened
        action_flat_dim = action_dim * action_horizon
        q_input_dim = vision_dim + state_dim + action_flat_dim

        self.q1 = QMlpHead(q_input_dim, list(hidden_dims))
        self.q2 = QMlpHead(q_input_dim, list(hidden_dims))

    def forward(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            table_img: (B, 3, H, W)
            wrist_img: (B, 3, H, W)
            state: (B, state_dim)
            action: (B, horizon, action_dim) or (B, action_dim*horizon)

        Returns:
            (q1, q2): each (B,)
        """
        vis = self.vision_encoder(table_img, wrist_img)
        if action.dim() == 3:
            action = action.flatten(1)
        x = torch.cat([vis, state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward only Q1 (for actor gradient)."""
        vis = self.vision_encoder(table_img, wrist_img)
        if action.dim() == 3:
            action = action.flatten(1)
        x = torch.cat([vis, state, action], dim=-1)
        return self.q1(x)
