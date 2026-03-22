"""SAC (Soft Actor-Critic) Networks for Diffusion Policy fine-tuning.

Architecture Overview:
- Twin Q-Networks (Critic): Shared vision encoder + MLP heads
  - Input: (obs_embedding, action_flat) → Q scalar
  - Clipped double-Q for conservative value estimation
- Squashed Gaussian Actor: Generates actions via reparameterization trick
  - Input: obs_embedding → (mean, log_std) → tanh squashed action
  - Can also wrap a Diffusion Policy as the actor (SAC-Diffusion)
- Automatic Entropy Tuning: Learns temperature α to match target entropy

References:
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
  with a Stochastic Actor" (ICML 2018)
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================================
# Vision Encoder (shared with scripts_rl/q_network.py design)
# ============================================================================

class VisionEncoder(nn.Module):
    """ResNet18-based visual feature extractor.

    Weights can be initialized from a pretrained DiffusionPolicy's backbone.
    During SAC training the encoder can be frozen or partially frozen.
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

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, output_dim)."""
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


# ============================================================================
# Q-Network (Critic)
# ============================================================================

class QMlpHead(nn.Module):
    """MLP head: (obs_embed + action) → scalar Q-value."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = (256, 256)):
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


class SACTwinQNetwork(nn.Module):
    """Twin Q-networks for SAC clipped double-Q.

    SAC uses min(Q1, Q2) - α * log π to avoid overestimation.
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 8,
        action_horizon: int = 16,
        vision_dim: int = 512,
        hidden_dims: list[int] = (256, 256),
        pretrained_vision: bool = True,
    ):
        super().__init__()
        self.vision_encoder = DualVisionEncoder(
            output_dim=vision_dim, pretrained=pretrained_vision,
        )

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

    def q_min(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return min(Q1, Q2) for conservative estimation."""
        q1, q2 = self.forward(table_img, wrist_img, state, action)
        return torch.min(q1, q2)


# ============================================================================
# Squashed Gaussian Actor
# ============================================================================

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class SquashedGaussianActor(nn.Module):
    """SAC actor that outputs tanh-squashed Gaussian actions.

    The actor maps observations to a distribution over actions:
        π(a|s) = tanh(μ(s) + σ(s) * ε),   ε ~ N(0, I)

    Log probability is corrected for the tanh squashing:
        log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u_i))
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 8,
        action_horizon: int = 16,
        vision_dim: int = 512,
        hidden_dims: list[int] = (256, 256),
        pretrained_vision: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        action_flat_dim = action_dim * action_horizon

        self.vision_encoder = DualVisionEncoder(
            output_dim=vision_dim, pretrained=pretrained_vision,
        )

        # Shared trunk
        layers = []
        d_in = vision_dim + state_dim
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU(inplace=True))
            d_in = d_out
        self.trunk = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(d_in, action_flat_dim)
        self.log_std_head = nn.Linear(d_in, action_flat_dim)

    def forward(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the Gaussian before squashing."""
        vis = self.vision_encoder(table_img, wrist_img)
        h = self.trunk(torch.cat([vis, state], dim=-1))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action via reparameterization trick + tanh squashing.

        Returns:
            (action, log_prob): action (B, action_flat), log_prob (B,)
        """
        mean, log_std = self.forward(table_img, wrist_img, state)
        std = log_std.exp()

        dist = Normal(mean, std)
        u = dist.rsample()  # reparameterized sample (before tanh)
        action = torch.tanh(u)

        # Log probability with tanh correction
        log_prob = dist.log_prob(u)  # (B, action_flat)
        # Correction for tanh squashing: -log(1 - tanh²(u)) = -log(1 - a²)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1)  # sum over action dims → (B,)

        return action, log_prob

    def deterministic_action(
        self,
        table_img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return deterministic action (tanh of mean) for evaluation."""
        mean, _ = self.forward(table_img, wrist_img, state)
        return torch.tanh(mean)


# ============================================================================
# Automatic Entropy Tuning
# ============================================================================

class AutoEntropyTuning(nn.Module):
    """Automatic temperature (α) tuning for SAC.

    Optimizes:  α* = arg min_α  E[-α * (log π(a|s) + H_target)]

    where H_target = -dim(A) is the target entropy (heuristic from the paper).
    """

    def __init__(self, action_dim: int, action_horizon: int = 1, init_alpha: float = 0.2):
        super().__init__()
        self.target_entropy = -float(action_dim * action_horizon)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute alpha loss: -α * (log π + H_target).

        Args:
            log_prob: (B,) log probabilities from the actor.
        Returns:
            Scalar loss for updating α.
        """
        return -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
