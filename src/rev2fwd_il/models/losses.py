"""Loss functions for behavior cloning training."""

from __future__ import annotations

import torch
import torch.nn as nn


def pose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    quat_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute pose loss with separate position and quaternion components.

    Args:
        pred: Predicted pose tensor of shape (B, 7) [pos(3), quat(4)].
        target: Target pose tensor of shape (B, 7) [pos(3), quat(4)].
        pos_weight: Weight for position loss.
        quat_weight: Weight for quaternion loss.

    Returns:
        Tuple of (total_loss, pos_loss, quat_loss).
    """
    pos_pred = pred[:, :3]
    pos_target = target[:, :3]
    quat_pred = pred[:, 3:7]
    quat_target = target[:, 3:7]

    # Position loss: MSE
    pos_loss = nn.functional.mse_loss(pos_pred, pos_target)

    # Quaternion loss: 1 - |dot(q_pred, q_target)|
    # This handles q and -q equivalence
    dot = torch.sum(quat_pred * quat_target, dim=-1)  # (B,)
    quat_loss = (1.0 - torch.abs(dot)).mean()

    total_loss = pos_weight * pos_loss + quat_weight * quat_loss

    return total_loss, pos_loss, quat_loss


def gripper_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute gripper loss using MSE.

    MSE works well since labels are -1/+1 and predictions are in [-1, 1].

    Args:
        pred: Predicted gripper tensor of shape (B, 1) or (B,).
        target: Target gripper tensor of shape (B, 1) or (B,).

    Returns:
        MSE loss scalar.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    return nn.functional.mse_loss(pred, target)


def bc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 1.0,
    quat_weight: float = 1.0,
    gripper_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute full BC loss for action prediction.

    Args:
        pred: Predicted action tensor of shape (B, 8) [pos(3), quat(4), gripper(1)].
        target: Target action tensor of shape (B, 8) [pos(3), quat(4), gripper(1)].
        pos_weight: Weight for position loss.
        quat_weight: Weight for quaternion loss.
        gripper_weight: Weight for gripper loss.

    Returns:
        Dictionary with:
            - "total": Total weighted loss
            - "pos": Position loss (unweighted)
            - "quat": Quaternion loss (unweighted)
            - "gripper": Gripper loss (unweighted)
    """
    # Split predictions and targets
    pose_pred = pred[:, :7]
    pose_target = target[:, :7]
    grip_pred = pred[:, 7:]
    grip_target = target[:, 7:]

    # Compute individual losses
    _, pos_l, quat_l = pose_loss(pose_pred, pose_target, pos_weight=1.0, quat_weight=1.0)
    grip_l = gripper_loss(grip_pred, grip_target)

    # Total weighted loss
    total = pos_weight * pos_l + quat_weight * quat_l + gripper_weight * grip_l

    return {
        "total": total,
        "pos": pos_l,
        "quat": quat_l,
        "gripper": grip_l,
    }
