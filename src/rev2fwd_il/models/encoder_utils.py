#!/usr/bin/env python3
"""Utility functions for loading a frozen DiffusionRgbEncoder from a Policy checkpoint.

Reuses the existing DiffusionRgbEncoder architecture (ResNet18 + SpatialSoftmax + Linear)
and the weight extraction pattern from CriticModel._load_vision_weights_from_action_model().

Usage:
    from rev2fwd_il.models.encoder_utils import load_frozen_rgb_encoder, encode_images_batch

    encoder = load_frozen_rgb_encoder("/path/to/pretrained_model", device="cuda:0")
    latents = encode_images_batch(encoder, images_uint8, device="cuda:0", batch_size=512)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from rev2fwd_il.models.critic_model import DiffusionRgbEncoder


def _build_diffusion_config_for_encoder(
    image_shape: tuple[int, int, int] = (3, 128, 128),
    vision_backbone: str = "resnet18",
    crop_shape: tuple[int, int] | None = (128, 128),
    crop_is_random: bool = True,
    use_group_norm: bool = True,
    spatial_softmax_num_keypoints: int = 32,
    pretrained_backbone_weights: str | None = None,
) -> DiffusionConfig:
    """Build a minimal DiffusionConfig sufficient for constructing DiffusionRgbEncoder."""
    config = DiffusionConfig()
    config.input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=list(image_shape)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=[15]),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=[8]),
    }
    config.vision_backbone = vision_backbone
    config.crop_shape = list(crop_shape) if crop_shape is not None else None
    config.crop_is_random = crop_is_random
    config.use_group_norm = use_group_norm
    config.spatial_softmax_num_keypoints = spatial_softmax_num_keypoints
    config.pretrained_backbone_weights = pretrained_backbone_weights
    return config


def load_frozen_rgb_encoder(
    policy_checkpoint_path: str,
    device: str = "cuda:0",
    image_shape: tuple[int, int, int] = (3, 128, 128),
    crop_shape: tuple[int, int] | None = (128, 128),
    spatial_softmax_num_keypoints: int = 32,
) -> DiffusionRgbEncoder:
    """Load a frozen DiffusionRgbEncoder from a Policy A checkpoint.

    Args:
        policy_checkpoint_path: Path to the pretrained_model directory (containing config.json
            and model.safetensors) or to a .safetensors / .pt file directly.
        device: Device to load the encoder onto.
        image_shape: (C, H, W) of input images.
        crop_shape: Crop shape for the encoder (should match training config).
        spatial_softmax_num_keypoints: Number of spatial softmax keypoints (default 32 → 64-d output).

    Returns:
        Frozen DiffusionRgbEncoder in eval mode, on the specified device.
    """
    ckpt_path = Path(policy_checkpoint_path)

    # Try to load config from checkpoint directory
    config_loaded = False
    if ckpt_path.is_dir():
        config_json = ckpt_path / "config.json"
        if config_json.exists():
            import json
            with open(config_json) as f:
                cfg_dict = json.load(f)
            # Extract relevant encoder params from saved config
            image_shape = tuple(image_shape)
            crop_shape_val = cfg_dict.get("crop_shape", list(crop_shape) if crop_shape else None)
            crop_shape = tuple(crop_shape_val) if crop_shape_val else None
            spatial_softmax_num_keypoints = cfg_dict.get(
                "spatial_softmax_num_keypoints", spatial_softmax_num_keypoints
            )
            config_loaded = True

    # Build config for encoder construction
    config = _build_diffusion_config_for_encoder(
        image_shape=image_shape,
        crop_shape=crop_shape,
        spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
        use_group_norm=True,
        pretrained_backbone_weights=None,
    )

    # Build encoder
    encoder = DiffusionRgbEncoder(config)

    # Resolve checkpoint file
    if ckpt_path.is_dir():
        safetensors_path = ckpt_path / "model.safetensors"
        pt_path = ckpt_path / "model.pt"
        if safetensors_path.exists():
            ckpt_path = safetensors_path
        elif pt_path.exists():
            ckpt_path = pt_path
        else:
            raise FileNotFoundError(
                f"No model.safetensors or model.pt found in {policy_checkpoint_path}"
            )

    # Load state dict
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_path))
    else:
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]

    # Extract rgb_encoder keys (strip "diffusion.rgb_encoder." prefix)
    prefix = "diffusion.rgb_encoder."
    vision_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            vision_sd[k[len(prefix):]] = v

    if len(vision_sd) == 0:
        raise ValueError(
            f"No rgb_encoder weights found in checkpoint: {policy_checkpoint_path}. "
            f"Available keys (first 10): {list(state_dict.keys())[:10]}"
        )

    missing, unexpected = encoder.load_state_dict(vision_sd, strict=False)
    print(f"[encoder_utils] Loaded frozen encoder from: {policy_checkpoint_path}")
    print(f"  matched keys: {len(vision_sd) - len(unexpected)}, feature_dim: {encoder.feature_dim}")
    if missing:
        print(f"  missing keys: {missing}")
    if unexpected:
        print(f"  unexpected keys: {unexpected}")

    # Freeze and eval
    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)

    return encoder


@torch.no_grad()
def encode_images_batch(
    encoder: DiffusionRgbEncoder,
    images_uint8: np.ndarray,
    device: str = "cuda:0",
    batch_size: int = 512,
) -> np.ndarray:
    """Encode a batch of uint8 images to latent vectors using the frozen encoder.

    Args:
        encoder: Frozen DiffusionRgbEncoder.
        images_uint8: (N, H, W, C) uint8 images.
        device: Device for computation.
        batch_size: Batch size for inference.

    Returns:
        (N, feature_dim) float32 numpy array of latent vectors.
    """
    N = len(images_uint8)
    feature_dim = encoder.feature_dim
    latents = np.empty((N, feature_dim), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        # (B, H, W, C) uint8 → (B, C, H, W) float32 [0, 1]
        batch_np = images_uint8[start:end].astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
        out = encoder(batch_tensor)  # (B, feature_dim)
        latents[start:end] = out.cpu().numpy()

    return latents
