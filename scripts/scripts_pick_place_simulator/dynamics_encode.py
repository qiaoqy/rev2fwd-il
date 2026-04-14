#!/usr/bin/env python3
"""Pre-encode images to latent observations using a frozen ResNet18 encoder from Policy A.

For each episode, encodes table images and wrist images into 64-d latent vectors,
then concatenates with state information to produce 143-d obs_latent per frame:
    obs_latent = [table_latent(64), wrist_latent(64), ee_pose(7), obj_pose(7), gripper(1)]

Usage:
    python scripts/scripts_pick_place_simulator/dynamics_encode.py \
        --input_npz data/.../iter1_collect_A.npz \
        --output_npz data/.../iter1_collect_A_encoded.npz \
        --policy_checkpoint data/.../weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
        --batch_size 512 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rev2fwd_il.models.encoder_utils import load_frozen_rgb_encoder, encode_images_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-encode images to latent obs using frozen ResNet18.")
    parser.add_argument("--input_npz", type=str, required=True, help="Input .npz with episodes.")
    parser.add_argument("--output_npz", type=str, required=True, help="Output .npz with obs_latent added.")
    parser.add_argument("--policy_checkpoint", type=str, required=True,
                        help="Path to Policy A pretrained_model directory.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoder inference.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Load encoder
    print(f"Loading frozen encoder from: {args.policy_checkpoint}")
    encoder = load_frozen_rgb_encoder(args.policy_checkpoint, device=args.device)
    feature_dim = encoder.feature_dim  # Expected: 64
    print(f"  Encoder feature_dim: {feature_dim}")

    # Load episodes
    print(f"Loading episodes from: {args.input_npz}")
    with np.load(args.input_npz, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"  Loaded {len(episodes)} episodes")

    total_frames = 0
    encoded_episodes = []

    for ep_idx, ep in enumerate(episodes):
        T = len(ep["images"])
        total_frames += T

        # Encode table images → (T, feature_dim)
        table_latent = encode_images_batch(encoder, ep["images"], device=args.device,
                                           batch_size=args.batch_size)

        # Encode wrist images → (T, feature_dim)
        if "wrist_images" in ep:
            wrist_latent = encode_images_batch(encoder, ep["wrist_images"], device=args.device,
                                               batch_size=args.batch_size)
        else:
            wrist_latent = np.zeros((T, feature_dim), dtype=np.float32)

        # State components
        ee_pose = ep["ee_pose"].astype(np.float32)  # (T, 7)
        obj_pose = ep["obj_pose"].astype(np.float32) if "obj_pose" in ep else np.zeros((T, 7), dtype=np.float32)

        if "gripper" in ep:
            gripper = ep["gripper"].astype(np.float32).reshape(T, 1)
        else:
            gripper = ep["action"][:, 7:8].astype(np.float32)

        # Concatenate: [table_latent(64), wrist_latent(64), ee_pose(7), obj_pose(7), gripper(1)] = 143
        obs_latent = np.concatenate([table_latent, wrist_latent, ee_pose, obj_pose, gripper], axis=-1)

        # Build output episode dict: keep all original fields + add obs_latent
        out_ep = {}
        for k, v in ep.items():
            out_ep[k] = v
        out_ep["obs_latent"] = obs_latent.astype(np.float32)

        encoded_episodes.append(out_ep)

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx + 1}/{len(episodes)}: T={T}, obs_latent shape={obs_latent.shape}")

    # Save
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, episodes=np.array(encoded_episodes, dtype=object))

    elapsed = time.time() - t0
    print(f"\nDone! Encoded {len(encoded_episodes)} episodes ({total_frames} frames) in {elapsed:.1f}s")
    print(f"  obs_latent dim: {encoded_episodes[0]['obs_latent'].shape[1]}")
    print(f"  Output: {args.output_npz}")


if __name__ == "__main__":
    main()
