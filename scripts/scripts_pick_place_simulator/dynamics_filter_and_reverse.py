#!/usr/bin/env python3
"""Dynamics-based filtering and reversal of Task B rollout data for Exp49.

For each Task B success episode:
  1. IDM predicts reverse actions between consecutive frames
  2. FDM verifies cycle-consistency of each frame pair
  3. Pairs with error below threshold are kept
  4. Contiguous kept segments are reversed to create Task A direction trajectories
  5. Output uses original RGB images + IDM-predicted actions

Usage:
    python scripts/scripts_pick_place_simulator/dynamics_filter_and_reverse.py \
        --encoded_B data/.../iter1_collect_B_encoded.npz \
        --raw_B data/.../iter1_collect_B.npz \
        --fdm_checkpoint data/.../iter1_dynamics/fdm_best.pt \
        --idm_checkpoint data/.../iter1_dynamics/idm_best.pt \
        --norm_stats data/.../iter1_dynamics/norm_stats.json \
        --filter_threshold data/.../iter1_dynamics/filter_threshold.json \
        --output data/.../iter1_collect_B_dynamics_filtered.npz \
        --stats_output data/.../iter1_dynamics_filter/filter_stats.json \
        --min_segment_length 16 \
        --w_state 1.0 --w_visual 0.1 \
        --success_only 1 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rev2fwd_il.models.dynamics_model import ForwardDynamicsModel, InverseDynamicsModel


# Obs layout (same as dynamics_train.py)
STATE_START = 128
STATE_END = 143
VISUAL_START = 0
VISUAL_END = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamics-based filter & reverse for Task B.")
    parser.add_argument("--encoded_B", type=str, required=True, help="Encoded Task B .npz (obs_latent).")
    parser.add_argument("--raw_B", type=str, required=True, help="Raw Task B .npz (original images).")
    parser.add_argument("--fdm_checkpoint", type=str, required=True, help="FDM best checkpoint.")
    parser.add_argument("--idm_checkpoint", type=str, required=True, help="IDM best checkpoint.")
    parser.add_argument("--norm_stats", type=str, required=True, help="Normalization stats JSON.")
    parser.add_argument("--filter_threshold", type=str, required=True, help="Filter threshold JSON.")
    parser.add_argument("--output", type=str, required=True, help="Output filtered .npz.")
    parser.add_argument("--stats_output", type=str, required=True, help="Output filter stats JSON.")
    parser.add_argument("--min_segment_length", type=int, default=16, help="Minimum segment length.")
    parser.add_argument("--w_state", type=float, default=1.0, help="Weight for state error.")
    parser.add_argument("--w_visual", type=float, default=0.1, help="Weight for visual error.")
    parser.add_argument("--success_only", type=int, default=1, choices=[0, 1],
                        help="Only process successful episodes.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for IDM/FDM inference.")
    return parser.parse_args()


def load_models(args) -> tuple[ForwardDynamicsModel, InverseDynamicsModel,
                                dict, float, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]:
    """Load FDM, IDM, norm stats, and threshold."""
    device = args.device

    # Load norm stats
    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    obs_mean = torch.tensor(norm_stats["obs_mean"], dtype=torch.float32, device=device)
    obs_std = torch.tensor(norm_stats["obs_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(norm_stats["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(norm_stats["action_std"], dtype=torch.float32, device=device)

    # Determine dims from norm stats
    obs_dim = len(norm_stats["obs_mean"])
    action_dim = len(norm_stats["action_mean"])

    # Load threshold
    with open(args.filter_threshold) as f:
        threshold_info = json.load(f)
    threshold = threshold_info["threshold"]

    # Load FDM
    fdm = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    fdm.load_state_dict(torch.load(args.fdm_checkpoint, map_location=device, weights_only=True))
    fdm.eval()

    # Load IDM
    idm = InverseDynamicsModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    idm.load_state_dict(torch.load(args.idm_checkpoint, map_location=device, weights_only=True))
    idm.eval()

    return fdm, idm, norm_stats, threshold, obs_mean, obs_std, action_mean, action_std


@torch.no_grad()
def process_episode(
    obs_latent: np.ndarray,      # (T, 143)
    fdm: ForwardDynamicsModel,
    idm: InverseDynamicsModel,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    threshold: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run IDM reverse prediction + FDM verification on one episode.

    Returns:
        keep_mask: (T-1,) bool — whether pair (t, t+1) passes consistency check
        act_rev: (T, 8) float32 — IDM-predicted reverse actions (denormalized)
            act_rev[t] = IDM(obs[t], obs[t-1]) for t=1..T-1, act_rev[0] = zeros
        errors: (T-1,) float32 — consistency error per pair
    """
    device = args.device
    T = len(obs_latent)

    obs_tensor = torch.from_numpy(obs_latent).float().to(device)  # (T, 143)
    obs_norm = (obs_tensor - obs_mean) / obs_std

    # IDM: predict reverse actions
    # For each t in 1..T-1: act_rev[t] = IDM(obs_norm[t], obs_norm[t-1])
    # "Standing at t, what action goes to t-1?"
    obs_curr = obs_norm[1:]       # (T-1, 143) — frames 1..T-1
    obs_prev = obs_norm[:-1]      # (T-1, 143) — frames 0..T-2

    # Batch inference
    act_rev_norm_list = []
    for start in range(0, T - 1, args.batch_size):
        end = min(start + args.batch_size, T - 1)
        act_rev_norm_list.append(idm(obs_curr[start:end], obs_prev[start:end]))
    act_rev_norm = torch.cat(act_rev_norm_list, dim=0)  # (T-1, 8)

    # Denormalize actions
    act_rev_denorm = act_rev_norm * action_std + action_mean  # (T-1, 8)

    # FDM: verify cycle consistency
    # From obs[t] (t=1..T-1), apply act_rev to reach obs[t-1]
    # obs_pred[t-1] = obs_norm[t] + FDM(obs_norm[t], act_rev_norm[t])
    obs_pred_list = []
    for start in range(0, T - 1, args.batch_size):
        end = min(start + args.batch_size, T - 1)
        obs_pred_list.append(fdm.predict(obs_curr[start:end], act_rev_norm[start:end]))
    obs_pred = torch.cat(obs_pred_list, dim=0)  # (T-1, 143)

    # Compute consistency error: compare obs_pred with obs_prev (target)
    diff = obs_pred - obs_prev
    error_state = torch.norm(diff[:, STATE_START:STATE_END], dim=1)    # (T-1,)
    error_visual = torch.norm(diff[:, VISUAL_START:VISUAL_END], dim=1) # (T-1,)
    errors = args.w_state * error_state + args.w_visual * error_visual  # (T-1,)

    keep_mask = (errors < threshold).cpu().numpy()  # (T-1,)
    errors_np = errors.cpu().numpy()

    # Build full act_rev array: (T, 8)
    # act_rev[0] = zeros (no previous frame for frame 0)
    act_rev_full = np.zeros((T, 8), dtype=np.float32)
    act_rev_full[1:] = act_rev_denorm.cpu().numpy()

    return keep_mask, act_rev_full, errors_np


def find_contiguous_segments(keep_mask: np.ndarray, min_length: int) -> list[list[int]]:
    """Find contiguous kept segments from the pair-level keep mask.

    keep_mask[i] = True means pair (i, i+1) is kept.
    A contiguous segment of kept pairs [i, i+1, ..., j] corresponds to
    frame indices [i, i+1, ..., j+1] (inclusive).

    Returns list of frame index lists, each of length >= min_length.
    """
    segments = []
    current_start = None

    for i in range(len(keep_mask)):
        if keep_mask[i]:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                # Segment of pairs [current_start, ..., i-1]
                # Frame indices: [current_start, ..., i]
                frame_indices = list(range(current_start, i + 1))
                if len(frame_indices) >= min_length:
                    segments.append(frame_indices)
                current_start = None

    # Handle last segment
    if current_start is not None:
        frame_indices = list(range(current_start, len(keep_mask) + 1))
        if len(frame_indices) >= min_length:
            segments.append(frame_indices)

    return segments


def assemble_reversed_episode(
    raw_ep: dict,
    segment_frames: list[int],
    act_rev: np.ndarray,
) -> dict:
    """Assemble a reversed episode from a contiguous segment.

    Args:
        raw_ep: Original Task B episode (with images, wrist_images, ee_pose, etc.)
        segment_frames: Frame indices in original order [t_start, ..., t_end]
        act_rev: (T, 8) IDM-predicted reverse actions for the full episode.
            act_rev[t] = IDM(obs[t], obs[t-1]) — "at t, action to reach t-1"

    Returns:
        Reversed episode dict compatible with Policy A training pipeline.
    """
    # Reverse frame order
    reversed_frames = segment_frames[::-1]  # [t_end, ..., t_start]
    T_rev = len(reversed_frames)

    # Extract original data at reversed frame indices
    result = {
        "images": raw_ep["images"][reversed_frames],
        "ee_pose": raw_ep["ee_pose"][reversed_frames].astype(np.float32),
        "success": True,
    }

    if "wrist_images" in raw_ep:
        result["wrist_images"] = raw_ep["wrist_images"][reversed_frames]
    if "obj_pose" in raw_ep:
        result["obj_pose"] = raw_ep["obj_pose"][reversed_frames].astype(np.float32)

    # Gripper: from original data or action
    if "gripper" in raw_ep:
        result["gripper"] = raw_ep["gripper"][reversed_frames].astype(np.float32)
    else:
        result["gripper"] = raw_ep["action"][reversed_frames, 7].astype(np.float32)

    # Assemble actions for the reversed trajectory
    # For reversed trajectory step i (i=0..T_rev-2):
    #   We're at original frame reversed_frames[i] = orig_t
    #   We want to move to reversed_frames[i+1] = orig_t - 1
    #   This is exactly act_rev[orig_t] = IDM(obs[orig_t], obs[orig_t-1])
    actions = np.zeros((T_rev, 8), dtype=np.float32)
    for i in range(T_rev - 1):
        orig_t = reversed_frames[i]
        actions[i] = act_rev[orig_t]
    # Last frame: copy second-to-last action (no next target)
    if T_rev >= 2:
        actions[T_rev - 1] = actions[T_rev - 2]
    # Override IDM-predicted gripper dim with ground-truth gripper state
    # (IDM only predicts pose dims 0-6; gripper is a discrete state, not a regression target)
    actions[:, 7] = result["gripper"]
    result["action"] = actions

    # Copy metadata if present
    if "place_pose" in raw_ep:
        result["place_pose"] = raw_ep["place_pose"].copy() if hasattr(raw_ep["place_pose"], "copy") else raw_ep["place_pose"]
    if "goal_pose" in raw_ep:
        result["goal_pose"] = raw_ep["goal_pose"].copy() if hasattr(raw_ep["goal_pose"], "copy") else raw_ep["goal_pose"]

    # Synthesize obs field (zeros, compatible with pipeline)
    result["obs"] = np.zeros((T_rev, 36), dtype=np.float32)
    result["fsm_state"] = np.zeros(T_rev, dtype=np.int32)

    return result


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Load models and stats
    print("Loading dynamics models...")
    fdm, idm, norm_stats, threshold, obs_mean, obs_std, action_mean, action_std = load_models(args)
    print(f"  Threshold: {threshold:.6f}")

    # Load encoded B (obs_latent)
    print(f"Loading encoded B: {args.encoded_B}")
    with np.load(args.encoded_B, allow_pickle=True) as data:
        encoded_episodes = list(data["episodes"])
    print(f"  {len(encoded_episodes)} episodes")

    # Load raw B (images etc.)
    print(f"Loading raw B: {args.raw_B}")
    with np.load(args.raw_B, allow_pickle=True) as data:
        raw_episodes = list(data["episodes"])
    print(f"  {len(raw_episodes)} episodes")

    assert len(encoded_episodes) == len(raw_episodes), \
        f"Episode count mismatch: encoded={len(encoded_episodes)} vs raw={len(raw_episodes)}"

    # Process episodes
    output_episodes = []
    per_episode_stats = []
    total_frames_in = 0
    total_frames_kept = 0
    total_segments = 0

    for ep_idx in range(len(encoded_episodes)):
        enc_ep = encoded_episodes[ep_idx]
        raw_ep = raw_episodes[ep_idx]

        # Skip non-success if requested
        is_success = enc_ep.get("success", False)
        if args.success_only and not is_success:
            continue

        T = len(enc_ep["obs_latent"])
        total_frames_in += T

        if T < args.min_segment_length:
            per_episode_stats.append({
                "ep_idx": int(ep_idx),
                "original_length": int(T),
                "kept_frames": 0,
                "num_segments": 0,
                "skipped": "too_short",
            })
            continue

        # Run IDM + FDM filtering
        keep_mask, act_rev, errors = process_episode(
            enc_ep["obs_latent"], fdm, idm,
            obs_mean, obs_std, action_mean, action_std,
            threshold, args,
        )

        # Find contiguous segments
        segments = find_contiguous_segments(keep_mask, args.min_segment_length)

        ep_kept_frames = sum(len(seg) for seg in segments)
        total_frames_kept += ep_kept_frames
        total_segments += len(segments)

        per_episode_stats.append({
            "ep_idx": int(ep_idx),
            "original_length": int(T),
            "pairs_checked": int(len(keep_mask)),
            "pairs_kept": int(keep_mask.sum()),
            "pair_keep_ratio": float(keep_mask.mean()),
            "num_segments": len(segments),
            "segment_lengths": [len(seg) for seg in segments],
            "kept_frames": int(ep_kept_frames),
            "error_mean": float(errors.mean()),
            "error_std": float(errors.std()),
            "error_max": float(errors.max()),
        })

        # Assemble reversed episodes from each segment
        for seg_idx, segment_frames in enumerate(segments):
            rev_ep = assemble_reversed_episode(raw_ep, segment_frames, act_rev)
            output_episodes.append(rev_ep)

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            stats = per_episode_stats[-1]
            print(f"  Episode {ep_idx}: T={T}, kept={stats['pairs_kept']}/{stats['pairs_checked']} "
                  f"({stats['pair_keep_ratio']:.2%}), segments={stats['num_segments']}, "
                  f"err_mean={stats['error_mean']:.4f}")

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, episodes=np.array(output_episodes, dtype=object))

    # Save stats
    keep_ratio = total_frames_kept / total_frames_in if total_frames_in > 0 else 0.0
    filter_stats = {
        "total_input_episodes": len(encoded_episodes),
        "processed_episodes": len(per_episode_stats),
        "total_frames_in": int(total_frames_in),
        "total_frames_kept": int(total_frames_kept),
        "keep_ratio": float(keep_ratio),
        "total_output_segments": int(total_segments),
        "total_output_episodes": len(output_episodes),
        "threshold": float(threshold),
        "min_segment_length": args.min_segment_length,
        "per_episode": per_episode_stats,
    }
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_output, "w") as f:
        json.dump(filter_stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Dynamics Filter & Reverse — Summary")
    print(f"{'='*60}")
    print(f"  Input episodes:      {len(encoded_episodes)} ({len(per_episode_stats)} processed)")
    print(f"  Total frames in:     {total_frames_in}")
    print(f"  Total frames kept:   {total_frames_kept} ({keep_ratio:.1%})")
    print(f"  Output segments:     {total_segments}")
    print(f"  Output episodes:     {len(output_episodes)}")
    print(f"  Threshold:           {threshold:.6f}")
    print(f"  Time:                {elapsed:.1f}s")
    print(f"  Output:              {args.output}")
    print(f"  Stats:               {args.stats_output}")


if __name__ == "__main__":
    main()
