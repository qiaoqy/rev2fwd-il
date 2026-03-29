#!/usr/bin/env python3
"""Speed-adjust episode trajectories: accelerate slow aerial moves, decelerate near gripper actions.

Two-phase temporal rescaling:
1. **Acceleration** – For segments where the EE moves slowly (displacement per
   frame below ``speed_threshold``), uniformly subsample frames to bring the
   effective step size closer to ``target_step_size``.
2. **Deceleration** – Around gripper state transitions (open→close or
   close→open), linearly interpolate extra frames within
   ``gripper_slow_radius`` to increase precision of these critical actions.

After all transformations, actions are recomputed:
    action[t][:7] = ee_pose[t+1]
    action[t][7]  = gripper[t]

Usage:
    python scripts/scripts_pick_place_simulator/3_speed_adjust_episodes.py \\
        --input data/exp24/rollout_A_200_filtered.npz \\
        --out data/exp24/rollout_A_200_speed_adjusted.npz \\
        --speed_threshold 2e-3 \\
        --target_step_size 4e-3 \\
        --min_slow_frames 8 \\
        --gripper_slow_radius 16 \\
        --interp_factor 2

    # Debug mode (render videos for selected episodes)
    python scripts/scripts_pick_place_simulator/3_speed_adjust_episodes.py \\
        --input data/exp24/rollout_A_200_filtered.npz \\
        --out data/exp24/rollout_A_200_speed_adjusted.npz \\
        --debug --debug_dir data/exp24/debug_videos --debug_episodes 0
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cv2
import imageio
import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speed-adjust episode data: accelerate slow moves, "
                    "decelerate near gripper transitions.",
    )
    # I/O
    p.add_argument("--input", type=str, required=True, help="Input NPZ file.")
    p.add_argument("--out", type=str, required=True, help="Output NPZ file.")
    p.add_argument("--success_only", type=int, default=0, choices=[0, 1],
                   help="Only keep successful episodes (default: 0).")

    # Acceleration parameters
    p.add_argument("--speed_threshold", type=float, default=2e-3,
                   help="Frames with EE displacement below this are 'slow'. "
                        "Default: 2e-3 (2 mm/frame).")
    p.add_argument("--target_step_size", type=float, default=4e-3,
                   help="Target EE displacement per frame after acceleration. "
                        "Default: 4e-3 (4 mm/frame).")
    p.add_argument("--min_slow_frames", type=int, default=8,
                   help="Min consecutive slow frames to trigger acceleration. "
                        "Default: 8.")

    # Deceleration parameters
    p.add_argument("--gripper_slow_radius", type=int, default=16,
                   help="Number of frames before/after gripper transitions to "
                        "decelerate. Default: 16.")
    p.add_argument("--interp_factor", type=int, default=2,
                   help="Interpolation factor for deceleration. 2 means insert "
                        "1 frame between each pair. Default: 2.")

    # Debug / video
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode: render annotated videos.")
    p.add_argument("--debug_dir", type=str, default=None,
                   help="Output directory for debug videos.")
    p.add_argument("--debug_episodes", type=int, nargs="+", default=None,
                   help="Episode indices to render. Default: first 5.")
    p.add_argument("--video_fps", type=int, default=20,
                   help="FPS for debug videos. Default: 20.")
    return p.parse_args()


# ── Core algorithms ──────────────────────────────────────────────────────

def compute_displacements(ee_pose: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean displacement of EE xyz. Returns (T-1,) array."""
    return np.linalg.norm(np.diff(ee_pose[:, :3], axis=0), axis=1)


def find_gripper_transitions(episode: dict) -> np.ndarray:
    """Return indices where gripper state changes significantly."""
    gripper = episode.get("gripper", episode["action"][:, 7])
    gripper_flat = gripper.flatten()[:len(episode["ee_pose"])]
    return np.where(np.abs(np.diff(gripper_flat)) > 0.5)[0]


def find_slow_runs(displacements: np.ndarray, speed_threshold: float,
                   min_slow_frames: int) -> list[tuple[int, int]]:
    """Find consecutive runs of slow frames. Returns (start, end) inclusive.

    Note: displacements has length T-1 for T frames. Frame t is 'slow' if
    displacement[t] < speed_threshold.
    """
    T = len(displacements) + 1
    is_slow = np.zeros(T, dtype=bool)
    is_slow[:-1] = displacements < speed_threshold
    if T > 1:
        is_slow[-1] = is_slow[-2]

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for t, s in enumerate(is_slow):
        if s:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None:
                if t - run_start >= min_slow_frames:
                    runs.append((run_start, t - 1))
                run_start = None
    if run_start is not None and T - run_start >= min_slow_frames:
        runs.append((run_start, T - 1))
    return runs


def build_gripper_protection_set(
    gripper_transitions: np.ndarray,
    gripper_slow_radius: int,
    T: int,
) -> set[int]:
    """Build set of frame indices protected by gripper transitions."""
    protected: set[int] = set()
    for gc in gripper_transitions:
        for f in range(max(0, gc - gripper_slow_radius),
                       min(T, gc + gripper_slow_radius + 1)):
            protected.add(f)
    return protected


def accelerate_slow_segments(
    episode: dict,
    speed_threshold: float,
    target_step_size: float,
    min_slow_frames: int,
    gripper_transitions: np.ndarray,
    gripper_slow_radius: int,
) -> tuple[dict, dict]:
    """Accelerate slow-moving segments by uniform subsampling.

    Frames near gripper transitions are protected from acceleration.
    """
    ee_pose = episode["ee_pose"]
    T = len(ee_pose)
    displacements = compute_displacements(ee_pose)

    slow_runs = find_slow_runs(displacements, speed_threshold, min_slow_frames)
    protected = build_gripper_protection_set(
        gripper_transitions, gripper_slow_radius, T)

    # Build keep mask
    keep = np.ones(T, dtype=bool)
    accel_info: list[dict] = []

    for start, end in slow_runs:
        run_len = end - start + 1

        # Decide subsampling ratio
        # Mean displacement in this run
        d_start = max(0, start)
        d_end = min(len(displacements), end + 1)
        if d_start < d_end:
            mean_disp = displacements[d_start:d_end].mean()
        else:
            mean_disp = 0.0

        if mean_disp <= 0 or target_step_size <= 0:
            continue

        subsample_ratio = max(1.0, target_step_size / mean_disp)
        keep_every = int(round(subsample_ratio))
        if keep_every <= 1:
            continue

        # Build indices to keep within this run, respecting protection
        run_indices = list(range(start, end + 1))
        new_keep_in_run: list[int] = []
        count = 0
        for idx in run_indices:
            if idx in protected:
                # Always keep protected frames
                new_keep_in_run.append(idx)
                count = 0  # Reset counter so next non-protected frame counts correctly
            elif count % keep_every == 0:
                new_keep_in_run.append(idx)
                count += 1
            else:
                keep[idx] = False
                count += 1

        # Always keep first and last frame of run for continuity
        keep[start] = True
        keep[end] = True

        removed = run_len - int(keep[start:end + 1].sum())
        accel_info.append({
            "start": int(start), "end": int(end),
            "run_len": run_len,
            "mean_disp": round(float(mean_disp), 8),
            "subsample_ratio": round(float(subsample_ratio), 2),
            "keep_every": keep_every,
            "removed": removed,
        })

    keep_indices = np.where(keep)[0]

    # Subsample all temporal arrays
    TEMPORAL_KEYS = [
        "images", "ee_pose", "obj_pose", "action", "obs",
        "gripper", "fsm_state", "wrist_images",
    ]
    result: dict = {}
    for key, val in episode.items():
        if key in TEMPORAL_KEYS and hasattr(val, "__len__") and len(val) == T:
            result[key] = val[keep_indices].copy()
        elif hasattr(val, "copy"):
            result[key] = val.copy()
        else:
            result[key] = val

    info = {
        "T_before_accel": T,
        "T_after_accel": len(keep_indices),
        "slow_runs": len(slow_runs),
        "frames_removed_accel": int((~keep).sum()),
        "accel_details": accel_info,
        "keep_indices_accel": keep_indices.tolist(),
    }
    return result, info


def decelerate_gripper_region(
    episode: dict,
    gripper_transitions: np.ndarray,
    gripper_slow_radius: int,
    interp_factor: int,
) -> tuple[dict, dict]:
    """Decelerate (insert interpolated frames) around gripper transitions.

    For frames within gripper_slow_radius of a transition, insert
    (interp_factor - 1) interpolated frames between each consecutive pair.
    """
    ee_pose = episode["ee_pose"]
    T = len(ee_pose)

    if len(gripper_transitions) == 0 or interp_factor <= 1:
        return episode, {"T_before_decel": T, "T_after_decel": T,
                         "interp_regions": [], "frames_added": 0}

    # Find which frames are in deceleration zones
    decel_set: set[int] = set()
    for gc in gripper_transitions:
        for f in range(max(0, gc - gripper_slow_radius),
                       min(T, gc + gripper_slow_radius + 1)):
            decel_set.add(f)

    TEMPORAL_KEYS = [
        "images", "ee_pose", "obj_pose", "action", "obs",
        "gripper", "fsm_state", "wrist_images",
    ]

    # Build interpolated data
    all_arrays: dict[str, list] = {key: [] for key in TEMPORAL_KEYS if key in episode}
    interp_regions: list[dict] = []
    frames_added = 0

    # Keys that should be linearly interpolated
    INTERP_KEYS = {"ee_pose", "obj_pose"}
    # Keys that use nearest-neighbor (repeat previous frame)
    NN_KEYS = {"images", "wrist_images", "obs", "gripper", "fsm_state", "action"}

    for t in range(T):
        # Always add the current frame
        for key in all_arrays:
            all_arrays[key].append(episode[key][t])

        # If this frame and the next are both in decel zone, interpolate between them
        if t < T - 1 and t in decel_set and (t + 1) in decel_set:
            n_insert = interp_factor - 1
            for k in range(1, n_insert + 1):
                alpha = k / interp_factor
                for key in all_arrays:
                    if key in INTERP_KEYS:
                        # Linear interpolation
                        val = (1 - alpha) * episode[key][t] + alpha * episode[key][t + 1]
                        all_arrays[key].append(val.astype(episode[key].dtype))
                    else:
                        # Nearest neighbor: use current frame's value
                        all_arrays[key].append(episode[key][t].copy())
            frames_added += n_insert

    # Rebuild episode
    result: dict = {}
    for key, val_list in all_arrays.items():
        result[key] = np.array(val_list)
    # Copy non-temporal keys
    for key, val in episode.items():
        if key not in result:
            result[key] = val.copy() if hasattr(val, "copy") else val

    info = {
        "T_before_decel": T,
        "T_after_decel": len(result["ee_pose"]),
        "frames_added": frames_added,
        "gripper_transitions": gripper_transitions.tolist(),
        "decel_zone_size": len(decel_set),
    }
    return result, info


def recompute_actions(ep: dict) -> None:
    """Recompute action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]."""
    ee = ep["ee_pose"]
    T = len(ee)
    actions = np.zeros((T, 8), dtype=np.float32)
    actions[:T - 1, :7] = ee[1:]
    actions[T - 1, :7] = ee[T - 1]

    if "gripper" in ep:
        actions[:, 7] = ep["gripper"].flatten()[:T]
    elif "action" in ep:
        actions[:, 7] = ep["action"][:, 7]

    ep["action"] = actions


def speed_adjust_episode(
    episode: dict,
    speed_threshold: float,
    target_step_size: float,
    min_slow_frames: int,
    gripper_slow_radius: int,
    interp_factor: int,
) -> tuple[dict, dict]:
    """Apply full speed adjustment pipeline to one episode."""
    T_original = len(episode["ee_pose"])

    # Step 1: Find gripper transitions (on original data)
    gripper_transitions = find_gripper_transitions(episode)

    # Step 2: Accelerate slow segments (protect gripper regions)
    accel_ep, accel_info = accelerate_slow_segments(
        episode, speed_threshold, target_step_size, min_slow_frames,
        gripper_transitions, gripper_slow_radius,
    )

    # Step 3: Find gripper transitions on accelerated data
    gripper_transitions_accel = find_gripper_transitions(accel_ep)

    # Step 4: Decelerate around gripper transitions
    final_ep, decel_info = decelerate_gripper_region(
        accel_ep, gripper_transitions_accel, gripper_slow_radius, interp_factor,
    )

    # Step 5: Recompute actions
    recompute_actions(final_ep)

    info = {
        "T_original": T_original,
        "T_after_accel": accel_info["T_after_accel"],
        "T_final": len(final_ep["ee_pose"]),
        "frames_removed_accel": accel_info["frames_removed_accel"],
        "frames_added_decel": decel_info["frames_added"],
        "gripper_transitions_original": gripper_transitions.tolist(),
        "accel_info": accel_info,
        "decel_info": decel_info,
    }
    return final_ep, info


# ── Debug video rendering ────────────────────────────────────────────────

def _write_video_h264(frames: list[np.ndarray], out_path: str | Path,
                      fps: int = 20) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    H, W = frames[0].shape[:2]
    H2 = H if H % 2 == 0 else H + 1
    W2 = W if W % 2 == 0 else W + 1
    adjusted = []
    for f in frames:
        if f.shape[0] != H2 or f.shape[1] != W2:
            f = cv2.resize(f, (W2, H2))
        adjusted.append(f)
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p") as w:
        for f in adjusted:
            w.append_data(f)
    print(f"  Video saved: {out_path} ({len(frames)} frames, {fps} fps)")


def _add_text(img: np.ndarray, text: str, position: tuple[int, int],
              font_scale: float = 0.4, color: tuple = (255, 255, 255),
              bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)
    return img


def render_debug_video(
    original_ep: dict,
    adjusted_ep: dict,
    info: dict,
    ep_idx: int,
    out_dir: Path,
    fps: int = 20,
) -> None:
    """Render a side-by-side comparison video for one episode."""
    scale = 2
    images = adjusted_ep["images"]
    ee_pose = adjusted_ep["ee_pose"]
    T = len(images)
    displacements = compute_displacements(ee_pose) if T > 1 else np.array([0.0])

    gripper = adjusted_ep.get("gripper", adjusted_ep["action"][:, 7])
    gripper_flat = gripper.flatten()[:T]

    frames = []
    for t in range(T):
        img = images[t]
        H, W = img.shape[:2]
        img = cv2.resize(img, (W * scale, H * scale),
                         interpolation=cv2.INTER_NEAREST)

        disp = displacements[t] if t < T - 1 else displacements[-1] if len(displacements) > 0 else 0
        ee = ee_pose[t]
        grip_state = "OPEN" if gripper_flat[t] > 0 else "CLOSED"

        lines = [
            f"Ep{ep_idx} F{t}/{T-1} | Orig:{info['T_original']}→{T}",
            f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
            f"Disp: {disp:.5f} m | Grip: {grip_state}",
            f"Accel-rm:{info['frames_removed_accel']} Decel+{info['frames_added_decel']}",
        ]
        for i, text in enumerate(lines):
            img = _add_text(img, text, (6, 14 + i * 16), font_scale=0.38)
        frames.append(img)

    _write_video_h264(frames, out_dir / f"ep{ep_idx}_speed_adjusted.mp4", fps)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t0 = datetime.datetime.now()
    print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] Speed-adjust episodes start")
    print(f"  Input:              {args.input}")
    print(f"  Output:             {args.out}")
    print(f"  speed_threshold:    {args.speed_threshold}")
    print(f"  target_step_size:   {args.target_step_size}")
    print(f"  min_slow_frames:    {args.min_slow_frames}")
    print(f"  gripper_slow_radius:{args.gripper_slow_radius}")
    print(f"  interp_factor:      {args.interp_factor}")
    print(f"  debug:              {args.debug}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return

    print("Loading episodes...")
    with np.load(args.input, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes")

    if args.success_only:
        n_before = len(episodes)
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful (dropped {n_before - len(episodes)})")

    if not episodes:
        print("ERROR: No episodes to process!")
        return

    # Debug setup
    debug_dir = None
    debug_episode_set: set[int] = set()
    if args.debug:
        debug_dir = Path(args.debug_dir) if args.debug_dir else Path(args.out).parent / "debug_videos"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_episode_set = set(args.debug_episodes) if args.debug_episodes else set(range(min(5, len(episodes))))
        print(f"  Debug videos → {debug_dir}")
        print(f"  Debug episodes: {sorted(debug_episode_set)}")

    # Process
    adjusted_episodes: list[dict] = []
    all_infos: list[dict] = []
    total_orig = 0
    total_adjusted = 0

    for i, ep in enumerate(episodes):
        adjusted_ep, info = speed_adjust_episode(
            ep,
            speed_threshold=args.speed_threshold,
            target_step_size=args.target_step_size,
            min_slow_frames=args.min_slow_frames,
            gripper_slow_radius=args.gripper_slow_radius,
            interp_factor=args.interp_factor,
        )
        adjusted_episodes.append(adjusted_ep)
        all_infos.append(info)

        total_orig += info["T_original"]
        total_adjusted += info["T_final"]

        # Debug video
        if args.debug and i in debug_episode_set:
            print(f"  Rendering debug video for ep {i}...")
            render_debug_video(ep, adjusted_ep, info, i, debug_dir, args.video_fps)

        if i < 5 or (i + 1) % 50 == 0 or i == len(episodes) - 1:
            print(f"  ep {i:4d}: {info['T_original']:5d} → {info['T_final']:5d} "
                  f"(accel-rm:{info['frames_removed_accel']:3d}, "
                  f"decel+{info['frames_added_decel']:3d}) "
                  f"grip_trans={len(info['gripper_transitions_original'])}")

    # Summary
    compression = total_orig / total_adjusted if total_adjusted else 0
    avg_orig = total_orig / len(episodes)
    avg_adj = total_adjusted / len(adjusted_episodes)
    lens = [len(ep["images"]) for ep in adjusted_episodes]

    print(f"\n{'=' * 70}")
    print(f"Speed Adjust Statistics")
    print(f"{'=' * 70}")
    print(f"Input episodes:          {len(episodes)}")
    print(f"Output episodes:         {len(adjusted_episodes)}")
    print(f"Original total frames:   {total_orig}")
    print(f"Adjusted total frames:   {total_adjusted}")
    print(f"Net frames change:       {total_adjusted - total_orig:+d}")
    print(f"Compression ratio:       {compression:.2f}x")
    print(f"Average orig length:     {avg_orig:.1f}")
    print(f"Average adj length:      {avg_adj:.1f}")
    print(f"Min/Max adj length:      {min(lens)}/{max(lens)}")
    print(f"{'=' * 70}")

    # Verify actions
    print("Verifying action consistency...")
    all_ok = True
    for i, ep in enumerate(adjusted_episodes):
        ee = ep["ee_pose"]
        act = ep["action"]
        T = len(ee)
        if T < 2:
            continue
        diff = np.linalg.norm(act[:T - 1, :7] - ee[1:], axis=1)
        max_diff = diff.max()
        if max_diff >= 1e-6:
            print(f"  MISMATCH ep {i}: max_diff={max_diff:.8f}")
            all_ok = False
    if all_ok:
        print(f"  All {len(adjusted_episodes)} episodes verified OK")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=adjusted_episodes)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path} ({file_mb:.1f} MB)")

    # Save stats
    stats_path = out_path.with_suffix(".stats.json")
    stats = {
        "source": str(args.input),
        "speed_threshold": args.speed_threshold,
        "target_step_size": args.target_step_size,
        "min_slow_frames": args.min_slow_frames,
        "gripper_slow_radius": args.gripper_slow_radius,
        "interp_factor": args.interp_factor,
        "num_input_episodes": len(episodes),
        "num_output_episodes": len(adjusted_episodes),
        "original_total_frames": total_orig,
        "adjusted_total_frames": total_adjusted,
        "net_change": total_adjusted - total_orig,
        "compression_ratio": round(compression, 4),
        "avg_original_length": round(avg_orig, 1),
        "avg_adjusted_length": round(avg_adj, 1),
        "min_adjusted_length": min(lens),
        "max_adjusted_length": max(lens),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: {stats_path}")

    # Save debug stats
    if args.debug and debug_dir:
        debug_stats = {
            "config": vars(args),
            "global_stats": stats,
            "per_episode": [{k: v for k, v in inf.items()
                            if k not in ("keep_indices_accel",)}
                           for inf in all_infos],
        }
        debug_stats_path = debug_dir / "speed_adjust_debug_stats.json"
        with open(debug_stats_path, "w") as f:
            json.dump(debug_stats, f, indent=2)
        print(f"Debug stats: {debug_stats_path}")

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
