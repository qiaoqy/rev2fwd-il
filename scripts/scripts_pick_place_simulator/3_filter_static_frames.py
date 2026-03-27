#!/usr/bin/env python3
"""Filter out static (non-moving) segments from episode trajectories.

Uses a two-pass hybrid algorithm:
1. **Fast pass** – per-frame EE displacement flags frames as "static" when the
   step is below ``velocity_threshold``.
2. **Spatial verification** – each candidate run of >= ``min_static_frames``
   consecutive static frames is verified by computing the spatial diameter
   (max pairwise EE distance within the run).  Only runs whose diameter is
   below ``spatial_threshold`` are actually removed.  This prevents deleting
   slow-but-purposeful drift segments.

Removed runs keep a small boundary buffer on each side, and frames near
gripper state transitions are always protected.  After filtering, remaining
frames are split into contiguous segments – each segment becomes a separate
episode to avoid temporal discontinuities.  Actions are recomputed within
each segment so that action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t].

An optional **debug mode** renders annotated MP4 videos showing which frames
were kept/removed and overlays pose information.

Usage:
    conda activate rev2fwd_il

    # Basic filtering
    python scripts/scripts_pick_place_simulator/3_filter_static_frames.py \\
        --input data/exp21/rollout_A_200.npz \\
        --out data/exp21/rollout_A_200_filtered.npz \\
        --velocity_threshold 1e-3 \\
        --min_static_frames 16

    # Debug mode (render before/after videos)
    python scripts/scripts_pick_place_simulator/3_filter_static_frames.py \\
        --input data/exp21/rollout_A_200.npz \\
        --out data/exp21/rollout_A_200_filtered.npz \\
        --debug --debug_dir data/exp21/debug_filter_videos \\
        --debug_episodes 0 1 2 3 4
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
        description="Filter static (non-moving) segments from episode data.",
    )
    # I/O
    p.add_argument("--input", type=str, required=True, help="Input NPZ file.")
    p.add_argument("--out", type=str, required=True, help="Output NPZ file.")
    p.add_argument("--success_only", type=int, default=0, choices=[0, 1],
                   help="Only keep successful episodes (default: 0).")

    # Filter parameters
    p.add_argument("--velocity_threshold", type=float, default=1e-3,
                   help="EE displacement threshold (m) below which a frame is "
                        "'static'. Default: 1e-3 (1 mm/frame).")
    p.add_argument("--min_static_frames", type=int, default=16,
                   help="Minimum consecutive static frames to trigger removal. "
                        "Default: 16.")
    p.add_argument("--gripper_protect_radius", type=int, default=8,
                   help="Protect frames within this many frames of a gripper "
                        "state change (won't be removed). Default: 8.")
    p.add_argument("--spatial_threshold", type=float, default=None,
                   help="Max spatial diameter (m) for a static run to be "
                        "removed. If the furthest two EE positions within a "
                        "candidate run exceed this, the run is kept (slow "
                        "drift, not truly static). Default: 8 * velocity_threshold.")
    p.add_argument("--min_segment_length", type=int, default=16,
                   help="Discard segments shorter than this after splitting. "
                        "Default: 16.")

    # Debug / video
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode: render annotated videos.")
    p.add_argument("--debug_dir", type=str, default=None,
                   help="Output directory for debug videos (required if --debug).")
    p.add_argument("--debug_episodes", type=int, nargs="+", default=None,
                   help="Episode indices to render in debug mode (default: first 5).")
    p.add_argument("--video_fps", type=int, default=20,
                   help="FPS for debug videos. Default: 20.")
    return p.parse_args()


# ── Core filtering logic ─────────────────────────────────────────────────

def compute_displacements(ee_pose: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame Euclidean displacement of EE position (xyz).

    Returns array of length T-1 where element i is ||ee[i+1,:3] - ee[i,:3]||.
    """
    return np.linalg.norm(np.diff(ee_pose[:, :3], axis=0), axis=1)


def compute_spatial_diameter(ee_pose_segment: np.ndarray) -> float:
    """Max pairwise Euclidean distance (xyz) within an EE pose segment."""
    xyz = ee_pose_segment[:, :3]
    n = len(xyz)
    if n <= 1:
        return 0.0
    # For typical segment lengths (~16-100 frames) full pairwise is fine.
    # Use broadcasting: (n,1,3) - (1,n,3) -> (n,n,3)
    diffs = xyz[:, None, :] - xyz[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (n, n)
    return float(dists.max())


def find_static_runs(is_static: np.ndarray) -> list[tuple[int, int]]:
    """Find consecutive runs of True values. Returns list of (start, end) inclusive."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for t, s in enumerate(is_static):
        if s:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None:
                runs.append((run_start, t - 1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(is_static) - 1))
    return runs


def find_contiguous_segments(keep_indices: np.ndarray) -> list[np.ndarray]:
    """Split sorted keep-indices into contiguous segments.

    Each segment is a group of consecutive original frame indices.
    Gaps in keep_indices (where removed frames were) become split points,
    so each segment can become a separate episode without temporal jumps.
    """
    if len(keep_indices) == 0:
        return []
    breaks = np.where(np.diff(keep_indices) > 1)[0] + 1
    return list(np.split(keep_indices, breaks))


def build_removal_mask(
    episode: dict,
    velocity_threshold: float,
    min_static_frames: int,
    gripper_protect_radius: int,
    spatial_threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a boolean mask indicating which frames to *remove*.

    Two-pass hybrid algorithm:
    1. Per-frame displacement flags candidate static frames.
    2. Each long candidate run is verified via spatial diameter.

    Returns:
        remove_mask: bool array of shape (T,), True = remove this frame
        info: dict with diagnostic info (static runs, protected frames, etc.)
    """
    if spatial_threshold is None:
        spatial_threshold = 8.0 * velocity_threshold

    ee_pose = episode["ee_pose"]
    T = len(ee_pose)

    # 1. Compute per-frame displacement
    displacements = compute_displacements(ee_pose)  # (T-1,)

    # 2. Mark each frame as static/moving
    #    Frame t is static if the step t→t+1 is below threshold.
    #    Last frame inherits status from frame T-2.
    is_static = np.zeros(T, dtype=bool)
    is_static[:-1] = displacements < velocity_threshold
    if T > 1:
        is_static[-1] = is_static[-2]

    # 3. Find consecutive static runs
    runs = find_static_runs(is_static)

    # 4. Mark frames for removal in long static runs (with spatial verification)
    remove_mask = np.zeros(T, dtype=bool)
    removed_runs: list[dict] = []
    skipped_runs: list[dict] = []  # runs that failed spatial check

    for start, end in runs:
        run_len = end - start + 1
        if run_len >= min_static_frames:
            # Spatial verification: check if the run is truly static
            diameter = compute_spatial_diameter(ee_pose[start:end + 1])
            if diameter > spatial_threshold:
                # Slow drift, not truly static — keep these frames
                skipped_runs.append({
                    "start": int(start), "end": int(end),
                    "run_len": run_len,
                    "diameter": round(float(diameter), 8),
                    "reason": "spatial_diameter_exceeded",
                })
                continue

            # Remove the entire static run
            remove_mask[start:end + 1] = True
            removed_runs.append({
                "start": int(start), "end": int(end),
                "run_len": run_len,
                "diameter": round(float(diameter), 8),
                "removed_count": run_len,
            })

    # 5. Protect frames near gripper state transitions
    gripper = episode.get("gripper", episode["action"][:, 7])
    gripper_flat = gripper.flatten()[:T]
    gripper_changes = np.where(np.abs(np.diff(gripper_flat)) > 0.5)[0]

    protected_frames: list[int] = []
    for gc in gripper_changes:
        prot_start = max(0, gc - gripper_protect_radius)
        prot_end = min(T - 1, gc + gripper_protect_radius)
        for f in range(prot_start, prot_end + 1):
            if remove_mask[f]:
                remove_mask[f] = False
                protected_frames.append(int(f))

    info = {
        "T_original": T,
        "num_static_frames": int(is_static.sum()),
        "static_runs_total": len(runs),
        "static_runs_long": len(removed_runs),
        "static_runs_skipped": len(skipped_runs),
        "removed_runs": removed_runs,
        "skipped_runs": skipped_runs,
        "spatial_threshold": spatial_threshold,
        "gripper_change_frames": gripper_changes.tolist(),
        "protected_frames": sorted(set(protected_frames)),
        "num_removed": int(remove_mask.sum()),
        "num_kept": int((~remove_mask).sum()),
    }
    return remove_mask, info


def filter_episode(
    episode: dict,
    velocity_threshold: float,
    min_static_frames: int,
    gripper_protect_radius: int,
    spatial_threshold: float | None = None,
) -> tuple[dict, dict]:
    """Filter static frames from an episode.

    Returns:
        filtered_episode: new dict with static frames removed
        info: diagnostic dict
    """
    remove_mask, info = build_removal_mask(
        episode, velocity_threshold, min_static_frames,
        gripper_protect_radius, spatial_threshold,
    )

    T = len(episode["images"])
    keep_indices = np.where(~remove_mask)[0]
    info["keep_indices"] = keep_indices.tolist()

    # Subsample all time-indexed arrays
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

    # Recompute actions
    _recompute_actions(result)
    return result, info


def _recompute_actions(ep: dict) -> None:
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
    """Add text with background rectangle."""
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


def _add_border(img: np.ndarray, color: tuple, width: int = 4) -> np.ndarray:
    """Add a colored border around the image."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                  color, width)
    return img


def render_debug_videos(
    episode: dict,
    remove_mask: np.ndarray,
    info: dict,
    ep_idx: int,
    out_dir: Path,
    fps: int = 20,
) -> None:
    """Render debug videos for one episode: original (annotated) + filtered."""
    ee_pose = episode["ee_pose"]
    T = len(episode["images"])
    displacements = compute_displacements(ee_pose)

    gripper = episode.get("gripper", episode["action"][:, 7])
    gripper_flat = gripper.flatten()[:T]
    gripper_change_set = set(info.get("gripper_change_frames", []))
    protected_set = set(info.get("protected_frames", []))

    scale = 2  # upscale for readability

    # ── Original video with annotations ──
    original_frames = []
    for t in range(T):
        img = episode["images"][t]
        H, W = img.shape[:2]
        img = cv2.resize(img, (W * scale, H * scale),
                         interpolation=cv2.INTER_NEAREST)

        # Border: red = removed, green = kept, yellow = protected
        if t in protected_set:
            img = _add_border(img, (255, 255, 0), width=4)
        elif remove_mask[t]:
            img = _add_border(img, (255, 0, 0), width=4)
        else:
            img = _add_border(img, (0, 255, 0), width=3)

        # Text overlay
        disp = displacements[t] if t < T - 1 else displacements[-1]
        ee = ee_pose[t]
        grip_state = "OPEN" if gripper_flat[t] > 0 else "CLOSED"
        status = "REMOVE" if remove_mask[t] else ("PROTECT" if t in protected_set else "KEEP")

        lines = [
            f"Ep {ep_idx} | Frame {t}/{T-1} | {status}",
            f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
            f"Disp: {disp:.6f} m | Grip: {grip_state}",
        ]
        if t in gripper_change_set:
            lines.append("** GRIPPER TRANSITION **")

        for i, text in enumerate(lines):
            img = _add_text(img, text, (6, 14 + i * 16), font_scale=0.38)

        original_frames.append(img)

    _write_video_h264(original_frames, out_dir / f"ep{ep_idx}_original.mp4", fps)

    # ── Filtered video (with segment separators) ──
    keep_indices = np.where(~remove_mask)[0]
    segments = find_contiguous_segments(keep_indices)
    filtered_frames = []
    for seg_idx, seg_indices in enumerate(segments):
        # Add separator frame between segments
        if seg_idx > 0:
            H0, W0 = episode["images"][0].shape[:2]
            sep = np.zeros((H0 * scale, W0 * scale, 3), dtype=np.uint8)
            sep = _add_text(sep, f"=== SEGMENT {seg_idx} ===",
                            (6, H0 * scale // 2), font_scale=0.5,
                            color=(255, 255, 0))
            filtered_frames.append(sep)

        for local_t, orig_t in enumerate(seg_indices):
            img = episode["images"][orig_t]
            H, W = img.shape[:2]
            img = cv2.resize(img, (W * scale, H * scale),
                             interpolation=cv2.INTER_NEAREST)

            disp = displacements[orig_t] if orig_t < T - 1 else displacements[-1]
            ee = ee_pose[orig_t]
            grip_state = "OPEN" if gripper_flat[orig_t] > 0 else "CLOSED"

            lines = [
                f"Ep {ep_idx} Seg{seg_idx} | F{local_t}/{len(seg_indices)-1} (Orig {orig_t})",
                f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
                f"Disp: {disp:.6f} m | Grip: {grip_state}",
            ]

            for i, text in enumerate(lines):
                img = _add_text(img, text, (6, 14 + i * 16), font_scale=0.38)

            filtered_frames.append(img)

    _write_video_h264(filtered_frames, out_dir / f"ep{ep_idx}_filtered.mp4", fps)

    # Wrist camera videos if available
    if "wrist_images" in episode:
        wrist_orig_frames = []
        for t in range(T):
            img = episode["wrist_images"][t]
            H, W = img.shape[:2]
            img = cv2.resize(img, (W * scale, H * scale),
                             interpolation=cv2.INTER_NEAREST)
            if remove_mask[t]:
                img = _add_border(img, (255, 0, 0), width=4)
            else:
                img = _add_border(img, (0, 255, 0), width=3)
            img = _add_text(img, f"WRIST Ep{ep_idx} F{t} {'RM' if remove_mask[t] else 'OK'}",
                            (6, 14), font_scale=0.35)
            wrist_orig_frames.append(img)
        _write_video_h264(wrist_orig_frames, out_dir / f"ep{ep_idx}_wrist_original.mp4", fps)

        wrist_filt_frames = []
        for seg_idx, seg_indices in enumerate(segments):
            if seg_idx > 0:
                H0, W0 = episode["wrist_images"][0].shape[:2]
                sep = np.zeros((H0 * scale, W0 * scale, 3), dtype=np.uint8)
                sep = _add_text(sep, f"=== SEGMENT {seg_idx} ===",
                                (6, H0 * scale // 2), font_scale=0.5,
                                color=(255, 255, 0))
                wrist_filt_frames.append(sep)
            for local_t, orig_t in enumerate(seg_indices):
                img = episode["wrist_images"][orig_t]
                H, W = img.shape[:2]
                img = cv2.resize(img, (W * scale, H * scale),
                                 interpolation=cv2.INTER_NEAREST)
                img = _add_text(img, f"WRIST Ep{ep_idx} Seg{seg_idx} F{local_t} (Orig{orig_t})",
                                (6, 14), font_scale=0.35)
                wrist_filt_frames.append(img)
        _write_video_h264(wrist_filt_frames, out_dir / f"ep{ep_idx}_wrist_filtered.mp4", fps)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t0 = datetime.datetime.now()
    print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] Filter static frames start")
    # Resolve spatial_threshold default
    if args.spatial_threshold is None:
        args.spatial_threshold = 8.0 * args.velocity_threshold

    print(f"  Input:                 {args.input}")
    print(f"  Output:                {args.out}")
    print(f"  velocity_threshold:    {args.velocity_threshold}")
    print(f"  min_static_frames:     {args.min_static_frames}")
    print(f"  gripper_protect_radius:{args.gripper_protect_radius}")
    print(f"  spatial_threshold:     {args.spatial_threshold}")
    print(f"  min_segment_length:    {args.min_segment_length}")
    print(f"  debug:                 {args.debug}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return
    input_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"  Input size: {input_mb:.1f} MB")

    # Load
    print("Loading episodes...")
    with np.load(args.input, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes")

    if args.success_only:
        n_before = len(episodes)
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful episodes "
              f"(dropped {n_before - len(episodes)})")

    if not episodes:
        print("ERROR: No episodes to process!")
        return

    # Debug setup
    debug_dir = None
    debug_episode_set: set[int] = set()
    if args.debug:
        if args.debug_dir is None:
            debug_dir = Path(args.out).parent / "debug_filter_videos"
        else:
            debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        if args.debug_episodes is not None:
            debug_episode_set = set(args.debug_episodes)
        else:
            debug_episode_set = set(range(min(5, len(episodes))))
        print(f"  Debug videos → {debug_dir}")
        print(f"  Debug episodes: {sorted(debug_episode_set)}")

    # Process
    filtered_episodes: list[dict] = []
    all_infos: list[dict] = []
    total_orig = 0
    total_filtered = 0
    total_segments = 0

    TEMPORAL_KEYS = [
        "images", "ee_pose", "obj_pose", "action", "obs",
        "gripper", "fsm_state", "wrist_images",
    ]

    for i, ep in enumerate(episodes):
        orig_len = len(ep["images"])

        # Build removal mask (but don't filter yet, so debug can use original)
        remove_mask, info = build_removal_mask(
            ep, args.velocity_threshold, args.min_static_frames,
            args.gripper_protect_radius,
            args.spatial_threshold,
        )

        # Debug rendering (uses original episode + mask)
        if args.debug and i in debug_episode_set:
            print(f"  Rendering debug videos for ep {i}...")
            render_debug_videos(ep, remove_mask, info, i, debug_dir, args.video_fps)

        # Split kept frames into contiguous segments (each becomes a separate episode)
        keep_indices = np.where(~remove_mask)[0]
        all_segments = find_contiguous_segments(keep_indices)
        # Drop segments shorter than min_segment_length
        segments = [s for s in all_segments if len(s) >= args.min_segment_length]
        dropped_segs = len(all_segments) - len(segments)

        ep_filtered_frames = 0
        for seg_indices in segments:
            T = orig_len
            result: dict = {}
            for key, val in ep.items():
                if key in TEMPORAL_KEYS and hasattr(val, "__len__") and len(val) == T:
                    result[key] = val[seg_indices].copy()
                elif hasattr(val, "copy"):
                    result[key] = val.copy()
                else:
                    result[key] = val
            _recompute_actions(result)
            filtered_episodes.append(result)
            ep_filtered_frames += len(seg_indices)

        # Add segment info to diagnostics
        info["num_segments"] = len(segments)
        info["dropped_short_segments"] = dropped_segs
        info["segment_lengths"] = [len(s) for s in segments]
        info["retention_pct"] = round(100 * ep_filtered_frames / orig_len, 2) if orig_len else 0

        # Don't store keep_indices in info for serialisation (can be large)
        info_for_json = {k: v for k, v in info.items() if k != "keep_indices"}
        all_infos.append(info_for_json)

        total_orig += orig_len
        total_filtered += ep_filtered_frames
        total_segments += len(segments)

        if i < 5 or (i + 1) % 20 == 0 or i == len(episodes) - 1:
            pct = 100 * info["num_removed"] / orig_len if orig_len else 0
            seg_lens = ",".join(str(l) for l in info["segment_lengths"])
            drop_str = f" drop={dropped_segs}" if dropped_segs else ""
            print(f"  ep {i:3d}: {orig_len:5d} → {ep_filtered_frames:4d} frames  "
                  f"(removed {orig_len - ep_filtered_frames:3d}, {pct:.1f}%)  "
                  f"segs={info['num_segments']} [{seg_lens}]{drop_str}  "
                  f"runs_removed={info['static_runs_long']}  "
                  f"runs_skipped={info['static_runs_skipped']}  "
                  f"gripper_changes={len(info['gripper_change_frames'])}")

    # Summary statistics
    avg_orig = total_orig / len(episodes) if episodes else 0
    avg_filt = total_filtered / len(filtered_episodes) if filtered_episodes else 0
    compression = total_orig / total_filtered if total_filtered else 0
    retention_pct = 100 * total_filtered / total_orig if total_orig else 0
    lens = [len(ep["images"]) for ep in filtered_episodes]

    print(f"\n{'=' * 70}")
    print(f"Filter Statistics")
    print(f"{'=' * 70}")
    print(f"Input episodes:           {len(episodes)}")
    print(f"Output episodes (segs):   {len(filtered_episodes)}")
    print(f"Total segments:           {total_segments}")
    print(f"velocity_threshold:       {args.velocity_threshold}")
    print(f"spatial_threshold:        {args.spatial_threshold}")
    print(f"min_static_frames:        {args.min_static_frames}")
    print(f"gripper_protect_radius:   {args.gripper_protect_radius}")
    print(f"min_segment_length:       {args.min_segment_length}")
    print(f"Original total frames:    {total_orig}")
    print(f"Filtered total frames:    {total_filtered}")
    print(f"Frames removed:           {total_orig - total_filtered}")
    print(f"Retention:                {retention_pct:.1f}%")
    print(f"Compression ratio:        {compression:.2f}x")
    print(f"Original avg length:      {avg_orig:.1f}")
    print(f"Filtered avg seg length:  {avg_filt:.1f}")
    print(f"Filtered min/max seg len: {min(lens)}/{max(lens)}")
    print(f"{'=' * 70}\n")

    # Verify actions
    print("Verifying action[t][:7] == ee_pose[t+1] for ALL episodes...")
    all_ok = True
    for i, ep in enumerate(filtered_episodes):
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
        elif i < 5:
            print(f"  ep {i}: max_diff={max_diff:.8f} [OK]")
    if all_ok:
        print(f"  All {len(filtered_episodes)} episodes verified OK")
    else:
        print("  WARNING: Some episodes have action mismatches!")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=filtered_episodes)
    file_mb = out_path.stat().st_size / (1024 * 1024)

    # Save stats JSON
    stats_path = out_path.with_suffix(".stats.json")
    stats = {
        "source": str(args.input),
        "velocity_threshold": args.velocity_threshold,
        "spatial_threshold": args.spatial_threshold,
        "min_static_frames": args.min_static_frames,
        "gripper_protect_radius": args.gripper_protect_radius,
        "num_input_episodes": len(episodes),
        "num_output_episodes": len(filtered_episodes),
        "num_episodes": len(filtered_episodes),
        "total_segments": total_segments,
        "min_segment_length": args.min_segment_length,
        "original_total_frames": total_orig,
        "filtered_total_frames": total_filtered,
        "frames_removed": total_orig - total_filtered,
        "retention_pct": round(retention_pct, 2),
        "compression_ratio": compression,
        "avg_original_length": avg_orig,
        "avg_filtered_length": avg_filt,
        "min_filtered_length": min(lens),
        "max_filtered_length": max(lens),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # Save debug info
    if args.debug and debug_dir:
        debug_stats_path = debug_dir / "filter_debug_stats.json"
        debug_data = {
            "config": {
                "velocity_threshold": args.velocity_threshold,
                "spatial_threshold": args.spatial_threshold,
                "min_static_frames": args.min_static_frames,
                "gripper_protect_radius": args.gripper_protect_radius,
            },
            "overall": {
                "num_input_episodes": len(episodes),
                "num_output_episodes": len(filtered_episodes),
                "total_segments": total_segments,
                "retention_pct": round(retention_pct, 2),
                "total_original_frames": total_orig,
                "total_filtered_frames": total_filtered,
            },
            "episodes": {
                str(i): all_infos[i]
                for i in sorted(debug_episode_set)
                if i < len(all_infos)
            },
        }
        with open(debug_stats_path, "w") as f:
            json.dump(debug_data, f, indent=2)
        print(f"Debug stats saved: {debug_stats_path}")

    t1 = datetime.datetime.now()
    elapsed = (t1 - t0).total_seconds()
    print(f"\nSaved {len(filtered_episodes)} episodes to {out_path}  ({file_mb:.1f} MB)")
    print(f"[{t1.strftime('%Y-%m-%d %H:%M:%S')}] Filter done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
