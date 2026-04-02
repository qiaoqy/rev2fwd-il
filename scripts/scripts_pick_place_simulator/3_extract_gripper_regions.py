#!/usr/bin/env python3
"""Extract only the two real gripper-action regions from pick-and-place episodes.

In a pick-and-place Task B rollout, the gripper performs exactly two actions:
  1. **Close** (pick): gripper transitions from open (~+1) to closed (~-1)
  2. **Open** (place): gripper transitions from closed (~-1) to open (~+1)

The raw gripper signal may contain noisy jitter (multiple threshold crossings
for a single physical action).  This script robustly identifies the two real
transitions by looking at **stable gripper states** rather than per-frame
changes, then extracts ±``gripper_region_radius`` frames around each.

Algorithm:
  1. Binarize: frame is "open" if gripper > +open_threshold, "closed" if
     gripper < -open_threshold, else "transitioning".
  2. Find the **first frame that enters the closed state** after being open
     → pick transition center.
  3. Find the **last frame that enters the open state** after being closed
     → place transition center.
  4. Extract [center - radius, center + radius] for each, clamp to [0, T-1].

Usage:
    python scripts/scripts_pick_place_simulator/3_extract_gripper_regions.py \\
        --input data/exp32/iter1_collect_B.npz \\
        --out data/exp32/iter1_collect_B_gripper_regions.npz \\
        --gripper_region_radius 16 \\
        --success_only 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract the two real gripper-action regions (pick & place) "
                    "from pick-and-place episode data.",
    )
    p.add_argument("--input", type=str, required=True, help="Input NPZ file.")
    p.add_argument("--out", type=str, required=True, help="Output NPZ file.")
    p.add_argument("--success_only", type=int, default=0, choices=[0, 1],
                   help="Only keep successful episodes (default: 0).")
    p.add_argument("--gripper_region_radius", type=int, default=16,
                   help="Number of frames before/after each gripper transition "
                        "to keep. Default: 16 (total ~32 frames per transition).")
    p.add_argument("--open_threshold", type=float, default=0.5,
                   help="Gripper value above +threshold = open, below "
                        "-threshold = closed. Default: 0.5.")
    p.add_argument("--min_region_length", type=int, default=8,
                   help="Discard extracted regions shorter than this. Default: 8.")
    return p.parse_args()


# ── Core logic ───────────────────────────────────────────────────────────

def find_pick_place_transitions(episode: dict,
                                open_threshold: float = 0.5,
                                ) -> tuple[int | None, int | None]:
    """Find the two real gripper transitions: pick (close) and place (open).

    Returns (pick_frame, place_frame) — the frame index at which the gripper
    first becomes stably closed (pick) and the frame at which it first becomes
    stably open again after the pick (place).  Returns None if not found.

    Strategy:
      - Binarize gripper into open (> +thr) / closed (< -thr) / transitioning.
      - Pick = first "closed" frame after we've seen an "open" frame (forward scan).
      - Place = first "open" frame after the pick that follows a "closed" frame.
    """
    gripper = episode.get("gripper", episode["action"][:, 7])
    g = gripper.flatten()[:len(episode["ee_pose"])]
    T = len(g)

    # Binarize: +1 = open, -1 = closed, 0 = transitioning
    state = np.zeros(T, dtype=np.int8)
    state[g > open_threshold] = 1    # open
    state[g < -open_threshold] = -1  # closed

    # --- Find PICK: first frame that enters closed after having been open ---
    pick_frame = None
    seen_open = False
    for t in range(T):
        if state[t] == 1:
            seen_open = True
        elif state[t] == -1 and seen_open:
            pick_frame = t
            break

    if pick_frame is None:
        return None, None

    # --- Find PLACE: first frame that enters open after pick, having been closed ---
    place_frame = None
    seen_closed = False
    for t in range(pick_frame, T):
        if state[t] == -1:
            seen_closed = True
        elif state[t] == 1 and seen_closed:
            place_frame = t
            break

    return pick_frame, place_frame


def extract_region(episode: dict, start: int, end: int) -> dict:
    """Extract frames [start, end] (inclusive) from episode as a new sub-episode."""
    T = len(episode["ee_pose"])
    sl = slice(start, end + 1)

    TEMPORAL_KEYS = [
        "images", "ee_pose", "obj_pose", "action", "obs",
        "gripper", "fsm_state", "wrist_images",
    ]

    result: dict = {}
    for key, val in episode.items():
        if key in TEMPORAL_KEYS and hasattr(val, "__len__") and len(val) == T:
            result[key] = val[sl].copy()
        elif isinstance(val, np.ndarray):
            result[key] = val.copy()
        else:
            result[key] = val

    return result


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


def extract_gripper_regions(
    episode: dict,
    gripper_region_radius: int,
    open_threshold: float,
    min_region_length: int,
) -> tuple[list[dict], dict]:
    """Extract the two real gripper-action regions from one episode.

    Returns:
        sub_episodes: list of extracted sub-episode dicts (up to 2)
        info: statistics about the extraction
    """
    T = len(episode["ee_pose"])
    pick_frame, place_frame = find_pick_place_transitions(episode, open_threshold)

    if pick_frame is None and place_frame is None:
        return [], {
            "T_original": T,
            "pick_frame": None,
            "place_frame": None,
            "num_sub_episodes": 0,
            "total_frames_kept": 0,
        }

    sub_episodes: list[dict] = []
    region_details: list[dict] = []

    for label, center in [("pick", pick_frame), ("place", place_frame)]:
        if center is None:
            region_details.append({"label": label, "center": None, "kept": False,
                                   "reason": "transition not found"})
            continue

        start = max(0, center - gripper_region_radius)
        end = min(T - 1, center + gripper_region_radius)
        region_len = end - start + 1

        if region_len < min_region_length:
            region_details.append({
                "label": label, "center": int(center),
                "start": start, "end": end,
                "length": region_len, "kept": False,
                "reason": f"too short ({region_len} < {min_region_length})",
            })
            continue

        sub_ep = extract_region(episode, start, end)
        recompute_actions(sub_ep)
        sub_episodes.append(sub_ep)

        region_details.append({
            "label": label, "center": int(center),
            "start": start, "end": end,
            "length": region_len, "kept": True,
        })

    total_kept = sum(d["length"] for d in region_details if d.get("kept", False))

    info = {
        "T_original": T,
        "pick_frame": int(pick_frame) if pick_frame is not None else None,
        "place_frame": int(place_frame) if place_frame is not None else None,
        "num_sub_episodes": len(sub_episodes),
        "total_frames_kept": total_kept,
        "compression_ratio": round(total_kept / T, 4) if T > 0 else 0,
        "region_details": region_details,
    }
    return sub_episodes, info


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}")
        return

    data = np.load(str(in_path), allow_pickle=True)
    episodes = list(data["episodes"])

    if args.success_only:
        before_filter = len(episodes)
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Success filter: {before_filter} → {len(episodes)} episodes")

    if not episodes:
        print(f"WARNING: No episodes to process in {in_path.name}")
        np.savez_compressed(args.out, episodes=np.array([], dtype=object))
        return

    all_sub_episodes: list[dict] = []
    all_infos: list[dict] = []
    total_frames_before = 0
    total_frames_after = 0

    for i, ep in enumerate(episodes):
        sub_eps, info = extract_gripper_regions(
            ep,
            gripper_region_radius=args.gripper_region_radius,
            open_threshold=args.open_threshold,
            min_region_length=args.min_region_length,
        )
        all_sub_episodes.extend(sub_eps)
        all_infos.append(info)
        total_frames_before += info["T_original"]
        total_frames_after += info["total_frames_kept"]

        if i < 5 or (i + 1) % 10 == 0:
            pick = info["pick_frame"]
            place = info["place_frame"]
            print(f"  Episode {i}: {info['T_original']} frames → "
                  f"{info['num_sub_episodes']} sub-episodes "
                  f"({info['total_frames_kept']} frames), "
                  f"pick@{pick}, place@{place}")

    print(f"\nSummary: {in_path.name}")
    print(f"  Input episodes: {len(episodes)}")
    print(f"  Output sub-episodes: {len(all_sub_episodes)}")
    print(f"  Total frames: {total_frames_before} → {total_frames_after} "
          f"({total_frames_after/total_frames_before*100:.1f}%)" if total_frames_before > 0 else "")
    print(f"  Parameters: radius={args.gripper_region_radius}, "
          f"open_threshold={args.open_threshold}, "
          f"min_length={args.min_region_length}")

    # Save
    np.savez_compressed(args.out,
                        episodes=np.array(all_sub_episodes, dtype=object))
    out_size = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"  Saved: {args.out} ({out_size:.1f} MB)")

    # Save stats alongside
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    stats_path = Path(args.out).with_suffix(".stats.json")
    stats = {
        "input": str(in_path),
        "output": str(args.out),
        "params": {
            "gripper_region_radius": args.gripper_region_radius,
            "open_threshold": args.open_threshold,
            "min_region_length": args.min_region_length,
            "success_only": bool(args.success_only),
        },
        "summary": {
            "input_episodes": len(episodes),
            "output_sub_episodes": len(all_sub_episodes),
            "total_frames_before": total_frames_before,
            "total_frames_after": total_frames_after,
            "compression_ratio": round(total_frames_after / total_frames_before, 4)
            if total_frames_before > 0 else 0,
        },
        "per_episode": all_infos,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=_json_default)
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
