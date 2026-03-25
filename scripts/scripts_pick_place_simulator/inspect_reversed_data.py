#!/usr/bin/env python3
"""Inspect and visualize reversed rollout data for exp20.

For each iteration, loads both the original rollout and its reversed version,
verifies correctness, and generates visualization videos + XYZ trajectory plots.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/inspect_reversed_data.py \
        --exp_dir data/pick_place_isaac_lab_simulation/exp20 \
        --episode 0
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect reversed data for exp20")
    parser.add_argument("--exp_dir", type=str, default="data/pick_place_isaac_lab_simulation/exp20")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--max_episodes", type=int, default=3, help="Max episodes to visualize per file")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def load_episodes(path: str) -> list[dict]:
    """Load episodes from npz file."""
    print(f"Loading {path} ...")
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"  → {len(episodes)} episodes loaded")
    return episodes


def verify_reversal(orig_eps: list[dict], rev_eps: list[dict], label: str) -> dict:
    """Verify that the reversed data is correct w.r.t. original data.
    
    Checks:
    1. action[t][:7] == ee_pose[t+1] in the reversed episode
    2. Images are correctly reversed (first frame of reversed == last frame of original)
    3. Episode lengths: reversed = original - 1 (due to last frame drop)
    """
    results = {
        "label": label,
        "num_orig": len(orig_eps),
        "num_rev": len(rev_eps),
        "checks": []
    }
    
    # Match episodes: reversed only contains successful ones
    orig_success = [ep for ep in orig_eps if ep.get("success", False)]
    
    print(f"\n{'='*60}")
    print(f"Verifying: {label}")
    print(f"{'='*60}")
    print(f"  Original: {len(orig_eps)} eps ({len(orig_success)} successful)")
    print(f"  Reversed: {len(rev_eps)} eps")
    
    if len(rev_eps) != len(orig_success):
        print(f"  WARNING: Expected {len(orig_success)} reversed eps, got {len(rev_eps)}")
    
    for i, rev_ep in enumerate(rev_eps):
        T_rev = len(rev_ep["images"])
        
        # Check action[t][:7] == ee_pose[t+1]
        ee = rev_ep["ee_pose"]
        act = rev_ep["action"]
        diff = np.linalg.norm(act[:T_rev-1, :7] - ee[1:], axis=1)
        max_diff = float(diff.max())
        action_ok = max_diff < 1e-5
        
        # Check length (should be original - 1)
        if i < len(orig_success):
            orig_ep = orig_success[i]
            T_orig = len(orig_ep["images"])
            length_ok = (T_rev == T_orig - 1)
            
            # Check first reversed image == last original image
            img_match = np.array_equal(rev_ep["images"][0], orig_ep["images"][-1])
            # Check last reversed image == first original image (+1 for dropped frame)
            img_match_last = np.array_equal(rev_ep["images"][-1], orig_ep["images"][1])
        else:
            T_orig = -1
            length_ok = None
            img_match = None
            img_match_last = None
        
        check = {
            "ep_idx": i,
            "T_orig": T_orig,
            "T_rev": T_rev,
            "action_max_diff": max_diff,
            "action_ok": action_ok,
            "length_ok": length_ok,
            "first_img_match": img_match,
            "last_img_match": img_match_last,
        }
        results["checks"].append(check)
        
        if i < 5 or not action_ok or not length_ok:
            status = "✓" if (action_ok and length_ok) else "✗"
            print(f"  ep {i:3d}: T_orig={T_orig:4d} T_rev={T_rev:4d} "
                  f"action_diff={max_diff:.2e} "
                  f"len_ok={length_ok} img_first={img_match} img_last={img_match_last} [{status}]")
    
    # Summary
    n_ok = sum(1 for c in results["checks"] if c["action_ok"] and c.get("length_ok", True))
    print(f"\n  Summary: {n_ok}/{len(rev_eps)} episodes pass all checks")
    
    return results


def plot_trajectory_comparison(orig_ep: dict, rev_ep: dict, out_path: str, 
                                ep_idx: int, label: str):
    """Plot XYZ trajectories of original vs reversed episode."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f"{label} — Episode {ep_idx}\n"
                 f"Original ({len(orig_ep['images'])} frames) vs "
                 f"Reversed ({len(rev_ep['images'])} frames)", fontsize=14)
    
    xyz_labels = ["X", "Y", "Z"]
    
    for row, (dim_name, dim_idx) in enumerate(zip(xyz_labels, range(3))):
        # Left column: EE pose
        ax = axes[row, 0]
        orig_ee = orig_ep["ee_pose"][:, dim_idx]
        rev_ee = rev_ep["ee_pose"][:, dim_idx]
        ax.plot(orig_ee, 'b-', alpha=0.7, label=f"Original (T={len(orig_ee)})")
        ax.plot(rev_ee, 'r-', alpha=0.7, label=f"Reversed (T={len(rev_ee)})")
        # Also plot original reversed in time for comparison
        ax.plot(orig_ee[::-1][:len(rev_ee)], 'g--', alpha=0.5, label="Orig flipped")
        ax.set_ylabel(f"EE {dim_name}")
        ax.legend(fontsize=8)
        ax.set_title(f"EE Pose {dim_name}" if row == 0 else "")
        ax.grid(True, alpha=0.3)
        
        # Right column: Action
        ax = axes[row, 1]
        orig_act = orig_ep["action"][:, dim_idx]
        rev_act = rev_ep["action"][:, dim_idx]
        ax.plot(orig_act, 'b-', alpha=0.7, label=f"Original")
        ax.plot(rev_act, 'r-', alpha=0.7, label=f"Reversed")
        ax.set_ylabel(f"Action {dim_name}")
        ax.legend(fontsize=8)
        ax.set_title(f"Action {dim_name}" if row == 0 else "")
        ax.grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel("Timestep")
    axes[2, 1].set_xlabel("Timestep")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory plot: {out_path}")


def plot_gripper_comparison(orig_ep: dict, rev_ep: dict, out_path: str,
                            ep_idx: int, label: str):
    """Plot gripper state comparison between original and reversed."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"{label} — Episode {ep_idx} — Gripper", fontsize=14)
    
    # Original gripper from action[:, 7]
    orig_gripper = orig_ep["action"][:, 7]
    rev_gripper = rev_ep["action"][:, 7]
    
    axes[0].plot(orig_gripper, 'b-', label="Original action[:,7]")
    axes[0].plot(orig_gripper[::-1][:len(rev_gripper)], 'g--', alpha=0.5, label="Orig flipped")
    axes[0].set_ylabel("Gripper action")
    axes[0].set_title("Original")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(rev_gripper, 'r-', label="Reversed action[:,7]")
    if "gripper" in rev_ep:
        axes[1].plot(rev_ep["gripper"], 'k--', alpha=0.5, label="Reversed gripper field")
    axes[1].set_ylabel("Gripper action")
    axes[1].set_title("Reversed")
    axes[1].set_xlabel("Timestep")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved gripper plot: {out_path}")


def _write_video_h264(frames: list[np.ndarray], out_path: str, fps: int = 20):
    """Write frames as H.264 MP4 using imageio-ffmpeg (VS Code compatible)."""
    import imageio
    
    # Ensure even dimensions (H.264 requirement)
    h, w = frames[0].shape[:2]
    h = h if h % 2 == 0 else h + 1
    w = w if w % 2 == 0 else w + 1
    
    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        # Pad to even if needed
        if frame.shape[0] != h or frame.shape[1] != w:
            padded = np.full((h, w, 3), 255, dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded
        writer.append_data(frame)
    writer.close()


def create_side_by_side_video(orig_ep: dict, rev_ep: dict, out_path: str, fps: int = 20):
    """Create a side-by-side video: original (forward) on left, reversed on right.
    
    For clearer comparison, the original is shown with the timeline reversed
    (last frame first) so both should show similar motion.
    """
    import cv2
    
    orig_imgs = orig_ep["images"]
    rev_imgs = rev_ep["images"]
    
    # Flip original timeline for comparison
    orig_flipped = orig_imgs[::-1]
    
    T = min(len(orig_flipped), len(rev_imgs))
    H, W = orig_imgs.shape[1], orig_imgs.shape[2]
    
    frames = []
    for t in range(T):
        canvas = np.ones((H + 20, W * 2 + 10, 3), dtype=np.uint8) * 255
        canvas[16:16+H, :W] = orig_flipped[t]
        canvas[16:16+H, W+10:] = rev_imgs[t]
        # Add labels
        cv2.putText(canvas, f"Orig(flip) t={t}", (2, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(canvas, f"Reversed t={t}", (W + 12, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        frames.append(canvas)
    
    _write_video_h264(frames, out_path, fps)
    print(f"  Saved comparison video ({T} frames): {out_path}")


def create_episode_video_simple(images: np.ndarray, out_path: str, fps: int = 20,
                                 wrist_images: np.ndarray = None):
    """Create a simple video from images array (H.264, VS Code compatible)."""
    H, W = images.shape[1], images.shape[2]
    
    frames = []
    if wrist_images is not None:
        total_w = W * 2 + 10
        for t in range(len(images)):
            canvas = np.ones((H, total_w, 3), dtype=np.uint8) * 255
            canvas[:, :W] = images[t]
            canvas[:, W+10:] = wrist_images[t]
            frames.append(canvas)
    else:
        frames = list(images)
    
    _write_video_h264(frames, out_path, fps)
    print(f"  Saved video ({len(images)} frames): {out_path}")


def plot_all_trajectories_2d(episodes: list[dict], out_path: str, title: str, max_eps: int = 20):
    """Plot XY trajectories of all episodes on a single 2D plot."""
    SIZE = 8  # equal width and height
    fig, ax = plt.subplots(1, 1, figsize=(SIZE, SIZE))
    
    n = min(len(episodes), max_eps)
    for i in range(n):
        ee = episodes[i]["ee_pose"]
        color = plt.cm.tab20(i % 20)
        ax.plot(ee[:, 0], ee[:, 1], '-', color=color, alpha=0.6, linewidth=1)
        ax.plot(ee[0, 0], ee[0, 1], 'o', color=color, markersize=5)  # start
        ax.plot(ee[-1, 0], ee[-1, 1], 's', color=color, markersize=5)  # end
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title}\n({n} episodes, o=start, □=end)")
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    
    # Force square output: use the data range to set equal axis limits
    ax.autoscale()
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo
    max_range = max(x_range, y_range) * 1.05
    x_mid = (x_lo + x_hi) / 2
    y_mid = (y_lo + y_hi) / 2
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Saved 2D trajectory plot: {out_path}")


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = exp_dir / f"inspect_reversed_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {out_dir}")
    
    # Find all iteration data
    iterations = {}
    for f in sorted(exp_dir.iterdir()):
        if f.name.startswith("iter") and f.name.endswith(".npz"):
            # Parse: iter{N}_collect_{A|B}[_reversed].npz
            parts = f.stem.split("_")
            iter_num = int(parts[0].replace("iter", ""))
            if iter_num not in iterations:
                iterations[iter_num] = {}
            
            if "reversed" in f.stem:
                if "collect_A" in f.stem:
                    iterations[iter_num]["A_reversed"] = str(f)
                elif "collect_B" in f.stem:
                    iterations[iter_num]["B_reversed"] = str(f)
            else:
                if f.name.endswith(".stats.json"):
                    continue
                if "collect_A" in f.stem:
                    iterations[iter_num]["A_orig"] = str(f)
                elif "collect_B" in f.stem:
                    iterations[iter_num]["B_orig"] = str(f)
    
    print(f"\nFound iterations: {sorted(iterations.keys())}")
    for it, files in sorted(iterations.items()):
        print(f"  Iter {it}: {sorted(files.keys())}")
    
    # Also check initial data
    init_files = {}
    for name in ["task_A_reversed_100.npz", "task_B_100.npz"]:
        p = exp_dir / name
        if p.exists():
            init_files[name] = str(p)
            print(f"  Initial: {name}")
    
    # Process each iteration
    for it in sorted(iterations.keys()):
        files = iterations[it]
        iter_dir = out_dir / f"iter{it}"
        iter_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"  ITERATION {it}")
        print(f"{'='*70}")
        
        # For each pair (A_orig + A_reversed, B_orig + B_reversed)
        for task in ["A", "B"]:
            orig_key = f"{task}_orig"
            rev_key = f"{task}_reversed"
            
            if rev_key not in files:
                print(f"\n  [Skip] No reversed data for task {task} in iter {it}")
                continue
            
            rev_eps = load_episodes(files[rev_key])
            
            if orig_key in files:
                orig_eps = load_episodes(files[orig_key])
                
                # Verify reversal correctness
                verify_reversal(orig_eps, rev_eps, f"iter{it} collect_{task}")
                
                # Visualize episodes
                orig_success = [ep for ep in orig_eps if ep.get("success", False)]
                n_vis = min(args.max_episodes, len(rev_eps), len(orig_success))
                
                for ei in range(n_vis):
                    ep_dir = iter_dir / f"task_{task}_ep{ei}"
                    ep_dir.mkdir(exist_ok=True)
                    
                    # Trajectory comparison plot
                    plot_trajectory_comparison(
                        orig_success[ei], rev_eps[ei],
                        str(ep_dir / "trajectory_xyz.png"),
                        ei, f"Iter{it} Task {task}"
                    )
                    
                    # Gripper comparison
                    plot_gripper_comparison(
                        orig_success[ei], rev_eps[ei],
                        str(ep_dir / "gripper.png"),
                        ei, f"Iter{it} Task {task}"
                    )
                    
                    # Videos of the reversed episode
                    wrist = rev_eps[ei].get("wrist_images", None)
                    create_episode_video_simple(
                        rev_eps[ei]["images"],
                        str(ep_dir / "reversed_video.mp4"),
                        fps=args.fps,
                        wrist_images=wrist,
                    )
                    
                    # Side-by-side comparison video
                    create_side_by_side_video(
                        orig_success[ei], rev_eps[ei],
                        str(ep_dir / "comparison_video.mp4"),
                        fps=args.fps,
                    )
                
                # 2D trajectory overview for all reversed episodes
                plot_all_trajectories_2d(
                    rev_eps,
                    str(iter_dir / f"task_{task}_reversed_2d_trajectories.png"),
                    f"Iter{it} Task {task} Reversed — All Trajectories"
                )
                
                # Also for originals for comparison
                plot_all_trajectories_2d(
                    orig_success,
                    str(iter_dir / f"task_{task}_original_2d_trajectories.png"),
                    f"Iter{it} Task {task} Original (success only) — All Trajectories"
                )
            else:
                print(f"\n  No original data for task {task} in iter {it}, "
                      f"visualizing reversed only")
                
                # Verify action consistency
                print(f"\n  Verifying reversed data internal consistency...")
                for i, ep in enumerate(rev_eps[:5]):
                    T = len(ep["images"])
                    diff = np.linalg.norm(ep["action"][:T-1, :7] - ep["ee_pose"][1:], axis=1)
                    max_diff = float(diff.max())
                    print(f"    ep {i}: T={T} action_diff={max_diff:.2e} "
                          f"{'✓' if max_diff < 1e-5 else '✗'}")
                
                n_vis = min(args.max_episodes, len(rev_eps))
                for ei in range(n_vis):
                    ep_dir = iter_dir / f"task_{task}_ep{ei}"
                    ep_dir.mkdir(exist_ok=True)
                    
                    wrist = rev_eps[ei].get("wrist_images", None)
                    create_episode_video_simple(
                        rev_eps[ei]["images"],
                        str(ep_dir / "reversed_video.mp4"),
                        fps=args.fps,
                        wrist_images=wrist,
                    )
                
                plot_all_trajectories_2d(
                    rev_eps,
                    str(iter_dir / f"task_{task}_reversed_2d_trajectories.png"),
                    f"Iter{it} Task {task} Reversed — All Trajectories"
                )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Output: {out_dir}")
    
    # List what's in the pipeline for training
    print(f"\n  Pipeline Data Flow Check:")
    print(f"  {'Iter':<6} {'Policy A trains on':<40} {'Policy B trains on':<40} {'Status'}")
    print(f"  {'----':<6} {'---'*13:<40} {'---'*13:<40} {'------'}")
    
    for it in sorted(iterations.keys()):
        files = iterations[it]
        a_data = "B_reversed" if "B_reversed" in files else "MISSING"
        b_data = "A_reversed" if "A_reversed" in files else "MISSING"
        
        a_exists = "B_reversed" in files
        b_exists = "A_reversed" in files
        
        if a_exists and b_exists:
            status = "✓ Both ready"
        elif a_exists:
            status = "⚠ Only A data (B missing)"
        elif b_exists:
            status = "⚠ Only B data (A missing)"
        else:
            status = "✗ Both missing"
        
        a_info = f"iter{it}_collect_B_reversed.npz" if a_exists else "MISSING (no B rollout)"
        b_info = f"iter{it}_collect_A_reversed.npz" if b_exists else "MISSING (no A rollout)"
        
        print(f"  {it:<6} {a_info:<40} {b_info:<40} {status}")
    
    print(f"\n  Note: Missing reversed data means no successful rollouts from the")
    print(f"  opposite policy in that iteration (--success_only 1 filters them).")
    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
