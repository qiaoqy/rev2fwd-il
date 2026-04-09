#!/usr/bin/env python3
"""Inspect and visualize peg insertion trajectory data.

Produces a video with:
- Two camera views (table + wrist)
- 6D force/torque plots
- EE position / rotation plots
- Peg position / rotation plots (with phase shading)

Usage:
    python scripts/scripts_peg_insert/2_inspect_peg_data.py \
        --dataset data/pick_place_isaac_lab_simulation/exp39/task_B_peg_insert_3.npz \
        --episode 0
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def _parse_args():
    parser = argparse.ArgumentParser(description="Inspect peg insertion data.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def load_episode_data(npz_path, episode_idx):
    data = np.load(npz_path, allow_pickle=True)
    episodes = data["episodes"]
    if episode_idx >= len(episodes):
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {len(episodes)} episodes.")
    return episodes[episode_idx]


def quat_to_euler(quat):
    """(w,x,y,z) -> (roll, pitch, yaw) in degrees, unwrapped."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.stack([np.degrees(np.unwrap(roll)),
                     np.degrees(np.unwrap(pitch)),
                     np.degrees(np.unwrap(yaw))], axis=-1)


def create_combined_video(episode_data, output_path, fps):
    """Video layout (4 rows × 2 cols):
    Row 0: table_cam | wrist_cam
    Row 1: Force (Fx,Fy,Fz) | Torque (Tx,Ty,Tz)
    Row 2: EE pos (Δx,Δy,Δz) | EE rot (roll,pitch,yaw)
    Row 3: Peg pos (Δx,Δy,Δz) | Peg rot (roll,pitch,yaw)
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    images = episode_data["images"]
    ft_force_raw = episode_data["ft_force_raw"]
    ee_pose = episode_data["ee_pose"]
    peg_pose = episode_data["peg_pose"]
    T = len(images)

    ft_force = ft_force_raw[:, :3]
    ft_torque = ft_force_raw[:, 3:6]
    ee_pos = ee_pose[:, :3]
    ee_euler = quat_to_euler(ee_pose[:, 3:7])
    peg_pos = peg_pose[:, :3]
    peg_euler = quat_to_euler(peg_pose[:, 3:7])

    phase_data = episode_data.get("phase", None)
    task_type = str(episode_data.get("task_type", "insert"))
    phase_names = episode_data.get("phase_names", ["APPROACH", "ALIGN", "INSERT", "DONE"])

    # Reference index: start of main action phase
    # For insert: start of INSERT (phase 2)
    # For extract: start of EXTRACT (phase 0) → always 0
    ref_idx = 0
    if phase_data is not None:
        if task_type == "extract":
            ref_idx = 0  # EXTRACT starts at beginning
        else:
            for i in range(len(phase_data)):
                if int(phase_data[i]) == 2:
                    ref_idx = i
                    break

    ee_pos_rel = (ee_pos - ee_pos[ref_idx:ref_idx+1]) * 1000.0  # mm
    peg_pos_rel = (peg_pos - peg_pos[ref_idx:ref_idx+1]) * 1000.0
    if task_type == "extract":
        phase_colors = {0: "#98FB98", 1: "#E6E6FA", 2: "#D3D3D3"}
        phase_text_colors = {0: "#006400", 1: "#4B0082", 2: "#404040"}
    else:
        phase_colors = {0: "#FFE4E1", 1: "#E6E6FA", 2: "#98FB98", 3: "#D3D3D3"}
        phase_text_colors = {0: "#8B0000", 1: "#4B0082", 2: "#006400", 3: "#404040"}

    # Wrist camera
    wrist_images = None
    for cn in episode_data.get("wrist_cam_names", []):
        key = f"wrist_{cn}"
        if key in episode_data:
            img = episode_data[key]
            if img.ndim >= 3 and img.shape[1] > 0 and not np.all(img == 0):
                wrist_images = img
                break

    def _ylim(arr, margin=0.1):
        lo, hi = float(arr.min()), float(arr.max())
        s = max(hi - lo, 1e-6)
        return lo - s * margin, hi + s * margin

    force_ylim = _ylim(ft_force)
    torque_ylim = _ylim(ft_torque)
    pos_ylim = _ylim(ee_pos_rel)
    rot_ylim = _ylim(ee_euler)
    peg_pos_ylim = _ylim(peg_pos_rel)
    peg_rot_ylim = _ylim(peg_euler)

    def _shade(ax, pd):
        if pd is None:
            return
        starts, vals = [0], [int(pd[0])]
        for i in range(1, len(pd)):
            if pd[i] != pd[i-1]:
                starts.append(i); vals.append(int(pd[i]))
        starts.append(len(pd))
        for i, v in enumerate(vals):
            ax.axvspan(starts[i], starts[i+1], alpha=0.25,
                       color=phase_colors.get(v, "#FFF"), zorder=0)

    print(f"Creating video ({T} frames @ {fps} fps) ...")
    frames = []

    cam_h, cam_bot = 0.22, 0.76
    r1_h, r1_bot = 0.15, 0.56
    r2_h, r2_bot = 0.15, 0.36
    r3_h, r3_bot = 0.15, 0.05
    lm, rm, pw = 0.07, 0.50, 0.40

    for t in range(T):
        fig = plt.figure(figsize=(14, 12))
        cur_phase = int(phase_data[t]) if phase_data is not None else -1
        cur_name = phase_names[cur_phase] if 0 <= cur_phase < len(phase_names) else ""
        ts = np.arange(t + 1)

        # Cameras
        ax_table = fig.add_axes([0.02, cam_bot, 0.46, cam_h])
        ax_table.imshow(images[t])
        ax_table.set_title(f"Table cam | Frame {t}/{T}", fontsize=10); ax_table.axis("off")
        if cur_name:
            ax_table.text(0.5, 0.95, cur_name, transform=ax_table.transAxes,
                          fontsize=14, fontweight="bold",
                          color=phase_text_colors.get(cur_phase, "#000"),
                          ha="center", va="top",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=phase_colors.get(cur_phase, "#FFF"),
                                    edgecolor="black", alpha=0.9))

        ax_wrist = fig.add_axes([0.52, cam_bot, 0.46, cam_h])
        if wrist_images is not None:
            ax_wrist.imshow(wrist_images[t])
        else:
            ax_wrist.text(0.5, 0.5, "No wrist camera", transform=ax_wrist.transAxes,
                          ha="center", va="center", fontsize=12, color="gray")
        ax_wrist.set_title("Wrist cam", fontsize=10); ax_wrist.axis("off")

        # Force
        ax_f = fig.add_axes([lm, r1_bot, pw, r1_h])
        _shade(ax_f, phase_data)
        ax_f.plot(ts, ft_force[:t+1, 0], "r-", lw=1.2, label="Fx")
        ax_f.plot(ts, ft_force[:t+1, 1], "g-", lw=1.2, label="Fy")
        ax_f.plot(ts, ft_force[:t+1, 2], "b-", lw=1.2, label="Fz")
        ax_f.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_f.set_xlim(0, T); ax_f.set_ylim(*force_ylim)
        ax_f.set_ylabel("Force (N)", fontsize=9); ax_f.set_title("Force", fontsize=10)
        ax_f.legend(loc="upper right", fontsize=7, ncol=3)
        ax_f.grid(True, alpha=0.3); ax_f.tick_params(labelsize=7); ax_f.set_xticklabels([])

        # Torque
        ax_t = fig.add_axes([rm, r1_bot, pw, r1_h])
        _shade(ax_t, phase_data)
        ax_t.plot(ts, ft_torque[:t+1, 0], "r-", lw=1.2, label="Tx")
        ax_t.plot(ts, ft_torque[:t+1, 1], "g-", lw=1.2, label="Ty")
        ax_t.plot(ts, ft_torque[:t+1, 2], "b-", lw=1.2, label="Tz")
        ax_t.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_t.set_xlim(0, T); ax_t.set_ylim(*torque_ylim)
        ax_t.set_ylabel("Torque (Nm)", fontsize=9); ax_t.set_title("Torque", fontsize=10)
        ax_t.legend(loc="upper right", fontsize=7, ncol=3)
        ax_t.grid(True, alpha=0.3); ax_t.tick_params(labelsize=7); ax_t.set_xticklabels([])

        # EE pos
        ax_p = fig.add_axes([lm, r2_bot, pw, r2_h])
        _shade(ax_p, phase_data)
        ax_p.plot(ts, ee_pos_rel[:t+1, 0], "r-", lw=1.2, label="\u0394x")
        ax_p.plot(ts, ee_pos_rel[:t+1, 1], "g-", lw=1.2, label="\u0394y")
        ax_p.plot(ts, ee_pos_rel[:t+1, 2], "b-", lw=1.2, label="\u0394z")
        ax_p.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_p.set_xlim(0, T); ax_p.set_ylim(*pos_ylim)
        ax_p.set_ylabel("\u0394 Pos (mm)", fontsize=9)
        ref_label = "EXTRACT" if task_type == "extract" else "INSERT"
        ax_p.set_title(f"EE Position (rel. {ref_label} start)", fontsize=10)
        ax_p.legend(loc="upper right", fontsize=7, ncol=3)
        ax_p.grid(True, alpha=0.3); ax_p.tick_params(labelsize=7); ax_p.set_xticklabels([])

        # EE rot
        ax_r = fig.add_axes([rm, r2_bot, pw, r2_h])
        _shade(ax_r, phase_data)
        ax_r.plot(ts, ee_euler[:t+1, 0], "r-", lw=1.2, label="roll")
        ax_r.plot(ts, ee_euler[:t+1, 1], "g-", lw=1.2, label="pitch")
        ax_r.plot(ts, ee_euler[:t+1, 2], "b-", lw=1.2, label="yaw")
        ax_r.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_r.set_xlim(0, T); ax_r.set_ylim(*rot_ylim)
        ax_r.set_ylabel("Rot (deg)", fontsize=9); ax_r.set_title("EE Rotation", fontsize=10)
        ax_r.legend(loc="upper right", fontsize=7, ncol=3)
        ax_r.grid(True, alpha=0.3); ax_r.tick_params(labelsize=7); ax_r.set_xticklabels([])

        # Peg pos
        ax_pp = fig.add_axes([lm, r3_bot, pw, r3_h])
        _shade(ax_pp, phase_data)
        ax_pp.plot(ts, peg_pos_rel[:t+1, 0], "r-", lw=1.2, label="\u0394x")
        ax_pp.plot(ts, peg_pos_rel[:t+1, 1], "g-", lw=1.2, label="\u0394y")
        ax_pp.plot(ts, peg_pos_rel[:t+1, 2], "b-", lw=1.2, label="\u0394z")
        ax_pp.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_pp.set_xlim(0, T); ax_pp.set_ylim(*peg_pos_ylim)
        ax_pp.set_xlabel("Step", fontsize=9); ax_pp.set_ylabel("\u0394 Pos (mm)", fontsize=9)
        ax_pp.set_title(f"Peg Position (rel. {ref_label} start)", fontsize=10)
        ax_pp.legend(loc="upper right", fontsize=7, ncol=3)
        ax_pp.grid(True, alpha=0.3); ax_pp.tick_params(labelsize=7)

        # Peg rot
        ax_pr = fig.add_axes([rm, r3_bot, pw, r3_h])
        _shade(ax_pr, phase_data)
        ax_pr.plot(ts, peg_euler[:t+1, 0], "r-", lw=1.2, label="roll")
        ax_pr.plot(ts, peg_euler[:t+1, 1], "g-", lw=1.2, label="pitch")
        ax_pr.plot(ts, peg_euler[:t+1, 2], "b-", lw=1.2, label="yaw")
        ax_pr.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_pr.set_xlim(0, T); ax_pr.set_ylim(*peg_rot_ylim)
        ax_pr.set_xlabel("Step", fontsize=9); ax_pr.set_ylabel("Rot (deg)", fontsize=9)
        ax_pr.set_title("Peg Rotation", fontsize=10)
        ax_pr.legend(loc="upper right", fontsize=7, ncol=3)
        ax_pr.grid(True, alpha=0.3); ax_pr.tick_params(labelsize=7)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T} frames rendered")

    with imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    print(f"Saved video: {output_path}")


def print_episode_stats(ep, idx):
    print(f"\n{'='*60}")
    print(f"Episode {idx}")
    print(f"{'='*60}")
    T = ep.get("episode_length", len(ep["obs"]))
    print(f"Length: {T}")
    print(f"Success: {ep.get('success', 'N/A')}")
    task_type = str(ep.get('task_type', 'insert'))
    if task_type == 'extract':
        print(f"Z above tip: {ep.get('z_above_tip', 0)*1000:.2f} mm")
    else:
        print(f"Z progress: {ep.get('z_progress', 0)*1000:.2f} mm")
    print(f"Images: {ep['images'].shape}")
    ee = ep["ee_pose"][:, :3]
    print(f"EE Z range: [{ee[:, 2].min():.4f}, {ee[:, 2].max():.4f}]")
    ft = ep["ft_force_raw"]
    print(f"Force max: {np.abs(ft[:, :3]).max():.2f} N")
    print(f"Torque max: {np.abs(ft[:, 3:6]).max():.4f} Nm")
    phase = ep.get("phase", None)
    if phase is not None:
        names = ep.get("phase_names", ["APPROACH", "ALIGN", "INSERT", "DONE"])
        for i, n in enumerate(names):
            c = int(np.sum(phase == i))
            if c > 0:
                print(f"  {n}: {c} steps ({100*c/T:.1f}%)")
    print(f"{'='*60}\n")


def main():
    args = _parse_args()
    dpath = Path(args.dataset)
    if not dpath.exists():
        print(f"Error: {dpath} not found"); return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or dpath.stem
    out_dir = dpath.parent / f"inspect_{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = load_episode_data(str(dpath), args.episode)
    print_episode_stats(ep, args.episode)

    if ep["images"].ndim < 3 or ep["images"].shape[1] == 0:
        print("ERROR: No images."); return

    video_path = out_dir / f"episode_{args.episode}.mp4"
    create_combined_video(ep, video_path, args.fps)
    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
