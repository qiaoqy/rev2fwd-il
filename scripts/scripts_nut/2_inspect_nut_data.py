#!/usr/bin/env python3
"""Step 2: Inspect and visualize nut threading trajectory data.

Produces a SINGLE video containing:
- Two camera views (table + wrist)
- 6D force/torque plots (Fx, Fy, Fz, Tx, Ty, Tz)
- 6D EE translation/rotation plots (x, y, z, roll, pitch, yaw)
- Nut pose plots (position xyz + rotation)

=============================================================================
USAGE
=============================================================================
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz --episode 0

# Custom name & fps
python scripts/scripts_nut/2_inspect_nut_data.py --dataset data/nut_thread.npz \
    --episode 0 --name my_run --fps 20
=============================================================================
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect nut threading data.")
    parser.add_argument("--dataset", type=str, default="data/nut_thread.npz",
                        help="Path to the NPZ file.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index.")
    parser.add_argument("--name", type=str, default=None, help="Custom output folder name.")
    parser.add_argument("--fps", type=int, default=15, help="Video fps.")
    return parser.parse_args()


def load_episode_data(npz_path: str, episode_idx: int) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if 'episodes' not in data.files:
        raise ValueError(f"Invalid dataset. Expected 'episodes' key, found: {data.files}")
    episodes = data['episodes']
    if episode_idx >= len(episodes):
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {len(episodes)} episodes.")
    return episodes[episode_idx]


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees.

    Applies np.unwrap along the time axis to eliminate ±180° discontinuities
    (e.g. when the gripper points straight down, roll sits right at the
    ±180° wrap boundary and flickers between +180° and -180°).
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    # Unwrap along time axis (axis 0) to remove ±180° jumps
    roll = np.unwrap(roll, axis=0)
    pitch = np.unwrap(pitch, axis=0)
    yaw = np.unwrap(yaw, axis=0)
    return np.stack([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)], axis=-1)


def create_combined_video(episode_data: dict, output_path: Path, fps: int) -> None:
    """Create a single video with cameras, force/torque, EE pose, and nut pose.

    Layout (4 rows x 2 cols):
        Row 0: table_cam | wrist_cam
        Row 1: Force (Fx,Fy,Fz) | Torque (Tx,Ty,Tz)
        Row 2: EE pos (x,y,z) | EE rot (roll,pitch,yaw)
        Row 3: Nut pos (x,y,z) | Nut rot (roll,pitch,yaw)
    """
    import imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    images = episode_data["images"]
    ft_force_raw = episode_data["ft_force_raw"]  # (T, 6)
    ee_pose = episode_data["ee_pose"]  # (T, 7)
    nut_pose = episode_data["nut_pose"]  # (T, 7)
    T = len(images)

    ft_force = ft_force_raw[:, :3]
    ft_torque = ft_force_raw[:, 3:6]
    ee_pos = ee_pose[:, :3]
    ee_quat = ee_pose[:, 3:7]
    ee_euler = quat_to_euler(ee_quat)
    nut_pos = nut_pose[:, :3]
    nut_quat = nut_pose[:, 3:7]
    nut_euler = quat_to_euler(nut_quat)

    phase_data = episode_data.get("phase", None)
    phase_names = episode_data.get(
        "phase_names",
        ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE",
         "RELEASE", "REPOSITION", "REGRASP"],
    )
    phase_colors = {
        0: "#FFE4E1", 1: "#E6E6FA", 2: "#FFFACD", 3: "#98FB98",
        4: "#D3D3D3", 5: "#FFD700", 6: "#87CEEB", 7: "#DDA0DD",
    }
    phase_text_colors = {
        0: "#8B0000", 1: "#4B0082", 2: "#8B8000", 3: "#006400",
        4: "#404040", 5: "#B8860B", 6: "#00688B", 7: "#8B008B",
    }

    # Find first valid wrist camera
    wrist_images = None
    wrist_cam_names = episode_data.get("wrist_cam_names", None)
    if wrist_cam_names is not None:
        for cam_name in wrist_cam_names:
            key = f"wrist_{cam_name}"
            if key in episode_data:
                img = episode_data[key]
                if img.ndim >= 3 and img.shape[1] > 0 and not np.all(img == 0):
                    wrist_images = img
                    break
    if wrist_images is None:
        wi = episode_data.get("wrist_images", None)
        if wi is not None and wi.ndim >= 3 and wi.shape[1] > 0 and not np.all(wi == 0):
            wrist_images = wi

    def _ylim(arr, margin_frac=0.1):
        lo, hi = float(arr.min()), float(arr.max())
        span = max(hi - lo, 1e-6)
        return lo - span * margin_frac, hi + span * margin_frac

    force_ylim = _ylim(ft_force)
    torque_ylim = _ylim(ft_torque)
    pos_ylim = _ylim(ee_pos)
    rot_ylim = _ylim(ee_euler)
    nut_pos_ylim = _ylim(nut_pos)
    nut_rot_ylim = _ylim(nut_euler)

    def _shade_phases(ax, phase_data):
        if phase_data is None:
            return
        starts, vals = [0], [int(phase_data[0])]
        for i in range(1, len(phase_data)):
            if phase_data[i] != phase_data[i - 1]:
                starts.append(i)
                vals.append(int(phase_data[i]))
        starts.append(len(phase_data))
        for i, v in enumerate(vals):
            ax.axvspan(starts[i], starts[i + 1], alpha=0.25,
                       color=phase_colors.get(v, "#FFFFFF"), zorder=0)

    print(f"Creating combined video ({T} frames @ {fps} fps) ...")

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(14, 12))

        # Vertical layout: 4 rows of plots + camera row on top
        # Row heights: cameras=0.22, force/torque=0.17, EE=0.17, nut=0.17
        cam_h = 0.22; cam_bot = 0.76
        r1_h = 0.15; r1_bot = 0.56
        r2_h = 0.15; r2_bot = 0.36
        r3_h = 0.15; r3_bot = 0.05
        lm = 0.07; rm = 0.50; pw = 0.40

        # Row 0: cameras
        ax_table = fig.add_axes([0.02, cam_bot, 0.46, cam_h])
        ax_wrist = fig.add_axes([0.52, cam_bot, 0.46, cam_h])

        cur_phase = int(phase_data[t]) if phase_data is not None else -1
        cur_name = phase_names[cur_phase] if 0 <= cur_phase < len(phase_names) else ""

        ax_table.imshow(images[t])
        ax_table.set_title(f"Table cam  |  Frame {t}/{T}", fontsize=10)
        ax_table.axis("off")
        if cur_name:
            ax_table.text(
                0.5, 0.95, cur_name, transform=ax_table.transAxes,
                fontsize=14, fontweight="bold",
                color=phase_text_colors.get(cur_phase, "#000"),
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=phase_colors.get(cur_phase, "#FFF"),
                          edgecolor="black", alpha=0.9))

        if wrist_images is not None:
            ax_wrist.imshow(wrist_images[t])
        else:
            ax_wrist.text(0.5, 0.5, "No wrist camera",
                          transform=ax_wrist.transAxes,
                          ha="center", va="center", fontsize=12, color="gray")
        ax_wrist.set_title("Wrist cam", fontsize=10)
        ax_wrist.axis("off")

        ts = np.arange(t + 1)

        # Row 1 left: Force
        ax_f = fig.add_axes([lm, r1_bot, pw, r1_h])
        _shade_phases(ax_f, phase_data)
        ax_f.plot(ts, ft_force[:t+1, 0], "r-", lw=1.2, label="Fx")
        ax_f.plot(ts, ft_force[:t+1, 1], "g-", lw=1.2, label="Fy")
        ax_f.plot(ts, ft_force[:t+1, 2], "b-", lw=1.2, label="Fz")
        ax_f.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_f.set_xlim(0, T); ax_f.set_ylim(*force_ylim)
        ax_f.set_ylabel("Force (N)", fontsize=9)
        ax_f.set_title("Force", fontsize=10)
        ax_f.legend(loc="upper right", fontsize=7, ncol=3)
        ax_f.grid(True, alpha=0.3); ax_f.tick_params(labelsize=7)
        ax_f.set_xticklabels([])

        # Row 1 right: Torque
        ax_t = fig.add_axes([rm, r1_bot, pw, r1_h])
        _shade_phases(ax_t, phase_data)
        ax_t.plot(ts, ft_torque[:t+1, 0], "r-", lw=1.2, label="Tx")
        ax_t.plot(ts, ft_torque[:t+1, 1], "g-", lw=1.2, label="Ty")
        ax_t.plot(ts, ft_torque[:t+1, 2], "b-", lw=1.2, label="Tz")
        ax_t.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_t.set_xlim(0, T); ax_t.set_ylim(*torque_ylim)
        ax_t.set_ylabel("Torque (Nm)", fontsize=9)
        ax_t.set_title("Torque", fontsize=10)
        ax_t.legend(loc="upper right", fontsize=7, ncol=3)
        ax_t.grid(True, alpha=0.3); ax_t.tick_params(labelsize=7)
        ax_t.set_xticklabels([])

        # Row 2 left: EE position
        ax_p = fig.add_axes([lm, r2_bot, pw, r2_h])
        _shade_phases(ax_p, phase_data)
        ax_p.plot(ts, ee_pos[:t+1, 0], "r-", lw=1.2, label="x")
        ax_p.plot(ts, ee_pos[:t+1, 1], "g-", lw=1.2, label="y")
        ax_p.plot(ts, ee_pos[:t+1, 2], "b-", lw=1.2, label="z")
        ax_p.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_p.set_xlim(0, T); ax_p.set_ylim(*pos_ylim)
        ax_p.set_ylabel("Position (m)", fontsize=9)
        ax_p.set_title("EE Position", fontsize=10)
        ax_p.legend(loc="upper right", fontsize=7, ncol=3)
        ax_p.grid(True, alpha=0.3); ax_p.tick_params(labelsize=7)
        ax_p.set_xticklabels([])

        # Row 2 right: EE rotation
        ax_r = fig.add_axes([rm, r2_bot, pw, r2_h])
        _shade_phases(ax_r, phase_data)
        ax_r.plot(ts, ee_euler[:t+1, 0], "r-", lw=1.2, label="roll")
        ax_r.plot(ts, ee_euler[:t+1, 1], "g-", lw=1.2, label="pitch")
        ax_r.plot(ts, ee_euler[:t+1, 2], "b-", lw=1.2, label="yaw")
        ax_r.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_r.set_xlim(0, T); ax_r.set_ylim(*rot_ylim)
        ax_r.set_ylabel("Rotation (deg)", fontsize=9)
        ax_r.set_title("EE Rotation", fontsize=10)
        ax_r.legend(loc="upper right", fontsize=7, ncol=3)
        ax_r.grid(True, alpha=0.3); ax_r.tick_params(labelsize=7)
        ax_r.set_xticklabels([])

        # Row 3 left: Nut position
        ax_np = fig.add_axes([lm, r3_bot, pw, r3_h])
        _shade_phases(ax_np, phase_data)
        ax_np.plot(ts, nut_pos[:t+1, 0], "r-", lw=1.2, label="x")
        ax_np.plot(ts, nut_pos[:t+1, 1], "g-", lw=1.2, label="y")
        ax_np.plot(ts, nut_pos[:t+1, 2], "b-", lw=1.2, label="z")
        ax_np.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_np.set_xlim(0, T); ax_np.set_ylim(*nut_pos_ylim)
        ax_np.set_xlabel("Step", fontsize=9)
        ax_np.set_ylabel("Position (m)", fontsize=9)
        ax_np.set_title("Nut Position", fontsize=10)
        ax_np.legend(loc="upper right", fontsize=7, ncol=3)
        ax_np.grid(True, alpha=0.3); ax_np.tick_params(labelsize=7)

        # Row 3 right: Nut rotation
        ax_nr = fig.add_axes([rm, r3_bot, pw, r3_h])
        _shade_phases(ax_nr, phase_data)
        ax_nr.plot(ts, nut_euler[:t+1, 0], "r-", lw=1.2, label="roll")
        ax_nr.plot(ts, nut_euler[:t+1, 1], "g-", lw=1.2, label="pitch")
        ax_nr.plot(ts, nut_euler[:t+1, 2], "b-", lw=1.2, label="yaw")
        ax_nr.axvline(t, color="k", ls="--", alpha=0.5, lw=0.8)
        ax_nr.set_xlim(0, T); ax_nr.set_ylim(*nut_rot_ylim)
        ax_nr.set_xlabel("Step", fontsize=9)
        ax_nr.set_ylabel("Rotation (deg)", fontsize=9)
        ax_nr.set_title("Nut Rotation", fontsize=10)
        ax_nr.legend(loc="upper right", fontsize=7, ncol=3)
        ax_nr.grid(True, alpha=0.3); ax_nr.tick_params(labelsize=7)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T} frames rendered")

    with imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    print(f"Saved video to {output_path}")


def print_episode_stats(episode_data: dict, episode_idx: int) -> None:
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx} Statistics")
    print(f"{'='*60}")
    T = episode_data.get("episode_length", len(episode_data["obs"]))
    print(f"Length: {T}")
    print(f"Images shape: {episode_data['images'].shape}")
    ee = episode_data["ee_pose"][:, :3]
    print(f"EE Z range: [{ee[:, 2].min():.4f}, {ee[:, 2].max():.4f}]")
    ft = episode_data["ft_force_raw"]
    print(f"Force max abs: {np.abs(ft[:, :3]).max():.2f} N")
    print(f"Torque max abs: {np.abs(ft[:, 3:6]).max():.4f} Nm")
    phase = episode_data.get("phase", None)
    if phase is not None:
        names = episode_data.get("phase_names",
                                 ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE",
                                  "RELEASE", "REPOSITION", "REGRASP"])
        for i, n in enumerate(names):
            c = int(np.sum(phase == i))
            if c > 0:
                print(f"  {n}: {c} steps ({100*c/len(phase):.1f}%)")
    print(f"{'='*60}\n")


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or dataset_path.stem
    output_dir = Path("data") / f"inspect_{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_data = load_episode_data(str(dataset_path), args.episode)
    print_episode_stats(episode_data, args.episode)

    images = episode_data["images"]
    if images.ndim < 3 or images.shape[1] == 0:
        print("ERROR: No valid images. Use --disable_fabric 1 during collection.")
        return

    video_path = output_dir / f"episode_{args.episode}.mp4"
    create_combined_video(episode_data, video_path, args.fps)
    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
