#!/usr/bin/env python3
"""Replay reversed rollout actions in Task B environment and render videos.

Loads episodes from a reversed NPZ file, replays their actions in the
Isaac Lab simulator (Task B setup: cube starts at goal, target is red region),
records frames, and renders annotated MP4 videos.

Usage:
    conda activate rev2fwd_il

    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/10_replay_reversed_data.py \
        --npz data/pick_place_isaac_lab_simulation/exp21/rollout_A_200_reversed.npz \
        --episode_indices 0 1 2 \
        --out_dir data/pick_place_isaac_lab_simulation/exp21/replay_reversed_videos \
        --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch


# ── time-reversal (from 2_reverse_to_task_A.py) ─────────────────────────
def reverse_episode(ep: dict) -> dict:
    """Time-reverse an episode.  action[t][:7] = ee_pose[t+1] after reversal."""
    T = len(ep["images"])
    images_rev = ep["images"][::-1].copy()
    ee_rev = ep["ee_pose"][::-1].copy()
    obj_rev = ep["obj_pose"][::-1].copy()

    has_gripper = "gripper" in ep
    if has_gripper:
        gripper_rev = ep["gripper"][::-1].copy()
    else:
        gripper_rev = ep["action"][:, 7][::-1].copy()

    has_wrist = "wrist_images" in ep
    if has_wrist:
        wrist_rev = ep["wrist_images"][::-1].copy()

    new_actions = np.zeros((T, 8), dtype=np.float32)
    new_actions[:T - 1, :7] = ee_rev[1:]
    new_actions[T - 1, :7] = ee_rev[T - 1]
    new_actions[:, 7] = gripper_rev

    result = {
        "images": images_rev[:-1],
        "ee_pose": ee_rev[:-1].astype(np.float32),
        "obj_pose": obj_rev[:-1].astype(np.float32),
        "action": new_actions[:-1].astype(np.float32),
        "gripper": gripper_rev[:-1].astype(np.float32),
    }
    if has_wrist:
        result["wrist_images"] = wrist_rev[:-1]
    if "place_pose" in ep:
        result["place_pose"] = ep["place_pose"].copy() if hasattr(ep["place_pose"], "copy") else ep["place_pose"]
    if "goal_pose" in ep:
        result["goal_pose"] = ep["goal_pose"].copy() if hasattr(ep["goal_pose"], "copy") else ep["goal_pose"]
    return result


# ── subsample (from 3_subsample_episodes.py) ─────────────────────────────
def subsample_episode(ep: dict, target_frames: int) -> dict:
    """Uniformly subsample an episode to target_frames, then recompute actions."""
    T = len(ep["images"])
    if T <= target_frames:
        result = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ep.items()}
        _recompute_actions(result)
        return result

    indices = np.round(np.linspace(0, T - 1, target_frames)).astype(int)

    TEMPORAL_KEYS = ["images", "ee_pose", "obj_pose", "action", "obs",
                     "gripper", "fsm_state", "wrist_images"]
    result = {}
    for key, val in ep.items():
        if key in TEMPORAL_KEYS and hasattr(val, "__len__") and len(val) == T:
            result[key] = val[indices].copy()
        elif hasattr(val, "copy"):
            result[key] = val.copy()
        else:
            result[key] = val
    _recompute_actions(result)
    return result


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


# ── video helpers ────────────────────────────────────────────────────────
def _write_video_h264(frames: list[np.ndarray], out_path: str | Path,
                      fps: int = 20) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    # ensure even dimensions for H.264
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


def add_text_overlay(img: np.ndarray, text: str,
                     position=(4, 16), font_scale=0.45,
                     color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)
    return img


def render_annotated_video(
    sim_frames: list[np.ndarray],
    data_frames: np.ndarray,
    obj_poses_sim: list[np.ndarray],
    obj_poses_data: np.ndarray,
    ee_poses_sim: list[np.ndarray],
    actions: np.ndarray,
    ep_idx: int,
    out_path: str | Path,
    goal_xy: tuple[float, float],
    region_center_xy: tuple[float, float],
    region_size_xy: tuple[float, float],
    fps: int = 20,
    wrist_frames_sim: list[np.ndarray] | None = None,
    wrist_frames_data: np.ndarray | None = None,
) -> None:
    """Render side-by-side video: left=sim replay, right=original reversed data."""
    T = min(len(sim_frames), len(data_frames))
    H, W = sim_frames[0].shape[:2]
    scale = max(1, 256 // W)
    out_H, out_W = H * scale, W * scale

    annotated = []
    for t in range(T):
        # Sim frame (left)
        sim_f = sim_frames[t]
        if scale > 1:
            sim_f = cv2.resize(sim_f, (out_W, out_H),
                               interpolation=cv2.INTER_NEAREST)

        obj_sim = obj_poses_sim[t] if t < len(obj_poses_sim) else np.zeros(3)
        obj_data = obj_poses_data[t] if t < len(obj_poses_data) else np.zeros(3)
        ee_sim = ee_poses_sim[t] if t < len(ee_poses_sim) else np.zeros(7)
        gripper = actions[t, 7] if t < len(actions) else 0

        # Overlay on sim frame
        lines_sim = [
            f"SIM REPLAY | Ep {ep_idx} | Step {t+1}/{T}",
            f"ObjXY: ({obj_sim[0]:.3f}, {obj_sim[1]:.3f})  Z:{obj_sim[2]:.3f}",
            f"EE: ({ee_sim[0]:.3f}, {ee_sim[1]:.3f}, {ee_sim[2]:.3f})",
            f"Grip: {'OPEN' if gripper > 0.5 else 'CLOSED'} ({gripper:.2f})",
        ]
        for i, text in enumerate(lines_sim):
            y = 4 + (i + 1) * int(14 * scale / 2 + 2)
            sim_f = add_text_overlay(sim_f, text, position=(4, y),
                                     font_scale=0.35 * scale / 2 + 0.1)

        # Data frame (right)
        data_f = data_frames[t]
        if scale > 1:
            data_f = cv2.resize(data_f, (out_W, out_H),
                                interpolation=cv2.INTER_NEAREST)

        lines_data = [
            f"REVERSED DATA | Ep {ep_idx} | Step {t+1}/{T}",
            f"ObjXY: ({obj_data[0]:.3f}, {obj_data[1]:.3f})  Z:{obj_data[2]:.3f}",
        ]
        for i, text in enumerate(lines_data):
            y = 4 + (i + 1) * int(14 * scale / 2 + 2)
            data_f = add_text_overlay(data_f, text, position=(4, y),
                                      font_scale=0.35 * scale / 2 + 0.1)

        # Side-by-side
        combined = np.concatenate([sim_f, data_f], axis=1)

        # Wrist cameras below: sim wrist (left), data wrist (right)
        has_sim_wrist = (wrist_frames_sim is not None and t < len(wrist_frames_sim))
        has_data_wrist = (wrist_frames_data is not None and t < len(wrist_frames_data))
        if has_sim_wrist or has_data_wrist:
            if has_sim_wrist:
                wf_sim = wrist_frames_sim[t]
                if scale > 1:
                    wf_sim = cv2.resize(wf_sim, (out_W, out_H),
                                        interpolation=cv2.INTER_NEAREST)
            else:
                wf_sim = np.zeros((out_H, out_W, 3), dtype=np.uint8)
            if has_data_wrist:
                wf_data = wrist_frames_data[t]
                if scale > 1:
                    wf_data = cv2.resize(wf_data, (out_W, out_H),
                                         interpolation=cv2.INTER_NEAREST)
            else:
                wf_data = np.zeros((out_H, out_W, 3), dtype=np.uint8)
            wrist_row = np.concatenate([wf_sim, wf_data], axis=1)
            combined = np.concatenate([combined, wrist_row], axis=0)

        annotated.append(combined)

    _write_video_h264(annotated, out_path, fps=fps)


# ── arg parsing ──────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay reversed rollout actions in Task B env and render videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--npz", type=str, required=True,
                        help="NPZ file path (reversed data or original rollout).")
    parser.add_argument("--episode_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Episode indices to replay (default: 0 1 2).")
    parser.add_argument("--reverse_on_load", action="store_true",
                        help="If set, time-reverse episodes on load (use with "
                             "original rollout data, not already-reversed).")
    parser.add_argument("--subsample_frames", type=int, default=None,
                        help="If set, subsample each episode to this many frames "
                             "(uniform temporal downsampling) before replay.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for videos.")

    parser.add_argument("--horizon", type=int, default=None,
                        help="Override max steps (default: use episode length).")
    parser.add_argument("--distance_threshold", type=float, default=0.03)

    # Region parameters (exp21 defaults)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3])

    # Env / sim
    parser.add_argument("--env_task", type=str,
                        default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--command_source", type=str,
                        choices=["action", "ee_pose"], default="action",
                        help="Replay from 'action' field or 'ee_pose' + gripper.")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_ee_pose_w,
            get_object_pose_w,
            teleport_object_to_pose,
            pre_position_gripper_down,
        )

        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).resolve().parent.parent
                / "scripts_pick_place" / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        create_target_markers = _alt_mod.create_target_markers
        update_target_markers = _alt_mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        goal_xy = np.array(args.goal_xy)
        region_center = tuple(args.red_region_center_xy)
        region_size = tuple(args.red_region_size_xy)

        # ── Load episodes from NPZ ──
        print(f"\nLoading {args.npz} ...")
        t0 = time.time()
        with np.load(args.npz, allow_pickle=True) as data:
            all_episodes = list(data["episodes"])
        print(f"  Loaded {len(all_episodes)} episodes in {time.time() - t0:.1f}s")

        # Optionally filter to successful episodes and reverse
        if args.reverse_on_load:
            all_episodes = [ep for ep in all_episodes
                           if ep.get("success", False)]
            print(f"  Filtered to {len(all_episodes)} successful episodes")
            print("  Reversing episodes on-the-fly...")
            all_episodes = [reverse_episode(ep) for ep in all_episodes]
            print(f"  Reversed {len(all_episodes)} episodes")

        # Optionally subsample
        if args.subsample_frames is not None:
            target = args.subsample_frames
            print(f"  Subsampling to {target} frames per episode...")
            for i, ep in enumerate(all_episodes):
                orig_T = len(ep["images"])
                all_episodes[i] = subsample_episode(ep, target)
                new_T = len(all_episodes[i]["images"])
                if i < 3:
                    print(f"    ep {i}: {orig_T} -> {new_T} frames")
            print(f"  Subsampled {len(all_episodes)} episodes")

        # Validate indices
        for idx in args.episode_indices:
            if idx < 0 or idx >= len(all_episodes):
                raise IndexError(f"Episode index {idx} out of range "
                                 f"(total: {len(all_episodes)})")

        selected = [(idx, all_episodes[idx]) for idx in args.episode_indices]
        del all_episodes  # free memory

        for idx, ep in selected:
            T = len(ep["images"])
            obj_start = ep["obj_pose"][0, :3] if ep["obj_pose"].ndim == 2 else ep["obj_pose"][0]
            obj_end = ep["obj_pose"][-1, :3] if ep["obj_pose"].ndim == 2 else ep["obj_pose"][-1]
            print(f"  Episode {idx}: T={T}, "
                  f"obj start=({obj_start[0]:.3f},{obj_start[1]:.3f},{obj_start[2]:.3f}), "
                  f"obj end=({obj_end[0]:.3f},{obj_end[1]:.3f},{obj_end[2]:.3f})")

        # ── Create environment ──
        print("\nCreating environment...")
        env = make_env_with_camera(
            task_id=args.env_task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)

        env.reset()
        pre_position_gripper_down(env)

        # Create markers
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        marker_xy = region_center
        update_target_markers(
            place_markers, goal_markers,
            marker_xy, tuple(goal_xy), marker_z, env,
        )

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _settle(steps: int = 10):
            ee_hold = get_ee_pose_w(env)
            act = torch.zeros(1, env.action_space.shape[-1], device=device)
            act[0, :7] = ee_hold[0, :7]
            act[0, 7] = 1.0
            for _ in range(steps):
                env.step(act)

        # ── Replay each episode ──
        for ep_idx, ep in selected:
            print(f"\n{'='*60}")
            print(f"  Replaying Episode {ep_idx}")
            print(f"{'='*60}")

            # Extract action sequence
            if args.command_source == "action":
                action_seq = ep["action"]  # (T, 8)
            else:
                ee = ep["ee_pose"]
                gripper = ep.get("gripper", ep["action"][:, 7])
                action_seq = np.concatenate(
                    [ee[:, :7], gripper.reshape(-1, 1)], axis=1
                ).astype(np.float32)

            T_replay = len(action_seq)
            if args.horizon is not None:
                T_replay = min(T_replay, args.horizon)

            data_images = ep["images"]
            data_wrist_images = ep.get("wrist_images", None)
            data_obj_poses = ep["obj_pose"]

            # Get the obj start position from the reversed data
            obj_start = ep["obj_pose"][0]
            start_xy = (float(obj_start[0]), float(obj_start[1]))
            print(f"  Cube start from data: ({start_xy[0]:.4f}, {start_xy[1]:.4f})")
            print(f"  Action sequence length: {T_replay}")

            # Hard reset environment
            env.reset()
            pre_position_gripper_down(env)
            update_target_markers(
                place_markers, goal_markers,
                marker_xy, tuple(goal_xy), marker_z, env,
            )

            # Teleport cube to recorded start position
            obj_pose_t = torch.tensor(
                [start_xy[0], start_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose_t, name="object")
            _settle(steps=20)

            # Verify cube position
            init_obj = get_object_pose_w(env)[0].cpu().numpy()
            init_ee = get_ee_pose_w(env)[0].cpu().numpy()
            print(f"  After setup: obj=({init_obj[0]:.3f},{init_obj[1]:.3f},{init_obj[2]:.3f})  "
                  f"ee=({init_ee[0]:.3f},{init_ee[1]:.3f},{init_ee[2]:.3f})")

            # Replay actions and record
            sim_frames = []
            wrist_frames = []
            obj_poses_sim = []
            ee_poses_sim = []

            for t in range(T_replay):
                # Record observation
                table_rgb = table_camera.data.output["rgb"]
                if table_rgb.shape[-1] > 3:
                    table_rgb = table_rgb[..., :3]
                sim_frames.append(table_rgb.cpu().numpy().astype(np.uint8)[0])

                if wrist_camera is not None:
                    wrist_rgb = wrist_camera.data.output["rgb"]
                    if wrist_rgb.shape[-1] > 3:
                        wrist_rgb = wrist_rgb[..., :3]
                    wrist_frames.append(wrist_rgb.cpu().numpy().astype(np.uint8)[0])

                ee_pose = get_ee_pose_w(env)[0].cpu().numpy()
                obj_pose = get_object_pose_w(env)[0].cpu().numpy()
                ee_poses_sim.append(ee_pose)
                obj_poses_sim.append(obj_pose[:3])

                # Execute action
                act = action_seq[t]
                action_t = torch.from_numpy(act).float().unsqueeze(0).to(device)
                env.step(action_t)

                if (t + 1) % 100 == 0:
                    cur_obj = get_object_pose_w(env)[0].cpu().numpy()
                    print(f"    [Replay] Step {t+1}/{T_replay}  "
                          f"obj=({cur_obj[0]:.3f},{cur_obj[1]:.3f},{cur_obj[2]:.3f})")

            # Post-replay settle
            _settle(steps=20)

            # Check final position
            final_obj = get_object_pose_w(env)[0].cpu().numpy()
            # For Task B (reversed A): target is red region
            dist_to_region_center = float(np.linalg.norm(
                final_obj[:2] - np.array(region_center)))
            in_region = (abs(final_obj[0] - region_center[0]) <= region_size[0] / 2 and
                         abs(final_obj[1] - region_center[1]) <= region_size[1] / 2)
            is_low = final_obj[2] < 0.15

            print(f"\n  Final obj: ({final_obj[0]:.4f}, {final_obj[1]:.4f}, {final_obj[2]:.4f})")
            print(f"  Dist to region center: {dist_to_region_center:.4f}m")
            print(f"  In region: {in_region}  Low: {is_low}")
            if in_region and is_low:
                print(f"  >>> Task B SUCCESS")
            else:
                print(f"  >>> Task B FAILED")

            # Render video
            video_path = out_dir / f"replay_ep{ep_idx}.mp4"
            render_annotated_video(
                sim_frames=sim_frames,
                data_frames=data_images[:T_replay],
                obj_poses_sim=obj_poses_sim,
                obj_poses_data=data_obj_poses[:T_replay],
                ee_poses_sim=ee_poses_sim,
                actions=action_seq[:T_replay],
                ep_idx=ep_idx,
                out_path=video_path,
                goal_xy=tuple(args.goal_xy),
                region_center_xy=region_center,
                region_size_xy=region_size,
                fps=args.video_fps,
                wrist_frames_sim=wrist_frames if wrist_frames else None,
                wrist_frames_data=data_wrist_images[:T_replay] if data_wrist_images is not None else None,
            )

            # Also save sim-only video (just the sim replay frames)
            sim_only_path = out_dir / f"replay_ep{ep_idx}_sim_only.mp4"
            _write_video_h264(sim_frames, sim_only_path, fps=args.video_fps)

        env.close()
        print(f"\nAll replays complete. Videos saved to: {out_dir}")

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
