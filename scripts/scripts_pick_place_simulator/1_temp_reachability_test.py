#!/usr/bin/env python3
"""Reachability test: check if the robot arm can reach key waypoints in/above the red region.

This script commands the robot to a grid of key positions (red region corners,
edges, center, and hover points above them) one by one, records the commanded
vs. actual EE pose, and saves images + video for visual inspection.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_temp_reachability_test.py \
        --out data/pick_place_isaac_lab_simulation/exp16/reachability_test.npz \
        --headless
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test robot arm reachability for red region waypoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    parser.add_argument("--settle_steps", type=int, default=60,
                        help="Steps to hold at each waypoint for the arm to converge.")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    # Region parameters (same defaults as 1_collect_task_B.py)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2, default=[0.3, 0.3])
    parser.add_argument("--hover_z", type=float, default=0.25)
    parser.add_argument("--place_z", type=float, default=0.055)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def build_waypoints(args):
    """Build a list of (name, x, y, z) waypoints to test."""
    cx, cy = args.red_region_center_xy
    sx, sy = args.red_region_size_xy
    half_x, half_y = sx / 2, sy / 2
    hover_z = args.hover_z
    place_z = args.place_z
    gx, gy = args.goal_xy

    # Corners and edges of the red region
    region_pts = {
        "center":       (cx, cy),
        "top_left":     (cx - half_x, cy + half_y),
        "top_right":    (cx + half_x, cy + half_y),
        "bot_left":     (cx - half_x, cy - half_y),
        "bot_right":    (cx + half_x, cy - half_y),
        "mid_top":      (cx, cy + half_y),
        "mid_bot":      (cx, cy - half_y),
        "mid_left":     (cx - half_x, cy),
        "mid_right":    (cx + half_x, cy),
    }

    waypoints = []

    # Goal position (green marker)
    waypoints.append(("goal_hover", gx, gy, hover_z))
    waypoints.append(("goal_surface", gx, gy, place_z))
    waypoints.append(("goal_hover_back", gx, gy, hover_z))

    # For each region point: hover above, then descend to surface, then back up
    for name, (px, py) in region_pts.items():
        waypoints.append((f"red_{name}_hover", px, py, hover_z))
        waypoints.append((f"red_{name}_surface", px, py, place_z))
        waypoints.append((f"red_{name}_hover_back", px, py, hover_z))

    # Extra: high approach points (hover_z + 0.08)
    high_z = hover_z + 0.08
    for name in ["center", "top_left", "top_right", "bot_left", "bot_right"]:
        px, py = region_pts[name]
        waypoints.append((f"red_{name}_high", px, py, high_z))

    return waypoints


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- imports that require Isaac Sim running ----
    import importlib.util
    from pathlib import Path

    import numpy as np
    import torch

    from rev2fwd_il.sim.scene_api import get_ee_pose_w
    from rev2fwd_il.utils.seed import set_seed

    # Reuse env creation from the original collect script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)
    make_env_with_camera = _orig.make_env_with_camera
    create_target_markers = _orig.create_target_markers
    update_target_markers = _orig.update_target_markers

    try:
        set_seed(42)

        # ---- Create environment (single env) ----
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=1,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=1000.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device

        # Camera sensors
        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

        # ---- Reset environment ----
        obs_dict, _ = env.reset()

        # Gripper-down quaternion (wxyz)
        grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

        # ---- Pre-position robot to gripper-down rest pose ----
        ee_pose = get_ee_pose_w(env)
        rest_action = torch.zeros(1, env.action_space.shape[-1], device=device)
        rest_action[0, :3] = ee_pose[0, :3]
        rest_action[0, 3:7] = grasp_quat
        rest_action[0, 7] = 1.0  # gripper open
        print("Pre-positioning robot to gripper-down pose...")
        for _ in range(100):
            env.step(rest_action)
        print("Pre-positioning done.")

        # ---- Create visual markers for red region and goal ----
        start_markers, goal_markers, marker_z = create_target_markers(
            1, device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        update_target_markers(
            start_markers, goal_markers,
            start_xys=[tuple(args.red_region_center_xy)],
            goal_xy=tuple(args.goal_xy),
            marker_z=marker_z,
            env=env,
        )
        # Settle marker rendering
        for _ in range(5):
            env.step(rest_action)

        # ---- Build waypoints ----
        waypoints = build_waypoints(args)
        print(f"\nTesting {len(waypoints)} waypoints:\n")
        for i, (name, x, y, z) in enumerate(waypoints):
            print(f"  [{i:2d}] {name:30s}  target=({x:.3f}, {y:.3f}, {z:.3f})")

        # ---- Test each waypoint ----
        results = []
        all_table_images = []
        all_wrist_images = []

        for wp_idx, (name, tx, ty, tz) in enumerate(waypoints):
            print(f"\n--- Waypoint {wp_idx}/{len(waypoints)-1}: {name} -> ({tx:.3f}, {ty:.3f}, {tz:.3f})")

            # Command robot to target
            action = torch.zeros(1, env.action_space.shape[-1], device=device)
            action[0, 0] = tx
            action[0, 1] = ty
            action[0, 2] = tz
            action[0, 3:7] = grasp_quat
            action[0, 7] = 1.0  # gripper open

            # Step and record convergence
            for step in range(args.settle_steps):
                obs_dict, _, _, _, _ = env.step(action)

                ee_now = get_ee_pose_w(env)
                actual = ee_now[0, :3].cpu().numpy()
                err = np.sqrt(np.sum((actual - np.array([tx, ty, tz])) ** 2))

                # Capture images at last few steps
                if step >= args.settle_steps - 5:
                    table_rgb = table_camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                    wrist_rgb = wrist_camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                    all_table_images.append(table_rgb)
                    all_wrist_images.append(wrist_rgb)

            # Final measurement
            ee_final = get_ee_pose_w(env)
            actual_final = ee_final[0, :3].cpu().numpy()
            pos_err = np.sqrt(np.sum((actual_final - np.array([tx, ty, tz])) ** 2))
            reached = pos_err < 0.02  # 2cm threshold

            result = {
                "name": name,
                "target_xyz": np.array([tx, ty, tz]),
                "actual_xyz": actual_final.copy(),
                "pos_error": pos_err,
                "reached": reached,
            }
            results.append(result)

            status = "OK" if reached else "FAIL"
            print(f"    actual=({actual_final[0]:.4f}, {actual_final[1]:.4f}, {actual_final[2]:.4f})  "
                  f"err={pos_err:.4f}m  [{status}]")

        # ---- Summary ----
        print(f"\n{'='*70}")
        print(f"REACHABILITY TEST SUMMARY")
        print(f"{'='*70}")
        n_ok = sum(1 for r in results if r["reached"])
        n_fail = len(results) - n_ok
        print(f"  Passed: {n_ok}/{len(results)}  |  Failed: {n_fail}/{len(results)}")
        print()
        if n_fail > 0:
            print("FAILED waypoints:")
            for r in results:
                if not r["reached"]:
                    t = r["target_xyz"]
                    a = r["actual_xyz"]
                    print(f"  {r['name']:30s}  target=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})  "
                          f"actual=({a[0]:.4f},{a[1]:.4f},{a[2]:.4f})  err={r['pos_error']:.4f}m")
        print(f"{'='*70}\n")

        # ---- Save data ----
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build episode-like dict for the inspect script
        table_imgs = np.stack(all_table_images, axis=0)  # (N, H, W, 3)
        wrist_imgs = np.stack(all_wrist_images, axis=0)  # (N, H, W, 3)

        # Build a dummy obs/ee_pose/action timeline from results for compatibility
        n_frames = len(table_imgs)
        dummy_obs = np.zeros((n_frames, 36), dtype=np.float32)
        dummy_ee = np.zeros((n_frames, 7), dtype=np.float32)
        dummy_gripper = np.ones(n_frames, dtype=np.float32)
        dummy_actions = np.zeros((n_frames, 8), dtype=np.float32)

        episode = {
            "obs": dummy_obs,
            "images": table_imgs,
            "wrist_images": wrist_imgs,
            "ee_pose": dummy_ee,
            "obj_pose": np.zeros((n_frames, 7), dtype=np.float32),
            "gripper": dummy_gripper,
            "place_pose": np.array(args.red_region_center_xy + [args.place_z, 1, 0, 0, 0], dtype=np.float32),
            "goal_pose": np.array(args.goal_xy + [args.place_z, 1, 0, 0, 0], dtype=np.float32),
            "success": n_fail == 0,
            "actions": dummy_actions,
        }

        # Fill ee_pose with actual data (repeat result per 5 image frames)
        frame_i = 0
        for r in results:
            for _ in range(5):
                if frame_i < n_frames:
                    dummy_ee[frame_i, :3] = r["actual_xyz"]
                    dummy_ee[frame_i, 3] = 0.0
                    dummy_ee[frame_i, 4] = 1.0
                    frame_i += 1

        np.savez(str(out_path), episodes=np.array([episode], dtype=object))
        print(f"Saved data to {out_path}")
        print(f"  Images: {table_imgs.shape} (table), {wrist_imgs.shape} (wrist)")

        # Also save a human-readable text report
        report_path = out_path.with_suffix(".txt")
        with open(report_path, "w") as f:
            f.write("REACHABILITY TEST REPORT\n")
            f.write(f"Red region center: {args.red_region_center_xy}\n")
            f.write(f"Red region size: {args.red_region_size_xy}\n")
            f.write(f"Hover Z: {args.hover_z}\n")
            f.write(f"Place Z: {args.place_z}\n")
            f.write(f"Settle steps per waypoint: {args.settle_steps}\n\n")
            f.write(f"{'Name':30s}  {'Target XYZ':30s}  {'Actual XYZ':30s}  {'Error':8s}  Status\n")
            f.write("-" * 110 + "\n")
            for r in results:
                t = r["target_xyz"]
                a = r["actual_xyz"]
                status = "OK" if r["reached"] else "FAIL"
                f.write(f"{r['name']:30s}  ({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})          "
                        f"({a[0]:.4f},{a[1]:.4f},{a[2]:.4f})          {r['pos_error']:.4f}m    {status}\n")
            f.write(f"\nPassed: {n_ok}/{len(results)}  Failed: {n_fail}/{len(results)}\n")
        print(f"Saved report to {report_path}")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
