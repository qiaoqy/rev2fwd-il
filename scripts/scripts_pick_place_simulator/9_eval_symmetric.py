#!/usr/bin/env python3
"""Exp31: Symmetric evaluation with green/red rectangle markers.

Both Task A and Task B use region-based success checking:
  - Task A: cube from red rectangle → green rectangle (region check)
  - Task B: cube from green rectangle → red rectangle (region check)

The visual markers are two rectangles of identical size, symmetric about y=0,
matching the data collection environment in ``1_collect_task_B_symmetric.py``.

Usage:
    conda activate rev2fwd_il

    # Task A (5 episodes, with videos)
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/9_eval_symmetric.py \
        --policy data/pick_place_isaac_lab_simulation/exp31/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
        --task A --num_episodes 5 --run_id viz_A \
        --out_dir data/pick_place_isaac_lab_simulation/exp31 \
        --headless --render_videos

    # Task B (5 episodes, with videos)
    CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place_simulator/9_eval_symmetric.py \
        --policy data/pick_place_isaac_lab_simulation/exp31/weights/PP_B/checkpoints/checkpoints/last/pretrained_model \
        --task B --num_episodes 5 --run_id viz_B \
        --out_dir data/pick_place_isaac_lab_simulation/exp31 \
        --headless --render_videos

    # Large-scale eval (50 episodes, no videos)
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/9_eval_symmetric.py \
        --policy <checkpoint_path> \
        --task A --num_episodes 50 --run_id shard0 \
        --out_dir <out_dir> --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch


# =========================================================================
# Symmetric marker helpers
# =========================================================================

def create_symmetric_markers(num_envs, device, region_size_xy):
    """Create symmetric green + red rectangle markers (identical to data collection).

    Returns:
        (red_markers, green_markers, marker_z)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    marker_height = 0.002
    table_z = 0.0
    marker_z = table_z + marker_height / 2 + 0.001

    sx, sy = float(region_size_xy[0]), float(region_size_xy[1])

    # Red rectangle — place target zone
    red_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PlaceMarkers",
        markers={
            "place": sim_utils.CuboidCfg(
                size=(sx, sy, marker_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        },
    )
    red_markers = VisualizationMarkers(red_cfg)

    # Green rectangle — goal zone (same size!)
    green_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(sx, sy, marker_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
        },
    )
    green_markers = VisualizationMarkers(green_cfg)

    return red_markers, green_markers, marker_z


def update_symmetric_markers(red_markers, green_markers,
                             red_center_xy, green_center_xy,
                             marker_z, env):
    """Position both rectangle markers at their fixed centers."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins

    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)

    for markers, cx, cy in [
        (red_markers, float(red_center_xy[0]), float(red_center_xy[1])),
        (green_markers, float(green_center_xy[0]), float(green_center_xy[1])),
    ]:
        pos = torch.zeros((num_envs, 3), device=device)
        pos[:, 0] = cx
        pos[:, 1] = cy
        pos[:, 2] = marker_z
        pos_w = pos + env_origins
        markers.visualize(pos_w, identity_quat)


def is_xy_inside_region(xy, center_xy, size_xy):
    """Check if (x, y) is inside a rectangle region."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    half_x, half_y = sx * 0.5, sy * 0.5
    return bool(abs(xy[0] - cx) <= half_x and abs(xy[1] - cy) <= half_y)


def sample_xy_in_region(rng, center_xy, size_xy, cube_half_size=0.02):
    """Sample a random point inside a rectangle, shrunk by cube_half_size."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    half_x = max(sx * 0.5 - cube_half_size, 0.0)
    half_y = max(sy * 0.5 - cube_half_size, 0.0)
    x = float(rng.uniform(cx - half_x, cx + half_x))
    y = float(rng.uniform(cy - half_y, cy + half_y))
    return (x, y)


# =========================================================================
# Video rendering (simplified, with region overlays for both tasks)
# =========================================================================

def render_episode_video(
    frames, out_path, ep_index, success, success_step,
    obj_poses, ee_poses, actions,
    task_type, goal_xy, red_center_xy, green_center_xy, region_size_xy,
    wrist_frames=None, fps=30,
):
    """Render annotated MP4 showing table + wrist cameras with info overlay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(frames)
    H, W = frames[0].shape[:2]
    scale = max(1, 384 // W)
    sH, sW = H * scale, W * scale
    has_wrist = wrist_frames is not None and len(wrist_frames) > 0

    info_h = 90
    out_W = sW * 2 if has_wrist else sW
    out_H = sH + info_h

    goal_xy = np.array(goal_xy)
    target_xy = np.array(green_center_xy if task_type == "A" else red_center_xy)

    annotated = []
    for t in range(T):
        frame = np.zeros((out_H, out_W, 3), dtype=np.uint8)

        table_img = cv2.resize(frames[t], (sW, sH), interpolation=cv2.INTER_NEAREST)
        frame[:sH, :sW] = table_img

        if has_wrist and t < len(wrist_frames):
            wrist_img = cv2.resize(wrist_frames[t], (sW, sH), interpolation=cv2.INTER_NEAREST)
            frame[:sH, sW:] = wrist_img

        # Info overlay
        ee_xyz = ee_poses[t, :3] if t < len(ee_poses) else np.zeros(3)
        obj_xyz = obj_poses[t, :3] if t < len(obj_poses) else np.zeros(3)
        obj_xy = obj_xyz[:2]
        dist_to_target = np.linalg.norm(obj_xy - target_xy)
        gripper_val = actions[t, -1] if t < len(actions) else 0.0
        gripper_str = "OPEN" if gripper_val > 0.0 else "CLOSED"
        in_region = is_xy_inside_region(obj_xy, target_xy, region_size_xy)

        tag = "SUCCESS" if success else "FAIL"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, clr = 0.38, (200, 200, 200)
        y0 = sH + 4

        lines = [
            f"Ep{ep_index} Task{task_type} Step {t}/{T}  {tag}  (step@{success_step})" if success
            else f"Ep{ep_index} Task{task_type} Step {t}/{T}  {tag}",
            f"EE: ({ee_xyz[0]:.3f}, {ee_xyz[1]:.3f}, {ee_xyz[2]:.3f})  Grip: {gripper_str}",
            f"Obj: ({obj_xyz[0]:.3f}, {obj_xyz[1]:.3f}, {obj_xyz[2]:.3f})  Dist: {dist_to_target:.3f}  InRegion: {in_region}",
            f"Target region: center=({target_xy[0]:.2f}, {target_xy[1]:.2f}) size=({region_size_xy[0]:.1f}x{region_size_xy[1]:.1f})",
        ]
        for i, line in enumerate(lines):
            y = y0 + (i + 1) * 16
            cv2.putText(frame, line, (4, y), font, fs, clr, 1, cv2.LINE_AA)

        if success and success_step is not None and t >= success_step:
            cv2.rectangle(frame, (out_W - 100, y0), (out_W - 4, y0 + 20), (0, 200, 0), -1)
            cv2.putText(frame, "SUCCESS", (out_W - 96, y0 + 14), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        annotated.append(frame)

    imageio.mimsave(str(out_path), annotated, fps=fps)


# =========================================================================
# Argument parser
# =========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Symmetric evaluation for exp31 (green/red rectangles).",
    )
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--run_id", type=str, default="run0")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--render_videos", action="store_true",
                        help="Render MP4 video for every episode.")

    # Symmetric region parameters (must match data collection)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2],
                        help="Green rectangle center.")
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2],
                        help="Red rectangle center.")
    parser.add_argument("--region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3],
                        help="Size of both rectangles.")

    parser.add_argument("--env_task", type=str,
                        default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    if args.seed is None:
        args.seed = hash(args.run_id) % (2**31)
    return args


# =========================================================================
# Main
# =========================================================================

def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_ee_pose_w,
            get_object_pose_w,
            teleport_object_to_pose,
            pre_position_gripper_down,
        )

        # Load AlternatingTester and helpers
        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).resolve().parent.parent
                / "scripts_pick_place" / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        AlternatingTester = _alt_mod.AlternatingTester

        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        task_label = args.task
        green_xy = tuple(args.goal_xy)           # (0.5, -0.2)
        red_xy = tuple(args.red_region_center_xy) # (0.5, +0.2)
        region_sz = tuple(args.region_size_xy)     # (0.3, 0.3)

        print(f"\n{'='*60}")
        print(f"  Exp31 Symmetric Eval — Task {task_label} ({args.run_id})")
        print(f"  Policy: {args.policy}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Green rect: center={green_xy}, size={region_sz}")
        print(f"  Red   rect: center={red_xy}, size={region_sz}")
        if task_label == "A":
            print(f"  Task A: cube spawns in RED rect → must reach GREEN rect")
        else:
            print(f"  Task B: cube spawns in GREEN rect → must reach RED rect")
        print(f"{'='*60}\n")

        # ---- Load policy ----
        config = load_policy_config(args.policy)
        env = make_env_with_camera(
            task_id=args.env_task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        policy, preproc, postproc, _, n_act = load_policy_auto(
            args.policy, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy.eval()

        # Create AlternatingTester with region-based success for Task B
        # (Task A success will be overridden below)
        tester = AlternatingTester(
            env=env,
            policy_A=policy, preprocessor_A=preproc, postprocessor_A=postproc,
            policy_B=policy, preprocessor_B=preproc, postprocessor_B=postproc,
            n_action_steps_A=n_act, n_action_steps_B=n_act,
            goal_xy=green_xy,
            red_marker_shape="rectangle",
            red_marker_size_xy=region_sz,
            fix_red_marker_pose=True,
            taskA_source_mode="red_region",
            taskB_target_mode="red_region",
            red_region_center_xy=red_xy,
            red_region_size_xy=region_sz,
            height_threshold=0.15,
            distance_threshold=0.03,
            horizon=args.horizon,
            has_wrist_A=config["has_wrist"],
            has_wrist_B=config["has_wrist"],
            include_obj_pose_A=config["include_obj_pose"],
            include_obj_pose_B=config["include_obj_pose"],
            include_gripper_A=config["include_gripper"],
            include_gripper_B=config["include_gripper"],
        )

        # ---- Override Task A success check to use GREEN REGION ----
        # The default check_task_A_success uses point-distance to goal_xy.
        # We override it with region-based check (cube inside green rectangle).
        _green_xy = np.array(green_xy)
        _region_sz = region_sz

        def check_task_A_success_symmetric():
            obj_pose = get_object_pose_w(env)[0].cpu().numpy()
            obj_z = obj_pose[2]
            obj_xy = obj_pose[:2]
            is_z_low = obj_z < 0.15
            is_in_green = is_xy_inside_region(obj_xy, _green_xy, _region_sz)
            is_gripper_open = tester.current_gripper_state > 0.5
            return is_z_low and is_in_green and is_gripper_open

        tester.check_task_A_success = check_task_A_success_symmetric

        # ---- Initial env setup ----
        env.reset()
        pre_position_gripper_down(env)

        # Create SYMMETRIC rectangle markers (both are rectangles!)
        red_markers, green_markers, marker_z = create_symmetric_markers(
            num_envs=1, device=device, region_size_xy=region_sz,
        )
        tester.place_markers = red_markers
        tester.goal_markers = green_markers
        tester.marker_z = marker_z

        # Position markers at their fixed centers
        update_symmetric_markers(
            red_markers, green_markers, red_xy, green_xy, marker_z, env,
        )

        # Output dirs
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_dir = out_dir / f"sym_eval_videos_{task_label}_{args.run_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / f"sym_eval_{task_label}_{args.run_id}.json"

        # ---- Run episodes ----
        results = []
        episode_details = []
        start_time = time.time()

        for ep_idx in range(args.num_episodes):
            env.reset()
            pre_position_gripper_down(env)

            # Re-position markers after reset
            update_symmetric_markers(
                red_markers, green_markers, red_xy, green_xy, marker_z, env,
            )

            if task_label == "A":
                # Spawn cube at random position in RED rectangle
                spawn_xy = sample_xy_in_region(rng, red_xy, region_sz)
                tester.current_place_xy = spawn_xy
            else:
                # Spawn cube at random position in GREEN rectangle
                spawn_xy = sample_xy_in_region(rng, green_xy, region_sz)
                # For Task B, place target is a random point in red region
                place_xy = sample_xy_in_region(rng, red_xy, region_sz)
                tester.current_place_xy = place_xy

            obj_pose_t = torch.tensor(
                [spawn_xy[0], spawn_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose_t, name="object")

            # Settle
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

            print(f"\n  [Task {task_label}] Ep {ep_idx + 1}/{args.num_episodes}  "
                  f"spawn=({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f})")

            # Run task
            if task_label == "A":
                ep_data, success = tester.run_task_A()
            else:
                ep_data, success = tester.run_task_B()

            results.append(success)
            rate = sum(results) / len(results) * 100
            print(f"    {'SUCCESS' if success else 'FAILED'}  "
                  f"(running: {sum(results)}/{len(results)} = {rate:.1f}%)")

            # Episode detail
            detail = {
                "episode_index": ep_idx,
                "success": bool(success),
                "success_step": ep_data.get("success_step"),
                "total_steps": len(ep_data.get("images", [])),
                "spawn_xy": list(spawn_xy),
            }
            if "obj_pose" in ep_data and len(ep_data["obj_pose"]) > 0:
                final_xy = ep_data["obj_pose"][-1][:2]
                target_center = green_xy if task_label == "A" else red_xy
                detail["final_obj_xy"] = final_xy.tolist()
                detail["final_in_region"] = is_xy_inside_region(
                    final_xy, target_center, region_sz)
            episode_details.append(detail)

            # Render video
            if args.render_videos:
                tag = "success" if success else "fail"
                vid_path = video_dir / f"{tag}_ep{ep_idx:03d}.mp4"
                print(f"    Rendering video -> {vid_path}")
                wrist_imgs = (list(ep_data["wrist_images"])
                              if ep_data.get("wrist_images") is not None else None)
                render_episode_video(
                    frames=list(ep_data["images"]),
                    out_path=vid_path,
                    ep_index=ep_idx,
                    success=success,
                    success_step=ep_data.get("success_step"),
                    obj_poses=ep_data["obj_pose"],
                    ee_poses=ep_data["ee_pose"],
                    actions=ep_data["action"],
                    task_type=task_label,
                    goal_xy=green_xy,
                    red_center_xy=red_xy,
                    green_center_xy=green_xy,
                    region_size_xy=region_sz,
                    wrist_frames=wrist_imgs,
                    fps=args.video_fps,
                )

        elapsed = time.time() - start_time

        # ---- Summary ----
        n_success = sum(results)
        n_total = len(results)
        rate = n_success / n_total if n_total else 0.0

        print(f"\n{'='*60}")
        print(f"  Exp31 Symmetric Eval — Task {task_label} ({args.run_id})")
        print(f"{'='*60}")
        print(f"  Total: {n_total}  Success: {n_success}  Failed: {n_total - n_success}")
        print(f"  Success rate: {rate:.1%}")
        print(f"  Elapsed: {elapsed:.1f}s")

        stats = {
            "experiment": f"exp31_symmetric_eval_task_{task_label}",
            "timestamp": datetime.now().isoformat(),
            "run_id": args.run_id,
            "config": {
                "policy": args.policy,
                "task": task_label,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "region_size_xy": list(args.region_size_xy),
                "seed": args.seed,
                "success_criteria": "region_based (cube center inside rectangle)",
            },
            "summary": {
                "total_episodes": n_total,
                "success_count": n_success,
                "fail_count": n_total - n_success,
                "success_rate": rate,
            },
            "episodes": episode_details,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats: {stats_path}")
        if args.render_videos:
            print(f"  Videos: {video_dir}/")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
