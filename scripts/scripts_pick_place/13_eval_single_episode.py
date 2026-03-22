#!/usr/bin/env python3
"""Run N Task A episodes, render every episode as an annotated video.

Modes:
  --mode all    (default) Run exactly N episodes, save all videos.
  --mode first  Stop after finding the first success (or failure via --want).

Usage:
  # Run 20 episodes, save all videos:
  CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place/13_eval_single_episode.py \
      --policy_A data/.../iter9_ckpt_A \
      --out_dir data/.../videos_iter9 \
      --num_episodes 20 --headless

  # Stop after first success:
  CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place/13_eval_single_episode.py \
      --policy_A data/.../iter9_ckpt_A \
      --out_dir data/.../exp5 \
      --mode first --want success --headless
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch


# ── video helpers ───────────────────────────────────────────────────────────

def _overlay(img, text, pos=(4, 16), fs=0.45, color=(255, 255, 255), bg=(0, 0, 0)):
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, 1)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + bl + 2), bg, -1)
    cv2.putText(img, text, (x, y), font, fs, color, 1, cv2.LINE_AA)
    return img


def _multi_overlay(img, lines, start_y=4, lh=16, fs=0.40,
                   color=(255, 255, 255), bg=(0, 0, 0)):
    for i, t in enumerate(lines):
        img = _overlay(img, t, (4, start_y + (i + 1) * lh), fs, color, bg)
    return img


def render_video(frames, out_path, ep_idx, success, success_step,
                 target_xy, obj_poses, ee_poses, actions,
                 dist_thr, goal_xy, fps=30):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(frames)
    H, W = frames[0].shape[:2]
    scale = max(1, 384 // W)
    oH, oW = H * scale, W * scale
    ann = []
    for t in range(T):
        f = frames[t]
        if scale > 1:
            f = cv2.resize(f, (oW, oH), interpolation=cv2.INTER_NEAREST)
        obj_xy = obj_poses[t, :2] if t < len(obj_poses) else [0, 0]
        obj_z  = obj_poses[t, 2]  if t < len(obj_poses) else 0
        grip   = actions[t, 7]    if t < len(actions)   else 0
        d2t = np.linalg.norm(np.array(obj_xy) - np.array(target_xy))
        lines = [
            f"Ep {ep_idx} | Step {t+1}/{T}",
            f"Target: ({target_xy[0]:.3f}, {target_xy[1]:.3f})",
            f"ObjXY: ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})  Z:{obj_z:.3f}",
            f"Dist: {d2t:.4f}  Thr: {dist_thr}",
            f"Grip: {'OPEN' if grip > 0.5 else 'CLOSED'} ({grip:.2f})",
        ]
        if success and success_step is not None and t + 1 >= success_step:
            lines.append("STATUS: SUCCESS")
        else:
            lines.append("STATUS: SUCCESS" if success else "STATUS: RUNNING...")
        f = _multi_overlay(f, lines, 2, int(14 * scale / 3 + 2),
                           0.35 * scale / 3 + 0.1)
        bar_h = max(4, scale * 2)
        bar_y = oH - bar_h - 2
        p = (t + 1) / T
        bw = int(p * (oW - 8))
        bc = (0, 200, 0) if success else (0, 100, 200)
        cv2.rectangle(f, (4, bar_y), (4 + bw, bar_y + bar_h), bc, -1)
        cv2.rectangle(f, (4, bar_y), (oW - 4, bar_y + bar_h), (128, 128, 128), 1)
        if t == T - 1:
            rt = f"SUCCESS (step {success_step})" if success else "FAILED"
            rc = (0, 255, 0) if success else (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fss = 0.6 * scale / 3 + 0.2
            (tw, th), _ = cv2.getTextSize(rt, font, fss, 2)
            cx, cy = (oW - tw) // 2, oH // 2
            cv2.rectangle(f, (cx - 6, cy - th - 6), (cx + tw + 6, cy + 10), (0,0,0), -1)
            cv2.putText(f, rt, (cx, cy), font, fss, rc, 2, cv2.LINE_AA)
        ann.append(f)
    imageio.mimsave(str(out_path), ann, fps=fps)


# ── main ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run Task A episodes and render annotated videos.")
    p.add_argument("--policy_A", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", choices=["all", "first"], default="all",
                   help="'all': run N episodes. 'first': stop after first match.")
    p.add_argument("--want", choices=["success", "failure"], default="success",
                   help="(mode=first only) Stop after first success or failure.")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="Number of episodes to run (mode=all) or max attempts (mode=first).")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--distance_threshold", type=float, default=0.03)
    p.add_argument("--height_threshold", type=float, default=0.15)
    p.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])
    p.add_argument("--n_action_steps", type=int, default=16)
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--task", default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    p.add_argument("--image_width", type=int, default=128)
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    # Red region / marker args
    p.add_argument("--taskA_source_mode", type=str, default="legacy",
                   choices=["legacy", "green_region", "red_region"])
    p.add_argument("--red_region_center_xy", type=float, nargs=2, default=None)
    p.add_argument("--red_region_size_xy", type=float, nargs=2, default=None)
    p.add_argument("--red_marker_shape", type=str, default="circle",
                   choices=["circle", "rectangle"])
    p.add_argument("--red_marker_size_xy", type=float, nargs=2, default=None)
    p.add_argument("--fix_red_marker_pose", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    return args


def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, pre_position_gripper_down, teleport_object_to_pose

        _spec = importlib.util.spec_from_file_location(
            "test_alternating", str(Path(__file__).parent / "6_test_alternating.py"))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        make_env_with_camera = _mod.make_env_with_camera
        load_policy_config   = _mod.load_policy_config
        load_diffusion_policy = _mod.load_diffusion_policy
        AlternatingTester    = _mod.AlternatingTester
        create_target_markers = _mod.create_target_markers
        update_target_markers = _mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        goal_xy = np.array(args.goal_xy)

        print(f"\n{'='*60}")
        print(f"  Task A Evaluation  (mode={args.mode}, episodes={args.num_episodes})")
        print(f"  Policy: {args.policy_A}")
        print(f"  Output: {args.out_dir}")
        print(f"{'='*60}\n")

        cfg_A = load_policy_config(args.policy_A)
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )
        pol_A, pre_A, post_A, _, n_act_A = load_diffusion_policy(
            args.policy_A, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        pol_A.eval()

        tester = AlternatingTester(
            env=env,
            policy_A=pol_A, preprocessor_A=pre_A, postprocessor_A=post_A,
            policy_B=pol_A, preprocessor_B=pre_A, postprocessor_B=post_A,
            n_action_steps_A=n_act_A, n_action_steps_B=n_act_A,
            goal_xy=tuple(args.goal_xy),
            height_threshold=args.height_threshold,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=cfg_A["has_wrist"], has_wrist_B=cfg_A["has_wrist"],
            include_obj_pose_A=cfg_A["include_obj_pose"],
            include_obj_pose_B=cfg_A["include_obj_pose"],
            include_gripper_A=cfg_A["include_gripper"],
            include_gripper_B=cfg_A["include_gripper"],
            taskA_source_mode=args.taskA_source_mode,
            red_region_center_xy=args.red_region_center_xy,
            red_region_size_xy=args.red_region_size_xy,
            red_marker_shape=args.red_marker_shape,
            red_marker_size_xy=args.red_marker_size_xy,
            fix_red_marker_pose=bool(args.fix_red_marker_pose),
        )

        env.reset()
        pre_position_gripper_down(env)
        pm, gm, mz = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape=args.red_marker_shape,
            red_marker_size_xy=args.red_marker_size_xy,
        )
        tester.place_markers = pm
        tester.goal_markers = gm
        tester.marker_z = mz
        first_place = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place
        marker_xy = first_place
        if args.fix_red_marker_pose and args.red_region_center_xy is not None:
            marker_xy = tuple(args.red_region_center_xy)
        update_target_markers(pm, gm, marker_xy, tuple(goal_xy), mz, env)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        start_time = time.time()

        for ep in range(args.num_episodes):
            # Hard reset for Task A: object at random position
            env.reset()
            pre_position_gripper_down(env)
            rand_xy = tester._sample_taskA_source_target()
            marker_xy_ep = rand_xy
            if args.fix_red_marker_pose and args.red_region_center_xy is not None:
                marker_xy_ep = tuple(args.red_region_center_xy)
            update_target_markers(pm, gm, marker_xy_ep, tuple(goal_xy), mz, env)
            obj_pose = torch.tensor(
                [rand_xy[0], rand_xy[1], 0.022, 1, 0, 0, 0],
                dtype=torch.float32, device=device).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[:, :7] = ee_hold[:, :7]
            hold_action[:, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            pol_A.reset()
            tester.video_frames = []

            print(f"  [Task A] Ep {ep+1}/{args.num_episodes}  "
                  f"obj=[{rand_xy[0]:.3f},{rand_xy[1]:.3f}]  goal=[{goal_xy[0]:.3f},{goal_xy[1]:.3f}]")

            ep_data, success = tester.run_task_A()
            results.append(success)
            tag = "SUCCESS" if success else "FAILED"
            rate = sum(results) / len(results) * 100
            print(f"    {tag}  (running: {sum(results)}/{len(results)} = {rate:.1f}%)")

            # Render video for this episode
            label = "success" if success else "fail"
            vid_path = out_dir / f"ep{ep:02d}_{label}.mp4"
            print(f"    Rendering → {vid_path}")
            render_video(
                frames=list(ep_data["images"]),
                out_path=vid_path, ep_idx=ep, success=success,
                success_step=ep_data.get("success_step"),
                target_xy=(float(goal_xy[0]), float(goal_xy[1])),
                obj_poses=ep_data["obj_pose"], ee_poses=ep_data["ee_pose"],
                actions=ep_data["action"], dist_thr=args.distance_threshold,
                goal_xy=tuple(args.goal_xy), fps=args.video_fps,
            )

            # mode=first: stop after finding the wanted outcome
            if args.mode == "first" and success == (args.want == "success"):
                print(f"\n  Found {args.want}! Stopping early.")
                break

        elapsed = time.time() - start_time
        n_success = sum(results)
        n_total = len(results)

        print(f"\n{'='*60}")
        print(f"  Results: {n_success}/{n_total} success ({n_success/n_total*100:.1f}%)")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Videos:  {out_dir}/")
        print(f"{'='*60}")

        # Save summary JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "policy": args.policy_A,
            "num_episodes": n_total,
            "success_count": n_success,
            "success_rate": n_success / n_total if n_total else 0,
            "elapsed_seconds": elapsed,
            "episodes": [{"ep": i, "success": bool(r)} for i, r in enumerate(results)],
        }
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary: {summary_path}")

        env.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
