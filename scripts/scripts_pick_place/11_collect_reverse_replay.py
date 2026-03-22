#!/usr/bin/env python3
"""Step 11: Combined Policy-B rollout + time-reverse + replay for exp15.

This script runs N cycles of:
  1. Policy B rollout: cube at goal → Policy B picks → places at red region.
     Record trajectory as Task B data.
  2. Time-reverse the successful episode offline (no sim needed).
  3. Replay the reversed actions in sim: cube starts where B left it (red
     region) → reversed replay drives it back toward the goal.
     Record observations as Task A data.

After each cycle the cube should be back near the goal, ready for the next
Policy B rollout.  If Policy B fails, the environment is hard-reset and the
failed episode is discarded. If the replay doesn't bring the cube accurately
back to the goal, both the Task B and Task A episodes are still saved, but
a hard-reset is performed before the next cycle.

=============================================================================
OUTPUTS
=============================================================================
- ``--out_B``:  NPZ with successful Policy B rollout episodes.
- ``--out_A``:  NPZ with replayed Task A episodes (from reversed replay).
- ``<out_B>.stats.json``: Per-cycle statistics and success rates.

=============================================================================
USAGE
=============================================================================
CUDA_VISIBLE_DEVICES=5 python scripts/scripts_pick_place/11_collect_reverse_replay.py \\
    --policy_B runs/.../pretrained_model \\
    --out_B data/pick_place_isaac_lab_simulation/exp15/iter1_policyB_rollout.npz \\
    --out_A data/pick_place_isaac_lab_simulation/exp15/iter1_replay_taskA.npz \\
    --num_cycles 50 --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


# ── arg parsing ──────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Policy B rollout + time-reverse + replay for reverse-replay reset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--policy_B", type=str, required=True,
                        help="Path to Policy B pretrained_model checkpoint.")
    parser.add_argument("--out_B", type=str, required=True,
                        help="Output NPZ for Task B rollout episodes (successful).")
    parser.add_argument("--out_A", type=str, required=True,
                        help="Output NPZ for replayed Task A episodes.")

    parser.add_argument("--num_cycles", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])

    # Region parameters
    parser.add_argument("--taskB_target_mode", type=str, default="red_region",
                        choices=["legacy", "red_region"])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.45, 0.15])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.12, 0.10])
    parser.add_argument("--red_marker_shape", type=str, default="rectangle",
                        choices=["circle", "rectangle"])
    parser.add_argument("--red_marker_size_xy", type=float, nargs=2,
                        default=[0.12, 0.10])
    parser.add_argument("--fix_red_marker_pose", type=int, default=1,
                        choices=[0, 1])

    # Env / sim
    parser.add_argument("--task", type=str,
                        default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    # Isaac Lab AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ── time-reversal (from 3_1_make_forward_data.py) ────────────────────────
def reverse_episode(ep: dict) -> dict:
    """Time-reverse an episode.  action[t][:7] = ee_pose[t+1] after reversal."""
    T = len(ep["images"])

    images_rev = ep["images"][::-1].copy()
    ee_rev = ep["ee_pose"][::-1].copy()
    obj_rev = ep["obj_pose"][::-1].copy()
    gripper_rev = ep["gripper"][::-1].copy()

    has_wrist = "wrist_images" in ep
    if has_wrist:
        wrist_rev = ep["wrist_images"][::-1].copy()

    new_actions = np.zeros((T, 8), dtype=np.float32)
    new_actions[:T - 1, :7] = ee_rev[1:]        # next-frame ee_pose
    new_actions[T - 1, :7] = ee_rev[T - 1]      # last: hold in place
    new_actions[:, 7] = gripper_rev

    # drop last frame (no valid next-frame target)
    result = {
        "images":   images_rev[:-1],
        "ee_pose":  ee_rev[:-1].astype(np.float32),
        "obj_pose": obj_rev[:-1].astype(np.float32),
        "action":   new_actions[:-1].astype(np.float32),
        "gripper":  gripper_rev[:-1].astype(np.float32),
    }
    if has_wrist:
        result["wrist_images"] = wrist_rev[:-1]
    if "place_pose" in ep:
        result["place_pose"] = ep["place_pose"].copy()
    if "goal_pose" in ep:
        result["goal_pose"] = ep["goal_pose"].copy()
    return result


# ── region helpers ───────────────────────────────────────────────────────
def _is_xy_inside_region(xy, center_xy, size_xy) -> bool:
    """Check whether cube center falls inside the region (no shrink)."""
    cube_half_size = 0.0  # No shrink — full region acceptance
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    half_x = max(sx * 0.5 - cube_half_size, 0.0)
    half_y = max(sy * 0.5 - cube_half_size, 0.0)
    return bool(abs(xy[0] - cx) <= half_x and abs(xy[1] - cy) <= half_y)


def _sample_xy_in_rectangle(rng, center_xy, size_xy):
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    x = rng.uniform(cx - sx * 0.5, cx + sx * 0.5)
    y = rng.uniform(cy - sy * 0.5, cy + sy * 0.5)
    return (x, y)


# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # Launch Isaac Sim
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

        # -- reuse utilities from 6_test_alternating.py --
        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).parent / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        create_target_markers = _alt_mod.create_target_markers
        update_target_markers = _alt_mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        goal_xy = np.array(args.goal_xy)
        rng = np.random.default_rng(args.seed)
        region_center = tuple(args.red_region_center_xy)
        region_size = tuple(args.red_region_size_xy)

        # == load policy B ==
        print(f"\n{'=' * 60}")
        print("Loading Policy B …")
        print(f"{'=' * 60}")
        config_B = load_policy_config(args.policy_B)
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )
        policy_B, preproc_B, postproc_B, _, n_act_B = load_policy_auto(
            args.policy_B, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()

        include_obj_pose_B = config_B["include_obj_pose"]
        include_gripper_B = config_B["include_gripper"]
        has_wrist_B = config_B["has_wrist"]

        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)

        # == initial env setup ==
        env.reset()
        pre_position_gripper_down(env)

        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape=args.red_marker_shape,
            red_marker_size_xy=tuple(args.red_marker_size_xy) if args.red_marker_size_xy else None,
        )

        # place marker at region center (for visual display)
        marker_xy = region_center if args.fix_red_marker_pose else region_center
        update_target_markers(
            place_markers, goal_markers,
            place_xy=marker_xy,
            goal_xy=tuple(goal_xy),
            marker_z=marker_z, env=env,
        )

        # ──────────────────────────────────────────────────────────
        #  Observation helper (mirrors AlternatingTester._get_observation)
        # ──────────────────────────────────────────────────────────
        current_gripper_state = 1.0  # open

        def _get_obs():
            nonlocal current_gripper_state
            table_rgb = table_camera.data.output["rgb"]
            if table_rgb.shape[-1] > 3:
                table_rgb = table_rgb[..., :3]
            table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)[0]
            wrist_rgb_np = None
            if wrist_camera is not None:
                wrist_rgb = wrist_camera.data.output["rgb"]
                if wrist_rgb.shape[-1] > 3:
                    wrist_rgb = wrist_rgb[..., :3]
                wrist_rgb_np = wrist_rgb.cpu().numpy().astype(np.uint8)[0]
            ee = get_ee_pose_w(env)[0].cpu().numpy()
            obj = get_object_pose_w(env)[0].cpu().numpy()
            return table_rgb_np, wrist_rgb_np, ee, obj, current_gripper_state

        def _prepare_input(table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state):
            table_t = torch.from_numpy(table_rgb).float().div(255.0)
            table_t = table_t.permute(2, 0, 1).unsqueeze(0).to(device)
            ee_t = torch.from_numpy(ee_pose).float().unsqueeze(0).to(device)
            parts = [ee_t]
            if include_obj_pose_B:
                parts.append(torch.from_numpy(obj_pose).float().unsqueeze(0).to(device))
            if include_gripper_B:
                parts.append(torch.tensor([[gripper_state]], dtype=torch.float32, device=device))
            state = torch.cat(parts, dim=-1)
            inp = {"observation.image": table_t, "observation.state": state}
            if wrist_rgb is not None and has_wrist_B:
                w = torch.from_numpy(wrist_rgb).float().div(255.0)
                w = w.permute(2, 0, 1).unsqueeze(0).to(device)
                inp["observation.wrist_image"] = w
            return inp

        def _hard_reset_to_goal():
            """Reset env and place cube at goal for the next Policy B rollout."""
            nonlocal current_gripper_state
            env.reset()
            pre_position_gripper_down(env)
            update_target_markers(place_markers, goal_markers,
                                  place_xy=marker_xy, goal_xy=tuple(goal_xy),
                                  marker_z=marker_z, env=env)
            obj_pose_t = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose_t, name="object")
            _settle()
            current_gripper_state = 1.0
            policy_B.reset()

        def _settle(steps: int = 10):
            ee_hold = get_ee_pose_w(env)
            act = torch.zeros(1, env.action_space.shape[-1], device=device)
            act[0, :7] = ee_hold[0, :7]
            act[0, 7] = 1.0
            for _ in range(steps):
                env.step(act)

        # ── main loop ────────────────────────────────────────────────
        episodes_B = []
        episodes_A = []
        results_B = []
        results_A_reset = []  # whether replay brought cube back near goal
        start_time = time.time()

        for cycle in range(args.num_cycles):
            print(f"\n{'=' * 50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'=' * 50}")

            # ──────────────── Phase 1: Policy B rollout ────────────────
            # Teleport cube to goal
            _hard_reset_to_goal()
            # Debug: verify cube is at goal
            dbg_init_obj = get_object_pose_w(env)[0].cpu().numpy()
            dbg_init_ee = get_ee_pose_w(env)[0].cpu().numpy()
            print(f"  After reset: obj=({dbg_init_obj[0]:.3f},{dbg_init_obj[1]:.3f},{dbg_init_obj[2]:.3f})  "
                  f"ee=({dbg_init_ee[0]:.3f},{dbg_init_ee[1]:.3f},{dbg_init_ee[2]:.3f})")

            # Optionally sample a target inside the red region (for marker)
            if not args.fix_red_marker_pose:
                sample_xy = _sample_xy_in_rectangle(rng, region_center, region_size)
                update_target_markers(place_markers, goal_markers,
                                      place_xy=sample_xy, goal_xy=tuple(goal_xy),
                                      marker_z=marker_z, env=env)

            # Build place_pose / goal_pose for episode metadata
            goal_pose_7 = np.array(
                [goal_xy[0], goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            place_pose_7 = np.array(
                [region_center[0], region_center[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            policy_B.reset()
            current_gripper_state = 1.0

            imgs, wrists, ees, objs, acts, grips = [], [], [], [], [], []
            b_success = False
            b_success_step = None

            print("  Running Policy B (goal → red region) …")
            for t in range(args.horizon):
                table_rgb, wrist_rgb, ee_pose, obj_pose, gs = _get_obs()
                imgs.append(table_rgb)
                if wrist_rgb is not None:
                    wrists.append(wrist_rgb)
                ees.append(ee_pose)
                objs.append(obj_pose)
                grips.append(np.float32(gs))

                inp = _prepare_input(table_rgb, wrist_rgb, ee_pose, obj_pose, gs)
                if preproc_B is not None:
                    inp = preproc_B(inp)

                with torch.no_grad():
                    action = policy_B.select_action(inp)
                if postproc_B is not None:
                    action = postproc_B(action)

                action_np = action[0].cpu().numpy()
                acts.append(action_np)
                current_gripper_state = float(action_np[7])

                action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(device)
                env.step(action_t)

                # success check
                if not b_success:
                    cur_obj = get_object_pose_w(env)[0].cpu().numpy()
                    obj_z = cur_obj[2]
                    obj_xy = cur_obj[:2]
                    is_low = obj_z < 0.15
                    is_open = current_gripper_state > 0.5
                    if args.taskB_target_mode == "red_region":
                        is_at_target = _is_xy_inside_region(obj_xy, region_center, region_size)
                    else:
                        is_at_target = np.linalg.norm(obj_xy - np.array(region_center)) < args.distance_threshold
                    if is_low and is_open and is_at_target:
                        b_success = True
                        b_success_step = t + 1
                        print(f"    ✓ Task B SUCCESS at step {t + 1}")
                        # record 20 extra frames
                        for _ in range(20):
                            table_rgb, wrist_rgb, ee_pose, obj_pose, gs = _get_obs()
                            imgs.append(table_rgb)
                            if wrist_rgb is not None:
                                wrists.append(wrist_rgb)
                            ees.append(ee_pose)
                            objs.append(obj_pose)
                            grips.append(np.float32(gs))
                            inp = _prepare_input(table_rgb, wrist_rgb, ee_pose, obj_pose, gs)
                            if preproc_B is not None:
                                inp = preproc_B(inp)
                            with torch.no_grad():
                                action = policy_B.select_action(inp)
                            if postproc_B is not None:
                                action = postproc_B(action)
                            action_np = action[0].cpu().numpy()
                            acts.append(action_np)
                            current_gripper_state = float(action_np[7])
                            action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(device)
                            env.step(action_t)
                        break

                if (t + 1) % 100 == 0:
                    dbg_obj = get_object_pose_w(env)[0].cpu().numpy()
                    print(f"    [Task B] Step {t + 1}/{args.horizon}  "
                          f"obj=({dbg_obj[0]:.3f},{dbg_obj[1]:.3f},{dbg_obj[2]:.3f})  "
                          f"grip={current_gripper_state:.2f}")

            if not b_success:
                final_obj_fail = get_object_pose_w(env)[0].cpu().numpy()
                print(f"    ✗ Task B FAILED — obj_xy=({final_obj_fail[0]:.3f}, {final_obj_fail[1]:.3f}), "
                      f"obj_z={final_obj_fail[2]:.3f}, gripper={current_gripper_state:.2f}")
                print(f"      Region center=({region_center[0]:.2f}, {region_center[1]:.2f}), "
                      f"size=({region_size[0]:.2f}, {region_size[1]:.2f})")
                results_B.append(False)
                continue

            results_B.append(True)

            # Build Task B episode dict
            ep_B = {
                "images":   np.array(imgs, dtype=np.uint8),
                "ee_pose":  np.array(ees, dtype=np.float32),
                "obj_pose": np.array(objs, dtype=np.float32),
                "action":   np.array(acts, dtype=np.float32),
                "gripper":  np.array(grips, dtype=np.float32),
                "success":  True,
                "success_step": b_success_step,
                "place_pose": place_pose_7,
                "goal_pose":  goal_pose_7,
            }
            if wrists:
                ep_B["wrist_images"] = np.array(wrists, dtype=np.uint8)

            # record final cube position
            final_obj = get_object_pose_w(env)[0].cpu().numpy()
            ep_B["final_obj_pose"] = final_obj.astype(np.float32)
            episodes_B.append(ep_B)

            # ──────────────── Phase 2: time-reverse ────────────────
            ep_A_rev = reverse_episode(ep_B)

            # ──────────────── Phase 3: replay reversed actions ─────
            print("  Replaying reversed actions (red region → goal) …")
            rev_actions = ep_A_rev["action"]  # (T-1, 8)
            T_replay = len(rev_actions)

            # The cube is already where Policy B left it — we just need to
            # reset the robot arm to its rest pose without moving the cube.
            # We do NOT call env.reset() (which would reset the cube too).
            # Instead we just pre-position the gripper.
            pre_position_gripper_down(env)
            update_target_markers(place_markers, goal_markers,
                                  place_xy=marker_xy, goal_xy=tuple(goal_xy),
                                  marker_z=marker_z, env=env)
            _settle(steps=20)
            current_gripper_state = 1.0

            ra_imgs, ra_wrists, ra_ees, ra_objs, ra_acts, ra_grips = [], [], [], [], [], []

            for t in range(T_replay):
                table_rgb, wrist_rgb, ee_pose, obj_pose, gs = _get_obs()
                ra_imgs.append(table_rgb)
                if wrist_rgb is not None:
                    ra_wrists.append(wrist_rgb)
                ra_ees.append(ee_pose)
                ra_objs.append(obj_pose)
                ra_grips.append(np.float32(gs))

                act = rev_actions[t]  # (8,)
                ra_acts.append(act.copy())
                current_gripper_state = float(act[7])

                action_t = torch.from_numpy(act).float().unsqueeze(0).to(device)
                env.step(action_t)

                if (t + 1) % 100 == 0:
                    print(f"    [Replay] Step {t + 1}/{T_replay}")

            # settle after replay
            _settle(steps=20)

            # check if cube returned to goal
            final_obj_A = get_object_pose_w(env)[0].cpu().numpy()
            dist_to_goal = float(np.linalg.norm(final_obj_A[:2] - goal_xy))
            reset_success = dist_to_goal < args.distance_threshold
            results_A_reset.append(reset_success)
            print(f"    Replay done — cube dist to goal: {dist_to_goal:.4f}m "
                  f"({'✓ reset OK' if reset_success else '✗ drift, will hard-reset'})")

            ep_A = {
                "images":   np.array(ra_imgs, dtype=np.uint8),
                "ee_pose":  np.array(ra_ees, dtype=np.float32),
                "obj_pose": np.array(ra_objs, dtype=np.float32),
                "action":   np.array(ra_acts, dtype=np.float32),
                "gripper":  np.array(ra_grips, dtype=np.float32),
                "success":  True,  # replay data is always saved
                "success_step": None,
                "place_pose": place_pose_7,
                "goal_pose":  goal_pose_7,
                "replay_dist_to_goal": np.float32(dist_to_goal),
            }
            if ra_wrists:
                ep_A["wrist_images"] = np.array(ra_wrists, dtype=np.uint8)
            episodes_A.append(ep_A)

            # if replay didn't reset accurately, hard-reset for next cycle
            if not reset_success:
                print("    Hard-resetting environment for next cycle …")
                # _hard_reset_to_goal will be called at top of next cycle anyway
            # (if it DID reset, the cube is already at goal — ready for next B rollout)

            # running stats
            b_succ = sum(results_B)
            b_total = len(results_B)
            a_reset = sum(results_A_reset)
            a_total = len(results_A_reset)
            print(f"  Running: B success {b_succ}/{b_total} = {100 * b_succ / b_total:.1f}%  |  "
                  f"Replay reset {a_reset}/{a_total} = {100 * a_reset / a_total:.1f}%")

        elapsed = time.time() - start_time

        # ── summary ──────────────────────────────────────────────────
        n_b_success = sum(results_B)
        b_rate = n_b_success / len(results_B) if results_B else 0.0
        n_reset_ok = sum(results_A_reset) if results_A_reset else 0
        reset_rate = n_reset_ok / len(results_A_reset) if results_A_reset else 0.0

        print(f"\n{'=' * 60}")
        print("Collection Summary")
        print(f"{'=' * 60}")
        print(f"  Attempted cycles:   {args.num_cycles}")
        print(f"  Task B success:     {n_b_success}/{len(results_B)} = {b_rate:.1%}")
        print(f"  Replay reset OK:    {n_reset_ok}/{len(results_A_reset)} = {reset_rate:.1%}")
        print(f"  Task B episodes:    {len(episodes_B)}")
        print(f"  Task A episodes:    {len(episodes_A)}")
        print(f"  Total time:         {elapsed:.1f}s")

        # ── save NPZ ────────────────────────────────────────────────
        out_b = Path(args.out_B)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_b, episodes=np.array(episodes_B, dtype=object))
        print(f"  Saved {len(episodes_B)} Task B episodes → {out_b}")

        out_a = Path(args.out_A)
        out_a.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_a, episodes=np.array(episodes_A, dtype=object))
        print(f"  Saved {len(episodes_A)} Task A episodes → {out_a}")

        # ── save stats JSON ──────────────────────────────────────────
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_B": args.policy_B,
                "num_cycles": args.num_cycles,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "goal_xy": list(args.goal_xy),
                "taskB_target_mode": args.taskB_target_mode,
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
            },
            "summary": {
                "total_cycles": args.num_cycles,
                "task_B_success_count": n_b_success,
                "total_task_B_attempts": len(results_B),
                "task_B_success_rate": b_rate,
                "task_A_episodes_collected": len(episodes_A),
                "replay_reset_success_count": n_reset_ok,
                "replay_reset_success_rate": reset_rate,
                "total_elapsed_seconds": elapsed,
            },
            "per_cycle_results": [],
        }
        b_idx = 0
        for i in range(len(results_B)):
            entry: Dict[str, Any] = {"cycle": i + 1, "task_B_success": results_B[i]}
            if results_B[i]:
                if b_idx < len(results_A_reset):
                    entry["replay_reset_success"] = results_A_reset[b_idx]
                    entry["replay_dist_to_goal"] = float(episodes_A[b_idx]["replay_dist_to_goal"])
                b_idx += 1
            stats["per_cycle_results"].append(entry)

        stats_path = out_b.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics → {stats_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
