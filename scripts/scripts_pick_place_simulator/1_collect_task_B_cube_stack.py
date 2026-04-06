#!/usr/bin/env python3
"""Collect Task B data for the 3-cube unstack task (Exp38).

The robot unstacks three cubes (top→bottom) from a vertical stack at the green
marker and places each at a random position inside the red rectangular region.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B_cube_stack.py \
        --out data/pick_place_isaac_lab_simulation/exp38/task_B_cube_stack_5.npz \
        --num_episodes 5 --num_envs 1 --horizon 1500 --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import numpy as np


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


TASK_ID = "Isaac-Lift-CubeStack-Franka-IK-Abs-v0"


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect 3-cube unstack data (Exp38).")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2, default=[0.30, 0.30])
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ======================================================================
# Place-target sampling with minimum separation
# ======================================================================

def sample_place_targets(rng, red_center, red_size, cube_defs_pick_order, min_sep=0.10):
    """Sample 3 placement XY positions that are mutually ≥ *min_sep* apart.

    Returns list of 3 (x, y) tuples.
    """
    cx, cy = float(red_center[0]), float(red_center[1])
    sx, sy = float(red_size[0]), float(red_size[1])
    max_half = max(c.edge_length for c in cube_defs_pick_order) / 2
    safe_hx = sx / 2 - max_half
    safe_hy = sy / 2 - max_half

    placed = []
    for _ in range(3):
        for _attempt in range(200):
            x = rng.uniform(cx - safe_hx, cx + safe_hx)
            y = rng.uniform(cy - safe_hy, cy + safe_hy)
            if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_sep for px, py in placed):
                placed.append((x, y))
                break
        else:
            # deterministic fallback grid
            grid = [
                (cx - 0.08, cy - 0.08),
                (cx + 0.08, cy + 0.08),
                (cx + 0.08, cy - 0.08),
            ]
            placed.append(grid[len(placed)])
    return placed


# ======================================================================
# Main
# ======================================================================

def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- post-sim imports ----
    import importlib.util
    from pathlib import Path

    import torch

    from rev2fwd_il.sim.cube_stack_registry import (
        CUBE_DEFS,
        DEFAULT_STACK_CONFIG,
        PICK_ORDER,
        get_cube_def,
    )
    from rev2fwd_il.experts.cube_unstack_expert import CubeUnstackExpert, UnstackState
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.utils.seed import set_seed

    # Reuse camera / marker helpers from original collection script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)

    add_camera_to_env_cfg = _orig.add_camera_to_env_cfg
    create_target_markers = _orig.create_target_markers
    update_target_markers = _orig.update_target_markers

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    # Load our local env config directly (Isaac Sim resolves isaaclab_tasks to its
    # bundled site-packages copy, not the local editable install)
    _env_cfg_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "manipulation"
        / "lift"
        / "config"
        / "franka"
        / "cube_stack_env_cfg.py"
    )
    _env_cfg_spec = importlib.util.spec_from_file_location("cube_stack_env_cfg", _env_cfg_path)
    _env_cfg_mod = importlib.util.module_from_spec(_env_cfg_spec)
    _env_cfg_spec.loader.exec_module(_env_cfg_mod)
    FrankaCubeStackEnvCfg = _env_cfg_mod.FrankaCubeStackEnvCfg

    # ---- helpers ----
    stack_cfg = DEFAULT_STACK_CONFIG
    cube_params = {}
    for cd in CUBE_DEFS:
        cube_params[cd.name] = dict(
            grasp_z_offset=cd.grasp_z_offset,
            release_z_offset=cd.release_z_offset,
            place_z=cd.place_z,
        )
    pick_order_defs = [get_cube_def(n) for n in PICK_ORDER]

    # ---- create env ----
    def _make_env():
        # Directly instantiate config to bypass gym registry lookup issues
        env_cfg = FrankaCubeStackEnvCfg()
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.sim.device = args.device
        if bool(args.disable_fabric):
            env_cfg.sim.use_fabric = False
        env_cfg.episode_length_s = 200.0
        # disable terminations
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "time_out"):
                env_cfg.terminations.time_out = None
            if hasattr(env_cfg.terminations, "object_dropping"):
                env_cfg.terminations.object_dropping = None
        add_camera_to_env_cfg(env_cfg, args.image_width, args.image_height)

        # Register the env temporarily so gym.make works
        if TASK_ID not in gym.registry:
            gym.register(
                id=TASK_ID,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": "dummy"},
                disable_env_checker=True,
            )
        return gym.make(TASK_ID, cfg=env_cfg)

    # ---- rollout ----
    def rollout_cube_unstack(env, expert, rng, horizon, settle_steps, markers):
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

        # 1. reset env
        obs_dict, _ = env.reset()

        # 2. teleport cubes to stacked position
        gx, gy = stack_cfg.goal_xy
        for cd in CUBE_DEFS:
            pose = torch.zeros(num_envs, 7, device=device)
            pose[:, 0] = gx
            pose[:, 1] = gy
            pose[:, 2] = cd.init_z
            pose[:, 3] = 1.0  # qw
            teleport_object_to_pose(env, pose, name=cd.scene_key)
        print(f"[{_ts()}] Cubes teleported to stack position", flush=True)

        # settle physics
        ee_hold = get_ee_pose_w(env)
        hold_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        hold_action[:, :7] = ee_hold[:, :7]
        hold_action[:, 7] = 1.0
        for _ in range(15):
            env.step(hold_action)

        # 3. sample place targets per env
        all_place_xys = []  # list of list of 3 (x,y)
        for _ in range(num_envs):
            pts = sample_place_targets(
                rng,
                args.red_region_center_xy,
                args.red_region_size_xy,
                pick_order_defs,
                min_sep=stack_cfg.min_place_separation,
            )
            all_place_xys.append(pts)
        # (N, 3, 2)
        place_targets_t = torch.tensor(all_place_xys, dtype=torch.float32, device=device)

        # 4. pre-position robot
        ee_pose = get_ee_pose_w(env)
        expert.reset(ee_pose, place_targets_t)

        prepos_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        prepos_action[:, :7] = expert.rest_pose[:, :7]
        prepos_action[:, 7] = 1.0
        for _ in range(80):
            obs_dict, _, _, _, _ = env.step(prepos_action)
        for _ in range(10):
            obs_dict, _, _, _, _ = env.step(prepos_action)
        print(f"[{_ts()}] Robot pre-positioned", flush=True)

        # 5. markers
        if markers is None:
            start_markers, goal_markers, marker_z = create_target_markers(
                num_envs, device,
                red_marker_shape="rectangle",
                red_marker_size_xy=tuple(args.red_region_size_xy),
            )
            markers = (start_markers, goal_markers, marker_z)
        else:
            start_markers, goal_markers, marker_z = markers

        red_center = tuple(args.red_region_center_xy)
        marker_xys = [red_center for _ in range(num_envs)]
        update_target_markers(
            start_markers, goal_markers,
            start_xys=marker_xys,
            goal_xy=tuple(args.goal_xy),
            marker_z=marker_z,
            env=env,
        )
        # let cameras capture markers
        m_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        m_action[:, :7] = get_ee_pose_w(env)[:, :7]
        m_action[:, 7] = 1.0
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step(m_action)
        print(f"[{_ts()}] Markers placed", flush=True)

        # 6. recording buffers
        CUBE_KEYS = [cd.scene_key for cd in CUBE_DEFS]
        obs_lists = [[] for _ in range(num_envs)]
        image_lists = [[] for _ in range(num_envs)]
        wrist_image_lists = [[] for _ in range(num_envs)]
        ee_pose_lists = [[] for _ in range(num_envs)]
        action_lists = [[] for _ in range(num_envs)]
        gripper_lists = [[] for _ in range(num_envs)]
        fsm_state_lists = [[] for _ in range(num_envs)]
        fsm_round_lists = [[] for _ in range(num_envs)]
        cube_pose_lists = {k: [[] for _ in range(num_envs)] for k in CUBE_KEYS}

        expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # 7. main loop
        print(f"[{_ts()}] Starting main loop (horizon={horizon})...", flush=True)
        for t in range(horizon):
            if t == 0 or (t + 1) % 100 == 0:
                states = expert.get_state_names()
                rounds = expert.round_idx.tolist()
                print(f"[{_ts()}] Step {t+1}/{horizon}  state={states}  round={rounds}", flush=True)

            ee_pose = get_ee_pose_w(env)
            all_cube_poses = {}
            for k in CUBE_KEYS:
                all_cube_poses[k] = get_object_pose_w(env, name=k)

            # obs vector
            if isinstance(obs_dict, dict):
                obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
                if obs_vec is None:
                    obs_vec = next(iter(obs_dict.values()))
            else:
                obs_vec = obs_dict

            table_rgb = table_camera.data.output["rgb"]
            wrist_rgb = wrist_camera.data.output["rgb"]
            if table_rgb.shape[-1] > 3:
                table_rgb = table_rgb[..., :3]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]

            # get gripper command from expert state before act()
            gripper_cmd = torch.where(
                (expert.state >= UnstackState.CLOSE) & (expert.state <= UnstackState.LOWER_TO_RELEASE),
                torch.tensor(-1.0, device=device),
                torch.tensor(1.0, device=device),
            )

            obs_np = obs_vec.cpu().numpy()
            ee_np = ee_pose.cpu().numpy()
            table_np = table_rgb.cpu().numpy().astype(np.uint8)
            wrist_np = wrist_rgb.cpu().numpy().astype(np.uint8)
            gripper_np = gripper_cmd.cpu().numpy()
            fsm_np = expert.state.cpu().numpy()
            round_np = expert.round_idx.cpu().numpy()

            for i in range(num_envs):
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_np[i])
                wrist_image_lists[i].append(wrist_np[i])
                ee_pose_lists[i].append(ee_np[i])
                gripper_lists[i].append(gripper_np[i])
                fsm_state_lists[i].append(fsm_np[i])
                fsm_round_lists[i].append(round_np[i])
                for k in CUBE_KEYS:
                    cube_pose_lists[k][i].append(all_cube_poses[k][i].cpu().numpy())

            # expert step
            action = expert.act(ee_pose, all_cube_poses)
            obs_dict, _, _, _, _ = env.step(action)

            # record next-frame ee_pose as action
            next_ee = get_ee_pose_w(env).cpu().numpy()
            for i in range(num_envs):
                act = np.zeros(8, dtype=np.float32)
                act[:7] = next_ee[i]
                act[7] = gripper_np[i]
                action_lists[i].append(act)

            expert_completed |= expert.is_done()
            if expert_completed.all():
                print(f"[{_ts()}] All envs done at step {t+1}", flush=True)
                break

        # 8. settle
        print(f"[{_ts()}] Settle steps ({settle_steps})...", flush=True)
        for t in range(settle_steps):
            ee_pose = get_ee_pose_w(env)
            all_cube_poses = {}
            for k in CUBE_KEYS:
                all_cube_poses[k] = get_object_pose_w(env, name=k)

            if isinstance(obs_dict, dict):
                obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
                if obs_vec is None:
                    obs_vec = next(iter(obs_dict.values()))
            else:
                obs_vec = obs_dict

            table_rgb = table_camera.data.output["rgb"]
            wrist_rgb = wrist_camera.data.output["rgb"]
            if table_rgb.shape[-1] > 3:
                table_rgb = table_rgb[..., :3]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]

            obs_np = obs_vec.cpu().numpy()
            ee_np = ee_pose.cpu().numpy()
            table_np = table_rgb.cpu().numpy().astype(np.uint8)
            wrist_np = wrist_rgb.cpu().numpy().astype(np.uint8)
            fsm_np = expert.state.cpu().numpy()
            round_np = expert.round_idx.cpu().numpy()

            for i in range(num_envs):
                if expert_completed[i]:
                    obs_lists[i].append(obs_np[i])
                    image_lists[i].append(table_np[i])
                    wrist_image_lists[i].append(wrist_np[i])
                    ee_pose_lists[i].append(ee_np[i])
                    gripper_lists[i].append(1.0)
                    fsm_state_lists[i].append(fsm_np[i])
                    fsm_round_lists[i].append(round_np[i])
                    for k in CUBE_KEYS:
                        cube_pose_lists[k][i].append(all_cube_poses[k][i].cpu().numpy())

            rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
            rest_action[:, :7] = expert.rest_pose[:, :7]
            rest_action[:, 7] = 1.0
            obs_dict, _, _, _, _ = env.step(rest_action)

            next_ee = get_ee_pose_w(env).cpu().numpy()
            for i in range(num_envs):
                if expert_completed[i]:
                    act = np.zeros(8, dtype=np.float32)
                    act[:7] = next_ee[i]
                    act[7] = 1.0
                    action_lists[i].append(act)

        # 9. success check — all 3 cubes must be inside the red region
        print(f"[{_ts()}] Checking success (red-region containment)...", flush=True)
        rcx, rcy = float(args.red_region_center_xy[0]), float(args.red_region_center_xy[1])
        rsx, rsy = float(args.red_region_size_xy[0]), float(args.red_region_size_xy[1])
        results = []
        for i in range(num_envs):
            success_per_cube = []
            for cube_name in PICK_ORDER:
                cp = get_object_pose_w(env, name=cube_name)[i, :2].cpu().numpy()
                in_x = abs(cp[0] - rcx) <= rsx / 2
                in_y = abs(cp[1] - rcy) <= rsy / 2
                success_per_cube.append(bool(in_x and in_y))

            all_success = all(success_per_cube)

            # build place_poses (3, 7) and place_targets (3, 2) arrays
            place_targets_np = np.array(all_place_xys[i], dtype=np.float32)  # (3, 2)
            goal_pose_np = np.array([gx, gy, 0.03, 1, 0, 0, 0], dtype=np.float32)

            ep = {
                "obs": np.array(obs_lists[i], dtype=np.float32),
                "images": np.array(image_lists[i], dtype=np.uint8),
                "wrist_images": np.array(wrist_image_lists[i], dtype=np.uint8),
                "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
                "action": np.array(action_lists[i], dtype=np.float32),
                "gripper": np.array(gripper_lists[i], dtype=np.float32),
                "fsm_state": np.array(fsm_state_lists[i], dtype=np.int32),
                "fsm_round": np.array(fsm_round_lists[i], dtype=np.int32),
                "cube_large_pose": np.array(cube_pose_lists["cube_large"][i], dtype=np.float32),
                "cube_medium_pose": np.array(cube_pose_lists["cube_medium"][i], dtype=np.float32),
                "cube_small_pose": np.array(cube_pose_lists["cube_small"][i], dtype=np.float32),
                # Compatibility key for inspect script (uses first picked cube)
                "obj_pose": np.array(cube_pose_lists["cube_small"][i], dtype=np.float32),
                "place_targets": place_targets_np,
                "goal_pose": goal_pose_np,
                "success": all_success,
                "expert_completed": bool(expert_completed[i].item()),
                "success_per_cube": success_per_cube,
            }
            results.append((ep, expert_completed[i].item()))
            print(
                f"[{_ts()}] Env {i}: completed={expert_completed[i].item()}, "
                f"success={all_success}, per_cube={success_per_cube}, "
                f"steps={len(obs_lists[i])}",
                flush=True,
            )

        return results, markers

    # ---- main loop ----
    gx, gy = args.goal_xy

    try:
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        print(f"[{_ts()}] Creating environment...", flush=True)
        env = _make_env()
        device = env.unwrapped.device
        print(f"[{_ts()}] Env created. device={device}", flush=True)

        expert = CubeUnstackExpert(
            num_envs=args.num_envs,
            device=device,
            pick_order=PICK_ORDER,
            cube_params=cube_params,
            hover_z=stack_cfg.hover_z,
            position_threshold=stack_cfg.position_threshold,
            wait_steps=stack_cfg.wait_steps,
        )

        episodes = []
        markers = None
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        start_time = time.time()

        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] Collecting {args.num_episodes} cube-stack unstack episodes", flush=True)
        print(f"[{_ts()}]   num_envs={args.num_envs}, horizon={args.horizon}", flush=True)
        print(f"[{_ts()}] {'='*60}\n", flush=True)

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            print(f"\n[{_ts()}] === Batch {batch_count} ===", flush=True)
            results, markers = rollout_cube_unstack(
                env, expert, rng, args.horizon, args.settle_steps, markers,
            )
            for ep_dict, completed in results:
                episodes.append(ep_dict)
                if len(episodes) >= args.num_episodes:
                    break

            elapsed = time.time() - start_time
            print(
                f"[{_ts()}] Saved: {len(episodes)}/{args.num_episodes} "
                f"| elapsed={elapsed:.1f}s",
                flush=True,
            )

        # ---- save ----
        episodes = episodes[: args.num_episodes]
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, episodes=episodes)
        print(f"\n[{_ts()}] Saved {len(episodes)} episodes to {out_path}", flush=True)

        if episodes:
            ep0 = episodes[0]
            print(f"  obs: {ep0['obs'].shape}")
            print(f"  images: {ep0['images'].shape}")
            print(f"  wrist_images: {ep0['wrist_images'].shape}")
            print(f"  action: {ep0['action'].shape}")
            print(f"  ee_pose: {ep0['ee_pose'].shape}")
            print(f"  fsm_state: {ep0['fsm_state'].shape}")
            print(f"  fsm_round: {ep0['fsm_round'].shape}")
            print(f"  cube_large_pose: {ep0['cube_large_pose'].shape}")

            succ = sum(1 for e in episodes if e["success"])
            comp = sum(1 for e in episodes if e["expert_completed"])
            print(f"  Completed: {comp}/{len(episodes)}, Success: {succ}/{len(episodes)}")

        env.close()

    except Exception:
        import traceback
        print(f"\n[{_ts()}] *** EXCEPTION ***", flush=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"[{_ts()}] Shutting down...", flush=True)
        simulation_app.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
