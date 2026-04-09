#!/usr/bin/env python3
"""Collect Task B data for the 3-cube partial-unstack task (Exp44).

The robot picks cube_small (top, red) then cube_medium (yellow) from a
vertical stack and places each at a *fixed* position on the table.
cube_large (blue, 8cm) stays in place.  No table markers.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B_exp44.py \
        --out data/pick_place_isaac_lab_simulation/exp44/task_B_exp44_5.npz \
        --num_episodes 5 --num_envs 1 --horizon 1200 --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import numpy as np


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


TASK_ID = "Isaac-Exp44-CubeStack-Franka-IK-Abs-v0"


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect partial-unstack data (Exp44).")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


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

    from rev2fwd_il.sim.exp44_registry import (
        CUBE_DEFS_44,
        DEFAULT_EXP44_CONFIG,
        PICK_ORDER_44,
    )
    from rev2fwd_il.experts.exp44_unstack_expert import (
        make_exp44_expert,
        make_fixed_place_targets,
        UnstackState,
    )
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.utils.seed import set_seed

    # Reuse camera helper from original collection script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)

    add_camera_to_env_cfg = _orig.add_camera_to_env_cfg

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401

    # Load local env config directly (bypass Isaac Sim's bundled isaaclab_tasks)
    _env_cfg_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "manipulation"
        / "lift"
        / "config"
        / "franka"
        / "exp44_env_cfg.py"
    )
    _env_cfg_spec = importlib.util.spec_from_file_location("exp44_env_cfg", _env_cfg_path)
    _env_cfg_mod = importlib.util.module_from_spec(_env_cfg_spec)
    _env_cfg_spec.loader.exec_module(_env_cfg_mod)
    FrankaExp44EnvCfg = _env_cfg_mod.FrankaExp44EnvCfg

    # ---- helpers ----
    exp44_cfg = DEFAULT_EXP44_CONFIG
    CUBE_KEYS = [cd.scene_key for cd in CUBE_DEFS_44]

    # ---- create env ----
    def _make_env():
        env_cfg = FrankaExp44EnvCfg()
        env_cfg.scene.num_envs = int(args.num_envs)
        env_cfg.sim.device = args.device
        if bool(args.disable_fabric):
            env_cfg.sim.use_fabric = False
        env_cfg.episode_length_s = 200.0
        if hasattr(env_cfg, "terminations"):
            if hasattr(env_cfg.terminations, "time_out"):
                env_cfg.terminations.time_out = None
            if hasattr(env_cfg.terminations, "object_dropping"):
                env_cfg.terminations.object_dropping = None
        add_camera_to_env_cfg(env_cfg, args.image_width, args.image_height)

        if TASK_ID not in gym.registry:
            gym.register(
                id=TASK_ID,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                kwargs={"env_cfg_entry_point": "dummy"},
                disable_env_checker=True,
            )
        return gym.make(TASK_ID, cfg=env_cfg)

    # ---- rollout ----
    def rollout_exp44(env, expert, horizon, settle_steps):
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs
        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

        # 1. reset env
        obs_dict, _ = env.reset()

        # 2. teleport cubes to stacked position at stack_xy
        sx, sy = exp44_cfg.stack_xy
        for cd in CUBE_DEFS_44:
            pose = torch.zeros(num_envs, 7, device=device)
            pose[:, 0] = sx
            pose[:, 1] = sy
            pose[:, 2] = cd.init_z
            pose[:, 3] = 1.0  # qw
            teleport_object_to_pose(env, pose, name=cd.scene_key)
        print(f"[{_ts()}] Cubes teleported to stack position ({sx}, {sy})", flush=True)

        # settle physics
        ee_hold = get_ee_pose_w(env)
        hold_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        hold_action[:, :7] = ee_hold[:, :7]
        hold_action[:, 7] = 1.0
        for _ in range(15):
            env.step(hold_action)

        # 3. fixed place targets (no random sampling)
        place_targets_t = make_fixed_place_targets(num_envs, device)

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

        # 5. No markers — skip marker creation

        # Let cameras stabilise
        m_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        m_action[:, :7] = get_ee_pose_w(env)[:, :7]
        m_action[:, 7] = 1.0
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step(m_action)

        # 6. recording buffers
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

            # gripper command from expert state
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

        # 9. success check — small and medium must be >10cm from stack and z<5cm
        print(f"[{_ts()}] Checking success...", flush=True)
        stack_xy = np.array(exp44_cfg.stack_xy, dtype=np.float32)
        results = []
        for i in range(num_envs):
            small_pose = get_object_pose_w(env, name="cube_small")[i].cpu().numpy()
            medium_pose = get_object_pose_w(env, name="cube_medium")[i].cpu().numpy()

            small_dist = float(np.linalg.norm(small_pose[:2] - stack_xy))
            medium_dist = float(np.linalg.norm(medium_pose[:2] - stack_xy))
            small_z = float(small_pose[2])
            medium_z = float(medium_pose[2])

            small_ok = bool(small_dist > 0.10 and small_z < 0.05)
            medium_ok = bool(medium_dist > 0.10 and medium_z < 0.05)
            all_success = small_ok and medium_ok

            # place targets array for this episode
            place_targets_np = np.array(
                [list(exp44_cfg.small_place_xy), list(exp44_cfg.medium_place_xy)],
                dtype=np.float32,
            )  # (2, 2)
            goal_pose_np = np.array(
                [sx, sy, 0.04, 1, 0, 0, 0], dtype=np.float32
            )

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
                "place_targets": place_targets_np,
                "goal_pose": goal_pose_np,
                "success": all_success,
                "expert_completed": bool(expert_completed[i].item()),
                "success_per_cube": [small_ok, medium_ok],
            }
            results.append((ep, expert_completed[i].item()))
            print(
                f"[{_ts()}] Env {i}: completed={expert_completed[i].item()}, "
                f"success={all_success}, per_cube=[small={small_ok}, medium={medium_ok}], "
                f"steps={len(obs_lists[i])}",
                flush=True,
            )

        return results

    # ---- main loop ----
    sx, sy = exp44_cfg.stack_xy

    try:
        set_seed(args.seed)

        print(f"[{_ts()}] Creating environment...", flush=True)
        env = _make_env()
        device = env.unwrapped.device
        print(f"[{_ts()}] Env created. device={device}", flush=True)

        expert = make_exp44_expert(num_envs=args.num_envs, device=device)

        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        start_time = time.time()

        print(f"\n[{_ts()}] {'='*60}", flush=True)
        print(f"[{_ts()}] Collecting {args.num_episodes} Exp44 partial-unstack episodes", flush=True)
        print(f"[{_ts()}]   num_envs={args.num_envs}, horizon={args.horizon}", flush=True)
        print(f"[{_ts()}]   stack_xy={exp44_cfg.stack_xy}", flush=True)
        print(f"[{_ts()}]   small_place={exp44_cfg.small_place_xy}, medium_place={exp44_cfg.medium_place_xy}", flush=True)
        print(f"[{_ts()}] {'='*60}\n", flush=True)

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            print(f"\n[{_ts()}] === Batch {batch_count} ===", flush=True)
            results = rollout_exp44(
                env, expert, args.horizon, args.settle_steps,
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
