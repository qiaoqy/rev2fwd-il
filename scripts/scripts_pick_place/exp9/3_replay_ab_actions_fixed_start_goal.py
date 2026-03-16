#!/usr/bin/env python3
"""Replay first Task A/B trajectories in Isaac Sim with fixed start/goal.

This script replays the first episode action sequence from:
- Task A dataset: data/pick_place_isaac_lab_simulation/exp9/iter0_taskA_from_reverse.npz
- Task B dataset: data/pick_place_isaac_lab_simulation/exp9/iter0_taskB_20.npz

It initializes a fixed start/goal setup from exp9 planning:
- fixed_start_xy = (0.45, 0.15)
- goal_xy = (0.50, 0.00)

Then it runs two single-episode replays (Task A once, Task B once) and saves
both executions as new NPZ files under exp9.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay first Task A/B actions in fixed-start fixed-goal environment.",
    )

    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--auto_select_gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 and CUDA_VISIBLE_DEVICES is not preset, auto-pick GPU with most free memory.",
    )
    parser.add_argument(
        "--min_free_gpu_mb",
        type=int,
        default=12000,
        help="Minimum free GPU memory (MB) required for auto-selected device.",
    )
    parser.add_argument(
        "--dataset_a",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/iter0_taskA_from_reverse.npz",
        help="Task A dataset path.",
    )
    parser.add_argument(
        "--dataset_b",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/iter0_taskB_20.npz",
        help="Task B dataset path.",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to replay from both datasets.",
    )
    parser.add_argument(
        "--command_source",
        type=str,
        choices=["ee_pose", "action"],
        default="ee_pose",
        help="Replay command source: ee_pose (trajectory-faithful) or action (goal-action replay).",
    )
    parser.add_argument(
        "--fixed_start_xy",
        type=float,
        nargs=2,
        default=(0.45, 0.15),
        help="Fixed start XY from exp9 plan.",
    )
    parser.add_argument(
        "--goal_xy",
        type=float,
        nargs=2,
        default=(0.50, 0.00),
        help="Fixed goal XY from exp9 plan.",
    )
    parser.add_argument(
        "--success_radius",
        type=float,
        default=0.03,
        help="Success radius in XY distance.",
    )
    parser.add_argument(
        "--out_a",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/replay_taskA_from_iter0_ep0.npz",
        help="Output NPZ for Task A replay.",
    )
    parser.add_argument(
        "--out_b",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/replay_taskB_from_iter0_ep0.npz",
        help="Output NPZ for Task B replay.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def _load_script1_module():
    script1_path = Path(__file__).resolve().parents[1] / "1_collect_data_pick_place.py"
    spec = spec_from_file_location("collect_b_script1", str(script1_path))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _pick_best_gpu_by_free_mem(min_free_mb: int) -> tuple[int, int] | None:
    """Pick the physical GPU index with the maximum free memory.

    Returns:
        (gpu_index, free_mb) if a candidate is found and passes threshold, else None.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to query nvidia-smi for auto GPU selection: {exc}")
        return None

    best_idx = None
    best_free = -1
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx = int(parts[0])
            free_mb = int(parts[1])
        except ValueError:
            continue
        if free_mb > best_free:
            best_idx = idx
            best_free = free_mb

    if best_idx is None or best_free < int(min_free_mb):
        return None
    return best_idx, best_free


def _load_first_episode_command(npz_path: str, episode_idx: int, command_source: str):
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    if "episodes" not in data.files:
        raise ValueError(f"Invalid dataset format: {npz_path}, keys={data.files}")

    episodes = data["episodes"]
    if episode_idx < 0 or episode_idx >= len(episodes):
        raise IndexError(f"episode_idx={episode_idx} out of range, total={len(episodes)}")

    ep = episodes[episode_idx]
    if not isinstance(ep, dict):
        raise TypeError(f"Episode is not dict: {type(ep)}")

    if command_source == "action":
        if "action" not in ep:
            raise KeyError(f"Episode has no 'action' key: {npz_path}")
        action = np.asarray(ep["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 8:
            raise ValueError(f"Action must have shape (T, >=8), got shape={action.shape}")
        return action[:, :8]

    if "ee_pose" not in ep:
        raise KeyError(f"Episode has no 'ee_pose' key: {npz_path}")

    ee_pose = np.asarray(ep["ee_pose"], dtype=np.float32)
    if ee_pose.ndim != 2 or ee_pose.shape[1] < 7:
        raise ValueError(f"ee_pose must have shape (T, >=7), got shape={ee_pose.shape}")

    if "gripper" in ep:
        gripper = np.asarray(ep["gripper"], dtype=np.float32).reshape(-1)
    elif "action" in ep:
        action = np.asarray(ep["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 8:
            raise ValueError(f"Action must have shape (T, >=8), got shape={action.shape}")
        gripper = action[:, 7]
    else:
        raise KeyError(f"Episode has neither 'gripper' nor valid 'action' key: {npz_path}")

    if ee_pose.shape[0] != len(gripper):
        raise ValueError(
            f"Length mismatch in {npz_path}: len(ee_pose)={ee_pose.shape[0]} vs len(gripper)={len(gripper)}"
        )

    command = np.concatenate([ee_pose[:, :7], gripper[:, None]], axis=1).astype(np.float32)
    return command


def _extract_policy_obs(obs_dict):
    if isinstance(obs_dict, dict):
        obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
        if obs_vec is None:
            obs_vec = next(iter(obs_dict.values()))
        return obs_vec
    return obs_dict


def _replay_single_episode(
    env,
    action_seq,
    object_start_xy,
    target_xy,
    goal_xy,
    success_radius,
    settle_steps=10,
    markers=None,
    update_markers_fn=None,
    red_xy=None,
):
    import numpy as np
    import torch

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    if num_envs != 1:
        raise ValueError(f"Replay script expects num_envs=1, got {num_envs}")

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

    obs_dict, _ = env.reset()

    # Set fixed object initial pose on table.
    start_pose = torch.zeros((1, 7), device=device)
    start_pose[0, 0] = float(object_start_xy[0])
    start_pose[0, 1] = float(object_start_xy[1])
    start_pose[0, 2] = 0.055
    start_pose[0, 3] = 1.0
    teleport_object_to_pose(env, start_pose, name="object")

    zero_action = torch.zeros((1, env.action_space.shape[-1]), device=device)
    for _ in range(10):
        obs_dict, _, _, _, _ = env.step(zero_action)

    # Update visualization markers (red = table position, green = goal position)
    if markers is not None and update_markers_fn is not None and red_xy is not None:
        start_markers, goal_markers, marker_z = markers
        update_markers_fn(
            start_markers=start_markers,
            goal_markers=goal_markers,
            start_xys=[red_xy],
            goal_xy=goal_xy,
            marker_z=marker_z,
            env=env,
        )

    obs_list = []
    image_list = []
    wrist_image_list = []
    ee_pose_list = []
    obj_pose_list = []
    action_list = []
    gripper_list = []

    for t in range(action_seq.shape[0]):
        ee_pose = get_ee_pose_w(env)
        obj_pose = get_object_pose_w(env)
        obs_vec = _extract_policy_obs(obs_dict)

        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        obs_np = obs_vec.cpu().numpy().astype(np.float32)
        ee_pose_np = ee_pose.cpu().numpy().astype(np.float32)
        obj_pose_np = obj_pose.cpu().numpy().astype(np.float32)
        table_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_np = wrist_rgb.cpu().numpy().astype(np.uint8)

        act_t = action_seq[t]
        act_env = torch.from_numpy(act_t).to(device=device, dtype=torch.float32).unsqueeze(0)

        obs_list.append(obs_np[0])
        image_list.append(table_np[0])
        wrist_image_list.append(wrist_np[0])
        ee_pose_list.append(ee_pose_np[0])
        obj_pose_list.append(obj_pose_np[0])
        action_list.append(act_t)
        gripper_list.append(float(act_t[7]))

        obs_dict, _, _, _, _ = env.step(act_env)

    for _ in range(int(settle_steps)):
        obs_dict, _, _, _, _ = env.step(zero_action)

    final_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
    dist = float(
        np.sqrt((final_obj_pose[0] - float(target_xy[0])) ** 2 + (final_obj_pose[1] - float(target_xy[1])) ** 2)
    )
    success = dist < float(success_radius)

    place_pose = np.array([target_xy[0], target_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal_pose = np.array([goal_xy[0], goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    episode = {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "images": np.asarray(image_list, dtype=np.uint8),
        "wrist_images": np.asarray(wrist_image_list, dtype=np.uint8),
        "ee_pose": np.asarray(ee_pose_list, dtype=np.float32),
        "obj_pose": np.asarray(obj_pose_list, dtype=np.float32),
        "action": np.asarray(action_list, dtype=np.float32),
        "gripper": np.asarray(gripper_list, dtype=np.float32),
        "place_pose": place_pose,
        "goal_pose": goal_pose,
        "success": bool(success),
        "final_xy_dist_to_target": np.float32(dist),
    }
    return episode


def _save_single_episode(out_path: str, episode: dict) -> None:
    import numpy as np

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, episodes=[episode])
    print(f"Saved replay episode to {out}")


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    # Avoid defaulting to an OOM GPU on multi-GPU shared servers.
    if bool(args.auto_select_gpu) and str(args.device).startswith("cuda"):
        preset_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if preset_visible:
            print(f"[INFO] Respecting preset CUDA_VISIBLE_DEVICES={preset_visible}")
        else:
            picked = _pick_best_gpu_by_free_mem(args.min_free_gpu_mb)
            if picked is not None:
                gpu_idx, free_mb = picked
                args.device = f"cuda:{gpu_idx}"
                print(
                    f"[INFO] Auto-selected GPU {gpu_idx} "
                    f"(free {free_mb} MB); using device {args.device}."
                )
            else:
                print(
                    "[WARN] Auto GPU selection found no device above threshold; "
                    f"continuing with args.device={args.device}."
                )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np

    from rev2fwd_il.utils.seed import set_seed

    script1 = _load_script1_module()
    make_env_with_camera = script1.make_env_with_camera

    dataset_a = Path(args.dataset_a)
    dataset_b = Path(args.dataset_b)
    if not dataset_a.exists():
        raise FileNotFoundError(f"Task A dataset not found: {dataset_a}")
    if not dataset_b.exists():
        raise FileNotFoundError(f"Task B dataset not found: {dataset_b}")

    fixed_start_xy = (float(args.fixed_start_xy[0]), float(args.fixed_start_xy[1]))
    goal_xy = (float(args.goal_xy[0]), float(args.goal_xy[1]))

    try:
        set_seed(args.seed)

        env = make_env_with_camera(
            task_id=args.task,
            num_envs=1,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )

        # Create visualization markers (red = table position, green = goal position)
        create_target_markers = script1.create_target_markers
        update_target_markers = script1.update_target_markers
        markers = create_target_markers(
            num_envs=1,
            device=str(env.unwrapped.device),
        )

        action_a = _load_first_episode_command(str(dataset_a), args.episode_idx, args.command_source)
        action_b = _load_first_episode_command(str(dataset_b), args.episode_idx, args.command_source)

        print("=" * 60)
        print("Fixed setup")
        print(f"  command_source: {args.command_source}")
        print(f"  fixed_start_xy: {fixed_start_xy}")
        print(f"  goal_xy:        {goal_xy}")
        print(f"  Task A action length: {action_a.shape[0]}")
        print(f"  Task B action length: {action_b.shape[0]}")
        print("=" * 60)

        # Task A: start at fixed_start -> target goal
        ep_a = _replay_single_episode(
            env=env,
            action_seq=action_a,
            object_start_xy=fixed_start_xy,
            target_xy=goal_xy,
            goal_xy=goal_xy,
            success_radius=args.success_radius,
            settle_steps=10,
            markers=markers,
            update_markers_fn=update_target_markers,
            red_xy=fixed_start_xy,
        )
        _save_single_episode(args.out_a, ep_a)
        print(
            f"Task A replay done | success={ep_a['success']} | dist={float(ep_a['final_xy_dist_to_target']):.4f}"
        )

        # Task B: start at goal -> target fixed_start
        ep_b = _replay_single_episode(
            env=env,
            action_seq=action_b,
            object_start_xy=goal_xy,
            target_xy=fixed_start_xy,
            goal_xy=goal_xy,
            success_radius=args.success_radius,
            settle_steps=10,
            markers=markers,
            update_markers_fn=update_target_markers,
            red_xy=fixed_start_xy,
        )
        _save_single_episode(args.out_b, ep_b)
        print(
            f"Task B replay done | success={ep_b['success']} | dist={float(ep_b['final_xy_dist_to_target']):.4f}"
        )

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
