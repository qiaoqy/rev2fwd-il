#!/usr/bin/env python3
"""Step 6: Rod Insert/Extract cyclic evaluation with recovery resets.

Runs N insert→extract cycles using Policy A (insert) and Policy B (extract).
Successful cycles continue from the current simulator state:
    - A success -> B starts from the inserted state.
    - B success -> next A starts from the extracted home state.
Hard resets are only used for initial setup and failed transitions.
Episodes are saved for later finetune (iterative data collection pipeline).

Success criteria:
  Task A (insert): insertion_depth > 15mm  (rod_bottom_z < block_top_z - 15mm)
  Task B (extract): rod center Z within ±2cm of home Z (default 0.15m)

Both use per-frame success detection with a 20-frame buffer after success.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_peg_insert/6_eval_rod_cyclic.py \\
        --policy_A weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --policy_B weights/PP_B/checkpoints/checkpoints/last/pretrained_model \\
        --out_A data/exp46/iter1_collect_A.npz \\
        --out_B data/exp46/iter1_collect_B.npz \\
        --num_cycles 10 --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


# =========================================================================
# Args
# =========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rod Insert/Extract cyclic evaluation with hard-reset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--policy_A", type=str, required=True,
                        help="Policy A (insert) weights path.")
    parser.add_argument("--policy_B", type=str, required=True,
                        help="Policy B (extract) weights path.")

    parser.add_argument("--out_A", type=str, required=True,
                        help="Output NPZ for Task A (insert) episodes.")
    parser.add_argument("--out_B", type=str, required=True,
                        help="Output NPZ for Task B (extract) episodes.")

    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200,
                        help="Max steps per task (default: 200).")
    parser.add_argument("--early_stop_buffer", type=int, default=20,
                        help="Frames to collect after success detection.")
    parser.add_argument("--n_action_steps", type=int, default=16)

    parser.add_argument("--success_depth", type=float, default=0.015,
                        help="Insert success: depth > this (m). Default: 15mm.")
    parser.add_argument("--extract_home_z", type=float, default=0.15,
                        help="Expected rod center Z at home position (m). Default: 0.15.")
    parser.add_argument("--extract_tolerance", type=float, default=0.02,
                        help="Extract success tolerance: |rod_z - home_z| < this (m). Default: ±2cm.")

    parser.add_argument("--save_all", action="store_true",
                        help="Save all episodes (success + failure).")

    parser.add_argument("--task", type=str,
                        default="Isaac-RodInsert-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--settle_steps", type=int, default=200)
    parser.add_argument("--grip_steps", type=int, default=80)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# =========================================================================
# Env creation — reuse from 3_collect_rod_extract.py
# =========================================================================

def _import_rod_env_helpers():
    """Import make_env_with_camera from 3_collect_rod_extract.py."""
    collect_path = str(Path(__file__).resolve().parent / "3_collect_rod_extract.py")
    spec = importlib.util.spec_from_file_location("collect_rod_extract", collect_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_policy_loader():
    """Import policy loading utilities from 6_test_alternating.py."""
    alt_path = str(
        Path(__file__).resolve().parent.parent
        / "scripts_pick_place" / "6_test_alternating.py"
    )
    spec = importlib.util.spec_from_file_location("test_alternating", alt_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================
# Constants
# =========================================================================

BLOCK_TOP_Z = 0.040
ROD_HALF_HEIGHT = 0.030
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)
ROD_GRIP_Z_OFFSET = -0.02


# =========================================================================
# Hard reset: settle + teleport rod + grip + tighten
# =========================================================================

def hard_reset(env, settle_steps=200, grip_steps=80):
    """Full hard reset: env.reset() + settle + teleport rod + grip + tighten.

    Returns the environment in a state where the gripper holds the rod
    at the home position, ready for either Task A or B.
    """
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    rl_env = env.unwrapped

    ee_frame = rl_env.scene["ee_frame"]
    rod = rl_env.scene["object"]
    block = rl_env.scene["block"]

    DOWN_QUAT_T = torch.tensor([DOWN_QUAT], device=device).expand(num_envs, -1)

    # Reset env
    env.reset()

    # Settle: move to approach position with fingers barely open
    settle_target = torch.tensor([[0.5, 0.0, 0.10]], device=device).expand(num_envs, -1)
    for s in range(settle_steps):
        action = torch.cat([settle_target, DOWN_QUAT_T,
                            torch.full((num_envs, 1), 1.0, device=device)], dim=-1)
        env.step(action)

    # Teleport rod between finger pads
    cur_ee_pos_w = ee_frame.data.target_pos_w[:, 0, :].clone()
    rod_target_pos = cur_ee_pos_w.clone()
    rod_target_pos[:, 2] += ROD_GRIP_Z_OFFSET
    rod_teleport_pose = torch.cat([
        rod_target_pos,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1),
    ], dim=-1)
    rod.write_root_pose_to_sim(rod_teleport_pose)
    rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    # Close fingers to grip
    for s in range(grip_steps):
        action = torch.cat([settle_target, DOWN_QUAT_T,
                            torch.full((num_envs, 1), -1.0, device=device)], dim=-1)
        env.step(action)
        if s < 20:
            rod.write_root_pose_to_sim(rod_teleport_pose)
            rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    # Tighten gripper
    rl_env.action_manager._terms["gripper_action"]._open_command.fill_(0.0)

    # Get block pose for success checks
    block_pos = block.data.root_pos_w - rl_env.scene.env_origins
    return block_pos


# =========================================================================
# Run single task (insert or extract) with a given policy
# =========================================================================

def run_task(
    env,
    policy, preprocessor, postprocessor,
    n_action_steps: int,
    horizon: int,
    early_stop_buffer: int,
    check_success_fn,
    has_wrist: bool,
    include_obj_pose: bool,
    include_gripper: bool,
    task_name: str,
    gripper_override: float = -1.0,
):
    """Run a single task episode, recording data for NPZ output.

    Args:
        check_success_fn: callable(rod_pos_local, block_pos_local) -> bool
        gripper_override: force gripper value (-1.0 = closed for rod tasks)

    Returns:
        episode_dict, success_bool
    """
    device = env.unwrapped.device
    rl_env = env.unwrapped

    table_camera = rl_env.scene.sensors["table_cam"]
    wrist_camera = rl_env.scene.sensors["wrist_cam"]
    ee_frame = rl_env.scene["ee_frame"]
    rod = rl_env.scene["object"]
    block = rl_env.scene["block"]

    # Recording buffers
    ee_pose_list = []
    action_list = []
    image_list = []
    wrist_image_list = []
    gripper_list = []

    success = False
    success_step = -1
    policy.reset()

    action_queue = []

    for t in range(horizon):
        # ---- Observation ----
        ee_pos = ee_frame.data.target_pos_w[:, 0, :] - rl_env.scene.env_origins
        ee_quat = ee_frame.data.target_quat_w[:, 0, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)  # (1, 7)

        table_rgb = table_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]

        wrist_rgb = wrist_camera.data.output["rgb"]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        # ---- Record observation ----
        ee_pose_np = ee_pose[0].cpu().numpy()
        table_img_np = table_rgb[0].cpu().numpy().astype(np.uint8)
        wrist_img_np = wrist_rgb[0].cpu().numpy().astype(np.uint8)

        ee_pose_list.append(ee_pose_np)
        image_list.append(table_img_np)
        wrist_image_list.append(wrist_img_np)

        # ---- Policy inference (when action queue is empty) ----
        if len(action_queue) == 0:
            # Build state
            state_parts = [ee_pose[0:1]]  # (1, 7)
            if include_gripper:
                gripper_t = torch.tensor([[gripper_override]], device=device, dtype=torch.float32)
                state_parts.append(gripper_t)
            state = torch.cat(state_parts, dim=-1)  # (1, 7|8)

            # Build image input: (1, 3, H, W) float [0,1]
            table_chw = table_rgb[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            policy_inputs = {
                "observation.image": table_chw,
                "observation.state": state,
            }
            if has_wrist:
                wrist_chw = wrist_rgb[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                policy_inputs["observation.wrist_image"] = wrist_chw

            # Preprocess (normalize)
            if preprocessor is not None:
                policy_inputs = preprocessor(policy_inputs)

            with torch.no_grad():
                raw_action = policy.select_action(policy_inputs)

            # Postprocess (unnormalize)
            if postprocessor is not None:
                raw_action = postprocessor(raw_action)

            # raw_action shape: (1, 8) or (n_action_steps, 8)
            if raw_action.dim() == 2 and raw_action.shape[0] > 1:
                action_queue = list(raw_action.cpu().numpy())
            elif raw_action.dim() == 3:
                action_queue = list(raw_action[0].cpu().numpy())
            else:
                action_queue = [raw_action[0].cpu().numpy()] * n_action_steps

        # ---- Get action from queue ----
        action_np = action_queue.pop(0)
        # Override gripper to stay closed for rod tasks
        action_np[7] = gripper_override
        action_list.append(action_np.copy())
        gripper_list.append(gripper_override)

        # ---- Step environment ----
        action_t = torch.tensor(action_np, device=device, dtype=torch.float32).unsqueeze(0)
        env.step(action_t)

        # ---- Success check ----
        rod_pos = rod.data.root_pos_w - rl_env.scene.env_origins
        block_pos = block.data.root_pos_w - rl_env.scene.env_origins

        if not success and check_success_fn(rod_pos[0], block_pos[0]):
            success = True
            success_step = t
            print(f"  [{_ts()}] {task_name}: SUCCESS at step {t+1}", flush=True)

        # Early stop after buffer
        if success and (t - success_step) >= early_stop_buffer:
            print(f"  [{_ts()}] {task_name}: early stop at step {t+1} "
                  f"(buffer={early_stop_buffer})", flush=True)
            break

    # Build episode dict
    episode = {
        "ee_pose": np.array(ee_pose_list, dtype=np.float32),
        "action": np.array(action_list, dtype=np.float32),
        "images": np.array(image_list, dtype=np.uint8),
        "wrist_images": np.array(wrist_image_list, dtype=np.uint8),
        "gripper": np.array(gripper_list, dtype=np.float32),
        "success": success,
    }
    return episode, success


# =========================================================================
# Success checks
# =========================================================================

def make_check_insert_success(success_depth):
    """Task A: insertion_depth > success_depth.

    insertion_depth = block_top_z - rod_bottom_z  (positive when rod is below block top)
    block_top_z = block_center_z + 0.020  (block half-height)
    rod_bottom_z = rod_center_z - 0.030   (rod half-height)
    """
    def check(rod_pos, block_pos):
        rod_bottom_z = rod_pos[2].item() - ROD_HALF_HEIGHT
        block_top_z = block_pos[2].item() + 0.020  # block half-height
        depth = block_top_z - rod_bottom_z
        return depth > success_depth
    return check


def make_check_extract_success(home_z, tolerance=0.02):
    """Task B: rod center Z is within ±tolerance of home_z.

    home_z:    expected rod center Z at home position (default 0.15m).
    tolerance: ±tolerance around home_z (default 0.02m = ±2cm).

    FSM extract expert targets EE home Z = 0.15–0.18m.  With
    ROD_GRIP_Z_OFFSET = -0.02m the rod center sits ~0.13–0.16m.
    home_z = 0.15 with ±0.02 gives [0.13, 0.17], covering the
    full FSM range with margin.
    """
    def check(rod_pos, block_pos):
        rod_z = rod_pos[2].item()
        return abs(rod_z - home_z) < tolerance
    return check


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from rev2fwd_il.utils.seed import set_seed
        set_seed(args.seed)

        # Import helpers
        rod_helpers = _import_rod_env_helpers()
        policy_loader = _import_policy_loader()

        load_policy_config = policy_loader.load_policy_config
        load_policy_auto = policy_loader.load_policy_auto
        make_env_fn = rod_helpers.make_env_with_camera

        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load policy configs ----
        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

        # ---- Create environment ----
        print(f"[{_ts()}] Creating environment...", flush=True)
        env = make_env_fn(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
        )
        print(f"[{_ts()}] Environment created. device={device}", flush=True)

        # ---- Load policies ----
        policy_A, preproc_A, postproc_A, _, n_act_A = load_policy_auto(
            args.policy_A, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_A.eval()
        print(f"[{_ts()}] Policy A loaded: has_wrist={config_A['has_wrist']}, "
              f"include_obj_pose={config_A['include_obj_pose']}, "
              f"include_gripper={config_A['include_gripper']}", flush=True)

        policy_B, preproc_B, postproc_B, _, n_act_B = load_policy_auto(
            args.policy_B, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()
        print(f"[{_ts()}] Policy B loaded: has_wrist={config_B['has_wrist']}, "
              f"include_obj_pose={config_B['include_obj_pose']}, "
              f"include_gripper={config_B['include_gripper']}", flush=True)

        # ---- Success checkers ----
        check_insert = make_check_insert_success(args.success_depth)
        check_extract = make_check_extract_success(args.extract_home_z, args.extract_tolerance)

        # ---- Initial hard reset ----
        print(f"[{_ts()}] Initial hard reset...", flush=True)
        hard_reset(env, args.settle_steps, args.grip_steps)

        # ---- Main loop ----
        episodes_A = []
        episodes_B = []
        results_A = []
        results_B = []
        start_time = time.time()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}", flush=True)
            print(f"Cycle {cycle + 1}/{args.num_cycles}", flush=True)
            print(f"{'='*50}", flush=True)

            # ---- Task A (insert) ----
            ep_A, success_A = run_task(
                env=env,
                policy=policy_A, preprocessor=preproc_A, postprocessor=postproc_A,
                n_action_steps=n_act_A,
                horizon=args.horizon,
                early_stop_buffer=args.early_stop_buffer,
                check_success_fn=check_insert,
                has_wrist=config_A["has_wrist"],
                include_obj_pose=config_A["include_obj_pose"],
                include_gripper=config_A["include_gripper"],
                task_name="Task A (insert)",
            )
            episodes_A.append(ep_A)
            results_A.append(success_A)
            print(f"  Task A: {'SUCCESS' if success_A else 'FAILED'} "
                  f"({len(ep_A['ee_pose'])} steps)", flush=True)

            # ---- A→B transition ----
            # If A succeeded, rod is in hole → B can start extracting from current state.
            # If A failed, rod position is uncertain → hard reset + setup insert.
            if not success_A:
                print(f"  [{_ts()}] A failed → hard reset + setup insert for B", flush=True)
                hard_reset(env, args.settle_steps, args.grip_steps)
                # Setup: insert rod using FSM expert so B can extract
                rod_helpers.setup_insert_rod(
                    env, settle_steps=30, insert_horizon=350,
                )

            # ---- Task B (extract) ----
            ep_B, success_B = run_task(
                env=env,
                policy=policy_B, preprocessor=preproc_B, postprocessor=postproc_B,
                n_action_steps=n_act_B,
                horizon=args.horizon,
                early_stop_buffer=args.early_stop_buffer,
                check_success_fn=check_extract,
                has_wrist=config_B["has_wrist"],
                include_obj_pose=config_B["include_obj_pose"],
                include_gripper=config_B["include_gripper"],
                task_name="Task B (extract)",
            )
            episodes_B.append(ep_B)
            results_B.append(success_B)
            print(f"  Task B: {'SUCCESS' if success_B else 'FAILED'} "
                  f"({len(ep_B['ee_pose'])} steps)", flush=True)

            # ---- B→A transition ----
            # If B succeeded, rod is already extracted near home and the next A
            # should continue from the current state. Only failed B transitions
            # need recovery via hard reset.
            if success_B:
                print(f"  [{_ts()}] B succeeded -> continue directly into next A", flush=True)
            else:
                print(f"  [{_ts()}] B failed -> hard reset for next A", flush=True)
                hard_reset(env, args.settle_steps, args.grip_steps)

            # Running stats
            a_rate = sum(results_A) / len(results_A) * 100
            b_rate = sum(results_B) / len(results_B) * 100
            print(f"  Running: A={a_rate:.1f}% ({sum(results_A)}/{len(results_A)})  "
                  f"B={b_rate:.1f}% ({sum(results_B)}/{len(results_B)})", flush=True)

        # ---- Summary ----
        elapsed = time.time() - start_time
        n_success_A = sum(results_A)
        n_success_B = sum(results_B)
        rate_A = n_success_A / len(results_A) if results_A else 0.0
        rate_B = n_success_B / len(results_B) if results_B else 0.0

        print(f"\n{'='*60}", flush=True)
        print("Evaluation Results", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Cycles:         {args.num_cycles}")
        print(f"  Task A success: {n_success_A}/{len(results_A)} = {rate_A:.1%}")
        print(f"  Task B success: {n_success_B}/{len(results_B)} = {rate_B:.1%}")
        print(f"  Time:           {elapsed:.1f}s", flush=True)

        # ---- Save episodes ----
        for out_path, episodes, label in [
            (args.out_A, episodes_A, "A"),
            (args.out_B, episodes_B, "B"),
        ]:
            if args.save_all:
                save_eps = episodes
            else:
                save_eps = [ep for ep in episodes if ep["success"]]

            if save_eps:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, episodes=np.array(save_eps, dtype=object))
                n_succ = sum(1 for ep in save_eps if ep["success"])
                print(f"  Saved {len(save_eps)} Task {label} episodes "
                      f"({n_succ} success) to {out_path}", flush=True)
            else:
                print(f"  No Task {label} episodes to save.", flush=True)

        # ---- Save statistics JSON ----
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "num_cycles": args.num_cycles,
                "horizon": args.horizon,
                "early_stop_buffer": args.early_stop_buffer,
                "n_action_steps": args.n_action_steps,
                "success_depth": args.success_depth,
                "extract_home_z": args.extract_home_z,
                "extract_tolerance": args.extract_tolerance,
                "task": args.task,
                "save_all": args.save_all,
            },
            "summary": {
                "total_cycles": args.num_cycles,
                "task_A_success_count": n_success_A,
                "task_B_success_count": n_success_B,
                "task_A_success_rate": rate_A,
                "task_B_success_rate": rate_B,
                "total_elapsed_seconds": elapsed,
            },
            "per_cycle_results": [
                {
                    "cycle": i + 1,
                    "task_A_success": results_A[i],
                    "task_B_success": results_B[i],
                }
                for i in range(len(results_A))
            ],
        }

        stats_path = Path(args.out_A).with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {stats_path}", flush=True)

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
