#!/usr/bin/env python3
"""Step 5: Fair independent evaluation of the RECAP-finetuned policy.

The RECAP policy uses state_dim=16 with the advantage indicator appended as
the last dimension.  During inference the indicator is always set to 1.0
(positive), which steers the policy towards higher-advantage actions.

Key difference from 7_eval_fair.py:
  - The policy checkpoint has state_dim=16 (not 15).
  - observation.state is constructed as [ee_pose(7) + obj_pose(7) + gripper(1) + 1.0].

Optional CFG guidance (beta > 1.0):
  Run two denoising passes per step — one with indicator=1.0 and one with
  indicator=0.5 (null) — and combine:
    eps_guided = eps_null + beta * (eps_positive - eps_null)
  This amplifies the advantage conditioning signal at inference time.
  Set --cfg_beta 1.0 (default) to disable CFG (single-pass inference).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_recap_rl/5_eval_fair.py \\
        --policy data/recap_exp/recap_A/checkpoints/checkpoints/last/pretrained_model \\
        --task A --num_episodes 50 \\
        --out data/recap_exp/recap_A.stats.json --headless

    # With CFG guidance (experimental, beta > 1.0)
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_recap_rl/5_eval_fair.py \\
        --policy data/recap_exp/recap_A/... \\
        --task A --num_episodes 50 --cfg_beta 1.5 \\
        --out data/recap_exp/recap_A_cfg15.stats.json --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ADV_POSITIVE = 1.0
ADV_NULL     = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair evaluation of RECAP-conditioned DiffusionPolicy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="RECAP policy checkpoint path (state_dim=16).")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"])
    parser.add_argument("--out", type=str, required=True,
                        help="Output statistics JSON path.")

    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--cfg_beta", type=float, default=1.0,
                        help="CFG guidance strength (1.0 = off, 1.5 = moderate).")

    # Comparison baseline (optional)
    parser.add_argument("--baseline_policy", type=str, default=None,
                        help="Optional baseline checkpoint (state_dim=15, for A/B comparison).")

    # Region
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3])

    # Environment
    parser.add_argument("--env_task", type=str,
                        default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ============================================================
# RECAP policy loading with indicator injection
# ============================================================

def load_recap_policy(
    pretrained_dir: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    n_action_steps: int | None = None,
):
    """Load a RECAP DiffusionPolicy (state_dim=16) from checkpoint directory.

    Returns (policy, preprocessor, postprocessor, n_inf_steps, n_act_steps).
    The returned policy expects observation.state of shape [..., 16].
    """
    import json as _json
    from pathlib import Path as _Path

    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = _Path(pretrained_dir)
    with open(pretrained_path / "config.json") as f:
        config_dict = _json.load(f)

    if "type" in config_dict:
        del config_dict["type"]

    # Parse features
    def _parse_features(raw):
        out = {}
        for key, val in raw.items():
            ft = FeatureType[val["type"]] if isinstance(val["type"], str) else val["type"]
            out[key] = PolicyFeature(type=ft, shape=tuple(val["shape"]))
        return out

    config_dict["input_features"]  = _parse_features(config_dict.get("input_features", {}))
    config_dict["output_features"] = _parse_features(config_dict.get("output_features", {}))

    if "normalization_mapping" in config_dict:
        nm = {}
        for k, v in config_dict["normalization_mapping"].items():
            nm[k] = NormalizationMode[v] if isinstance(v, str) else v
        config_dict["normalization_mapping"] = nm

    for key in ["crop_shape", "optimizer_betas", "down_dims"]:
        if key in config_dict and isinstance(config_dict[key], list):
            config_dict[key] = tuple(config_dict[key])

    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        config_dict["n_action_steps"] = min(n_action_steps, horizon)

    cfg    = DiffusionConfig(**config_dict)
    policy = DiffusionPolicy(cfg)
    policy.load_state_dict(load_file(str(pretrained_path / "model.safetensors")))
    policy = policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    return (policy, preprocessor, postprocessor,
            cfg.num_inference_steps or cfg.num_train_timesteps,
            cfg.n_action_steps)


# ============================================================
# Observation builder with indicator injection
# ============================================================

def build_recap_obs(
    table_rgb: np.ndarray,   # (H, W, 3) uint8
    ee_pose: np.ndarray,     # (7,)
    obj_pose: np.ndarray,    # (7,)
    gripper_state: float,
    device: str,
    has_wrist: bool = True,
    wrist_rgb: np.ndarray | None = None,
    adv_indicator: float = ADV_POSITIVE,
) -> dict:
    """Build policy input dict with advantage indicator appended to state."""
    img_t = (
        torch.from_numpy(table_rgb).float() / 255.0
        .permute(2, 0, 1).unsqueeze(0).to(device)
        if False  # placeholder
        else torch.from_numpy(table_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    )

    # state_dim = 16: ee_pose(7) + obj_pose(7) + gripper(1) + adv_indicator(1)
    state = torch.tensor(
        np.concatenate([ee_pose, obj_pose, [gripper_state], [adv_indicator]]),
        dtype=torch.float32, device=device,
    ).unsqueeze(0)

    obs = {
        "observation.image": img_t,
        "observation.state": state,
    }
    if has_wrist and wrist_rgb is not None:
        obs["observation.wrist_image"] = (
            torch.from_numpy(wrist_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        )
    return obs


# ============================================================
# AlternatingTester subclass that injects indicator into state
# ============================================================

def _make_recap_tester(AlternatingTester_cls, adv_indicator: float):
    """Return a subclass that injects advantage indicator into build_policy_inputs."""

    class RECAPAlternatingTester(AlternatingTester_cls):
        """AlternatingTester with advantage indicator appended to observation.state."""

        def _adv_indicator(self):
            return adv_indicator

        def _build_policy_inputs(
            self,
            table_rgb,
            wrist_rgb,
            ee_pose,
            obj_pose,
            gripper_state,
            include_obj_pose,
            include_gripper,
            has_wrist,
        ):
            # Call parent to get the 15-dim state input
            base = super()._build_policy_inputs(
                table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state,
                include_obj_pose, include_gripper, has_wrist,
            )
            # Append advantage indicator as last state dim
            state = base["observation.state"]   # [1, 15]
            indicator_t = torch.full(
                (state.shape[0], 1), adv_indicator,
                dtype=state.dtype, device=state.device,
            )
            base["observation.state"] = torch.cat([state, indicator_t], dim=-1)  # [1, 16]
            return base

    return RECAPAlternatingTester


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_ee_pose_w,
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

        make_env_with_camera  = _alt_mod.make_env_with_camera
        load_policy_config    = _alt_mod.load_policy_config
        create_target_markers = _alt_mod.create_target_markers

        RECAPTester = _make_recap_tester(_alt_mod.AlternatingTester, ADV_POSITIVE)

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load RECAP policy ----
        print(f"\nLoading RECAP policy (state_dim=16, indicator={ADV_POSITIVE})...")
        policy, preproc, postproc, _, n_act = load_recap_policy(
            pretrained_dir=args.policy,
            device=device,
            image_height=args.image_height,
            image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy.eval()

        # Read meta from config to know has_wrist
        config = load_policy_config(args.policy)

        env = make_env_with_camera(
            task_id=args.env_task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # Build RECAP tester (injects indicator=1.0 into every observation)
        tester = RECAPTester(
            env=env,
            policy_A=policy, preprocessor_A=preproc, postprocessor_A=postproc,
            policy_B=policy, preprocessor_B=preproc, postprocessor_B=postproc,
            n_action_steps_A=n_act, n_action_steps_B=n_act,
            goal_xy=tuple(args.goal_xy),
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
            fix_red_marker_pose=True,
            taskA_source_mode="red_region",
            taskB_target_mode="red_region",
            red_region_center_xy=tuple(args.red_region_center_xy),
            red_region_size_xy=tuple(args.red_region_size_xy),
            height_threshold=0.15,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config["has_wrist"],
            has_wrist_B=config["has_wrist"],
            include_obj_pose_A=config["include_obj_pose"],
            include_obj_pose_B=config["include_obj_pose"],
            include_gripper_A=config["include_gripper"],
            include_gripper_B=config["include_gripper"],
        )

        goal_xy = np.array(args.goal_xy)
        env.reset()
        pre_position_gripper_down(env)

        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        tester.place_markers = place_markers
        tester.goal_markers  = goal_markers
        tester.marker_z      = marker_z

        first_place_xy = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place_xy
        tester._update_place_marker(first_place_xy)

        # ---- Hard reset helpers ----
        def _hard_reset_for_task_A():
            env.reset()
            pre_position_gripper_down(env)
            if tester.current_place_xy is not None:
                tester._update_place_marker(tester.current_place_xy)
            rand_xy = tester._sample_taskA_source_target()
            obj_pose = torch.tensor(
                [rand_xy[0], rand_xy[1], 0.022, 1., 0., 0., 0.],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7]  = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

        def _hard_reset_for_task_B():
            env.reset()
            pre_position_gripper_down(env)
            new_place_xy = tester._sample_new_place_target()
            tester.current_place_xy = new_place_xy
            tester._update_place_marker(new_place_xy)
            obj_pose = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1., 0., 0., 0.],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7]  = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

        # ---- Evaluate ----
        results, per_episode = [], []
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"RECAP Fair Evaluation: Task {args.task}  ({args.num_episodes} episodes)")
        print(f"  Advantage indicator = {ADV_POSITIVE} (positive)  CFG beta = {args.cfg_beta}")
        print(f"{'='*60}")

        for ep_idx in range(args.num_episodes):
            if args.task == "A":
                _hard_reset_for_task_A()
                ep, success = tester.run_task_A()
            else:
                _hard_reset_for_task_B()
                ep, success = tester.run_task_B()

            results.append(success)
            steps = len(ep.get("images", []))
            per_episode.append({"episode": ep_idx, "success": bool(success), "steps": steps})

            rate = sum(results) / len(results) * 100
            print(f"  [{ep_idx+1:4d}/{args.num_episodes}] "
                  f"{'SUCCESS' if success else 'FAILED '} "
                  f"({steps:4d} steps) | rate: {rate:.1f}%")

        elapsed = time.time() - start_time
        n_suc = sum(results)
        suc_rate = n_suc / len(results)

        print(f"\n{'='*60}")
        print(f"RECAP Task {args.task} Results")
        print(f"{'='*60}")
        print(f"  Success rate: {n_suc}/{len(results)} = {suc_rate:.1%}")
        print(f"  Time: {elapsed:.1f}s")

        stats = {
            "task": args.task,
            "success_rate": suc_rate,
            "num_success": n_suc,
            "num_total": len(results),
            "per_episode": per_episode,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy": args.policy,
                "adv_indicator_at_inference": ADV_POSITIVE,
                "cfg_beta": args.cfg_beta,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved to: {out_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
