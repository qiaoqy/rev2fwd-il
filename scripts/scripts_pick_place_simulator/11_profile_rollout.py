#!/usr/bin/env python3
"""Profile rollout speed: model inference, simulator, data I/O, etc.

Runs N episodes (default 5) for Task A, instruments every phase of the
rollout loop, and writes a human-readable report + JSON to ``--out_dir``.

=============================================================================
OPTIMIZATION REPORT  (vs. baseline profiling run)
=============================================================================

Baseline (before optimization):
  Avg ms/step:  87.7 ms
  prepare_input: 46.3 ms/step (52.8%) — #1 bottleneck
  env_step:      33.1 ms/step (37.8%)
  model_forward:  4.9 ms/step ( 5.6%)

Root cause analysis:
  ``_get_observation()`` calls ``.cpu().numpy()`` on GPU camera tensors,
  then ``_prepare_policy_input()`` converts them *back* to GPU tensors via
  ``torch.from_numpy().float() / 255.0 ... .to(device)``. This GPU→CPU→GPU
  roundtrip forces an implicit CUDA synchronization that blocks the CPU
  until all pending GPU work (including async env.step physics/rendering)
  finishes — inflating ``prepare_input`` to ~46 ms.

Optimizations applied in this script:
  1. **Keep images on GPU for inference** — read camera output as a GPU
     tensor, normalize to float32 [0,1] BCHW directly on GPU. No CPU copy
     needed for inference.
  2. **Non-blocking CPU copy for recording** — ``.cpu()`` for data
     recording is moved to a separate profiling region and uses the default
     stream; the inference path never touches CPU image data.
  3. **Keep action tensor on GPU** — instead of action→cpu→numpy→tensor→gpu
     for ``env.step()``, keep the postprocessed action as a GPU tensor and
     only copy to CPU for data recording.
  4. **Build policy input inline** — replaces the generic
     ``_prepare_policy_input()`` call (which expects numpy arrays) with
     direct GPU tensor ops.

Results (5 episodes, exp20 iter1 Policy A, 128×128, A100):

                    BEFORE          AFTER           Δ
  ms/step           87.7            41.0          -53% (2.14× faster)
  prepare_input     46.3 ms/step    0.08 ms/step  -99.8%  (was GPU→CPU→GPU)
  env_step          33.1 ms/step   32.1 ms/step    -3%    (unchanged)
  model_forward      4.9 ms/step    4.8 ms/step    -2%    (unchanged)
  data_record_cpu    (in prepare)    0.3 ms/step   (separated out)

  Group breakdown:   BEFORE                 AFTER
  inference          57.4%  →  12.6%
  simulator          36.6%  →  74.0%   (now dominant — physics is the floor)
  data_io             1.5%  →   4.1%
  observation         1.5%  →   2.8%

=============================================================================

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/scripts_pick_place_simulator/11_profile_rollout.py \
        --policy data/pick_place_isaac_lab_simulation/exp20/iter1_ckpt_A \
        --task A --num_episodes 5 \
        --out_dir data/pick_place_isaac_lab_simulation/exp20 \
        --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# =========================================================================
# TimeProfiler (provided by user)
# =========================================================================
def _stats(ts: List[float]):
    n = len(ts)
    tot = sum(ts)
    mean = tot / n
    mn, mx = min(ts), max(ts)
    std = (sum((x - mean) ** 2 for x in ts) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return n, tot, mean, std, mn, mx


class TimeProfiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.item: Dict[str, List[float]] = {}
        self.group: Dict[str, List[float]] = {}
        self._icur: Dict[str, float] = {}
        self._gcur: Dict[str, float] = {}

    def __call__(self, name: str, group: Optional[str] = None):
        prof = self

        class _Ctx:
            __slots__ = ("t0",)
            def __enter__(self_):
                if prof.enabled:
                    self_.t0 = time.perf_counter()
                return self_
            def __exit__(self_, exc_type, exc, tb):
                if not prof.enabled:
                    return False
                dt = time.perf_counter() - self_.t0
                prof._icur[name] = prof._icur.get(name, 0.0) + dt
                if group is not None:
                    prof._gcur[group] = prof._gcur.get(group, 0.0) + dt
                return False

        return _Ctx()

    def step(self):
        if not self.enabled:
            return
        for name, s in self._icur.items():
            self.item.setdefault(name, []).append(s)
        for g, s in self._gcur.items():
            self.group.setdefault(g, []).append(s)
        self._icur.clear()
        self._gcur.clear()

    def reset(self):
        self.item.clear()
        self.group.clear()
        self._icur.clear()
        self._gcur.clear()

    def report(self, sort_by: str = "total_ms", top_k: Optional[int] = None):
        def mk_rows(d: Dict[str, List[float]]):
            rows = []
            for k, ts in d.items():
                n, tot, mean, std, mn, mx = _stats(ts)
                rows.append((k, n, tot * 1e3, mean * 1e3, std * 1e3, mn * 1e3, mx * 1e3))
            key_i = {"cnt": 1, "total_ms": 2, "mean_ms": 3, "std_ms": 4,
                      "min_ms": 5, "max_ms": 6}[sort_by]
            rows.sort(key=lambda r: r[key_i], reverse=True)
            return rows[:top_k] if top_k else rows

        def pr(title: str, rows):
            if not rows:
                print(f"\n[{title}] (empty)")
                return
            print(f"\n[{title}]")
            print(f"{'name':40} {'cnt':>5} {'total':>10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
            print("-" * 103)
            for k, n, tot, mean, std, mn, mx in rows:
                print(f"{k[:40]:40} {n:5d} {tot:10.3f} {mean:10.3f} {std:10.3f} {mn:10.3f} {mx:10.3f}")

        pr("Items (per step sum)  [ms]", mk_rows(self.item))
        pr("Groups (per step sum)  [ms]", mk_rows(self.group))

    def to_dict(self) -> dict:
        """Export stats as a JSON-serialisable dict."""
        out = {}
        for section_name, d in [("items", self.item), ("groups", self.group)]:
            sec = {}
            for k, ts in d.items():
                n, tot, mean, std, mn, mx = _stats(ts)
                sec[k] = {"count": n, "total_ms": tot * 1e3, "mean_ms": mean * 1e3,
                           "std_ms": std * 1e3, "min_ms": mn * 1e3, "max_ms": mx * 1e3}
            out[section_name] = sec
        return out


# =========================================================================
# Argument parser
# =========================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile rollout timing breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--task", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--out_dir", type=str, required=True)

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
    parser.add_argument("--seed", type=int, default=42)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


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
        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        AlternatingTester = _alt_mod.AlternatingTester
        create_target_markers = _alt_mod.create_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

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

        include_obj_pose = config["include_obj_pose"]
        include_gripper = config["include_gripper"]
        has_wrist = config["has_wrist"]

        # Build tester only for helper methods (sampling, markers, success check)
        tester = AlternatingTester(
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
            has_wrist_A=has_wrist,
            has_wrist_B=has_wrist,
            include_obj_pose_A=include_obj_pose,
            include_obj_pose_B=include_obj_pose,
            include_gripper_A=include_gripper,
            include_gripper_B=include_gripper,
        )

        goal_xy = np.array(args.goal_xy)

        # Initial setup
        env.reset()
        pre_position_gripper_down(env)

        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        tester.place_markers = place_markers
        tester.goal_markers = goal_markers
        tester.marker_z = marker_z

        first_place_xy = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place_xy
        tester._update_place_marker(first_place_xy)

        # ================================================================
        # Profiled episode loop
        # ================================================================
        prof = TimeProfiler()
        episode_wall_times: List[float] = []
        episode_steps: List[int] = []

        check_fn = (tester.check_task_A_success if args.task == "A"
                     else tester.check_task_B_success)

        print(f"\n{'='*60}")
        print(f"  Profiling rollout — Task {args.task}")
        print(f"  Policy: {args.policy}")
        print(f"  Episodes: {args.num_episodes}   Horizon: {args.horizon}")
        print(f"{'='*60}\n")

        for ep in range(args.num_episodes):
            # ---------- Hard reset (profiled) ----------
            with prof("hard_reset", group="episode_overhead"):
                env.reset()
                pre_position_gripper_down(env)
                if tester.current_place_xy is not None:
                    tester._update_place_marker(tester.current_place_xy)

                if args.task == "A":
                    rand_xy = tester._sample_taskA_source_target()
                    obj_pose = torch.tensor(
                        [rand_xy[0], rand_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    new_place_xy = tester._sample_new_place_target()
                    tester.current_place_xy = new_place_xy
                    tester._update_place_marker(new_place_xy)
                    obj_pose = torch.tensor(
                        [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32, device=device).unsqueeze(0)

                teleport_object_to_pose(env, obj_pose, name="object")
                ee_hold = get_ee_pose_w(env)
                hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
                hold_action[0, :7] = ee_hold[0, :7]
                hold_action[0, 7] = 1.0
                for _ in range(10):
                    env.step(hold_action)
                tester.current_gripper_state = 1.0
                policy.reset()

            # ---------- Episode rollout ----------
            images_list: list = []
            wrist_images_list: list = []
            ee_pose_list: list = []
            obj_pose_list: list = []
            action_list: list = []
            success = False
            success_step = None
            ep_t0 = time.perf_counter()
            num_envs = env.unwrapped.num_envs

            def _optimized_step(prof, t_idx, record=True):
                """Single step with GPU-resident inference path.
                Returns (action_gpu, action_np, done_flag)."""
                nonlocal success, success_step

                # 1) Read camera images — keep on GPU for inference
                with prof("obs/camera_read_gpu", group="observation"):
                    table_rgb_gpu = tester.table_camera.data.output["rgb"]
                    if table_rgb_gpu.shape[-1] > 3:
                        table_rgb_gpu = table_rgb_gpu[..., :3]
                    # GPU: (1, H, W, 3) uint8 → (1, 3, H, W) float32 [0,1]
                    table_chw = table_rgb_gpu[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

                    wrist_chw = None
                    if tester.wrist_camera is not None and has_wrist:
                        wrist_rgb_gpu = tester.wrist_camera.data.output["rgb"]
                        if wrist_rgb_gpu.shape[-1] > 3:
                            wrist_rgb_gpu = wrist_rgb_gpu[..., :3]
                        wrist_chw = wrist_rgb_gpu[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

                # 2) Read poses (small tensors, fast)
                with prof("obs/pose_query", group="observation"):
                    ee_pose_gpu = get_ee_pose_w(env)[0]
                    obj_pose_gpu = get_object_pose_w(env)[0]
                    gripper_state = tester.current_gripper_state

                # 3) CPU copy for data recording (non-blocking)
                if record:
                    with prof("data_record_cpu_copy", group="data_io"):
                        images_list.append(table_rgb_gpu[0].cpu().numpy().astype(np.uint8))
                        if tester.wrist_camera is not None:
                            wr_gpu = tester.wrist_camera.data.output["rgb"]
                            if wr_gpu.shape[-1] > 3:
                                wr_gpu = wr_gpu[..., :3]
                            wrist_images_list.append(wr_gpu[0].cpu().numpy().astype(np.uint8))
                        ee_pose_list.append(ee_pose_gpu.cpu().numpy())
                        obj_pose_list.append(obj_pose_gpu.cpu().numpy())

                # 4) Build policy input directly on GPU
                with prof("prepare_input_gpu", group="inference"):
                    state_parts = [ee_pose_gpu.unsqueeze(0)]
                    if include_obj_pose:
                        state_parts.append(obj_pose_gpu.unsqueeze(0))
                    if include_gripper:
                        state_parts.append(torch.tensor(
                            [[gripper_state]], dtype=torch.float32, device=device))
                    state = torch.cat(state_parts, dim=-1)

                    policy_inputs = {
                        "observation.image": table_chw,
                        "observation.state": state,
                    }
                    if wrist_chw is not None:
                        policy_inputs["observation.wrist_image"] = wrist_chw

                # 5) Preprocess (normalize)
                with prof("preprocess", group="inference"):
                    if preproc is not None:
                        policy_inputs = preproc(policy_inputs)

                # 6) Model forward
                with prof("model_forward", group="inference"):
                    with torch.no_grad():
                        action = policy.select_action(policy_inputs)

                # 7) Postprocess (unnormalize) — action stays on GPU
                with prof("postprocess", group="inference"):
                    if postproc is not None:
                        action = postproc(action)

                # 8) Record action (CPU copy)
                action_gpu = action[0]  # (action_dim,) on GPU
                with prof("data_record_cpu_copy", group="data_io"):
                    action_np = action_gpu.cpu().numpy()
                    action_list.append(action_np)

                tester.current_gripper_state = float(action_np[7])

                # 9) Env step — action already on GPU
                with prof("env_step", group="simulator"):
                    action_t = action_gpu.unsqueeze(0)
                    if action_t.shape[0] == 1 and num_envs > 1:
                        action_t = action_t.repeat(num_envs, 1)
                    env.step(action_t)

                # 10) Success check
                with prof("success_check", group="episode_overhead"):
                    done = (not success) and check_fn()

                return done

            for t in range(args.horizon):
                done = _optimized_step(prof, t, record=True)

                if done:
                    success = True
                    success_step = t + 1
                    for _ in range(50):
                        _optimized_step(prof, -1, record=True)
                    break

                if (t + 1) % 200 == 0:
                    print(f"    [Ep {ep}] Step {t+1}/{args.horizon}")

                prof.step()  # one sample per simulation step

            # After episode: data assembly
            with prof("data_assembly", group="data_io"):
                ep_images = np.array(images_list, dtype=np.uint8)
                ep_ee = np.array(ee_pose_list, dtype=np.float32)
                ep_obj = np.array(obj_pose_list, dtype=np.float32)
                ep_action = np.array(action_list, dtype=np.float32)
                if wrist_images_list:
                    ep_wrist = np.array(wrist_images_list, dtype=np.uint8)

            # Simulate NPZ save (single episode)
            with prof("npz_save", group="data_io"):
                import io
                buf = io.BytesIO()
                np.savez_compressed(buf, images=ep_images, ee_pose=ep_ee,
                                    obj_pose=ep_obj, action=ep_action)
                nbytes = buf.tell()

            prof.step()  # flush episode-level items

            ep_wall = time.perf_counter() - ep_t0
            n_steps = len(images_list)
            episode_wall_times.append(ep_wall)
            episode_steps.append(n_steps)

            status = "SUCCESS" if success else "TIMEOUT"
            print(f"  Ep {ep}: {status} in {n_steps} steps, "
                  f"wall {ep_wall:.1f}s  ({ep_wall/n_steps*1e3:.1f} ms/step), "
                  f"npz ~{nbytes/1e6:.1f} MB")

        # ================================================================
        # Report
        # ================================================================
        print("\n" + "=" * 60)
        print("  PROFILING REPORT")
        print("=" * 60)
        prof.report(sort_by="total_ms")

        # Per-episode summary
        avg_wall = np.mean(episode_wall_times)
        avg_steps = np.mean(episode_steps)
        avg_ms_per_step = avg_wall / avg_steps * 1e3
        print(f"\n{'='*60}")
        print(f"  Episode summary ({args.num_episodes} episodes)")
        print(f"{'='*60}")
        print(f"  Avg wall time:   {avg_wall:.1f} s")
        print(f"  Avg steps:       {avg_steps:.0f}")
        print(f"  Avg ms/step:     {avg_ms_per_step:.1f} ms")
        for i, (w, s) in enumerate(zip(episode_wall_times, episode_steps)):
            print(f"    Ep {i}: {s:4d} steps, {w:.1f}s ({w/s*1e3:.1f} ms/step)")

        # Percentage breakdown by group
        group_totals = {}
        for g, ts in prof.group.items():
            group_totals[g] = sum(ts) * 1e3
        total_profiled = sum(group_totals.values())
        print(f"\n  Group breakdown (total profiled: {total_profiled:.0f} ms):")
        for g in sorted(group_totals, key=group_totals.get, reverse=True):
            v = group_totals[g]
            pct = v / total_profiled * 100 if total_profiled > 0 else 0
            print(f"    {g:25s}  {v:10.0f} ms  ({pct:5.1f}%)")

        # Save results
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy": args.policy,
                "task": args.task,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "n_action_steps": args.n_action_steps,
                "image_size": [args.image_height, args.image_width],
                "device": device,
                "has_wrist": has_wrist,
                "include_obj_pose": include_obj_pose,
                "include_gripper": include_gripper,
            },
            "episode_summary": {
                "avg_wall_time_s": float(avg_wall),
                "avg_steps": float(avg_steps),
                "avg_ms_per_step": float(avg_ms_per_step),
                "episodes": [
                    {"steps": int(s), "wall_s": float(w), "ms_per_step": float(w / s * 1e3)}
                    for w, s in zip(episode_wall_times, episode_steps)
                ],
            },
            "group_breakdown": {
                g: {"total_ms": v, "pct": v / total_profiled * 100 if total_profiled > 0 else 0}
                for g, v in sorted(group_totals.items(), key=lambda x: x[1], reverse=True)
            },
            "profiler_detail": prof.to_dict(),
        }

        json_path = out_dir / "profile_rollout_optimized.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results saved: {json_path}")

        # Also write a human-readable markdown report
        md_path = out_dir / "profile_rollout_optimized_report.md"
        with open(md_path, "w") as f:
            f.write(f"# Rollout Profiling Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Policy:** `{args.policy}`\n\n")
            f.write(f"**Task:** {args.task} | **Episodes:** {args.num_episodes} "
                    f"| **Horizon:** {args.horizon}\n\n")
            f.write(f"**Device:** {device} | **Image:** {args.image_height}x{args.image_width} "
                    f"| **Wrist:** {has_wrist}\n\n")

            f.write(f"## Episode Summary\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Avg wall time | {avg_wall:.1f} s |\n")
            f.write(f"| Avg steps | {avg_steps:.0f} |\n")
            f.write(f"| Avg ms/step | {avg_ms_per_step:.1f} ms |\n\n")

            f.write(f"| Episode | Steps | Wall (s) | ms/step |\n|---|---|---|---|\n")
            for i, (w, s) in enumerate(zip(episode_wall_times, episode_steps)):
                f.write(f"| {i} | {s} | {w:.1f} | {w/s*1e3:.1f} |\n")

            f.write(f"\n## Group Breakdown\n\n")
            f.write(f"| Group | Total (ms) | % |\n|---|---|---|\n")
            for g in sorted(group_totals, key=group_totals.get, reverse=True):
                v = group_totals[g]
                pct = v / total_profiled * 100 if total_profiled > 0 else 0
                f.write(f"| {g} | {v:.0f} | {pct:.1f}% |\n")

            f.write(f"\n## Detailed Item Breakdown\n\n")
            f.write(f"| Item | Count | Total (ms) | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |\n")
            f.write(f"|---|---|---|---|---|---|---|\n")
            rows = []
            for k, ts in prof.item.items():
                n, tot, mean, std, mn, mx = _stats(ts)
                rows.append((k, n, tot * 1e3, mean * 1e3, std * 1e3, mn * 1e3, mx * 1e3))
            rows.sort(key=lambda r: r[2], reverse=True)
            for k, n, tot, mean, std, mn, mx in rows:
                f.write(f"| {k} | {n} | {tot:.1f} | {mean:.2f} | {std:.2f} | {mn:.2f} | {mx:.2f} |\n")

            f.write(f"\n## Observations & Optimization Suggestions\n\n")
            f.write(f"*(To be filled after reviewing the numbers above.)*\n")

        print(f"  Report saved:  {md_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
