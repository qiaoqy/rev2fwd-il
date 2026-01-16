from __future__ import annotations

"""Isaac Lab environment sanity-check runner.

This script is intentionally minimal and is meant to answer one question:

    "Can my Isaac Lab 2.x + Isaac Sim 5.x installation launch a gymnasium task and step it?"

What it does:
    1) Uses :class:`isaaclab.app.AppLauncher` to start Isaac Sim (GUI or headless).
    2) Creates an Isaac Lab gym environment via ``gym.make(task_id, cfg=env_cfg)``.
    3) Calls ``env.reset()`` and prints:
        - action_space / observation_space
        - observation dict keys
        - flattened policy observation shape/dtype/device
        - major scene entities available under ``env.unwrapped.scene``
    4) Runs a short stepping loop and prints per-step diagnostics.

What it does NOT do:
    - No RL training, no policy learning.
    - No "expert" policy; actions are zero / random / small debug perturbation.

Notes on Isaac Lab specifics:
    - Isaac Lab environments typically expect **torch tensors on the simulation device**
      (often ``cuda:0``) for actions. Passing numpy arrays may work sometimes but can
      lead to silent no-ops depending on wrappers.
    - Many ``scene[...].data`` fields are **live tensor views** that update in-place.
      For proper "previous vs current" comparisons, we clone the tensors.
"""

import argparse
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rev2fwd_il.sim.make_env import make_env
from rev2fwd_il.sim.obs_utils import extract_policy_obs
from rev2fwd_il.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Important:
        ``AppLauncher.add_app_launcher_args(parser)`` will inject standard Isaac Lab / Isaac Sim
        launcher arguments (e.g. ``--headless`` and ``--device``). We must not add those names
        ourselves to avoid conflicts.

    Returns:
        Parsed args (argparse Namespace) containing both our custom args and AppLauncher args.
    """

    parser = argparse.ArgumentParser(description="Minimal Isaac Lab env sanity check.")

    # ----------------------------
    # Task / env configuration
    # ----------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Gym task id to create via gym.make(...).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (vectorized) inside the same simulation.",
    )

    # "device" is reserved by AppLauncher. Provide a non-conflicting alias.
    parser.add_argument(
        "--rl_device",
        type=str,
        default=None,
        help="Alias for AppLauncher --device (e.g., cuda:0/cuda/cpu). If set, overrides args.device.",
    )

    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric in env cfg (some builds log fewer Fabric warnings but may be slower).",
    )

    # ----------------------------
    # Loop control
    # ----------------------------
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of env.step(...) calls to run after reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for python/numpy/torch RNG (note: env internal seed may still be None unless set in cfg).",
    )

    # ----------------------------
    # Action generation
    # ----------------------------
    parser.add_argument(
        "--random_action",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, samples actions from env.action_space each step. Otherwise uses zero action.",
    )
    parser.add_argument(
        "--nudge_action",
        type=float,
        default=0.0,
        help=(
            "If non-zero, uses a stable zero action plus a small deterministic perturbation on action dim-0. "
            "This is useful to debug whether actions have any effect (recommended magnitude <= 0.05)."
        ),
    )

    # ----------------------------
    # Debug logging
    # ----------------------------
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print diagnostics every N steps (1 prints every step).",
    )
    parser.add_argument(
        "--step_timeout_s",
        type=float,
        default=20.0,
        help="If a single env.step(...) call takes longer than this, print a warning.",
    )

    # Add Isaac Lab launcher flags (headless, device, experience, etc.).
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Map alias args -> AppLauncher args if provided.
    if args.rl_device is not None:
        args.device = args.rl_device

    return args


def _zero_action_for_space(space: gym.Space, num_envs: int) -> Any:
    """Create a zero action with the correct batch shape.

    For most Isaac Lab tasks, action space is a gymnasium Box.

    Args:
        space: env.action_space.
        num_envs: Number of parallel environments.

    Returns:
        Numpy array with zeros and shape (num_envs, action_dim) for Box spaces.
    """

    if isinstance(space, gym.spaces.Box):
        # Many Isaac Lab tasks expose batched spaces: (num_envs, action_dim).
        shape = (num_envs, *space.shape)
        return np.zeros(shape, dtype=space.dtype)

    # Generic fallback for non-Box spaces.
    sample = space.sample()
    try:
        z = np.zeros_like(sample)
        if num_envs == 1:
            return z
        return np.stack([z for _ in range(num_envs)], axis=0)
    except Exception:
        if num_envs == 1:
            return sample
        return np.stack([space.sample() for _ in range(num_envs)], axis=0)


def _action_to_env(action: Any, *, device: str, num_envs: int, action_dim: int) -> torch.Tensor:
    """Convert action to a torch tensor expected by Isaac Lab.

    Isaac Lab environments generally expect actions as torch tensors on the simulation device
    (usually ``cuda:0``). This helper ensures:
        - dtype: float32
        - shape: (num_envs, action_dim)
        - device: matches env device
        - NaN / Inf are replaced with 0 to avoid controller/solver issues

    Args:
        action: Numpy array or torch tensor action.
        device: Torch device (e.g. "cuda:0").
        num_envs: Number of parallel envs.
        action_dim: Flat action dimension.

    Returns:
        Torch Tensor action on the target device.
    """

    if isinstance(action, torch.Tensor):
        t = action
    else:
        t = torch.as_tensor(action, dtype=torch.float32)

    # Ensure (N, D).
    if t.ndim == 1:
        t = t.unsqueeze(0)

    # Ensure correct batch dimension.
    if t.shape[0] != num_envs:
        if t.shape[0] == 1 and num_envs > 1:
            t = t.repeat(num_envs, 1)
        else:
            t = t.view(num_envs, -1)

    # Ensure correct action dimension.
    if t.shape[1] != action_dim:
        t = t.view(num_envs, action_dim)

    t = t.to(device=device, dtype=torch.float32)

    # Guard against NaN/Inf (can stall some controllers/solvers).
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    return t


def _clone_if_tensor(x: Any) -> Any:
    """Snapshot a tensor value.

    Many Isaac Lab ``.data`` tensors are live views updated in-place each simulation step.
    We clone them so that "previous" and "current" snapshots do not alias same memory.
    """

    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    return x


def _try_get_scene_state(env: gym.Env) -> dict[str, Any]:
    """Try to fetch a small set of useful state tensors from the scene.

    This is best-effort (keys/types depend on the specific task).

    Returns:
        Dict with optional entries:
            - robot_joint_pos: (N, dof)
            - robot_joint_vel: (N, dof)
            - object_root_pos: (N, 3)
    """

    out: dict[str, Any] = {}
    try:
        scene = env.unwrapped.scene
    except Exception:
        return out

    # Robot joint state.
    try:
        robot = scene["robot"]
        out["robot_joint_pos"] = _clone_if_tensor(getattr(robot.data, "joint_pos", None))
        out["robot_joint_vel"] = _clone_if_tensor(getattr(robot.data, "joint_vel", None))
    except Exception:
        pass

    # Object root position.
    try:
        obj = scene["object"]
        out["object_root_pos"] = _clone_if_tensor(getattr(obj.data, "root_pos_w", None))
    except Exception:
        pass

    return out


def main() -> None:
    """Entry point."""

    args_cli = _parse_args()

    # Seed python/numpy/torch.
    set_seed(args_cli.seed)

    # Launch Isaac Sim.
    from isaaclab.app import AppLauncher

    args_cli.headless = bool(args_cli.headless)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # Create environment after Isaac Sim is initialized.
        env = make_env(
            task_id=args_cli.task,
            num_envs=args_cli.num_envs,
            device=args_cli.device,
            use_fabric=not bool(args_cli.disable_fabric),
        )

        # Reset once and print static info.
        obs_dict, _ = env.reset()

        print("action_space:", env.action_space)
        print("observation_space:", env.observation_space)

        print("obs_dict keys:", list(obs_dict.keys()))
        policy_obs = extract_policy_obs(obs_dict)
        print(
            "policy_obs:",
            "shape=",
            tuple(policy_obs.shape),
            "dtype=",
            policy_obs.dtype,
            "device=",
            policy_obs.device,
        )

        # Scene entities registered in the InteractiveScene.
        try:
            scene_keys = list(env.unwrapped.scene.keys())
        except Exception as e:
            scene_keys = []
            print("Failed to read env.unwrapped.scene.keys():", repr(e))
        print("scene keys:", scene_keys)

        # Snapshot initial state.
        prev_state = _try_get_scene_state(env)
        if prev_state:
            jp = prev_state.get("robot_joint_pos")
            if isinstance(jp, torch.Tensor):
                print("initial robot_joint_pos mean:", float(jp.mean().item()))

        # Flat action dimension: action space is typically shaped (num_envs, action_dim).
        action_dim = int(np.prod(env.action_space.shape[1:])) if hasattr(env.action_space, "shape") else 0

        t0 = time.perf_counter()
        for i in range(int(args_cli.steps)):
            # ----------------------------
            # Build action
            # ----------------------------
            if float(args_cli.nudge_action) != 0.0:
                # Stable zero + small nudge is more predictable than sampling from an unbounded Box.
                base = _zero_action_for_space(env.action_space, args_cli.num_envs)
                action_t = _action_to_env(
                    base,
                    device=str(args_cli.device),
                    num_envs=args_cli.num_envs,
                    action_dim=action_dim,
                )

                # Limit the magnitude to avoid pushing IK targets too far.
                n = float(np.clip(float(args_cli.nudge_action), -0.05, 0.05))
                action_t = action_t.clone()
                action_t[:, 0] += n
            else:
                if bool(args_cli.random_action):
                    # Random action (mainly to confirm actuation is alive).
                    if args_cli.num_envs == 1:
                        action = env.action_space.sample()
                    else:
                        action = np.stack([env.action_space.sample() for _ in range(args_cli.num_envs)], axis=0)
                else:
                    # Zero action (baseline).
                    action = _zero_action_for_space(env.action_space, args_cli.num_envs)

                action_t = _action_to_env(
                    action,
                    device=str(args_cli.device),
                    num_envs=args_cli.num_envs,
                    action_dim=action_dim,
                )

            # ----------------------------
            # Step environment
            # ----------------------------
            if (i % int(args_cli.print_every)) == 0:
                print(f"calling env.step at i={i} ...")

            step_start = time.perf_counter()
            obs_dict, reward, terminated, truncated, _ = env.step(action_t)
            step_dt = time.perf_counter() - step_start

            if step_dt > float(args_cli.step_timeout_s):
                print(f"WARNING: env.step took {step_dt:.2f}s (timeout={args_cli.step_timeout_s}s)")

            if (i % int(args_cli.print_every)) == 0:
                print(f"returned from env.step at i={i} (dt={step_dt*1000:.2f}ms)")

            # ----------------------------
            # Diagnostics
            # ----------------------------
            policy_obs = extract_policy_obs(obs_dict)

            if (i % int(args_cli.print_every)) == 0:
                # Reward may be float/np/torch.
                if isinstance(reward, torch.Tensor):
                    r_mean = float(reward.mean().item())
                else:
                    r_mean = float(np.mean(reward))

                def _shape(x: Any) -> Any:
                    if isinstance(x, torch.Tensor):
                        return tuple(x.shape)
                    if isinstance(x, np.ndarray):
                        return x.shape
                    return type(x)

                print(
                    f"step {i+1}/{args_cli.steps} | dt={step_dt*1000:.2f}ms | reward_mean={r_mean:.4f} | "
                    f"terminated={_shape(terminated)} truncated={_shape(truncated)} | action_mean={float(action_t.mean().item()):.4f}"
                )

                # Compare state snapshots.
                cur_state = _try_get_scene_state(env)
                if prev_state and cur_state:
                    jp0 = prev_state.get("robot_joint_pos")
                    jp1 = cur_state.get("robot_joint_pos")
                    if isinstance(jp0, torch.Tensor) and isinstance(jp1, torch.Tensor):
                        djp = float((jp1 - jp0).abs().mean().item())
                        print(f"  robot_joint_pos | mean_abs_delta={djp:.6e} | mean={float(jp1.mean().item()):.6f}")

                    jv0 = prev_state.get("robot_joint_vel")
                    jv1 = cur_state.get("robot_joint_vel")
                    if isinstance(jv0, torch.Tensor) and isinstance(jv1, torch.Tensor):
                        djv = float((jv1 - jv0).abs().mean().item())
                        print(f"  robot_joint_vel | mean_abs_delta={djv:.6e} | mean={float(jv1.mean().item()):.6f}")

                    op0 = prev_state.get("object_root_pos")
                    op1 = cur_state.get("object_root_pos")
                    if isinstance(op0, torch.Tensor) and isinstance(op1, torch.Tensor):
                        dop = float((op1 - op0).abs().mean().item())
                        print(f"  object_root_pos  | mean_abs_delta={dop:.6e} | mean={float(op1.mean().item()):.6f}")

                    prev_state = cur_state

                # Flattened policy obs stats.
                mean = float(torch.mean(policy_obs).item())
                std = float(torch.std(policy_obs).item())
                print(f"  policy_obs | mean={mean:.6f} std={std:.6f} | device={policy_obs.device}")

            # ----------------------------
            # Reset logic
            # ----------------------------
            # For simplicity, reset the whole vector env if any sub-env terminates/truncates.
            def _any_true(x: Any) -> bool:
                if isinstance(x, torch.Tensor):
                    return bool(torch.any(x).item())
                if isinstance(x, np.ndarray):
                    return bool(np.any(x))
                if isinstance(x, (bool, np.bool_)):
                    return bool(x)
                return False

            if _any_true(terminated) or _any_true(truncated):
                obs_dict, _ = env.reset()
                prev_state = _try_get_scene_state(env)

        t_total = time.perf_counter() - t0
        print(
            f"done. steps={args_cli.steps} total_time={t_total:.3f}s avg_step={(t_total/max(1,args_cli.steps))*1000:.2f}ms"
        )

        env.close()

    finally:
        # Always close Kit.
        simulation_app.close()


if __name__ == "__main__":
    main()
