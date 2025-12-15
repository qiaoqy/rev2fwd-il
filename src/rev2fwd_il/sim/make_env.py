from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_env(task_id: str, num_envs: int, device: str, use_fabric: bool) -> gym.Env:
    """Create an Isaac Lab gym environment.

    Notes:
        - This function assumes Isaac Sim is already launched via isaaclab.app.AppLauncher
          in the calling script.
        - We import isaaclab_tasks here to ensure tasks are registered with gym.

    Args:
        task_id: Gym task id, e.g. "Isaac-Lift-Cube-Franka-IK-Abs-v0".
        num_envs: Number of vectorized env instances.
        device: Torch device string, e.g. "cuda" or "cpu".
        use_fabric: If False, disables Fabric backend in the parsed env cfg.

    Returns:
        The created gym environment.
    """

    # Ensure Isaac Lab tasks are registered.
    import isaaclab_tasks  # noqa: F401

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    env = gym.make(task_id, cfg=env_cfg)
    return env
