from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_env(
    task_id: str,
    num_envs: int,
    device: str,
    use_fabric: bool,
    episode_length_s: float | None = None,
    disable_terminations: bool = False,
) -> gym.Env:
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
        episode_length_s: Override episode length in seconds. If None, uses default.
            Set to a large value (e.g., 100.0) to prevent auto-reset during data collection.
        disable_terminations: If True, disables all termination conditions (time_out, object_dropping).
            This prevents the environment from auto-resetting when terminated/truncated,
            allowing the robot to move naturally back to rest position.

    Returns:
        The created gym environment.
    """

    # Ensure Isaac Lab tasks are registered.
    import isaaclab_tasks  # noqa: F401

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    
    # Override episode length if specified (prevents auto-reset during data collection)
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
    
    # Disable terminations to prevent auto-reset (robot won't teleport back)
    if disable_terminations:
        # Disable time_out termination
        if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'time_out'):
            env_cfg.terminations.time_out = None
        # Disable object_dropping termination
        if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'object_dropping'):
            env_cfg.terminations.object_dropping = None
    
    env = gym.make(task_id, cfg=env_cfg)
    return env
