"""Bellman return computation for critic training."""

import numpy as np


def compute_bellman_returns(
    episodes: list[dict],
    gamma: float = 0.99,
    success_reward: float = 1.0,
) -> list[np.ndarray]:
    """Compute discounted Bellman returns for each episode.

    Reward definition (sparse):
      - Successful episode: r(success_step) = success_reward, r(t != success_step) = 0
      - Failed episode: r(t) = 0 for all t

    Bellman return (backward from terminal step):
      V(T-1) = r(T-1)
      V(t)   = r(t) + gamma * V(t+1)

    For sparse reward with success_step S, this gives:
      - t <= S: V(t) = gamma^(S - t) * success_reward
      - t >  S: V(t) = 0

    Args:
        episodes: List of episode dicts. Each dict must contain:
            - "action": np.ndarray of shape (T, action_dim), used to determine episode length
            - "success": bool
            - "success_step": int (required if success=True), the timestep where success is first achieved
        gamma: Discount factor.
        success_reward: Reward value at the success timestep.

    Returns:
        List of np.ndarray, each of shape (T,), aligned with the corresponding episode's action length.
    """
    bellman_values = []
    for ep in episodes:
        T = len(ep["action"])
        success = ep.get("success", False)

        if not success:
            bellman_values.append(np.zeros(T, dtype=np.float32))
            continue

        success_step = ep.get("success_step", T - 1)
        # Clamp to valid range
        success_step = min(success_step, T - 1)

        values = np.zeros(T, dtype=np.float32)
        # Assign reward at success_step, then propagate backward
        values[success_step] = success_reward
        for t in range(success_step - 1, -1, -1):
            values[t] = gamma * values[t + 1]
        # Steps after success_step remain 0

        bellman_values.append(values)

    return bellman_values
