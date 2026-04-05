"""Bellman return computation for critic training."""

import numpy as np


def compute_bellman_returns(
    episodes: list[dict],
    gamma: float = 0.995,
    success_reward: float = 1.0,
) -> list[np.ndarray]:
    """Compute discounted Bellman returns for each episode.

    Reward definition (sparse):
      - Successful episode: virtual reward at step T (episode end), r = success_reward
      - Failed episode: r(t) = 0 for all t

    Value assignment for successful episodes (T = total episode length):
      - V(t) = gamma^(T - t) * success_reward  (discounted from episode end)
      - V(T-1) = gamma * success_reward  (last frame, one step from virtual terminal)
      - V(0) = gamma^T * success_reward  (first frame)

    This treats the entire episode as necessary for success — the success point
    is the episode boundary rather than an intermediate success_step.

    Failed episodes: V(t) = 0 for all t.

    Args:
        episodes: List of episode dicts. Each dict must contain:
            - "action": np.ndarray of shape (T, action_dim), used to determine episode length
            - "success": bool
        gamma: Discount factor.
        success_reward: Reward value at the virtual terminal state.

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

        # Discount from virtual terminal at step T
        # V(t) = gamma^(T - t) * success_reward
        exponents = np.arange(T, 0, -1, dtype=np.float32)  # [T, T-1, ..., 1]
        values = (gamma ** exponents) * success_reward

        bellman_values.append(values)

    return bellman_values
