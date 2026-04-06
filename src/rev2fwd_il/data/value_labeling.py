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


def compute_mc_returns(
    episodes: list[dict],
    max_episode_length: int = 3000,
) -> list[np.ndarray]:
    """Monte Carlo return with per-step penalty and success terminal reward.

    Reward definition:
        r(t) = -1  for all t in {0, ..., T-1}
        R_success = max_episode_length  (added at final step of success episodes)

    Value definition (undiscounted, gamma=1):
        Success episode:  V(t) = R_success - (T - t)
        Failure episode:  V(t) = -(T - t)

    Normalized to [-1, 0):
        V_bar(t) = (V(t) - R_success) / (2 * R_success)

    Properties:
        - Success episodes: V_bar(t) in (-0.5, 0)
          V_bar(T-1) = -1/(2R) ≈ 0,  V_bar(0) when T=R: -0.5
        - Failure episodes: V_bar(t) in (-1, -0.5)
          V_bar(T-1) = -0.5 - 1/(2R) ≈ -0.5,  V_bar(0) when T=R: -1
        - -0.5 is the natural success/failure boundary
        - Shorter success episodes have higher (less negative) V_bar(0)

    Args:
        episodes: List of episode dicts with "action" (T, D) and "success" (bool).
        max_episode_length: R_success = normalization constant. Should be >= longest
            episode length to ensure success V(0) >= -0.5.

    Returns:
        List of np.ndarray, each (T,) float32.
    """
    mc_values = []
    R = max_episode_length
    for ep in episodes:
        T = len(ep["action"])
        success = ep.get("success", False)
        remaining = np.arange(T, 0, -1, dtype=np.float32)  # [T, T-1, ..., 1]
        if success:
            # V_mc = R - remaining; V_bar = (V_mc - R) / (2R) = -remaining / (2R)
            values = -remaining / (2 * R)
        else:
            # V_mc = -remaining; V_bar = (-remaining - R) / (2R) = -0.5 - remaining / (2R)
            values = -0.5 - remaining / (2 * R)
        mc_values.append(values)
    return mc_values
