"""GAE advantage estimation and frame-level filtering for reversed B rollout data.

Core workflow:
  1. Run critic inference → per-timestep predicted values V(t)
  2. Compute GAE advantage from V(t) → per-timestep A(t)
  3. Sliding-window smoothing of advantage → identify low-advantage segments to remove
  4. Output per-frame keep/drop mask
"""

from __future__ import annotations

import numpy as np


def compute_gae_from_values(
    values: np.ndarray,
    gamma: float = 0.995,
    lam: float = 0.95,
    rewards: np.ndarray | None = None,
    terminal_bootstrap: bool = True,
) -> np.ndarray:
    """Compute GAE advantage from a predicted value sequence.

    Args:
        values: (T,) critic predicted per-timestep value.
        gamma: Discount factor (should match critic training).
        lam: GAE λ. λ=1 → MC return diff, λ=0 → 1-step TD.
        rewards: (T,) optional per-step reward. Default None → all zeros (sparse reward).
        terminal_bootstrap: If True, set next_value at terminal = values[-1]
            (assume value stays constant after episode end) to avoid ghost negative spike.
            If False, set next_value = 0 at terminal.

    Returns:
        advantages: (T,) per-timestep GAE advantage.
    """
    T = len(values)
    if rewards is None:
        rewards = np.zeros(T, dtype=np.float32)

    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_value = values[t] if terminal_bootstrap else 0.0
        else:
            next_value = values[t + 1]
        delta_t = rewards[t] + gamma * next_value - values[t]
        last_gae = delta_t + gamma * lam * last_gae
        advantages[t] = last_gae

    return advantages


def compute_frame_filter_mask(
    advantages: np.ndarray,
    smooth_window: int = 51,
    drop_threshold: float = 0.0,
    min_drop_length: int = 50,
    min_keep_length: int = 50,
) -> np.ndarray:
    """Compute per-frame keep/drop mask from advantage sequence.

    Pipeline:
      1. Smooth advantages with a centered moving average (window=smooth_window).
      2. Mark frames where smoothed advantage < drop_threshold as candidates for dropping.
      3. Remove short drop segments (< min_drop_length) → keep them instead.
      4. Remove short keep segments (< min_keep_length) surrounded by drops → drop them too.

    This avoids overly fragmented filtering.

    Args:
        advantages: (T,) raw GAE advantages.
        smooth_window: Moving average window size (odd recommended).
        drop_threshold: Frames with smoothed advantage below this are drop candidates.
        min_drop_length: Drop segments shorter than this are converted back to keep.
        min_keep_length: Keep segments shorter than this (sandwiched by drops) are converted to drop.

    Returns:
        keep_mask: (T,) bool array. True = keep frame, False = drop frame.
    """
    T = len(advantages)
    if T == 0:
        return np.array([], dtype=bool)

    # Step 1: Smooth advantages with centered moving average
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    # Pad edges with reflection to reduce boundary artifacts
    pad = smooth_window // 2
    padded = np.pad(advantages, (pad, pad), mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")[:T]

    # Step 2: Initial mask — keep frames with smoothed adv >= threshold
    keep_mask = smoothed >= drop_threshold

    # Step 3: Remove short drop runs (< min_drop_length)
    keep_mask = _remove_short_runs(keep_mask, target_value=False, min_length=min_drop_length)

    # Step 4: Remove short keep runs sandwiched by drops (< min_keep_length)
    keep_mask = _remove_short_runs(keep_mask, target_value=True, min_length=min_keep_length)

    return keep_mask


def _remove_short_runs(mask: np.ndarray, target_value: bool, min_length: int) -> np.ndarray:
    """Flip runs of `target_value` shorter than `min_length` to `not target_value`.

    For target_value=False, min_length=50: drop segments shorter than 50 become keep.
    For target_value=True, min_length=50: keep segments shorter than 50 (sandwiched by drops) become drop.
    """
    mask = mask.copy()
    T = len(mask)
    i = 0
    while i < T:
        if mask[i] == target_value:
            # Find the end of this run
            j = i
            while j < T and mask[j] == target_value:
                j += 1
            run_length = j - i
            if run_length < min_length:
                # For keep segments: only flip if truly sandwiched (not at episode boundaries)
                if target_value:
                    if i > 0 and j < T:
                        mask[i:j] = not target_value
                else:
                    # For drop segments: always flip short ones back to keep
                    mask[i:j] = not target_value
            i = j
        else:
            i += 1
    return mask


def filter_episode_frames(
    episode: dict,
    keep_mask: np.ndarray,
) -> dict:
    """Apply keep_mask to an episode, returning a new episode with only kept frames.

    Time-indexed arrays (images, ee_pose, obj_pose, action, etc.) are filtered.
    Scalar fields (success, place_pose, goal_pose) are copied as-is.

    Also recomputes action[t][:7] = ee_pose[t+1] for consistency after frame removal.

    Args:
        episode: Episode dict with time-indexed arrays of length T.
        keep_mask: (T,) bool array.

    Returns:
        Filtered episode dict. Also includes:
            "_original_length": int, original T
            "_kept_length": int, filtered T
            "_keep_ratio": float
    """
    T_orig = len(keep_mask)
    indices = np.where(keep_mask)[0]
    T_new = len(indices)

    if T_new == 0:
        # Edge case: nothing kept. Return empty episode with metadata.
        result = {}
        for key, val in episode.items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == T_orig:
                result[key] = val[:0]  # empty array with same dtype/dims
            else:
                result[key] = val
        result["_original_length"] = T_orig
        result["_kept_length"] = 0
        result["_keep_ratio"] = 0.0
        return result

    result = {}
    for key, val in episode.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == T_orig:
            result[key] = val[indices].copy()
        else:
            result[key] = val

    # Recompute action[:, :7] = ee_pose of next frame for consistency
    # action[t][:7] should be ee_pose[t+1], and action[-1][:7] = ee_pose[-1]
    if "action" in result and "ee_pose" in result and T_new > 1:
        result["action"][:-1, :7] = result["ee_pose"][1:]
        result["action"][-1, :7] = result["ee_pose"][-1]

    result["_original_length"] = T_orig
    result["_kept_length"] = T_new
    result["_keep_ratio"] = T_new / T_orig

    return result
