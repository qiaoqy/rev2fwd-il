from __future__ import annotations

from typing import Any, Mapping

import torch


def _as_2d_tensor(x: Any) -> torch.Tensor:
    """Convert x to a 2D torch tensor (num_envs, dim)."""

    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)

    if t.ndim == 1:
        # (dim,) -> (1, dim)
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        # (N, ...) -> (N, prod(...))
        t = t.view(t.shape[0], -1)

    return t


def extract_policy_obs(obs_dict: Mapping[str, Any]) -> torch.Tensor:
    """Extract policy observation tensor from Isaac Lab env obs dict.

    Isaac Lab tasks typically return observations as a dict with a "policy" entry.

    Behavior:
        - Prefer obs_dict["policy"].
        - If obs_dict["policy"] is already a Tensor/array, it is reshaped to (num_envs, obs_dim).
        - If obs_dict["policy"] is itself a dict, we sort keys lexicographically,
          convert each leaf value to a 2D tensor (num_envs, dim_i), then concatenate
          along the last dimension to produce (num_envs, sum(dim_i)).

    Args:
        obs_dict: The observation dict returned by env.reset() or env.step().

    Returns:
        A tensor of shape (num_envs, obs_dim).

    Raises:
        KeyError: If "policy" key is missing.
        ValueError: If dict leaves have incompatible num_envs.
    """

    if "policy" not in obs_dict:
        raise KeyError(f"Observation dict has no 'policy' key. keys={list(obs_dict.keys())}")

    policy = obs_dict["policy"]

    if isinstance(policy, Mapping):
        parts: list[torch.Tensor] = []
        num_envs: int | None = None
        for k in sorted(policy.keys()):
            t = _as_2d_tensor(policy[k])
            if num_envs is None:
                num_envs = int(t.shape[0])
            elif int(t.shape[0]) != num_envs:
                raise ValueError(
                    f"Inconsistent num_envs when concatenating policy obs: key={k}, got={t.shape[0]}, expected={num_envs}"
                )
            parts.append(t)

        if not parts:
            raise ValueError("obs_dict['policy'] is an empty dict.")

        return torch.cat(parts, dim=-1)

    return _as_2d_tensor(policy)
