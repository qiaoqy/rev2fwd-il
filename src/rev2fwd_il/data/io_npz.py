"""I/O utilities for saving and loading episode datasets."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .episode import Episode


def save_episodes(path: str | Path, episodes: List[Episode]) -> None:
    """Save a list of episodes to a compressed npz file.

    Args:
        path: Output file path (should end with .npz).
        episodes: List of Episode objects to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert episodes to list of dicts
    episode_dicts = [ep.to_dict() for ep in episodes]

    # Save with allow_pickle for list of dicts
    np.savez_compressed(path, episodes=episode_dicts)
    print(f"Saved {len(episodes)} episodes to {path}")


def load_episodes(path: str | Path) -> List[Episode]:
    """Load episodes from a compressed npz file.

    Args:
        path: Input file path.

    Returns:
        List of Episode objects.
    """
    path = Path(path)

    with np.load(path, allow_pickle=True) as data:
        episode_dicts = data["episodes"]

    episodes = [Episode.from_dict(d) for d in episode_dicts]
    print(f"Loaded {len(episodes)} episodes from {path}")
    return episodes
