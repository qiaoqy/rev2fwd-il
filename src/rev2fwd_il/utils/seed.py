from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = False) -> int:
    """Set RNG seed for python/random, numpy and torch.

    Args:
        seed: The seed to set.
        deterministic_torch: If True, enables PyTorch deterministic mode.
            This may reduce performance and is not always fully deterministic on GPU.

    Returns:
        The seed used.
    """

    seed_int = int(seed)

    random.seed(seed_int)
    np.random.seed(seed_int)

    try:
        import torch

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older/newer torch builds may not support all flags.
                pass
    except ModuleNotFoundError:
        # Torch is optional for some tooling.
        pass

    return seed_int
