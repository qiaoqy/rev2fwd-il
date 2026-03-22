#!/usr/bin/env python3
"""Step 3: Inspect and visualize collected trajectory data.

Thin wrapper around scripts_pick_place/2_inspect_data.py — delegates all
logic to that script's functions.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/3_inspect_data.py \
        --dataset data/exp_new/task_B_100.npz --episode 0 --frame 30

    python scripts/scripts_pick_place_simulator/3_inspect_data.py \
        --dataset data/exp_new/task_A_reversed_100.npz \
        --enable_xyz_viz --episode 0
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Import the original inspect script via importlib (filename starts with digit)
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SCRIPT = _THIS_DIR.parent / "scripts_pick_place" / "2_inspect_data.py"

_spec = importlib.util.spec_from_file_location("_orig_inspect", _ORIG_SCRIPT)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)

if __name__ == "__main__":
    _orig_mod.main()
