#!/usr/bin/env python3
"""Step 5: Finetune policy with rollout data (DAgger-style data aggregation).

Merges new rollout data into an existing LeRobot dataset and prepares
checkpoint structure for ``4_train.py --resume``.  Uses 1:1 weighted
sampling between original and new data.

Usage:
    conda activate rev2fwd_il

    # Prepare only (recommended for multi-GPU)
    python scripts/scripts_pick_place_simulator/5_finetune.py \\
        --original_lerobot data/exp_new/weights/PP_A/lerobot_dataset \\
        --rollout_data data/exp_new/iter1_collect_A.npz \\
        --checkpoint data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out data/exp_new/work/PP_A_iter1 --prepare_only

    # Then train with multi-GPU
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        scripts/scripts_pick_place_simulator/4_train.py \\
        --dataset dummy.npz \\
        --lerobot_dataset_dir data/exp_new/work/PP_A_iter1/lerobot_dataset \\
        --out data/exp_new/work/PP_A_iter1 \\
        --steps 36072 --batch_size 128 --skip_convert --resume \\
        --sample_weights data/exp_new/work/PP_A_iter1/lerobot_dataset/meta/sampling_weights.json
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Import the original finetune script
_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SCRIPT = _THIS_DIR.parent / "scripts_pick_place" / "7_finetune_with_rollout.py"

_spec = importlib.util.spec_from_file_location("_orig_finetune", _ORIG_SCRIPT)
_orig_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig_mod)

# Re-export key functions
copy_lerobot_dataset = _orig_mod.copy_lerobot_dataset
add_episodes_to_lerobot_dataset = _orig_mod.add_episodes_to_lerobot_dataset
load_episodes_from_npz = _orig_mod.load_episodes_from_npz

if __name__ == "__main__":
    _orig_mod.main()
