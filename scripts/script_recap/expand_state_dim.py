#!/usr/bin/env python3
"""Expand checkpoint state dimension from old_state_dim to new_state_dim.

Modifies the DiffusionPolicy (UNet-based) checkpoint to support an increased
state dimension. The extra dimensions are zero-initialized in all weight
matrices that depend on global_cond_dim.

This is used for Exp56 CFG-RL where we add an indicator dimension to the state
(15 → 16) for advantage-conditioned policy training.

Changes applied:
1. All cond_encoder weight matrices in UNet (down_modules, mid_modules, up_modules)
   → input dim increases by (new_state_dim - old_state_dim) * n_obs_steps
2. config.json → observation.state shape updated
3. Preprocessor normalizer → observation.state shape + stats extended
4. Postprocessor normalizer → observation.state shape + stats extended (if present)

Usage:
    python scripts/script_recap/expand_state_dim.py \
        --input weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
        --output weights/PP_A_cfg/checkpoints/checkpoints/last/pretrained_model \
        --old_state_dim 15 \
        --new_state_dim 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand checkpoint state dimension with zero-init.",
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input pretrained_model directory.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output pretrained_model directory.")
    parser.add_argument("--old_state_dim", type=int, default=15)
    parser.add_argument("--new_state_dim", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    return parser.parse_args()


def expand_cond_encoder_weights(
    state_dict: dict[str, torch.Tensor],
    old_cond_dim: int,
    new_cond_dim: int,
) -> dict[str, torch.Tensor]:
    """Expand all cond_encoder weight matrices from old_cond_dim to new_cond_dim.

    The cond_encoder layers are Linear(diffusion_step_embed_dim + global_cond_dim, out_dim).
    We need to expand the input dimension by (new_cond_dim - old_cond_dim).
    New columns are zero-initialized.
    """
    delta = new_cond_dim - old_cond_dim
    assert delta > 0, f"new_cond_dim ({new_cond_dim}) must be > old_cond_dim ({old_cond_dim})"

    modified_keys = []
    new_state_dict = {}

    for key, tensor in state_dict.items():
        if "cond_encoder" in key and key.endswith(".weight") and tensor.ndim == 2:
            # Linear weight shape: (out_features, in_features)
            out_features, in_features = tensor.shape
            if in_features == old_cond_dim:
                # Expand input dimension
                expansion = torch.zeros(out_features, delta, dtype=tensor.dtype, device=tensor.device)
                new_tensor = torch.cat([tensor, expansion], dim=1)
                new_state_dict[key] = new_tensor
                modified_keys.append(key)
                print(f"  Expanded {key}: ({out_features}, {in_features}) → ({out_features}, {in_features + delta})")
            else:
                new_state_dict[key] = tensor
        else:
            new_state_dict[key] = tensor

    print(f"\n  Modified {len(modified_keys)} cond_encoder weight matrices")
    return new_state_dict


def expand_normalizer_stats(
    stats: dict[str, torch.Tensor],
    old_dim: int,
    new_dim: int,
    indicator_min: float = -1.0,
    indicator_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Expand observation.state normalizer statistics for the extra dimension(s).

    The indicator dimension is in [-1, +1], so min=-1, max=+1.
    Other stats (mean, std, quantiles) are set to reasonable defaults.
    """
    delta = new_dim - old_dim
    new_stats = {}

    for key, tensor in stats.items():
        if key.startswith("observation.state."):
            stat_name = key.split(".")[-1]  # e.g., "min", "max", "mean", ...

            if tensor.shape[0] == old_dim:
                if stat_name == "min":
                    pad = torch.full((delta,), indicator_min, dtype=tensor.dtype)
                elif stat_name == "max":
                    pad = torch.full((delta,), indicator_max, dtype=tensor.dtype)
                elif stat_name == "mean":
                    pad = torch.full((delta,), 0.0, dtype=tensor.dtype)
                elif stat_name == "std":
                    pad = torch.full((delta,), 1.0, dtype=tensor.dtype)
                elif stat_name == "count":
                    new_stats[key] = tensor
                    continue
                elif stat_name.startswith("q"):
                    # Quantiles: set to 0 (neutral value)
                    pad = torch.full((delta,), 0.0, dtype=tensor.dtype)
                else:
                    pad = torch.full((delta,), 0.0, dtype=tensor.dtype)

                new_stats[key] = torch.cat([tensor, pad])
                print(f"  Expanded {key}: {old_dim} → {new_dim}")
            else:
                new_stats[key] = tensor
        else:
            new_stats[key] = tensor

    return new_stats


def main() -> None:
    args = _parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    delta_state = args.new_state_dim - args.old_state_dim
    delta_cond = delta_state * args.n_obs_steps

    print(f"{'='*60}")
    print(f"Expanding State Dimension")
    print(f"  Input:          {input_dir}")
    print(f"  Output:         {output_dir}")
    print(f"  State dim:      {args.old_state_dim} → {args.new_state_dim} (+{delta_state})")
    print(f"  n_obs_steps:    {args.n_obs_steps}")
    print(f"  Cond dim delta: +{delta_cond}")
    print(f"{'='*60}")

    # Step 1: Copy entire directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(input_dir, output_dir)
    print(f"\nCopied checkpoint to {output_dir}")

    # Step 2: Modify model weights
    print(f"\n--- Expanding model weights ---")
    import safetensors.torch

    model_path = output_dir / "model.safetensors"
    if not model_path.exists():
        print(f"ERROR: model.safetensors not found in {output_dir}")
        sys.exit(1)

    state_dict = safetensors.torch.load_file(str(model_path))

    # Find the current cond_dim from any cond_encoder weight
    old_cond_dim = None
    for key, tensor in state_dict.items():
        if "cond_encoder" in key and key.endswith(".weight") and tensor.ndim == 2:
            old_cond_dim = tensor.shape[1]
            break

    if old_cond_dim is None:
        print("ERROR: Could not find any cond_encoder weight matrices")
        sys.exit(1)

    new_cond_dim = old_cond_dim + delta_cond
    print(f"  cond_encoder input dim: {old_cond_dim} → {new_cond_dim}")

    new_state_dict = expand_cond_encoder_weights(state_dict, old_cond_dim, new_cond_dim)
    safetensors.torch.save_file(new_state_dict, str(model_path))
    print(f"  Saved expanded model.safetensors")

    # Step 3: Update config.json
    print(f"\n--- Updating config.json ---")
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    old_shape = config["input_features"]["observation.state"]["shape"]
    config["input_features"]["observation.state"]["shape"] = [args.new_state_dim]
    print(f"  observation.state shape: {old_shape} → [{args.new_state_dim}]")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Step 4: Update preprocessor normalizer stats
    print(f"\n--- Updating preprocessor normalizer ---")
    preproc_json_path = output_dir / "policy_preprocessor.json"
    if preproc_json_path.exists():
        with open(preproc_json_path) as f:
            preproc = json.load(f)
        # Update the state shape in normalizer config
        for step in preproc.get("steps", []):
            if step.get("registry_name") == "normalizer_processor":
                features = step.get("config", {}).get("features", {})
                if "observation.state" in features:
                    features["observation.state"]["shape"] = [args.new_state_dim]
                    print(f"  Updated preprocessor normalizer state shape")
        with open(preproc_json_path, "w") as f:
            json.dump(preproc, f, indent=2)

        # Update the normalizer safetensors
        preproc_stats_path = output_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        if preproc_stats_path.exists():
            stats = safetensors.torch.load_file(str(preproc_stats_path))
            new_stats = expand_normalizer_stats(stats, args.old_state_dim, args.new_state_dim)
            safetensors.torch.save_file(new_stats, str(preproc_stats_path))
            print(f"  Updated preprocessor normalizer stats")

    # Step 5: Update postprocessor (if it has state normalization)
    postproc_json_path = output_dir / "policy_postprocessor.json"
    if postproc_json_path.exists():
        with open(postproc_json_path) as f:
            postproc = json.load(f)
        for step in postproc.get("steps", []):
            if step.get("registry_name") == "unnormalizer_processor":
                features = step.get("config", {}).get("features", {})
                if "observation.state" in features:
                    features["observation.state"]["shape"] = [args.new_state_dim]
                    print(f"  Updated postprocessor state shape")
        with open(postproc_json_path, "w") as f:
            json.dump(postproc, f, indent=2)

        postproc_stats_path = output_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if postproc_stats_path.exists():
            stats = safetensors.torch.load_file(str(postproc_stats_path))
            # Check if observation.state stats exist in postprocessor
            has_state_stats = any(k.startswith("observation.state.") for k in stats)
            if has_state_stats:
                new_stats = expand_normalizer_stats(stats, args.old_state_dim, args.new_state_dim)
                safetensors.torch.save_file(new_stats, str(postproc_stats_path))
                print(f"  Updated postprocessor normalizer stats")

    # Step 6: Also copy the training_state directory if it exists alongside
    # (The caller is responsible for copying the full checkpoint structure)

    print(f"\n{'='*60}")
    print(f"State dimension expansion complete!")
    print(f"  {args.old_state_dim} → {args.new_state_dim}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
