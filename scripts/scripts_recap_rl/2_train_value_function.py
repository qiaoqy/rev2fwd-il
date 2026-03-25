#!/usr/bin/env python3
"""Step 2: Train the distributional value function for RECAP.

Builds a FrozenEncoderValueFunction on top of a pretrained DiffusionPolicy
(frozen ResNet18 encoder + new 3-layer MLP head).

Training data: demo NPZ file(s) + rollout NPZ file(s) collected by step 1.
Using all available data (demos + rollouts) provides a better coverage of
the state space and stabilises the value estimate.

The value function is supervised with Monte Carlo returns:
  R_t = (-(T-t) + penalty) / (MAX_EP_LEN + C_FAIL)
  penalty = 0 if success, -C_FAIL = -1200 if failure

Training uses cross-entropy loss on discretised (32-bin) return targets
instead of MSE, which is more stable under bimodal success/failure returns.

Usage:
    # Train on Task A demos + rollouts (no simulator needed for this step)
    python scripts/scripts_recap_rl/2_train_value_function.py \\
        --policy data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --npz_paths data/exp_new/task_A_reversed_100.npz \\
                    data/recap_exp/rollouts_A_200.npz \\
        --out data/recap_exp/vf_A.pt \\
        --epochs 200 --batch_size 512
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train distributional value function (no simulator needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Pretrained DiffusionPolicy checkpoint path.")
    parser.add_argument("--npz_paths", type=str, nargs="+", required=True,
                        help="NPZ data files (demo + rollout, can repeat).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output value function checkpoint path (.pt).")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_bins", type=int, default=32)
    parser.add_argument("--c_fail", type=float, default=1200.0)
    parser.add_argument("--max_ep_len", type=float, default=1200.0)

    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precompute_batch_size", type=int, default=64,
                        help="Batch size for feature pre-computation pass.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import (
        build_vf_model,
        load_episodes_from_npz_list,
        precompute_features,
        save_vf_checkpoint,
        C_FAIL, MAX_EP_LEN,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"RECAP — Value Function Training")
    print(f"  Policy:  {args.policy}")
    print(f"  Device:  {device}")
    print(f"  C_fail:  {args.c_fail}  MAX_EP_LEN: {args.max_ep_len}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    episodes = load_episodes_from_npz_list(args.npz_paths)
    n_suc  = sum(1 for ep in episodes if ep["success"])
    n_fail = len(episodes) - n_suc
    print(f"  Dataset: {len(episodes)} episodes  ({n_suc} success, {n_fail} failure)")

    # ---- Build frozen-encoder value function ----
    print("\nBuilding value function model...")
    vf_model, meta = build_vf_model(
        pretrained_dir=args.policy,
        device=device,
        image_height=args.image_height,
        image_width=args.image_width,
        num_bins=args.num_bins,
    )
    print(f"  Feature dim: {meta['feat_dim']}  bins: {args.num_bins}")
    n_params = sum(p.numel() for p in vf_model.value_head.parameters())
    print(f"  ValueHead parameters: {n_params:,}")

    # ---- Pre-compute features (one pass through frozen encoder) ----
    print("\nPre-computing features (frozen encoder)...")
    all_feats, all_rets = precompute_features(
        episodes=episodes,
        vf_model=vf_model,
        device=device,
        batch_size=args.precompute_batch_size,
        progress=True,
    )
    print(f"  Features: {all_feats.shape}  Returns: {all_rets.shape}")
    print(f"  Return range: [{all_rets.min():.3f}, {all_rets.max():.3f}]")

    # Move everything to device for training
    all_feats = all_feats.to(device)
    all_rets  = all_rets.to(device)
    n_total   = all_feats.shape[0]

    # ---- Optimizer (only trains ValueHead) ----
    optimizer = optim.AdamW(
        vf_model.value_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    print(f"\nTraining ValueHead for {args.epochs} epochs "
          f"(batch_size={args.batch_size})...")
    best_loss = float("inf")
    rng = np.random.default_rng(args.seed)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        vf_model.value_head.train()
        idx = rng.permutation(n_total)
        epoch_losses = []

        for start in range(0, n_total, args.batch_size):
            end  = min(start + args.batch_size, n_total)
            bidx = torch.from_numpy(idx[start:end].copy()).long().to(device)
            feats_b = all_feats[bidx]
            rets_b  = all_rets[bidx]

            optimizer.zero_grad()
            loss = vf_model.value_head.loss(feats_b, rets_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf_model.value_head.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_vf_checkpoint(vf_model, args.out, meta)

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"loss: {avg_loss:.4f} | best: {best_loss:.4f} | "
                  f"lr: {lr_now:.2e} | {elapsed:.0f}s elapsed")

    # ---- Final evaluation: value distribution ----
    vf_model.value_head.eval()
    with torch.no_grad():
        val_preds = vf_model.value_head(all_feats).cpu().numpy()
    ret_np  = all_rets.cpu().numpy()
    suc_mask = np.array([ep["success"] for ep in episodes for _ in range(len(ep["action"]))])

    print(f"\n  Value estimates — SUCCESS: mean={val_preds[suc_mask].mean():.3f} "
          f"std={val_preds[suc_mask].std():.3f}")
    print(f"  Value estimates — FAILURE: mean={val_preds[~suc_mask].mean():.3f} "
          f"std={val_preds[~suc_mask].std():.3f}")
    print(f"  Best validation loss: {best_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Best checkpoint: {args.out}")


if __name__ == "__main__":
    main()
