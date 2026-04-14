#!/usr/bin/env python3
"""Train Forward Dynamics Model (FDM) and Inverse Dynamics Model (IDM) on Task A rollout transitions.

After training, computes cycle-consistency error on the validation set to determine
the filtering threshold (data-driven, percentile-based).

Usage:
    python scripts/scripts_pick_place_simulator/dynamics_train.py \
        --data_dir data/.../iter1_dynamics_data/ \
        --output_dir data/.../iter1_dynamics/ \
        --fdm_steps 2000 --idm_steps 2000 \
        --lr 1e-3 --batch_size 256 --weight_decay 1e-4 \
        --loss_weight_state 1.0 --loss_weight_visual 0.1 \
        --filter_percentile 80 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rev2fwd_il.models.dynamics_model import ForwardDynamicsModel, InverseDynamicsModel


# ── Obs layout ──────────────────────────────────────────────────────────────
# obs_latent (143-d):
#   [0:128]  = visual (table_latent 64 + wrist_latent 64)
#   [128:135] = ee_pose (7)
#   [135:142] = obj_pose (7)
#   [142:143] = gripper (1)
STATE_START = 128
STATE_END = 143
VISUAL_START = 0
VISUAL_END = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FDM + IDM for dynamics-based filtering.")
    parser.add_argument("--data_dir", type=str, required=True, help="Dir with train.npz & val.npz.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    # FDM
    parser.add_argument("--fdm_steps", type=int, default=2000)
    parser.add_argument("--loss_weight_state", type=float, default=1.0)
    parser.add_argument("--loss_weight_visual", type=float, default=0.1)
    # IDM
    parser.add_argument("--idm_steps", type=int, default=2000)
    # Shared training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--patience", type=int, default=500)
    # Filter threshold
    parser.add_argument("--filter_percentile", type=float, default=80)
    parser.add_argument("--w_state", type=float, default=1.0, help="Weight for state part in consistency error.")
    parser.add_argument("--w_visual", type=float, default=0.1, help="Weight for visual part in consistency error.")
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_data(data_dir: str) -> tuple[dict, dict]:
    """Load train and val transition data."""
    train_data = dict(np.load(Path(data_dir) / "train.npz"))
    val_data = dict(np.load(Path(data_dir) / "val.npz"))
    return train_data, val_data


def compute_norm_stats(obs: np.ndarray, action: np.ndarray) -> dict:
    """Compute z-score normalization statistics from training data."""
    return {
        "obs_mean": obs.mean(axis=0).tolist(),
        "obs_std": np.maximum(obs.std(axis=0), 1e-8).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.maximum(action.std(axis=0), 1e-8).tolist(),
    }


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def make_dataloader(obs: np.ndarray, action: np.ndarray, obs_next: np.ndarray,
                    batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(obs),
        torch.from_numpy(action),
        torch.from_numpy(obs_next),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)


def cosine_lr_lambda(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    """Cosine decay schedule from lr_start to lr_end."""
    if total_steps <= 0:
        return 1.0
    progress = min(step / total_steps, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (lr_end + (lr_start - lr_end) * cosine_decay) / lr_start


def train_fdm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[ForwardDynamicsModel, dict]:
    """Train the Forward Dynamics Model."""
    device = args.device
    obs_dim = obs_mean.shape[0]
    action_dim = action_mean.shape[0]

    model = ForwardDynamicsModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.fdm_steps, args.lr, args.lr_end),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    log = {"train_loss": [], "val_loss": [], "best_step": 0}

    train_iter = iter(train_loader)
    model.train()

    for step in range(1, args.fdm_steps + 1):
        try:
            obs_b, act_b, obs_next_b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            obs_b, act_b, obs_next_b = next(train_iter)

        obs_b, act_b, obs_next_b = obs_b.to(device), act_b.to(device), obs_next_b.to(device)

        # Normalize
        obs_norm = normalize(obs_b, obs_mean, obs_std)
        act_norm = normalize(act_b, action_mean, action_std)
        obs_next_norm = normalize(obs_next_b, obs_mean, obs_std)

        # Target: residual in normalized space
        target_delta = obs_next_norm - obs_norm

        # Forward
        pred_delta = model(obs_norm, act_norm)

        # Weighted loss
        loss_state = F.mse_loss(pred_delta[:, STATE_START:STATE_END],
                                target_delta[:, STATE_START:STATE_END])
        loss_visual = F.mse_loss(pred_delta[:, VISUAL_START:VISUAL_END],
                                 target_delta[:, VISUAL_START:VISUAL_END])
        loss = args.loss_weight_state * loss_state + args.loss_weight_visual * loss_visual

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        log["train_loss"].append({"step": step, "loss": loss.item()})

        # Eval
        if step % args.eval_freq == 0 or step == args.fdm_steps:
            val_loss = eval_fdm(model, val_loader, obs_mean, obs_std, action_mean, action_std, args)
            log["val_loss"].append({"step": step, "loss": val_loss})

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [FDM] step {step}/{args.fdm_steps} | train={loss.item():.6f} | "
                  f"val={val_loss:.6f} | lr={lr_now:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log["best_step"] = step
                patience_counter = 0
            else:
                patience_counter += args.eval_freq
                if patience_counter >= args.patience:
                    print(f"  [FDM] Early stopping at step {step} (patience={args.patience})")
                    break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [FDM] Best val loss: {best_val_loss:.6f} at step {log['best_step']}")
    return model, log


@torch.no_grad()
def eval_fdm(model, val_loader, obs_mean, obs_std, action_mean, action_std, args):
    device = args.device
    model.eval()
    total_loss, n = 0.0, 0
    for obs_b, act_b, obs_next_b in val_loader:
        obs_b, act_b, obs_next_b = obs_b.to(device), act_b.to(device), obs_next_b.to(device)
        obs_norm = normalize(obs_b, obs_mean, obs_std)
        act_norm = normalize(act_b, action_mean, action_std)
        obs_next_norm = normalize(obs_next_b, obs_mean, obs_std)
        target_delta = obs_next_norm - obs_norm
        pred_delta = model(obs_norm, act_norm)
        loss_state = F.mse_loss(pred_delta[:, STATE_START:STATE_END],
                                target_delta[:, STATE_START:STATE_END], reduction="sum")
        loss_visual = F.mse_loss(pred_delta[:, VISUAL_START:VISUAL_END],
                                 target_delta[:, VISUAL_START:VISUAL_END], reduction="sum")
        total_loss += (args.loss_weight_state * loss_state + args.loss_weight_visual * loss_visual).item()
        n += len(obs_b)
    model.train()
    return total_loss / max(n, 1)


def train_idm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[InverseDynamicsModel, dict]:
    """Train the Inverse Dynamics Model."""
    device = args.device
    obs_dim = obs_mean.shape[0]
    action_dim = action_mean.shape[0]

    model = InverseDynamicsModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.idm_steps, args.lr, args.lr_end),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    log = {"train_loss": [], "val_loss": [], "best_step": 0}

    train_iter = iter(train_loader)
    model.train()

    for step in range(1, args.idm_steps + 1):
        try:
            obs_b, act_b, obs_next_b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            obs_b, act_b, obs_next_b = next(train_iter)

        obs_b, act_b, obs_next_b = obs_b.to(device), act_b.to(device), obs_next_b.to(device)

        # Normalize obs (action stays in original space for target)
        obs_norm = normalize(obs_b, obs_mean, obs_std)
        obs_next_norm = normalize(obs_next_b, obs_mean, obs_std)
        act_norm = normalize(act_b, action_mean, action_std)

        # IDM predicts normalized action
        pred_act_norm = model(obs_norm, obs_next_norm)
        loss = F.mse_loss(pred_act_norm, act_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        log["train_loss"].append({"step": step, "loss": loss.item()})

        # Eval
        if step % args.eval_freq == 0 or step == args.idm_steps:
            val_loss = eval_idm(model, val_loader, obs_mean, obs_std, action_mean, action_std, args)
            log["val_loss"].append({"step": step, "loss": val_loss})

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [IDM] step {step}/{args.idm_steps} | train={loss.item():.6f} | "
                  f"val={val_loss:.6f} | lr={lr_now:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log["best_step"] = step
                patience_counter = 0
            else:
                patience_counter += args.eval_freq
                if patience_counter >= args.patience:
                    print(f"  [IDM] Early stopping at step {step} (patience={args.patience})")
                    break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  [IDM] Best val loss: {best_val_loss:.6f} at step {log['best_step']}")
    return model, log


@torch.no_grad()
def eval_idm(model, val_loader, obs_mean, obs_std, action_mean, action_std, args):
    device = args.device
    model.eval()
    total_loss, n = 0.0, 0
    for obs_b, act_b, obs_next_b in val_loader:
        obs_b, act_b, obs_next_b = obs_b.to(device), act_b.to(device), obs_next_b.to(device)
        obs_norm = normalize(obs_b, obs_mean, obs_std)
        obs_next_norm = normalize(obs_next_b, obs_mean, obs_std)
        act_norm = normalize(act_b, action_mean, action_std)
        pred_act_norm = model(obs_norm, obs_next_norm)
        total_loss += F.mse_loss(pred_act_norm, act_norm, reduction="sum").item()
        n += len(obs_b)
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def compute_cycle_consistency_threshold(
    fdm: ForwardDynamicsModel,
    idm: InverseDynamicsModel,
    val_loader: DataLoader,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    """Compute cycle-consistency errors on val set and derive threshold.

    Calibrate in the REVERSE direction to match the actual filtering usage:
    For each (obs[t], action[t], obs[t+1]):
      1. act_rev = IDM(obs[t+1], obs[t])            (normalized, reverse query)
      2. obs_prev_pred = FDM(obs[t+1], act_rev)     (normalized)
      3. error = w_state * ||state_diff||₂ + w_visual * ||visual_diff||₂
         where diff = obs_prev_pred - obs[t]
    """
    device = args.device
    fdm.eval()
    idm.eval()

    errors = []
    for obs_b, act_b, obs_next_b in val_loader:
        obs_b, obs_next_b = obs_b.to(device), obs_next_b.to(device)
        obs_norm = normalize(obs_b, obs_mean, obs_std)
        obs_next_norm = normalize(obs_next_b, obs_mean, obs_std)

        # Step 1: IDM reverse query — predict action from obs[t+1] back to obs[t]
        act_rev_norm = idm(obs_next_norm, obs_norm)

        # Step 2: FDM predicts previous obs from obs[t+1] using reverse action
        obs_prev_pred_norm = fdm.predict(obs_next_norm, act_rev_norm)

        # Step 3: Error between predicted and actual previous obs (normalized space)
        diff = obs_prev_pred_norm - obs_norm
        error_state = torch.norm(diff[:, STATE_START:STATE_END], dim=1)    # (B,)
        error_visual = torch.norm(diff[:, VISUAL_START:VISUAL_END], dim=1) # (B,)
        error = args.w_state * error_state + args.w_visual * error_visual   # (B,)
        errors.append(error.cpu().numpy())

    errors = np.concatenate(errors)
    threshold = float(np.percentile(errors, args.filter_percentile))

    result = {
        "threshold": threshold,
        "percentile": args.filter_percentile,
        "w_state": args.w_state,
        "w_visual": args.w_visual,
        "n_val_transitions": int(len(errors)),
        "error_mean": float(errors.mean()),
        "error_std": float(errors.std()),
        "error_p50": float(np.percentile(errors, 50)),
        "error_p80": float(np.percentile(errors, 80)),
        "error_p90": float(np.percentile(errors, 90)),
        "error_p95": float(np.percentile(errors, 95)),
        "error_min": float(errors.min()),
        "error_max": float(errors.max()),
    }
    return result


def main() -> None:
    args = parse_args()
    t0 = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # Load data
    print(f"Loading data from {args.data_dir}")
    train_data, val_data = load_data(args.data_dir)
    print(f"  Train: {len(train_data['obs'])} transitions")
    print(f"  Val:   {len(val_data['obs'])} transitions")

    # Normalization stats (from training set)
    norm_stats = compute_norm_stats(train_data["obs"], train_data["action"])
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("  Saved norm_stats.json")

    obs_mean = torch.tensor(norm_stats["obs_mean"], dtype=torch.float32, device=device)
    obs_std = torch.tensor(norm_stats["obs_std"], dtype=torch.float32, device=device)
    action_mean = torch.tensor(norm_stats["action_mean"], dtype=torch.float32, device=device)
    action_std = torch.tensor(norm_stats["action_std"], dtype=torch.float32, device=device)

    # Dataloaders
    train_loader = make_dataloader(train_data["obs"], train_data["action"], train_data["obs_next"],
                                   args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_data["obs"], val_data["action"], val_data["obs_next"],
                                 args.batch_size, shuffle=False)

    # ── Train FDM ──
    print("\n" + "=" * 60)
    print("Training Forward Dynamics Model (FDM)")
    print("=" * 60)
    fdm, fdm_log = train_fdm(train_loader, val_loader, obs_mean, obs_std, action_mean, action_std, args)
    torch.save(fdm.state_dict(), out_dir / "fdm_best.pt")
    print(f"  Saved fdm_best.pt")

    # ── Train IDM ──
    print("\n" + "=" * 60)
    print("Training Inverse Dynamics Model (IDM)")
    print("=" * 60)
    idm, idm_log = train_idm(train_loader, val_loader, obs_mean, obs_std, action_mean, action_std, args)
    torch.save(idm.state_dict(), out_dir / "idm_best.pt")
    print(f"  Saved idm_best.pt")

    # ── Compute threshold ──
    print("\n" + "=" * 60)
    print("Computing cycle-consistency threshold on val set")
    print("=" * 60)
    threshold_info = compute_cycle_consistency_threshold(
        fdm, idm, val_loader, obs_mean, obs_std, action_mean, action_std, args
    )
    with open(out_dir / "filter_threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"  Threshold ({args.filter_percentile}th percentile): {threshold_info['threshold']:.6f}")
    print(f"  Error stats: mean={threshold_info['error_mean']:.6f}, "
          f"std={threshold_info['error_std']:.6f}, "
          f"p50={threshold_info['error_p50']:.6f}, "
          f"p80={threshold_info['error_p80']:.6f}, "
          f"p90={threshold_info['error_p90']:.6f}")

    # ── Save train log ──
    train_log = {
        "fdm": fdm_log,
        "idm": idm_log,
        "threshold": threshold_info,
        "args": vars(args),
    }
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {args.output_dir}")
    print(f"  Files: fdm_best.pt, idm_best.pt, norm_stats.json, filter_threshold.json, train_log.json")


if __name__ == "__main__":
    main()
