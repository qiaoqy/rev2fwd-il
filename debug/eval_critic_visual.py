#!/usr/bin/env python3
"""Evaluate trained critic model on test episodes and visualize predicted vs GT values.

For each episode, generates:
1. Value curve plot: predicted V(t) vs ground-truth V(t) overlayed
2. Overview image: value curves + sampled camera frames with predicted/GT annotations
3. Video: camera images + real-time predicted vs GT value curve

Usage:
    python debug/eval_critic_visual.py \
        --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
        --dataset debug/data/critic_A_test.npz \
        --num_episodes 5 \
        --out_dir debug/data/eval_critic_A_v2
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.configs.types import FeatureType, PolicyFeature
from rev2fwd_il.models.critic_config import CriticConfig
from rev2fwd_il.models.critic_model import CriticModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate critic model on test episodes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint.pt")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to critic_*_test.npz")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_video_frames", type=int, default=1500)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (sliding windows)")
    return parser.parse_args()


def load_critic_model(checkpoint_path: str, device: str) -> CriticModel:
    """Load trained critic model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract training config from checkpoint to match architecture
    ckpt_cfg = ckpt.get("config", {})
    dropout = ckpt_cfg.get("dropout", 0.0)
    crop_shape = tuple(ckpt_cfg.get("crop_shape", [128, 128]))
    include_obj_pose = not ckpt_cfg.get("no_obj_pose", False)
    state_dim = 7 + (7 if include_obj_pose else 0) + 1

    config = CriticConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        n_obs_steps=2,
        crop_shape=crop_shape,
        value_loss_type="mse",
        action_model_checkpoint=None,
        vision_backbone="resnet18",
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        use_separate_rgb_encoder_per_camera=False,
        mlp_hidden_dims=(512, 512, 256, 256),
        mlp_dropout=dropout,
    )

    model = CriticModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    model._critic_include_obj_pose = include_obj_pose
    model._critic_state_dim = state_dim
    print(
        f"Loaded critic model from {checkpoint_path} "
        f"(step={ckpt.get('step', '?')}, dropout={dropout}, crop={crop_shape}, "
        f"include_obj_pose={include_obj_pose}, state_dim={state_dim})"
    )
    return model


def build_state(ep, t, include_obj_pose: bool = True):
    """Build state from episode data.

    Supports both legacy 15D critic inputs (ee + obj + gripper) and the
    Exp45 8D variant (ee + gripper only).
    """
    parts = [ep["ee_pose"][t]]
    if include_obj_pose:
        if "obj_pose" in ep:
            obj_pose = ep["obj_pose"][t]
        elif "cube_small_pose" in ep:
            obj_pose = ep["cube_small_pose"][t]
        else:
            raise KeyError(
                "Episode is missing object pose data required by the critic "
                "(expected 'obj_pose' or 'cube_small_pose')."
            )
        parts.append(obj_pose)
    parts.append(np.array([ep["action"][t, 7]], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def predict_episode_values(
    model: CriticModel,
    ep: dict,
    horizon: int,
    n_obs_steps: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Run critic model on every frame of an episode, return predicted V(t).

    For each timestep t, constructs observation (state + images) and
    predicts a scalar value V(s_t).

    Args:
        horizon: Unused, kept for backward compatibility with caller signatures.

    Returns:
        pred_values: (T,) array of predicted values, one per frame.
    """
    T = len(ep["action"])
    pred_values = np.zeros(T, dtype=np.float32)
    include_obj_pose = getattr(model, "_critic_include_obj_pose", True)

    # Pre-build all samples
    all_obs_states = []
    all_obs_images = []

    for start in range(T):
        # Observation
        obs_states = []
        obs_imgs = []
        for i in range(n_obs_steps):
            obs_t = max(0, min(start - n_obs_steps + 1 + i, T - 1))
            obs_states.append(build_state(ep, obs_t, include_obj_pose=include_obj_pose))
            table = np.transpose(ep["images"][obs_t], (2, 0, 1)).astype(np.float32) / 255.0
            wrist = np.transpose(ep["wrist_images"][obs_t], (2, 0, 1)).astype(np.float32) / 255.0
            obs_imgs.append(np.stack([table, wrist], axis=0))

        all_obs_states.append(np.stack(obs_states, axis=0))
        all_obs_images.append(np.stack(obs_imgs, axis=0))

    # Batched inference
    all_obs_states = np.stack(all_obs_states)
    all_obs_images = np.stack(all_obs_images)   # (T, n_obs, 2, 3, H, W)

    with torch.no_grad():
        for i in range(0, T, batch_size):
            j = min(i + batch_size, T)
            batch = {
                "observation.state": torch.from_numpy(all_obs_states[i:j]).to(device),
                "observation.images": torch.from_numpy(all_obs_images[i:j]).to(device),
            }
            out = model(batch)  # (B, 1)
            pred_values[i:j] = out[:, 0].cpu().numpy()

    return pred_values


def plot_pred_vs_gt(ep: dict, pred_values: np.ndarray, ep_idx: int, out_path: str):
    """Plot predicted vs ground-truth Bellman value curves."""
    bv = ep["mc_value"]
    T = len(bv)
    success = ep.get("success", False)
    success_step = ep.get("success_step", None)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[2, 1, 1])
    fig.suptitle(
        f"Episode {ep_idx} — T={T}, success={success}"
        + (f", success_step={success_step}" if success_step is not None else ""),
        fontsize=14,
    )
    ts = np.arange(T)

    # --- Top: Predicted vs GT value curves ---
    ax = axes[0]
    ax.plot(ts, bv, "b-", linewidth=1.5, alpha=0.8, label="GT Bellman V(t)")
    ax.plot(ts, pred_values, "r-", linewidth=1.2, alpha=0.7, label="Predicted V(t)")
    ax.set_ylabel("Value")
    ax.set_xlabel("Timestep")
    ax.set_title("Predicted vs Ground-Truth Value")
    ax.set_xlim(0, T)
    ax.set_ylim(-1.1, 0.1)
    ax.grid(True, alpha=0.3)
    if success and success_step is not None:
        ss = min(success_step, T - 1)
        ax.axvline(x=ss, color="green", linestyle="--", alpha=0.7, label=f"success_step={ss}")
    ax.legend(loc="upper left", fontsize=10)

    # --- Middle: Prediction error ---
    ax2 = axes[1]
    error = pred_values - bv
    ax2.plot(ts, error, "m-", linewidth=0.8, alpha=0.7)
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.fill_between(ts, error, 0, alpha=0.2, color="magenta")
    ax2.set_ylabel("Error (pred - GT)")
    ax2.set_xlabel("Timestep")
    ax2.set_title(f"Prediction Error — MAE={np.abs(error).mean():.6f}, MSE={np.square(error).mean():.6f}")
    ax2.set_xlim(0, T)
    ax2.grid(True, alpha=0.3)
    if success and success_step is not None:
        ax2.axvline(x=ss, color="green", linestyle="--", alpha=0.5)

    # --- Bottom: EE pose XYZ ---
    ax3 = axes[2]
    ee = ep["ee_pose"]
    for dim, (color, label) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        ax3.plot(ts, ee[:, dim], color=color, alpha=0.7, linewidth=1, label=label)
    ax3.set_ylabel("EE Position")
    ax3.set_xlabel("Timestep")
    ax3.set_title("End-Effector XYZ Trajectory")
    ax3.set_xlim(0, T)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    if success and success_step is not None:
        ax3.axvline(x=ss, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Value curve saved: {out_path}")


def plot_overview_with_frames(ep: dict, pred_values: np.ndarray, ep_idx: int, out_path: str):
    """Overview image: value curves + sampled camera frames with pred/GT annotations."""
    bv = ep["mc_value"]
    T = len(bv)
    success = ep.get("success", False)
    success_step = ep.get("success_step", None)
    images = ep["images"]

    frame_indices = np.linspace(0, T - 1, 6, dtype=int)

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 6, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    # Top: value curves
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(np.arange(T), bv, "b-", linewidth=1.5, alpha=0.8, label="GT V(t)")
    ax_curve.plot(np.arange(T), pred_values, "r-", linewidth=1.2, alpha=0.7, label="Pred V(t)")
    ax_curve.set_ylabel("Value V(t)")
    ax_curve.set_xlabel("Timestep")
    ax_curve.set_xlim(0, T)
    ax_curve.set_ylim(-1.1, 0.1)
    ax_curve.grid(True, alpha=0.3)

    if success and success_step is not None:
        ss = min(success_step, T - 1)
        ax_curve.axvline(x=ss, color="green", linestyle="--", alpha=0.6)

    for fi in frame_indices:
        ax_curve.axvline(x=fi, color="orange", linestyle=":", alpha=0.5)
        ax_curve.plot(fi, pred_values[fi], "ro", markersize=5)
        ax_curve.plot(fi, bv[fi], "bo", markersize=5)

    mae = np.abs(pred_values - bv).mean()
    ax_curve.set_title(
        f"Episode {ep_idx} — T={T}, success={success}"
        + (f", ss={success_step}" if success_step is not None else "")
        + f" | MAE={mae:.5f}",
        fontsize=12,
    )
    ax_curve.legend(loc="upper left")

    # Bottom: sampled camera frames
    for col, fi in enumerate(frame_indices):
        ax_img = fig.add_subplot(gs[1, col])
        ax_img.imshow(images[fi])
        ax_img.set_title(
            f"t={fi}\nGT={bv[fi]:.3f}\nPred={pred_values[fi]:.3f}",
            fontsize=8,
        )
        ax_img.axis("off")

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Overview saved: {out_path}")


def create_episode_video(ep: dict, pred_values: np.ndarray, out_path: str,
                         ep_idx: int = 0, fps: int = 20, max_frames: int = 1500):
    """Video with camera images + real-time predicted vs GT value curve."""
    import imageio

    images = ep["images"]
    wrist = ep.get("wrist_images", None)
    bv = ep.get("mc_value", None)
    success_step = ep.get("success_step", None)
    T_full = len(images)
    T = min(T_full, max_frames)

    H, W = images.shape[1], images.shape[2]
    cam_w = W * 2 + 10 if wrist is not None else W
    plot_w = cam_w
    plot_h = max(int(H * 0.75), 96)
    total_h = H + plot_h
    total_h = total_h if total_h % 2 == 0 else total_h + 1
    total_w = cam_w if cam_w % 2 == 0 else cam_w + 1

    if T < T_full:
        print(f"    Video: using first {T}/{T_full} frames")

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
        canvas[:H, :W] = images[t]
        if wrist is not None:
            canvas[:H, W + 10: W + 10 + W] = wrist[t]

        if bv is not None:
            curve_img = _render_pred_vs_gt_frame(
                bv, pred_values, t, success_step, plot_w, plot_h
            )
            canvas[H: H + plot_h, :plot_w] = curve_img

        writer.append_data(canvas)

    writer.close()
    print(f"    Video saved: {out_path}  ({T} frames)")


def _render_pred_vs_gt_frame(
    bv: np.ndarray, pred: np.ndarray, t: int,
    success_step: int | None, plot_w: int, plot_h: int,
) -> np.ndarray:
    """Render a single frame of the pred vs GT value curve."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(plot_w / dpi, plot_h / dpi), dpi=dpi)
    T = len(bv)
    ts = np.arange(T)

    # Full trajectories (ghost)
    ax.plot(ts, bv, color="lightblue", linewidth=1)
    ax.plot(ts, pred, color="lightsalmon", linewidth=1)
    # Up to current t
    ax.plot(ts[:t+1], bv[:t+1], "b-", linewidth=1.5, label="GT")
    ax.plot(ts[:t+1], pred[:t+1], "r-", linewidth=1.2, label="Pred")
    # Current points
    ax.plot(t, bv[t], "bo", markersize=4)
    ax.plot(t, pred[t], "ro", markersize=4)

    ax.text(t, max(bv[t], pred[t]) + 0.03,
            f"GT={bv[t]:.3f} Pred={pred[t]:.3f}",
            fontsize=7, ha="center", color="black", clip_on=True)

    if success_step is not None and success_step < T:
        ax.axvline(x=success_step, color="green", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlim(0, T)
    ax.set_ylim(-1.1, 0.1)
    ax.set_xlabel("Timestep", fontsize=7)
    ax.set_ylabel("V(t)", fontsize=7)
    ax.set_title(f"Pred vs GT  t={t}/{T}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    if img.shape[0] != plot_h or img.shape[1] != plot_w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((plot_w, plot_h), Image.LANCZOS))

    return img


def plot_aggregate_pred_vs_gt(all_results: list[dict], out_path: str):
    """Plot all episodes' pred vs GT in one figure."""
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Critic Prediction vs Ground Truth (Test Set)", fontsize=14, y=1.002)

    for i, res in enumerate(all_results):
        ax = axes[i, 0]
        bv = res["gt"]
        pv = res["pred"]
        T = len(bv)
        ts = np.arange(T)
        success = res["success"]
        ss = res.get("success_step", None)
        mae = np.abs(pv - bv).mean()

        ax.plot(ts, bv, "b-", linewidth=1.5, alpha=0.8, label="GT")
        ax.plot(ts, pv, "r-", linewidth=1.0, alpha=0.7, label="Pred")
        ax.set_xlim(0, T)
        ax.set_ylim(-0.1, 1.15)
        ax.grid(True, alpha=0.3)

        tag = "success" if success else "failure"
        title = f"Ep {res['ep_idx']} ({tag}) — T={T}"
        if ss is not None:
            title += f", ss={ss}"
            ax.axvline(x=min(ss, T-1), color="green", linestyle="--", alpha=0.6)
        title += f" | MAE={mae:.5f}"
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregate plot saved: {out_path}")


def main():
    args = parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"debug/data/eval_critic_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Critic Model Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Load model
    model = load_critic_model(args.checkpoint, args.device)

    # Load test episodes
    print(f"\nLoading test episodes from {args.dataset}...")
    with np.load(args.dataset, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    n_success = sum(1 for ep in episodes if ep.get("success", False))
    n_failure = len(episodes) - n_success
    print(f"  {len(episodes)} episodes ({n_success} success, {n_failure} failure)")

    # Select episodes: mix of success + failure
    success_eps = [(i, ep) for i, ep in enumerate(episodes) if ep.get("success", False)]
    failure_eps = [(i, ep) for i, ep in enumerate(episodes) if not ep.get("success", False)]

    n_vis = min(args.num_episodes, len(episodes))
    n_fail_vis = min(max(1, n_vis // 3), len(failure_eps))
    n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))
    if n_fail_vis > len(failure_eps):
        n_fail_vis = len(failure_eps)
        n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))

    selected = success_eps[:n_succ_vis] + failure_eps[:n_fail_vis]

    # Run inference and visualization
    all_results = []
    for idx, (ep_i, ep) in enumerate(selected):
        success_tag = "success" if ep.get("success", False) else "failure"
        T = len(ep["action"])
        print(f"\n  [{idx+1}/{len(selected)}] Episode {ep_i} ({success_tag}, T={T})...")

        # Predict values for every frame
        pred_values = predict_episode_values(
            model, ep,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            device=args.device,
            batch_size=args.batch_size,
        )

        bv = ep["mc_value"]
        mae = np.abs(pred_values - bv).mean()
        mse = np.square(pred_values - bv).mean()
        print(f"    MAE={mae:.6f}, MSE={mse:.6f}")

        all_results.append({
            "ep_idx": ep_i,
            "gt": bv,
            "pred": pred_values,
            "success": ep.get("success", False),
            "success_step": ep.get("success_step", None),
        })

        ep_dir = out_dir / f"ep{ep_i}_{success_tag}"
        ep_dir.mkdir(exist_ok=True)

        # Static plots
        plot_pred_vs_gt(ep, pred_values, ep_i, str(ep_dir / "value_curve.png"))
        plot_overview_with_frames(ep, pred_values, ep_i, str(ep_dir / "overview.png"))

        # Video
        create_episode_video(
            ep, pred_values, str(ep_dir / "video.mp4"),
            ep_idx=ep_i, fps=args.fps, max_frames=args.max_video_frames,
        )

    # Aggregate plot
    print(f"\nGenerating aggregate plot...")
    plot_aggregate_pred_vs_gt(all_results, str(out_dir / "aggregate_pred_vs_gt.png"))

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  Summary")
    print(f"  {'='*70}")
    print(f"  {'Ep':>4} {'T':>6} {'Success':>8} {'MAE':>10} {'MSE':>10}")
    print(f"  {'----':>4} {'------':>6} {'--------':>8} {'----------':>10} {'----------':>10}")
    for r in all_results:
        mae = np.abs(r["pred"] - r["gt"]).mean()
        mse = np.square(r["pred"] - r["gt"]).mean()
        print(f"  {r['ep_idx']:>4} {len(r['gt']):>6} {str(r['success']):>8} {mae:>10.6f} {mse:>10.6f}")

    print(f"\n{'='*60}")
    print(f"Done! Results saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
