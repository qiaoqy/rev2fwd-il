#!/usr/bin/env python3
"""
Script 5 — Evaluate a trained DiT Flow policy on Inovo data.

Supports two evaluation modes:
  (A) **Offline** — Replay recorded RoboKit episodes through the policy to
      compare predicted actions with ground-truth.  Generates per-episode
      comparison plots and aggregate metrics.
  (B) **Online** — Connect to a live Inovo arm via a generic robot interface
      and run the closed-loop inference loop.  (Placeholder: the concrete
      RoboKit connector must be supplied by the user.)

Usage (offline):
    python scripts/scripts_task_inovo/5_eval_policy.py \\
        --checkpoint runs/inovo_A/checkpoints/pretrained_model \\
        --data_dir data/inovo_data/0209_tower_boby \\
        --mode offline \\
        --num_episodes 5 \\
        --out runs/inovo_A/eval_offline

Usage (online, requires robot access):
    python scripts/scripts_task_inovo/5_eval_policy.py \\
        --checkpoint runs/inovo_A/checkpoints/pretrained_model \\
        --mode online \\
        --num_episodes 10 \\
        --max_steps 800 \\
        --control_freq 30 \\
        --out runs/inovo_A/eval_online

Reference: scripts/scripts_piper_local/8_eval_ditflow_piper.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

# Default image shape expected by the policy (C, H, W)
DEFAULT_IMAGE_SHAPE = (3, 240, 320)

# Inovo workspace limits (metres).  These are conservative estimates — adjust
# to match the actual hardware setup.
WORKSPACE_LIMITS = {
    "x_min": -1.0,
    "x_max": 1.0,
    "y_min": -1.0,
    "y_max": 1.0,
    "z_min": 0.0,
    "z_max": 1.0,
}


# ============================================================================
# RoboKit NPZ helpers (same as other scripts)
# ============================================================================


def _load_pickled(raw_bytes: bytes) -> Any:
    """Unpickle bytes stored in a RoboKit NPZ."""
    return pickle.loads(raw_bytes)


def _decode_jpeg(raw_bytes: bytes) -> np.ndarray:
    """Decode a JPEG-encoded image from RoboKit NPZ → RGB uint8 HWC array."""
    if CV2_AVAILABLE:
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image

        return np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))


def load_robokit_frame(path: str | Path) -> Dict[str, Any]:
    """Load a single RoboKit per-frame NPZ and decode all fields."""
    with np.load(str(path), allow_pickle=True) as npz:
        frame: Dict[str, Any] = {}
        for key in npz.files:
            raw = npz[key]
            val = raw.item() if raw.ndim == 0 else raw
            if isinstance(val, bytes):
                if key in ("primary_rgb", "gripper_rgb"):
                    frame[key] = _decode_jpeg(val)
                else:
                    try:
                        frame[key] = _load_pickled(val)
                    except Exception:
                        frame[key] = val
            else:
                frame[key] = val
        return frame


def discover_episodes(data_dir: str | Path) -> List[Path]:
    """Return sorted list of episode directories inside *data_dir*."""
    data_dir = Path(data_dir)
    episodes = sorted(
        [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and d.name not in ("extracted", "__pycache__")
        ]
    )
    return episodes


def load_episode_frames(
    episode_dir: Path,
    max_frames: int | None = None,
) -> List[Dict[str, Any]]:
    """Load all (or up to *max_frames*) decoded frames from an episode."""
    frame_files = sorted(episode_dir.glob("*.npz"))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    return [load_robokit_frame(f) for f in frame_files]


# ============================================================================
# DiT Flow Policy Loading (adapted from piper eval)
# ============================================================================


def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """Parse *config.json* from a checkpoint directory.

    Returns a dict with keys: ``has_wrist``, ``image_shape``, ``state_dim``,
    ``action_dim``, ``n_obs_steps``, ``n_action_steps``, ``policy_type``,
    ``raw_config``.
    """
    config_path = Path(pretrained_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})

    has_wrist = "observation.wrist_image" in input_features

    image_shape = None
    if "observation.image" in input_features:
        image_shape = tuple(input_features["observation.image"]["shape"])

    state_dim = None
    if "observation.state" in input_features:
        state_dim = input_features["observation.state"]["shape"][0]

    action_dim = None
    if "action" in output_features:
        action_dim = output_features["action"]["shape"][0]

    policy_type = config_dict.get("type", "unknown")

    return {
        "has_wrist": has_wrist,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_obs_steps": config_dict.get("n_obs_steps", 2),
        "n_action_steps": config_dict.get("n_action_steps", 8),
        "policy_type": policy_type,
        "raw_config": config_dict,
    }


def load_ditflow_policy(
    pretrained_dir: str,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> Tuple[Any, Any, Any, int, int]:
    """Load a DiT Flow policy from checkpoint.

    Returns
    -------
    policy, preprocessor, postprocessor, num_inference_steps, n_action_steps
    """
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.factory import make_pre_post_processors

    # Register ditflow policy type
    from lerobot_policy_ditflow import DiTFlowConfig, DiTFlowPolicy

    pretrained_path = Path(pretrained_dir)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"

    print(f"[Policy] Loading config from {config_path} …")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Remove non-constructor fields
    config_dict.pop("type", None)

    # Parse features
    for feat_key in ("input_features", "output_features"):
        raw = config_dict.get(feat_key, {})
        parsed = {}
        for k, v in raw.items():
            ft = FeatureType[v["type"]] if isinstance(v["type"], str) else v["type"]
            parsed[k] = PolicyFeature(type=ft, shape=tuple(v["shape"]))
        config_dict[feat_key] = parsed

    # Parse normalization_mapping
    if "normalization_mapping" in config_dict:
        nm = {}
        for k, v in config_dict["normalization_mapping"].items():
            nm[k] = NormalizationMode[v] if isinstance(v, str) else v
        config_dict["normalization_mapping"] = nm

    # Convert lists → tuples for field types that expect tuples
    for fname in ("crop_shape", "optimizer_betas"):
        if fname in config_dict and isinstance(config_dict[fname], list):
            config_dict[fname] = tuple(config_dict[fname])

    # Overrides
    if num_inference_steps is not None:
        config_dict["num_inference_steps"] = num_inference_steps
        print(f"[Policy] Override num_inference_steps = {num_inference_steps}")
    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        n_action_steps = min(n_action_steps, horizon)
        config_dict["n_action_steps"] = n_action_steps
        print(f"[Policy] Override n_action_steps = {n_action_steps}")

    # Filter unknown keys
    import dataclasses as _dc

    valid_fields = {f.name for f in _dc.fields(DiTFlowConfig)}
    unknown = set(config_dict.keys()) - valid_fields
    if unknown:
        print(f"[Policy] Ignoring unknown config keys: {unknown}")
        for k in unknown:
            del config_dict[k]

    cfg = DiTFlowConfig(**config_dict)
    policy = DiTFlowPolicy(cfg)

    # Load weights (strict=False for mlp aliasing)
    print(f"[Policy] Loading weights from {model_path} …")
    state_dict = load_file(model_path)
    result = policy.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        unexpected_missing = [
            k
            for k in result.missing_keys
            if not ((".mlp.0." in k or ".mlp.3." in k) and "decoder.layers" in k)
        ]
        if unexpected_missing:
            raise RuntimeError(
                f"Unexpected missing keys:\n  {unexpected_missing}"
            )
        print(
            f"[Policy] {len(result.missing_keys)} aliased mlp keys auto-populated (safe)."
        )
    if result.unexpected_keys:
        print(f"[Policy] Warning: unexpected keys: {result.unexpected_keys}")

    policy = policy.to(device)
    policy.eval()

    # Pre/post processors (normalisation)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    actual_inference_steps = cfg.num_inference_steps or 100
    actual_n_action_steps = cfg.n_action_steps

    print(
        f"[Policy] Loaded! inference_steps={actual_inference_steps}, "
        f"n_action_steps={actual_n_action_steps}"
    )
    return (
        policy,
        preprocessor,
        postprocessor,
        actual_inference_steps,
        actual_n_action_steps,
    )


# ============================================================================
# Observation building
# ============================================================================


def build_observation(
    primary_rgb: np.ndarray,
    gripper_rgb: Optional[np.ndarray],
    state_vec: np.ndarray,
    policy_config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Construct an observation dict consumable by the policy.

    *state_vec* should be float32 of length ``state_dim`` (6 or 7).
    Images should be RGB uint8 HWC.
    """
    image_shape = policy_config.get("image_shape", DEFAULT_IMAGE_SHAPE)
    target_h, target_w = image_shape[1], image_shape[2]

    obs: Dict[str, torch.Tensor] = {}

    # Primary camera
    if CV2_AVAILABLE:
        img = cv2.resize(primary_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        from PIL import Image as _PILImage

        img = np.array(
            _PILImage.fromarray(primary_rgb).resize(
                (target_w, target_h), _PILImage.BILINEAR
            )
        )
    obs["observation.image"] = (
        torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    )

    # Wrist / gripper camera
    if policy_config.get("has_wrist", False) and gripper_rgb is not None:
        if CV2_AVAILABLE:
            wimg = cv2.resize(gripper_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            wimg = np.array(
                _PILImage.fromarray(gripper_rgb).resize(
                    (target_w, target_h), _PILImage.BILINEAR
                )
            )
        obs["observation.wrist_image"] = (
            torch.from_numpy(wimg).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        )

    # State vector
    obs["observation.state"] = (
        torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)
    )
    return obs


# ============================================================================
# Offline evaluation
# ============================================================================


@dataclass
class OfflineEpisodeResult:
    episode_id: str
    num_frames: int
    action_mse: float  # mean-squared error over all timesteps (L2)
    action_mae: float  # mean absolute error
    per_dim_mae: np.ndarray  # per-action-dimension MAE
    pos_mae: float  # position dims only
    rot_mae: float  # rotation dims only
    gripper_accuracy: float  # fraction of correct binary gripper predictions


@dataclass
class OfflineResults:
    episodes: List[OfflineEpisodeResult] = field(default_factory=list)

    @property
    def mean_mse(self) -> float:
        return float(np.mean([e.action_mse for e in self.episodes]))

    @property
    def mean_mae(self) -> float:
        return float(np.mean([e.action_mae for e in self.episodes]))

    @property
    def mean_pos_mae(self) -> float:
        return float(np.mean([e.pos_mae for e in self.episodes]))

    @property
    def mean_rot_mae(self) -> float:
        return float(np.mean([e.rot_mae for e in self.episodes]))

    @property
    def mean_gripper_acc(self) -> float:
        return float(np.mean([e.gripper_accuracy for e in self.episodes]))

    def to_dict(self) -> Dict:
        return {
            "num_episodes": len(self.episodes),
            "mean_mse": self.mean_mse,
            "mean_mae": self.mean_mae,
            "mean_pos_mae": self.mean_pos_mae,
            "mean_rot_mae": self.mean_rot_mae,
            "mean_gripper_accuracy": self.mean_gripper_acc,
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "num_frames": e.num_frames,
                    "action_mse": e.action_mse,
                    "action_mae": e.action_mae,
                    "per_dim_mae": e.per_dim_mae.tolist(),
                    "pos_mae": e.pos_mae,
                    "rot_mae": e.rot_mae,
                    "gripper_accuracy": e.gripper_accuracy,
                }
                for e in self.episodes
            ],
        }

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Results] Saved to {path}")


def _compute_gt_action_delta_pose(
    frames: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ground-truth delta-pose actions and state vectors.

    Returns
    -------
    gt_actions : (T-1, 7)  delta_pose[0:6] + gripper[6]
    states     : (T, state_dim)
    """
    obs_list = []
    gripper_list = []
    for fr in frames:
        rob = np.asarray(fr["robot_obs"], dtype=np.float32)
        tcp_pose = rob[:6]
        gripper_width = rob[6]
        obs_list.append(tcp_pose)
        gripper_list.append(gripper_width)

    obs_arr = np.stack(obs_list)  # (T, 6)
    gripper_arr = np.array(gripper_list, dtype=np.float32)  # (T,)

    # Delta pose: obs[t+1] - obs[t]
    delta_pose = np.diff(obs_arr, axis=0)  # (T-1, 6)

    # Gripper action from raw actions field
    gripper_actions = []
    for fr in frames[:-1]:
        act = np.asarray(fr["actions"], dtype=np.float32)
        gripper_actions.append(act[6])
    gripper_actions = np.array(gripper_actions, dtype=np.float32).reshape(-1, 1)

    gt_actions = np.concatenate([delta_pose, gripper_actions], axis=1)  # (T-1, 7)

    # State vectors: tcp_pose + gripper_width
    states = np.concatenate(
        [obs_arr, gripper_arr.reshape(-1, 1)], axis=1
    )  # (T, 7)

    return gt_actions, states


def _compute_gt_action_velocity(
    frames: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Use raw velocity actions from the data as ground truth.

    Returns
    -------
    gt_actions : (T-1, 7)  velocity[0:6] + gripper[6]
    states     : (T, state_dim)
    """
    obs_list, gripper_list, action_list = [], [], []
    for i, fr in enumerate(frames):
        rob = np.asarray(fr["robot_obs"], dtype=np.float32)
        obs_list.append(rob[:6])
        gripper_list.append(rob[6])
        if i < len(frames) - 1:
            act = np.asarray(fr["actions"], dtype=np.float32)
            action_list.append(act[:7])

    obs_arr = np.stack(obs_list)
    gripper_arr = np.array(gripper_list, dtype=np.float32)
    gt_actions = np.stack(action_list)

    states = np.concatenate(
        [obs_arr, gripper_arr.reshape(-1, 1)], axis=1
    )

    return gt_actions, states


def run_offline_episode(
    episode_dir: Path,
    policy,
    preprocessor,
    postprocessor,
    policy_config: Dict[str, Any],
    device: str,
    action_format: str = "delta_pose",
    out_dir: Optional[Path] = None,
) -> OfflineEpisodeResult:
    """Run open-loop policy inference on one recorded episode.

    For each timestep *t*, the policy sees the *recorded* observation (images +
    state) and predicts the next action.  The first action of the chunk is
    compared against the ground-truth action at *t*.
    """
    episode_id = episode_dir.name
    print(f"\n[Offline] Episode: {episode_id}")

    frames = load_episode_frames(episode_dir)
    T = len(frames)
    if T < 3:
        print(f"  Skipping (only {T} frames)")
        return OfflineEpisodeResult(
            episode_id=episode_id,
            num_frames=T,
            action_mse=float("nan"),
            action_mae=float("nan"),
            per_dim_mae=np.zeros(7),
            pos_mae=float("nan"),
            rot_mae=float("nan"),
            gripper_accuracy=float("nan"),
        )

    # Ground-truth
    if action_format == "velocity":
        gt_actions, states = _compute_gt_action_velocity(frames)
    else:
        gt_actions, states = _compute_gt_action_delta_pose(frames)

    state_dim = policy_config.get("state_dim", 7)
    if state_dim == 6:
        states = states[:, :6]

    # Run policy on each frame and collect first-action predictions
    pred_actions: List[np.ndarray] = []
    policy.reset()

    n_eval = min(T - 1, gt_actions.shape[0])
    log_freq = max(1, n_eval // 10)

    for t in range(n_eval):
        fr = frames[t]
        primary_rgb = fr["primary_rgb"]
        gripper_rgb = fr.get("gripper_rgb")
        state_vec = states[t]

        obs = build_observation(
            primary_rgb=primary_rgb,
            gripper_rgb=gripper_rgb,
            state_vec=state_vec,
            policy_config=policy_config,
        )
        obs_norm = preprocessor(obs)

        with torch.no_grad():
            action_tensor = policy.select_action(obs_norm)

        action_tensor = postprocessor(action_tensor)

        if isinstance(action_tensor, torch.Tensor):
            action_np = action_tensor.cpu().numpy()
        else:
            action_np = np.asarray(action_tensor)
        if action_np.ndim == 2:
            action_np = action_np[0]

        pred_actions.append(action_np.copy())

        if t % log_freq == 0:
            print(
                f"  [{t}/{n_eval}] pred={action_np[:3].round(4)}, "
                f"gt={gt_actions[t, :3].round(4)}"
            )

    pred_arr = np.stack(pred_actions)  # (n_eval, action_dim)
    gt_arr = gt_actions[:n_eval]

    # Align action dims
    min_dim = min(pred_arr.shape[1], gt_arr.shape[1])
    pred_arr = pred_arr[:, :min_dim]
    gt_arr = gt_arr[:, :min_dim]

    # Metrics
    diff = pred_arr - gt_arr
    action_mse = float(np.mean(diff ** 2))
    action_mae = float(np.mean(np.abs(diff)))
    per_dim_mae = np.mean(np.abs(diff), axis=0)

    pos_mae = float(np.mean(np.abs(diff[:, :3])))
    rot_mae = float(np.mean(np.abs(diff[:, 3:6]))) if min_dim > 3 else float("nan")

    # Gripper accuracy (binary: >0.5 = open)
    if min_dim > 6:
        pred_binary = (pred_arr[:, 6] > 0.5).astype(float)
        gt_binary = (gt_arr[:, 6] > 0.5).astype(float)
        gripper_accuracy = float(np.mean(pred_binary == gt_binary))
    else:
        gripper_accuracy = float("nan")

    print(
        f"  MSE={action_mse:.6f}  MAE={action_mae:.6f}  "
        f"pos_MAE={pos_mae:.6f}  rot_MAE={rot_mae:.6f}  "
        f"grip_acc={gripper_accuracy:.3f}"
    )

    # Save per-episode visualisation
    if out_dir is not None:
        _save_offline_plots(
            episode_id=episode_id,
            gt_actions=gt_arr,
            pred_actions=pred_arr,
            states=states[:n_eval],
            out_dir=out_dir,
        )

    return OfflineEpisodeResult(
        episode_id=episode_id,
        num_frames=T,
        action_mse=action_mse,
        action_mae=action_mae,
        per_dim_mae=per_dim_mae,
        pos_mae=pos_mae,
        rot_mae=rot_mae,
        gripper_accuracy=gripper_accuracy,
    )


def _save_offline_plots(
    episode_id: str,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    states: np.ndarray,
    out_dir: Path,
):
    """Generate comparison plots for one offline episode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = gt_actions.shape[0]
    t_axis = np.arange(T)

    # --- Action comparison plot (3 rows: position, rotation, gripper) ---
    dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    n_dims = min(gt_actions.shape[1], len(dim_labels))

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for d in range(n_dims):
        ax = axes[d]
        ax.plot(t_axis, gt_actions[:, d], "b-", alpha=0.7, label="Ground truth")
        ax.plot(t_axis, pred_actions[:, d], "r--", alpha=0.7, label="Predicted")
        ax.set_ylabel(dim_labels[d])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Episode {episode_id} — Action Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{episode_id}_action_comparison.png", dpi=120)
    plt.close(fig)

    # --- Trajectory comparison (XYZ) ---
    if states.shape[1] >= 3:
        fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        xyz_labels = ["X (m)", "Y (m)", "Z (m)"]

        # Integrate predicted actions from the first state
        pred_traj = np.zeros((T + 1, 3))
        pred_traj[0] = states[0, :3]
        for t in range(T):
            pred_traj[t + 1] = pred_traj[t] + pred_actions[t, :3]

        for d in range(3):
            ax = axes2[d]
            ax.plot(np.arange(T), states[:T, d], "b-", label="Actual")
            ax.plot(np.arange(T + 1), pred_traj[:, d], "r--", label="Pred (integrated)")
            ax.set_ylabel(xyz_labels[d])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        axes2[-1].set_xlabel("Timestep")
        fig2.suptitle(f"Episode {episode_id} — Trajectory Comparison (XYZ)", fontsize=14)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{episode_id}_trajectory.png", dpi=120)
        plt.close(fig2)

    print(f"  Plots saved to {out_dir}")


def run_offline_evaluation(args):
    """Main entry for offline evaluation mode."""
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy_config = load_policy_config(args.checkpoint)
    policy, preprocessor, postprocessor, _, _ = load_ditflow_policy(
        pretrained_dir=args.checkpoint,
        device=device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )

    # Discover episodes
    episodes = discover_episodes(args.data_dir)
    if not episodes:
        print(f"[Error] No episodes found in {args.data_dir}")
        sys.exit(1)

    # Select subset
    if args.num_episodes is not None and args.num_episodes < len(episodes):
        if args.shuffle:
            import random
            random.seed(args.seed)
            random.shuffle(episodes)
        episodes = episodes[: args.num_episodes]

    print(f"\n[Offline] Evaluating {len(episodes)} episodes from {args.data_dir}")
    print(f"  Action format: {args.action_format}")
    print(f"  Policy type  : {policy_config['policy_type']}")
    print(f"  State dim    : {policy_config['state_dim']}")
    print(f"  Action dim   : {policy_config['action_dim']}")
    print(f"  Has wrist    : {policy_config['has_wrist']}")
    print(f"  Image shape  : {policy_config['image_shape']}")
    print(f"  Output dir   : {out_dir}")

    results = OfflineResults()

    for ep_dir in episodes:
        ep_result = run_offline_episode(
            episode_dir=ep_dir,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_config=policy_config,
            device=device,
            action_format=args.action_format,
            out_dir=out_dir / "plots",
        )
        results.episodes.append(ep_result)

    # Summary
    print(f"\n{'='*60}")
    print(f"[Offline] Evaluation Summary ({len(results.episodes)} episodes)")
    print(f"{'='*60}")
    print(f"  Mean MSE           : {results.mean_mse:.6f}")
    print(f"  Mean MAE           : {results.mean_mae:.6f}")
    print(f"  Mean pos MAE       : {results.mean_pos_mae:.6f}")
    print(f"  Mean rot MAE       : {results.mean_rot_mae:.6f}")
    print(f"  Mean gripper acc   : {results.mean_gripper_acc:.3f}")

    # Per-dim MAE
    if results.episodes:
        pdm = np.mean([e.per_dim_mae for e in results.episodes], axis=0)
        dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        for d in range(min(len(pdm), len(dim_labels))):
            print(f"    {dim_labels[d]:8s}: {pdm[d]:.6f}")

    results.save(out_dir / "eval_results.json")

    # Aggregate comparison plot (bar chart of per-episode MAE)
    if len(results.episodes) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(results.episodes) * 0.5), 5))
        ep_ids = [e.episode_id[:12] for e in results.episodes]
        maes = [e.action_mae for e in results.episodes]
        ax.bar(range(len(ep_ids)), maes, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(ep_ids)))
        ax.set_xticklabels(ep_ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Action MAE")
        ax.set_title("Per-Episode Action MAE")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "per_episode_mae.png", dpi=120)
        plt.close(fig)
        print(f"  Aggregate plot saved to {out_dir / 'per_episode_mae.png'}")

    print("\n[Offline] Done.")


# ============================================================================
# Online evaluation (real robot)
# ============================================================================


class InovoRobotInterface:
    """Abstract interface for Inovo arm control.

    Subclass and implement the abstract methods to connect to the actual
    hardware via RoboKit or another SDK.  The default implementation raises
    ``NotImplementedError`` so that users know what needs to be provided.
    """

    def connect(self):
        raise NotImplementedError("Implement connect() for your Inovo SDK")

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose as float32 array [x, y, z, rx, ry, rz]."""
        raise NotImplementedError

    def get_gripper_width(self) -> float:
        """Return normalised gripper width in [0, 1]."""
        raise NotImplementedError

    def get_primary_rgb(self) -> np.ndarray:
        """Return current primary camera RGB image (H, W, 3) uint8."""
        raise NotImplementedError

    def get_gripper_rgb(self) -> Optional[np.ndarray]:
        """Return current gripper camera RGB image, or None."""
        raise NotImplementedError

    def send_pose_target(self, target_pose: np.ndarray, gripper: float):
        """Send an absolute target pose and gripper command to the robot.

        The evaluation loop uses **integral-based control**: predicted deltas
        are accumulated onto the first-frame pose, and the resulting absolute
        target is passed here.

        Args:
            target_pose: float32 array [x, y, z, rx, ry, rz] (metres / radians)
                in the robot base frame.
            gripper: Normalised gripper target in [0, 1] (0 = close, 1 = open).
        """
        raise NotImplementedError

    def go_home(self):
        """Move the arm to a safe home position."""
        raise NotImplementedError

    def disconnect(self):
        pass


@dataclass
class OnlineEpisodeResult:
    episode_id: int
    steps: int
    duration: float
    notes: str = ""


@dataclass
class OnlineResults:
    episodes: List[OnlineEpisodeResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "num_episodes": len(self.episodes),
            "avg_steps": float(np.mean([e.steps for e in self.episodes])) if self.episodes else 0,
            "avg_duration": float(np.mean([e.duration for e in self.episodes])) if self.episodes else 0,
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "steps": e.steps,
                    "duration": e.duration,
                    "notes": e.notes,
                }
                for e in self.episodes
            ],
        }

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Results] Saved to {path}")


def run_online_episode(
    robot: InovoRobotInterface,
    policy,
    preprocessor,
    postprocessor,
    policy_config: Dict[str, Any],
    episode_id: int,
    max_steps: int,
    control_freq: int,
    device: str,
    action_format: str = "delta_pose",
    out_dir: Optional[Path] = None,
    record_video: bool = True,
) -> OnlineEpisodeResult:
    """Run one online evaluation episode on the real robot.

    Uses **integral-based control**: the first-frame TCP pose is captured,
    then predicted deltas are accumulated onto it.  The integrated target is
    sent to the robot each step so that small deltas still move the arm.
    """
    print(f"\n{'='*60}")
    print(f"[Online] Episode {episode_id} — max {max_steps} steps @ {control_freq} Hz")
    print(f"{'='*60}")

    policy.reset()
    control_period = 1.0 / control_freq
    step = 0
    start_time = time.time()
    integrated_pose: Optional[np.ndarray] = None

    # Trajectory recording
    actual_positions: List[np.ndarray] = []
    integrated_positions: List[np.ndarray] = []
    recorded_actions: List[np.ndarray] = []

    video_frames: List[np.ndarray] = []

    state_dim = policy_config.get("state_dim", 7)

    while step < max_steps:
        loop_start = time.time()

        # --- Get observations ---
        tcp_pose = robot.get_tcp_pose()  # (6,)
        gripper_w = robot.get_gripper_width()
        primary_rgb = robot.get_primary_rgb()
        gripper_rgb = robot.get_gripper_rgb()

        if tcp_pose is None or primary_rgb is None:
            time.sleep(control_period)
            continue

        # Initialise integral base on first valid frame
        if integrated_pose is None:
            integrated_pose = tcp_pose.copy()
            print(
                f"  [Integral] Base pose captured: "
                f"xyz=({tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f})"
            )

        actual_positions.append(tcp_pose[:3].copy())

        # Build state vector
        if state_dim == 7:
            state_vec = np.concatenate([tcp_pose, [gripper_w]]).astype(np.float32)
        else:
            state_vec = tcp_pose[:state_dim].astype(np.float32)

        # --- Policy inference ---
        obs = build_observation(
            primary_rgb=primary_rgb,
            gripper_rgb=gripper_rgb,
            state_vec=state_vec,
            policy_config=policy_config,
        )
        obs_norm = preprocessor(obs)

        with torch.no_grad():
            action_tensor = policy.select_action(obs_norm)
        action_tensor = postprocessor(action_tensor)

        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.asarray(action_tensor)
        if action.ndim == 2:
            action = action[0]

        recorded_actions.append(action.copy())

        # --- Gripper forcing ---
        gripper_val = float(action[6]) if len(action) > 6 else 1.0
        # When model predicts "close" (val < 0.9), clamp harder to ensure grip
        if gripper_val < 0.9:
            gripper_val = 0.3
        # Last 50 steps: force gripper open for reset
        if step >= max_steps - 50:
            gripper_val = 1.0

        # --- Integral accumulation ---
        integrated_pose[:6] += action[:6]

        # Clamp to workspace limits
        for i, key in enumerate(["x", "y", "z"]):
            lo = WORKSPACE_LIMITS[f"{key}_min"]
            hi = WORKSPACE_LIMITS[f"{key}_max"]
            integrated_pose[i] = np.clip(integrated_pose[i], lo, hi)

        integrated_positions.append(integrated_pose[:3].copy())

        # --- Send to robot ---
        robot.send_pose_target(integrated_pose[:6], gripper_val)

        # --- Video recording ---
        if record_video and primary_rgb is not None:
            video_frames.append(primary_rgb.copy())

        # --- Logging ---
        if step % max(1, control_freq) == 0:
            print(
                f"  Step {step}: action={action[:3].round(4)}, "
                f"actual=({tcp_pose[0]:.3f},{tcp_pose[1]:.3f},{tcp_pose[2]:.3f}), "
                f"integ=({integrated_pose[0]:.3f},{integrated_pose[1]:.3f},{integrated_pose[2]:.3f})"
            )

        step += 1

        elapsed = time.time() - loop_start
        sleep_time = control_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    duration = time.time() - start_time
    print(f"[Online] Episode {episode_id}: {step} steps in {duration:.1f}s ({step/duration:.1f} Hz)")

    # Save video and trajectory plot
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        if video_frames and record_video:
            video_path = out_dir / f"episode_{episode_id:03d}.mp4"
            writer = imageio.get_writer(str(video_path), fps=control_freq, codec="libx264")
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Video saved to {video_path}")

        if len(actual_positions) > 1:
            _save_trajectory_plot(
                actual=np.array(actual_positions),
                integrated=np.array(integrated_positions),
                episode_id=episode_id,
                out_path=out_dir / f"episode_{episode_id:03d}_trajectory.png",
            )

    return OnlineEpisodeResult(
        episode_id=episode_id,
        steps=step,
        duration=duration,
    )


def _save_trajectory_plot(
    actual: np.ndarray,
    integrated: np.ndarray,
    episode_id: int,
    out_path: Path,
):
    """Save XYZ trajectory comparison plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X (m)", "Y (m)", "Z (m)"]
    T = actual.shape[0]
    t_axis = np.arange(T)

    for d in range(3):
        axes[d].plot(t_axis, actual[:, d], "b-", alpha=0.7, label="Actual")
        axes[d].plot(
            t_axis[: integrated.shape[0]],
            integrated[:, d],
            "r--",
            alpha=0.7,
            label="Integrated",
        )
        axes[d].set_ylabel(labels[d])
        axes[d].legend(fontsize=8)
        axes[d].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Episode {episode_id} — Actual vs Integrated Trajectory", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Trajectory plot saved to {out_path}")


def run_online_evaluation(args):
    """Main entry for online (real-robot) evaluation mode."""
    # ---------------------------------------------------
    # Robot interface — replace with actual implementation
    # ---------------------------------------------------
    print(
        "\n[Online] NOTE: Online mode requires a concrete InovoRobotInterface "
        "implementation.\nSee the InovoRobotInterface class in this file and "
        "provide your RoboKit / SDK connector.\n"
    )

    # Example: subclass InovoRobotInterface and pass here
    # robot = MyInovoRobot(ip="192.168.1.100")
    robot = InovoRobotInterface()

    try:
        robot.connect()
    except NotImplementedError:
        print(
            "[Error] InovoRobotInterface.connect() not implemented.\n"
            "To use online mode, subclass InovoRobotInterface and implement\n"
            "all abstract methods (connect, get_tcp_pose, get_gripper_width,\n"
            "get_primary_rgb, get_gripper_rgb, send_action, go_home).\n"
            "\nFalling back to --mode offline if data_dir is provided."
        )
        if args.data_dir:
            print("[Fallback] Switching to offline mode.\n")
            run_offline_evaluation(args)
            return
        sys.exit(1)

    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy_config = load_policy_config(args.checkpoint)
    policy, preprocessor, postprocessor, _, _ = load_ditflow_policy(
        pretrained_dir=args.checkpoint,
        device=device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )

    results = OnlineResults()

    for ep_id in range(args.num_episodes):
        print(f"\n--- Online episode {ep_id + 1}/{args.num_episodes} ---")
        input("Press Enter to start episode (move robot to start position first) …")

        ep_result = run_online_episode(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            policy_config=policy_config,
            episode_id=ep_id,
            max_steps=args.max_steps,
            control_freq=args.control_freq,
            device=device,
            action_format=args.action_format,
            out_dir=out_dir / "episodes",
            record_video=not args.no_record_video,
        )
        results.episodes.append(ep_result)

        # Move home between episodes
        try:
            robot.go_home()
        except NotImplementedError:
            pass

    results.save(out_dir / "eval_results.json")
    robot.disconnect()
    print("\n[Online] Done.")


# ============================================================================
# Argument parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DiT Flow policy on Inovo data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline evaluation on recorded data
  python scripts/scripts_task_inovo/5_eval_policy.py \\
      --checkpoint runs/inovo_A/checkpoints/pretrained_model \\
      --data_dir data/inovo_data/0209_tower_boby \\
      --mode offline --num_episodes 5 \\
      --out runs/inovo_A/eval_offline

  # Online evaluation (requires InovoRobotInterface implementation)
  python scripts/scripts_task_inovo/5_eval_policy.py \\
      --checkpoint runs/inovo_A/checkpoints/pretrained_model \\
      --mode online --num_episodes 10 --max_steps 800 \\
      --out runs/inovo_A/eval_online
""",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained_model directory (must contain config.json + model.safetensors)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="Evaluation mode (default: offline)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to RoboKit episode data directory (required for offline mode)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (auto-generated if None)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate (None = all for offline)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
        help="Max steps per episode (online mode, default: 800)",
    )
    parser.add_argument(
        "--control_freq",
        type=int,
        default=30,
        help="Control frequency in Hz (online mode, default: 30)",
    )
    parser.add_argument(
        "--action_format",
        type=str,
        choices=["delta_pose", "velocity"],
        default="delta_pose",
        help="Action format to compare against (default: delta_pose)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override n_action_steps (None = use model config)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override flow integration steps (None = use model config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle episodes before selecting subset (offline mode)",
    )
    parser.add_argument(
        "--no_record_video",
        action="store_true",
        help="Disable video recording (online mode)",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    # Auto-generate output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"runs/inovo_eval_{args.mode}_{ts}"

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "offline":
        if args.data_dir is None:
            print("[Error] --data_dir is required for offline mode.")
            sys.exit(1)
        run_offline_evaluation(args)
    else:
        run_online_evaluation(args)


if __name__ == "__main__":
    main()
