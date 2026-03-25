"""Shared utilities for RECAP RL scripts.

Implements the core RECAP algorithm components:
  - Distributional ValueHead with cross-entropy loss
  - FrozenEncoderValueFunction (frozen DP encoder + value head)
  - Monte Carlo return computation
  - Advantage computation & binary indicator generation
  - Single-frame observation batching for VF inference
  - Checkpoint migration for state-dim expansion (15→16 for adv indicator)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Hyperparameters (confirmed for pick-and-place simulator)
# ============================================================
C_FAIL = 1200       # Failure penalty (= MAX_EP_LEN → perfect -0.5 split)
MAX_EP_LEN = 1200   # Maximum episode length for return normalization
PERCENTILE = 30     # Success rate ~50% → percentile-30 → ~70% positive frames
NUM_BINS = 32       # Distributional VF bins (success/failure each get 16 bins)


# ============================================================
# Distributional Value Head
# ============================================================

class ValueHead(nn.Module):
    """Distributional value function head.

    Outputs a probability distribution over NUM_BINS bins covering (-1, 0).
    The expected value is used as a scalar value estimate.

    Bins are uniform: bin_width = 1/NUM_BINS.
    With NUM_BINS=32 and C_FAIL=MAX_EP_LEN=1200:
      - Success returns ∈ (-0.5, 0)  → upper 16 bins
      - Failure returns ∈ (-1, -0.5) → lower 16 bins
    """

    def __init__(self, feat_dim: int, num_bins: int = NUM_BINS, dropout: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_bins),
        )
        # Bin centers uniformly spaced in (-1, 0)
        centers = torch.linspace(-1.0 + 0.5 / num_bins, -0.5 / num_bins, num_bins)
        self.register_buffer("bin_centers", centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, feat_dim] → expected value [B]"""
        logits = self.net(z)                           # [B, num_bins]
        probs = F.softmax(logits, dim=-1)              # [B, num_bins]
        return (probs * self.bin_centers).sum(-1)      # [B]

    def loss(self, z: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss: map continuous returns to nearest bin.

        z:       [B, feat_dim]
        returns: [B] float, normalized ∈ (-1, 0)
        """
        logits = self.net(z)                                                       # [B, num_bins]
        dists = (returns.unsqueeze(1) - self.bin_centers.unsqueeze(0)).abs()       # [B, num_bins]
        target_bins = dists.argmin(dim=1)                                           # [B] int64
        return F.cross_entropy(logits, target_bins)


# ============================================================
# Frozen-Encoder Value Function
# ============================================================

class FrozenEncoderValueFunction(nn.Module):
    """Value function using a frozen copy of the DiffusionPolicy encoder.

    The DiffusionPolicy's rgb_encoder (ResNet18 + pooling) is deepcopied and
    frozen.  Only the ValueHead MLP is trained.

    Feature vector per frame:
      visual_feats (all cameras concatenated) + robot state
      ≈ 1024 + 15  = 1039-d  (2 cameras, ResNet18, state_dim=15)
    """

    def __init__(
        self,
        rgb_encoder: nn.Module,
        normalize_fn,            # callable: batch_dict → normalized_batch_dict
        feat_dim: int,
        num_bins: int = NUM_BINS,
        has_wrist: bool = True,
    ):
        super().__init__()
        self.rgb_encoder = copy.deepcopy(rgb_encoder)
        self.rgb_encoder.requires_grad_(False)
        self.normalize_fn = normalize_fn
        self.value_head = ValueHead(feat_dim, num_bins=num_bins)
        self.has_wrist = has_wrist
        self.feat_dim = feat_dim

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode single-frame observations to feature vector.

        batch keys expected:
          observation.image:       [B, 1, C, H, W] float32 in [0, 1]
          observation.state:       [B, 1, state_dim] float32
          observation.wrist_image: [B, 1, C, H, W] (optional)

        Returns [B, feat_dim].
        """
        with torch.no_grad():
            # Extract image tensor and squeeze obs-step dim: [B, 1, C, H, W] → [B, C, H, W]
            img = batch["observation.image"]
            if img.ndim == 5:
                img = img[:, -1]
            cam_feats = [self.rgb_encoder(img)]  # [B, feat_per_cam]
            if self.has_wrist and "observation.wrist_image" in batch:
                wrist = batch["observation.wrist_image"]
                if wrist.ndim == 5:
                    wrist = wrist[:, -1]
                cam_feats.append(self.rgb_encoder(wrist))
            img_feats = torch.cat(cam_feats, dim=-1)  # [B, num_cams * feat_per_cam]
            state = batch.get("observation.state", None)
            if state is not None:
                if state.ndim == 3:
                    state = state[:, -1, :]            # [B, state_dim] (take last step)
                z = torch.cat([img_feats, state], dim=-1)
            else:
                z = img_feats
        return z

    def predict_value(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """batch → scalar value estimate [B]"""
        z = self.encode(batch)
        return self.value_head(z)

    def compute_loss(self, batch: Dict[str, torch.Tensor], returns: torch.Tensor) -> torch.Tensor:
        z = self.encode(batch)
        return self.value_head.loss(z, returns)


# ============================================================
# Return computation
# ============================================================

def compute_normalized_returns(
    trajectory: dict,
    c_fail: float = C_FAIL,
    max_len: float = MAX_EP_LEN,
) -> List[float]:
    """Compute per-frame normalized Monte Carlo returns from end-of-episode label.

    Reward structure:
      r_t = -1 for t < T (time penalty)
      r_T = 0           if success
      r_T = -c_fail     if failure

    Normalized by (max_len + c_fail) so all returns ∈ (-1, 0).
    With c_fail = max_len: cut is exactly at -0.5 (success ∈ (-0.5, 0), failure ∈ (-1, -0.5)).
    """
    T = len(trajectory["action"])
    norm = max_len + c_fail
    penalty = 0.0 if trajectory["success"] else float(-c_fail)
    return [((-(T - t)) + penalty) / norm for t in range(T)]


# ============================================================
# Observation batching helpers
# ============================================================

def build_single_frame_batch(
    episode: dict,
    t: int,
    device: str,
    has_wrist: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a LeRobot-style single-frame batch from a NPZ episode dict.

    The returned batch has shape [1, 1, ...] to match the LeRobot convention
    of [B, n_obs_steps, ...].  The normalizer and encoder will squeeze this
    appropriately.

    episode keys expected:
      images: (T, H, W, 3) uint8
      ee_pose: (T, 7)
      obj_pose: (T, 7)
      gripper (optional): (T,) or (T, 1)  -- if absent, derived from action[:, 7]
      action (optional): (T, 8)           -- fallback for gripper when gripper key missing
      wrist_images (optional): (T, H, W, 3) uint8
    Note: DAgger collect NPZs from AlternatingTester have no 'gripper' key; gripper
    is encoded as action[:, 7]. Both formats are supported.
    """
    img = episode["images"][t]   # (H, W, 3) uint8
    img_t = (
        torch.from_numpy(img.copy()).float()
        .permute(2, 0, 1)           # (3, H, W)
        .unsqueeze(0)               # (1, 3, H, W) — B=1
        .unsqueeze(1)               # (1, 1, 3, H, W) — n_obs_steps=1
        .to(device)
        / 255.0
    )

    ee = episode["ee_pose"][t]     # (7,)
    obj = episode["obj_pose"][t]   # (7,)
    if "gripper" in episode:
        g_arr = episode["gripper"]
        gripper = g_arr[t] if g_arr.ndim == 1 else g_arr[t, 0]
    else:
        # DAgger collect format: gripper encoded as action[:, 7]
        gripper = episode["action"][t, 7]
    state = np.concatenate([ee, obj, [float(gripper)]], axis=0).astype(np.float32)  # (15,)
    state_t = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 15)

    batch = {
        "observation.image": img_t,
        "observation.state": state_t,
    }
    if has_wrist and "wrist_images" in episode:
        w = episode["wrist_images"][t]
        wrist_t = (
            torch.from_numpy(w.copy()).float()
            .permute(2, 0, 1)
            .unsqueeze(0).unsqueeze(1)
            .to(device)
            / 255.0
        )
        batch["observation.wrist_image"] = wrist_t

    return batch


def precompute_features(
    episodes: List[dict],
    vf_model: FrozenEncoderValueFunction,
    device: str,
    batch_size: int = 64,
    progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute all frame features and returns for efficient VH training.

    Returns:
      all_feats:   [N_total_frames, feat_dim]  float32, on CPU
      all_returns: [N_total_frames]             float32, on CPU
    """
    all_feats = []
    all_rets = []

    # Collect all (episode_idx, frame_idx, return) entries
    frame_list = []
    for ep in episodes:
        rets = compute_normalized_returns(ep)
        for t, R_t in enumerate(rets):
            frame_list.append((ep, t, R_t))

    n_total = len(frame_list)
    if progress:
        print(f"  Pre-computing features for {n_total} frames (batch={batch_size})...")

    # Process in batches
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_frames = frame_list[start:end]

        imgs, wrists, states, rets_batch = [], [], [], []
        use_wrist = vf_model.has_wrist
        for ep, t, R_t in batch_frames:
            img = ep["images"][t]
            imgs.append(torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0)
            ee = ep["ee_pose"][t]
            obj = ep["obj_pose"][t]
            if "gripper" in ep:
                g_arr = ep["gripper"]
                g = g_arr[t] if g_arr.ndim == 1 else g_arr[t, 0]
            else:
                g = ep["action"][t, 7]
            states.append(torch.from_numpy(
                np.concatenate([ee, obj, [float(g)]], axis=0).astype(np.float32)))
            if use_wrist and "wrist_images" in ep:
                w = ep["wrist_images"][t]
                wrists.append(torch.from_numpy(w.copy()).float().permute(2, 0, 1) / 255.0)
            rets_batch.append(R_t)

        # Stack to batch: [B, 1, C, H, W] for images
        imgs_t = torch.stack(imgs).unsqueeze(1).to(device)       # [B, 1, 3, H, W]
        states_t = torch.stack(states).unsqueeze(1).to(device)   # [B, 1, 15]
        batch = {"observation.image": imgs_t, "observation.state": states_t}
        if use_wrist and wrists:
            batch["observation.wrist_image"] = torch.stack(wrists).unsqueeze(1).to(device)

        feats = vf_model.encode(batch).cpu()
        all_feats.append(feats)
        all_rets.extend(rets_batch)

        if progress and (start // batch_size) % 20 == 0:
            print(f"    {end}/{n_total} frames processed", end="\r")

    if progress:
        print()

    all_feats = torch.cat(all_feats, dim=0)              # [N_total, feat_dim]
    all_rets = torch.tensor(all_rets, dtype=torch.float32)  # [N_total]
    return all_feats, all_rets


# ============================================================
# Advantage computation
# ============================================================

def compute_advantages_and_indicators(
    episodes: List[dict],
    vf_model: FrozenEncoderValueFunction,
    device: str,
    percentile: float = PERCENTILE,
    c_fail: float = C_FAIL,
    max_len: float = MAX_EP_LEN,
    batch_size: int = 64,
) -> float:
    """Compute per-frame advantages and write binary indicators into episodes.

    Modifies episodes in-place by adding 'indicators' field (List[int]).
    Returns the threshold value used.
    """
    # Collect all (episode_idx, frame_idx, return) triples
    ep_frame_return = []
    for ep_idx, ep in enumerate(episodes):
        rets = compute_normalized_returns(ep, c_fail=c_fail, max_len=max_len)
        for t, R_t in enumerate(rets):
            ep_frame_return.append((ep_idx, t, R_t))

    n_total = len(ep_frame_return)
    print(f"  Computing advantages for {n_total} frames...")

    # Pre-compute values in batches
    all_values = []
    use_wrist = vf_model.has_wrist
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_entries = ep_frame_return[start:end]

        imgs, wrists, states = [], [], []
        for ep_idx, t, _ in batch_entries:
            ep = episodes[ep_idx]
            img = ep["images"][t]
            imgs.append(torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0)
            ee = ep["ee_pose"][t]
            obj = ep["obj_pose"][t]
            if "gripper" in ep:
                g_arr = ep["gripper"]
                g = g_arr[t] if g_arr.ndim == 1 else g_arr[t, 0]
            else:
                g = ep["action"][t, 7]
            states.append(torch.from_numpy(
                np.concatenate([ee, obj, [float(g)]]).astype(np.float32)))
            if use_wrist and "wrist_images" in episodes[ep_idx]:
                w = episodes[ep_idx]["wrist_images"][t]
                wrists.append(torch.from_numpy(w.copy()).float().permute(2, 0, 1) / 255.0)

        imgs_t = torch.stack(imgs).unsqueeze(1).to(device)
        states_t = torch.stack(states).unsqueeze(1).to(device)
        batch = {"observation.image": imgs_t, "observation.state": states_t}
        if use_wrist and wrists:
            batch["observation.wrist_image"] = torch.stack(wrists).unsqueeze(1).to(device)

        with torch.no_grad():
            vals = vf_model.predict_value(batch).cpu().numpy()
        all_values.extend(vals.tolist())

        if (start // batch_size) % 20 == 0:
            print(f"    {end}/{n_total} frames done", end="\r")
    print()

    # Compute advantages and threshold
    all_returns = np.array([r for _, _, r in ep_frame_return])
    all_values = np.array(all_values)
    all_advantages = all_returns - all_values

    threshold = float(np.percentile(all_advantages, percentile))

    # Write indicators into episodes
    for ep in episodes:
        ep["indicators"] = [0] * len(ep["action"])

    for (ep_idx, t, _), A_t in zip(ep_frame_return, all_advantages):
        episodes[ep_idx]["indicators"][t] = int(A_t > threshold)

    pos_ratio = float(np.mean(all_advantages > threshold))
    print(f"  Advantage threshold (p{percentile:.0f}): {threshold:.4f}"
          f"  |  Positive: {pos_ratio:.1%}  |  "
          f"Val range: [{all_values.min():.3f}, {all_values.max():.3f}]"
          f"  |  Adv range: [{all_advantages.min():.3f}, {all_advantages.max():.3f}]")

    return threshold


# ============================================================
# Checkpoint migration: state_dim 15 → 16
# ============================================================

def migrate_checkpoint_for_recap(
    pretrained_state_dict: Dict[str, torch.Tensor],
    new_policy: nn.Module,
    extra_state_dims: int = 1,
    n_obs_steps: int = 2,
) -> Dict[str, torch.Tensor]:
    """Expand cond_encoder input layers to accommodate extra state dimensions.

    When state_dim increases by `extra_state_dims`, global_cond_dim increases
    by `n_obs_steps * extra_state_dims`.  All Linear layers inside the UNet
    that take global_cond as (part of) their input are expanded with zero-init
    for the new columns.

    Strategy: compare old and new state_dict shapes; for any 2D weight tensor
    where the column count (in_features) has grown, pad with zeros.
    """
    new_sd = new_policy.state_dict()
    migrated = {}

    n_expanded = 0
    for key, new_tensor in new_sd.items():
        if key in pretrained_state_dict:
            old_tensor = pretrained_state_dict[key]
            if (
                old_tensor.ndim == 2
                and new_tensor.ndim == 2
                and new_tensor.shape[0] == old_tensor.shape[0]
                and new_tensor.shape[1] > old_tensor.shape[1]
            ):
                # Input dimension grew → pad with zeros (zero init preserves
                # the pretrained behaviour: the new indicator dims start neutral)
                delta = new_tensor.shape[1] - old_tensor.shape[1]
                pad = torch.zeros(
                    old_tensor.shape[0], delta,
                    dtype=old_tensor.dtype, device=old_tensor.device,
                )
                migrated[key] = torch.cat([old_tensor, pad], dim=1)
                n_expanded += 1
            else:
                migrated[key] = old_tensor
        else:
            # New key not in pretrained → keep random init
            migrated[key] = new_tensor

    print(f"  Checkpoint migration: {n_expanded} layers expanded for state_dim+{extra_state_dims}.")
    return migrated


# ============================================================
# Policy loading helper (reuses 6_test_alternating.py logic)
# ============================================================

def load_dp_policy(
    pretrained_dir: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    n_action_steps: Optional[int] = None,
):
    """Load a DiffusionPolicy using the existing load_diffusion_policy function.

    Returns (policy, preprocessor, postprocessor, n_inference_steps, n_action_steps).
    """
    import importlib.util

    _alt_spec = importlib.util.spec_from_file_location(
        "test_alternating",
        str(Path(__file__).resolve().parent.parent
            / "scripts_pick_place" / "6_test_alternating.py"),
    )
    _alt_mod = importlib.util.module_from_spec(_alt_spec)
    _alt_spec.loader.exec_module(_alt_mod)

    policy, preproc, postproc, n_steps, n_act = _alt_mod.load_diffusion_policy(
        pretrained_dir=pretrained_dir,
        device=device,
        image_height=image_height,
        image_width=image_width,
        n_action_steps=n_action_steps,
    )
    return policy, preproc, postproc, n_steps, n_act


def get_policy_meta(pretrained_dir: str) -> dict:
    """Return has_wrist, include_obj_pose, include_gripper from checkpoint config."""
    import importlib.util

    _alt_spec = importlib.util.spec_from_file_location(
        "test_alternating",
        str(Path(__file__).resolve().parent.parent
            / "scripts_pick_place" / "6_test_alternating.py"),
    )
    _alt_mod = importlib.util.module_from_spec(_alt_spec)
    _alt_spec.loader.exec_module(_alt_mod)
    return _alt_mod.load_policy_config(pretrained_dir)


def infer_vf_feat_dim(
    policy,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    state_dim: int = 15,
    has_wrist: bool = True,
) -> int:
    """Do a single forward pass to discover the feature dimension output by the encoder."""
    enc = policy.diffusion.rgb_encoder
    dummy_img = torch.zeros(1, 3, image_height, image_width, device=device)
    with torch.no_grad():
        feat = enc(dummy_img)  # [1, feat_per_cam]
    num_cameras = 2 if has_wrist else 1
    return feat.shape[-1] * num_cameras + state_dim


def build_vf_model(
    pretrained_dir: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    num_bins: int = NUM_BINS,
) -> Tuple[FrozenEncoderValueFunction, dict]:
    """Build a FrozenEncoderValueFunction from a pretrained DiffusionPolicy checkpoint.

    Returns (vf_model, meta_dict).
    """
    meta = get_policy_meta(pretrained_dir)
    has_wrist = meta.get("has_wrist", True)
    state_dim_base = 15  # ee_pose(7) + obj_pose(7) + gripper(1)

    policy, _, _, _, _ = load_dp_policy(pretrained_dir, device, image_height, image_width)
    policy.eval()

    feat_dim = infer_vf_feat_dim(
        policy, device, image_height, image_width, state_dim_base, has_wrist)

    vf_model = FrozenEncoderValueFunction(
        rgb_encoder=policy.diffusion.rgb_encoder,
        normalize_fn=lambda batch: batch,
        feat_dim=feat_dim,
        num_bins=num_bins,
        has_wrist=has_wrist,
    )
    vf_model = vf_model.to(device)

    return vf_model, {"feat_dim": feat_dim, "has_wrist": has_wrist, "state_dim": state_dim_base}


def save_vf_checkpoint(
    vf_model: FrozenEncoderValueFunction,
    path: str,
    meta: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "value_head_state_dict": vf_model.value_head.state_dict(),
        "meta": meta,
    }, path)
    print(f"  Saved VF checkpoint to: {path}")


def load_vf_checkpoint(
    pretrained_dir: str,
    ckpt_path: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
) -> FrozenEncoderValueFunction:
    """Load a previously saved VF checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    vf_model, _ = build_vf_model(
        pretrained_dir=pretrained_dir,
        device=device,
        image_height=image_height,
        image_width=image_width,
    )
    vf_model.value_head.load_state_dict(ckpt["value_head_state_dict"])
    vf_model = vf_model.to(device)
    print(f"  Loaded VF checkpoint from: {ckpt_path}  feat_dim={meta['feat_dim']}")
    return vf_model


# ============================================================
# NPZ helpers
# ============================================================

def load_episodes_from_npz_list(npz_paths: List[str]) -> List[dict]:
    """Load and merge episodes from one or more NPZ files."""
    episodes = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        eps = list(data["episodes"])
        episodes.extend(eps)
        print(f"  Loaded {len(eps)} episodes from {path}")
    print(f"  Total: {len(episodes)} episodes")
    return episodes


def save_episodes_npz(path: str, episodes: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), episodes=np.array(episodes, dtype=object))
    print(f"  Saved {len(episodes)} episodes to: {path}")
