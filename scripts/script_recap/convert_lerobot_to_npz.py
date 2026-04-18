#!/usr/bin/env python3
"""Convert LeRobot dataset format to standard NPZ episode format.

Reads a LeRobot dataset directory and outputs a merged NPZ file with
episodes list, each dict containing the standard fields used by the
rest of the pipeline (images, wrist_images, ee_pose, obj_pose, action, success).

Usage:
    python scripts/script_recap/convert_lerobot_to_npz.py \
        --lerobot_dir data/pick_place_isaac_lab_simulation/exp56/lerobot/task_A \
        --out data/pick_place_isaac_lab_simulation/exp56/task_A_demo.npz \
        --mark_success
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to NPZ episode format.",
    )
    parser.add_argument("--lerobot_dir", type=str, required=True,
                        help="Path to LeRobot dataset directory.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output NPZ file path.")
    parser.add_argument("--mark_success", action="store_true",
                        help="Mark all episodes as success=True.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    lerobot_dir = Path(args.lerobot_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load LeRobot dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info_path = lerobot_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: info.json not found at {info_path}")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    repo_id = info.get("repo_id", "local/dataset")
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    features = info["features"]
    has_wrist = "observation.wrist_image" in features

    state_shape = features.get("observation.state", {}).get("shape", [15])
    state_dim = state_shape[0] if isinstance(state_shape, list) else state_shape

    print(f"{'='*60}")
    print(f"Converting LeRobot → NPZ")
    print(f"  LeRobot dir:    {lerobot_dir}")
    print(f"  Output:         {out_path}")
    print(f"  Episodes:       {total_episodes}")
    print(f"  Total frames:   {total_frames}")
    print(f"  State dim:      {state_dim}")
    print(f"  Has wrist cam:  {has_wrist}")
    print(f"  Mark success:   {args.mark_success}")
    print(f"{'='*60}")

    # Load the dataset
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(lerobot_dir),
    )

    # ---- Fast episode boundary detection via parquet ----
    import pyarrow.parquet as pq

    episodes_data = []

    parquet_files = sorted((lerobot_dir / "data").rglob("*.parquet"))
    if parquet_files:
        print("\nReading episode boundaries from parquet (fast path)...")
        tables = [pq.read_table(str(f)) for f in parquet_files]
        import pyarrow as pa
        full_table = pa.concat_tables(tables)
        ep_col = full_table.column("episode_index").to_pylist()
        all_ep_indices = np.array(ep_col, dtype=np.int64)

        # Also read state and action from parquet (avoids video decode)
        state_col = full_table.column("observation.state")
        action_col = full_table.column("action")
        all_states = np.array([s.as_py() for s in state_col], dtype=np.float32)
        all_actions = np.array([a.as_py() for a in action_col], dtype=np.float32)
        print(f"  Loaded {len(all_ep_indices)} frames from parquet")
        use_parquet_data = True
    else:
        print("\nNo parquet files found, falling back to dataset scan...")
        all_ep_indices = []
        for i in range(len(dataset)):
            all_ep_indices.append(dataset[i]["episode_index"].item())
        all_ep_indices = np.array(all_ep_indices)
        use_parquet_data = False

    unique_eps = sorted(set(all_ep_indices.tolist()))
    print(f"  Found {len(unique_eps)} unique episodes")

    for ep_idx in unique_eps:
        frame_mask = all_ep_indices == ep_idx
        frame_indices = np.where(frame_mask)[0]
        T = len(frame_indices)

        if T == 0:
            continue

        # Collect frames for this episode
        images_list = []
        wrist_images_list = []

        for global_idx in frame_indices:
            sample = dataset[int(global_idx)]

            # Image: (C, H, W) float → (H, W, C) uint8
            img = sample["observation.image"]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            images_list.append(img)

            if has_wrist:
                wimg = sample["observation.wrist_image"]
                if isinstance(wimg, torch.Tensor):
                    wimg = wimg.numpy()
                if wimg.dtype == np.float32 or wimg.dtype == np.float64:
                    wimg = (wimg * 255).clip(0, 255).astype(np.uint8)
                if wimg.ndim == 3 and wimg.shape[0] in (1, 3):
                    wimg = np.transpose(wimg, (1, 2, 0))
                wrist_images_list.append(wimg)

        images = np.stack(images_list)       # (T, H, W, 3)

        if use_parquet_data:
            states = all_states[frame_indices]   # (T, state_dim)
            actions = all_actions[frame_indices]  # (T, action_dim)
        else:
            states_list = []
            actions_list = []
            for global_idx in frame_indices:
                sample = dataset[int(global_idx)]
                state = sample["observation.state"]
                if isinstance(state, torch.Tensor):
                    state = state.numpy()
                states_list.append(state.astype(np.float32))
                action = sample["action"]
                if isinstance(action, torch.Tensor):
                    action = action.numpy()
                actions_list.append(action.astype(np.float32))
            states = np.stack(states_list)
            actions = np.stack(actions_list)

        # Extract ee_pose and obj_pose from state
        # state = [ee_pose(7), obj_pose(7), gripper(1)] = 15 dims
        ee_pose = states[:, :7]              # (T, 7)
        if state_dim >= 14:
            obj_pose = states[:, 7:14]       # (T, 7)
        else:
            obj_pose = np.zeros((T, 7), dtype=np.float32)

        ep_dict = {
            "images": images,
            "ee_pose": ee_pose,
            "obj_pose": obj_pose,
            "action": actions,
            "success": True if args.mark_success else False,
        }

        if has_wrist and wrist_images_list:
            ep_dict["wrist_images"] = np.stack(wrist_images_list)

        episodes_data.append(ep_dict)

        if (len(episodes_data)) % 10 == 0:
            print(f"  Processed {len(episodes_data)}/{len(unique_eps)} episodes "
                  f"(current: {T} frames)")

    # Save
    np.savez_compressed(
        str(out_path),
        episodes=np.array(episodes_data, dtype=object),
    )

    total_frames_out = sum(len(ep["action"]) for ep in episodes_data)
    n_success = sum(1 for ep in episodes_data if ep["success"])
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Episodes:     {len(episodes_data)}")
    print(f"  Total frames: {total_frames_out}")
    print(f"  Success:      {n_success}")
    print(f"  Output:       {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
