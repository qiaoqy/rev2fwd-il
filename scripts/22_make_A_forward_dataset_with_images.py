#!/usr/bin/env python3
"""Build forward BC dataset with images from reverse rollouts (from script 12).

This script extends 20_make_A_forward_dataset.py to handle image data.
It processes reverse rollouts from Expert B (collected with script 12) and
converts them into forward training data, preserving the image observations.

=============================================================================
CORE IDEA: TIME REVERSAL WITH IMAGE DATA
=============================================================================
Same as script 20, but also reverses the image sequence along with state data.
The reversed images represent what the forward policy A should "see" when
executing the table -> goal task.

=============================================================================
INPUT/OUTPUT DATA FORMATS
=============================================================================
INPUT (from script 12_collect_B_with_images.py):
    NPZ file with episodes list, each dict containing:
    - obs:        (T, 36)  State observations (REVERSE order: goal -> table)
    - images:     (T, H, W, 3)  RGB images (REVERSE order)
    - ee_pose:    (T, 7)   EE poses (REVERSE order)
    - obj_pose:   (T, 7)   Object poses (REVERSE order)
    - gripper:    (T,)     Gripper actions (REVERSE order)
    - place_pose: (7,)     Place target (was Expert B's target)
    - goal_pose:  (7,)     Goal position (plate center)
    - success:    bool     Whether reverse task succeeded

OUTPUT (Episode format, same as script 12):
    NPZ file with episodes list, each dict containing:
    - obs:        (T-1, 36)  State observations (FORWARD order: table -> goal)
    - images:     (T-1, H, W, 3)  RGB images (FORWARD order)
    - ee_pose:    (T-1, 7)   EE poses (FORWARD order)
    - obj_pose:   (T-1, 7)   Object poses (FORWARD order)
    - gripper:    (T-1,)     Gripper actions (FORWARD order)
    - place_pose: (7,)     Original place target (now becomes start position)
    - goal_pose:  (7,)     Goal position (plate center)
    - success:    bool     Whether reverse task succeeded

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
python scripts/22_make_A_forward_dataset_with_images.py \
    --input data/B_images_latest.npz \
    --out data/A_forward_with_images.npz

# Only use successful episodes
python scripts/22_make_A_forward_dataset_with_images.py \
    --input data/B_images_100eps.npz \
    --out data/A_forward_with_images.npz \
    --success_only 1

=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace containing all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Build forward BC dataset with images from reverse rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Input/Output paths
    # -----------------------------------------------------------------
    parser.add_argument(
        "--input",
        type=str,
        default="data/B_images_latest.npz",
        help="Input NPZ file containing reverse episodes with images from Expert B. "
             "This file is produced by script 12_collect_B_with_images.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/A_forward_with_images.npz",
        help="Output NPZ file for the forward BC dataset with images. "
             "This file can be used to train Policy A with behavior cloning.",
    )
    
    # -----------------------------------------------------------------
    # Data filtering options
    # -----------------------------------------------------------------
    parser.add_argument(
        "--success_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, only use episodes where the reverse task succeeded. "
             "If 0, use all episodes.",
    )

    args = parser.parse_args()
    return args


def load_episodes_with_images(path: str) -> list[dict]:
    """Load episodes with images from NPZ file.
    
    Args:
        path: Path to the NPZ file.
        
    Returns:
        List of episode dictionaries.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


def reverse_episode_with_images(ep: dict) -> dict:
    """Reverse an episode in time and return a new episode dict in forward order.
    
    Args:
        ep: Episode dictionary from script 12 (reverse order).
        
    Returns:
        New episode dictionary with all sequences reversed (forward order).
        Same format as input, compatible with script 13 for inspection.
    """
    # Time reversal of all sequences
    obs_fwd = ep["obs"][::-1].copy()          # (T, obs_dim)
    images_fwd = ep["images"][::-1].copy()    # (T, H, W, 3)
    ee_fwd = ep["ee_pose"][::-1].copy()       # (T, 7)
    obj_fwd = ep["obj_pose"][::-1].copy()     # (T, 7)
    gripper_fwd = ep["gripper"][::-1].copy()  # (T,)

    # Remove last timestep (no action target for it)
    # This keeps consistency: obs[t] corresponds to action target ee_pose[t+1]
    obs_out = obs_fwd[:-1]        # (T-1, obs_dim)
    images_out = images_fwd[:-1]  # (T-1, H, W, 3)
    ee_out = ee_fwd[:-1]          # (T-1, 7) - current ee pose at each step
    obj_out = obj_fwd[:-1]        # (T-1, 7)
    gripper_out = gripper_fwd[:-1]  # (T-1,)

    # Build new episode dict (same format as script 12 output)
    return {
        "obs": obs_out.astype(np.float32),
        "images": images_out,  # Keep as uint8
        "ee_pose": ee_out.astype(np.float32),
        "obj_pose": obj_out.astype(np.float32),
        "gripper": gripper_out.astype(np.float32),
        # place_pose in reverse task becomes start position in forward task
        "place_pose": ep["place_pose"].copy(),
        # goal_pose stays the same (plate center)
        "goal_pose": ep["goal_pose"].copy(),
        # success flag carried over
        "success": ep["success"],
    }


def save_episodes_with_images(path: str, episodes: list[dict]) -> None:
    """Save episodes with images to an NPZ file.
    
    Args:
        path: Output file path.
        episodes: List of episode dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(path, episodes=episodes)
    print(f"Saved {len(episodes)} episodes with images to {path}")
    
    if episodes:
        ep0 = episodes[0]
        print(f"  - State obs shape: {ep0['obs'].shape}")
        print(f"  - Image shape: {ep0['images'].shape}")
        print(f"  - Episode length: {ep0['obs'].shape[0]}")


def main() -> None:
    """Main entry point for forward dataset generation with images.
    
    This function:
        1. Loads reverse episodes with images from the input NPZ file
        2. Optionally filters to success-only episodes
        3. For each episode, reverses the trajectory in time (including images)
        4. Saves the forward episodes in Episode format (same as script 12)
    """
    args = _parse_args()

    # =====================================================================
    # Step 1: Load reverse episodes with images
    # =====================================================================
    print(f"\nLoading episodes from {args.input}")
    episodes = load_episodes_with_images(args.input)
    print(f"Loaded {len(episodes)} episodes")

    # Check that data has images
    if episodes and "images" not in episodes[0]:
        print("ERROR: Input data does not contain images!")
        print("       This script is designed for data from script 12.")
        print("       For data from script 10, use script 20 instead.")
        return

    # =====================================================================
    # Step 2: Filter to successful episodes (optional)
    # =====================================================================
    if args.success_only:
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful episodes")

    if len(episodes) == 0:
        print("ERROR: No episodes to process!")
        print("Hint: Try --success_only 0 to use all episodes")
        return

    # =====================================================================
    # Step 3: Process each episode - reverse in time
    # =====================================================================
    forward_episodes = []
    total_close_count = 0
    total_steps = 0

    print(f"\n{'='*60}")
    print(f"Processing {len(episodes)} episodes")
    print(f"{'='*60}\n")

    for ep_idx, ep in enumerate(episodes):
        # Reverse trajectory to get forward order
        fwd_ep = reverse_episode_with_images(ep)
        forward_episodes.append(fwd_ep)

        T = len(fwd_ep["obs"])
        gripper = fwd_ep["gripper"]
        close_count = np.sum(gripper < 0)
        total_close_count += close_count
        total_steps += T

        # Print progress every 20 episodes
        if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
            close_ratio = close_count / T if T > 0 else 0
            print(f"Episode {ep_idx + 1:4d} | Length: {T:4d} | CLOSE ratio: {100*close_ratio:.1f}%")

    # =====================================================================
    # Step 4: Print dataset statistics
    # =====================================================================
    total_close_ratio = total_close_count / total_steps if total_steps > 0 else 0
    avg_length = total_steps / len(forward_episodes) if forward_episodes else 0

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total episodes: {len(forward_episodes)}")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {avg_length:.1f}")
    if forward_episodes:
        print(f"Observation dim: {forward_episodes[0]['obs'].shape[1]}")
        print(f"Image shape: {forward_episodes[0]['images'].shape[1:]}")
    print(f"Gripper CLOSE ratio: {100*total_close_ratio:.1f}%")
    print(f"{'='*60}\n")

    # Sanity check for gripper ratio
    if total_close_ratio < 0.1 or total_close_ratio > 0.9:
        print("WARNING: Gripper CLOSE ratio is outside expected range (10-90%)!")
        print("         This may indicate a problem with the gripper heuristic.")

    # =====================================================================
    # Step 5: Save forward episodes (Episode format, same as script 12)
    # =====================================================================
    save_episodes_with_images(args.out, forward_episodes)
    
    # Calculate file size
    out_path = Path(args.out)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  file size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
