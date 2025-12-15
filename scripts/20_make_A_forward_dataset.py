#!/usr/bin/env python3
"""Build forward BC dataset for policy A from reverse rollouts.

This script implements Step 2 of the reverse-to-forward imitation learning pipeline:
converting REVERSE rollouts from Expert B into FORWARD training data for Policy A.

=============================================================================
CORE IDEA: TIME REVERSAL FOR IMITATION LEARNING
=============================================================================
The key insight is that a reverse trajectory (goal -> table) when played backwards
becomes a forward trajectory (table -> goal). However, we cannot simply reverse
the actions because:
    1. The gripper open/close timing would be wrong
    2. IK-Abs actions are absolute poses, not deltas

Instead, we:
    1. Reverse the state sequences (obs, ee_pose, obj_pose) in time
    2. RECONSTRUCT action labels using heuristics based on the reversed states

=============================================================================
ACTION RECONSTRUCTION STRATEGY
=============================================================================
For the FORWARD task (table -> goal), the action at time t should command:
    - EE pose: The NEXT timestep's EE pose (where we want to go)
    - Gripper: Inferred from EE-object distance and object-goal proximity

Gripper Heuristic (for forward task):
    - CLOSE when: EE is close to object AND object is far from goal
      (robot should grasp the object to move it toward goal)
    - OPEN when: EE is far from object OR object is close to goal
      (robot is approaching or has finished placing)

This heuristic works because:
    - In the forward task, we pick up from table (close gripper when near object)
    - Then place at goal (open gripper when object reaches goal)

=============================================================================
INPUT/OUTPUT DATA FORMATS
=============================================================================
INPUT (from script 10_collect_B_reverse_rollouts.py):
    NPZ file with episodes, each containing:
    - obs_i:        (T, 36)  Observations (REVERSE order: goal -> table)
    - ee_pose_i:    (T, 7)   EE poses (REVERSE order)
    - obj_pose_i:   (T, 7)   Object poses (REVERSE order)
    - gripper_i:    (T,)     Gripper actions (REVERSE order)
    - place_pose_i: (7,)     Place target (was Expert B's target)
    - goal_pose_i:  (7,)     Goal position (plate center)
    - success_i:    bool     Whether reverse task succeeded

OUTPUT (for BC training):
    NPZ file with flat arrays:
    - obs:    (N, 36)  Observations (FORWARD order: table -> goal)
    - act:    (N, 8)   Actions [ee_pose(7), gripper(1)] (reconstructed)
    - ep_id:  (N,)     Episode index for each sample

    Where N = sum of (T_i - 1) for all episodes (we lose one step per episode
    because action[t] = next_ee_pose[t+1], so the last step has no action).

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (process all episodes, including failures)
python scripts/20_make_A_forward_dataset.py \\
    --input data/B_reverse_100eps.npz \\
    --out data/A_forward_from_reverse.npz \\
    --success_only 0

# Only use successful episodes (stricter but cleaner data)
python scripts/20_make_A_forward_dataset.py \\
    --input data/B_reverse_100eps.npz \\
    --out data/A_forward_from_reverse.npz \\
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
        description="Build forward BC dataset from reverse rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Input/Output paths
    # -----------------------------------------------------------------
    parser.add_argument(
        "--input",
        type=str,
        default="data/B_reverse_100eps.npz",
        help="Input NPZ file containing reverse episodes from Expert B. "
             "This file is produced by script 10_collect_B_reverse_rollouts.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/A_forward_from_reverse.npz",
        help="Output NPZ file for the forward BC dataset. "
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
        help="If 1, only use episodes where the reverse task succeeded "
             "(cube ended up near target position). If 0, use all episodes. "
             "Using all episodes provides more data but may include noisy samples.",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for forward dataset generation.
    
    This function:
        1. Loads reverse episodes from the input NPZ file
        2. Optionally filters to success-only episodes
        3. For each episode:
           a. Reverses the trajectory in time
           b. Reconstructs action labels using gripper heuristics
        4. Concatenates all (obs, action) pairs into flat arrays
        5. Saves the dataset for BC training
    """
    args = _parse_args()

    # =====================================================================
    # Step 1: Import data utilities
    # =====================================================================
    from rev2fwd_il.data.io_npz import load_episodes
    from rev2fwd_il.data.reverse_time import reverse_episode_build_forward_pairs
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec

    # =====================================================================
    # Step 2: Load reverse episodes
    # =====================================================================
    print(f"\nLoading episodes from {args.input}")
    episodes = load_episodes(args.input)
    print(f"Loaded {len(episodes)} episodes from {args.input}")

    # =====================================================================
    # Step 3: Filter to successful episodes (optional)
    # =====================================================================
    # "Success" here means the REVERSE task succeeded (cube ended up at
    # Expert B's target position). For forward training, this means the
    # initial cube position (after reversal) is at a valid table location.
    if args.success_only:
        episodes = [ep for ep in episodes if ep.success]
        print(f"Filtered to {len(episodes)} successful episodes")

    if len(episodes) == 0:
        print("ERROR: No episodes to process!")
        print("Hint: Try --success_only 0 to use all episodes")
        return

    # =====================================================================
    # Step 4: Define task specification for reverse operation
    # =====================================================================
    # This must match the task spec used during collection.
    # The goal_xy is needed for the gripper heuristic (determining when
    # the object is "close to goal" and gripper should open).
    task_spec = PickPlaceTaskSpec(
        goal_xy=(0.5, 0.0),      # Plate center (forward task goal)
        hover_z=0.25,            # Hover height
        grasp_z_offset=0.0,      # Grasp offset
        success_radius=0.03,     # Success threshold (used for gripper heuristic)
        settle_steps=10,         # Not used here
    )

    # =====================================================================
    # Step 5: Process each episode
    # =====================================================================
    # We'll collect all (obs, action) pairs and their episode IDs
    all_obs = []      # List of observation arrays
    all_act = []      # List of action arrays
    all_ep_id = []    # List of episode ID arrays

    # Statistics tracking
    total_close_count = 0  # Total gripper CLOSE actions
    total_steps = 0        # Total number of steps

    print(f"\n{'='*60}")
    print(f"Processing {len(episodes)} episodes")
    print(f"{'='*60}\n")

    for ep_idx, ep in enumerate(episodes):
        # -------------------------------------------------------------
        # Reverse trajectory and build forward (obs, action) pairs
        # -------------------------------------------------------------
        # This function:
        # 1. Reverses obs, ee_pose, obj_pose sequences in time
        # 2. Builds action labels: act[t] = [ee_pose[t+1], gripper_heuristic[t]]
        # 3. Returns dict with "obs" and "act" arrays
        pairs = reverse_episode_build_forward_pairs(ep, task_spec)

        obs = pairs["obs"]  # (T-1, obs_dim) - forward order observations
        act = pairs["act"]  # (T-1, 8) - reconstructed actions [ee_pose, gripper]

        T = len(obs)

        # -------------------------------------------------------------
        # Compute gripper statistics
        # -------------------------------------------------------------
        # The gripper heuristic should produce a mix of OPEN (+1) and CLOSE (-1).
        # A healthy ratio is typically 30-50% CLOSE (grasping/transporting phase).
        # If ratio is 0% or 100%, something is wrong with the heuristic.
        gripper = act[:, 7]  # Last dimension is gripper
        close_count = np.sum(gripper < 0)  # Count CLOSE actions
        total_close_count += close_count
        total_steps += T

        # -------------------------------------------------------------
        # Append to collection with episode ID
        # -------------------------------------------------------------
        all_obs.append(obs)
        all_act.append(act)
        all_ep_id.append(np.full(T, ep_idx, dtype=np.int32))

        # Print progress every 20 episodes
        if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
            close_ratio = close_count / T if T > 0 else 0
            print(f"Episode {ep_idx + 1:4d} | Length: {T:4d} | CLOSE ratio: {100*close_ratio:.1f}%")

    # =====================================================================
    # Step 6: Concatenate all episodes into flat arrays
    # =====================================================================
    # This creates a single large dataset where samples from different
    # episodes are concatenated. The ep_id array tracks which episode
    # each sample came from (useful for analysis or episode-aware batching).
    obs_all = np.concatenate(all_obs, axis=0)      # (N, obs_dim)
    act_all = np.concatenate(all_act, axis=0)      # (N, 8)
    ep_id_all = np.concatenate(all_ep_id, axis=0)  # (N,)

    # =====================================================================
    # Step 7: Print dataset statistics
    # =====================================================================
    total_close_ratio = total_close_count / total_steps if total_steps > 0 else 0
    avg_length = total_steps / len(episodes) if len(episodes) > 0 else 0

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Total steps: {total_steps}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Observation dim: {obs_all.shape[1]}")
    print(f"Action dim: {act_all.shape[1]}")
    print(f"Gripper CLOSE ratio: {100*total_close_ratio:.1f}%")
    print(f"{'='*60}\n")

    # Sanity check for gripper ratio
    if total_close_ratio < 0.1 or total_close_ratio > 0.9:
        print("WARNING: Gripper CLOSE ratio is outside expected range (10-90%)!")
        print("         This may indicate a problem with the gripper heuristic.")

    # =====================================================================
    # Step 8: Save dataset as NPZ file
    # =====================================================================
    # We use np.savez_compressed for efficient storage.
    # The output format is simple flat arrays that can be directly used
    # with PyTorch Dataset/DataLoader for BC training.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        obs=obs_all,      # Observations: (N, 36)
        act=act_all,      # Actions: (N, 8) = [ee_pose(7), gripper(1)]
        ep_id=ep_id_all,  # Episode IDs: (N,)
    )
    
    print(f"Saved forward BC dataset to {out_path}")
    print(f"  obs shape: {obs_all.shape}")
    print(f"  act shape: {act_all.shape}")
    print(f"  ep_id shape: {ep_id_all.shape}")


if __name__ == "__main__":
    main()
