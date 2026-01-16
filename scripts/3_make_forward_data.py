#!/usr/bin/env python3
"""Step 3: Build forward training data by time-reversing reverse rollouts.

This script processes reverse rollouts from script 1 (Task B: goal -> table) and
converts them into forward training data (Task A: table -> goal) by reversing
the trajectories in time.

=============================================================================
CORE IDEA: TIME REVERSAL
=============================================================================
A reverse trajectory (goal -> table) when played backwards becomes a forward
trajectory (table -> goal). This script:
1. Reverses all sequences (obs, images, poses) in time
2. Recomputes goal actions appropriate for forward execution

=============================================================================
INPUT/OUTPUT DATA FORMATS
=============================================================================
INPUT (from script 1_collect_data.py):
    NPZ file with reverse trajectories (goal -> table)

OUTPUT:
    NPZ file with forward trajectories (table -> goal)
    Same format, but sequences are time-reversed.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
python scripts/3_make_forward_data.py \
    --input data/B_2images_goal.npz \
    --out data/A_2images_goal.npz

# Only use successful episodes
python scripts/3_make_forward_data.py \
    --input data/B_2images_goal.npz \
    --out data/A_2images_goal.npz \
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
        description="Build forward BC dataset with goal-based actions from reverse rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Input/Output paths
    # -----------------------------------------------------------------
    parser.add_argument(
        "--input",
        type=str,
        default="data/B_2images_goal.npz",
        help="Input NPZ file containing reverse episodes with goal actions from Expert B. "
             "This file is produced by script 14_collect_B_with_goal_actions.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/A_2images_goal.npz",
        help="Output NPZ file for the forward BC dataset with goal actions. "
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


def load_episodes_with_goal_actions(path: str) -> list[dict]:
    """Load episodes with goal actions from NPZ file.
    
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


def compute_forward_goal_actions(fsm_state_rev: np.ndarray, action_rev: np.ndarray) -> np.ndarray:
    """Compute goal actions for the forward trajectory.
    
    The key insight is that in the reversed trajectory:
    - We're now moving from the END of task B to the START of task B
    - At each timestep, the goal should be where we're heading NEXT
    - This is the goal of the PREVIOUS state in the reversed fsm_state
    
    Strategy:
    - Find state transition points in the reversed FSM state sequence
    - For each segment between transitions, the goal is the action value
      at the START of that segment (which was the end of the original segment)
    - This gives us "look-ahead" goals: at each step, we know where we're going
    
    Args:
        fsm_state_rev: Reversed FSM state sequence (T,)
        action_rev: Reversed action sequence (T, 8)
        
    Returns:
        New action sequence (T, 8) with forward goal actions
    """
    T = len(fsm_state_rev)
    new_actions = np.zeros_like(action_rev)
    
    # Find state transition points in the reversed sequence
    # A transition occurs when fsm_state changes from t to t+1
    transitions = np.where(np.diff(fsm_state_rev) != 0)[0]
    
    # Add boundaries: start and end of sequence
    boundaries = np.concatenate([[0], transitions + 1, [T]])
    
    # For each segment, the goal action is the action at the END of the segment
    # (because we want to reach where that segment ends)
    # In the reversed trajectory, this is actually the action from the NEXT segment
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # For all timesteps in this segment, the goal is where this segment ends
        # The action at the end of the segment tells us where we're going
        if end < T:
            # Use the action at the start of the next segment
            # (which is the goal of the next state)
            goal_action = action_rev[end].copy()
        else:
            # For the last segment, the goal is the final position
            # Use the last action value
            goal_action = action_rev[end - 1].copy()
        
        new_actions[start:end] = goal_action
    
    return new_actions


def reverse_episode_with_goal_actions(ep: dict) -> dict:
    """Reverse an episode in time with goal-based actions.
    
    This function:
    1. Reverses all sequences in time (observations, images, poses, etc.)
    2. Recomputes goal actions to be appropriate for the forward trajectory
    
    Args:
        ep: Episode dictionary from script 14 (reverse order).
        
    Returns:
        New episode dictionary with all sequences reversed (forward order)
        and goal actions recomputed for forward execution.
    """
    # Time reversal of all sequences
    obs_fwd = ep["obs"][::-1].copy()          # (T, obs_dim)
    images_fwd = ep["images"][::-1].copy()    # (T, H, W, 3)
    ee_fwd = ep["ee_pose"][::-1].copy()       # (T, 7)
    obj_fwd = ep["obj_pose"][::-1].copy()     # (T, 7)
    gripper_fwd = ep["gripper"][::-1].copy()  # (T,)
    fsm_state_fwd = ep["fsm_state"][::-1].copy()  # (T,) - reversed FSM states
    action_fwd = ep["action"][::-1].copy()    # (T, 8) - reversed actions (to be recomputed)
    
    # Check if wrist camera images exist
    has_wrist = "wrist_images" in ep
    if has_wrist:
        wrist_images_fwd = ep["wrist_images"][::-1].copy()  # (T, H, W, 3)

    # Recompute goal actions for forward trajectory
    # The original reversed actions point to where task B was going
    # We need to transform them to point to where task A should go
    new_actions = compute_forward_goal_actions(fsm_state_fwd, action_fwd)
    
    # Remove last timestep (no action target for it)
    # This keeps consistency: obs[t] corresponds to action target at step t
    obs_out = obs_fwd[:-1]           # (T-1, obs_dim)
    images_out = images_fwd[:-1]     # (T-1, H, W, 3)
    ee_out = ee_fwd[:-1]             # (T-1, 7)
    obj_out = obj_fwd[:-1]           # (T-1, 7)
    gripper_out = gripper_fwd[:-1]   # (T-1,)
    fsm_state_out = fsm_state_fwd[:-1]  # (T-1,)
    action_out = new_actions[:-1]    # (T-1, 8)
    
    if has_wrist:
        wrist_images_out = wrist_images_fwd[:-1]  # (T-1, H, W, 3)

    # Build new episode dict
    result = {
        "obs": obs_out.astype(np.float32),
        "images": images_out,  # Keep as uint8
        "ee_pose": ee_out.astype(np.float32),
        "obj_pose": obj_out.astype(np.float32),
        "action": action_out.astype(np.float32),
        "gripper": gripper_out.astype(np.float32),
        "fsm_state": fsm_state_out.astype(np.int32),
        # place_pose in reverse task becomes start position in forward task
        "place_pose": ep["place_pose"].copy(),
        # goal_pose stays the same (plate center)
        "goal_pose": ep["goal_pose"].copy(),
        # success flag carried over
        "success": ep["success"],
    }
    
    # Add wrist camera images if present
    if has_wrist:
        result["wrist_images"] = wrist_images_out  # Keep as uint8
    
    return result


def save_episodes_with_goal_actions(path: str, episodes: list[dict]) -> None:
    """Save episodes with goal actions to an NPZ file.
    
    Args:
        path: Output file path.
        episodes: List of episode dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(path, episodes=episodes)
    print(f"Saved {len(episodes)} episodes with goal actions to {path}")
    
    if episodes:
        ep0 = episodes[0]
        print(f"  - State obs shape: {ep0['obs'].shape}")
        print(f"  - Table camera image shape: {ep0['images'].shape}")
        if "wrist_images" in ep0:
            print(f"  - Wrist camera image shape: {ep0['wrist_images'].shape}")
        print(f"  - Action shape: {ep0['action'].shape}")
        print(f"  - Episode length: {ep0['obs'].shape[0]}")


def analyze_action_transitions(episodes: list[dict], num_episodes: int = 3) -> None:
    """Analyze and print action transitions for debugging.
    
    Args:
        episodes: List of forward episode dictionaries.
        num_episodes: Number of episodes to analyze.
    """
    print(f"\n{'='*60}")
    print("Action Transition Analysis (Forward Trajectory)")
    print(f"{'='*60}")
    
    for ep_idx, ep in enumerate(episodes[:num_episodes]):
        print(f"\nEpisode {ep_idx}:")
        fsm_state = ep["fsm_state"]
        action = ep["action"]
        
        # Find transitions
        transitions = np.where(np.diff(fsm_state) != 0)[0]
        
        print(f"  FSM states present: {np.unique(fsm_state)}")
        print(f"  Number of state transitions: {len(transitions)}")
        
        # Print a few transitions
        for t in transitions[:5]:
            print(f"  Step {t}: state {fsm_state[t]} -> {fsm_state[t+1]}")
            print(f"    Action XYZ before: {action[t, :3]}")
            print(f"    Action XYZ after:  {action[t+1, :3]}")
            print(f"    Action gripper before: {action[t, 7]:.1f}, after: {action[t+1, 7]:.1f}")


def main() -> None:
    """Main entry point for forward dataset generation with goal actions.
    
    This function:
        1. Loads reverse episodes with goal actions from the input NPZ file
        2. Optionally filters to success-only episodes
        3. For each episode, reverses the trajectory and recomputes goal actions
        4. Saves the forward episodes
    """
    args = _parse_args()

    # =====================================================================
    # Step 1: Load reverse episodes with goal actions
    # =====================================================================
    print(f"\nLoading episodes from {args.input}")
    episodes = load_episodes_with_goal_actions(args.input)
    print(f"Loaded {len(episodes)} episodes")

    # Check that data has required fields
    if episodes:
        required_fields = ["images", "action", "fsm_state"]
        missing = [f for f in required_fields if f not in episodes[0]]
        if missing:
            print(f"ERROR: Input data is missing required fields: {missing}")
            print("       This script is designed for data from script 14.")
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
    # Step 3: Process each episode - reverse in time with goal actions
    # =====================================================================
    forward_episodes = []
    total_close_count = 0
    total_steps = 0

    print(f"\n{'='*60}")
    print(f"Processing {len(episodes)} episodes")
    print(f"{'='*60}\n")

    for ep_idx, ep in enumerate(episodes):
        # Reverse trajectory and recompute goal actions
        fwd_ep = reverse_episode_with_goal_actions(ep)
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
    # Step 4: Analyze action transitions for debugging
    # =====================================================================
    analyze_action_transitions(forward_episodes, num_episodes=2)

    # =====================================================================
    # Step 5: Print dataset statistics
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
        print(f"Table camera image shape: {forward_episodes[0]['images'].shape[1:]}")
        if "wrist_images" in forward_episodes[0]:
            print(f"Wrist camera image shape: {forward_episodes[0]['wrist_images'].shape[1:]}")
        print(f"Action dim: {forward_episodes[0]['action'].shape[1]}")
    print(f"Gripper CLOSE ratio: {100*total_close_ratio:.1f}%")
    print(f"{'='*60}\n")

    # Sanity check for gripper ratio
    if total_close_ratio < 0.1 or total_close_ratio > 0.9:
        print("WARNING: Gripper CLOSE ratio is outside expected range (10-90%)!")
        print("         This may indicate a problem with the gripper heuristic.")

    # =====================================================================
    # Step 6: Save forward episodes
    # =====================================================================
    save_episodes_with_goal_actions(args.out, forward_episodes)
    
    # Calculate file size
    out_path = Path(args.out)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  file size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
