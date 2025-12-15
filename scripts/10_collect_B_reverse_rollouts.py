#!/usr/bin/env python3
"""Collect reverse rollouts from Expert B for imitation learning.

This script implements Step 1 of the reverse-to-forward imitation learning pipeline:
collecting demonstration data from Expert B performing the REVERSE pick-and-place task.

=============================================================================
REVERSE TASK DEFINITION (Expert B)
=============================================================================
- Initial state: Cube is at the GOAL position (plate center at ~[0.5, 0.0])
- Task: Pick up the cube and place it at a RANDOM position on the table
- Terminal state: Cube is placed at random table position, robot returns to REST

This is the OPPOSITE of what we want Policy A to learn (forward task: table -> goal).
By collecting reverse rollouts and time-reversing them, we can train Policy A.

=============================================================================
DATA COLLECTION PROCESS
=============================================================================
For each episode:
    1. Reset environment and teleport cube to GOAL position
    2. Sample a random target position on the table
    3. Run Expert B's finite state machine (FSM):
       REST -> GO_ABOVE_OBJ -> GO_TO_OBJ -> CLOSE -> LIFT -> 
       GO_ABOVE_PLACE -> GO_TO_PLACE -> LOWER_TO_RELEASE -> OPEN -> RETURN_REST -> DONE
    4. Add settle steps to let the cube come to rest
    5. Check if expert completed the full FSM (reached DONE state)
    6. Save episode data: (obs, ee_pose, obj_pose, gripper, place_pose, goal_pose, success)

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode i, the following arrays are saved:
    - obs_i:        (T, 36)  Policy observation sequence
    - ee_pose_i:    (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - obj_pose_i:   (T, 7)   Object (cube) pose [x, y, z, qw, qx, qy, qz]
    - gripper_i:    (T,)     Gripper action (+1=open, -1=close)
    - place_pose_i: (7,)     Target place position (random table position)
    - goal_pose_i:  (7,)     Goal position (plate center, fixed)
    - success_i:    bool     Whether cube ended up near target position

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (headless mode, 100 episodes)
python scripts/10_collect_B_reverse_rollouts.py --headless --num_episodes 100

# Custom output path and seed
python scripts/10_collect_B_reverse_rollouts.py --headless --num_episodes 50 \\
    --out data/B_reverse_50eps.npz --seed 42

# With GUI visualization (slower but useful for debugging)
python scripts/10_collect_B_reverse_rollouts.py --num_episodes 10

=============================================================================
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace containing all configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Collect reverse rollouts from Expert B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Task and environment configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab Gym task ID. Must use IK-Abs control for time reversal to work.",
    )
    
    # -----------------------------------------------------------------
    # Data collection parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect. Only episodes where Expert B "
             "completes the full FSM (reaches DONE state) are saved.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Maximum simulation steps per episode (before settle steps). "
             "Expert B typically needs 300-400 steps to complete the full FSM. "
             "If horizon is too short, the expert won't reach DONE state.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=40,
        help="Additional steps after expert finishes to let the cube settle. "
             "This ensures the cube comes to rest before measuring success.",
    )
    
    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Controls: "
             "- NumPy random generator (for sampling table positions) "
             "- Python random module "
             "- PyTorch random generators",
    )
    
    # -----------------------------------------------------------------
    # Output configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--out",
        type=str,
        default="data/B_reverse_100eps.npz",
        help="Output path for the NPZ file containing collected episodes.",
    )
    
    # -----------------------------------------------------------------
    # Simulation backend options
    # -----------------------------------------------------------------
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric backend (PhysX GPU acceleration). "
             "Use this if you encounter GPU memory issues.",
    )

    # -----------------------------------------------------------------
    # Isaac Lab AppLauncher arguments (--headless, --device, etc.)
    # -----------------------------------------------------------------
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for reverse rollout collection.
    
    This function:
        1. Initializes Isaac Sim via AppLauncher
        2. Creates the gymnasium environment
        3. Sets up Expert B with FSM-based control
        4. Collects episodes in a loop until num_episodes are saved
        5. Saves all collected episodes to an NPZ file
    """
    args = _parse_args()

    # =====================================================================
    # Step 1: Launch Isaac Sim
    # =====================================================================
    # AppLauncher must be called before importing any Isaac Lab modules.
    # It initializes the Omniverse runtime and loads necessary extensions.
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # =====================================================================
    # Step 2: Import modules (must be after Isaac Sim initialization)
    # =====================================================================
    import numpy as np

    from rev2fwd_il.sim.make_env import make_env
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.data.recorder import rollout_expert_B_reverse
    from rev2fwd_il.data.io_npz import save_episodes
    from rev2fwd_il.utils.seed import set_seed

    try:
        # =================================================================
        # Step 3: Set random seeds for reproducibility
        # =================================================================
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        # =================================================================
        # Step 4: Create the gymnasium environment
        # =================================================================
        # We use num_envs=1 because the recorder is designed for single-env
        # sequential data collection. For parallel collection, use a
        # different approach with vectorized environments.
        env = make_env(
            task_id=args.task,
            num_envs=1,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
        )

        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        # =================================================================
        # Step 5: Define task specification
        # =================================================================
        # The task spec defines:
        # - goal_xy: The "plate center" where cube should end up (forward task)
        # - hover_z: Height for hovering above objects during manipulation
        # - success_radius: How close the cube must be to target for "success"
        task_spec = PickPlaceTaskSpec(
            goal_xy=(0.5, 0.0),      # Plate center position
            hover_z=0.25,            # Hover height (25cm above table)
            grasp_z_offset=0.0,      # Grasp at object center height
            success_radius=0.08,     # 8cm tolerance for success (relaxed)
            settle_steps=10,         # Wait steps at each FSM state
        )

        # =================================================================
        # Step 6: Create Expert B with tuned parameters
        # =================================================================
        # Expert B uses a finite state machine (FSM) to execute the reverse
        # pick-and-place task. Key parameters:
        # - release_z_offset: How much to lower before releasing (negative = lower)
        #   This helps the cube land stably on the table.
        # - position_threshold: Distance threshold for "reached" detection
        # - wait_steps: How many steps to wait at each state before transitioning
        expert = PickPlaceExpertB(
            num_envs=num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.015,   # Lower 1.5cm before releasing for stability
            position_threshold=0.015,  # 1.5cm threshold for reaching waypoints
            wait_steps=task_spec.settle_steps,
        )

        # =================================================================
        # Step 7: Data collection loop
        # =================================================================
        episodes = []           # List to store collected Episode objects
        completed_count = 0     # Number of episodes where Expert B reached DONE
        success_count = 0       # Number of episodes where cube is near target

        # Print collection settings
        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} reverse rollouts")
        print(f"Settings:")
        print(f"  - horizon: {args.horizon}")
        print(f"  - settle_steps: {args.settle_steps}")
        print(f"  - success_radius: {task_spec.success_radius}m")
        print(f"  - release_z_offset: {expert.release_z_offset}m")
        print(f"  - Only saving episodes with completed FSM (DONE state)")
        print(f"{'='*60}\n")

        start_time = time.time()
        attempt_count = 0
        max_attempts = args.num_episodes * 3  # Limit total attempts to 3x requested

        # Keep collecting until we have enough episodes or hit max attempts
        while len(episodes) < args.num_episodes and attempt_count < max_attempts:
            attempt_count += 1

            # Run one episode with Expert B
            # Returns: (Episode object, whether expert completed FSM)
            episode, expert_completed = rollout_expert_B_reverse(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
            )

            # Only save episodes where Expert B completed the full FSM
            # This ensures we have complete trajectories for time reversal
            if expert_completed:
                completed_count += 1
                episodes.append(episode)
                
                # Track success rate (cube near target position)
                if episode.success:
                    success_count += 1

            # Print progress every 10 episodes
            if attempt_count % 10 == 0 or attempt_count == 1:
                elapsed = time.time() - start_time
                rate = attempt_count / elapsed
                print(
                    f"Attempt {attempt_count:4d} | "
                    f"Saved: {len(episodes)}/{args.num_episodes} | "
                    f"Completed: {completed_count} ({100*completed_count/attempt_count:.1f}%) | "
                    f"Success: {success_count} | "
                    f"Rate: {rate:.2f} ep/s | "
                    f"Length: {episode.length}"
                )

        # =================================================================
        # Step 8: Print summary statistics
        # =================================================================
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Total attempts: {attempt_count}")
        print(f"Completed FSM: {completed_count} ({100*completed_count/attempt_count:.1f}%)")
        print(f"Saved episodes: {len(episodes)}")
        print(f"Success (cube at target): {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        # =================================================================
        # Step 9: Save collected episodes to NPZ file
        # =================================================================
        save_episodes(args.out, episodes)

        # Clean up environment
        env.close()

    finally:
        # Always close the simulation app to release GPU resources
        simulation_app.close()


if __name__ == "__main__":
    main()
