#!/usr/bin/env python3
"""Debug script for Expert B pick-and-place one episode.

This script demonstrates the "reverse" task for imitation learning data collection:
    - In the FORWARD task: Expert A picks cube from random table position → places at goal (plate center)
    - In the REVERSE task: Expert B picks cube from goal (plate center) → places at random table position

This script specifically tests Expert B, which performs the reverse trajectory.

Workflow:
    1. Launch Isaac Sim and create a Franka robot environment
    2. Reset the environment to initial state
    3. Teleport the cube to the goal position (simulating end state of forward task)
    4. Sample a random target position on the table (excluding goal vicinity)
    5. Run Expert B finite state machine to pick & place the cube
    6. Verify the robot returns to REST pose after completion

Expert B State Machine:
    REST → GO_ABOVE_OBJ → GO_TO_OBJ → CLOSE → LIFT → GO_ABOVE_PLACE → GO_TO_PLACE → OPEN → RETURN_REST → DONE

Usage:
    # Run with GUI visualization
    python scripts/01_debug_expert_B_one_episode.py --headless 0

    # Run in headless mode (faster, no rendering)
    python scripts/01_debug_expert_B_one_episode.py --headless

    # Custom parameters
    python scripts/01_debug_expert_B_one_episode.py --headless --seed 123 --steps 600 --goal_xy 0.5 0.1

Note:
    This script requires Isaac Sim to be properly installed and the conda environment
    'rev2fwd_il' to be activated before running.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    This function sets up argument parsing for both custom script arguments
    and Isaac Lab's AppLauncher arguments (e.g., --headless, --device).

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - task: Gym environment ID for the manipulation task
            - num_envs: Number of parallel simulation environments
            - steps: Maximum simulation steps per episode
            - seed: Random seed for reproducibility
            - goal_xy: (x, y) coordinates for initial cube placement
            - disable_fabric: Flag to disable Fabric backend
            - headless: Flag for headless rendering (from AppLauncher)
            - device: Compute device, e.g., 'cuda:0' (from AppLauncher)
    """
    parser = argparse.ArgumentParser(
        description="Debug Expert B pick-and-place one episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with visualization
    python scripts/01_debug_expert_B_one_episode.py --headless 0

    # Run headless with custom seed
    python scripts/01_debug_expert_B_one_episode.py --headless --seed 123
        """,
    )

    # ----- Task Configuration -----
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Gym task ID. Default uses Franka robot with IK absolute pose control.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments. Use 1 for debugging, higher for batch testing.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum simulation steps per episode. Expert typically finishes in ~300 steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/torch/python RNG. Ensures reproducible place positions.",
    )
    parser.add_argument(
        "--goal_xy",
        type=float,
        nargs=2,
        default=[0.5, 0.0],
        help="Goal (x, y) position where cube is initially placed (plate center in forward task).",
    )
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric backend in simulation. May reduce warnings but slower.",
    )

    # ----- Isaac Lab AppLauncher Arguments -----
    # AppLauncher.add_app_launcher_args() injects standard Isaac Sim arguments:
    #   --headless: Run without GUI (0=GUI, 1=headless, or just --headless for headless)
    #   --device: Torch device for simulation (e.g., 'cuda:0', 'cpu')
    #   --experience: Path to Kit experience file
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for the Expert B debugging script.

    This function orchestrates the entire pick-and-place demonstration:
        1. Initializes Isaac Sim via AppLauncher
        2. Creates the simulation environment
        3. Sets up the Expert B controller
        4. Runs the pick-and-place episode
        5. Reports success/failure metrics
    """
    # =========================================================================
    # Step 1: Parse arguments and launch Isaac Sim
    # =========================================================================
    args = _parse_args()

    # IMPORTANT: AppLauncher must be called BEFORE any other Isaac/Omniverse imports.
    # This initializes the Omniverse Kit application and USD runtime.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # Reference to the Kit application

    # =========================================================================
    # Step 2: Import modules (AFTER Isaac Sim is initialized)
    # =========================================================================
    # These imports depend on Isaac Sim being running, so they must come after AppLauncher
    from rev2fwd_il.sim.make_env import make_env
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec, make_goal_pose_w, is_pose_close_xy
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.utils.seed import set_seed

    try:
        # =====================================================================
        # Step 3: Initialize RNG and create environment
        # =====================================================================
        # Set random seeds for reproducibility across numpy, torch, and python
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)  # NumPy Generator for sampling

        # Create Isaac Lab gym environment
        # - task_id: Registered gym environment name
        # - num_envs: Vectorized env count (parallel simulations)
        # - device: GPU/CPU for tensor operations
        # - use_fabric: Fabric backend for faster USD operations
        # - episode_length_s: Set large value to prevent auto-reset during debugging
        # - disable_terminations: Prevent robot from teleporting back on task completion
        env = make_env(
            task_id=args.task,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            episode_length_s=100.0,  # Prevent auto-reset (default is 5.0s = ~250 steps)
            disable_terminations=True,  # Prevent robot teleport on task completion
        )

        device = env.unwrapped.device  # Actual device used by simulation
        num_envs = env.unwrapped.num_envs  # Actual number of environments

        # =====================================================================
        # Step 4: Configure task specification
        # =====================================================================
        # PickPlaceTaskSpec defines geometric parameters for the pick-place task:
        #   - goal_xy: Fixed target position (plate center for forward task)
        #   - hover_z: Height for safe movement above objects
        #   - grasp_z_offset: Vertical offset when grasping (0 = grasp at object center)
        #   - success_radius: XY distance threshold for success detection
        #   - settle_steps: Wait steps after reaching each waypoint
        task_spec = PickPlaceTaskSpec(
            goal_xy=tuple(args.goal_xy),
            hover_z=0.25,  # 25cm above table for safe transit
            grasp_z_offset=0.0,  # Grasp at object center height
            success_radius=0.03,  # 3cm tolerance for success
            settle_steps=10,  # Wait 10 steps at each waypoint
        )

        # =====================================================================
        # Step 5: Reset environment and get initial state
        # =====================================================================
        obs_dict, _ = env.reset()

        # Get initial end-effector (EE) pose - this becomes the REST pose
        # Pose format: [x, y, z, qw, qx, qy, qz] (position + wxyz quaternion)
        ee_pose = get_ee_pose_w(env)
        print(f"Initial EE pose: {ee_pose[0, :3].cpu().numpy()}")

        # =====================================================================
        # Step 6: Teleport cube to goal position (simulate forward task end state)
        # =====================================================================
        # In the reverse task, we start with the cube at the goal position
        # (as if Expert A just finished placing it there)
        goal_pose = make_goal_pose_w(env, task_spec.goal_xy, z=0.055)  # Cube height ~5.5cm
        teleport_object_to_pose(env, goal_pose, name="object")

        # Step simulation a few times to let physics settle after teleport
        # This prevents the cube from "bouncing" due to sudden position change
        zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        for _ in range(10):
            env.step(zero_action)

        # Verify object position after teleport
        object_pose = get_object_pose_w(env)
        print(f"Object teleported to: {object_pose[0, :3].cpu().numpy()}")

        # =====================================================================
        # Step 7: Sample random placement target on table
        # =====================================================================
        # Expert B will place the cube here (random position, avoiding goal area)
        place_xy = task_spec.sample_table_xy(rng)
        print(f"Target place position: {place_xy}")

        # =====================================================================
        # Step 8: Initialize Expert B finite state machine
        # =====================================================================
        # Expert B is a scripted policy that outputs:
        #   - EE target pose (7D: position + quaternion)
        #   - Gripper command (1D: +1.0=open, -1.0=close)
        expert = PickPlaceExpertB(
            num_envs=num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            position_threshold=0.015,  # 1.5cm threshold for "reached" detection
            wait_steps=task_spec.settle_steps,
        )

        # Reset expert with current EE pose (becomes REST target) and place target
        ee_pose = get_ee_pose_w(env)
        expert.reset(ee_pose, place_xy, place_z=0.055)

        print("\n" + "=" * 60)
        print("Starting Expert B episode")
        print("=" * 60)

        # =====================================================================
        # Step 9: Run episode loop
        # =====================================================================
        for t in range(args.steps):
            # --- Get current state from simulation ---
            ee_pose = get_ee_pose_w(env)  # End-effector pose in local env frame
            object_pose = get_object_pose_w(env)  # Object pose in local env frame

            # --- Compute expert action based on current state ---
            # Action shape: (num_envs, 8) = [x, y, z, qw, qx, qy, qz, gripper]
            action = expert.act(ee_pose, object_pose)

            # --- Step the simulation with the action ---
            obs_dict, reward, terminated, truncated, info = env.step(action)

            # --- Print diagnostics every 20 steps ---
            if t % 20 == 0:
                state_names = expert.get_state_names()  # Current FSM state name
                gripper_val = action[0, 7].item()  # Gripper command value
                print(
                    f"Step {t:4d} | State: {state_names[0]:16s} | "
                    f"EE: [{ee_pose[0, 0]:.3f}, {ee_pose[0, 1]:.3f}, {ee_pose[0, 2]:.3f}] | "
                    f"Obj: [{object_pose[0, 0]:.3f}, {object_pose[0, 1]:.3f}, {object_pose[0, 2]:.3f}] | "
                    f"Gripper: {gripper_val:+.1f}"
                )

            # --- Check if expert finished (all envs in DONE state) ---
            if expert.is_done().all():
                print(f"\nExpert finished at step {t}")
                break

        # =====================================================================
        # Step 10: Evaluate episode results
        # =====================================================================
        print("\n" + "=" * 60)
        print("Episode finished")
        print("=" * 60)

        # Get final poses
        object_pose = get_object_pose_w(env)
        ee_pose = get_ee_pose_w(env)

        print(f"Final object position: [{object_pose[0, 0]:.3f}, {object_pose[0, 1]:.3f}, {object_pose[0, 2]:.3f}]")
        print(f"Target place position: [{place_xy[0]:.3f}, {place_xy[1]:.3f}]")

        # --- Check placement success ---
        # Success = object XY position within success_radius of target
        success = is_pose_close_xy(object_pose, place_xy, task_spec.success_radius)
        xy_dist = torch.norm(object_pose[:, :2] - torch.tensor([place_xy], device=device), dim=-1)

        print(f"XY distance to target: {xy_dist[0].item():.4f}")
        print(f"Success (within {task_spec.success_radius}m): {success[0].item()}")

        # --- Check if robot returned to REST pose ---
        rest_pose = expert.rest_pose
        ee_dist = torch.norm(ee_pose[:, :3] - rest_pose[:, :3], dim=-1)
        print(f"EE distance to rest: {ee_dist[0].item():.4f}")

        # =====================================================================
        # Step 11: Cleanup
        # =====================================================================
        env.close()

    finally:
        # IMPORTANT: Always close the simulation app to release GPU resources
        simulation_app.close()


if __name__ == "__main__":
    main()
