#!/usr/bin/env python3
"""Evaluate Policy A on the forward pick-and-place task in Isaac Lab.

=============================================================================
OVERVIEW
=============================================================================
This script implements Step 4 of the reverse-to-forward imitation learning
pipeline: evaluating a trained BC policy on the actual FORWARD task in the
Isaac Lab simulator.

The evaluation measures whether the learned policy can successfully pick up
a cube from random starting positions and place it at the goal location
(plate center).

=============================================================================
FORWARD TASK DEFINITION
=============================================================================
Task Setup:
    - Environment: Isaac-Lift-Cube-Franka-IK-Abs-v0 (Franka robot with IK control)
    - Object: A cube spawns at a RANDOM position on the table (env.reset default)
    - Goal: Fixed position at the plate center (default: x=0.5, y=0.0)
    - Action Space: 8-dim [ee_pose(7), gripper(1)] with IK-Abs control

Success Criterion:
    - The cube's final XY position must be within success_radius (3cm) of the goal
    - NOTE: Gripper state is NOT considered for success determination

Evaluation Metrics:
    - Success Rate: Percentage of rollouts where the cube reaches the goal
    - Average Final Distance: Mean XY distance from cube to goal at episode end
    - Min/Max Final Distance: Best and worst performance across rollouts

=============================================================================
EVALUATION PROCEDURE
=============================================================================
For each rollout:
    1. Reset the environment (cube spawns at random table position)
    2. For each timestep up to horizon:
       a. Get observation from environment
       b. Normalize observation using saved mean/std statistics
       c. Run policy inference: obs -> action
       d. Execute action in environment
       e. Check for episode termination (timeout or failure)
    3. Measure final cube-to-goal distance
    4. Record success/failure based on distance threshold

=============================================================================
USAGE EXAMPLES
=============================================================================
# Standard evaluation (headless mode, no GUI)
python scripts/40_eval_A.py \\
    --checkpoint runs/bc_A/model.pt \\
    --norm runs/bc_A/norm.json \\
    --num_rollouts 50 \\
    --horizon 450 \\
    --headless

# Quick validation (fewer rollouts)
python scripts/40_eval_A.py \\
    --checkpoint runs/bc_A/model.pt \\
    --norm runs/bc_A/norm.json \\
    --num_rollouts 5 \\
    --horizon 450 \\
    --headless

# With GUI visualization (for debugging)
python scripts/40_eval_A.py \\
    --checkpoint runs/bc_A/model.pt \\
    --norm runs/bc_A/norm.json \\
    --num_rollouts 10 \\
    --horizon 450

# Using CPU (if no GPU available)
python scripts/40_eval_A.py \\
    --checkpoint runs/bc_A/model.pt \\
    --norm runs/bc_A/norm.json \\
    --num_rollouts 20 \\
    --headless \\
    --device cpu

=============================================================================
NOTES ON ISAAC LAB INTEGRATION
=============================================================================
- Isaac Sim must be launched via AppLauncher BEFORE importing Isaac Lab modules
- The --headless flag runs without GUI (faster, suitable for batch evaluation)
- Environment uses Fabric backend by default (--disable_fabric 1 to disable)
- The default episode length in the environment is 250 steps; use --horizon
  to specify the maximum steps per evaluation rollout

=============================================================================
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    This function is called BEFORE Isaac Sim is launched. It uses Isaac Lab's
    AppLauncher.add_app_launcher_args() to add simulator-specific arguments
    like --headless, --device, etc.

    Returns:
        Namespace containing all parsed arguments:
            - checkpoint: Path to model.pt
            - norm: Path to norm.json
            - task: Isaac Lab task ID
            - num_envs: Number of parallel environments
            - num_rollouts: Number of evaluation episodes
            - horizon: Maximum steps per episode
            - seed: Random seed
            - disable_fabric: Whether to disable Fabric backend
            - (plus AppLauncher args: headless, device, etc.)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Policy A on the forward pick-and-place task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Model Checkpoint Arguments
    # =========================================================================
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/bc_A/model.pt",
        help="Path to the trained model checkpoint (model.pt). This file contains "
             "the policy network weights and architecture information.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="runs/bc_A/norm.json",
        help="Path to the observation normalization statistics (norm.json). "
             "This file contains mean and std arrays for z-score normalization.",
    )

    # =========================================================================
    # Environment Configuration
    # =========================================================================
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab Gym task ID. Default uses Franka robot with IK-Abs control.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments. For evaluation, typically 1 is used "
             "to ensure deterministic sequential rollouts. Default: 1.",
    )

    # =========================================================================
    # Evaluation Parameters
    # =========================================================================
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=50,
        help="Number of independent evaluation episodes to run. More rollouts "
             "give more statistically reliable results. Default: 50.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=250,
        help="Maximum number of environment steps per rollout. The episode may "
             "terminate earlier due to timeout or failure conditions. Default: 250.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Controls environment randomization "
             "and policy inference. Default: 42.",
    )

    # =========================================================================
    # Simulator Options
    # =========================================================================
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disable the Fabric backend in Isaac Lab. Fabric provides "
             "faster tensor operations but may have compatibility issues. Default: 0.",
    )

    # =========================================================================
    # AppLauncher Arguments (--headless, --device, etc.)
    # =========================================================================
    # These are added by Isaac Lab's AppLauncher and include:
    #   --headless: Run without GUI (faster)
    #   --device: Specify GPU device
    #   --enable_cameras: Enable camera rendering
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for Policy A evaluation.

    Execution Flow:
        1. Parse command-line arguments (before Isaac Sim launch)
        2. Launch Isaac Sim via AppLauncher
        3. Import Isaac Lab modules (must be after launch)
        4. Load trained policy and normalization statistics
        5. Create evaluation environment
        6. Run evaluation rollouts
        7. Print results and cleanup

    Note: Isaac Sim must be launched via AppLauncher before importing any
    Isaac Lab modules. This is why imports are done inside the function.
    """
    # =========================================================================
    # Step 1: Parse Arguments and Launch Isaac Sim
    # =========================================================================
    args = _parse_args()

    # Launch Isaac Sim - this must happen BEFORE importing Isaac Lab modules
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # =========================================================================
    # Step 2: Import Modules (AFTER Isaac Sim is launched)
    # =========================================================================
    import torch
    import numpy as np

    from rev2fwd_il.sim.make_env import make_env
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.eval.rollout import evaluate_A_forward, load_policy_and_norm
    from rev2fwd_il.utils.seed import set_seed

    # =========================================================================
    # Step 3: Set Random Seed
    # =========================================================================
    set_seed(args.seed)

    # Determine device for policy inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # =========================================================================
    # Step 4: Load Policy and Normalization Statistics
    # =========================================================================
    print(f"\nLoading checkpoint: {args.checkpoint}")
    print(f"Loading norm: {args.norm}")

    policy, mean, std = load_policy_and_norm(
        checkpoint_path=args.checkpoint,
        norm_path=args.norm,
        device=device,
    )

    print(f"Policy loaded: obs_dim={policy.obs_dim}, act_dim={policy.act_dim}")

    # =========================================================================
    # Step 5: Create Evaluation Environment
    # =========================================================================
    print(f"\nCreating environment: {args.task}")

    # env = make_env(
    #     task_id=args.task,
    #     num_envs=args.num_envs,
    #     device=device,
    #     use_fabric=(args.disable_fabric == 0),
    # )
    env = make_env(
        task_id=args.task,
        num_envs=args.num_envs,
        device=device,
        use_fabric=(args.disable_fabric == 0),
        episode_length_s=100.0,  # Prevent auto-reset (default is 5.0s)
        disable_terminations=True,  # Prevent robot teleport on task completion
    )

    print(f"Environment created with {env.unwrapped.num_envs} envs")

    # =========================================================================
    # Step 6: Define Task Specification
    # =========================================================================
    # The task spec defines the goal position and success criteria
    task_spec = PickPlaceTaskSpec(
        goal_xy=(0.5, 0.0),      # Plate center (forward task goal)
        success_radius=0.03,     # 3cm success threshold
    )

    # =========================================================================
    # Step 7: Run Evaluation
    # =========================================================================
    results = evaluate_A_forward(
        env=env,
        policy=policy,
        mean=mean,
        std=std,
        task_spec=task_spec,
        num_rollouts=args.num_rollouts,
        horizon=args.horizon,
        device=device,
        verbose=True,
    )

    # =========================================================================
    # Step 8: Print Final Summary
    # =========================================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Horizon: {args.horizon}")
    print("-"*60)
    print(f"SUCCESS RATE: {100*results['success_rate']:.1f}%")
    print(f"AVG FINAL DIST: {results['avg_final_dist']:.4f}m")
    print(f"MIN FINAL DIST: {min(results['final_dists']):.4f}m")
    print(f"MAX FINAL DIST: {max(results['final_dists']):.4f}m")
    print("="*60 + "\n")

    # =========================================================================
    # Step 9: Cleanup
    # =========================================================================
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
