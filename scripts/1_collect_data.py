#!/usr/bin/env python3
"""Step 1: Collect reverse rollouts from Expert B with goal-based actions.

This script collects reverse trajectory data using an FSM-based expert policy.
The expert performs Task B (goal -> table): picks cube from goal position and
places it at a random table position.

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode, the following arrays are saved:
    - obs:           (T, 36)  Policy observation sequence
    - images:        (T, H, W, 3)  RGB images from table camera (uint8)
    - wrist_images:  (T, H, W, 3)  RGB images from wrist camera (uint8)
    - ee_pose:       (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - obj_pose:      (T, 7)   Object (cube) pose
    - action:        (T, 8)   Goal action [ee_pose, gripper]
    - gripper:       (T,)     Gripper action (+1=open, -1=close)
    - fsm_state:     (T,)     FSM state at each timestep (int)
    - place_pose:    (7,)     Target place position (random table position)
    - goal_pose:     (7,)     Goal position (plate center)
    - success:       bool     Whether cube ended up near target position

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (headless mode, 100 episodes)
python scripts/1_collect_data.py --headless --num_episodes 100

# Production run (500 episodes with parallel envs)
CUDA_VISIBLE_DEVICES=0 python scripts/1_collect_data.py \
    --headless --num_episodes 500 --num_envs 256 \
    --out data/B_2images_goal.npz

=============================================================================
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reverse rollouts from Expert B with goal-based actions.",
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
    # Image configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Width of captured images.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=128,
        help="Height of captured images.",
    )
    
    # -----------------------------------------------------------------
    # Data collection parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments. Use fewer envs for image collection "
             "due to increased memory usage.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Maximum simulation steps per episode (before settle steps).",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=40,
        help="Additional steps after expert finishes to let the cube settle.",
    )
    
    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    
    # -----------------------------------------------------------------
    # Output configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--out",
        type=str,
        default="data/B_goal_actions_latest.npz",
        help="Output path for the NPZ file containing collected episodes with goal actions.",
    )
    
    # -----------------------------------------------------------------
    # Simulation backend options
    # -----------------------------------------------------------------
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric backend (PhysX GPU acceleration).",
    )

    # -----------------------------------------------------------------
    # Isaac Lab AppLauncher arguments (--headless, --device, etc.)
    # -----------------------------------------------------------------
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    
    # IMPORTANT: Enable cameras for headless mode with image rendering
    # Without this, the simulation will hang when trying to use camera sensors
    args.enable_cameras = True
    
    return args


def compute_camera_quat_from_lookat(eye: tuple, target: tuple, up: tuple = (0, 0, 1)) -> tuple:
    """Compute camera rotation quaternion from eye position and lookat target.
    
    This allows defining camera orientation intuitively using:
    - eye: where the camera is located
    - target: what point the camera is looking at
    - up: which direction is "up" in world frame (default: Z-up)
    
    The function computes a quaternion for ROS camera convention where:
    - Camera +Z axis points toward the target (forward/optical axis)
    - Camera +X axis points right
    - Camera +Y axis points down
    
    Args:
        eye: Camera position (x, y, z) in world frame
        target: Point to look at (x, y, z) in world frame
        up: World up vector (default Z-up)
        
    Returns:
        Quaternion (w, x, y, z) for use with convention="ros"
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    # Forward direction: from eye toward target
    # This will be the camera's +Z axis in ROS convention
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Right direction: perpendicular to forward and world-up
    # This will be the camera's +X axis
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Camera down direction: perpendicular to forward and right
    # This will be the camera's +Y axis (down in ROS optical frame)
    down = np.cross(forward, right)
    
    # Build rotation matrix from world to camera frame
    # Columns are the camera axes expressed in world coordinates
    # For ROS optical frame: X=right, Y=down, Z=forward
    rotation_matrix = np.column_stack([right, down, forward])
    
    # Convert rotation matrix to quaternion using scipy
    # scipy returns quaternion as [x, y, z, w]
    rot = R.from_matrix(rotation_matrix)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    
    # Isaac Lab expects (w, x, y, z) format
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    
    return (qw, qx, qy, qz)


def create_target_markers(num_envs: int, device: str):
    """Create visualization markers for start and goal positions.
    
    Creates two sets of flat cylinder markers on the table surface:
    - Red markers: Start/place positions (where cube should be placed)
    - Green markers: Goal positions (fixed at plate center)
    
    These are visual-only markers with no physics interaction.
    
    Args:
        num_envs: Number of parallel environments.
        device: Torch device string.
        
    Returns:
        Tuple of (start_markers, goal_markers) VisualizationMarkers objects.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    
    # Marker parameters
    marker_radius = 0.05  # 5cm radius
    marker_height = 0.002  # 2mm height (flat disk)
    table_z = 0.0  # Table surface height
    marker_z = table_z + marker_height / 2 + 0.001  # Slightly above table
    
    # Red marker for start/place positions
    start_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/StartMarkers",
        markers={
            "start": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Red
                ),
            ),
        },
    )
    start_markers = VisualizationMarkers(start_marker_cfg)
    
    # Green marker for goal positions
    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # Green
                ),
            ),
        },
    )
    goal_markers = VisualizationMarkers(goal_marker_cfg)
    
    return start_markers, goal_markers, marker_z


def update_target_markers(
    start_markers,
    goal_markers,
    start_xys: list,
    goal_xy: tuple,
    marker_z: float,
    env,
):
    """Update the positions of start and goal markers.
    
    Args:
        start_markers: VisualizationMarkers for start positions.
        goal_markers: VisualizationMarkers for goal positions.
        start_xys: List of (x, y) tuples for start positions, one per env.
        goal_xy: Tuple (x, y) for the goal position (same for all envs).
        marker_z: Z height for markers.
        env: Isaac Lab environment (for env_origins).
    """
    import torch
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins  # (num_envs, 3)
    
    # Build start marker positions (one per env)
    start_positions = torch.zeros((num_envs, 3), device=device)
    for i, xy in enumerate(start_xys):
        start_positions[i, 0] = xy[0]
        start_positions[i, 1] = xy[1]
        start_positions[i, 2] = marker_z
    # Add env origins for world positions
    start_positions_w = start_positions + env_origins
    
    # Build goal marker positions (same XY for all envs)
    goal_positions = torch.zeros((num_envs, 3), device=device)
    goal_positions[:, 0] = goal_xy[0]
    goal_positions[:, 1] = goal_xy[1]
    goal_positions[:, 2] = marker_z
    # Add env origins for world positions
    goal_positions_w = goal_positions + env_origins
    
    # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    
    # Update marker visualizations
    start_markers.visualize(start_positions_w, identity_quat)
    goal_markers.visualize(goal_positions_w, identity_quat)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int):
    """Dynamically add camera sensors to the environment configuration.
    
    This function modifies env_cfg.scene to add:
    1. A table-view camera (Camera) - fixed third-person view
    2. A wrist camera (Camera) - mounted on robot's end-effector, follows the gripper
    
    Both cameras use standard CameraCfg for consistency and flexibility.
    
    Args:
        env_cfg: The environment configuration object (from parse_env_cfg).
        image_width: Width of captured images.
        image_height: Height of captured images.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    
    # =========================================================================
    # Table Camera (Camera) - Third-person fixed view
    # =========================================================================
    # Define camera using intuitive eye/lookat positions
    # Table center is at approximately (0.5, 0, 0)
    camera_eye = (1.6, 0.7, 0.8)      # Camera position: in front, slightly right, above
    camera_lookat = (0.4, 0.0, 0.2)   # Look at: table center, slightly above table surface
    
    # Compute rotation quaternion from eye and lookat
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)
    
    # Use standard CameraCfg (same as wrist camera for consistency)
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,  # Update every physics step
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            # Clip far objects to avoid seeing adjacent environments
            # Camera is ~1.2m from workspace center, so clip at ~2m to exclude neighbors
            clipping_range=(0.1, 2.5),
        ),
        # Camera position and orientation defined by eye/lookat above
        offset=CameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
            convention="ros",  # ROS convention: +Z forward, +X right, +Y down
        ),
    )
    
    # =========================================================================
    # Wrist Camera (Camera) - Eye-in-hand, attached to robot gripper
    # =========================================================================
    # Use standard CameraCfg (not TiledCamera) because we need it attached to a robot link.
    # The prim_path includes the robot's hand link so the camera follows the gripper.
    env_cfg.scene.wrist_cam = CameraCfg(
        # Mount camera on panda_hand link - it will move with the gripper
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,  # Update every physics step
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),  # Short range since it's close to objects
        ),
        # Offset relative to panda_hand frame
        # pos: forward from gripper, looking down at workspace
        # rot: orientation to look forward and slightly down
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),  # Forward, centered, below the hand link
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),  # Looking forward/down
            convention="ros",
        ),
    )
    
    # Increase environment spacing to prevent seeing adjacent environments
    # Default is 2.5m, increase to 5.0m so neighbors are far enough away
    env_cfg.scene.env_spacing = 5.0
    
    # Disable debug visualization markers (coordinate frame arrows)
    # These are useful for debugging but appear in rendered images
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'object_pose'):
        env_cfg.commands.object_pose.debug_vis = False
    
    # Also disable ee_frame debug visualization if present
    if hasattr(env_cfg, 'scene') and hasattr(env_cfg.scene, 'ee_frame'):
        env_cfg.scene.ee_frame.debug_vis = False
    
    # Ensure good lighting for image capture
    # The scene already has lights, but we can adjust render settings
    if hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'render'):
        env_cfg.sim.render.antialiasing_mode = "FXAA"  # Fast AA for efficiency


def make_env_with_camera(
    task_id: str,
    num_envs: int,
    device: str,
    use_fabric: bool,
    image_width: int,
    image_height: int,
    episode_length_s: float | None = None,
    disable_terminations: bool = False,
):
    """Create an Isaac Lab gym environment with an added camera sensor.
    
    This function wraps the standard make_env but adds a camera to the
    scene configuration before creating the environment.
    
    Args:
        task_id: Gym task id, e.g. "Isaac-Lift-Cube-Franka-IK-Abs-v0".
        num_envs: Number of vectorized env instances.
        device: Torch device string.
        use_fabric: If False, disables Fabric backend.
        image_width: Width of captured images.
        image_height: Height of captured images.
        episode_length_s: Override episode length in seconds.
        disable_terminations: If True, disables all termination conditions.
    
    Returns:
        The created gym environment with camera sensor.
    """
    import gymnasium as gym
    
    print("[DEBUG] make_env_with_camera: Importing isaaclab_tasks...")
    # Ensure Isaac Lab tasks are registered
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    print("[DEBUG] make_env_with_camera: Imports done")
    
    # Parse environment config
    print("[DEBUG] make_env_with_camera: Parsing env config...")
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    print("[DEBUG] make_env_with_camera: Config parsed")
    
    # Override episode length if specified
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
        print(f"[DEBUG] make_env_with_camera: Set episode_length_s={episode_length_s}")
    
    # Disable terminations if requested
    if disable_terminations:
        if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'time_out'):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'object_dropping'):
            env_cfg.terminations.object_dropping = None
        print("[DEBUG] make_env_with_camera: Terminations disabled")
    
    # Add camera to the environment configuration
    print("[DEBUG] make_env_with_camera: Adding camera to env config...")
    add_camera_to_env_cfg(env_cfg, image_width, image_height)
    print("[DEBUG] make_env_with_camera: Camera config added")
    
    # Create environment
    print("[DEBUG] make_env_with_camera: Creating gym environment (this may take a while)...")
    env = gym.make(task_id, cfg=env_cfg)
    print("[DEBUG] make_env_with_camera: Gym environment created")
    return env


def get_fsm_goal_action(expert, object_pose) -> "torch.Tensor":
    """Get the goal action for each environment based on FSM state.
    
    This function computes what the FSM's target position is for each environment,
    WITHOUT updating the FSM state or wait counters. It's a pure query function.
    
    Args:
        expert: PickPlaceExpertB instance with current state.
        object_pose: Current object pose (num_envs, 7) [x, y, z, qw, qx, qy, qz].
    
    Returns:
        Goal action tensor (num_envs, 8) [x, y, z, qw, qx, qy, qz, gripper].
        This is the position the FSM wants the robot to reach.
    """
    import torch
    from rev2fwd_il.experts.pickplace_expert_b import ExpertState, GRIPPER_OPEN, GRIPPER_CLOSE
    
    num_envs = expert.num_envs
    device = expert.device
    
    # Initialize goal action
    goal_action = torch.zeros(num_envs, 8, device=device)
    
    # Compute target positions for each state (same as in act())
    obj_xy = object_pose[:, :2]
    obj_z = object_pose[:, 2]
    
    # Above object position
    above_obj_pos = torch.zeros(num_envs, 3, device=device)
    above_obj_pos[:, :2] = obj_xy
    above_obj_pos[:, 2] = expert.hover_z
    
    # At object position (for grasping)
    at_obj_pos = torch.zeros(num_envs, 3, device=device)
    at_obj_pos[:, :2] = obj_xy
    at_obj_pos[:, 2] = obj_z + expert.grasp_z_offset
    
    # Above place position
    above_place_pos = torch.zeros(num_envs, 3, device=device)
    above_place_pos[:, :2] = expert.place_pose[:, :2]
    above_place_pos[:, 2] = expert.hover_z
    
    # At place position
    at_place_pos = expert.place_pose[:, :3].clone()
    
    # Release position (lower than place position)
    release_pos = expert.place_pose[:, :3].clone()
    release_pos[:, 2] = expert.place_pose[:, 2] + expert.release_z_offset
    
    # Set goal based on current state
    for state_val in ExpertState:
        mask = expert.state == state_val
        
        if not mask.any():
            continue
        
        if state_val == ExpertState.REST:
            goal_action[mask, :3] = expert.rest_pose[mask, :3]
            goal_action[mask, 3:7] = expert.rest_pose[mask, 3:7]
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.GO_ABOVE_OBJ:
            goal_action[mask, :3] = above_obj_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.GO_TO_OBJ:
            goal_action[mask, :3] = at_obj_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.CLOSE:
            goal_action[mask, :3] = at_obj_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_CLOSE
            
        elif state_val == ExpertState.LIFT:
            goal_action[mask, :3] = above_obj_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_CLOSE
            
        elif state_val == ExpertState.GO_ABOVE_PLACE:
            goal_action[mask, :3] = above_place_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_CLOSE
            
        elif state_val == ExpertState.GO_TO_PLACE:
            goal_action[mask, :3] = at_place_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_CLOSE
            
        elif state_val == ExpertState.LOWER_TO_RELEASE:
            goal_action[mask, :3] = release_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_CLOSE
            
        elif state_val == ExpertState.OPEN:
            goal_action[mask, :3] = release_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.LIFT_AFTER_RELEASE:
            goal_action[mask, :3] = above_place_pos[mask]
            goal_action[mask, 3:7] = expert.grasp_quat
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.RETURN_REST:
            goal_action[mask, :3] = expert.rest_pose[mask, :3]
            goal_action[mask, 3:7] = expert.rest_pose[mask, 3:7]
            goal_action[mask, 7] = GRIPPER_OPEN
            
        elif state_val == ExpertState.DONE:
            goal_action[mask, :3] = expert.rest_pose[mask, :3]
            goal_action[mask, 3:7] = expert.rest_pose[mask, 3:7]
            goal_action[mask, 7] = GRIPPER_OPEN
    
    return goal_action


def rollout_expert_B_with_goal_actions(
    env,
    expert,
    task_spec,
    rng,
    horizon: int,
    settle_steps: int = 30,
    markers: tuple = None,
):
    """Run parallel reverse rollouts with Expert B and record goal-based actions.
    
    The key difference from rollout_expert_B_with_images is that we record
    the FSM's goal position as the action, not the smooth interpolated output.
    
    Args:
        env: Isaac Lab gymnasium environment with camera sensor.
        expert: PickPlaceExpertB instance.
        task_spec: Task specification with goal/table bounds.
        rng: NumPy random generator.
        horizon: Maximum steps for each episode.
        settle_steps: Extra steps after expert finishes to let cube settle.
        markers: Tuple of (start_markers, goal_markers, marker_z) for visualization.
                 If None, markers will be created (for first call).
    
    Returns:
        Tuple of (results, markers) where:
        - results: List of (episode_dict, expert_completed) tuples, one per environment.
        - markers: The markers tuple (for reuse in subsequent calls).
    """
    import numpy as np
    import torch
    
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w, is_pose_close_xy
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    print(f"[DEBUG] rollout: device={device}, num_envs={num_envs}")
    
    # Get camera sensor references
    print("[DEBUG] rollout: Getting camera sensor references...")
    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]
    print(f"[DEBUG] rollout: Table camera acquired: {type(table_camera)}")
    print(f"[DEBUG] rollout: Wrist camera acquired: {type(wrist_camera)}")
    
    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    print("[DEBUG] rollout: Resetting environment...")
    obs_dict, _ = env.reset()
    print(f"[DEBUG] rollout: Environment reset done. obs_dict keys: {obs_dict.keys() if isinstance(obs_dict, dict) else 'not a dict'}")
    
    ee_pose = get_ee_pose_w(env)
    print(f"[DEBUG] rollout: EE pose shape: {ee_pose.shape}")
    
    # Test camera data access
    print("[DEBUG] rollout: Testing camera data access...")
    try:
        test_rgb = table_camera.data.output["rgb"]
        print(f"[DEBUG] rollout: Table camera RGB shape: {test_rgb.shape}, dtype: {test_rgb.dtype}")
        test_wrist_rgb = wrist_camera.data.output["rgb"]
        print(f"[DEBUG] rollout: Wrist camera RGB shape: {test_wrist_rgb.shape}, dtype: {test_wrist_rgb.dtype}")
    except Exception as e:
        print(f"[ERROR] rollout: Failed to access camera data: {e}")
    
    # =========================================================================
    # Step 2: Teleport cube to goal position for all envs
    # =========================================================================
    print("[DEBUG] rollout: Teleporting cube to goal position...")
    goal_pose = make_goal_pose_w(env, task_spec.goal_xy, z=0.055)
    teleport_object_to_pose(env, goal_pose, name="object")
    print("[DEBUG] rollout: Cube teleported")
    
    # Let physics settle after teleport
    print("[DEBUG] rollout: Settling physics (10 steps)...")
    zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    for _ in range(10):
        env.step(zero_action)
    print("[DEBUG] rollout: Physics settled")
    
    # =========================================================================
    # Step 3: Sample random place targets for each env
    # =========================================================================
    # Sample place targets that are at least 0.1 away from the goal position
    min_dist_from_goal = 0.1
    place_xys = []
    for _ in range(num_envs):
        while True:
            xy = task_spec.sample_table_xy(rng)
            dist = np.sqrt((xy[0] - task_spec.goal_xy[0])**2 + (xy[1] - task_spec.goal_xy[1])**2)
            if dist >= min_dist_from_goal:
                place_xys.append(xy)
                break
    place_poses_np = np.array([
        [xy[0], xy[1], 0.055, 1.0, 0.0, 0.0, 0.0] for xy in place_xys
    ], dtype=np.float32)
    goal_pose_np = np.array(
        [task_spec.goal_xy[0], task_spec.goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], 
        dtype=np.float32
    )
    
    # =========================================================================
    # Step 4: Initialize expert for each env with different place targets
    # =========================================================================
    ee_pose = get_ee_pose_w(env)
    expert.reset(ee_pose, place_xys[0], place_z=0.055)
    for i, xy in enumerate(place_xys):
        expert.place_pose[i, 0] = xy[0]
        expert.place_pose[i, 1] = xy[1]
    
    # =========================================================================
    # Step 4.5: Create (if needed) and update visual markers for start/goal positions
    # =========================================================================
    if markers is None:
        print("[DEBUG] rollout: Creating target markers (first time)...")
        start_markers, goal_markers, marker_z = create_target_markers(num_envs, device)
        markers = (start_markers, goal_markers, marker_z)
    else:
        print("[DEBUG] rollout: Reusing existing markers...")
        start_markers, goal_markers, marker_z = markers
    update_target_markers(
        start_markers, goal_markers,
        start_xys=place_xys,
        goal_xy=task_spec.goal_xy,
        marker_z=marker_z,
        env=env,
    )
    print("[DEBUG] rollout: Target markers created and positioned")
    
    # =========================================================================
    # Step 5: Initialize per-env recording buffers
    # =========================================================================
    obs_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]           # Table camera images
    wrist_image_lists = [[] for _ in range(num_envs)]     # Wrist camera images
    ee_pose_lists = [[] for _ in range(num_envs)]
    obj_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]          # Goal-based actions
    gripper_lists = [[] for _ in range(num_envs)]
    fsm_state_lists = [[] for _ in range(num_envs)]       # FSM state at each step
    
    expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done_at_step = torch.full((num_envs,), horizon + settle_steps, dtype=torch.int32, device=device)
    
    # =========================================================================
    # Step 6: Run episode and record
    # =========================================================================
    print(f"[DEBUG] rollout: Starting main loop (horizon={horizon})...")
    for t in range(horizon):
        if t == 0 or (t + 1) % 50 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)
        
        # Get observation vector
        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict
        
        # Get camera images from both cameras
        # table_camera.data.output["rgb"] has shape (num_envs, H, W, C) where C is typically 3 or 4
        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        # Ensure we have 3 channels (RGB) - handles both RGB and RGBA formats
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]
        
        # =====================================================================
        # KEY DIFFERENCE: Get FSM goal as action (not smooth interpolated action)
        # =====================================================================
        # Get the FSM's goal position for current state (without state transition)
        goal_action = get_fsm_goal_action(expert, object_pose)
        
        # Record data for each env
        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        goal_action_np = goal_action.cpu().numpy()
        fsm_states_np = expert.state.cpu().numpy()
        
        for i in range(num_envs):
            obs_lists[i].append(obs_np[i])
            image_lists[i].append(table_images_np[i])
            wrist_image_lists[i].append(wrist_images_np[i])
            ee_pose_lists[i].append(ee_pose_np[i])
            obj_pose_lists[i].append(obj_pose_np[i])
            action_lists[i].append(goal_action_np[i])
            gripper_lists[i].append(goal_action_np[i, 7])
            fsm_state_lists[i].append(fsm_states_np[i])
        
        # Compute expert action (this also updates FSM state)
        action = expert.act(ee_pose, object_pose)
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        # Check which envs just finished
        just_done = expert.is_done() & ~expert_completed
        expert_completed = expert_completed | expert.is_done()
        done_at_step[just_done] = t + 1
        
        # Early exit if all envs are done
        if expert_completed.all():
            print(f"[DEBUG] rollout: All envs done at step {t+1}")
            break
    
    print(f"[DEBUG] rollout: Main loop finished. expert_completed={expert_completed}")
    
    # =========================================================================
    # Step 7: Settle steps (continue recording for all envs)
    # =========================================================================
    print(f"[DEBUG] rollout: Starting settle steps ({settle_steps})...")
    for t in range(settle_steps):
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)
        
        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict
        
        # Get camera images from both cameras
        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]
        
        # Get goal action for settle phase (should be rest pose)
        goal_action = get_fsm_goal_action(expert, object_pose)
        
        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        goal_action_np = goal_action.cpu().numpy()
        fsm_states_np = expert.state.cpu().numpy()
        
        for i in range(num_envs):
            if expert_completed[i]:
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_images_np[i])
                wrist_image_lists[i].append(wrist_images_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                action_lists[i].append(goal_action_np[i])
                gripper_lists[i].append(1.0)  # Open gripper during settle
                fsm_state_lists[i].append(fsm_states_np[i])
        
        # Rest action during settle
        rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        rest_action[:, :7] = expert.rest_pose
        rest_action[:, 7] = 1.0  # Open gripper
        
        obs_dict, _, _, _, _ = env.step(rest_action)
    
    print("[DEBUG] rollout: Settle steps finished")
    
    # =========================================================================
    # Step 8: Check success and build episode dicts
    # =========================================================================
    print("[DEBUG] rollout: Building episode dicts...")
    object_pose = get_object_pose_w(env)
    results = []
    
    for i in range(num_envs):
        # Check success for this env
        place_xy = place_xys[i]
        obj_xy = object_pose[i, :2].cpu().numpy()
        dist = np.sqrt((obj_xy[0] - place_xy[0])**2 + (obj_xy[1] - place_xy[1])**2)
        success_bool = dist < task_spec.success_radius
        
        episode_dict = {
            "obs": np.array(obs_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),  # (T, H, W, 3) - table camera
            "wrist_images": np.array(wrist_image_lists[i], dtype=np.uint8),  # (T, H, W, 3) - wrist camera
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "obj_pose": np.array(obj_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),  # (T, 8) - goal actions
            "gripper": np.array(gripper_lists[i], dtype=np.float32),
            "fsm_state": np.array(fsm_state_lists[i], dtype=np.int32),  # (T,) - FSM state
            "place_pose": place_poses_np[i],
            "goal_pose": goal_pose_np,
            "success": success_bool,
        }
        results.append((episode_dict, expert_completed[i].item()))
    
    print(f"[DEBUG] rollout: Done. Returning {len(results)} episodes")
    return results, markers


def save_episodes_with_goal_actions(path: str, episodes: list[dict]) -> None:
    """Save episodes with goal actions to an NPZ file.
    
    Args:
        path: Output file path.
        episodes: List of episode dictionaries.
    """
    from pathlib import Path
    import numpy as np
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with allow_pickle for list of dicts
    np.savez_compressed(path, episodes=episodes)
    print(f"Saved {len(episodes)} episodes with goal actions to {path}")
    
    # Print some statistics
    if episodes:
        ep0 = episodes[0]
        print(f"  - State obs shape: {ep0['obs'].shape}")
        print(f"  - Table camera image shape: {ep0['images'].shape}")
        print(f"  - Wrist camera image shape: {ep0['wrist_images'].shape}")
        print(f"  - Goal action shape: {ep0['action'].shape}")
        print(f"  - FSM state shape: {ep0['fsm_state'].shape}")
        print(f"  - Episode length: {ep0['obs'].shape[0]}")
        
        # Show FSM state distribution
        from rev2fwd_il.experts.pickplace_expert_b import ExpertState
        fsm_states = ep0['fsm_state']
        print(f"  - FSM state distribution (episode 0):")
        for state in ExpertState:
            count = (fsm_states == state.value).sum()
            if count > 0:
                print(f"      {state.name}: {count} steps")


def main() -> None:
    """Main entry point for collecting reverse rollouts with goal-based actions."""
    args = _parse_args()
    
    # =====================================================================
    # Step 1: Launch Isaac Sim
    # =====================================================================
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # =====================================================================
    # Step 2: Import modules (must be after Isaac Sim initialization)
    # =====================================================================
    import numpy as np
    
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.utils.seed import set_seed
    
    try:
        # =================================================================
        # Step 3: Set random seeds for reproducibility
        # =================================================================
        print("[DEBUG] Step 3: Setting random seeds...")
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        print("[DEBUG] Step 3: Done setting seeds")
        
        # =================================================================
        # Step 4: Create environment with camera
        # =================================================================
        print("[DEBUG] Step 4: Creating environment with camera...")
        print(f"[DEBUG]   task_id={args.task}")
        print(f"[DEBUG]   num_envs={args.num_envs}")
        print(f"[DEBUG]   device={args.device}")
        print(f"[DEBUG]   image_size={args.image_width}x{args.image_height}")
        
        num_envs = args.num_envs
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,  # Prevent auto-reset
            disable_terminations=True,  # Prevent robot teleport on task completion
        )
        print("[DEBUG] Step 4: Environment created successfully")
        
        device = env.unwrapped.device
        print(f"[DEBUG] Environment device: {device}")
        
        # Check if camera sensor exists
        print("[DEBUG] Checking camera sensor...")
        if "table_cam" in env.unwrapped.scene.sensors:
            camera = env.unwrapped.scene.sensors["table_cam"]
            print(f"[DEBUG] Camera sensor found: {type(camera)}")
            print(f"[DEBUG] Camera data types: {camera.cfg.data_types}")
            print(f"[DEBUG] Camera resolution: {camera.cfg.width}x{camera.cfg.height}")
        else:
            print("[ERROR] Camera sensor 'table_cam' not found in scene.sensors!")
            print(f"[DEBUG] Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        
        # =================================================================
        # Step 5: Define task specification
        # =================================================================
        print("[DEBUG] Step 5: Creating task specification...")
        task_spec = PickPlaceTaskSpec(
            goal_xy=(0.5, 0.0),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )
        print("[DEBUG] Step 5: Task specification created")
        
        # =================================================================
        # Step 6: Create Expert B
        # =================================================================
        print("[DEBUG] Step 6: Creating Expert B...")
        expert = PickPlaceExpertB(
            num_envs=num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.04,
            position_threshold=0.015,
            wait_steps=task_spec.settle_steps,
        )
        print("[DEBUG] Step 6: Expert B created")
        
        # =================================================================
        # Step 7: Data collection loop
        # =================================================================
        episodes = []
        completed_count = 0
        success_count = 0
        
        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} reverse rollouts with GOAL-BASED actions")
        print(f"Settings:")
        print(f"  - num_envs (parallel): {num_envs}")
        print(f"  - horizon: {args.horizon}")
        print(f"  - settle_steps: {args.settle_steps}")
        print(f"  - image_size: {args.image_width}x{args.image_height}")
        print(f"  - Action type: FSM goal position (NOT smooth interpolation)")
        print(f"  - Only saving episodes with completed FSM (DONE state)")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        batch_count = 0
        max_batches = (args.num_episodes // num_envs + 1) * 3
        markers = None  # Will be created on first rollout and reused
        
        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            
            # Run parallel episodes with Expert B and goal-based action capture
            results, markers = rollout_expert_B_with_goal_actions(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
                markers=markers,
            )
            
            # Process results from all parallel envs
            batch_completed = 0
            batch_success = 0
            for episode_dict, expert_completed_flag in results:
                if expert_completed_flag:
                    completed_count += 1
                    batch_completed += 1
                    episodes.append(episode_dict)
                    
                    if episode_dict["success"]:
                        success_count += 1
                        batch_success += 1
                    
                    if len(episodes) >= args.num_episodes:
                        break
            
            # Print progress
            elapsed = time.time() - start_time
            total_attempts = batch_count * num_envs
            rate = total_attempts / elapsed
            print(
                f"Batch {batch_count:3d} ({num_envs} envs) | "
                f"Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{num_envs} completed, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s"
            )
        
        # =================================================================
        # Step 8: Print summary statistics
        # =================================================================
        elapsed = time.time() - start_time
        total_attempts = batch_count * num_envs
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Parallel envs: {num_envs}")
        print(f"Total batches: {batch_count}")
        print(f"Total attempts: {total_attempts}")
        print(f"Completed FSM: {completed_count} ({100*completed_count/total_attempts:.1f}%)")
        print(f"Saved episodes: {len(episodes)}")
        print(f"Success (cube at target): {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"Effective rate: {len(episodes)/elapsed:.2f} episodes/s")
        print(f"{'='*60}\n")
        
        # =================================================================
        # Step 9: Save collected episodes to NPZ file
        # =================================================================
        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        
        # Clean up environment
        env.close()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
