#!/usr/bin/env python3
"""Step 1: Collect nut threading demonstration data with force sensing.

This script collects nut threading trajectory data from the FORGE environment.
The robot performs a nut-bolt fastening task where it threads a nut onto a bolt.

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode, the following arrays are saved:
    - obs:              (T, obs_dim)  Policy observation sequence
    - state:            (T, state_dim) Full state observation (privileged)
    - images:           (T, H, W, 3)  RGB images from table camera (uint8)
    - wrist_wrist_cam_front:      (T, H, W, 3)  Wrist camera looking forward
    - wrist_wrist_cam_down:       (T, H, W, 3)  Wrist camera looking down
    - wrist_wrist_cam_front_down: (T, H, W, 3)  Wrist camera 45 deg pitch
    - wrist_wrist_cam_front_slight: (T, H, W, 3) Wrist camera 30 deg pitch
    - ee_pose:          (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - nut_pose:         (T, 7)   Nut (held asset) pose
    - bolt_pose:        (T, 7)   Bolt (fixed asset) pose
    - action:           (T, 7)   Action [pos(3), rot(3), gripper]
    - ft_force:         (T, 3)   Force/torque sensor readings (force xyz)
    - ft_force_raw:     (T, 6)   Raw force/torque readings (force + torque)
    - joint_pos:        (T, 7)   Robot joint positions
    - episode_length:   int      Total timesteps in episode
    - success:          bool     Whether task was successful
    - success_threshold: float   Threshold used for success determination
    - wrist_cam_names:  list     Names of all wrist cameras

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (headless mode, 100 episodes)
python scripts_nut/1_collect_data_nut_thread.py --headless --num_episodes 100

# Production run (500 episodes with parallel envs)
CUDA_VISIBLE_DEVICES=2 python scripts_nut/1_collect_data_nut_thread.py \
    --headless --num_episodes 1 \
    --out data/nut_thread.npz

# With custom image size
python scripts_nut/1_collect_data_nut_thread.py --headless --num_episodes 100 \
    --image_width 128 --image_height 128 --out data/nut_thread_128.npz

=============================================================================
"""

from __future__ import annotations

import argparse
import time
import sys

# Disable output buffering for immediate debug output
sys.stdout.reconfigure(line_buffering=True)
print("[DEBUG] Script started, output buffering disabled", flush=True)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect nut threading demonstration data with force sensing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Task and environment configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Forge-NutThread-Direct-v0",
        help="Isaac Lab Gym task ID for FORGE nut threading task.",
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
        default=1,
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
        default=600,
        help="Maximum simulation steps per episode. FORGE nut threading is 30s at 50Hz=1500 steps, "
             "but we use scripted motion which is faster.",
    )
    
    # -----------------------------------------------------------------
    # Expert policy parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--rotation_speed",
        type=float,
        default=0.5,
        help="Angular velocity for threading rotation (rad/s). Positive = tightening direction.",
    )
    parser.add_argument(
        "--downward_force",
        type=float,
        default=5.0,
        help="Target downward force during threading (N).",
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
        default="data/nut_thread.npz",
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
        help="If 1, disables Fabric backend (PhysX GPU acceleration).",
    )

    # -----------------------------------------------------------------
    # Isaac Lab AppLauncher arguments (--headless, --device, etc.)
    # -----------------------------------------------------------------
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    
    # IMPORTANT: Enable cameras for headless mode with image rendering
    args.enable_cameras = True
    
    return args


def compute_camera_quat_from_lookat(eye: tuple, target: tuple, up: tuple = (0, 0, 1)) -> tuple:
    """Compute camera rotation quaternion from eye position and lookat target.
    
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
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Right direction: perpendicular to forward and world-up
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Camera down direction: perpendicular to forward and right
    down = np.cross(forward, right)
    
    # Build rotation matrix from world to camera frame
    rotation_matrix = np.column_stack([right, down, forward])
    
    # Convert rotation matrix to quaternion using scipy
    rot = R.from_matrix(rotation_matrix)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    
    # Isaac Lab expects (w, x, y, z) format
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    
    return (qw, qx, qy, qz)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int):
    """Dynamically add camera sensors to the FORGE environment configuration.
    
    This function modifies env_cfg.scene to add:
    1. A table-view camera - fixed third-person view looking at the workspace
    
    NOTE: Wrist camera cannot be added here for FORGE environments because
    the robot prim doesn't exist yet during scene creation. The wrist camera
    is added after environment creation using add_wrist_camera_post_creation().
    
    Args:
        env_cfg: The environment configuration object.
        image_width: Width of captured images.
        image_height: Height of captured images.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    
    # =========================================================================
    # Table Camera - Third-person fixed view for nut threading task
    # =========================================================================
    # FORGE/Factory workspace is centered at (0, 0, 0), robot base is there
    # Use similar camera position as pick_place task but adjusted for FORGE workspace
    # Place camera further back to capture full workspace including table
    camera_eye = (0.7, 0.2, 0.2)      # Camera position: further back, right side, higher up
    camera_lookat = (0.6, 0.0, 0.1)   # Look at: workspace center
    
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)
    
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
            clipping_range=(0.1, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
            convention="ros",
        ),
    )
    
    # NOTE: Wrist camera is NOT added here - it will be added after env creation
    # using add_wrist_camera_post_creation() because FORGE creates robot prims
    # during scene setup, not before.
    
    # Increase environment spacing to prevent seeing adjacent environments
    # Default is 2.0m in Factory, increase to 5.0m so neighbors are far enough away
    env_cfg.scene.env_spacing = 5.0
    
    # Disable debug visualization markers if present
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'object_pose'):
        env_cfg.commands.object_pose.debug_vis = False
    
    # =========================================================================
    # Extend episode length to prevent auto-reset during data collection
    # =========================================================================
    # Default is 30s at 15Hz = 450 steps. We extend to allow horizon steps.
    # Control freq = 120Hz (sim) / 8 (decimation) = 15Hz
    # Set episode_length_s high enough to cover our horizon
    env_cfg.episode_length_s = 120.0  # 120 seconds = 1800 steps at 15Hz


def add_wrist_camera_post_creation(env, num_envs: int, image_width: int, image_height: int):
    """Add multiple wrist cameras after FORGE environment creation using replicator API.
    
    FORGE environments create the robot prim during scene setup, so we cannot
    add a camera attached to robot links in the env_cfg. Instead, we add the
    camera after the environment is created using USD and replicator APIs.
    
    Creates multiple wrist cameras with different orientations for debugging:
    - wrist_cam_front: Looking forward toward gripper fingers
    - wrist_cam_down: Looking straight down at workspace
    - wrist_cam_front_down: Looking forward and down (45 degree pitch)
    
    Args:
        env: The created FORGE gymnasium environment.
        num_envs: Number of parallel environments.
        image_width: Width of captured images.
        image_height: Height of captured images.
        
    Returns:
        Dict mapping camera name to (render_products, rgb_annotators) for each environment.
    """
    import omni.usd
    from pxr import UsdGeom, Gf
    import omni.replicator.core as rep
    
    print("[DEBUG] add_wrist_camera_post_creation: Starting (multiple cameras)...", flush=True)
    
    stage = omni.usd.get_context().get_stage()
    
    # Define multiple camera configurations
    # Each entry: (name, position, quaternion (w,x,y,z), description)
    # After testing, rotX_180 is the correct orientation for wrist camera
    # Camera convention: -Z is viewing direction
    camera_configs = [
        # Wrist camera: Rotate 180 deg around X - looks forward toward workspace
        ("wrist_cam",
         Gf.Vec3d(0.05, 0.0, 0.0),
         Gf.Quatf(0.0, 1.0, 0.0, 0.0),  # 180 deg around X
         "Wrist camera looking at workspace"),
    ]
    
    all_camera_data = {}
    
    for cam_name, cam_pos, cam_quat, cam_desc in camera_configs:
        render_products = []
        rgb_annotators = []
        
        for env_idx in range(num_envs):
            # Path to panda_hand in this environment
            panda_hand_path = f"/World/envs/env_{env_idx}/Robot/panda_hand"
            panda_hand_prim = stage.GetPrimAtPath(panda_hand_path)
            
            if not panda_hand_prim.IsValid():
                print(f"[ERROR] panda_hand not found at {panda_hand_path}", flush=True)
                render_products.append(None)
                rgb_annotators.append(None)
                continue
            
            # Create camera prim under panda_hand
            wrist_cam_path = f"{panda_hand_path}/{cam_name}"
            
            # Check if camera already exists
            if not stage.GetPrimAtPath(wrist_cam_path).IsValid():
                # Create camera using USD API
                camera_prim = UsdGeom.Camera.Define(stage, wrist_cam_path)
                
                # Set camera properties - moderate FOV for wrist camera
                camera_prim.GetFocalLengthAttr().Set(18.0)  # Longer focal length = narrower FOV
                camera_prim.GetHorizontalApertureAttr().Set(20.955)
                camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 2.0))
                
                # Set local transform (relative to panda_hand)
                xform = UsdGeom.Xformable(camera_prim.GetPrim())
                xform.ClearXformOpOrder()
                
                translate_op = xform.AddTranslateOp()
                translate_op.Set(cam_pos)
                
                orient_op = xform.AddOrientOp()
                orient_op.Set(cam_quat)
            
            # Create render product for this camera
            try:
                render_product = rep.create.render_product(wrist_cam_path, (image_width, image_height))
                
                # Create RGB annotator
                rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annotator.attach([render_product])
                
                render_products.append(render_product)
                rgb_annotators.append(rgb_annotator)
                
                if env_idx == 0:
                    print(f"[DEBUG] {cam_name} created: {cam_desc}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to create {cam_name} for env_{env_idx}: {e}", flush=True)
                render_products.append(None)
                rgb_annotators.append(None)
        
        all_camera_data[cam_name] = (render_products, rgb_annotators)
    
    print(f"[DEBUG] add_wrist_camera_post_creation: Created {len(camera_configs)} camera types", flush=True)
    
    return all_camera_data


def make_forge_env_with_camera(
    task_id: str,
    num_envs: int,
    device: str,
    use_fabric: bool,
    image_width: int,
    image_height: int,
):
    """Create a FORGE environment with added camera sensors.
    
    Args:
        task_id: Gym task id, e.g. "Isaac-Forge-NutThread-Direct-v0".
        num_envs: Number of vectorized env instances.
        device: Torch device string.
        use_fabric: If False, disables Fabric backend.
        image_width: Width of captured images.
        image_height: Height of captured images.
    
    Returns:
        The created gym environment with camera sensor and FORGE force sensing.
    """
    import gymnasium as gym
    
    print("[DEBUG] make_forge_env_with_camera: Importing isaaclab_tasks...")
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    print("[DEBUG] make_forge_env_with_camera: Imports done")
    
    # Parse environment config
    print("[DEBUG] make_forge_env_with_camera: Parsing env config...")
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    print("[DEBUG] make_forge_env_with_camera: Config parsed")
    
    # Add camera to the environment configuration
    print("[DEBUG] make_forge_env_with_camera: Adding camera to env config...", flush=True)
    add_camera_to_env_cfg(env_cfg, image_width, image_height)
    print("[DEBUG] make_forge_env_with_camera: Camera config added", flush=True)
    
    # Debug: print the scene config to verify table camera was added
    print(f"[DEBUG] make_forge_env_with_camera: scene has table_cam: {hasattr(env_cfg.scene, 'table_cam')}", flush=True)
    
    # Create environment
    print("[DEBUG] make_forge_env_with_camera: Creating gym environment...", flush=True)
    import sys
    sys.stdout.flush()
    try:
        env = gym.make(task_id, cfg=env_cfg)
        print("[DEBUG] make_forge_env_with_camera: Gym environment created", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR in gym.make: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # After env creation, check if scene is properly initialized
    print(f"[DEBUG] make_forge_env_with_camera: env type: {type(env)}", flush=True)
    print(f"[DEBUG] make_forge_env_with_camera: env.unwrapped type: {type(env.unwrapped)}", flush=True)
    
    # Check scene sensors (should have table_cam)
    print(f"[DEBUG] make_forge_env_with_camera: Checking scene.sensors...", flush=True)
    try:
        sensors = env.unwrapped.scene.sensors
        print(f"[DEBUG] make_forge_env_with_camera: sensors keys: {list(sensors.keys())}", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR accessing scene.sensors: {e}", flush=True)
    
    # Now add wrist cameras using replicator API (after robot prim exists)
    print("[DEBUG] make_forge_env_with_camera: Adding wrist cameras post-creation...", flush=True)
    wrist_camera_data = add_wrist_camera_post_creation(
        env, num_envs, image_width, image_height
    )
    
    # Store all wrist camera info in env for later access
    # wrist_camera_data is a dict: {cam_name: (render_products, rgb_annotators)}
    env.unwrapped._wrist_camera_data = wrist_camera_data
    
    # For backward compatibility, also store the first camera as default
    first_cam_name = list(wrist_camera_data.keys())[0] if wrist_camera_data else None
    if first_cam_name:
        env.unwrapped._wrist_render_products = wrist_camera_data[first_cam_name][0]
        env.unwrapped._wrist_rgb_annotators = wrist_camera_data[first_cam_name][1]
    
    return env


class NutThreadingExpert:
    """Force-feedback scripted expert for nut threading task.
    
    The expert follows a force-guided strategy for robust threading:
    
    PHASES:
    0. APPROACH: Move down toward bolt until contact detected
    1. SEARCH: Small spiral motion to find thread alignment
    2. ENGAGE: Gentle rotation + downward pressure to catch threads
    3. THREAD: Continuous rotation with force-guided downward motion
    4. DONE: Stopped because torque too high (can't rotate further) or timeout
    
    KEY IMPROVEMENTS over simple time-based control:
    - Force feedback to detect contact and thread engagement
    - Spiral search pattern for alignment correction
    - Adaptive pressure based on force readings
    - Rotation gating based on vertical force (only rotate when engaged)
    
    This handles the FORGE dead zone issue by maintaining consistent
    contact force above the dead zone threshold (~5N).
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        rotation_speed: float = 0.5,
        downward_force: float = 5.0,
        # Force thresholds
        contact_force_threshold: float = 2.0,    # Force to detect initial contact
        engage_force_threshold: float = 6.0,     # Force indicating thread engagement
        max_force_threshold: float = 25.0,       # Max force before backing off
        # Search parameters
        search_radius: float = 0.005,            # Spiral search radius (5mm, increased for faster coverage)
        search_speed: float = 4.0,               # Angular speed of spiral (rad/s, doubled for speed)
        # Threading parameters  
        thread_rotation_speed: float = 1.0,      # Yaw action during threading (max speed)
        thread_downward_action: float = -0.2,    # Z action during threading
        # Torque threshold for completion
        max_torque_threshold: float = 3.0,       # Max Z-torque before stopping (Nm, increased from 1.5)
        # Engagement parameters
        engage_rotation_speed: float = 0.4,      # Rotation during engage (faster than before)
        reverse_rotation_steps: int = 15,        # Steps to reverse-rotate to find thread start
    ):
        """Initialize the nut threading expert with force feedback.
        
        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
            rotation_speed: Angular velocity for threading (rad/s).
            downward_force: Target downward force during threading (N).
            contact_force_threshold: Force threshold to detect contact (N).
            engage_force_threshold: Force threshold indicating thread engagement (N).
            max_force_threshold: Maximum allowed force before easing pressure (N).
            search_radius: Radius of spiral search pattern (m).
            search_speed: Angular speed of spiral search (rad/s).
            thread_rotation_speed: Yaw action value during threading phase.
            thread_downward_action: Z action value during threading phase.
        """
        import torch
        
        self.num_envs = num_envs
        self.device = device
        self.rotation_speed = rotation_speed
        self.downward_force = downward_force
        
        # Force thresholds
        self.contact_force_threshold = contact_force_threshold
        self.engage_force_threshold = engage_force_threshold
        self.max_force_threshold = max_force_threshold
        
        # Search parameters
        self.search_radius = search_radius
        self.search_speed = search_speed
        
        # Threading parameters
        self.thread_rotation_speed = thread_rotation_speed
        self.thread_downward_action = thread_downward_action
        self.max_torque_threshold = max_torque_threshold
        self.engage_rotation_speed = engage_rotation_speed
        self.reverse_rotation_steps = reverse_rotation_steps
        
        # State tracking
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # Phases: 
        #   0=approach, 1=search, 2=engage, 3=thread, 4=done
        #   5=release (let go of nut), 6=reposition (rotate back to -1), 7=regrasp (grab nut again)
        
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Search state
        self.search_angle = torch.zeros(num_envs, device=device)
        self.search_attempts = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.max_search_attempts = 3  # Number of spiral cycles (reduced for speed)
        
        # Engagement tracking
        self.engage_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.engage_success_steps = 10  # Steps of sustained engagement (reduced from 20)
        
        # Position tracking for progress detection
        self.initial_z = torch.zeros(num_envs, device=device)
        self.z_progress = torch.zeros(num_envs, device=device)
        self.z_progress_total = torch.zeros(num_envs, device=device)  # Total Z progress across all cycles
        self.last_z_progress = torch.zeros(num_envs, device=device)  # For stall detection
        self.stall_counter = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.stall_threshold = 30  # Steps without progress before checking (reduced from 50)
        self.min_progress_per_window = 0.001  # 1mm minimum progress in stall_threshold steps (increased from 0.5mm)
        
        # Phase timing limits
        self.approach_timeout = 60     # Max steps in approach (reduced)
        self.search_timeout = 100      # Max steps in search (reduced from 200)
        self.engage_timeout = 60       # Max steps in engage attempt (reduced)
        self.thread_timeout = 400      # Max steps in threading
        self.release_timeout = 30      # Max steps in release phase
        self.reposition_timeout = 60   # Max steps in reposition phase
        self.regrasp_timeout = 60      # Max steps in regrasp phase
        
        # Debug: phase entry timestamps
        self.phase_entry_step = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # IMPORTANT: Cumulative rotation target for FORGE environment
        # FORGE uses target poses, not velocity commands!
        # action[5] maps to yaw angle: [-1,1] -> [-180°, +90°]
        # We need to gradually increase the target to achieve continuous rotation
        self.cumulative_yaw_target = torch.zeros(num_envs, device=device)  # Normalized [-1, 1]
        self.yaw_increment_per_step = 0.05  # How much to increment target per step
        
        # Multi-turn tracking
        self.regrasp_count = torch.zeros(num_envs, dtype=torch.int32, device=device)  # How many times we've regripped
        self.max_regrasp_cycles = 15  # Allow up to 15 regrasp cycles (~15 * 270° = 4050° = ~11 turns)
        self.yaw_threshold_for_regrasp = 0.9  # When to trigger regrasp (close to +1.0 limit)
        
    def reset(self, env_ids=None):
        """Reset expert state for specified environments."""
        import torch
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.step_count[env_ids] = 0
        self.phase[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.search_angle[env_ids] = 0.0
        self.search_attempts[env_ids] = 0
        self.engage_step_count[env_ids] = 0
        self.initial_z[env_ids] = 0.0
        self.z_progress[env_ids] = 0.0
        self.z_progress_total[env_ids] = 0.0
        self.last_z_progress[env_ids] = 0.0
        self.stall_counter[env_ids] = 0
        self.phase_entry_step[env_ids] = 0
        self.cumulative_yaw_target[env_ids] = 0.0  # Reset rotation target
        self.regrasp_count[env_ids] = 0  # Reset regrasp counter
    
    def compute_action(
        self,
        fingertip_pos: "torch.Tensor",
        fingertip_quat: "torch.Tensor",
        fixed_pos: "torch.Tensor",
        force_sensor: "torch.Tensor",
        torque_sensor: "torch.Tensor",
        dt: float,
    ) -> "torch.Tensor":
        """Compute expert action with force feedback control.
        
        The FORGE action space is 7D:
        - pos (3): Position target relative to bolt frame
        - rot (3): Rotation target (roll, pitch, yaw) 
        - gripper (1): Gripper control [-1=closed, +1=open]
        
        Force feedback logic:
        - APPROACH: Move down until force > contact_threshold
        - SEARCH: Spiral pattern while pressing down, looking for thread catch
        - ENGAGE: Small rotation + downward pressure, wait for force > engage_threshold
        - THREAD: Continuous rotation + adaptive downward pressure
        - DONE: Stop when Z-torque exceeds threshold (threading complete or jammed)
        
        Args:
            fingertip_pos: Current fingertip position (num_envs, 3)
            fingertip_quat: Current fingertip quaternion (num_envs, 4)
            fixed_pos: Position of fixed asset (bolt) (num_envs, 3)
            force_sensor: Force sensor readings (num_envs, 3) [Fx, Fy, Fz]
            torque_sensor: Torque sensor readings (num_envs, 3) [Tx, Ty, Tz]
            dt: Physics timestep
            
        Returns:
            Action tensor (num_envs, 7)
        """
        import torch
        import numpy as np
        
        # Initialize action
        action = torch.zeros(self.num_envs, 7, device=self.device)
        
        # Update counters
        self.step_count += 1
        self.phase_step_count += 1
        
        # Extract force components
        # In FORGE, positive Fz typically means upward force (reaction to pressing down)
        fz = force_sensor[:, 2]  # Vertical force
        fxy = torch.norm(force_sensor[:, :2], dim=1)  # Lateral force magnitude
        force_magnitude = torch.norm(force_sensor, dim=1)
        
        # Extract torque components
        tz = torque_sensor[:, 2]  # Z-axis torque (threading resistance)
        
        # Current position relative to bolt
        rel_pos = fingertip_pos - fixed_pos
        current_z = fingertip_pos[:, 2]
        
        # =====================================================================
        # IMPORTANT: Snapshot phase at start of step to prevent cascade transitions
        # This ensures only ONE phase transition per step
        # =====================================================================
        phase_snapshot = self.phase.clone()
        next_phase = self.phase.clone()  # Will be modified, applied at end
        
        # =====================================================================
        # PHASE 0: APPROACH - Move down until contact detected
        # =====================================================================
        approach_mask = phase_snapshot == 0
        if approach_mask.any():
            # Move downward steadily
            action[approach_mask, 0:2] = 0.0  # Stay centered XY
            action[approach_mask, 2] = -0.3   # Move down
            action[approach_mask, 3:6] = 0.0  # No rotation
            
            # Record initial Z when first entering approach
            first_step_mask = approach_mask & (self.phase_step_count == 1)
            if first_step_mask.any():
                self.initial_z[first_step_mask] = current_z[first_step_mask]
            
            # Transition to SEARCH when contact detected
            contact_detected = approach_mask & (force_magnitude > self.contact_force_threshold)
            next_phase = torch.where(contact_detected, torch.full_like(next_phase, 1), next_phase)
            
            # Timeout: move to search anyway after approach_timeout
            timeout = approach_mask & (self.phase_step_count > self.approach_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 1), next_phase)
        
        # =====================================================================
        # PHASE 1: SEARCH - Spiral motion to find thread alignment
        # =====================================================================
        search_mask = phase_snapshot == 1
        if search_mask.any():
            # Reset initial_z when first entering search phase
            first_step_search = search_mask & (self.phase_step_count == 1)
            if first_step_search.any():
                self.initial_z[first_step_search] = current_z[first_step_search]
                self.search_angle[first_step_search] = 0.0
                self.search_attempts[first_step_search] = 0
            
            # Update search angle
            self.search_angle[search_mask] += self.search_speed * dt
            
            # Spiral pattern: increasing radius circles
            # Reset angle when completing a circle
            circle_complete = search_mask & (self.search_angle > 2 * np.pi)
            if circle_complete.any():
                self.search_angle[circle_complete] -= 2 * np.pi
                self.search_attempts[circle_complete] += 1
            
            # Compute spiral offset (XY)
            progress = self.search_attempts.float() / self.max_search_attempts
            current_radius = self.search_radius * (0.5 + 0.5 * progress)  # Grow radius
            
            spiral_x = current_radius * torch.cos(self.search_angle)
            spiral_y = current_radius * torch.sin(self.search_angle)
            
            # Action: spiral XY + stronger downward pressure
            action[search_mask, 0] = spiral_x[search_mask] * 15.0  # Scale to action space (increased)
            action[search_mask, 1] = spiral_y[search_mask] * 15.0
            action[search_mask, 2] = -0.4   # Stronger downward pressure (increased from -0.25)
            
            # Larger rotation oscillation to help find thread
            # Also include reverse rotation trick during search
            rot_osc = 0.3 * torch.sin(self.search_angle * 2)  # Larger amplitude, slower frequency
            action[search_mask, 5] = rot_osc[search_mask]
            
            # Transition to ENGAGE if we detect significant vertical force
            # (indicating thread tips are touching)
            engage_ready = search_mask & (fz.abs() > self.engage_force_threshold * 0.5)
            next_phase = torch.where(engage_ready, torch.full_like(next_phase, 2), next_phase)
            
            # Also transition if Z has dropped (thread starting to catch)
            # Use initial_z set at SEARCH phase entry, not APPROACH phase
            z_dropped = search_mask & ((self.initial_z - current_z) > 0.002)
            next_phase = torch.where(z_dropped, torch.full_like(next_phase, 2), next_phase)
            
            # Timeout: force transition to engage
            timeout = search_mask & (self.phase_step_count > self.search_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 2), next_phase)
        
        # =====================================================================
        # PHASE 2: ENGAGE - Try to catch threads with rotation + pressure
        # =====================================================================
        engage_mask = phase_snapshot == 2
        if engage_mask.any():
            # Reset engage counter on phase entry
            first_step = engage_mask & (self.phase_step_count == 1)
            if first_step.any():
                self.engage_step_count[first_step] = 0
                self.initial_z[first_step] = current_z[first_step]
            
            # Center on bolt + downward pressure + rotation
            action[engage_mask, 0:2] = 0.0  # Center XY
            action[engage_mask, 2] = -0.4   # Firm downward pressure (increased)
            
            # TRICK: Reverse-then-forward rotation to find thread start
            # First few steps: rotate backwards (CCW) to find thread engagement point
            # Then: rotate forwards (CW) to catch and start threading
            # FORGE uses TARGET poses, so we increment cumulative target
            is_reverse_phase = self.phase_step_count <= self.reverse_rotation_steps
            yaw_delta = torch.where(
                is_reverse_phase,
                torch.full((self.num_envs,), -self.yaw_increment_per_step * 2, device=self.device),  # Reverse faster
                torch.full((self.num_envs,), self.yaw_increment_per_step, device=self.device)       # Forward
            )
            self.cumulative_yaw_target = torch.where(
                engage_mask,
                (self.cumulative_yaw_target + yaw_delta).clamp(-1.0, 1.0),
                self.cumulative_yaw_target
            )
            action[engage_mask, 5] = self.cumulative_yaw_target[engage_mask]
            
            # Check for successful engagement:
            # 1. Sustained force above threshold
            # 2. Z position is dropping (nut is threading down)
            is_engaged = fz.abs() > self.engage_force_threshold
            z_progress_engage = self.initial_z - current_z
            is_progressing = z_progress_engage > 0.001  # At least 1mm progress
            
            # Count consecutive engaged steps
            self.engage_step_count = torch.where(
                engage_mask & is_engaged,
                self.engage_step_count + 1,
                torch.zeros_like(self.engage_step_count)
            )
            
            # Transition to THREAD when engagement confirmed
            confirmed_engage = engage_mask & (self.engage_step_count > self.engage_success_steps)
            next_phase = torch.where(confirmed_engage, torch.full_like(next_phase, 3), next_phase)
            
            # Also transition if good progress regardless of force
            good_progress = engage_mask & (z_progress_engage > 0.003)  # 3mm drop
            next_phase = torch.where(good_progress, torch.full_like(next_phase, 3), next_phase)
            
            # If force too high without progress, back off to search
            high_force_stuck = engage_mask & (fz.abs() > self.max_force_threshold) & (~is_progressing)
            next_phase = torch.where(high_force_stuck, torch.full_like(next_phase, 1), next_phase)
            
            # Timeout
            timeout = engage_mask & (self.phase_step_count > self.engage_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 3), next_phase)
        
        # =====================================================================
        # PHASE 3: THREAD - Continuous rotation with adaptive pressure
        # =====================================================================
        thread_mask = phase_snapshot == 3
        if thread_mask.any():
            # Record starting Z on phase entry
            first_step = thread_mask & (self.phase_step_count == 1)
            if first_step.any():
                self.initial_z[first_step] = current_z[first_step]
                self.last_z_progress[first_step] = 0.0
                self.stall_counter[first_step] = 0
            
            # Track threading progress (only for envs in thread phase)
            z_progress_thread = self.initial_z - current_z
            # Only update z_progress for thread phase envs
            self.z_progress = torch.where(thread_mask, z_progress_thread, self.z_progress)
            
            # Stall detection: check if progress has stalled
            # Every stall_threshold steps, check if we made minimum progress
            check_stall = thread_mask & (self.phase_step_count % self.stall_threshold == 0) & (self.phase_step_count > 0)
            if check_stall.any():
                progress_delta = self.z_progress - self.last_z_progress
                is_stalled = check_stall & (progress_delta < self.min_progress_per_window)
                # If stalled and force is low (not engaged), go back to SEARCH
                stall_and_low_force = is_stalled & (fz.abs() < self.engage_force_threshold * 0.8)
                next_phase = torch.where(stall_and_low_force, torch.full_like(next_phase, 1), next_phase)
                # Update last_z_progress for next check
                self.last_z_progress = torch.where(check_stall, self.z_progress.clone(), self.last_z_progress)
            
            # Adaptive downward pressure based on force
            # If force is low, press harder; if force is high, ease up
            force_ratio = fz.abs() / self.engage_force_threshold
            
            # Base action: rotate + downward with small XY wobble to help find threads
            # Small oscillation helps threads catch if slightly misaligned
            wobble_freq = 2.0  # Hz
            wobble_amp = 0.02  # Small amplitude
            wobble_phase = self.phase_step_count.float() * wobble_freq * dt * 2 * 3.14159
            action[thread_mask, 0] = wobble_amp * torch.sin(wobble_phase)[thread_mask]
            action[thread_mask, 1] = wobble_amp * torch.cos(wobble_phase)[thread_mask]
            
            # Adaptive Z action: MUCH stronger pressure to ensure thread engagement
            # Real threading requires significant downward force to overcome thread resistance
            z_action = torch.where(
                force_ratio < 0.5,
                torch.full_like(fz, -0.8),    # Low force: press VERY hard
                torch.where(
                    force_ratio < 1.5,
                    torch.full_like(fz, -0.6),  # Normal force: still press hard
                    torch.full_like(fz, -0.3)   # High force: moderate pressure
                )
            )
            action[thread_mask, 2] = z_action[thread_mask]
            
            # Check if force is too high BEFORE updating rotation target
            too_high = thread_mask & (fz.abs() > self.max_force_threshold)
            should_rotate = thread_mask & ~too_high
            
            # Rotation: FORGE uses TARGET poses, not velocity!
            # We need to increment the target yaw gradually to achieve continuous rotation
            # FORGE maps action[5] from [-1, 1] to [-180°, +90°] yaw angle
            # Only increment if not in too_high state
            self.cumulative_yaw_target = torch.where(
                should_rotate,
                (self.cumulative_yaw_target + self.yaw_increment_per_step).clamp(-1.0, 1.0),
                self.cumulative_yaw_target  # Don't increment if force too high
            )
            action[thread_mask, 5] = self.cumulative_yaw_target[thread_mask]
            
            # If force gets too high, reduce downward pressure
            if too_high.any():
                action[too_high, 2] = -0.1  # Light pressure
            
            # =========== MULTI-TURN REGRASP LOGIC ===========
            # When yaw target reaches threshold, trigger release-reposition-regrasp cycle
            # This allows unlimited rotation (10+ turns) within FORGE's limited action space
            needs_regrasp = thread_mask & (self.cumulative_yaw_target >= self.yaw_threshold_for_regrasp) & (self.regrasp_count < self.max_regrasp_cycles)
            if needs_regrasp.any():
                # Accumulate total z progress before releasing
                self.z_progress_total = torch.where(needs_regrasp, 
                                                     self.z_progress_total + self.z_progress,
                                                     self.z_progress_total)
                # Transition to RELEASE phase
                next_phase = torch.where(needs_regrasp, torch.full_like(next_phase, 5), next_phase)
            # =========== END MULTI-TURN REGRASP LOGIC ===========
            
            # Transition to DONE if torque too high AND we have made some progress
            # (High torque alone is not enough - could just be initial contact resistance)
            min_progress_for_torque_done = 0.003  # At least 3mm progress required
            total_progress = self.z_progress_total + self.z_progress
            high_torque = thread_mask & (tz.abs() > self.max_torque_threshold) & (total_progress > min_progress_for_torque_done)
            next_phase = torch.where(high_torque, torch.full_like(next_phase, 4), next_phase)
            
            # Also transition to DONE if significant TOTAL progress made
            # With regrasp, we can achieve much more than 12mm
            target_total_progress = 0.040  # 40mm = ~10 turns on M8x1.25 thread
            done = thread_mask & (total_progress > target_total_progress)
            next_phase = torch.where(done, torch.full_like(next_phase, 4), next_phase)
            
            # Timeout
            timeout = thread_mask & (self.phase_step_count > self.thread_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 4), next_phase)
        
        # =====================================================================
        # PHASE 5: RELEASE - Lift up and open gripper to release nut
        # =====================================================================
        release_mask = phase_snapshot == 5
        if release_mask.any():
            # First step: record that we're starting a regrasp cycle
            first_step = release_mask & (self.phase_step_count == 1)
            if first_step.any():
                self.regrasp_count[first_step] += 1
            
            # Lift up to create clearance, maintain current yaw
            action[release_mask, 0:2] = 0.0  # Stay centered XY
            action[release_mask, 2] = 0.5    # Move UP
            action[release_mask, 3:5] = 0.0  # No roll/pitch change
            action[release_mask, 5] = self.cumulative_yaw_target[release_mask]  # Keep yaw
            action[release_mask, 6] = 1.0    # OPEN GRIPPER (release nut)
            
            # After some steps, transition to REPOSITION
            lifted_enough = release_mask & (self.phase_step_count > self.release_timeout)
            next_phase = torch.where(lifted_enough, torch.full_like(next_phase, 6), next_phase)
        
        # =====================================================================
        # PHASE 6: REPOSITION - Rotate yaw back to starting position (-1.0)
        # =====================================================================
        reposition_mask = phase_snapshot == 6
        if reposition_mask.any():
            # Keep lifted, rotate yaw back towards -1.0 (starting position)
            action[reposition_mask, 0:2] = 0.0  # Stay centered XY
            action[reposition_mask, 2] = 0.3    # Stay lifted
            action[reposition_mask, 3:5] = 0.0  # No roll/pitch change
            action[reposition_mask, 6] = 1.0    # Keep gripper OPEN
            
            # Decrement yaw target back towards -0.8 (leave some margin from -1.0)
            # Use larger steps for faster repositioning
            reposition_yaw_step = 0.08
            self.cumulative_yaw_target = torch.where(
                reposition_mask,
                (self.cumulative_yaw_target - reposition_yaw_step).clamp(-0.8, 1.0),
                self.cumulative_yaw_target
            )
            action[reposition_mask, 5] = self.cumulative_yaw_target[reposition_mask]
            
            # Transition to REGRASP when yaw is back near starting position
            yaw_repositioned = reposition_mask & (self.cumulative_yaw_target <= -0.75)
            next_phase = torch.where(yaw_repositioned, torch.full_like(next_phase, 7), next_phase)
            
            # Timeout: force transition
            timeout = reposition_mask & (self.phase_step_count > self.reposition_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 7), next_phase)
        
        # =====================================================================
        # PHASE 7: REGRASP - Descend and close gripper to regrasp nut
        # =====================================================================
        regrasp_mask = phase_snapshot == 7
        if regrasp_mask.any():
            # Descend towards nut
            action[regrasp_mask, 0:2] = 0.0  # Stay centered XY
            action[regrasp_mask, 3:5] = 0.0  # No roll/pitch change
            action[regrasp_mask, 5] = self.cumulative_yaw_target[regrasp_mask]  # Keep yaw
            
            # Two sub-phases: descend, then close gripper
            descend_steps = self.regrasp_timeout // 2
            is_descending = self.phase_step_count <= descend_steps
            
            # Descend phase: move down with open gripper
            action[regrasp_mask, 2] = torch.where(
                is_descending[regrasp_mask],
                torch.full((regrasp_mask.sum(),), -0.4, device=self.device),
                torch.full((regrasp_mask.sum(),), -0.2, device=self.device)  # Lighter when closing
            )
            action[regrasp_mask, 6] = torch.where(
                is_descending[regrasp_mask],
                torch.full((regrasp_mask.sum(),), 1.0, device=self.device),   # Open during descend
                torch.full((regrasp_mask.sum(),), -1.0, device=self.device)   # Close to grasp
            )
            
            # Transition back to THREAD after completing regrasp
            regrasp_done = regrasp_mask & (self.phase_step_count > self.regrasp_timeout)
            if regrasp_done.any():
                # Reset z_progress for the new threading cycle (keeping z_progress_total)
                self.z_progress[regrasp_done] = 0.0
                self.initial_z[regrasp_done] = current_z[regrasp_done]
            next_phase = torch.where(regrasp_done, torch.full_like(next_phase, 3), next_phase)
        
        # =====================================================================
        # PHASE 4: DONE - Maintain position
        # =====================================================================
        done_mask = phase_snapshot == 4
        if done_mask.any():
            action[done_mask, 0:6] = 0.0  # Hold position
        
        # =====================================================================
        # Apply phase transitions (only one transition per step)
        # =====================================================================
        phase_changed = next_phase != phase_snapshot
        if phase_changed.any():
            self.phase = next_phase
            # Reset phase step count for environments that changed phase
            self.phase_step_count = torch.where(phase_changed, 
                                                 torch.zeros_like(self.phase_step_count),
                                                 self.phase_step_count)
            self.phase_entry_step = torch.where(phase_changed,
                                                 self.step_count,
                                                 self.phase_entry_step)
        
        # =====================================================================
        # Gripper control: default CLOSED for phases 0-4
        # Phases 5-7 (RELEASE, REPOSITION, REGRASP) already set action[:, 6]
        # =====================================================================
        default_gripper_mask = phase_snapshot <= 4
        action[default_gripper_mask, 6] = -1.0  # Gripper closed (grasp nut)
        
        return action
    
    def is_done(self) -> "torch.Tensor":
        """Check which environments have completed the task."""
        # Only phase 4 (DONE) is truly done
        # Phases 5-7 (RELEASE, REPOSITION, REGRASP) are part of multi-turn cycling
        return self.phase == 4
    
    def get_phase_names(self) -> list:
        """Return human-readable phase names for debugging."""
        return ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE", "RELEASE", "REPOSITION", "REGRASP"]
    
    def get_debug_info(self) -> dict:
        """Return debug information for monitoring."""
        return {
            "phase": self.phase.cpu().numpy(),
            "step_count": self.step_count.cpu().numpy(),
            "phase_step_count": self.phase_step_count.cpu().numpy(),
            "search_attempts": self.search_attempts.cpu().numpy(),
            "engage_step_count": self.engage_step_count.cpu().numpy(),
            "z_progress": self.z_progress.cpu().numpy(),
            "z_progress_total": self.z_progress_total.cpu().numpy(),
            "cumulative_yaw_target": self.cumulative_yaw_target.cpu().numpy(),
            "regrasp_count": self.regrasp_count.cpu().numpy(),
        }


def rollout_nut_threading(
    env,
    expert: NutThreadingExpert,
    horizon: int,
):
    """Run parallel rollouts for nut threading and collect data.
    
    Args:
        env: FORGE gymnasium environment with camera and force sensors.
        expert: NutThreadingExpert instance.
        horizon: Maximum steps for each episode.
    
    Returns:
        List of episode dictionaries, one per environment.
    """
    import numpy as np
    import torch
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    physics_dt = env.unwrapped.physics_dt
    
    print(f"[DEBUG] rollout: device={device}, num_envs={num_envs}, dt={physics_dt}")
    
    # Get camera sensor references
    print("[DEBUG] rollout: Getting camera sensor references...")
    try:
        table_camera = env.unwrapped.scene.sensors["table_cam"]
        print(f"[DEBUG] rollout: table_camera acquired: {type(table_camera)}")
    except Exception as e:
        print(f"[DEBUG] ERROR getting table_cam: {e}")
        raise
    
    # Get wrist camera data (multiple cameras added post-creation using replicator API)
    wrist_camera_data = getattr(env.unwrapped, '_wrist_camera_data', None)
    if wrist_camera_data is not None:
        print(f"[DEBUG] rollout: wrist cameras acquired: {list(wrist_camera_data.keys())}")
    else:
        print("[DEBUG] rollout: wrist_camera_data not available, using zeros for wrist images")
    print(f"[DEBUG] rollout: Cameras acquired")
    
    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    print("[DEBUG] rollout: Resetting environment...")
    obs_dict, _ = env.reset()
    print(f"[DEBUG] rollout: obs_dict type: {type(obs_dict)}")
    if isinstance(obs_dict, dict):
        print(f"[DEBUG] rollout: obs_dict keys: {obs_dict.keys()}")
    expert.reset()
    print(f"[DEBUG] rollout: Reset done")
    
    # Get the underlying FORGE env
    forge_env = env.unwrapped
    
    # =========================================================================
    # Step 2: Initialize per-env recording buffers
    # =========================================================================
    obs_lists = [[] for _ in range(num_envs)]
    state_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    # Create separate lists for each wrist camera
    wrist_cam_names = list(wrist_camera_data.keys()) if wrist_camera_data else []
    wrist_image_lists_dict = {cam_name: [[] for _ in range(num_envs)] for cam_name in wrist_cam_names}
    ee_pose_lists = [[] for _ in range(num_envs)]
    nut_pose_lists = [[] for _ in range(num_envs)]
    bolt_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    ft_force_lists = [[] for _ in range(num_envs)]
    ft_force_raw_lists = [[] for _ in range(num_envs)]
    joint_pos_lists = [[] for _ in range(num_envs)]
    phase_lists = [[] for _ in range(num_envs)]  # Expert state machine phase
    
    # =========================================================================
    # Step 3: Run episode and record data
    # =========================================================================
    print(f"[DEBUG] rollout: Starting main loop (horizon={horizon})...")
    
    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")
        
        # Get observations from environment
        if isinstance(obs_dict, dict):
            policy_obs = obs_dict.get("policy", None)
            critic_state = obs_dict.get("critic", None)
            if policy_obs is None:
                policy_obs = next(iter(obs_dict.values()))
            if critic_state is None:
                critic_state = policy_obs
        else:
            policy_obs = obs_dict
            critic_state = obs_dict
        
        # Get camera images
        if t == 0:
            print(f"[DEBUG] rollout: Getting camera images...")
            print(f"[DEBUG] rollout: table_camera.data.output keys: {table_camera.data.output.keys()}")
        table_rgb = table_camera.data.output["rgb"]
        
        # Get wrist camera images from all cameras
        wrist_images_by_cam = {}
        if wrist_camera_data is not None:
            for cam_name, (render_products, rgb_annotators) in wrist_camera_data.items():
                wrist_images_list = []
                for i, annotator in enumerate(rgb_annotators):
                    if annotator is not None:
                        data = annotator.get_data()
                        if data is not None:
                            # Convert to torch tensor if needed
                            if not isinstance(data, torch.Tensor):
                                data = torch.from_numpy(data).to(device)
                            wrist_images_list.append(data)
                        else:
                            wrist_images_list.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                    else:
                        wrist_images_list.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                wrist_images_by_cam[cam_name] = torch.stack(wrist_images_list, dim=0)
        
        if t == 0:
            print(f"[DEBUG] rollout: table_rgb shape: {table_rgb.shape}, dtype: {table_rgb.dtype}")
            for cam_name, wrist_rgb in wrist_images_by_cam.items():
                print(f"[DEBUG] rollout: {cam_name} shape: {wrist_rgb.shape}, dtype: {wrist_rgb.dtype}")
        
        # Truncate to 3 channels if RGBA
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        for cam_name in wrist_images_by_cam:
            if wrist_images_by_cam[cam_name].shape[-1] > 3:
                wrist_images_by_cam[cam_name] = wrist_images_by_cam[cam_name][..., :3]
        
        # Get poses and force data from FORGE environment
        # These are internal tensors maintained by ForgeEnv
        if t == 0:
            print(f"[DEBUG] rollout: Getting fingertip_pos...")
            print(f"[DEBUG] rollout: forge_env attributes: fingertip_midpoint_pos={hasattr(forge_env, 'fingertip_midpoint_pos')}")
        fingertip_pos = forge_env.fingertip_midpoint_pos.clone()
        fingertip_quat = forge_env.fingertip_midpoint_quat.clone()
        if t == 0:
            print(f"[DEBUG] rollout: fingertip_pos shape: {fingertip_pos.shape}")
        
        # Held asset (nut) pose
        if t == 0:
            print(f"[DEBUG] rollout: Getting held_asset (nut) pose...")
            print(f"[DEBUG] rollout: _held_asset exists: {hasattr(forge_env, '_held_asset')}")
        nut_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
        nut_quat = forge_env._held_asset.data.root_quat_w
        nut_pose = torch.cat([nut_pos, nut_quat], dim=-1)
        if t == 0:
            print(f"[DEBUG] rollout: nut_pose shape: {nut_pose.shape}")
        
        # Fixed asset (bolt) pose  
        if t == 0:
            print(f"[DEBUG] rollout: Getting fixed_asset (bolt) pose...")
            print(f"[DEBUG] rollout: _fixed_asset exists: {hasattr(forge_env, '_fixed_asset')}")
        bolt_pos = forge_env._fixed_asset.data.root_pos_w - forge_env.scene.env_origins
        bolt_quat = forge_env._fixed_asset.data.root_quat_w
        bolt_pose = torch.cat([bolt_pos, bolt_quat], dim=-1)
        if t == 0:
            print(f"[DEBUG] rollout: bolt_pose shape: {bolt_pose.shape}")
        
        # Get force sensor data - this is the key FORGE feature!
        # force_sensor_smooth contains smoothed force/torque readings (6D: Fx, Fy, Fz, Tx, Ty, Tz)
        if t == 0:
            print(f"[DEBUG] rollout: Checking force_sensor_smooth...")
            print(f"[DEBUG] rollout: force_sensor_smooth exists: {hasattr(forge_env, 'force_sensor_smooth')}")
        if hasattr(forge_env, 'force_sensor_smooth'):
            ft_force_raw = forge_env.force_sensor_smooth.clone()
            ft_force = ft_force_raw[:, :3]  # Just the force components
            if t == 0:
                print(f"[DEBUG] rollout: ft_force_raw shape: {ft_force_raw.shape}")
        else:
            # Fallback if force sensor not available
            if t == 0:
                print(f"[DEBUG] rollout: WARNING - force_sensor_smooth not found, using zeros")
            ft_force_raw = torch.zeros(num_envs, 6, device=device)
            ft_force = torch.zeros(num_envs, 3, device=device)
        
        # Joint positions
        joint_pos = forge_env._robot.data.joint_pos[:, :7]
        
        # EE pose
        ee_pose = torch.cat([fingertip_pos, fingertip_quat], dim=-1)
        
        # Compute expert action
        # Extract torque from raw sensor data (indices 3:6 are Tx, Ty, Tz)
        ft_torque = ft_force_raw[:, 3:6]
        action = expert.compute_action(
            fingertip_pos=fingertip_pos,
            fingertip_quat=fingertip_quat,
            fixed_pos=bolt_pos,
            force_sensor=ft_force,
            torque_sensor=ft_torque,
            dt=physics_dt,
        )
        
        # Record data for each env
        policy_obs_np = policy_obs.cpu().numpy()
        critic_state_np = critic_state.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        # Convert all wrist camera images to numpy
        wrist_images_np_dict = {
            cam_name: wrist_images_by_cam[cam_name].cpu().numpy().astype(np.uint8)
            for cam_name in wrist_images_by_cam
        }
        # Get current phase from expert
        phase_np = expert.phase.cpu().numpy().copy()
        _ = {
        }
        ee_pose_np = ee_pose.cpu().numpy()
        nut_pose_np = nut_pose.cpu().numpy()
        bolt_pose_np = bolt_pose.cpu().numpy()
        action_np = action.cpu().numpy()
        ft_force_np = ft_force.cpu().numpy()
        ft_force_raw_np = ft_force_raw.cpu().numpy()
        joint_pos_np = joint_pos.cpu().numpy()
        
        for i in range(num_envs):
            obs_lists[i].append(policy_obs_np[i])
            state_lists[i].append(critic_state_np[i])
            image_lists[i].append(table_images_np[i])
            # Append each wrist camera image
            for cam_name in wrist_cam_names:
                wrist_image_lists_dict[cam_name][i].append(wrist_images_np_dict[cam_name][i])
            ee_pose_lists[i].append(ee_pose_np[i])
            nut_pose_lists[i].append(nut_pose_np[i])
            bolt_pose_lists[i].append(bolt_pose_np[i])
            action_lists[i].append(action_np[i])
            ft_force_lists[i].append(ft_force_np[i])
            ft_force_raw_lists[i].append(ft_force_raw_np[i])
            joint_pos_lists[i].append(joint_pos_np[i])
            phase_lists[i].append(phase_np[i])
        
        # Step environment
        if t == 0:
            print(f"[DEBUG] rollout: Calling env.step...")
            print(f"[DEBUG] rollout: action shape: {action.shape}, dtype: {action.dtype}")
        obs_dict, reward, terminated, truncated, info = env.step(action)
        if t == 0:
            print(f"[DEBUG] rollout: env.step returned successfully")
            print(f"[DEBUG] rollout: reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        
        # Debug: Print state machine status periodically
        if (t + 1) % 1 == 0:
            phase_names = expert.get_phase_names()
            debug_info = expert.get_debug_info()
            for i in range(min(num_envs, 2)):  # Only print first 2 envs
                phase_idx = int(debug_info["phase"][i])
                phase_name = phase_names[phase_idx] if phase_idx < len(phase_names) else "UNKNOWN"
                fz_val = ft_force_np[i, 2]
                tz_val = ft_force_raw_np[i, 5]  # Z-axis torque
                ty_val = ft_force_raw_np[i, 4]  # Y-axis torque
                z_prog = debug_info["z_progress"][i] * 1000  # mm
                z_prog_total = debug_info["z_progress_total"][i] * 1000  # mm
                yaw_target = debug_info["cumulative_yaw_target"][i]
                regrasp_cnt = debug_info["regrasp_count"][i]
                # Also print actual action values for debugging
                act_z = action_np[i, 2]
                act_yaw = action_np[i, 5]
                print(f"[DEBUG] t={t+1} env{i}: phase={phase_name}, Fz={fz_val:.2f}N, Ty={ty_val:.3f}Nm, "
                      f"z_prog={z_prog:.2f}mm, z_tot={z_prog_total:.2f}mm, yaw_tgt={yaw_target:.3f}, regrasp#{regrasp_cnt}")
        
        # Check for done status (but don't reset - we disabled auto-reset by extending episode length)
        done = terminated | truncated
        if done.any():
            print(f"[DEBUG] Step {t+1}: {done.sum().item()} envs signaled done (but continuing)")
    
    print("[DEBUG] rollout: Main loop finished")
    
    # =========================================================================
    # Step 4: Build episode dictionaries
    # =========================================================================
    print("[DEBUG] rollout: Building episode dicts...")
    results = []
    
    # Check success using FORGE's success metric
    success_threshold = forge_env.cfg_task.success_threshold
    
    for i in range(num_envs):
        # Determine success based on final state
        # In FORGE, success is based on how much the nut has been threaded onto the bolt
        episode_dict = {
            "obs": np.array(obs_lists[i], dtype=np.float32),
            "state": np.array(state_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),
            # Save all wrist cameras with their names as keys
            **{f"wrist_{cam_name}": np.array(wrist_image_lists_dict[cam_name][i], dtype=np.uint8) 
               for cam_name in wrist_cam_names},
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "nut_pose": np.array(nut_pose_lists[i], dtype=np.float32),
            "bolt_pose": np.array(bolt_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "ft_force": np.array(ft_force_lists[i], dtype=np.float32),
            "ft_force_raw": np.array(ft_force_raw_lists[i], dtype=np.float32),
            "joint_pos": np.array(joint_pos_lists[i], dtype=np.float32),
            "phase": np.array(phase_lists[i], dtype=np.int32),  # Expert state machine phase
            "phase_names": ["APPROACH", "SEARCH", "ENGAGE", "THREAD", "DONE"],  # Phase name mapping
            "episode_length": len(obs_lists[i]),
            "success": False,  # Will be determined by inspection
            "success_threshold": success_threshold,
            "wrist_cam_names": wrist_cam_names,  # Store camera names for reference
        }
        results.append(episode_dict)
    
    print(f"[DEBUG] rollout: Done. Returning {len(results)} episodes")
    return results


def save_episodes(path: str, episodes: list[dict]) -> None:
    """Save episodes to an NPZ file.
    
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
    print(f"Saved {len(episodes)} episodes to {path}")
    
    # Print some statistics
    if episodes:
        ep0 = episodes[0]
        print(f"  - Policy obs shape: {ep0['obs'].shape}")
        print(f"  - State shape: {ep0['state'].shape}")
        print(f"  - Table camera image shape: {ep0['images'].shape}")
        # Print all wrist camera shapes
        wrist_cam_names = ep0.get('wrist_cam_names', [])
        for cam_name in wrist_cam_names:
            key = f"wrist_{cam_name}"
            if key in ep0:
                print(f"  - {key} image shape: {ep0[key].shape}")
        print(f"  - EE pose shape: {ep0['ee_pose'].shape}")
        print(f"  - Action shape: {ep0['action'].shape}")
        print(f"  - Force sensor shape: {ep0['ft_force'].shape}")
        print(f"  - Force sensor raw shape: {ep0['ft_force_raw'].shape}")
        print(f"  - Joint pos shape: {ep0['joint_pos'].shape}")
        print(f"  - Episode length: {ep0['episode_length']}")
        
        # Show force statistics
        all_forces = np.concatenate([ep['ft_force'] for ep in episodes], axis=0)
        print(f"  - Force stats: min={all_forces.min():.2f}, max={all_forces.max():.2f}, "
              f"mean={all_forces.mean():.2f}, std={all_forces.std():.2f}")


def main() -> None:
    """Main entry point for collecting nut threading demonstration data."""
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
    import torch
    
    try:
        # =================================================================
        # Step 3: Set random seeds for reproducibility
        # =================================================================
        print("[DEBUG] Step 3: Setting random seeds...")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print("[DEBUG] Step 3: Done setting seeds")
        
        # =================================================================
        # Step 4: Create FORGE environment with camera
        # =================================================================
        print("[DEBUG] Step 4: Creating FORGE environment with camera...")
        print(f"[DEBUG]   task_id={args.task}")
        print(f"[DEBUG]   num_envs={args.num_envs}")
        print(f"[DEBUG]   device={args.device}")
        print(f"[DEBUG]   image_size={args.image_width}x{args.image_height}")
        
        num_envs = args.num_envs
        env = make_forge_env_with_camera(
            task_id=args.task,
            num_envs=num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
        )
        print("[DEBUG] Step 4: Environment created successfully")
        
        print("[DEBUG] Getting env.unwrapped.device...")
        device = env.unwrapped.device
        print(f"[DEBUG] Environment device: {device}")
        
        # Print FORGE environment info
        print("[DEBUG] Getting forge_env = env.unwrapped...")
        forge_env = env.unwrapped
        print(f"[DEBUG] forge_env type: {type(forge_env)}")
        
        print("[DEBUG] Accessing forge_env.cfg_task...")
        print(f"[DEBUG] FORGE task name: {forge_env.cfg_task.name}")
        
        print("[DEBUG] Accessing forge_env.cfg.obs_order...")
        print(f"[DEBUG] FORGE obs order: {forge_env.cfg.obs_order}")
        print(f"[DEBUG] FORGE state order: {forge_env.cfg.state_order}")
        
        print("[DEBUG] Accessing action/observation space...")
        print(f"[DEBUG] FORGE action space: {env.action_space}")
        print(f"[DEBUG] FORGE observation space: {env.observation_space}")
        
        # Check camera sensor
        print("[DEBUG] Checking camera sensors...")
        print(f"[DEBUG] Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        if "table_cam" in env.unwrapped.scene.sensors:
            camera = env.unwrapped.scene.sensors["table_cam"]
            print(f"[DEBUG] Table camera found: {camera.cfg.width}x{camera.cfg.height}")
            print(f"[DEBUG] Table camera data types: {camera.cfg.data_types}")
        else:
            print("[DEBUG] WARNING: table_cam NOT found in sensors!")
        
        # Check wrist cameras (added post-creation via replicator API)
        wrist_camera_data = getattr(env.unwrapped, '_wrist_camera_data', None)
        if wrist_camera_data is not None:
            print(f"[DEBUG] Wrist cameras found: {list(wrist_camera_data.keys())}")
            for cam_name, (render_products, rgb_annotators) in wrist_camera_data.items():
                valid_count = len([a for a in rgb_annotators if a is not None])
                print(f"[DEBUG]   - {cam_name}: {valid_count} valid annotators")
        else:
            print("[DEBUG] WARNING: wrist camera data NOT found!")
        
        # =================================================================
        # Step 5: Create expert policy
        # =================================================================
        print("[DEBUG] Step 5: Creating nut threading expert...")
        expert = NutThreadingExpert(
            num_envs=num_envs,
            device=device,
            rotation_speed=args.rotation_speed,
            downward_force=args.downward_force,
        )
        print("[DEBUG] Step 5: Expert created")
        
        # =================================================================
        # Step 6: Data collection loop
        # =================================================================
        episodes = []
        
        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} nut threading episodes with FORCE SENSING")
        print(f"Settings:")
        print(f"  - num_envs (parallel): {num_envs}")
        print(f"  - horizon: {args.horizon}")
        print(f"  - image_size: {args.image_width}x{args.image_height}")
        print(f"  - rotation_speed: {args.rotation_speed} rad/s")
        print(f"  - downward_force target: {args.downward_force} N")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        batch_count = 0
        
        while len(episodes) < args.num_episodes:
            batch_count += 1
            
            # Run parallel episodes
            results = rollout_nut_threading(
                env=env,
                expert=expert,
                horizon=args.horizon,
            )
            
            # Add results to episode list
            for episode_dict in results:
                episodes.append(episode_dict)
                if len(episodes) >= args.num_episodes:
                    break
            
            # Print progress
            elapsed = time.time() - start_time
            rate = len(episodes) / elapsed if elapsed > 0 else 0
            print(
                f"Batch {batch_count:3d} ({num_envs} envs) | "
                f"Collected: {len(episodes)}/{args.num_episodes} | "
                f"Rate: {rate:.1f} ep/s"
            )
        
        # =================================================================
        # Step 7: Print summary statistics
        # =================================================================
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Parallel envs: {num_envs}")
        print(f"Total batches: {batch_count}")
        print(f"Collected episodes: {len(episodes)}")
        print(f"Rate: {len(episodes)/elapsed:.2f} episodes/s")
        print(f"{'='*60}\n")
        
        # =================================================================
        # Step 8: Save collected episodes to NPZ file
        # =================================================================
        episodes = episodes[:args.num_episodes]
        save_episodes(args.out, episodes)
        
        # Clean up environment
        env.close()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
