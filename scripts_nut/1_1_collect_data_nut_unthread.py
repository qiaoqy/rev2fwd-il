#!/usr/bin/env python3
"""Step 1.1: Collect REVERSE nut unthreading demonstration data.

This script collects reverse trajectories - starting from a THREADED state
and performing unthreading (loosening) motion.

=============================================================================
REVERSE DATA COLLECTION STRATEGY (rev2fwd-il)
=============================================================================
The core idea from rev2fwd-il:
1. Initialize with nut ALREADY THREADED onto the bolt (goal state)
2. Execute a simple "unthreading" policy (rotate CCW + pull up)
3. Time-reverse the collected trajectories to get forward threading data

This is easier because:
- We can directly set the initial state to the threaded configuration
- Unthreading motion is simpler (just reverse rotation + pull)
- Gravity helps maintain contact during unthreading

=============================================================================
ENVIRONMENT DETAILS (from isaaclab_tasks)
=============================================================================
Initial State (MODIFIED for reverse collection):
- Nut is pre-threaded onto bolt (instead of held above)
- Robot gripper grasps the nut in the threaded position

Domain Randomization (inherited from FORGE):
- Bolt friction: 0.25 - 1.25 (128 buckets, static only)
- Nut friction: ~0.25 effective (0.01 offset from 0.75 base - 0.5)
- Mass: ±5g additive noise on nut
- Controller gains: ±41% noise
- Force observation noise: ±1N
- Position observation noise: ±0.25mm
- Rotation observation noise: ±0.1°

Force Sensing:
- 6-DOF F/T sensor on robot wrist (force_sensor link)
- Smoothed with EMA (α=0.25)
- Transformed to bolt-fixed frame

Success Threshold:
- success_threshold: 0.375 (fraction of thread height)
- thread_pitch: 2mm (BoltM16)
- Effective success depth: 0.75mm below engagement

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
Same format as forward collection, but trajectory goes from
threaded→unthreaded (reverse of goal):

    - obs:              (T, obs_dim)  Policy observation sequence
    - state:            (T, state_dim) Full state observation (privileged)
    - images:           (T, H, W, 3)  RGB images from table camera (uint8)
    - wrist_wrist_cam:  (T, H, W, 3)  Wrist camera
    - ee_pose:          (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - nut_pose:         (T, 7)   Nut (held asset) pose
    - bolt_pose:        (T, 7)   Bolt (fixed asset) pose
    - action:           (T, 7)   Action [pos(3), rot(3), success_pred]
    - ft_force:         (T, 3)   Force/torque sensor readings (force xyz)
    - ft_force_raw:     (T, 6)   Raw force/torque readings (force + torque)
    - joint_pos:        (T, 7)   Robot joint positions
    - episode_length:   int      Total timesteps in episode
    - success:          bool     Whether unthreading was successful
    - is_reverse:       bool     Always True (marks as reverse trajectory)

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (headless mode, 100 episodes)
python scripts_nut/1_1_collect_data_nut_unthread.py --headless --num_episodes 100

# Production run
CUDA_VISIBLE_DEVICES=2 python scripts_nut/1_1_collect_data_nut_unthread.py \
    --headless --num_episodes 500 \
    --out data/B_nut_unthread.npz

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
        description="Collect REVERSE nut unthreading demonstration data.",
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
        help="Number of parallel environments.",
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
        default=200,
        help="Maximum simulation steps per episode.",
    )
    
    # -----------------------------------------------------------------
    # Unthreading expert policy parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--unthread_rotation_speed",
        type=float,
        default=-0.5,
        help="Angular velocity for unthreading rotation (rad/s). Negative = loosening direction.",
    )
    parser.add_argument(
        "--lift_speed",
        type=float,
        default=0.001,
        help="Upward velocity during unthreading (m/s).",
    )
    parser.add_argument(
        "--force_threshold",
        type=float,
        default=2.0,
        help="Force threshold (N) to detect if nut is still engaged with bolt. "
             "If |Fz| > threshold during lift attempt, nut is still threaded.",
    )
    parser.add_argument(
        "--z_change_threshold",
        type=float,
        default=0.001,
        help="Minimum z position change (m) to consider nut is moving. "
             "If z change < threshold during lift, nut is stuck.",
    )
    
    # -----------------------------------------------------------------
    # Initial threaded state parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--thread_depth",
        type=float,
        default=0.008,
        help="Initial thread depth in meters (how deep the nut is threaded). "
             "BoltM16 thread height is 0.025m, pitch is 0.002m.",
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
        default="data/B_nut_unthread.npz",
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
    """Compute camera rotation quaternion from eye position and lookat target."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    
    rotation_matrix = np.column_stack([right, down, forward])
    rot = R.from_matrix(rotation_matrix)
    q_xyzw = rot.as_quat()
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    
    return (qw, qx, qy, qz)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int):
    """Dynamically add camera sensors to the FORGE environment configuration."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    
    camera_eye = (0.7, 0.4, 0.5)
    camera_lookat = (0.6, 0.0, 0.0)
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)
    
    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
            convention="ros",
        ),
    )
    
    env_cfg.scene.env_spacing = 5.0
    
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'object_pose'):
        env_cfg.commands.object_pose.debug_vis = False


def modify_env_cfg_for_threaded_init(env_cfg, thread_depth: float):
    """Modify environment config to initialize with nut already threaded on bolt.
    
    Key changes:
    1. Set hand_init_pos to be lower (at threaded position instead of above bolt)
    2. Reduce position noise to maintain proper threading alignment
    
    Args:
        env_cfg: The environment configuration object.
        thread_depth: How deep the nut should be threaded (meters).
    """
    print(f"[DEBUG] modify_env_cfg_for_threaded_init: thread_depth={thread_depth}")
    
    # Access the task config to modify initial positions
    # The hand position is relative to the bolt tip
    # Original: hand_init_pos = [0.0, 0.0, 0.015] (1.5cm above bolt tip)
    # For threaded state: we want the nut to be ON the bolt, so z should be negative
    
    # BoltM16 specs:
    # - thread_height = 0.025m (25mm)
    # - thread_pitch = 0.002m (2mm)
    # - The bolt tip is at z=0 in the bolt frame
    
    # For a threaded nut, we want to position it BELOW the bolt tip
    # thread_depth of 0.008m means 8mm into the threads
    
    if hasattr(env_cfg, 'hand_init_pos'):
        # Direct access (some configs)
        env_cfg.hand_init_pos = [0.0, 0.0, -thread_depth]
        env_cfg.hand_init_pos_noise = [0.002, 0.002, 0.001]  # Reduce noise for stability
        print(f"[DEBUG] Set hand_init_pos to {env_cfg.hand_init_pos}")
    
    # Also try via scene or task config
    if hasattr(env_cfg, 'scene'):
        scene_cfg = env_cfg.scene
        # Some FORGE configs store this differently
        print(f"[DEBUG] scene_cfg attributes: {[a for a in dir(scene_cfg) if not a.startswith('_')][:20]}")
    
    # The actual modification happens in the reset function override
    # We'll store the thread_depth in the env_cfg for later use
    env_cfg._reverse_thread_depth = thread_depth
    
    return env_cfg


def add_wrist_camera_post_creation(env, num_envs: int, image_width: int, image_height: int):
    """Add wrist camera after FORGE environment creation using replicator API."""
    import omni.usd
    from pxr import UsdGeom, Gf
    import omni.replicator.core as rep
    
    print("[DEBUG] add_wrist_camera_post_creation: Starting...", flush=True)
    
    stage = omni.usd.get_context().get_stage()
    
    camera_configs = [
        ("wrist_cam",
         Gf.Vec3d(0.05, 0.0, 0.0),
         Gf.Quatf(0.0, 1.0, 0.0, 0.0),
         "Wrist camera looking at workspace"),
    ]
    
    all_camera_data = {}
    
    for cam_name, cam_pos, cam_quat, cam_desc in camera_configs:
        render_products = []
        rgb_annotators = []
        
        for env_idx in range(num_envs):
            panda_hand_path = f"/World/envs/env_{env_idx}/Robot/panda_hand"
            panda_hand_prim = stage.GetPrimAtPath(panda_hand_path)
            
            if not panda_hand_prim.IsValid():
                print(f"[ERROR] panda_hand not found at {panda_hand_path}", flush=True)
                render_products.append(None)
                rgb_annotators.append(None)
                continue
            
            wrist_cam_path = f"{panda_hand_path}/{cam_name}"
            
            if not stage.GetPrimAtPath(wrist_cam_path).IsValid():
                camera_prim = UsdGeom.Camera.Define(stage, wrist_cam_path)
                camera_prim.GetFocalLengthAttr().Set(18.0)
                camera_prim.GetHorizontalApertureAttr().Set(20.955)
                camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 2.0))
                
                xform = UsdGeom.Xformable(camera_prim.GetPrim())
                xform.ClearXformOpOrder()
                
                translate_op = xform.AddTranslateOp()
                translate_op.Set(cam_pos)
                
                orient_op = xform.AddOrientOp()
                orient_op.Set(cam_quat)
            
            try:
                render_product = rep.create.render_product(wrist_cam_path, (image_width, image_height))
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
    thread_depth: float,
):
    """Create a FORGE environment configured for reverse (unthreading) data collection."""
    import gymnasium as gym
    
    print("[DEBUG] make_forge_env_with_camera: Importing isaaclab_tasks...")
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    print("[DEBUG] make_forge_env_with_camera: Imports done")
    
    print("[DEBUG] make_forge_env_with_camera: Parsing env config...")
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    print("[DEBUG] make_forge_env_with_camera: Config parsed")
    
    # Add camera to config
    print("[DEBUG] make_forge_env_with_camera: Adding camera to env config...", flush=True)
    add_camera_to_env_cfg(env_cfg, image_width, image_height)
    
    # Modify for threaded initial state
    print("[DEBUG] make_forge_env_with_camera: Modifying for threaded init...", flush=True)
    modify_env_cfg_for_threaded_init(env_cfg, thread_depth)
    
    print(f"[DEBUG] make_forge_env_with_camera: scene has table_cam: {hasattr(env_cfg.scene, 'table_cam')}", flush=True)
    
    # Create environment
    print("[DEBUG] make_forge_env_with_camera: Creating gym environment...", flush=True)
    sys.stdout.flush()
    try:
        env = gym.make(task_id, cfg=env_cfg)
        print("[DEBUG] make_forge_env_with_camera: Gym environment created", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR in gym.make: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[DEBUG] make_forge_env_with_camera: env type: {type(env)}", flush=True)
    print(f"[DEBUG] make_forge_env_with_camera: env.unwrapped type: {type(env.unwrapped)}", flush=True)
    
    # Store thread depth for custom reset
    env.unwrapped._reverse_thread_depth = thread_depth
    
    # Check scene sensors
    print(f"[DEBUG] make_forge_env_with_camera: Checking scene.sensors...", flush=True)
    try:
        sensors = env.unwrapped.scene.sensors
        print(f"[DEBUG] make_forge_env_with_camera: sensors keys: {list(sensors.keys())}", flush=True)
    except Exception as e:
        print(f"[DEBUG] ERROR accessing scene.sensors: {e}", flush=True)
    
    # Add wrist cameras
    print("[DEBUG] make_forge_env_with_camera: Adding wrist cameras post-creation...", flush=True)
    wrist_camera_data = add_wrist_camera_post_creation(
        env, num_envs, image_width, image_height
    )
    
    env.unwrapped._wrist_camera_data = wrist_camera_data
    
    first_cam_name = list(wrist_camera_data.keys())[0] if wrist_camera_data else None
    if first_cam_name:
        env.unwrapped._wrist_render_products = wrist_camera_data[first_cam_name][0]
        env.unwrapped._wrist_rgb_annotators = wrist_camera_data[first_cam_name][1]
    
    return env


def set_threaded_initial_state(env, thread_depth: float):
    """Set the environment to a threaded state after reset.
    
    This function manually positions the nut onto the bolt after the standard
    reset, simulating a pre-threaded initial condition.
    
    Args:
        env: The FORGE gymnasium environment.
        thread_depth: How deep the nut should be threaded (meters).
    """
    import torch
    
    forge_env = env.unwrapped
    device = forge_env.device
    num_envs = forge_env.num_envs
    
    print(f"[DEBUG] set_threaded_initial_state: Setting thread_depth={thread_depth}")
    
    # Get bolt (fixed asset) position
    bolt_pos = forge_env._fixed_asset.data.root_pos_w.clone()
    bolt_quat = forge_env._fixed_asset.data.root_quat_w.clone()
    
    # Calculate threaded nut position
    # Nut should be centered on bolt and lowered by thread_depth
    nut_target_pos = bolt_pos.clone()
    nut_target_pos[:, 2] -= thread_depth  # Lower the nut onto the bolt
    
    # Get current nut state
    nut_pos = forge_env._held_asset.data.root_pos_w.clone()
    nut_quat = forge_env._held_asset.data.root_quat_w.clone()
    
    print(f"[DEBUG] Bolt pos: {bolt_pos[0].cpu().numpy()}")
    print(f"[DEBUG] Original nut pos: {nut_pos[0].cpu().numpy()}")
    print(f"[DEBUG] Target nut pos: {nut_target_pos[0].cpu().numpy()}")
    
    # Set nut position to threaded state
    # We need to set both the nut and the robot gripper position
    
    # Option 1: Use teleport/set_world_pose if available
    if hasattr(forge_env._held_asset, 'write_root_pose_to_sim'):
        # Combine position and quaternion
        new_nut_pose = torch.cat([nut_target_pos, bolt_quat], dim=-1)
        forge_env._held_asset.write_root_pose_to_sim(new_nut_pose)
        print("[DEBUG] Used write_root_pose_to_sim for nut")
    
    # Also move the robot gripper to match
    # The gripper should be at the nut position
    fingertip_offset = torch.tensor([0.0, 0.0, 0.0], device=device)
    target_fingertip_pos = nut_target_pos + fingertip_offset
    
    print(f"[DEBUG] Target fingertip pos: {target_fingertip_pos[0].cpu().numpy()}")
    
    # Step the simulation a few times to let physics settle
    for _ in range(10):
        forge_env.scene.write_data_to_sim()
        forge_env.sim.step(render=False)
        forge_env.scene.update(forge_env.sim.get_physics_dt())


class NutUnthreadingExpert:
    """Scripted expert for nut UNthreading task (reverse of threading) with FORCE FEEDBACK.
    
    The expert follows a force-feedback strategy:
    1. Maintain: Hold position briefly to ensure stable grasp
    2. Unthread: Rotate CCW for a few steps, then try to lift
    3. Check: If |Fz| > threshold AND z position didn't change much -> still engaged
       - If still engaged: go back to unthread phase
       - If disengaged (Fz low or z increased): proceed to lift
    4. Lift: Pull up once threads are disengaged
    5. Done: Stop when nut is fully separated from bolt
    
    This generates reverse trajectories that can be time-flipped for forward training.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        unthread_rotation_speed: float = -0.5,
        lift_speed: float = 0.001,
        force_threshold: float = 2.0,
        z_change_threshold: float = 0.001,
    ):
        """Initialize the nut unthreading expert with force feedback.
        
        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
            unthread_rotation_speed: Angular velocity for loosening (rad/s). 
                                     Negative = counter-clockwise = loosening.
            lift_speed: Upward velocity during unthreading (m/s).
            force_threshold: Force threshold (N) to detect if nut is still engaged.
            z_change_threshold: Minimum z position change (m) to consider nut moving.
        """
        import torch
        
        self.num_envs = num_envs
        self.device = device
        self.unthread_rotation_speed = unthread_rotation_speed
        self.lift_speed = lift_speed
        self.force_threshold = force_threshold
        self.z_change_threshold = z_change_threshold
        
        # State tracking
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # Phases: 0=maintain, 1=unthread, 2=try_lift, 3=lift, 4=done
        
        # Phase timing
        self.maintain_steps = 20     # Steps to stabilize at start
        self.unthread_steps = 30     # Steps to rotate before trying to lift
        self.try_lift_steps = 15     # Steps to attempt lifting and check force
        
        # Position tracking for force feedback
        self.z_at_try_lift_start = torch.zeros(num_envs, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        
        # Track number of unthread cycles (for debugging)
        self.unthread_cycles = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.max_unthread_cycles = 10  # Safety limit
        
    def reset(self, env_ids=None):
        """Reset expert state for specified environments."""
        import torch
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.step_count[env_ids] = 0
        self.phase[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.z_at_try_lift_start[env_ids] = 0.0
        self.unthread_cycles[env_ids] = 0
    
    def compute_action(
        self,
        fingertip_pos: "torch.Tensor",
        fingertip_quat: "torch.Tensor",
        fixed_pos: "torch.Tensor",
        force_sensor: "torch.Tensor",
        dt: float,
    ) -> "torch.Tensor":
        """Compute expert action for unthreading with force feedback.
        
        The FORGE action space is 7D:
        - pos (3): Position delta in end-effector frame
        - rot (3): Rotation delta (roll, pitch, yaw) 
        - success_pred (1): Success prediction output
        
        Force feedback logic:
        - After rotating for unthread_steps, try to lift
        - If |Fz| > force_threshold AND z hasn't changed much -> still threaded
        - Go back to unthread phase and rotate more
        
        Args:
            fingertip_pos: Current fingertip position (num_envs, 3)
            fingertip_quat: Current fingertip quaternion (num_envs, 4)
            fixed_pos: Position of fixed asset (bolt) (num_envs, 3)
            force_sensor: Force sensor readings (num_envs, 3) [Fx, Fy, Fz]
            dt: Physics timestep
            
        Returns:
            Action tensor (num_envs, 7)
        """
        import torch
        
        # Initialize action
        action = torch.zeros(self.num_envs, 7, device=self.device)
        
        self.step_count += 1
        self.phase_step_count += 1
        
        # Get current z position and force
        current_z = fingertip_pos[:, 2]
        fz = force_sensor[:, 2]  # Z-axis force
        
        # =====================================================================
        # Phase 0: Maintain - hold position briefly
        # =====================================================================
        maintain_mask = self.phase == 0
        maintain_done = maintain_mask & (self.phase_step_count > self.maintain_steps)
        
        if maintain_mask.any():
            action[maintain_mask, 0:3] = 0.0  # No movement
            action[maintain_mask, 3:6] = 0.0  # No rotation
        
        # Transition: maintain -> unthread
        if maintain_done.any():
            self.phase[maintain_done] = 1
            self.phase_step_count[maintain_done] = 0
        
        # =====================================================================
        # Phase 1: Unthread - rotate CCW for a while
        # =====================================================================
        unthread_mask = self.phase == 1
        unthread_done = unthread_mask & (self.phase_step_count > self.unthread_steps)
        
        if unthread_mask.any():
            action[unthread_mask, 0:2] = 0.0   # Stay centered in XY
            action[unthread_mask, 2] = 0.02    # Slight upward bias
            action[unthread_mask, 5] = -0.3    # Rotate CCW (loosening)
        
        # Transition: unthread -> try_lift
        if unthread_done.any():
            self.phase[unthread_done] = 2
            self.phase_step_count[unthread_done] = 0
            self.z_at_try_lift_start[unthread_done] = current_z[unthread_done]
        
        # =====================================================================
        # Phase 2: Try lift - attempt to pull up and check force feedback
        # =====================================================================
        try_lift_mask = self.phase == 2
        try_lift_done = try_lift_mask & (self.phase_step_count > self.try_lift_steps)
        
        if try_lift_mask.any():
            action[try_lift_mask, 0:2] = 0.0   # Stay centered
            action[try_lift_mask, 2] = 0.1     # Try to lift
            action[try_lift_mask, 3:6] = 0.0   # No rotation while testing
        
        # Check force feedback after try_lift phase
        if try_lift_done.any():
            # Calculate z position change
            z_change = current_z - self.z_at_try_lift_start
            
            # Check if still engaged: high force AND small z change
            # |Fz| > threshold means resistance (could be positive or negative depending on direction)
            high_force = torch.abs(fz) > self.force_threshold
            small_z_change = z_change < self.z_change_threshold
            still_engaged = high_force & small_z_change & try_lift_done
            
            # Check if disengaged: low force OR significant z increase
            disengaged = (~still_engaged) & try_lift_done
            
            # Safety: limit number of unthread cycles
            max_cycles_reached = self.unthread_cycles >= self.max_unthread_cycles
            
            # Transition: still engaged -> back to unthread (if not max cycles)
            back_to_unthread = still_engaged & (~max_cycles_reached)
            if back_to_unthread.any():
                self.phase[back_to_unthread] = 1
                self.phase_step_count[back_to_unthread] = 0
                self.unthread_cycles[back_to_unthread] += 1
                # Debug print
                for i in range(self.num_envs):
                    if back_to_unthread[i]:
                        print(f"[DEBUG] Env {i}: Still engaged! |Fz|={abs(fz[i].item()):.2f}N, "
                              f"z_change={z_change[i].item()*1000:.2f}mm, cycle={self.unthread_cycles[i].item()}")
            
            # Transition: disengaged OR max cycles -> lift
            to_lift = disengaged | (still_engaged & max_cycles_reached)
            if to_lift.any():
                self.phase[to_lift] = 3
                self.phase_step_count[to_lift] = 0
                for i in range(self.num_envs):
                    if to_lift[i]:
                        print(f"[DEBUG] Env {i}: Disengaged! |Fz|={abs(fz[i].item()):.2f}N, "
                              f"z_change={z_change[i].item()*1000:.2f}mm, cycles={self.unthread_cycles[i].item()}")
        
        # =====================================================================
        # Phase 3: Lift - pull up now that threads are disengaged
        # =====================================================================
        lift_mask = self.phase == 3
        
        if lift_mask.any():
            action[lift_mask, 0:2] = 0.0
            action[lift_mask, 2] = 0.15    # Lift up
            action[lift_mask, 3:6] = 0.0   # No rotation
        
        # Transition to done after sufficient lift (based on z position relative to bolt)
        z_above_bolt = current_z - fixed_pos[:, 2]
        lift_complete = lift_mask & (z_above_bolt > 0.02)  # 2cm above bolt
        if lift_complete.any():
            self.phase[lift_complete] = 4
        
        # =====================================================================
        # Phase 4: Done - hold position
        # =====================================================================
        done_mask = self.phase == 4
        if done_mask.any():
            action[done_mask, :] = 0.0
        
        # =====================================================================
        # Success prediction
        # =====================================================================
        # Progress based on phase and cycles
        base_progress = self.phase.float() / 4.0
        action[:, 6] = (base_progress.clamp(max=1.0) * 2.0 - 1.0)
        
        return action
    
    def is_done(self) -> "torch.Tensor":
        """Check which environments have completed unthreading."""
        return self.phase >= 4


def rollout_nut_unthreading(
    env,
    expert: NutUnthreadingExpert,
    horizon: int,
    thread_depth: float,
):
    """Run parallel rollouts for nut UNthreading and collect data.
    
    Args:
        env: FORGE gymnasium environment with camera and force sensors.
        expert: NutUnthreadingExpert instance.
        horizon: Maximum steps for each episode.
        thread_depth: Initial thread depth for reset.
    
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
    
    wrist_camera_data = getattr(env.unwrapped, '_wrist_camera_data', None)
    if wrist_camera_data is not None:
        print(f"[DEBUG] rollout: wrist cameras acquired: {list(wrist_camera_data.keys())}")
    else:
        print("[DEBUG] rollout: wrist_camera_data not available")
    print(f"[DEBUG] rollout: Cameras acquired")
    
    # =========================================================================
    # Step 1: Reset environment and set to threaded state
    # =========================================================================
    print("[DEBUG] rollout: Resetting environment...")
    obs_dict, _ = env.reset()
    
    # Set to threaded initial state
    print("[DEBUG] rollout: Setting threaded initial state...")
    set_threaded_initial_state(env, thread_depth)
    
    expert.reset()
    print(f"[DEBUG] rollout: Reset done")
    
    forge_env = env.unwrapped
    
    # =========================================================================
    # Step 2: Initialize per-env recording buffers
    # =========================================================================
    obs_lists = [[] for _ in range(num_envs)]
    state_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_cam_names = list(wrist_camera_data.keys()) if wrist_camera_data else []
    wrist_image_lists_dict = {cam_name: [[] for _ in range(num_envs)] for cam_name in wrist_cam_names}
    ee_pose_lists = [[] for _ in range(num_envs)]
    nut_pose_lists = [[] for _ in range(num_envs)]
    bolt_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    ft_force_lists = [[] for _ in range(num_envs)]
    ft_force_raw_lists = [[] for _ in range(num_envs)]
    joint_pos_lists = [[] for _ in range(num_envs)]
    
    # =========================================================================
    # Step 3: Run episode and record data
    # =========================================================================
    print(f"[DEBUG] rollout: Starting main loop (horizon={horizon})...")
    
    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")
        
        # Get observations
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
        table_rgb = table_camera.data.output["rgb"]
        
        wrist_images_by_cam = {}
        if wrist_camera_data is not None:
            for cam_name, (render_products, rgb_annotators) in wrist_camera_data.items():
                wrist_images_list = []
                for i, annotator in enumerate(rgb_annotators):
                    if annotator is not None:
                        data = annotator.get_data()
                        if data is not None:
                            if not isinstance(data, torch.Tensor):
                                data = torch.from_numpy(data).to(device)
                            wrist_images_list.append(data)
                        else:
                            wrist_images_list.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                    else:
                        wrist_images_list.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                wrist_images_by_cam[cam_name] = torch.stack(wrist_images_list, dim=0)
        
        # Truncate to 3 channels if RGBA
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        for cam_name in wrist_images_by_cam:
            if wrist_images_by_cam[cam_name].shape[-1] > 3:
                wrist_images_by_cam[cam_name] = wrist_images_by_cam[cam_name][..., :3]
        
        # Get poses and force data
        fingertip_pos = forge_env.fingertip_midpoint_pos.clone()
        fingertip_quat = forge_env.fingertip_midpoint_quat.clone()
        
        nut_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
        nut_quat = forge_env._held_asset.data.root_quat_w
        nut_pose = torch.cat([nut_pos, nut_quat], dim=-1)
        
        bolt_pos = forge_env._fixed_asset.data.root_pos_w - forge_env.scene.env_origins
        bolt_quat = forge_env._fixed_asset.data.root_quat_w
        bolt_pose = torch.cat([bolt_pos, bolt_quat], dim=-1)
        
        if hasattr(forge_env, 'force_sensor_smooth'):
            ft_force_raw = forge_env.force_sensor_smooth.clone()
            ft_force = ft_force_raw[:, :3]
        else:
            ft_force_raw = torch.zeros(num_envs, 6, device=device)
            ft_force = torch.zeros(num_envs, 3, device=device)
        
        joint_pos = forge_env._robot.data.joint_pos[:, :7]
        ee_pose = torch.cat([fingertip_pos, fingertip_quat], dim=-1)
        
        # Compute expert action
        action = expert.compute_action(
            fingertip_pos=fingertip_pos,
            fingertip_quat=fingertip_quat,
            fixed_pos=bolt_pos,
            force_sensor=ft_force,
            dt=physics_dt,
        )
        
        # Record data
        policy_obs_np = policy_obs.cpu().numpy()
        critic_state_np = critic_state.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np_dict = {
            cam_name: wrist_images_by_cam[cam_name].cpu().numpy().astype(np.uint8)
            for cam_name in wrist_images_by_cam
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
            for cam_name in wrist_cam_names:
                wrist_image_lists_dict[cam_name][i].append(wrist_images_np_dict[cam_name][i])
            ee_pose_lists[i].append(ee_pose_np[i])
            nut_pose_lists[i].append(nut_pose_np[i])
            bolt_pose_lists[i].append(bolt_pose_np[i])
            action_lists[i].append(action_np[i])
            ft_force_lists[i].append(ft_force_np[i])
            ft_force_raw_lists[i].append(ft_force_raw_np[i])
            joint_pos_lists[i].append(joint_pos_np[i])
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        # Check for resets
        done = terminated | truncated
        if done.any():
            print(f"[DEBUG] Step {t+1}: {done.sum().item()} envs done")
    
    print("[DEBUG] rollout: Main loop finished")
    
    # =========================================================================
    # Step 4: Build episode dictionaries
    # =========================================================================
    print("[DEBUG] rollout: Building episode dicts...")
    results = []
    
    success_threshold = forge_env.cfg_task.success_threshold
    
    for i in range(num_envs):
        episode_dict = {
            "obs": np.array(obs_lists[i], dtype=np.float32),
            "state": np.array(state_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),
            **{f"wrist_{cam_name}": np.array(wrist_image_lists_dict[cam_name][i], dtype=np.uint8) 
               for cam_name in wrist_cam_names},
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "nut_pose": np.array(nut_pose_lists[i], dtype=np.float32),
            "bolt_pose": np.array(bolt_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "ft_force": np.array(ft_force_lists[i], dtype=np.float32),
            "ft_force_raw": np.array(ft_force_raw_lists[i], dtype=np.float32),
            "joint_pos": np.array(joint_pos_lists[i], dtype=np.float32),
            "episode_length": len(obs_lists[i]),
            "success": True,  # Assume success for reverse trajectories
            "success_threshold": success_threshold,
            "wrist_cam_names": wrist_cam_names,
            "is_reverse": True,  # Mark as reverse trajectory
            "thread_depth": thread_depth,
        }
        results.append(episode_dict)
    
    print(f"[DEBUG] rollout: Done. Returning {len(results)} episodes")
    return results


def save_episodes(path: str, episodes: list[dict]) -> None:
    """Save episodes to an NPZ file."""
    from pathlib import Path
    import numpy as np
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(path, episodes=episodes)
    print(f"Saved {len(episodes)} episodes to {path}")
    
    if episodes:
        ep0 = episodes[0]
        print(f"  - Policy obs shape: {ep0['obs'].shape}")
        print(f"  - State shape: {ep0['state'].shape}")
        print(f"  - Table camera image shape: {ep0['images'].shape}")
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
        print(f"  - Is reverse: {ep0['is_reverse']}")
        print(f"  - Thread depth: {ep0['thread_depth']}")
        
        all_forces = np.concatenate([ep['ft_force'] for ep in episodes], axis=0)
        print(f"  - Force stats: min={all_forces.min():.2f}, max={all_forces.max():.2f}, "
              f"mean={all_forces.mean():.2f}, std={all_forces.std():.2f}")


def main() -> None:
    """Main entry point for collecting reverse nut unthreading data."""
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
        # Step 4: Create FORGE environment configured for reverse collection
        # =================================================================
        print("[DEBUG] Step 4: Creating FORGE environment for REVERSE collection...")
        print(f"[DEBUG]   task_id={args.task}")
        print(f"[DEBUG]   num_envs={args.num_envs}")
        print(f"[DEBUG]   device={args.device}")
        print(f"[DEBUG]   image_size={args.image_width}x{args.image_height}")
        print(f"[DEBUG]   thread_depth={args.thread_depth}")
        
        num_envs = args.num_envs
        env = make_forge_env_with_camera(
            task_id=args.task,
            num_envs=num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            thread_depth=args.thread_depth,
        )
        print("[DEBUG] Step 4: Environment created successfully")
        
        device = env.unwrapped.device
        print(f"[DEBUG] Environment device: {device}")
        
        forge_env = env.unwrapped
        print(f"[DEBUG] FORGE task name: {forge_env.cfg_task.name}")
        print(f"[DEBUG] FORGE action space: {env.action_space}")
        print(f"[DEBUG] FORGE observation space: {env.observation_space}")
        
        # Check sensors
        print(f"[DEBUG] Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        
        # =================================================================
        # Step 5: Create UNTHREADING expert policy with force feedback
        # =================================================================
        print("[DEBUG] Step 5: Creating nut UNTHREADING expert with FORCE FEEDBACK...")
        expert = NutUnthreadingExpert(
            num_envs=num_envs,
            device=device,
            unthread_rotation_speed=args.unthread_rotation_speed,
            lift_speed=args.lift_speed,
            force_threshold=args.force_threshold,
            z_change_threshold=args.z_change_threshold,
        )
        print("[DEBUG] Step 5: Expert created")
        
        # =================================================================
        # Step 6: Data collection loop
        # =================================================================
        episodes = []
        
        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} REVERSE (unthreading) episodes")
        print(f"Settings:")
        print(f"  - num_envs (parallel): {num_envs}")
        print(f"  - horizon: {args.horizon}")
        print(f"  - image_size: {args.image_width}x{args.image_height}")
        print(f"  - unthread_rotation_speed: {args.unthread_rotation_speed} rad/s")
        print(f"  - lift_speed: {args.lift_speed} m/s")
        print(f"  - initial thread_depth: {args.thread_depth} m")
        print(f"Force feedback parameters:")
        print(f"  - force_threshold: {args.force_threshold} N")
        print(f"  - z_change_threshold: {args.z_change_threshold*1000:.1f} mm")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        batch_count = 0
        
        while len(episodes) < args.num_episodes:
            batch_count += 1
            
            results = rollout_nut_unthreading(
                env=env,
                expert=expert,
                horizon=args.horizon,
                thread_depth=args.thread_depth,
            )
            
            for episode_dict in results:
                episodes.append(episode_dict)
                if len(episodes) >= args.num_episodes:
                    break
            
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
        print(f"REVERSE collection finished in {elapsed:.1f}s")
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
        
        env.close()
        
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
