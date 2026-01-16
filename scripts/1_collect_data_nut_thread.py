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
    - wrist_images:     (T, H, W, 3)  RGB images from wrist camera (uint8)
    - ee_pose:          (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - nut_pose:         (T, 7)   Nut (held asset) pose
    - bolt_pose:        (T, 7)   Bolt (fixed asset) pose
    - action:           (T, 7)   Action [pos(3), rot(3), success_pred]
    - ft_force:         (T, 3)   Force/torque sensor readings (force xyz)
    - ft_force_raw:     (T, 6)   Raw force/torque readings (force + torque)
    - joint_pos:        (T, 7)   Robot joint positions
    - episode_length:   int      Total timesteps in episode
    - success:          bool     Whether task was successful
    - success_threshold: float   Threshold used for success determination

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage (headless mode, 100 episodes)
python scripts/1_collect_data_nut_thread.py --headless --num_episodes 100

# Production run (500 episodes with parallel envs)
CUDA_VISIBLE_DEVICES=1 python scripts/1_collect_data_nut_thread.py \
    --headless --num_episodes 300 --num_envs 4 \
    --out data/nut_thread.npz

# With custom image size
python scripts/1_collect_data_nut_thread.py --headless --num_episodes 100 \
    --image_width 128 --image_height 128 --out data/nut_thread_128.npz

=============================================================================
"""

from __future__ import annotations

import argparse
import time


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
    2. A wrist camera - mounted on robot's end-effector
    
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
    # FORGE task workspace is centered differently than lift task
    # The bolt is fixed at a certain position, we look at the workspace from above-front
    camera_eye = (0.6, 0.4, 0.5)      # Camera position: in front, slightly right, above
    camera_lookat = (0.0, 0.0, 0.15)  # Look at: workspace center, slightly above table
    
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
            clipping_range=(0.1, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
            convention="ros",
        ),
    )
    
    # =========================================================================
    # Wrist Camera - Eye-in-hand, attached to robot gripper
    # =========================================================================
    env_cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )
    
    # Increase environment spacing to prevent seeing adjacent environments
    env_cfg.scene.env_spacing = 5.0


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
    print("[DEBUG] make_forge_env_with_camera: Adding camera to env config...")
    add_camera_to_env_cfg(env_cfg, image_width, image_height)
    print("[DEBUG] make_forge_env_with_camera: Camera config added")
    
    # Create environment
    print("[DEBUG] make_forge_env_with_camera: Creating gym environment...")
    env = gym.make(task_id, cfg=env_cfg)
    print("[DEBUG] make_forge_env_with_camera: Gym environment created")
    return env


class NutThreadingExpert:
    """Simple scripted expert for nut threading task.
    
    The expert follows a simple strategy:
    1. Approach: Move down toward the bolt while maintaining alignment
    2. Engage: Press down gently to engage the threads
    3. Thread: Rotate in the threading direction while applying downward force
    4. Done: Stop when fully threaded or force exceeds threshold
    
    This is a simplified demonstration policy - real threading requires
    more sophisticated force-based control.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        rotation_speed: float = 0.5,
        downward_force: float = 5.0,
    ):
        """Initialize the nut threading expert.
        
        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
            rotation_speed: Angular velocity for threading (rad/s).
            downward_force: Target downward force during threading (N).
        """
        import torch
        
        self.num_envs = num_envs
        self.device = device
        self.rotation_speed = rotation_speed
        self.downward_force = downward_force
        
        # State tracking
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.phase = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # Phases: 0=approach, 1=engage, 2=thread, 3=done
        
        # Threading parameters
        self.approach_steps = 50  # Steps to approach
        self.engage_steps = 30   # Steps to engage threads
        
    def reset(self, env_ids=None):
        """Reset expert state for specified environments."""
        import torch
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.step_count[env_ids] = 0
        self.phase[env_ids] = 0
    
    def compute_action(
        self,
        fingertip_pos: "torch.Tensor",
        fingertip_quat: "torch.Tensor",
        fixed_pos: "torch.Tensor",
        force_sensor: "torch.Tensor",
        dt: float,
    ) -> "torch.Tensor":
        """Compute expert action based on current state.
        
        The FORGE action space is 7D:
        - pos (3): Position target relative to bolt frame
        - rot (3): Rotation target (roll, pitch, yaw) 
        - success_pred (1): Success prediction output
        
        Args:
            fingertip_pos: Current fingertip position (num_envs, 3)
            fingertip_quat: Current fingertip quaternion (num_envs, 4)
            fixed_pos: Position of fixed asset (bolt) (num_envs, 3)
            force_sensor: Force sensor readings (num_envs, 3)
            dt: Physics timestep
            
        Returns:
            Action tensor (num_envs, 7)
        """
        import torch
        import numpy as np
        
        # Initialize action
        action = torch.zeros(self.num_envs, 7, device=self.device)
        
        # Position action: relative to bolt position
        # For threading, we want to stay centered over the bolt
        rel_pos = fingertip_pos - fixed_pos
        
        # Simple phase-based control
        self.step_count += 1
        
        # Phase transitions
        approach_done = self.step_count > self.approach_steps
        engage_done = self.step_count > (self.approach_steps + self.engage_steps)
        
        self.phase = torch.where(
            (self.phase == 0) & approach_done,
            torch.ones_like(self.phase),  # -> engage
            self.phase
        )
        self.phase = torch.where(
            (self.phase == 1) & engage_done,
            torch.full_like(self.phase, 2),  # -> thread
            self.phase
        )
        
        # Phase 0: Approach - move down
        approach_mask = self.phase == 0
        if approach_mask.any():
            # Small downward motion
            action[approach_mask, 0:3] = 0.0  # Stay centered
            action[approach_mask, 2] = -0.1   # Move down slightly
            action[approach_mask, 3:6] = 0.0  # No rotation
        
        # Phase 1: Engage - press down gently
        engage_mask = self.phase == 1
        if engage_mask.any():
            action[engage_mask, 0:3] = 0.0
            action[engage_mask, 2] = -0.2   # Press down more
            action[engage_mask, 3:6] = 0.0
        
        # Phase 2: Thread - rotate while pressing down
        thread_mask = self.phase == 2
        if thread_mask.any():
            action[thread_mask, 0:3] = 0.0
            action[thread_mask, 2] = -0.15  # Maintain downward pressure
            # Yaw rotation for threading
            # Action[5] is yaw, normalized to [-1, 1] range
            # Maps to [-180, +90] degrees in FORGE
            action[thread_mask, 5] = 0.3   # Rotate in threading direction
        
        # Success prediction: estimate based on how long we've been threading
        thread_progress = (self.step_count - self.approach_steps - self.engage_steps).float()
        thread_progress = thread_progress.clamp(min=0) / 200.0  # Normalize
        action[:, 6] = thread_progress.clamp(max=1.0) * 2.0 - 1.0  # Scale to [-1, 1]
        
        return action
    
    def is_done(self) -> "torch.Tensor":
        """Check which environments have completed the task."""
        return self.phase >= 3


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
    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]
    print(f"[DEBUG] rollout: Cameras acquired")
    
    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    print("[DEBUG] rollout: Resetting environment...")
    obs_dict, _ = env.reset()
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
    wrist_image_lists = [[] for _ in range(num_envs)]
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
        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]
        
        # Get poses and force data from FORGE environment
        # These are internal tensors maintained by ForgeEnv
        fingertip_pos = forge_env.fingertip_midpoint_pos.clone()
        fingertip_quat = forge_env.fingertip_midpoint_quat.clone()
        
        # Held asset (nut) pose
        nut_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
        nut_quat = forge_env._held_asset.data.root_quat_w
        nut_pose = torch.cat([nut_pos, nut_quat], dim=-1)
        
        # Fixed asset (bolt) pose  
        bolt_pos = forge_env._fixed_asset.data.root_pos_w - forge_env.scene.env_origins
        bolt_quat = forge_env._fixed_asset.data.root_quat_w
        bolt_pose = torch.cat([bolt_pos, bolt_quat], dim=-1)
        
        # Get force sensor data - this is the key FORGE feature!
        # force_sensor_smooth contains smoothed force/torque readings (6D: Fx, Fy, Fz, Tx, Ty, Tz)
        if hasattr(forge_env, 'force_sensor_smooth'):
            ft_force_raw = forge_env.force_sensor_smooth.clone()
            ft_force = ft_force_raw[:, :3]  # Just the force components
        else:
            # Fallback if force sensor not available
            ft_force_raw = torch.zeros(num_envs, 6, device=device)
            ft_force = torch.zeros(num_envs, 3, device=device)
        
        # Joint positions
        joint_pos = forge_env._robot.data.joint_pos[:, :7]
        
        # EE pose
        ee_pose = torch.cat([fingertip_pos, fingertip_quat], dim=-1)
        
        # Compute expert action
        action = expert.compute_action(
            fingertip_pos=fingertip_pos,
            fingertip_quat=fingertip_quat,
            fixed_pos=bolt_pos,
            force_sensor=ft_force,
            dt=physics_dt,
        )
        
        # Record data for each env
        policy_obs_np = policy_obs.cpu().numpy()
        critic_state_np = critic_state.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
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
            wrist_image_lists[i].append(wrist_images_np[i])
            ee_pose_lists[i].append(ee_pose_np[i])
            nut_pose_lists[i].append(nut_pose_np[i])
            bolt_pose_lists[i].append(bolt_pose_np[i])
            action_lists[i].append(action_np[i])
            ft_force_lists[i].append(ft_force_np[i])
            ft_force_raw_lists[i].append(ft_force_raw_np[i])
            joint_pos_lists[i].append(joint_pos_np[i])
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        
        # Check for resets (environment may auto-reset on success)
        done = terminated | truncated
        if done.any():
            print(f"[DEBUG] Step {t+1}: {done.sum().item()} envs done")
            # Don't break - let the episode continue for all data
    
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
            "wrist_images": np.array(wrist_image_lists[i], dtype=np.uint8),
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "nut_pose": np.array(nut_pose_lists[i], dtype=np.float32),
            "bolt_pose": np.array(bolt_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "ft_force": np.array(ft_force_lists[i], dtype=np.float32),
            "ft_force_raw": np.array(ft_force_raw_lists[i], dtype=np.float32),
            "joint_pos": np.array(joint_pos_lists[i], dtype=np.float32),
            "episode_length": len(obs_lists[i]),
            "success": False,  # Will be determined by inspection
            "success_threshold": success_threshold,
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
        print(f"  - Wrist camera image shape: {ep0['wrist_images'].shape}")
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
        
        device = env.unwrapped.device
        print(f"[DEBUG] Environment device: {device}")
        
        # Print FORGE environment info
        forge_env = env.unwrapped
        print(f"[DEBUG] FORGE task name: {forge_env.cfg_task.name}")
        print(f"[DEBUG] FORGE obs order: {forge_env.cfg.obs_order}")
        print(f"[DEBUG] FORGE state order: {forge_env.cfg.state_order}")
        print(f"[DEBUG] FORGE action space: {env.action_space}")
        print(f"[DEBUG] FORGE observation space: {env.observation_space}")
        
        # Check camera sensor
        print("[DEBUG] Checking camera sensors...")
        if "table_cam" in env.unwrapped.scene.sensors:
            camera = env.unwrapped.scene.sensors["table_cam"]
            print(f"[DEBUG] Table camera found: {camera.cfg.width}x{camera.cfg.height}")
        if "wrist_cam" in env.unwrapped.scene.sensors:
            camera = env.unwrapped.scene.sensors["wrist_cam"]
            print(f"[DEBUG] Wrist camera found: {camera.cfg.width}x{camera.cfg.height}")
        
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
