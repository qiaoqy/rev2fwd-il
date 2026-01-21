"""Waypoint-based action chunk executor for diffusion policy.

This module provides a waypoint-based execution strategy for action chunks
predicted by the diffusion policy. Instead of executing actions step by step
at a fixed rate, this executor:

1. Predicts a full action chunk (e.g., 8 waypoints)
2. Sends each waypoint as a target position
3. Waits until the robot reaches the waypoint (within threshold)
4. Only then moves to the next waypoint
5. Only re-infers the next action chunk after all waypoints are reached
6. Only updates the observation queue when waypoints are reached

This approach is more aligned with real robot execution where:
- Actions represent target positions, not velocities
- The robot controller handles the motion between waypoints
- Observations should reflect the state at meaningful milestones
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class WaypointConfig:
    """Configuration for waypoint-based execution."""
    
    # Position threshold to consider waypoint reached (in meters)
    position_threshold: float = 0.01
    
    # Orientation threshold (in radians) - optional
    orientation_threshold: float = 0.1
    
    # Maximum simulation steps to wait for reaching a waypoint
    # After this many steps, force move to next waypoint
    max_steps_per_waypoint: int = 100
    
    # Minimum steps to execute at each waypoint before checking if reached
    # This helps avoid immediately "reaching" the waypoint due to latency
    min_steps_per_waypoint: int = 3
    
    # Whether to check orientation when determining if waypoint is reached
    check_orientation: bool = False
    
    # Number of observation steps to maintain for policy (n_obs_steps)
    n_obs_steps: int = 2


@dataclass 
class WaypointState:
    """State tracking for waypoint execution."""
    
    # Current action chunk (n_action_steps, action_dim)
    action_chunk: Optional[np.ndarray] = None
    action_chunk_raw: Optional[np.ndarray] = None  # Unnormalized version
    
    # Current waypoint index within the chunk (0 to n_action_steps-1)
    current_waypoint_idx: int = 0
    
    # Steps taken trying to reach current waypoint
    steps_at_current_waypoint: int = 0
    
    # Total simulation steps taken in episode
    total_steps: int = 0
    
    # Number of inference calls made
    inference_count: int = 0
    
    # History of reached waypoints for visualization
    waypoint_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Observation queue - rolling buffer storing observations at EVERY step
    # This ensures we always have the most recent n_obs_steps observations for inference
    obs_queue: deque = field(default_factory=lambda: deque(maxlen=2))


class WaypointExecutor:
    """Executes action chunks using waypoint-based control.
    
    This executor manages the execution of action chunks from a diffusion policy
    using waypoint-based control. It:
    
    1. Takes an action chunk (sequence of target poses)
    2. Sends waypoints one at a time to the robot
    3. Waits for each waypoint to be reached before moving to the next
    4. Re-infers a new action chunk only when all waypoints are completed
    5. Maintains an observation queue that only updates at waypoint arrivals
    
    The executor does NOT modify the lerobot policy internals. Instead, it
    manages its own action queue and directly calls the diffusion model to
    generate full action chunks.
    """
    
    def __init__(
        self,
        config: WaypointConfig,
        device: str = "cuda",
    ):
        """Initialize the waypoint executor.
        
        Args:
            config: Waypoint execution configuration.
            device: Torch device for computations.
        """
        self.config = config
        self.device = device
        self.state = WaypointState()
        
        # Store references to policy components (set via set_policy)
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._n_action_steps = None
    
    def set_policy(
        self,
        policy,
        preprocessor,
        postprocessor,
        n_action_steps: int,
    ):
        """Set the policy and processors for inference.
        
        Args:
            policy: The diffusion policy (DiffusionPolicy from lerobot).
            preprocessor: Input preprocessor pipeline.
            postprocessor: Output postprocessor pipeline (unnormalizes actions).
            n_action_steps: Number of action steps per inference chunk.
        """
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._n_action_steps = n_action_steps
        
        # Update obs queue maxlen based on policy config
        if hasattr(policy, 'config') and hasattr(policy.config, 'n_obs_steps'):
            self.config.n_obs_steps = policy.config.n_obs_steps
        
        self.state.obs_queue = deque(maxlen=self.config.n_obs_steps)
    
    def reset(self):
        """Reset the executor state for a new episode."""
        self.state = WaypointState()
        self.state.obs_queue = deque(maxlen=self.config.n_obs_steps)
        
        # Also reset the policy's internal queues
        if self._policy is not None:
            self._policy.reset()
    
    def _check_waypoint_reached(
        self,
        current_pose: np.ndarray,  # (7,) xyz + quat
        target_pose: np.ndarray,   # (7,) xyz + quat (or 8 if including gripper)
    ) -> bool:
        """Check if the current pose has reached the target waypoint.
        
        Args:
            current_pose: Current EE pose (x, y, z, qw, qx, qy, qz).
            target_pose: Target waypoint pose (x, y, z, qw, qx, qy, qz, [gripper]).
            
        Returns:
            True if waypoint is considered reached.
        """
        # Position check
        current_xyz = current_pose[:3]
        target_xyz = target_pose[:3]
        position_error = np.linalg.norm(current_xyz - target_xyz)
        
        position_reached = position_error < self.config.position_threshold
        
        if not self.config.check_orientation:
            return position_reached
        
        # Orientation check (quaternion distance)
        current_quat = current_pose[3:7]  # wxyz
        target_quat = target_pose[3:7]
        
        # Quaternion distance: 1 - |q1 · q2| gives a value in [0, 1]
        # 0 means identical, 1 means 180 degree difference
        dot_product = np.abs(np.dot(current_quat, target_quat))
        orientation_error = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 2  # Convert to angle
        
        orientation_reached = orientation_error < self.config.orientation_threshold
        
        return position_reached and orientation_reached
    
    def _generate_action_chunk(
        self,
        observation: Dict[str, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a full action chunk from the diffusion policy.
        
        This method directly calls the diffusion model to generate a full
        action sequence, bypassing the internal action queue of the policy.
        
        Uses the most recent n_obs_steps observations from obs_queue for
        temporal context. The queue should be updated BEFORE calling this.
        
        Args:
            observation: Dictionary with observation tensors (raw, un-preprocessed).
                Expected keys: 'observation.image', 'observation.state',
                optionally 'observation.wrist_image'.
                
        Returns:
            Tuple of (action_chunk_norm, action_chunk_raw):
                - action_chunk_norm: (n_action_steps, action_dim) normalized actions
                - action_chunk_raw: (n_action_steps, action_dim) unnormalized actions
        """
        if self._policy is None:
            raise RuntimeError("Policy not set. Call set_policy() first.")
        
        # Get the required n_obs_steps from policy config
        n_obs_steps = self._policy.config.n_obs_steps if hasattr(self._policy, 'config') else 2
        
        # Prepare batch for diffusion model using observations from queue
        inference_batch = {}
        
        # Check if we have enough observations in the queue
        queue_len = len(self.state.obs_queue)
        
        if queue_len >= n_obs_steps:
            # Use the most recent n_obs_steps observations from queue
            recent_obs = list(self.state.obs_queue)[-n_obs_steps:]
            
            # Stack states: (B, n_obs_steps, state_dim)
            states = [obs['state'] for obs in recent_obs]
            state_tensor = torch.stack(states, dim=1)  # (B, n_obs_steps, state_dim)
            
            # Preprocess state
            if self._preprocessor is not None:
                # Create a temp dict for preprocessing just the state
                temp_obs = {'observation.state': state_tensor[:, -1, :]}  # Latest state for shape
                for key in self._policy.config.image_features if hasattr(self._policy.config, 'image_features') else []:
                    if key in recent_obs[-1]['images']:
                        temp_obs[key] = recent_obs[-1]['images'][key]
                temp_obs = self._preprocessor(temp_obs)
                
                # Now preprocess each state in the sequence
                processed_states = []
                for obs in recent_obs:
                    temp = {'observation.state': obs['state']}
                    temp = self._preprocessor(temp)
                    processed_states.append(temp['observation.state'])
                state_tensor = torch.stack(processed_states, dim=1)
            
            inference_batch['observation.state'] = state_tensor
            
            # Stack images: (B, n_obs_steps, num_cams, C, H, W)
            if hasattr(self._policy, 'config') and hasattr(self._policy.config, 'image_features'):
                images_list = []
                for key in self._policy.config.image_features:
                    imgs = []
                    for obs in recent_obs:
                        if key in obs['images']:
                            img = obs['images'][key]  # (B, C, H, W)
                            imgs.append(img)
                    if imgs:
                        img_stack = torch.stack(imgs, dim=1)  # (B, n_obs_steps, C, H, W)
                        
                        # Preprocess images
                        if self._preprocessor is not None:
                            # Process each timestep's image
                            processed_imgs = []
                            for t_idx in range(img_stack.shape[1]):
                                temp = {key: img_stack[:, t_idx]}
                                temp = self._preprocessor(temp)
                                processed_imgs.append(temp[key])
                            img_stack = torch.stack(processed_imgs, dim=1)
                        
                        images_list.append(img_stack)
                
                if images_list:
                    inference_batch['observation.images'] = torch.stack(images_list, dim=2)
        else:
            # Not enough observations in queue, fall back to replicating current observation
            # First preprocess the observation
            if self._preprocessor is not None:
                observation = self._preprocessor(observation)
            
            # Replicate state across n_obs_steps
            if 'observation.state' in observation:
                state = observation['observation.state']  # (B, state_dim)
                if state.dim() == 2:
                    state = state.unsqueeze(1).repeat(1, n_obs_steps, 1)
                inference_batch['observation.state'] = state
            
            # Replicate images across n_obs_steps
            if hasattr(self._policy, 'config') and hasattr(self._policy.config, 'image_features'):
                images_list = []
                for key in self._policy.config.image_features:
                    if key in observation:
                        img = observation[key]  # (B, C, H, W)
                        if img.dim() == 4:
                            img = img.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
                        images_list.append(img)
                
                if images_list:
                    inference_batch['observation.images'] = torch.stack(images_list, dim=2)
        
        # Call diffusion model to generate actions
        with torch.no_grad():
            if hasattr(self._policy, 'diffusion'):
                full_chunk = self._policy.diffusion.generate_actions(inference_batch)
                # full_chunk shape: (B, horizon, action_dim)
            else:
                # Fallback: use select_action repeatedly (less efficient)
                raise RuntimeError("Policy does not have 'diffusion' attribute")
        
        # Extract action chunk: we use the first n_action_steps actions
        if full_chunk.dim() == 3:
            action_chunk_norm = full_chunk[0, :self._n_action_steps].cpu().numpy()
        else:
            action_chunk_norm = full_chunk[:self._n_action_steps].cpu().numpy()
        
        # Unnormalize actions using postprocessor
        if self._postprocessor is not None:
            action_chunk_raw = []
            for i in range(action_chunk_norm.shape[0]):
                action_tensor = torch.from_numpy(action_chunk_norm[i]).float().to(self.device)
                action_unnorm = self._postprocessor(action_tensor)
                action_chunk_raw.append(action_unnorm.cpu().numpy())
            action_chunk_raw = np.array(action_chunk_raw)
        else:
            action_chunk_raw = action_chunk_norm.copy()
        
        return action_chunk_norm, action_chunk_raw
    
    def _store_observation(
        self,
        observation: Dict[str, torch.Tensor],
    ):
        """Store observation in the rolling buffer.
        
        This should be called at EVERY simulation step to maintain a rolling
        buffer of the most recent observations. The buffer is used to provide
        temporal context (n_obs_steps frames) for action chunk inference.
        
        Args:
            observation: Raw (un-preprocessed) observation dictionary.
        """
        # Store relevant parts of observation
        obs_entry = {
            'state': observation.get('observation.state', None),
            'images': {},
        }
        
        # Clone tensors to avoid issues with in-place modifications
        if obs_entry['state'] is not None:
            obs_entry['state'] = obs_entry['state'].clone()
        
        # Store images
        if hasattr(self._policy, 'config') and hasattr(self._policy.config, 'image_features'):
            for key in self._policy.config.image_features:
                if key in observation:
                    obs_entry['images'][key] = observation[key].clone()
        
        self.state.obs_queue.append(obs_entry)
    
    def step(
        self,
        observation: Dict[str, torch.Tensor],
        current_ee_pose: np.ndarray,  # (7,) in raw coordinates
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute one simulation step of the waypoint executor.
        
        This method should be called at every simulation step. It manages:
        - Generating new action chunks when needed
        - Tracking progress toward current waypoint
        - Transitioning to next waypoint when reached
        
        Args:
            observation: Current observation dictionary (raw, un-preprocessed).
                Keys: 'observation.image', 'observation.state', etc.
            current_ee_pose: Current EE pose in raw coordinates (7,).
            
        Returns:
            Tuple of (action, info):
                - action: (action_dim,) current waypoint target to send to env
                - info: Dictionary with execution state information
        """
        info = {
            'inference_this_step': False,
            'waypoint_reached': False,
            'waypoint_idx': self.state.current_waypoint_idx,
            'steps_at_waypoint': self.state.steps_at_current_waypoint,
            'total_steps': self.state.total_steps,
            'inference_count': self.state.inference_count,
        }
        
        # IMPORTANT: Store observation at EVERY step to maintain a rolling buffer
        # This ensures we always have the most recent n_obs_steps observations
        # available for the next inference
        self._store_observation(observation)
        
        # Check if we need a new action chunk
        # Use actual chunk size instead of self._n_action_steps to handle cases where
        # the generated chunk may be smaller than expected
        actual_chunk_size = (
            len(self.state.action_chunk_raw) 
            if self.state.action_chunk_raw is not None 
            else 0
        )
        need_new_chunk = (
            self.state.action_chunk is None or 
            self.state.current_waypoint_idx >= actual_chunk_size
        )
        
        if need_new_chunk:
            # Generate new action chunk using observations from the rolling buffer
            action_chunk_norm, action_chunk_raw = self._generate_action_chunk(observation)
            
            self.state.action_chunk = action_chunk_norm
            self.state.action_chunk_raw = action_chunk_raw
            self.state.current_waypoint_idx = 0
            self.state.steps_at_current_waypoint = 0
            self.state.inference_count += 1
            
            info['inference_this_step'] = True
            info['action_chunk'] = action_chunk_raw.copy()
        
        # Get current waypoint target
        current_waypoint = self.state.action_chunk_raw[self.state.current_waypoint_idx]
        
        # Update step counters
        self.state.steps_at_current_waypoint += 1
        self.state.total_steps += 1
        
        # Check if waypoint is reached (after minimum steps)
        if self.state.steps_at_current_waypoint >= self.config.min_steps_per_waypoint:
            waypoint_reached = self._check_waypoint_reached(current_ee_pose, current_waypoint)
            timeout = self.state.steps_at_current_waypoint >= self.config.max_steps_per_waypoint
            
            if waypoint_reached or timeout:
                # Record waypoint history (no need to store observation again,
                # it's already stored at the beginning of each step)
                self.state.waypoint_history.append({
                    'waypoint_idx': self.state.current_waypoint_idx,
                    'target_pose': current_waypoint.copy(),
                    'actual_pose': current_ee_pose.copy(),
                    'steps_taken': self.state.steps_at_current_waypoint,
                    'reached': waypoint_reached,
                    'timeout': timeout,
                    'total_step': self.state.total_steps,
                })
                
                # Record steps taken BEFORE resetting
                steps_taken_for_waypoint = self.state.steps_at_current_waypoint
                
                # Move to next waypoint
                self.state.current_waypoint_idx += 1
                self.state.steps_at_current_waypoint = 0
                
                info['waypoint_reached'] = True
                info['reached_by_timeout'] = timeout and not waypoint_reached
                info['steps_taken_for_waypoint'] = steps_taken_for_waypoint
        
        # Update info
        info['waypoint_idx'] = self.state.current_waypoint_idx
        info['steps_at_waypoint'] = self.state.steps_at_current_waypoint
        info['total_steps'] = self.state.total_steps
        
        return current_waypoint, info
    
    def get_current_chunk(self) -> Optional[np.ndarray]:
        """Get the current action chunk (unnormalized).
        
        Returns:
            Action chunk array (n_action_steps, action_dim) or None if no chunk.
        """
        return self.state.action_chunk_raw
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for the episode.
        
        Returns:
            Dictionary with statistics:
                - total_steps: Total simulation steps
                - inference_count: Number of action chunk inferences
                - waypoints_reached: Number of waypoints successfully reached
                - waypoints_timeout: Number of waypoints reached by timeout
                - avg_steps_per_waypoint: Average steps to reach each waypoint
        """
        waypoints_reached = sum(1 for w in self.state.waypoint_history if w['reached'])
        waypoints_timeout = sum(1 for w in self.state.waypoint_history if w['timeout'])
        
        steps_list = [w['steps_taken'] for w in self.state.waypoint_history]
        avg_steps = np.mean(steps_list) if steps_list else 0.0
        
        return {
            'total_steps': self.state.total_steps,
            'inference_count': self.state.inference_count,
            'waypoints_completed': len(self.state.waypoint_history),
            'waypoints_reached': waypoints_reached,
            'waypoints_timeout': waypoints_timeout,
            'avg_steps_per_waypoint': avg_steps,
            'waypoint_history': self.state.waypoint_history,
        }
