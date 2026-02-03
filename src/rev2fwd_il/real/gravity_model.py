"""Analytical gravity compensation model for Piper robotic arm.

This module provides gravity torque calculation WITHOUT MuJoCo dependency.
Physics parameters are extracted from the verified MuJoCo model:
    piper-sdk-rs/crates/piper-physics/assets/piper_no_gripper.xml

The gravity torques are calculated analytically using the robot's kinematics
and link mass/COM properties.

Usage:
    from rev2fwd_il.real import PiperGravityModel
    
    model = PiperGravityModel()
    gravity_torques = model.compute_gravity_torques(joint_angles)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LinkParams:
    """Physical parameters for a single robot link."""
    name: str
    mass: float  # kg
    com: np.ndarray  # Center of mass in link frame [x, y, z] meters
    # Note: inertia not needed for gravity compensation, only for dynamics


# =============================================================================
# Piper Physical Parameters (from verified MuJoCo model)
# =============================================================================
# Source: piper-sdk-rs/crates/piper-physics/assets/piper_no_gripper.xml
# These are the VERIFIED parameters from the physical robot model.

PIPER_LINK_PARAMS = [
    # Link 0 (base_link) - fixed, doesn't contribute to gravity torques
    LinkParams(
        name="base_link",
        mass=1.02,
        com=np.array([-0.00473641164191482, 2.56829134630247e-05, 0.041451518036016]),
    ),
    # Link 1 - rotates around Z
    LinkParams(
        name="link1",
        mass=0.71,
        com=np.array([0.000121504734057468, 0.000104632162460536, -0.00438597309559853]),
    ),
    # Link 2 - rotates around Z (after -90° rotation about X from link1)
    LinkParams(
        name="link2",
        mass=1.17,
        com=np.array([0.198666145229743, -0.010926924140076, 0.00142121714502687]),
    ),
    # Link 3 - rotates around Z
    LinkParams(
        name="link3",
        mass=0.5,
        com=np.array([-0.0202737662122021, -0.133914995944595, -0.000458682652737356]),
    ),
    # Link 4 - rotates around Z
    LinkParams(
        name="link4",
        mass=0.38,
        com=np.array([-9.66635791618542e-05, 0.000876064475651083, -0.00496880904640868]),
    ),
    # Link 5 - rotates around Z
    LinkParams(
        name="link5",
        mass=0.383,
        com=np.array([-4.10554118924211e-05, -0.0566486692356075, -0.0037205791677906]),
    ),
    # Link 6 - end effector link
    LinkParams(
        name="link6",
        mass=0.007,
        com=np.array([-8.82590762930069e-05, 9.0598378529832e-06, -0.002]),
    ),
]

# Gripper parameters (estimated - gripper attached to link6)
# The gripper adds mass but COM doesn't change much during open/close
GRIPPER_PARAMS = LinkParams(
    name="gripper",
    mass=0.0,  # Estimated gripper mass in kg (adjust based on actual gripper)
    com=np.array([0.0, 0.0, 0.0]),  # Estimated COM offset from link6 in link6 frame
)


class PiperGravityModel:
    """Analytical gravity compensation model for Piper arm.
    
    Computes gravity torques at any joint configuration using the robot's
    link masses, COM positions, and forward kinematics.
    
    This is a standalone implementation that doesn't depend on MuJoCo.
    """
    
    # DH parameters (matching force_estimator.py for consistency)
    # Units: a in meters, alpha in radians, theta_offset in radians, d in meters
    DH_PARAMS = {
        'a':     [0.0,     0.0,       0.28503,   -0.02198,  0.0,       0.0],
        'alpha': [0.0,    -np.pi/2,   0.0,        np.pi/2, -np.pi/2,   np.pi/2],
        'theta': [0.0,    -3.0051,   -1.7941,     0.0,      0.0,       0.0],  # offset in radians
        'd':     [0.123,   0.0,       0.0,        0.25075,  0.0,       0.091],
    }
    
    def __init__(self, 
                 include_gripper: bool = True,
                 gripper_mass: Optional[float] = None,
                 gripper_com: Optional[np.ndarray] = None,
                 dh_is_offset: int = 0x01):
        """Initialize gravity model.
        
        Args:
            include_gripper: Whether to include gripper mass in calculations.
            gripper_mass: Override gripper mass (kg). If None, uses default.
            gripper_com: Override gripper COM in link6 frame. If None, uses default.
            dh_is_offset: DH parameter offset flag (0x01 = with 2° offset for j2, j3).
        """
        self.include_gripper = include_gripper
        self.link_params = list(PIPER_LINK_PARAMS)  # Copy to avoid modifying global
        
        # Setup gripper
        if include_gripper:
            gripper = LinkParams(
                name="gripper",
                mass=gripper_mass if gripper_mass is not None else GRIPPER_PARAMS.mass,
                com=gripper_com if gripper_com is not None else GRIPPER_PARAMS.com.copy(),
            )
            self.gripper_params = gripper
        else:
            self.gripper_params = None
        
        # DH parameters
        self.dh_is_offset = dh_is_offset
        if dh_is_offset == 0x01:
            self.a = np.array(self.DH_PARAMS['a'])
            self.alpha = np.array(self.DH_PARAMS['alpha'])
            self.theta_offset = np.array(self.DH_PARAMS['theta'])
            self.d = np.array(self.DH_PARAMS['d'])
        else:
            # Without offset (rare case)
            self.a = np.array([0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0])
            self.alpha = np.array([0.0, -np.pi/2, 0.0, np.pi/2, -np.pi/2, np.pi/2])
            self.theta_offset = np.array([0.0, -np.pi*174.22/180, -100.78/180*np.pi, 0.0, 0.0, 0.0])
            self.d = np.array([0.123, 0.0, 0.0, 0.25075, 0.0, 0.091])
        
        # Gravity vector in base frame (pointing down in -Z)
        self.gravity = np.array([0.0, 0.0, -9.81])
    
    def dh_transform(self, a: float, alpha: float, theta: float, d: float) -> np.ndarray:
        """Compute single DH transformation matrix.
        
        Args:
            a: Link length (m)
            alpha: Link twist (rad)
            theta: Joint angle (rad)
            d: Link offset (m)
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct,     -st,     0,   a],
            [st*ca,  ct*ca,  -sa, -sa*d],
            [st*sa,  ct*sa,   ca,  ca*d],
            [0,      0,       0,   1]
        ])
    
    def forward_kinematics(self, q: np.ndarray) -> List[np.ndarray]:
        """Compute forward kinematics for all joints.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            List of 7 transformation matrices [T_base, T_0, T_01, ..., T_06]
            Each T_0i transforms from base frame to frame i.
        """
        T = [np.eye(4)]  # Base frame
        
        for i in range(6):
            theta_i = q[i] + self.theta_offset[i]
            T_i = self.dh_transform(self.a[i], self.alpha[i], theta_i, self.d[i])
            T.append(T[i] @ T_i)
        
        return T
    
    def compute_link_com_world(self, q: np.ndarray) -> List[np.ndarray]:
        """Compute center of mass positions in world frame for all links.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            List of COM positions in world frame (one per link, starting from link1)
        """
        T = self.forward_kinematics(q)
        
        com_positions = []
        # Links 1-6 correspond to transforms T[1] through T[6]
        for i in range(1, 7):
            # Get link COM in link frame
            link_com_local = self.link_params[i].com
            # Transform to world frame
            com_homo = np.append(link_com_local, 1.0)
            com_world = (T[i] @ com_homo)[:3]
            com_positions.append(com_world)
        
        # Add gripper if included
        if self.include_gripper and self.gripper_params is not None:
            gripper_com_local = self.gripper_params.com
            com_homo = np.append(gripper_com_local, 1.0)
            gripper_com_world = (T[6] @ com_homo)[:3]
            com_positions.append(gripper_com_world)
        
        return com_positions
    
    def compute_gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques for given joint configuration.
        
        Uses the principle that gravity torque on joint i is the sum of 
        torques due to gravity forces on all links distal to joint i.
        
        τ_i = Σ_j (z_i × (p_com_j - p_i)) · (m_j * g)
        
        where:
            z_i = joint i axis in world frame
            p_com_j = COM of link j in world frame  
            p_i = origin of joint i in world frame
            m_j = mass of link j
            g = gravity vector
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            tau_gravity: Gravity torques (6,) in N·m
        """
        T = self.forward_kinematics(q)
        
        # Get link masses and COM positions in world frame
        masses = [self.link_params[i].mass for i in range(1, 7)]
        
        com_positions = []
        for i in range(1, 7):
            link_com_local = self.link_params[i].com
            com_homo = np.append(link_com_local, 1.0)
            com_world = (T[i] @ com_homo)[:3]
            com_positions.append(com_world)
        
        # Add gripper if included
        if self.include_gripper and self.gripper_params is not None:
            masses.append(self.gripper_params.mass)
            gripper_com_local = self.gripper_params.com
            com_homo = np.append(gripper_com_local, 1.0)
            gripper_com_world = (T[6] @ com_homo)[:3]
            com_positions.append(gripper_com_world)
        
        # Compute gravity torques for each joint
        tau_gravity = np.zeros(6)
        
        for joint_idx in range(6):
            # Joint axis in world frame (Z-axis of the joint frame)
            z_joint = T[joint_idx][:3, 2]
            # Joint origin in world frame
            p_joint = T[joint_idx][:3, 3]
            
            # Sum contributions from all distal links
            # Links distal to joint i are links i+1, i+2, ..., 6 (and gripper)
            for link_idx in range(joint_idx, len(masses)):
                m = masses[link_idx]
                p_com = com_positions[link_idx]
                
                # Gravity force on this link
                F_gravity = m * self.gravity
                
                # Moment arm from joint to COM
                r = p_com - p_joint
                
                # Torque contribution: τ = r × F
                torque = np.cross(r, F_gravity)
                
                # Project onto joint axis
                tau_gravity[joint_idx] += np.dot(z_joint, torque)
        
        return tau_gravity
    
    def compute_gravity_torques_with_payload(self, q: np.ndarray, 
                                              payload_mass: float,
                                              payload_com_ee: np.ndarray) -> np.ndarray:
        """Compute gravity torques including an additional payload at end-effector.
        
        Args:
            q: Joint angles (6,) in radians
            payload_mass: Mass of payload in kg
            payload_com_ee: Payload COM position in end-effector frame
            
        Returns:
            tau_gravity: Gravity torques (6,) in N·m
        """
        # First compute base gravity torques
        tau_gravity = self.compute_gravity_torques(q)
        
        if payload_mass <= 0:
            return tau_gravity
        
        # Get transforms
        T = self.forward_kinematics(q)
        
        # Payload COM in world frame
        payload_com_homo = np.append(payload_com_ee, 1.0)
        payload_com_world = (T[6] @ payload_com_homo)[:3]
        
        # Add payload contribution to each joint
        for joint_idx in range(6):
            z_joint = T[joint_idx][:3, 2]
            p_joint = T[joint_idx][:3, 3]
            
            F_gravity = payload_mass * self.gravity
            r = payload_com_world - p_joint
            torque = np.cross(r, F_gravity)
            tau_gravity[joint_idx] += np.dot(z_joint, torque)
        
        return tau_gravity


class GravityCompensator:
    """High-level gravity compensator with calibration support.
    
    This class wraps PiperGravityModel and adds:
    - Per-joint scaling factors to account for motor/transmission differences
    - Residual offset calibration
    - Filtering for smooth torque commands
    """
    
    def __init__(self,
                 include_gripper: bool = True,
                 gripper_mass: Optional[float] = None):
        """Initialize gravity compensator.
        
        Args:
            include_gripper: Whether to include gripper in model.
            gripper_mass: Override gripper mass if known.
        """
        self.model = PiperGravityModel(
            include_gripper=include_gripper,
            gripper_mass=gripper_mass,
        )
        
        # Per-joint scaling factors (can be calibrated)
        # These account for motor constants, gear ratios, friction, etc.
        # Default: based on piper_control's DIRECT_SCALING_FACTORS
        # J1-3 have 4x scaling in firmware, J4-6 are 1:1
        self._scaling = np.array([0.25, 0.25, 0.25, 1.0, 1.0, 1.0])
        
        # Per-joint offsets (can be calibrated)
        self._offsets = np.zeros(6)
        
        # Low-pass filter state
        self._filtered_tau: Optional[np.ndarray] = None
        self._filter_alpha: float = 0.3
    
    @property
    def scaling(self) -> np.ndarray:
        """Per-joint scaling factors."""
        return self._scaling.copy()
    
    @scaling.setter
    def scaling(self, value: np.ndarray):
        """Set per-joint scaling factors."""
        self._scaling = np.asarray(value).flatten()[:6]
    
    @property
    def offsets(self) -> np.ndarray:
        """Per-joint offset torques."""
        return self._offsets.copy()
    
    @offsets.setter
    def offsets(self, value: np.ndarray):
        """Set per-joint offsets."""
        self._offsets = np.asarray(value).flatten()[:6]
    
    def predict(self, q: np.ndarray, apply_filter: bool = False) -> np.ndarray:
        """Predict gravity compensation torques.
        
        Args:
            q: Joint angles (6,) in radians
            apply_filter: Whether to apply low-pass filter
            
        Returns:
            Gravity compensation torques (6,) in N·m
        """
        tau_model = self.model.compute_gravity_torques(q)
        tau_scaled = tau_model * self._scaling + self._offsets
        
        if apply_filter:
            if self._filtered_tau is None:
                self._filtered_tau = tau_scaled.copy()
            else:
                self._filtered_tau = (self._filter_alpha * tau_scaled + 
                                      (1 - self._filter_alpha) * self._filtered_tau)
            return self._filtered_tau.copy()
        
        return tau_scaled
    
    def calibrate_offsets(self, q: np.ndarray, measured_tau: np.ndarray):
        """Calibrate offset torques from measured data.
        
        Call this at a known pose with no external load.
        The offset is computed as: offset = measured - predicted
        
        Args:
            q: Joint angles (6,) in radians
            measured_tau: Measured joint torques (6,) in N·m
        """
        tau_model = self.model.compute_gravity_torques(q)
        tau_scaled = tau_model * self._scaling
        self._offsets = measured_tau - tau_scaled
        print(f"[GravityCompensator] Calibrated offsets: {self._offsets}")
    
    def calibrate_scaling(self, q_samples: np.ndarray, tau_samples: np.ndarray):
        """Calibrate scaling factors from multiple samples.
        
        Uses least-squares fit to find optimal scaling factors.
        
        Args:
            q_samples: Joint angle samples (N, 6) in radians
            tau_samples: Measured torque samples (N, 6) in N·m
        """
        n_samples = q_samples.shape[0]
        
        # Compute model predictions for all samples
        tau_model = np.array([self.model.compute_gravity_torques(q) for q in q_samples])
        
        # Fit scaling factors per joint using least squares
        for j in range(6):
            # Solve: tau_measured = scale * tau_model + offset
            A = np.column_stack([tau_model[:, j], np.ones(n_samples)])
            b = tau_samples[:, j]
            try:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                self._scaling[j] = x[0]
                self._offsets[j] = x[1]
            except np.linalg.LinAlgError:
                print(f"[GravityCompensator] Warning: lstsq failed for joint {j}")
        
        print(f"[GravityCompensator] Calibrated scaling: {self._scaling}")
        print(f"[GravityCompensator] Calibrated offsets: {self._offsets}")
    
    def reset_filter(self):
        """Reset the low-pass filter state."""
        self._filtered_tau = None
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to file.
        
        Args:
            filepath: Path to save calibration (.npz format)
        """
        np.savez(filepath,
                 scaling=self._scaling,
                 offsets=self._offsets,
                 gripper_mass=self.model.gripper_params.mass if self.model.gripper_params else 0.0)
        print(f"[GravityCompensator] Saved calibration to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration parameters from file.
        
        Args:
            filepath: Path to calibration file (.npz format)
        """
        data = np.load(filepath)
        self._scaling = data['scaling']
        self._offsets = data['offsets']
        if 'gripper_mass' in data and self.model.gripper_params is not None:
            self.model.gripper_params.mass = float(data['gripper_mass'])
        print(f"[GravityCompensator] Loaded calibration from {filepath}")
        print(f"[GravityCompensator] Scaling: {self._scaling}")
        print(f"[GravityCompensator] Offsets: {self._offsets}")
