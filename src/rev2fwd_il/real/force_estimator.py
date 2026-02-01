"""End-effector force estimation for Piper robotic arm.

This module provides force estimation based on joint torques and Jacobian matrix.
The estimation uses the relationship: τ = J^T * F_ext

Note: This is an approximation without gravity compensation. For better accuracy,
consider calibrating gravity torques at different poses.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

# Import Piper FK class for DH parameters
try:
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    PIPER_FK_AVAILABLE = True
except ImportError:
    PIPER_FK_AVAILABLE = False
    print("[Warning] piper_sdk.kinematics not available, using built-in DH parameters")

if TYPE_CHECKING:
    from piper_sdk import C_PiperInterface_V2


class PiperJacobian:
    """Jacobian matrix computation for Piper robotic arm.
    
    Based on DH parameters from piper_sdk/kinematics/piper_fk.py.
    """
    
    # DH parameters (with 2° offset for j2, j3)
    # Units: a in meters, alpha in radians, theta_offset in radians, d in meters
    DH_PARAMS = {
        'a':     [0.0,     0.0,       0.28503,   -0.02198,  0.0,       0.0],
        'alpha': [0.0,    -np.pi/2,   0.0,        np.pi/2, -np.pi/2,   np.pi/2],
        'theta': [0.0,    -3.0051,   -1.7941,     0.0,      0.0,       0.0],  # offset in radians
        'd':     [0.123,   0.0,       0.0,        0.25075,  0.0,       0.091],
    }
    
    def __init__(self, dh_is_offset: int = 0x01):
        """Initialize Jacobian calculator.
        
        Args:
            dh_is_offset: Whether j2, j3 have 2° offset in DH parameters.
                          0x01 = offset applied (default for most Piper arms)
                          0x00 = no offset
        """
        self.dh_is_offset = dh_is_offset
        
        # Load DH parameters from SDK if available
        if PIPER_FK_AVAILABLE:
            fk = C_PiperForwardKinematics(dh_is_offset)
            self.a = np.array(fk._a) / 1000.0  # mm -> m
            self.alpha = np.array(fk._alpha)
            self.theta_offset = np.array(fk._theta)
            self.d = np.array(fk._d) / 1000.0  # mm -> m
        else:
            # Use built-in parameters (with offset)
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
    
    def forward_kinematics(self, q: np.ndarray) -> list:
        """Compute forward kinematics for all joints.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            List of 7 transformation matrices [T_base, T_0, T_01, ..., T_06]
        """
        T = [np.eye(4)]  # Base frame
        
        for i in range(6):
            theta_i = q[i] + self.theta_offset[i]
            T_i = self.dh_transform(self.a[i], self.alpha[i], theta_i, self.d[i])
            T.append(T[i] @ T_i)
        
        return T
    
    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute geometric Jacobian matrix.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            J: 6x6 Jacobian matrix
               Rows 0-2: Linear velocity (m/rad)
               Rows 3-5: Angular velocity (rad/rad)
        """
        T = self.forward_kinematics(q)
        
        # End-effector position
        p_e = T[6][:3, 3]
        
        # Build Jacobian
        J = np.zeros((6, 6))
        
        for i in range(6):
            # Z-axis of joint i in base frame
            z_i = T[i][:3, 2]
            # Origin of joint i in base frame
            p_i = T[i][:3, 3]
            
            # Linear velocity contribution: z_i × (p_e - p_i)
            J[:3, i] = np.cross(z_i, p_e - p_i)
            # Angular velocity contribution: z_i
            J[3:, i] = z_i
        
        return J
    
    def get_end_effector_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and rotation matrix.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            position: (3,) position in meters
            rotation: (3, 3) rotation matrix
        """
        T = self.forward_kinematics(q)
        return T[6][:3, 3], T[6][:3, :3]


class PiperForceEstimator:
    """End-effector force estimator for Piper robotic arm.
    
    Estimates external forces/torques at end-effector from joint torques
    using the Jacobian transpose relationship: τ = J^T * F_ext
    
    Note: Without proper gravity compensation, the estimated forces will
    include gravitational effects. For better results:
    1. Calibrate gravity torques at various poses
    2. Use a low-pass filter on torque readings
    3. Consider only force changes (delta) rather than absolute values
    """
    
    def __init__(self, dh_is_offset: int = 0x01):
        """Initialize force estimator.
        
        Args:
            dh_is_offset: DH parameter offset flag (see PiperJacobian)
        """
        self.jacobian = PiperJacobian(dh_is_offset)
        
        # Gravity compensation baseline (can be calibrated)
        self._gravity_baseline: Optional[np.ndarray] = None
        self._baseline_pose: Optional[np.ndarray] = None
        
        # Low-pass filter state
        self._filtered_torque: Optional[np.ndarray] = None
        self._filter_alpha: float = 0.3  # EMA smoothing factor
    
    def set_gravity_baseline(self, q: np.ndarray, tau: np.ndarray):
        """Set gravity compensation baseline at current pose.
        
        Call this when the arm is stationary with no external load.
        
        Args:
            q: Joint angles (6,) in radians
            tau: Measured joint torques (6,) in N·m
        """
        self._gravity_baseline = tau.copy()
        self._baseline_pose = q.copy()
        print(f"[ForceEstimator] Gravity baseline set at pose: {np.rad2deg(q)}")
    
    def estimate_force(self, q: np.ndarray, tau: np.ndarray, 
                       use_filter: bool = True,
                       use_gravity_comp: bool = True) -> np.ndarray:
        """Estimate end-effector force from joint torques.
        
        Args:
            q: Joint angles (6,) in radians
            tau: Measured joint torques (6,) in N·m
            use_filter: Apply low-pass filter to torque readings
            use_gravity_comp: Subtract gravity baseline if available
            
        Returns:
            F_ext: Estimated external force/torque (6,)
                   [Fx, Fy, Fz, Mx, My, Mz]
                   Forces in N, torques in N·m
        """
        # Apply low-pass filter
        if use_filter:
            if self._filtered_torque is None:
                self._filtered_torque = tau.copy()
            else:
                self._filtered_torque = (self._filter_alpha * tau + 
                                         (1 - self._filter_alpha) * self._filtered_torque)
            tau_filtered = self._filtered_torque
        else:
            tau_filtered = tau
        
        # Gravity compensation
        tau_ext = tau_filtered.copy()
        if use_gravity_comp and self._gravity_baseline is not None:
            tau_ext = tau_filtered - self._gravity_baseline
        
        # Compute Jacobian
        J = self.jacobian.compute_jacobian(q)
        
        # Estimate force: F = (J^T)^{-1} * tau
        # Use pseudo-inverse for numerical stability
        try:
            J_T_pinv = np.linalg.pinv(J.T)
            F_ext = J_T_pinv @ tau_ext
        except np.linalg.LinAlgError:
            # Singular configuration
            F_ext = np.zeros(6)
        
        return F_ext
    
    def estimate_force_from_piper(self, piper: "C_PiperInterface_V2",
                                   use_filter: bool = True,
                                   use_gravity_comp: bool = True) -> Optional[np.ndarray]:
        """Estimate end-effector force directly from Piper interface.
        
        Args:
            piper: Piper SDK interface instance
            use_filter: Apply low-pass filter
            use_gravity_comp: Use gravity compensation
            
        Returns:
            F_ext: Estimated force (6,) or None if reading failed
        """
        try:
            # Get joint angles
            joint_msgs = piper.GetArmJointMsgs()
            q = np.deg2rad(np.array([
                joint_msgs.joint_state.joint_1 / 1000.0,
                joint_msgs.joint_state.joint_2 / 1000.0,
                joint_msgs.joint_state.joint_3 / 1000.0,
                joint_msgs.joint_state.joint_4 / 1000.0,
                joint_msgs.joint_state.joint_5 / 1000.0,
                joint_msgs.joint_state.joint_6 / 1000.0,
            ]))
            
            # Get joint torques
            high_spd = piper.GetArmHighSpdInfoMsgs()
            tau = np.array([
                high_spd.motor_1.effort / 1000.0,
                high_spd.motor_2.effort / 1000.0,
                high_spd.motor_3.effort / 1000.0,
                high_spd.motor_4.effort / 1000.0,
                high_spd.motor_5.effort / 1000.0,
                high_spd.motor_6.effort / 1000.0,
            ])
            
            return self.estimate_force(q, tau, use_filter, use_gravity_comp)
        
        except Exception as e:
            print(f"[ForceEstimator] Error reading from Piper: {e}")
            return None
    
    def get_joint_state_from_piper(self, piper: "C_PiperInterface_V2") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get joint angles and torques from Piper interface.
        
        Args:
            piper: Piper SDK interface instance
            
        Returns:
            (q, tau): Joint angles (rad) and torques (N·m), or None if failed
        """
        try:
            # Get joint angles
            joint_msgs = piper.GetArmJointMsgs()
            q = np.deg2rad(np.array([
                joint_msgs.joint_state.joint_1 / 1000.0,
                joint_msgs.joint_state.joint_2 / 1000.0,
                joint_msgs.joint_state.joint_3 / 1000.0,
                joint_msgs.joint_state.joint_4 / 1000.0,
                joint_msgs.joint_state.joint_5 / 1000.0,
                joint_msgs.joint_state.joint_6 / 1000.0,
            ]))
            
            # Get joint torques
            high_spd = piper.GetArmHighSpdInfoMsgs()
            tau = np.array([
                high_spd.motor_1.effort / 1000.0,
                high_spd.motor_2.effort / 1000.0,
                high_spd.motor_3.effort / 1000.0,
                high_spd.motor_4.effort / 1000.0,
                high_spd.motor_5.effort / 1000.0,
                high_spd.motor_6.effort / 1000.0,
            ])
            
            return q, tau
        
        except Exception as e:
            print(f"[ForceEstimator] Error: {e}")
            return None
    
    def calibrate_gravity_from_piper(self, piper: "C_PiperInterface_V2") -> bool:
        """Calibrate gravity baseline from current Piper state.
        
        Call this when the arm is stationary with no external load.
        
        Args:
            piper: Piper SDK interface instance
            
        Returns:
            True if calibration successful
        """
        result = self.get_joint_state_from_piper(piper)
        if result is None:
            return False
        
        q, tau = result
        self.set_gravity_baseline(q, tau)
        return True
    
    def reset_filter(self):
        """Reset the low-pass filter state."""
        self._filtered_torque = None
