"""End-effector force estimation for Piper robotic arm.

This module provides force estimation based on joint torques and Jacobian matrix.
The estimation uses the relationship: τ = J^T * F_ext

Gravity compensation is provided by the PiperGravityModel which uses analytically
computed gravity torques based on verified physics parameters from the MuJoCo model.
This approach avoids runtime MuJoCo dependency while maintaining accuracy.

Physics parameters source: piper-sdk-rs/crates/piper-physics/assets/piper_no_gripper.xml
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

# Import Piper FK class for DH parameters (optional, fallback to built-in)
try:
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    PIPER_FK_AVAILABLE = True
except ImportError:
    PIPER_FK_AVAILABLE = False

# Import gravity model
from .gravity_model import PiperGravityModel, GravityCompensator

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
    
    Gravity compensation is provided by PiperGravityModel which uses analytical
    calculation based on verified physics parameters. This avoids the need for
    single-pose calibration and works across the entire workspace.
    
    For best accuracy:
    1. Use analytical gravity compensation (use_model_gravity=True)
    2. Optionally calibrate scaling factors with calibrate_gravity_from_samples()
    3. Apply low-pass filter on torque readings
    """
    
    def __init__(self, 
                 dh_is_offset: int = 0x01,
                 include_gripper: bool = True,
                 gripper_mass: Optional[float] = None,
                 use_model_gravity: bool = True):
        """Initialize force estimator.
        
        Args:
            dh_is_offset: DH parameter offset flag (see PiperJacobian)
            include_gripper: Whether to include gripper mass in gravity model
            gripper_mass: Override gripper mass (kg) if known
            use_model_gravity: Use analytical gravity model instead of single-pose baseline
        """
        self.jacobian = PiperJacobian(dh_is_offset)
        
        # Analytical gravity compensation model
        self._use_model_gravity = use_model_gravity
        if use_model_gravity:
            self._gravity_compensator = GravityCompensator(
                include_gripper=include_gripper,
                gripper_mass=gripper_mass,
            )
        else:
            self._gravity_compensator = None
        
        # Legacy: single-pose gravity compensation baseline (fallback)
        self._gravity_baseline: Optional[np.ndarray] = None
        self._baseline_pose: Optional[np.ndarray] = None
        
        # Low-pass filter state
        self._filtered_torque: Optional[np.ndarray] = None
        self._filter_alpha: float = 0.3  # EMA smoothing factor
    
    @property
    def gravity_compensator(self) -> Optional[GravityCompensator]:
        """Access the gravity compensator for calibration."""
        return self._gravity_compensator
    
    def set_gravity_baseline(self, q: np.ndarray, tau: np.ndarray):
        """Set gravity compensation baseline at current pose (legacy method).
        
        This is a fallback for single-pose calibration. For better accuracy,
        use the analytical gravity model with use_model_gravity=True.
        
        Call this when the arm is stationary with no external load.
        
        Args:
            q: Joint angles (6,) in radians
            tau: Measured joint torques (6,) in N·m
        """
        self._gravity_baseline = tau.copy()
        self._baseline_pose = q.copy()
        
        # If using model gravity, calibrate offsets instead
        if self._use_model_gravity and self._gravity_compensator is not None:
            self._gravity_compensator.calibrate_offsets(q, tau)
        else:
            print(f"[ForceEstimator] Gravity baseline set at pose: {np.rad2deg(q)}")
    
    def calibrate_gravity_from_samples(self, q_samples: np.ndarray, tau_samples: np.ndarray):
        """Calibrate gravity compensation from multiple pose samples.
        
        This provides better accuracy than single-pose calibration by fitting
        scaling factors and offsets across the workspace.
        
        Args:
            q_samples: Joint angle samples (N, 6) in radians
            tau_samples: Measured torque samples (N, 6) in N·m
        """
        if self._gravity_compensator is not None:
            self._gravity_compensator.calibrate_scaling(q_samples, tau_samples)
        else:
            print("[ForceEstimator] Warning: No gravity compensator, using first sample as baseline")
            if len(q_samples) > 0:
                self.set_gravity_baseline(q_samples[0], tau_samples[0])
    
    def estimate_force(self, q: np.ndarray, tau: np.ndarray, 
                       use_filter: bool = True,
                       use_gravity_comp: bool = True) -> np.ndarray:
        """Estimate end-effector force from joint torques.
        
        Args:
            q: Joint angles (6,) in radians
            tau: Measured joint torques (6,) in N·m
            use_filter: Apply low-pass filter to torque readings
            use_gravity_comp: Apply gravity compensation
            
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
        if use_gravity_comp:
            if self._use_model_gravity and self._gravity_compensator is not None:
                # Use analytical gravity model
                tau_gravity = self._gravity_compensator.predict(q, apply_filter=False)
                tau_ext = tau_filtered - tau_gravity
            elif self._gravity_baseline is not None:
                # Fallback: single-pose baseline
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
    
    def get_gravity_torques(self, q: np.ndarray) -> Optional[np.ndarray]:
        """Get predicted gravity torques at given pose.
        
        Args:
            q: Joint angles (6,) in radians
            
        Returns:
            Predicted gravity torques (6,) in N·m, or None if model not available
        """
        if self._gravity_compensator is not None:
            return self._gravity_compensator.predict(q, apply_filter=False)
        return None
    
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
            if joint_msgs is None:
                print("[ForceEstimator] DEBUG: GetArmJointMsgs returned None")
                return None
            
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
            if high_spd is None:
                print("[ForceEstimator] DEBUG: GetArmHighSpdInfoMsgs returned None")
                return None
            
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
            print(f"[ForceEstimator] Error in get_joint_state_from_piper: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calibrate_gravity_from_piper(self, piper: "C_PiperInterface_V2") -> bool:
        """Calibrate gravity baseline from current Piper state.
        
        Call this when the arm is stationary with no external load.
        This calibrates offsets for the analytical gravity model if enabled,
        or sets a single-pose baseline otherwise.
        
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
        if self._gravity_compensator is not None:
            self._gravity_compensator.reset_filter()
    
    def save_gravity_calibration(self, filepath: str):
        """Save gravity calibration parameters to file.
        
        Args:
            filepath: Path to save calibration (.npz format)
        """
        if self._gravity_compensator is not None:
            self._gravity_compensator.save_calibration(filepath)
        else:
            print("[ForceEstimator] No gravity compensator to save")
    
    def load_gravity_calibration(self, filepath: str):
        """Load gravity calibration parameters from file.
        
        Args:
            filepath: Path to calibration file (.npz format)
        """
        if self._gravity_compensator is not None:
            self._gravity_compensator.load_calibration(filepath)
        else:
            print("[ForceEstimator] No gravity compensator to load into")
