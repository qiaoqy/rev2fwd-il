#!/usr/bin/env python3
"""Gyro Calibration Test Script for PS5 Controller.

This script helps find optimal gyro parameters by:
1. Calibrating when controller is stationary
2. Testing gyro response when user moves the controller
3. Finding appropriate deadzone, filter, and scale values

Usage:
    python 2a_gyro_calibration_test.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import numpy as np
from typing import Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS5", "1")

try:
    from pydualsense import pydualsense
    PYDUALSENSE_AVAILABLE = True
except ImportError:
    PYDUALSENSE_AVAILABLE = False
    print("[Error] pydualsense not installed. Install with: pip install pydualsense")
    sys.exit(1)


class GyroTester:
    """Interactive gyro calibration and testing."""
    
    def __init__(self):
        self.dualsense = None
        self.scale = (2000.0 / 32768.0) * (np.pi / 180.0)  # Raw to rad/s
        
        # Calibration data
        self.offset = [0.0, 0.0, 0.0]
        self.noise_std = [0.0, 0.0, 0.0]
        
        # Filter state
        self.filtered = [0.0, 0.0, 0.0]
        
    def connect(self) -> bool:
        """Connect to DualSense controller."""
        print("[Init] Connecting to DualSense controller...")
        try:
            self.dualsense = pydualsense()
            self.dualsense.init()
            print("[Init] ✓ Connected!")
            return True
        except Exception as e:
            print(f"[Init] ✗ Failed: {e}")
            return False
    
    def read_raw(self) -> Tuple[float, float, float]:
        """Read raw gyro values in rad/s."""
        pitch = self.dualsense.state.gyro.Pitch * self.scale
        yaw = self.dualsense.state.gyro.Yaw * self.scale
        roll = self.dualsense.state.gyro.Roll * self.scale
        return (pitch, yaw, roll)
    
    def read_calibrated(self) -> Tuple[float, float, float]:
        """Read calibrated gyro values (with offset removed)."""
        raw = self.read_raw()
        return (
            raw[0] - self.offset[0],
            raw[1] - self.offset[1],
            raw[2] - self.offset[2]
        )
    
    def calibrate_stationary(self, duration: float = 3.0) -> dict:
        """Calibrate gyro by measuring stationary readings.
        
        Returns statistics about the calibration.
        """
        print(f"\n{'='*60}")
        print("STATIONARY CALIBRATION")
        print(f"{'='*60}")
        print("Place the controller on a FLAT SURFACE and don't touch it!")
        print(f"Calibrating for {duration} seconds...")
        print()
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...")
            time.sleep(1.0)
        
        print("  Calibrating... DO NOT MOVE!")
        
        # Collect samples
        samples = []
        start = time.time()
        while time.time() - start < duration:
            samples.append(self.read_raw())
            time.sleep(0.01)  # 100Hz
        
        samples = np.array(samples)
        n = len(samples)
        
        # Statistics
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        min_val = np.min(samples, axis=0)
        max_val = np.max(samples, axis=0)
        median = np.median(samples, axis=0)
        
        # Use median as offset (more robust to outliers)
        self.offset = list(median)
        self.noise_std = list(std)
        
        print(f"\n  Collected {n} samples")
        print(f"\n  Raw Statistics (°/s):")
        print(f"  {'':12} {'Pitch':>12} {'Yaw':>12} {'Roll':>12}")
        print(f"  {'Mean':12} {np.rad2deg(mean[0]):>12.1f} {np.rad2deg(mean[1]):>12.1f} {np.rad2deg(mean[2]):>12.1f}")
        print(f"  {'Median':12} {np.rad2deg(median[0]):>12.1f} {np.rad2deg(median[1]):>12.1f} {np.rad2deg(median[2]):>12.1f}")
        print(f"  {'Std Dev':12} {np.rad2deg(std[0]):>12.1f} {np.rad2deg(std[1]):>12.1f} {np.rad2deg(std[2]):>12.1f}")
        print(f"  {'Min':12} {np.rad2deg(min_val[0]):>12.1f} {np.rad2deg(min_val[1]):>12.1f} {np.rad2deg(min_val[2]):>12.1f}")
        print(f"  {'Max':12} {np.rad2deg(max_val[0]):>12.1f} {np.rad2deg(max_val[1]):>12.1f} {np.rad2deg(max_val[2]):>12.1f}")
        
        # Recommended deadzone (3 sigma)
        recommended_dz = max(std) * 3
        print(f"\n  Recommended deadzone: {np.rad2deg(recommended_dz):.1f}°/s ({recommended_dz:.3f} rad/s)")
        
        # Check if gyro is noisy
        max_std_deg = np.rad2deg(max(std))
        if max_std_deg > 100:
            print(f"\n  ⚠ WARNING: Very high noise ({max_std_deg:.0f}°/s)!")
            print("     The gyro sensor may be defective or there's a connection issue.")
            print("     Try: 1) Reconnect controller  2) Use USB instead of Bluetooth")
        elif max_std_deg > 30:
            print(f"\n  ⚠ Moderate noise detected ({max_std_deg:.0f}°/s)")
        else:
            print(f"\n  ✓ Low noise level ({max_std_deg:.0f}°/s) - Good!")
        
        return {
            'offset': self.offset,
            'std': self.noise_std,
            'recommended_deadzone': recommended_dz,
            'n_samples': n
        }
    
    def test_motion(self, duration: float = 10.0, deadzone: float = 0.5, alpha: float = 0.2):
        """Test gyro with motion.
        
        Args:
            duration: Test duration in seconds
            deadzone: Deadzone threshold in rad/s
            alpha: Low-pass filter coefficient (0-1, lower = more smoothing)
        """
        print(f"\n{'='*60}")
        print("MOTION TEST")
        print(f"{'='*60}")
        print(f"Parameters: deadzone={np.rad2deg(deadzone):.1f}°/s, alpha={alpha}")
        print()
        print("Now PICK UP the controller and move it around!")
        print("Try slow and fast rotations in all directions.")
        print(f"Test will run for {duration} seconds.")
        print()
        print("Press ENTER to start...")
        input()
        
        # Reset filter
        self.filtered = [0.0, 0.0, 0.0]
        
        # Track statistics
        raw_readings = []
        filtered_readings = []
        
        start = time.time()
        last_print = start
        
        print(f"{'Time':>6}  {'--- Raw (°/s) ---':^36}  {'--- Filtered (°/s) ---':^36}")
        print(f"{'':>6}  {'Pitch':>12}{'Yaw':>12}{'Roll':>12}  {'Pitch':>12}{'Yaw':>12}{'Roll':>12}")
        print("-" * 90)
        
        while time.time() - start < duration:
            # Read calibrated values
            raw = self.read_calibrated()
            
            # Apply deadzone
            dz_applied = list(raw)
            for i in range(3):
                if abs(dz_applied[i]) < deadzone:
                    dz_applied[i] = 0.0
            
            # Apply low-pass filter
            for i in range(3):
                self.filtered[i] = alpha * dz_applied[i] + (1 - alpha) * self.filtered[i]
            
            raw_readings.append(raw)
            filtered_readings.append(list(self.filtered))
            
            # Print every 0.5 seconds
            now = time.time()
            if now - last_print >= 0.5:
                t = now - start
                print(f"{t:6.1f}  {np.rad2deg(raw[0]):>+12.1f}{np.rad2deg(raw[1]):>+12.1f}{np.rad2deg(raw[2]):>+12.1f}  "
                      f"{np.rad2deg(self.filtered[0]):>+12.1f}{np.rad2deg(self.filtered[1]):>+12.1f}{np.rad2deg(self.filtered[2]):>+12.1f}")
                last_print = now
            
            time.sleep(0.033)  # ~30Hz
        
        # Summary
        raw_arr = np.array(raw_readings)
        filt_arr = np.array(filtered_readings)
        
        print(f"\n{'='*60}")
        print("MOTION TEST RESULTS")
        print(f"{'='*60}")
        print(f"\n  Raw Range (°/s):")
        print(f"    Pitch: {np.rad2deg(np.min(raw_arr[:,0])):>+8.1f} to {np.rad2deg(np.max(raw_arr[:,0])):>+8.1f}")
        print(f"    Yaw:   {np.rad2deg(np.min(raw_arr[:,1])):>+8.1f} to {np.rad2deg(np.max(raw_arr[:,1])):>+8.1f}")
        print(f"    Roll:  {np.rad2deg(np.min(raw_arr[:,2])):>+8.1f} to {np.rad2deg(np.max(raw_arr[:,2])):>+8.1f}")
        
        print(f"\n  Filtered Range (°/s):")
        print(f"    Pitch: {np.rad2deg(np.min(filt_arr[:,0])):>+8.1f} to {np.rad2deg(np.max(filt_arr[:,0])):>+8.1f}")
        print(f"    Yaw:   {np.rad2deg(np.min(filt_arr[:,1])):>+8.1f} to {np.rad2deg(np.max(filt_arr[:,1])):>+8.1f}")
        print(f"    Roll:  {np.rad2deg(np.min(filt_arr[:,2])):>+8.1f} to {np.rad2deg(np.max(filt_arr[:,2])):>+8.1f}")
        
        # Recommend scale based on max filtered value
        max_filt = max(np.max(np.abs(filt_arr[:,0])), np.max(np.abs(filt_arr[:,1])), np.max(np.abs(filt_arr[:,2])))
        if max_filt > 0.1:
            # Target: max rotation ~0.1 rad/cycle at max input
            recommended_scale = 0.1 / max_filt
        else:
            recommended_scale = 0.05
        
        print(f"\n  Recommended gyro_scale: {recommended_scale:.4f}")
        print(f"    (This would give ~6° rotation per cycle at your maximum input)")
    
    def interactive_test(self):
        """Run interactive parameter tuning."""
        print(f"\n{'='*60}")
        print("INTERACTIVE PARAMETER TUNING")
        print(f"{'='*60}")
        print()
        print("This will help you find good parameters.")
        print("Hold controller STILL and watch the 'Calibrated' values.")
        print("They should stay close to 0 when not moving.")
        print()
        print("Controls:")
        print("  q / w  : Decrease / Increase DEADZONE")
        print("  a / s  : Decrease / Increase ALPHA (filter)")
        print("  z / x  : Decrease / Increase SCALE")
        print("  c      : Re-calibrate")
        print("  e      : Quit")
        print()
        
        deadzone = 0.5  # rad/s (~30°/s)
        alpha = 0.15
        scale = 0.03
        self.filtered = [0.0, 0.0, 0.0]
        
        import select
        import tty
        import termios
        
        # Set terminal to non-blocking
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            print(f"Starting... (deadzone={np.rad2deg(deadzone):.0f}°/s, alpha={alpha:.2f}, scale={scale:.3f})")
            print()
            
            last_print = 0
            running = True
            
            while running:
                # Check for input
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'e':
                        running = False
                    elif key == 'q':
                        deadzone = max(0.1, deadzone - 0.2)
                        print(f"\r  Deadzone: {np.rad2deg(deadzone):.0f}°/s ({deadzone:.2f} rad/s)                    ")
                    elif key == 'w':
                        deadzone = min(20.0, deadzone + 0.2)  # Max ~1150°/s
                        print(f"\r  Deadzone: {np.rad2deg(deadzone):.0f}°/s ({deadzone:.2f} rad/s)                    ")
                    elif key == 'a':
                        alpha = max(0.01, alpha - 0.02)
                        print(f"\r  Alpha: {alpha:.2f} (more smoothing)                    ")
                    elif key == 's':
                        alpha = min(0.9, alpha + 0.02)
                        print(f"\r  Alpha: {alpha:.2f} (less smoothing)                    ")
                    elif key == 'z':
                        scale = max(0.001, scale - 0.005)
                        print(f"\r  Scale: {scale:.3f} (slower rotation)                    ")
                    elif key == 'x':
                        scale = min(0.2, scale + 0.005)
                        print(f"\r  Scale: {scale:.3f} (faster rotation)                    ")
                    elif key == 'c':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        self.calibrate_stationary(2.0)
                        tty.setcbreak(sys.stdin.fileno())
                        self.filtered = [0.0, 0.0, 0.0]
                
                # Read and process gyro
                raw = self.read_calibrated()
                
                # Apply deadzone
                dz_applied = list(raw)
                for i in range(3):
                    if abs(dz_applied[i]) < deadzone:
                        dz_applied[i] = 0.0
                
                # Apply filter
                for i in range(3):
                    self.filtered[i] = alpha * dz_applied[i] + (1 - alpha) * self.filtered[i]
                
                # Calculate scaled output (what would be sent to robot)
                scaled = [f * scale for f in self.filtered]
                
                # Print periodically
                now = time.time()
                if now - last_print >= 0.1:
                    # Use magnitude to show activity
                    raw_mag = np.sqrt(raw[0]**2 + raw[1]**2 + raw[2]**2)
                    filt_mag = np.sqrt(self.filtered[0]**2 + self.filtered[1]**2 + self.filtered[2]**2)
                    scaled_mag = filt_mag * scale
                    
                    # Visual bar
                    bar_len = min(40, int(np.rad2deg(filt_mag) / 5))
                    bar = '█' * bar_len + '░' * (40 - bar_len)
                    
                    print(f"\r  Raw:{np.rad2deg(raw_mag):>6.1f}°/s  Filt:{np.rad2deg(filt_mag):>6.1f}°/s  Out:{np.rad2deg(scaled_mag):>5.2f}°  [{bar}]  ", end='', flush=True)
                    last_print = now
                
                time.sleep(0.02)
        
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        print(f"\n\n{'='*60}")
        print("FINAL RECOMMENDED PARAMETERS")
        print(f"{'='*60}")
        print(f"  _gyro_dynamic_deadzone = {deadzone:.2f}  # {np.rad2deg(deadzone):.0f}°/s")
        print(f"  _gyro_alpha = {alpha:.2f}")
        print(f"  _gyro_scale = {scale:.3f}")
        print()
        print("Copy these values to 2_teleop_ps5_controller.py")
        print()
    
    def close(self):
        """Close connection."""
        if self.dualsense:
            self.dualsense.close()


def main():
    print("=" * 60)
    print("PS5 DualSense Gyro Calibration & Test Tool")
    print("=" * 60)
    
    tester = GyroTester()
    
    if not tester.connect():
        sys.exit(1)
    
    try:
        while True:
            print("\n" + "=" * 60)
            print("MENU")
            print("=" * 60)
            print("  1. Stationary Calibration (measure noise)")
            print("  2. Motion Test (measure range)")
            print("  3. Interactive Parameter Tuning")
            print("  4. Quick Test (calibrate + motion)")
            print("  q. Quit")
            print()
            
            choice = input("Select option: ").strip().lower()
            
            if choice == '1':
                tester.calibrate_stationary(3.0)
            
            elif choice == '2':
                dz = input("  Deadzone (°/s, default 30): ").strip()
                dz = float(dz) if dz else 30.0
                dz_rad = np.deg2rad(dz)
                
                alpha = input("  Alpha (0-1, default 0.15): ").strip()
                alpha = float(alpha) if alpha else 0.15
                
                tester.test_motion(10.0, dz_rad, alpha)
            
            elif choice == '3':
                # Need to calibrate first
                if tester.offset == [0.0, 0.0, 0.0]:
                    tester.calibrate_stationary(2.0)
                tester.interactive_test()
            
            elif choice == '4':
                tester.calibrate_stationary(3.0)
                print("\n" + "-" * 60)
                tester.test_motion(10.0, 0.5, 0.15)
            
            elif choice == 'q':
                break
            
            else:
                print("Invalid option")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    finally:
        tester.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
