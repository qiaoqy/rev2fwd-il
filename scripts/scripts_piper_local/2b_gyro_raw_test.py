#!/usr/bin/env python3
"""Test raw gyro values from pydualsense to understand the actual data format."""

import time
import numpy as np

try:
    from pydualsense import pydualsense
except ImportError:
    print("pip install pydualsense")
    exit(1)

print("=" * 60)
print("PyDualSense Raw Gyro Value Test")
print("=" * 60)

ds = pydualsense()
ds.init()
print("Connected!\n")

# ============ Phase 1: Stationary ============
print("=" * 60)
print("PHASE 1: STATIONARY TEST")
print("=" * 60)
print("\n>>> Place controller FLAT and STILL on a surface <<<")
print(">>> DO NOT TOUCH IT <<<\n")

for i in range(3, 0, -1):
    print(f"  Starting in {i}...")
    time.sleep(1)

print("\nCollecting data for 3 seconds...\n")

stationary_samples = []
start = time.time()
while time.time() - start < 3.0:
    raw_pitch = ds.state.gyro.Pitch
    raw_yaw = ds.state.gyro.Yaw
    raw_roll = ds.state.gyro.Roll
    stationary_samples.append((raw_pitch, raw_yaw, raw_roll))
    time.sleep(0.02)

arr = np.array(stationary_samples)
print(f"Collected {len(arr)} samples\n")

print("Raw value statistics (STATIONARY):")
print(f"  {'':12} {'Pitch':>12} {'Yaw':>12} {'Roll':>12}")
print(f"  {'Mean':12} {np.mean(arr[:,0]):>12.1f} {np.mean(arr[:,1]):>12.1f} {np.mean(arr[:,2]):>12.1f}")
print(f"  {'Std':12} {np.std(arr[:,0]):>12.1f} {np.std(arr[:,1]):>12.1f} {np.std(arr[:,2]):>12.1f}")
print(f"  {'Min':12} {np.min(arr[:,0]):>12.1f} {np.min(arr[:,1]):>12.1f} {np.min(arr[:,2]):>12.1f}")
print(f"  {'Max':12} {np.max(arr[:,0]):>12.1f} {np.max(arr[:,1]):>12.1f} {np.max(arr[:,2]):>12.1f}")

stationary_mean = np.mean(arr, axis=0)
stationary_std = np.std(arr, axis=0)

# ============ Phase 2: Motion ============
print("\n" + "=" * 60)
print("PHASE 2: MOTION TEST")
print("=" * 60)
print("\n>>> NOW pick up the controller and ROTATE it! <<<")
print(">>> Rotate slowly and steadily in all directions <<<\n")

input("Press ENTER when ready...")
print("\nCollecting data for 5 seconds - ROTATE NOW!\n")

motion_samples = []
start = time.time()
last_print = start
while time.time() - start < 5.0:
    raw_pitch = ds.state.gyro.Pitch
    raw_yaw = ds.state.gyro.Yaw
    raw_roll = ds.state.gyro.Roll
    motion_samples.append((raw_pitch, raw_yaw, raw_roll))
    
    # Print live values
    now = time.time()
    if now - last_print >= 0.3:
        print(f"  Raw: pitch={raw_pitch:>8.1f}  yaw={raw_yaw:>8.1f}  roll={raw_roll:>8.1f}")
        last_print = now
    
    time.sleep(0.02)

arr_motion = np.array(motion_samples)
print(f"\nCollected {len(arr_motion)} samples\n")

print("Raw value statistics (MOTION):")
print(f"  {'':12} {'Pitch':>12} {'Yaw':>12} {'Roll':>12}")
print(f"  {'Mean':12} {np.mean(arr_motion[:,0]):>12.1f} {np.mean(arr_motion[:,1]):>12.1f} {np.mean(arr_motion[:,2]):>12.1f}")
print(f"  {'Min':12} {np.min(arr_motion[:,0]):>12.1f} {np.min(arr_motion[:,1]):>12.1f} {np.min(arr_motion[:,2]):>12.1f}")
print(f"  {'Max':12} {np.max(arr_motion[:,0]):>12.1f} {np.max(arr_motion[:,1]):>12.1f} {np.max(arr_motion[:,2]):>12.1f}")
print(f"  {'Range':12} {np.ptp(arr_motion[:,0]):>12.1f} {np.ptp(arr_motion[:,1]):>12.1f} {np.ptp(arr_motion[:,2]):>12.1f}")

# ============ Analysis ============
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

mean_magnitude = np.mean(np.abs(stationary_mean))
motion_range = np.max(np.ptp(arr_motion, axis=0))

print(f"\nStationary mean magnitude: {mean_magnitude:.1f}")
print(f"Motion range (max): {motion_range:.1f}")
print(f"Stationary std (max): {np.max(stationary_std):.1f}")

print("\n" + "-" * 60)
print("INTERPRETATION:")
print("-" * 60)

# Try to determine the correct interpretation
if mean_magnitude > 5000:
    print("\n✗ Raw values have HUGE offset!")
    print("  This is abnormal. The gyro might not be properly initialized.")
    print("  pydualsense might be returning garbage values.")
    
    # Check if it's fixed offset
    if np.max(stationary_std) < 1000:
        print("\n  However, std is relatively low, so it might be a constant offset bug.")
        print(f"  Offset appears to be: pitch={stationary_mean[0]:.0f}, yaw={stationary_mean[1]:.0f}, roll={stationary_mean[2]:.0f}")
        
        # See if motion values make sense after offset removal
        motion_corrected = arr_motion - stationary_mean
        corrected_range = np.max(np.ptp(motion_corrected, axis=0))
        print(f"\n  After subtracting offset, motion range = {corrected_range:.0f}")
        
        if corrected_range > 100 and corrected_range < 10000:
            print("  ✓ This looks reasonable! The raw values are likely in some unit like:")
            if corrected_range > 2000:
                print(f"    - Raw int16 (needs ÷ ~16 to get deg/s)")
            else:
                print(f"    - Possibly already deg/s or similar")
    
elif mean_magnitude > 100:
    print(f"\n  Values seem to be in raw units (likely deg/s already).")
    print(f"  Stationary offset: {stationary_mean}")
    
elif mean_magnitude < 10:
    print(f"\n  Values are very small - might already be in rad/s.")

# Suggest correct scale
print("\n" + "-" * 60)
print("SUGGESTED CORRECTION:")
print("-" * 60)
print(f"\n  1. Subtract offset: [{stationary_mean[0]:.1f}, {stationary_mean[1]:.1f}, {stationary_mean[2]:.1f}]")

# Estimate scale based on motion range
# A moderate rotation is about 90-180 deg/s
if motion_range > 0:
    # Assuming max motion was ~200 deg/s
    estimated_scale = 200.0 / (motion_range / 2)  # Use half range as typical
    print(f"  2. If values are raw int: scale by ~{estimated_scale:.4f} to get deg/s")
    print(f"     OR if already deg/s: no additional scaling needed")

print("\n  3. Apply deadzone of ~3x the stationary std:")
print(f"     Deadzone = {3 * np.max(stationary_std):.1f} (in raw units)")

ds.close()
print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
