#!/usr/bin/env python3
"""Quick test: does the RodInsert env register correctly?"""
import sys, argparse
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym

# Try importing isaaclab_tasks
print(">>> IMPORTING isaaclab_tasks", flush=True)
try:
    import isaaclab_tasks
    print(">>> IMPORT OK", flush=True)
except Exception as e:
    print(f">>> IMPORT FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

# Check registered
specs = [s for s in gym.registry if "Rod" in s]
print(f">>> ROD_ENVS: {specs}", flush=True)

if not specs:
    print(">>> No Rod envs found. Trying direct import...", flush=True)
    try:
        from isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.lift.config.franka import rod_insert_env_cfg
        print(f">>> Direct import OK: {rod_insert_env_cfg}", flush=True)
    except Exception as e:
        print(f">>> Direct import FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

    # Re-check
    specs2 = [s for s in gym.registry if "Rod" in s]
    print(f">>> ROD_ENVS after direct: {specs2}", flush=True)

app.close()
print(">>> DONE", flush=True)
