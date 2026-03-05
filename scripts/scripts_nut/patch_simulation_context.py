"""Apply camera rendering patch to Isaac Lab's simulation_context.py.

This script fixes the `forward()` method so that camera images update properly
when `--disable_fabric 1` is used. See README.md Patch 2 for full explanation.

Run with the target conda environment activated:
    conda activate rev2fwd_il
    python scripts/scripts_nut/patch_simulation_context.py
"""

import os
import sys


def main():
    try:
        import isaaclab.sim.simulation_context as m
    except ImportError:
        print("ERROR: Cannot import isaaclab.sim.simulation_context")
        print("Make sure the correct conda environment is activated.")
        sys.exit(1)

    simctx_path = os.path.abspath(m.__file__)
    print(f"Target file: {simctx_path}")

    with open(simctx_path, "r") as f:
        content = f.read()

    # Check if already patched
    if "[PATCHED]" in content:
        print("Already patched — skipping.")
        return

    # The original forward() code to find and replace
    ORIGINAL = (
        "    def forward(self) -> None:\n"
        '        """Updates articulation kinematics and fabric for rendering."""\n'
        "        if self._fabric_iface is not None:\n"
        "            if self.physics_sim_view is not None and self.is_playing():\n"
        "                # Update the articulations' link's poses before rendering\n"
        "                self.physics_sim_view.update_articulations_kinematic()\n"
        "            self._update_fabric(0.0, 0.0)"
    )

    PATCHED = (
        "    def forward(self) -> None:\n"
        '        """Updates articulation kinematics and fabric for rendering."""\n'
        "        # [PATCHED] Always sync articulation link poses to renderer, even without fabric.\n"
        "        # Original code gated this on _fabric_iface, causing stale camera images\n"
        "        # when disable_fabric=1 (Camera sensor requirement).\n"
        "        if self.physics_sim_view is not None and self.is_playing():\n"
        "            self.physics_sim_view.update_articulations_kinematic()\n"
        "        if self._fabric_iface is not None:\n"
        "            self._update_fabric(0.0, 0.0)\n"
        "        else:\n"
        "            # [PATCHED] When Isaac Lab's fabric is disabled (disable_fabric=1), we still\n"
        "            # need to sync PhysX data to the renderer via the physx fabric interface.\n"
        "            # The base class SimulationContext.render() does this lazily, but since Isaac Lab\n"
        "            # overrides render() and calls forward() instead of super().render(), we handle it here.\n"
        "            if not hasattr(self, '_physx_fabric_fallback'):\n"
        "                self._physx_fabric_fallback = None\n"
        "                try:\n"
        '                    if self._extension_manager.is_extension_enabled("omni.physx.fabric"):\n'
        "                        from omni.physxfabric import get_physx_fabric_interface\n"
        "                        self._physx_fabric_fallback = get_physx_fabric_interface()\n"
        "                except Exception:\n"
        "                    pass\n"
        "            if self._physx_fabric_fallback is not None:\n"
        "                self._physx_fabric_fallback.force_update(0.0, 0.0)"
    )

    if ORIGINAL not in content:
        print("ERROR: Could not find the original forward() code to patch.")
        print("The file may have been modified or the Isaac Lab version is different.")
        print("Please apply the patch manually — see README.md for the exact code.")
        sys.exit(1)

    # Back up original
    backup_path = simctx_path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f_bak:
            f_bak.write(content)
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Apply patch
    content = content.replace(ORIGINAL, PATCHED, 1)
    with open(simctx_path, "w") as f:
        f.write(content)
    print("Patch applied successfully.")

    # Clear .pyc cache
    cache_dir = os.path.join(os.path.dirname(simctx_path), "__pycache__")
    if os.path.isdir(cache_dir):
        cleared = 0
        for fname in os.listdir(cache_dir):
            if fname.startswith("simulation_context"):
                os.remove(os.path.join(cache_dir, fname))
                cleared += 1
        if cleared:
            print(f"Cleared {cleared} cached .pyc file(s).")

    print("Done. Camera images should now update when using --disable_fabric 1.")


if __name__ == "__main__":
    main()
