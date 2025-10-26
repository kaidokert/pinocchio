#!/usr/bin/env python3
"""
Standalone test to reproduce SO101 simulation crash
Tests collision detection without any visualizer code.
"""

import sys
import os
import numpy as np
from pathlib import Path
import pinocchio as pin

print("="*80)
print("SO101 Collision Crash Reproduction Test")
print("Testing collision detection without visualizer")
print("="*80)

# Step 1: Load model exactly as so101-sim.py does
print("\nStep 1: Loading SO101 model...")
try:
    model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
    mesh_dir = model_path.parent.parent
    urdf_model_path = model_path / "so_arm_description/urdf/so101.urdf"

    print(f"URDF path: {urdf_model_path}")
    print(f"Mesh dir: {mesh_dir}")

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
    )
    print(f"OK Model loaded: {model.nq} DOF, {collision_model.ngeoms} collision geometries")
except Exception as e:
    print(f"FAIL Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Setup collision pairs FIRST (before creating GeometryData!)
print("\nStep 2: Setting up collision pairs...")
try:
    print(f"Before: {len(collision_model.collisionPairs)} pairs")
    collision_model.addAllCollisionPairs()
    print(f"After: {len(collision_model.collisionPairs)} pairs")
    print("OK Collision pairs setup complete")
except Exception as e:
    print(f"FAIL Collision setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create data structures AFTER collision pairs are added
print("\nStep 3: Creating data structures...")
try:
    data = pin.Data(model)
    collision_data = pin.GeometryData(collision_model)
    print("OK Data and GeometryData created")
    print("IMPORTANT: GeometryData created AFTER addAllCollisionPairs() - correct order!")
except Exception as e:
    print(f"FAIL Failed to create data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Initialize robot configuration
print("\nStep 4: Initializing robot configuration...")
q = pin.neutral(model)
q[2] = 0.0  # Z position at ground level
print(f"Initial q: {q}")

# Step 5: Test collision detection with repeated calls
print("\nStep 5: Testing collision detection with repeated calls...")
print("This mimics what happens when you move joints in so101-sim.py")

for iteration in range(10):
    print(f"\n--- Iteration {iteration+1} ---")

    # Modify configuration slightly (like moving a joint)
    if iteration > 0:
        q[7] += np.deg2rad(5)  # Move shoulder_pan by 5 degrees
        print(f"Modified q[7] (shoulder_pan): {np.rad2deg(q[7]):.2f} deg")

    # Check collisions
    print("  Calling updateGeometryPlacements...")
    try:
        pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)
        print("  OK updateGeometryPlacements succeeded")
    except Exception as e:
        print(f"  FAIL updateGeometryPlacements crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("  Calling computeCollisions...")
    try:
        is_colliding = pin.computeCollisions(collision_model, collision_data, stop_at_first_collision=True)
        print(f"  OK computeCollisions succeeded: {'COLLISION' if is_colliding else 'NO COLLISION'}")
    except Exception as e:
        print(f"  FAIL computeCollisions crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print("clean pass")

print("\n" + "="*80)
print("OK ALL ITERATIONS COMPLETED - No crash!")
print("="*80)
print("\nThis is a minimal reproduction case for the collision crash bug.")
print("The crash occurs with repeated collision checks on SO101 URDF model.")
print("No visualizer, no UI, just pure collision detection in a loop.")
print()
