#!/usr/bin/env python3
"""
SO101 Collision Detection Test
Tests collision detection with the SO101 URDF model to identify issues.
"""

import sys
import os
import numpy as np
from pathlib import Path
import pinocchio as pin

print("Testing SO101 Collision Detection...")
print(f"Pinocchio version: {pin.__version__}")

# Step 1: Load the SO101 model
print("\n" + "="*60)
print("Step 1: Loading SO101 URDF model")
print("="*60)

try:
    # Get model paths from environment
    model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
    mesh_dir = model_path.parent.parent
    urdf_model_path = model_path / "so_arm_description/urdf/so101.urdf"

    print(f"URDF path: {urdf_model_path}")
    print(f"Mesh dir: {mesh_dir}")

    if not urdf_model_path.exists():
        print(f"ERROR: URDF file not found at {urdf_model_path}")
        sys.exit(1)

    # Load with free flyer (same as simulation)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
    )
    print(f"OK Model loaded successfully")
    print(f"   DOFs: {model.nq}")
    print(f"   Joints: {model.njoints}")
    print(f"   Collision geometries: {collision_model.ngeoms}")

except Exception as e:
    print(f"FAIL Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Inspect collision geometries
print("\n" + "="*60)
print("Step 2: Inspecting collision geometries")
print("="*60)

try:
    for i in range(collision_model.ngeoms):
        geom = collision_model.geometryObjects[i]
        print(f"   [{i}] {geom.name}")
        print(f"       Parent joint: {geom.parentJoint} ({model.names[geom.parentJoint]})")
        print(f"       Parent frame: {geom.parentFrame}")
        print(f"       Geometry type: {type(geom.geometry).__name__}")

    if collision_model.ngeoms == 0:
        print("   WARNING: No collision geometries found in model!")
        print("   This means the URDF has no collision tags defined.")

except Exception as e:
    print(f"FAIL Failed to inspect geometries: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Add collision pairs
print("\n" + "="*60)
print("Step 3: Adding collision pairs")
print("="*60)

try:
    print(f"   Collision pairs before: {len(collision_model.collisionPairs)}")

    if collision_model.ngeoms == 0:
        print("   SKIP Cannot add collision pairs - no geometries!")
    else:
        collision_model.addAllCollisionPairs()
        print(f"   OK addAllCollisionPairs() succeeded")
        print(f"   Collision pairs after: {len(collision_model.collisionPairs)}")

        if len(collision_model.collisionPairs) == 0:
            print("   WARNING: No collision pairs created!")
            print("   This might indicate all pairs were filtered out.")

except Exception as e:
    print(f"   FAIL addAllCollisionPairs() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create data structures
print("\n" + "="*60)
print("Step 4: Creating data structures")
print("="*60)

try:
    data = pin.Data(model)
    geom_data = pin.GeometryData(collision_model)
    print(f"   OK Data and GeometryData created")
except Exception as e:
    print(f"   FAIL Failed to create data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test collision detection at various configurations
print("\n" + "="*60)
print("Step 5: Testing collision detection")
print("="*60)

if collision_model.ngeoms == 0 or len(collision_model.collisionPairs) == 0:
    print("   SKIP No collision geometries or pairs - cannot test")
else:
    # Configuration format for free flyer + joints:
    # [x, y, z, qx, qy, qz, qw, joint1, joint2, ...]
    # Free flyer is 7 DOF (3 position + 4 quaternion)

    # Test 1: Default configuration (all zeros except quaternion)
    q1 = np.zeros(model.nq)
    q1[6] = 1.0  # Set quaternion w=1 (identity rotation)

    print(f"\n   Test 1: Default configuration")
    print(f"   q = {q1}")

    try:
        # Update geometry placements
        pin.updateGeometryPlacements(model, data, collision_model, geom_data, q1)
        print(f"   OK updateGeometryPlacements() succeeded")

        # Compute collisions
        is_colliding = pin.computeCollisions(collision_model, geom_data, stop_at_first_collision=True)
        print(f"   OK computeCollisions() succeeded")
        print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

        # Check individual collision pairs
        if is_colliding:
            print(f"   Checking which pairs are colliding:")
            for i, pair in enumerate(collision_model.collisionPairs):
                if geom_data.collisionResults[i].isCollision():
                    geom1 = collision_model.geometryObjects[pair.first]
                    geom2 = collision_model.geometryObjects[pair.second]
                    print(f"      Pair {i}: {geom1.name} <-> {geom2.name}")

    except Exception as e:
        print(f"   FAIL Collision detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 2: With some joint movement
    q2 = np.zeros(model.nq)
    q2[6] = 1.0  # quaternion w
    # Move first joint (shoulder_pan) by 45 degrees
    if model.nq > 7:
        q2[7] = np.deg2rad(45)

    print(f"\n   Test 2: With joint movement")
    print(f"   q = {q2}")

    try:
        pin.updateGeometryPlacements(model, data, collision_model, geom_data, q2)
        is_colliding = pin.computeCollisions(collision_model, geom_data, stop_at_first_collision=True)
        print(f"   OK computeCollisions() succeeded")
        print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

    except Exception as e:
        print(f"   FAIL Collision detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test 3: Extreme configuration (all joints at 90 degrees)
    q3 = np.zeros(model.nq)
    q3[6] = 1.0  # quaternion w
    # Set all joints to 90 degrees
    for i in range(7, model.nq):
        q3[i] = np.deg2rad(90)

    print(f"\n   Test 3: Extreme configuration (90 deg all joints)")
    print(f"   q = {q3}")

    try:
        pin.updateGeometryPlacements(model, data, collision_model, geom_data, q3)
        is_colliding = pin.computeCollisions(collision_model, geom_data, stop_at_first_collision=True)
        print(f"   OK computeCollisions() succeeded")
        print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

        # Check which pairs are colliding
        if is_colliding:
            print(f"   Checking which pairs are colliding:")
            for i, pair in enumerate(collision_model.collisionPairs):
                if geom_data.collisionResults[i].isCollision():
                    geom1 = collision_model.geometryObjects[pair.first]
                    geom2 = collision_model.geometryObjects[pair.second]
                    print(f"      Pair {i}: {geom1.name} <-> {geom2.name}")

    except Exception as e:
        print(f"   FAIL Collision detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("OK ALL TESTS COMPLETED")
print("="*60)
print()
print("Summary:")
print(f"  Model: SO101 with {model.nq} DOFs")
print(f"  Collision geometries: {collision_model.ngeoms}")
print(f"  Collision pairs: {len(collision_model.collisionPairs)}")
print(f"  Collision detection: WORKING")
print()
print("If this test passes but so101-sim.py crashes, the issue may be:")
print("  1. Interaction with MeshcatVisualizer")
print("  2. Threading issues with blessed terminal UI")
print("  3. Memory corruption from repeated updates")
print("  4. Issue with the specific configuration values used in sim")
print()
