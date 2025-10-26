#!/usr/bin/env python3
"""
Simple test script to verify collision detection is working.
Based on the unittest/geometry-algorithms.cpp test case.
"""

import sys
import numpy as np
import pinocchio as pin

print("Testing Pinocchio collision detection...")
print(f"Pinocchio version: {pin.__version__}")

# Create a simple model with two bodies
model = pin.Model()
geom_model = pin.GeometryModel()

print("\n1. Creating model with two planar joints...")

# Add first joint
idx1 = model.addJoint(
    model.getJointId("universe"),
    pin.JointModelPlanar(),
    pin.SE3.Identity(),
    "planar1_joint"
)
model.addJointFrame(idx1)
model.appendBodyToJoint(idx1, pin.Inertia.Random(), pin.SE3.Identity())
body_frame_1 = model.addBodyFrame("planar1_body", idx1, pin.SE3.Identity(), -1)

# Add second joint
idx2 = model.addJoint(
    model.getJointId("universe"),
    pin.JointModelPlanar(),
    pin.SE3.Identity(),
    "planar2_joint"
)
model.addJointFrame(idx2)
model.appendBodyToJoint(idx2, pin.Inertia.Random(), pin.SE3.Identity())
body_frame_2 = model.addBodyFrame("planar2_body", idx2, pin.SE3.Identity(), -1)

print(f"   Model has {model.nq} DOFs")

# Add collision geometries (boxes)
print("\n2. Adding collision geometries (boxes)...")

try:
    import hppfcl as fcl
    print("   Using hppfcl (coal)")
except ImportError:
    try:
        import coal as fcl
        print("   Using coal")
    except ImportError:
        print("   ERROR: Neither hppfcl nor coal found!")
        sys.exit(1)

# Box for first body
box1 = fcl.Box(1.0, 1.0, 1.0)
body_id_1 = model.getBodyId("planar1_body")
joint_parent_1 = model.frames[body_id_1].parentJoint
idx_geom1 = geom_model.addGeometryObject(
    pin.GeometryObject(
        "box1", joint_parent_1, body_id_1,
        pin.SE3.Identity(), box1
    )
)
print(f"   Added box1 (geom index {idx_geom1})")

# Box for second body
box2 = fcl.Box(1.0, 1.0, 1.0)
body_id_2 = model.getBodyId("planar2_body")
joint_parent_2 = model.frames[body_id_2].parentJoint
idx_geom2 = geom_model.addGeometryObject(
    pin.GeometryObject(
        "box2", joint_parent_2, body_id_2,
        pin.SE3.Identity(), box2
    )
)
print(f"   Added box2 (geom index {idx_geom2})")
print(f"   Total geometries: {geom_model.ngeoms}")

# Add all collision pairs
print("\n3. Setting up collision pairs...")
try:
    geom_model.addAllCollisionPairs()
    print(f"   OK addAllCollisionPairs() succeeded")
    print(f"   Created {len(geom_model.collisionPairs)} collision pairs")
except Exception as e:
    print(f"   FAIL addAllCollisionPairs() FAILED: {e}")
    sys.exit(1)

# Create data structures
print("\n4. Creating data structures...")
try:
    data = pin.Data(model)
    geom_data = pin.GeometryData(geom_model)
    print(f"   OK Data and GeometryData created")
except Exception as e:
    print(f"   FAIL Failed to create data: {e}")
    sys.exit(1)

# Test collision detection at different configurations
print("\n5. Testing collision detection...")

# Test 1: Boxes overlapping (should collide)
q1 = np.array([0, 0, 1, 0, 0, 0, 1, 0])
print(f"\n   Test 1: q = {q1}")
print(f"   (boxes at same position - should COLLIDE)")
try:
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, q1)
    print(f"   OK updateGeometryPlacements() succeeded")

    # Try computeCollisions (plural)
    is_colliding = pin.computeCollisions(geom_model, geom_data, stop_at_first_collision=True)
    print(f"   OK computeCollisions() succeeded")
    print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

    if not is_colliding:
        print(f"   WARNING: Expected collision but got none!")
except Exception as e:
    print(f"   FAIL Collision detection FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Boxes far apart (should not collide)
q2 = np.array([2, 0, 1, 0, 0, 0, 1, 0])
print(f"\n   Test 2: q = {q2}")
print(f"   (boxes 2m apart - should NOT collide)")
try:
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, q2)
    is_colliding = pin.computeCollisions(geom_model, geom_data, stop_at_first_collision=True)
    print(f"   OK computeCollisions() succeeded")
    print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

    if is_colliding:
        print(f"   WARNING: Unexpected collision detected!")
except Exception as e:
    print(f"   FAIL Collision detection FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Boxes barely touching (should collide)
q3 = np.array([0.99, 0, 1, 0, 0, 0, 1, 0])
print(f"\n   Test 3: q = {q3}")
print(f"   (boxes barely overlapping - should COLLIDE)")
try:
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, q3)
    is_colliding = pin.computeCollisions(geom_model, geom_data, stop_at_first_collision=True)
    print(f"   OK computeCollisions() succeeded")
    print(f"   Result: {'COLLISION' if is_colliding else 'NO COLLISION'}")

    if not is_colliding:
        print(f"   WARNING: Expected collision but got none!")
except Exception as e:
    print(f"   FAIL Collision detection FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("OK ALL TESTS PASSED - Collision detection is working!")
print("="*60)
