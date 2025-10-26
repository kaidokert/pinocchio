#!/usr/bin/env python3
"""
Test SO101 collision without visualizer to isolate the crash
"""

import sys
import os
import numpy as np
from pathlib import Path
import pinocchio as pin

print("="*80)
print("SO101 Collision Test WITHOUT Visualizer")
print("="*80)

# Load model
print("\nLoading SO101 model...")
model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
mesh_dir = model_path.parent.parent
urdf_model_path = model_path / "so_arm_description/urdf/so101.urdf"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
)
print(f"OK Model loaded: {model.nq} DOF, {collision_model.ngeoms} collision geometries")

# NO VISUALIZER - Skip MeshcatVisualizer entirely

# Create data structures
print("\nCreating data structures...")
data = pin.Data(model)
collision_data = pin.GeometryData(collision_model)
print("OK Data created")

# Setup collision pairs
print("\nSetting up collision pairs...")
print(f"Before: {len(collision_model.collisionPairs)} pairs")
collision_model.addAllCollisionPairs()
print(f"After: {len(collision_model.collisionPairs)} pairs")

# Initialize configuration
q = pin.neutral(model)
q[2] = 0.0  # Z position at ground level
print(f"\nInitial q: {q}")

# Test collision detection
print("\nTesting collision detection (10 iterations)...")
for iteration in range(10):
    print(f"Iteration {iteration+1}...", end=" ")

    # Modify configuration
    if iteration > 0:
        q[7] += np.deg2rad(5)

    # Update geometry placements
    pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)

    # Compute collisions
    is_colliding = pin.computeCollisions(collision_model, collision_data, stop_at_first_collision=True)

    print(f"{'COLLISION' if is_colliding else 'NO COLLISION'}")

print("\n" + "="*80)
print("OK ALL TESTS PASSED - No crash without visualizer!")
print("="*80)
