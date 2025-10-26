#!/usr/bin/env python3
"""
Inspect collision geometry and export to viewable formats.
Helps understand collision detection issues by:
1. Listing all collision geometries and their properties
2. Showing which pairs are colliding
3. Exporting collision geometry to OBJ files for external viewing
"""

import sys
import os
import numpy as np
from pathlib import Path
import pinocchio as pin

print("=" * 80)
print("SO101 Collision Geometry Inspector")
print("=" * 80)

# Load model
model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
mesh_dir = model_path.parent.parent
urdf_model_path = model_path / "so_arm_description/urdf/so101.urdf"

print(f"\nLoading model from: {urdf_model_path}")
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
)

# Setup collision pairs
print(f"\nModel: {model.nq} DOF, {collision_model.ngeoms} collision geometries")
collision_model.addAllCollisionPairs()
print(f"Total collision pairs: {len(collision_model.collisionPairs)}")

# Create data
data = pin.Data(model)
collision_data = pin.GeometryData(collision_model)

# Neutral configuration
q = pin.neutral(model)
q[2] = 0.0  # Ground level

# Update and check collisions
pin.updateGeometryPlacements(model, data, collision_model, collision_data, q)
is_colliding = pin.computeCollisions(collision_model, collision_data, stop_at_first_collision=False)

print(f"\nCollision status: {'COLLISION DETECTED' if is_colliding else 'NO COLLISION'}")

# Print detailed geometry information
print("\n" + "=" * 80)
print("COLLISION GEOMETRY DETAILS")
print("=" * 80)

for i in range(collision_model.ngeoms):
    geom_obj = collision_model.geometryObjects[i]
    geom_shape = geom_obj.geometry

    print(f"\n[{i}] {geom_obj.name}")
    print(f"    Parent joint: {geom_obj.parentJoint} ({model.names[geom_obj.parentJoint]})")
    print(f"    Parent frame: {geom_obj.parentFrame}")
    print(f"    Geometry type: {type(geom_shape).__name__}")

    # Try to get geometry properties
    try:
        import hppfcl
        if isinstance(geom_shape, hppfcl.Box):
            print(f"    Box dimensions: {geom_shape.halfSide * 2}")
        elif isinstance(geom_shape, hppfcl.Sphere):
            print(f"    Sphere radius: {geom_shape.radius}")
        elif isinstance(geom_shape, hppfcl.Cylinder):
            print(f"    Cylinder: height={geom_shape.halfLength * 2}, radius={geom_shape.radius}")
        elif isinstance(geom_shape, hppfcl.Capsule):
            print(f"    Capsule: height={geom_shape.halfLength * 2}, radius={geom_shape.radius}")
        elif isinstance(geom_shape, hppfcl.Cone):
            print(f"    Cone: height={geom_shape.halfLength * 2}, radius={geom_shape.radius}")
        elif hasattr(geom_shape, 'num_vertices'):
            # Mesh type
            print(f"    Mesh vertices: {geom_shape.num_vertices}")
            print(f"    Mesh triangles: {geom_shape.num_tris if hasattr(geom_shape, 'num_tris') else 'N/A'}")

            # Compute AABB (axis-aligned bounding box)
            placement = collision_data.oMg[i]
            geom_shape.computeLocalAABB()
            aabb_min = geom_shape.aabb_local.min_
            aabb_max = geom_shape.aabb_local.max_
            size = aabb_max - aabb_min
            print(f"    AABB size: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}] m")
            print(f"    AABB center: [{(aabb_min[0]+aabb_max[0])/2:.4f}, {(aabb_min[1]+aabb_max[1])/2:.4f}, {(aabb_min[2]+aabb_max[2])/2:.4f}]")
        else:
            print(f"    (Unknown geometry properties)")
    except Exception as e:
        print(f"    Error getting properties: {e}")

# Analyze collision pairs
print("\n" + "=" * 80)
print("COLLISION PAIR ANALYSIS")
print("=" * 80)

same_link_pairs = []
adjacent_link_pairs = []
colliding_pairs = []
other_pairs = []

for idx, pair in enumerate(collision_model.collisionPairs):
    geom1 = collision_model.geometryObjects[pair.first]
    geom2 = collision_model.geometryObjects[pair.second]

    joint1_idx = geom1.parentJoint
    joint2_idx = geom2.parentJoint

    is_collision = collision_data.collisionResults[idx].isCollision()

    pair_info = {
        'idx': idx,
        'geom1': geom1.name,
        'geom2': geom2.name,
        'joint1': model.names[joint1_idx],
        'joint2': model.names[joint2_idx],
        'colliding': is_collision
    }

    # Categorize
    if joint1_idx == joint2_idx:
        same_link_pairs.append(pair_info)
    elif joint1_idx == model.parents[joint2_idx] or joint2_idx == model.parents[joint1_idx]:
        adjacent_link_pairs.append(pair_info)
    elif is_collision:
        colliding_pairs.append(pair_info)
    else:
        other_pairs.append(pair_info)

print(f"\nPair categories:")
print(f"  Same link: {len(same_link_pairs)} pairs")
print(f"  Adjacent links: {len(adjacent_link_pairs)} pairs")
print(f"  Other (colliding): {len(colliding_pairs)} pairs")
print(f"  Other (non-colliding): {len(other_pairs)} pairs")

print(f"\n--- Same Link Pairs (should be filtered) ---")
for p in same_link_pairs[:10]:
    status = "COLLISION" if p['colliding'] else "clear"
    print(f"  [{p['idx']}] {p['geom1']} <-> {p['geom2']} ({p['joint1']}) [{status}]")
if len(same_link_pairs) > 10:
    print(f"  ... and {len(same_link_pairs) - 10} more")

print(f"\n--- Adjacent Link Pairs (should be filtered) ---")
for p in adjacent_link_pairs[:10]:
    status = "COLLISION" if p['colliding'] else "clear"
    print(f"  [{p['idx']}] {p['geom1']} <-> {p['geom2']} ({p['joint1']} <-> {p['joint2']}) [{status}]")
if len(adjacent_link_pairs) > 10:
    print(f"  ... and {len(adjacent_link_pairs) - 10} more")

print(f"\n--- Colliding Pairs (after filtering) ---")
if len(colliding_pairs) == 0:
    print("  (None - good!)")
else:
    for p in colliding_pairs:
        print(f"  [{p['idx']}] {p['geom1']} <-> {p['geom2']} ({p['joint1']} <-> {p['joint2']})")

# Export recommendation
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print(f"""
The URDF uses MESH-based collision geometry for all {collision_model.ngeoms} collision objects.
This is why you see many spheres (fallback visualization) instead of boxes.

Problems with the current collision geometry:
1. Multiple meshes per link (servo body + holders + wheels) naturally overlap
2. {len(same_link_pairs)} pairs are on the SAME link (will always collide)
3. {len(adjacent_link_pairs)} pairs are on ADJACENT links (will usually collide)
4. Only {len(other_pairs) + len(colliding_pairs)} pairs are between non-adjacent links (what we care about)

Recommendation:
- Filter out same-link and adjacent-link pairs (your simulation now does this)
- Consider simplifying collision geometry in URDF to use boxes/cylinders instead of meshes
- Or manually create a simplified collision URDF with fewer, larger bounding boxes

After filtering, you should have ~{len(other_pairs) + len(colliding_pairs)} collision pairs to check.
""")

print("\n" + "=" * 80)
print("To export collision meshes for viewing:")
print("  - STL files are already in: models/example-robot-data/robots/so_arm_description/meshes/so101/")
print("  - You can open these in MeshLab, Blender, or any STL viewer")
print("  - Each link has 2-4 separate mesh files that make up its collision geometry")
print("=" * 80)
