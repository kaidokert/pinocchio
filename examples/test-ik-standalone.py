#!/usr/bin/env python3
"""
Standalone IK test for SO101 - move end-effector forward by 1cm
Based on inverse-kinematics.py pattern
"""

import numpy as np
import pinocchio as pin
from numpy.linalg import norm, solve
import sys
from pathlib import Path

print("=" * 80)
print("SO101 IK Standalone Test")
print("=" * 80)

# Find URDF
urdf_path = Path(__file__).parent.parent / "models/example-robot-data/robots/so_arm_description/urdf/so101_simple_collision.urdf"
mesh_dir = Path(__file__).parent.parent / "models"

if not urdf_path.exists():
    print(f"ERROR: URDF not found at {urdf_path}")
    sys.exit(1)

print(f"Loading URDF: {urdf_path}")
print(f"Mesh dir: {mesh_dir}")

# Load model
model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
data = model.createData()

print(f"Model loaded: {model.nq} DOF")

# Get end-effector frame
EE_FRAME_NAME = "gripper"
EE_FRAME_ID = model.getFrameId(EE_FRAME_NAME)
EE_FRAME = model.frames[EE_FRAME_ID]
EE_JOINT_ID = EE_FRAME.parentJoint
print(f"End-effector frame: '{EE_FRAME_NAME}' (ID={EE_FRAME_ID})")
print(f"Parent joint: {model.names[EE_JOINT_ID]} (ID={EE_JOINT_ID})")

# Start from neutral configuration
q = pin.neutral(model)
print(f"Initial configuration: {q.T}")

# Compute initial end-effector pose (use joint pose, not frame pose)
pin.forwardKinematics(model, data, q)
initial_pose = data.oMi[EE_JOINT_ID]
initial_pos = initial_pose.translation
print(f"Initial EE position: {initial_pos.T}")

# Target: move forward by 1cm (in +X direction)
target_pos = initial_pos + np.array([0.01, 0.0, 0.0])
print(f"Target EE position:  {target_pos.T}")
print(f"Target distance: {norm(target_pos - initial_pos)*1000:.2f}mm")

# Create target pose (keep orientation, change position)
oMdes = pin.SE3(initial_pose.rotation, target_pos)

# IK parameters
eps = 1e-3  # 1mm tolerance
IT_MAX = 100
DT = 1e-1
damp = 1e-6

print()
print("Starting IK solver...")
print(f"Parameters: eps={eps*1000:.1f}mm, IT_MAX={IT_MAX}, DT={DT}, damp={damp:.1e}")
print()

i = 0
success = False

while True:
    # Forward kinematics
    pin.forwardKinematics(model, data, q)

    # Current end-effector pose (use joint pose)
    oMcurrent = data.oMi[EE_JOINT_ID]

    # SE3 error (relative pose from current to desired)
    iMd = oMcurrent.actInv(oMdes)

    # Position error (translation only - simpler approach, no log map needed!)
    err = iMd.translation
    pos_err_norm = norm(err)

    # Current position
    curr_pos = oMcurrent.translation

    if i % 5 == 0 or i < 5:  # Print first 5 and every 5th iteration
        print(f"iter {i:2d}: pos_err={pos_err_norm*1000:6.2f}mm, pos={curr_pos.T}")

    # Check convergence
    if pos_err_norm < eps:
        success = True
        break

    if i >= IT_MAX:
        success = False
        break

    # Compute JOINT Jacobian (like inverse-kinematics-3d.py)
    J = pin.computeJointJacobian(model, data, q, EE_JOINT_ID)  # Returns Jacobian directly!

    # Extract position part (first 3 rows = linear velocity)
    J = -J[:3, :]  # Only position, no orientation

    # Debug on first iteration
    if i == 0:
        print(f"  J shape: {J.shape}, norm: {np.linalg.norm(J):.4f}")
        print(f"  J[:, 6:9] (first 3 arm joints):\n{J[:, 6:9]}")

    # Lock base joints (first 6 DOF)
    # We only want to move the arm joints (DOF 6-12)
    J[:, :6] = 0.0  # Zero out base joint columns

    # Solve using damped least squares
    v = -J.T @ solve(J @ J.T + damp * np.eye(3), err)

    if i == 0:
        print(f"  v (joint velocities): norm={np.linalg.norm(v):.6f}")
        print(f"  v[6:13] (arm joints): {v[6:13]}")

    # Integrate (only arm joints will change due to locked base)
    q = pin.integrate(model, q, v * DT)

    i += 1

print()
print("=" * 80)
if success:
    print("SUCCESS: IK CONVERGED!")
    print(f"Iterations: {i}")
    print(f"Final position error: {pos_err_norm*1000:.3f}mm")
else:
    print("FAILED: IK DID NOT CONVERGE")
    print(f"Iterations: {i}")
    print(f"Final position error: {pos_err_norm*1000:.3f}mm")

print()
print(f"Final configuration: {q.T}")
print(f"Final EE position: {curr_pos.T}")
print(f"Target position:   {target_pos.T}")
print("=" * 80)
