# This example shows how to load the SO-ARM 101 robot model in MeshCat.
import logging

import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

logging.basicConfig(level=logging.DEBUG, stream= sys.stdout)

# Load the URDF model.
# The environment variable EXAMPLE_ROBOT_DATA_MODEL_DIR must be set to the .../robots directory.
model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))

# The mesh directory is the parent of the model_path parent, i.e., the .../models directory.
# This is necessary to resolve package:// URIs in the URDF.
mesh_dir = model_path.parent.parent

urdf_filename = "so101.urdf"
urdf_model_path = model_path / "so_arm_description/urdf" / urdf_filename

# Load the URDF model.
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer(),
    verbose=True
)

# Start a new MeshCat server and client.
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)

# Load the robot in the viewer.
viz.loadViewerModel()

# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)

# Keep the script alive to interact with the viewer.
input("Press Enter to exit...")
