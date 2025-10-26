#!/usr/bin/env python3
"""
Live Collision Geometry Tuner

Watch and auto-reload URDF collision geometry in real-time.
Edit the URDF box dimensions, save, and see changes immediately!

Usage:
    python live-collision-tuner.py [--urdf path/to/urdf]

Controls:
    V - Toggle visual geometry on/off
    C - Toggle collision geometry on/off
    R - Manual reload (also auto-reloads on file change)
    Q - Quit
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# File watching
import hashlib

def compute_file_hash(filepath):
    """Compute MD5 hash of file contents."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

class LiveCollisionTuner:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        # Compute mesh directory for package:// resolution
        # URDF is at: models/example-robot-data/robots/so_arm_description/urdf/file.urdf
        # Mesh dir should be: models/ (so package://example-robot-data/... resolves correctly)
        self.mesh_dir = self.urdf_path.parent.parent.parent.parent.parent
        self.last_hash = None
        self.show_visual = True
        self.show_collision = True

        # Load model and create visualizer
        self.load_model()

        print("=" * 80)
        print("LIVE COLLISION GEOMETRY TUNER")
        print("=" * 80)
        print(f"\nWatching: {self.urdf_path}")
        print(f"Mesh dir: {self.mesh_dir}")
        print(f"\nModel: {self.model.nq} DOF, {self.collision_model.ngeoms} collision geometries")
        print("\nControls:")
        print("  V - Toggle visual geometry")
        print("  C - Toggle collision geometry")
        print("  R - Manual reload")
        print("  Q - Quit")
        print("\nEdit the URDF file and save - it will auto-reload!")
        print("Tune the <box size=\"x y z\"/> values in <collision> tags")
        print("=" * 80)

    def load_model(self):
        """Load or reload the URDF model."""
        print(f"\n[{time.strftime('%H:%M:%S')}] Loading model...")

        try:
            # Load model with free flyer
            self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
                str(self.urdf_path),
                str(self.mesh_dir),
                pin.JointModelFreeFlyer()
            )

            # Create data
            self.data = pin.Data(self.model)

            # Initialize visualizer (or reuse existing)
            if not hasattr(self, 'viz'):
                self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
                self.viz.initViewer(open=True, loadModel=False)
            else:
                # Update visualizer models
                self.viz.model = self.model
                self.viz.collision_model = self.collision_model
                self.viz.visual_model = self.visual_model
                self.viz.data = self.data

            # Load geometry
            self.viz.loadViewerModel()

            # Set neutral configuration
            self.q = pin.neutral(self.model)
            self.q[2] = 0.0  # Ground level

            # Display
            self.viz.display(self.q)

            # Apply current visibility settings
            self.update_visibility()

            # Update file hash
            self.last_hash = compute_file_hash(self.urdf_path)

            print(f"[{time.strftime('%H:%M:%S')}] Model loaded successfully!")
            print(f"  Visual geometries: {self.visual_model.ngeoms}")
            print(f"  Collision geometries: {self.collision_model.ngeoms}")

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR loading model: {e}")
            import traceback
            traceback.print_exc()

    def update_visibility(self):
        """Update visual/collision geometry visibility."""
        try:
            # Visual geometry
            if hasattr(self.viz, 'displayVisuals'):
                self.viz.displayVisuals(self.show_visual)
            else:
                # Fallback: manually hide/show visual geometries
                for i in range(self.visual_model.ngeoms):
                    geom_name = self.visual_model.geometryObjects[i].name
                    if self.show_visual:
                        self.viz.viewer[f"visuals/{geom_name}"].set_property("visible", True)
                    else:
                        self.viz.viewer[f"visuals/{geom_name}"].set_property("visible", False)

            # Collision geometry
            if hasattr(self.viz, 'displayCollisions'):
                self.viz.displayCollisions(self.show_collision)
            else:
                # Fallback: manually hide/show collision geometries
                for i in range(self.collision_model.ngeoms):
                    geom_name = self.collision_model.geometryObjects[i].name
                    if self.show_collision:
                        self.viz.viewer[f"collision/{geom_name}"].set_property("visible", True)
                    else:
                        self.viz.viewer[f"collision/{geom_name}"].set_property("visible", False)

        except Exception as e:
            print(f"Error updating visibility: {e}")

    def check_file_changed(self):
        """Check if URDF file has been modified."""
        current_hash = compute_file_hash(self.urdf_path)
        if current_hash != self.last_hash:
            print(f"\n[{time.strftime('%H:%M:%S')}] File changed detected!")
            self.load_model()
            return True
        return False

    def run(self):
        """Main loop - watch file and handle user input."""
        print("\nWatching for file changes... (Ctrl+C to exit)")

        try:
            while True:
                # Check for file changes every 0.5 seconds
                time.sleep(0.5)
                self.check_file_changed()

        except KeyboardInterrupt:
            print("\n\nExiting...")

def main():
    parser = argparse.ArgumentParser(description="Live collision geometry tuner")
    parser.add_argument(
        '--urdf',
        type=str,
        default=None,
        help='Path to URDF file (default: so101_simple_collision.urdf)'
    )
    args = parser.parse_args()

    # Default URDF path
    if args.urdf is None:
        model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR", "."))
        urdf_path = model_path / "so_arm_description/urdf/so101_simple_collision.urdf"
    else:
        urdf_path = Path(args.urdf)

    # Create tuner and run
    tuner = LiveCollisionTuner(urdf_path)
    tuner.run()

if __name__ == "__main__":
    main()
