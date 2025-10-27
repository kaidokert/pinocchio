# SO101 Robot Arm Simulation with Keyboard Control
# Uses blessed for terminal UI and Pinocchio for dynamics

import argparse
import logging
import os
import statistics
import sys
import time
import traceback
import json
import numpy as np
from pathlib import Path
from blessed import Terminal

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# Setup logging to file
LOG_FILE = "so101_sim.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("SO101 Simulation Starting")
logger.info("=" * 80)

# Configuration
STEP_SIZE = 2.0  # degrees per keypress
LARGE_STEP_SIZE = 10.0  # degrees with Shift
DT = 0.01  # simulation timestep
STORE_DIR = Path("store")  # Directory for saved states

# Loop rate control
TARGET_LOOP_HZ = 20.0  # Target control loop frequency
WARMUP_FRAMES = 10  # Exclude first N frames from statistics

# Parse command line arguments BEFORE loading model
parser = argparse.ArgumentParser(description="SO101 Robot Arm Simulation")
parser.add_argument(
    '--collision',
    action='store_true',
    help='Enable collision detection'
)
parser.add_argument(
    '--urdf',
    type=str,
    default='so101.urdf',
    help='URDF filename (default: so101.urdf, or use: so101_simple_collision.urdf)'
)
args = parser.parse_args()

try:
    logger.info("Getting model paths from environment")
    # Get model paths from environment
    model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
    mesh_dir = model_path.parent.parent

    # Use specified URDF file
    urdf_model_path = model_path / f"so_arm_description/urdf/{args.urdf}"
    logger.info(f"URDF path: {urdf_model_path}")
    logger.info(f"Mesh dir: {mesh_dir}")

    if not urdf_model_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_model_path}")

    logger.info("Loading robot model from URDF...")
    # Load robot model
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
    )
    logger.info(f"Model loaded successfully: {model.nq} DOF")
    logger.info(f"Collision model: {collision_model.ngeoms} geometries")
except Exception as e:
    logger.critical(f"Failed to load model: {e}", exc_info=True)
    print(f"FATAL ERROR loading model: {e}")
    print(f"Check {LOG_FILE} for details")
    sys.exit(1)

# Joint names (excluding the 6 DOF free flyer = indices 7 onwards)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_INDICES = list(range(7, 7 + len(JOINT_NAMES)))  # Indices in configuration vector


class SO101Simulation:
    def __init__(self, enable_collision=False):
        logger.info("Initializing SO101Simulation")
        logger.info(f"Collision detection: {'ENABLED' if enable_collision else 'DISABLED'}")
        self.term = Terminal()

        try:
            logger.info("Creating MeshcatVisualizer")
            # Initialize visualizer
            self.viz = MeshcatVisualizer(model, collision_model, visual_model)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            logger.info("Visualizer initialized")

            # Add ground plane
            logger.info("Adding ground plane")
            self.add_ground_plane()

            # Add test ball
            logger.info("Adding test ball")
            self.ball_pos = np.array([0.3, 0.3, 0.5])  # Start 50cm above ground
            self.ball_vel = np.zeros(3)
            self.add_test_ball()

            # Create data object for dynamics
            logger.info("Creating dynamics data")
            self.data = pin.Data(model)

            # Setup collision pairs FIRST (before creating GeometryData!)
            if enable_collision:
                logger.info("Setting up collision pairs")
                self.setup_collision_pairs()
                logger.info("Collision setup complete")

                # Create collision data AFTER collision pairs are added
                logger.info("Creating collision data (after addAllCollisionPairs)")
                self.collision_data = pin.GeometryData(collision_model)
            else:
                logger.info("Skipping collision setup (disabled by command line)")
                # Still create collision_data but it won't be used
                self.collision_data = pin.GeometryData(collision_model)
        except Exception as e:
            logger.critical(f"Failed during initialization: {e}", exc_info=True)
            raise

        # Robot state
        self.q = pin.neutral(model)  # Configuration
        self.v = np.zeros(model.nv)  # Velocity

        # Set initial base position (slightly above ground)
        self.q[0] = 0.0  # X position
        self.q[1] = 0.0  # Y position
        self.q[2] = 0.004  # Z position (2cm above ground for clearance)

        # Control state
        self.running = True
        self.selected_joint = 0
        self.last_command = "Ready"
        self.command_history = []
        self.paused = False
        self.gravity_enabled = False
        self.collision_detected = False
        self.colliding_pairs = []
        self.enable_collision_detection = enable_collision
        self.show_collision_geom = enable_collision  # Show by default when collision enabled
        self.prevent_collisions = enable_collision  # Prevent commanded motions that would collide
        self.prevent_environment_collision = enable_collision  # Prevent robot-environment collisions

        # Track geometry indices by type
        self.robot_geom_indices = list(range(collision_model.ngeoms))  # Robot geometries (initially all)
        self.environment_geom_indices = []  # Environment geometries (ground, obstacles)

        # Add environment collision geometries
        if enable_collision:
            self.add_environment_collision()

        # Loop rate tracking
        self.frame_count = 0
        self.frame_times = []  # All frame times (for warmup exclusion)
        self.last_frame_time = None
        self.target_hz = TARGET_LOOP_HZ

        # Display initial state
        self.viz.display(self.q)

        # Show collision geometries by default if collision detection is enabled
        if self.enable_collision_detection:
            self.update_collision_visualization()

    def add_ground_plane(self):
        """Add a ground plane to MeshCat."""
        import meshcat.geometry as g
        import meshcat.transformations as tf

        # Create a large flat box for ground
        self.viz.viewer["ground"].set_object(
            g.Box([2.0, 2.0, 0.01]),  # 2m x 2m x 1cm
            g.MeshLambertMaterial(color=0x808080)  # Gray
        )

        # Position ground at z = -0.005 (half thickness below origin)
        self.viz.viewer["ground"].set_transform(
            tf.translation_matrix([0, 0, -0.005])
        )

    def add_environment_collision(self):
        """Add environment objects (ground, obstacles) to collision model."""
        try:
            import hppfcl
        except ImportError:
            logger.warning("hppfcl not available - skipping environment collision")
            return

        logger.info("Adding environment collision geometries...")

        # Ground plane collision geometry
        ground_shape = hppfcl.Box(2.0, 2.0, 0.01)  # 2m x 2m x 1cm
        ground_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.005]))

        ground_geom_obj = pin.GeometryObject(
            "collision_ground",
            0,  # world frame (universe frame)
            ground_placement,
            ground_shape
        )

        # Add to collision model
        ground_idx = collision_model.addGeometryObject(ground_geom_obj)
        self.environment_geom_indices.append(ground_idx)

        # Add collision pairs: each robot geometry <-> ground
        for robot_idx in self.robot_geom_indices:
            collision_model.addCollisionPair(
                pin.CollisionPair(robot_idx, ground_idx)
            )

        logger.info(f"Added ground collision geometry at index {ground_idx}")
        logger.info(f"Added {len(self.robot_geom_indices)} robot<->ground collision pairs")

        # Re-create collision data after modifying collision model
        self.collision_data = pin.GeometryData(collision_model)

        # Add cup to scene
        self.add_cup()

    def add_cup(self):
        """Add a cup mesh to the scene with visualization and collision."""
        import meshcat.geometry as g
        import meshcat.transformations as tf
        import trimesh
        import io
        import hppfcl

        # Cup position in world
        cup_pos = np.array([0.3, 0.0, 0.00])  # 30cm in front, 2cm above ground

        # SCALE FACTOR: Adjust this to change cup size (ideal ~0.035 for 8cm cup)
        CUP_SCALE_FACTOR = 0.055

        # Load cup_3.dae (path relative to examples directory)
        cup_mesh_path = (Path(__file__).parent.parent / "models/scene/cup_3.dae").absolute()
        logger.info(f"Loading cup mesh from: {cup_mesh_path}")

        # Load DAE with trimesh
        loaded = trimesh.load(str(cup_mesh_path))

        # Convert Scene to Mesh if needed
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded

        # Apply scale
        mesh.apply_scale(CUP_SCALE_FACTOR)

        # No rotation - keep original DAE orientation (opening points up in Z)
        # The DAE file is already oriented correctly

        # Fix normals to prevent disappearing faces (ensures outward-pointing normals)
        mesh.fix_normals()

        logger.debug(f"Mesh after transforms: bounds={mesh.bounds}, {len(mesh.vertices)} vertices")

        # Export as OBJ for visualization
        obj_data = mesh.export(file_type='obj')
        obj_bytes = obj_data.encode('utf-8') if isinstance(obj_data, str) else obj_data
        obj_stream = io.BytesIO(obj_bytes)

        # Load into MeshCat
        obj_geom = g.ObjMeshGeometry.from_stream(obj_stream)
        cup_material = g.MeshLambertMaterial(color=0xffaa88, opacity=0.8, side=2)  # side=2 = DoubleSide
        self.viz.viewer["scene"]["cup"].set_object(obj_geom, cup_material)
        self.viz.viewer["scene"]["cup"].set_transform(tf.translation_matrix(cup_pos))

        logger.info(f"Cup visualization loaded at {cup_pos} (scale={CUP_SCALE_FACTOR})")

        # Create collision geometry from mesh
        vertices = mesh.vertices
        faces = mesh.faces

        cup_collision_shape = hppfcl.BVHModelOBBRSS()
        cup_collision_shape.beginModel(len(vertices), len(faces))
        cup_collision_shape.addVertices(vertices)
        cup_collision_shape.addTriangles(faces)
        cup_collision_shape.endModel()

        # Mesh is already rotated and scaled, just translate
        cup_placement = pin.SE3(np.eye(3), cup_pos)

        cup_geom_obj = pin.GeometryObject(
            "collision_cup",
            0,  # world frame
            cup_placement,
            cup_collision_shape
        )

        # Add to collision model
        cup_idx = collision_model.addGeometryObject(cup_geom_obj)
        self.environment_geom_indices.append(cup_idx)

        # Add collision pairs with robot
        for robot_idx in self.robot_geom_indices:
            collision_model.addCollisionPair(
                pin.CollisionPair(robot_idx, cup_idx)
            )

        logger.info(f"Cup collision added: {len(vertices)} vertices, {len(faces)} triangles at index {cup_idx}")

        # Re-create collision data
        self.collision_data = pin.GeometryData(collision_model)

    def add_test_ball(self):
        """Add a test ball to demonstrate gravity."""
        import meshcat.geometry as g
        import meshcat.transformations as tf

        # Create a red sphere
        self.viz.viewer["test_ball"].set_object(
            g.Sphere(0.03),  # 3cm radius
            g.MeshLambertMaterial(color=0xff0000)  # Red
        )
        self.update_ball_position()

    def update_ball_position(self):
        """Update test ball position in viewer."""
        import meshcat.transformations as tf
        self.viz.viewer["test_ball"].set_transform(
            tf.translation_matrix(self.ball_pos)
        )

    def setup_collision_pairs(self):
        """Setup collision pairs for self-collision detection."""
        try:
            logger.info(f"Before setup: {len(collision_model.collisionPairs)} collision pairs")

            # Add all self-collision pairs
            # This checks all collision geometries against each other
            logger.info("Calling addAllCollisionPairs()...")
            collision_model.addAllCollisionPairs()

            logger.info(f"After addAllCollisionPairs: {len(collision_model.collisionPairs)} collision pairs")

            # Filter collision pairs based on geometry type
            logger.info("Filtering collision pairs...")
            pairs_to_remove = []

            for pair in collision_model.collisionPairs:
                geom1 = collision_model.geometryObjects[pair.first]
                geom2 = collision_model.geometryObjects[pair.second]

                joint1_idx = geom1.parentJoint
                joint2_idx = geom2.parentJoint

                # Only remove same-link pairs (geometries on the same link)
                # For simplified box geometry, adjacent links are designed with gaps
                # so we don't need to filter adjacent-link pairs anymore
                if joint1_idx == joint2_idx:
                    pairs_to_remove.append(pair)
                    logger.debug(f"Removing same-link pair: {geom1.name} <-> {geom2.name}")

            # Remove the filtered pairs
            for pair in pairs_to_remove:
                collision_model.removeCollisionPair(pair)

            logger.info(f"Removed {len(pairs_to_remove)} same-link collision pairs")
            logger.info(f"Final collision pairs: {len(collision_model.collisionPairs)} pairs")

        except Exception as e:
            logger.error(f"Error setting up collision pairs: {e}", exc_info=True)
            logger.warning("Continuing without collision detection")
            # Don't raise - just disable collision detection
            self.enable_collision_detection = False

    def check_collisions(self):
        """Check for self-collisions and store colliding pairs."""
        if not self.enable_collision_detection:
            self.colliding_pairs = []
            return False

        try:
            # Update collision geometry placements
            pin.updateGeometryPlacements(model, self.data, collision_model, self.collision_data, self.q)

            # Compute all collisions (don't stop at first - get all of them)
            is_colliding = pin.computeCollisions(collision_model, self.collision_data, stop_at_first_collision=False)

            # Collect all colliding pairs
            self.colliding_pairs = []
            for i, pair in enumerate(collision_model.collisionPairs):
                if self.collision_data.collisionResults[i].isCollision():
                    geom1 = collision_model.geometryObjects[pair.first]
                    geom2 = collision_model.geometryObjects[pair.second]
                    self.colliding_pairs.append((geom1.name, geom2.name))

            return is_colliding
        except Exception as e:
            logger.error(f"Error checking collisions: {e}", exc_info=True)
            # Disable collision detection on error
            self.enable_collision_detection = False
            self.colliding_pairs = []
            return False

    def update_collision_visualization(self):
        """Render collision geometries with color coding.

        Red = colliding pairs, Green = non-colliding geometries.
        """
        if not self.enable_collision_detection:
            return

        try:
            import meshcat.geometry as g

            # Get set of colliding geometry indices for quick lookup
            colliding_geom_indices = set()
            for i, pair in enumerate(collision_model.collisionPairs):
                if self.collision_data.collisionResults[i].isCollision():
                    colliding_geom_indices.add(pair.first)
                    colliding_geom_indices.add(pair.second)

            # Render each collision geometry
            for geom_idx in range(collision_model.ngeoms):
                geom_obj = collision_model.geometryObjects[geom_idx]
                geom_name = f"collision_viz/{geom_obj.name}"

                if self.show_collision_geom:
                    # Get the current placement of this geometry
                    placement = self.collision_data.oMg[geom_idx]

                    # Determine color: red if colliding, green otherwise
                    if geom_idx in colliding_geom_indices:
                        color = 0xff0000  # Red for collision
                        opacity = 0.5
                    else:
                        color = 0x00ff00  # Green for non-colliding
                        opacity = 0.2

                    # Get geometry shape and create appropriate meshcat geometry
                    geom_shape = geom_obj.geometry

                    # Try to create a bounding box visualization
                    try:
                        # For mesh geometries, create a box that represents the bounds
                        import hppfcl
                        if isinstance(geom_shape, hppfcl.Box):
                            # Use actual box dimensions
                            side = geom_shape.halfSide * 2
                            viz_geom = g.Box([side[0], side[1], side[2]])
                        elif isinstance(geom_shape, hppfcl.Sphere):
                            viz_geom = g.Sphere(geom_shape.radius)
                        elif isinstance(geom_shape, hppfcl.Cylinder):
                            viz_geom = g.Cylinder(geom_shape.halfLength * 2, geom_shape.radius)
                        elif isinstance(geom_shape, hppfcl.Capsule):
                            # Capsule is cylinder with spherical caps
                            viz_geom = g.Cylinder(geom_shape.halfLength * 2, geom_shape.radius)
                        elif hasattr(geom_shape, 'num_vertices'):
                            # Mesh type (BVHModel, Convex, etc.) - show AABB bounding box
                            geom_shape.computeLocalAABB()
                            aabb_min = geom_shape.aabb_local.min_
                            aabb_max = geom_shape.aabb_local.max_
                            size = aabb_max - aabb_min
                            center = (aabb_min + aabb_max) / 2

                            # Create box at AABB center and size
                            viz_geom = g.Box([abs(size[0]), abs(size[1]), abs(size[2])])

                            # Adjust placement to account for AABB offset
                            import meshcat.transformations as tf
                            T = np.eye(4)
                            T[:3, :3] = placement.rotation
                            T[:3, 3] = placement.translation + placement.rotation @ center
                            self.viz.viewer[geom_name].set_object(
                                viz_geom,
                                g.MeshLambertMaterial(color=color, opacity=opacity, transparent=True)
                            )
                            self.viz.viewer[geom_name].set_transform(T)
                            continue  # Skip the normal transform setting below
                        else:
                            # Unknown type - use small sphere marker
                            viz_geom = g.Sphere(0.01)

                        # Set the geometry with material
                        material = g.MeshLambertMaterial(color=color, opacity=opacity, transparent=True)
                        self.viz.viewer[geom_name].set_object(viz_geom, material)

                        # Set transform to match the collision geometry placement
                        import meshcat.transformations as tf
                        T = np.eye(4)
                        T[:3, :3] = placement.rotation
                        T[:3, 3] = placement.translation
                        self.viz.viewer[geom_name].set_transform(T)

                    except Exception as e:
                        # If we can't render this geometry type, skip it
                        logger.debug(f"Could not render collision geometry {geom_obj.name}: {e}")
                        pass
                else:
                    # Hide collision visualization
                    self.viz.viewer[geom_name].delete()

        except Exception as e:
            logger.error(f"Error updating collision visualization: {e}", exc_info=True)

    def save_state(self, slot=1):
        """Save current robot and world state to JSON files."""
        # Create store directory if it doesn't exist
        STORE_DIR.mkdir(exist_ok=True)

        # Save robot state
        robot_state = {
            "q": self.q.tolist(),  # Configuration (includes base pose + joints)
            "v": self.v.tolist(),  # Velocity
            "joint_names": JOINT_NAMES,
            "joint_positions_deg": [self.get_joint_position(i) for i in range(len(JOINT_NAMES))],
        }
        robot_file = STORE_DIR / f"so101_{slot:03d}.json"
        with open(robot_file, 'w') as f:
            json.dump(robot_state, f, indent=2)
        logger.info(f"Saved robot state to {robot_file}")

        # Save ball state
        ball_state = {
            "position": self.ball_pos.tolist(),
            "velocity": self.ball_vel.tolist(),
        }
        ball_file = STORE_DIR / f"ball_{slot:03d}.json"
        with open(ball_file, 'w') as f:
            json.dump(ball_state, f, indent=2)
        logger.info(f"Saved ball state to {ball_file}")

        self.add_command(f"Saved state to slot {slot}")

    def load_state(self, slot=1):
        """Load robot and world state from JSON files."""
        robot_file = STORE_DIR / f"so101_{slot:03d}.json"
        ball_file = STORE_DIR / f"ball_{slot:03d}.json"

        if not robot_file.exists():
            self.add_command(f"Slot {slot} not found")
            logger.warning(f"Save file not found: {robot_file}")
            return

        try:
            # Load robot state
            with open(robot_file, 'r') as f:
                robot_state = json.load(f)
            self.q = np.array(robot_state["q"])
            self.v = np.array(robot_state["v"])
            logger.info(f"Loaded robot state from {robot_file}")

            # Load ball state if it exists
            if ball_file.exists():
                with open(ball_file, 'r') as f:
                    ball_state = json.load(f)
                self.ball_pos = np.array(ball_state["position"])
                self.ball_vel = np.array(ball_state["velocity"])
                self.update_ball_position()
                logger.info(f"Loaded ball state from {ball_file}")

            # Update display
            self.update_display()
            self.add_command(f"Loaded state from slot {slot}")
        except Exception as e:
            self.add_command(f"Error loading slot {slot}")
            logger.error(f"Error loading state: {e}", exc_info=True)

    def list_saved_states(self):
        """List all saved states."""
        if not STORE_DIR.exists():
            return []

        robot_files = sorted(STORE_DIR.glob("so101_*.json"))
        slots = []
        for f in robot_files:
            # Extract slot number from filename
            try:
                slot = int(f.stem.split('_')[1])
                slots.append(slot)
            except (IndexError, ValueError):
                pass
        return slots

    def get_joint_position(self, joint_idx):
        """Get position of a joint in degrees."""
        config_idx = JOINT_INDICES[joint_idx]
        return np.degrees(self.q[config_idx])

    def set_joint_position(self, joint_idx, value_deg):
        """Set position of a joint in degrees."""
        config_idx = JOINT_INDICES[joint_idx]
        self.q[config_idx] = np.radians(value_deg)

    def would_collide(self, q_test):
        """Check if a given configuration would cause collision (without modifying current state)."""
        if not self.enable_collision_detection:
            return False

        # If both prevention flags are off, allow all motion
        if not self.prevent_collisions and not self.prevent_environment_collision:
            return False

        try:
            # Create temporary data for collision checking
            data_test = pin.Data(model)
            collision_data_test = pin.GeometryData(collision_model)

            # Update collision geometry placements with test configuration
            pin.updateGeometryPlacements(model, data_test, collision_model, collision_data_test, q_test)

            # Compute collisions (don't stop at first - need to categorize)
            pin.computeCollisions(collision_model, collision_data_test, stop_at_first_collision=False)

            # Categorize collisions
            has_self_collision = False
            has_env_collision = False

            for idx, pair in enumerate(collision_model.collisionPairs):
                if collision_data_test.collisionResults[idx].isCollision():
                    geom1_idx = pair.first
                    geom2_idx = pair.second

                    # Check if both are robot geometries (self-collision)
                    if (geom1_idx in self.robot_geom_indices and
                        geom2_idx in self.robot_geom_indices):
                        has_self_collision = True

                    # Check if one is environment
                    if (geom1_idx in self.environment_geom_indices or
                        geom2_idx in self.environment_geom_indices):
                        has_env_collision = True

            # Respect separate toggles
            if self.prevent_collisions and has_self_collision:
                return True
            if self.prevent_environment_collision and has_env_collision:
                return True

            return False
        except Exception as e:
            logger.warning(f"Error in would_collide: {e}")
            return False  # On error, allow the motion (fail-safe)

    def set_joint_position_safe(self, joint_idx, value_deg):
        """
        Safely set joint position, checking for collisions first.
        Returns True if motion was applied, False if rejected due to collision.
        """
        if not self.prevent_collisions:
            # No collision prevention - apply directly
            self.set_joint_position(joint_idx, value_deg)
            return True

        # Create test configuration
        q_test = self.q.copy()
        config_idx = JOINT_INDICES[joint_idx]
        q_test[config_idx] = np.radians(value_deg)

        # Check if this would cause collision
        if self.would_collide(q_test):
            # Collision detected - reject motion
            return False
        else:
            # Safe to apply
            self.set_joint_position(joint_idx, value_deg)
            return True

    def get_rate_stats(self):
        """Calculate rate statistics, excluding warmup frames."""
        # Filter out warmup frames
        valid_times = self.frame_times[WARMUP_FRAMES:] if len(self.frame_times) > WARMUP_FRAMES else []

        if len(valid_times) < 2:
            return None  # Not enough data yet

        # Calculate statistics
        stats = {
            'mean_hz': 1.0 / statistics.mean(valid_times),
            'median_hz': 1.0 / statistics.median(valid_times),
            'min_hz': 1.0 / max(valid_times),  # Longest dt = slowest rate
            'max_hz': 1.0 / min(valid_times),  # Shortest dt = fastest rate
            'current_hz': 1.0 / valid_times[-1] if valid_times else 0,
            'target_hz': self.target_hz
        }
        return stats

    def draw_bar(self, value, width=20, min_val=-180, max_val=180):
        """Draw a horizontal bar representing a value in degrees."""
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        filled = int(normalized * width)
        bar = "=" * filled + "-" * (width - filled)
        return bar

    def draw_ui(self):
        """Draw the complete UI using blessed."""
        t = self.term

        lines = []
        lines.append(t.bold + t.center("SO101 ARM SIMULATION") + t.normal)
        lines.append(t.center("=" * 60))
        lines.append("")

        # Status
        status_color = t.yellow if self.paused else t.green
        status_text = "PAUSED" if self.paused else "RUNNING"
        lines.append(f"  Status: {status_color}{status_text}{t.normal}")

        # Collision status
        collision_color = t.red if self.collision_detected else t.green
        collision_text = "COLLISION!" if self.collision_detected else "No collision"
        collision_status = "ON" if self.enable_collision_detection else "OFF"
        lines.append(f"  Collision Detection: {t.cyan}{collision_status}{t.normal} - {collision_color}{collision_text}{t.normal}")

        # Gravity status
        gravity_status = f"{t.green}ON{t.normal}" if self.gravity_enabled else f"{t.normal}OFF{t.normal}"
        lines.append(f"  Gravity: {gravity_status}")

        # Show colliding pairs if any
        if self.collision_detected and len(self.colliding_pairs) > 0:
            lines.append(f"  {t.red}Colliding Pairs ({len(self.colliding_pairs)}):{t.normal}")
            for geom1, geom2 in self.colliding_pairs[:5]:  # Show first 5
                lines.append(f"    - {geom1} <-> {geom2}")
            if len(self.colliding_pairs) > 5:
                lines.append(f"    ... and {len(self.colliding_pairs) - 5} more")
        lines.append("")

        # Loop rate statistics
        rate_stats = self.get_rate_stats()
        if rate_stats:
            # Color code based on performance
            current_hz = rate_stats['current_hz']
            target_hz = rate_stats['target_hz']
            if current_hz >= target_hz * 0.9:
                rate_color = t.green  # Within 90% of target
            elif current_hz >= target_hz * 0.7:
                rate_color = t.yellow  # Within 70% of target
            else:
                rate_color = t.red  # Below 70% of target

            lines.append(f"  Loop Rate: {rate_color}{current_hz:.1f} Hz{t.normal} " +
                        f"(target: {target_hz:.0f} Hz)")
            lines.append(f"    Mean: {rate_stats['mean_hz']:.1f} Hz  " +
                        f"Median: {rate_stats['median_hz']:.1f} Hz  " +
                        f"Min: {rate_stats['min_hz']:.1f}  " +
                        f"Max: {rate_stats['max_hz']:.1f}")
        else:
            lines.append(f"  Loop Rate: {t.dim}Measuring... (frame {self.frame_count}/{WARMUP_FRAMES + 2}){t.normal}")
        lines.append("")

        # Joint positions with bars
        lines.append(t.bold + "  JOINT POSITIONS:" + t.normal)
        lines.append("")

        for i, name in enumerate(JOINT_NAMES):
            pos = self.get_joint_position(i)

            # Highlight selected joint
            if i == self.selected_joint:
                color = t.black_on_green
                marker = " <"
            else:
                color = t.normal
                marker = "  "

            # Joint name and number (display as 1-6 to match keyboard)
            lines.append(f"  {color}[{i+1}] {name:20s}{t.normal}{marker}")

            # Value and bar
            bar = self.draw_bar(pos, width=20, min_val=-180, max_val=180)
            value_color = t.green if -180 <= pos <= 180 else t.red
            lines.append(f"      {value_color}{pos:+7.2f}deg{t.normal} |{bar}|")
            lines.append("")

        # Controls in three columns
        lines.append("  " + "-" * 60)
        lines.append("")
        lines.append(t.bold + "  CONTROLS:" + t.normal)

        # Define controls in three columns
        col1 = [
            (f"{t.yellow}1-6{t.normal}    Select joint"),
            (f"{t.yellow}W/UP{t.normal}   Increase (+{STEP_SIZE}deg)"),
            (f"{t.yellow}S/DN{t.normal}   Decrease (-{STEP_SIZE}deg)"),
            (f"{t.yellow}A/LT{t.normal}   Previous joint"),
            (f"{t.yellow}D/RT{t.normal}   Next joint"),
        ]

        col2 = [
            (f"{t.yellow}+/-{t.normal}    Large (+/-{LARGE_STEP_SIZE}deg)"),
            (f"{t.yellow}SPC{t.normal}    Pause/Resume"),
            (f"{t.yellow}G{t.normal}      Toggle gravity"),
            (f"{t.yellow}C{t.normal}      Toggle collision"),
            (f"{t.yellow}V{t.normal}      Toggle viz"),
            (f"{t.yellow}E{t.normal}      Toggle env"),
        ]

        col3 = [
            (f"{t.yellow}H{t.normal}      Home position"),
            (f"{t.yellow}R{t.normal}      Reset sim"),
            (f"{t.yellow}F1-9{t.normal}   Save state"),
            (f"{t.yellow}0,7-9{t.normal}  Load state"),
            (f"{t.yellow}Q/ESC{t.normal}  Quit"),
        ]

        # Print side by side in three columns
        for i in range(max(len(col1), len(col2), len(col3))):
            c1 = f"    {col1[i]}" if i < len(col1) else ""
            c2 = f"{col2[i]}" if i < len(col2) else ""
            c3 = f"{col3[i]}" if i < len(col3) else ""
            lines.append(f"{c1:28s} {c2:26s} {c3}")
        lines.append("")

        # Status line
        lines.append("  " + "-" * 60)
        lines.append(f"  Last: {t.cyan}{self.last_command}{t.normal}")

        # Command history
        if self.command_history:
            lines.append(f"  History: {t.dim}{' | '.join(self.command_history[-3:])}{t.normal}")

        # Draw everything at once
        with t.hidden_cursor():
            print(t.home + t.clear + '\n'.join(lines), end='', flush=True)

    def add_command(self, cmd):
        """Add a command to history and update status."""
        self.last_command = cmd
        self.command_history.append(f"{time.strftime('%H:%M:%S')} {cmd}")

    def update_display(self):
        """Update the MeshCat display with current configuration."""
        self.viz.display(self.q)

        # Check for collisions (only runs if enabled via flag)
        self.collision_detected = self.check_collisions()

        # Update collision visualization if enabled
        if self.enable_collision_detection:
            self.update_collision_visualization()

    def home_position(self):
        """Move all joints to zero position."""
        for i in range(len(JOINT_NAMES)):
            self.set_joint_position(i, 0.0)
        self.v = np.zeros(model.nv)
        self.update_display()
        self.add_command("Moved to home position")

    def reload_urdf(self):
        """Reload URDF from disk (for live tuning workflow)."""
        global model, collision_model, visual_model

        logger.info("Reloading URDF from disk...")
        self.add_command("Reloading URDF...")

        try:
            # Reload models from URDF
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
            )
            logger.info(f"URDF reloaded: {collision_model.ngeoms} collision geometries")

            # Recreate data structures
            self.data = pin.Data(model)

            # Re-setup collision if enabled
            if self.enable_collision_detection:
                self.setup_collision_pairs()
                self.collision_data = pin.GeometryData(collision_model)
                logger.info("Collision model reloaded")

            # Update visualizer with new models
            self.viz.model = model
            self.viz.collision_model = collision_model
            self.viz.visual_model = visual_model
            self.viz.data = self.data

            # Reload viewer geometry
            self.viz.loadViewerModel()

            # Reset configuration to neutral
            self.q = pin.neutral(model)
            self.v = np.zeros(model.nv)
            self.q[2] = 0.0

            # Update display
            self.viz.display(self.q)

            self.add_command(f"URDF reloaded ({collision_model.ngeoms} geoms)")
            logger.info("URDF reload complete")

        except Exception as e:
            logger.error(f"Failed to reload URDF: {e}", exc_info=True)
            self.add_command(f"ERROR: Reload failed - {e}")

    def reset_simulation(self):
        """Reset entire simulation to initial state and reload URDF."""
        # Reload URDF to pick up any changes
        self.reload_urdf()

        # Reset ball
        self.ball_pos = np.array([0.3, 0.3, 0.5])
        self.ball_vel = np.zeros(3)

        # Update displays
        self.update_display()
        self.update_ball_position()
        self.add_command("Simulation reset + URDF reloaded")

    def toggle_gravity(self):
        """Toggle gravity on/off."""
        self.gravity_enabled = not self.gravity_enabled
        if self.gravity_enabled:
            # Reset ball position for new drop
            self.ball_pos = np.array([0.3, 0.3, 0.5])
            self.ball_vel = np.zeros(3)
            self.add_command("Gravity ON")
        else:
            # Stop all motion
            self.v = np.zeros(model.nv)
            self.ball_vel = np.zeros(3)
            self.add_command("Gravity OFF")

    def update_physics(self, dt):
        """Update physics simulation for one timestep."""
        if not self.gravity_enabled or self.paused:
            return

        # Gravity constant
        gravity = np.array([0, 0, -9.81])  # m/s^2

        # ---- Robot dynamics ----
        tau = np.zeros(model.nv)  # No control torques

        # Compute forward dynamics (ABA algorithm)
        a = pin.aba(model, self.data, self.q, self.v, tau)

        # Lock the base position (first 6 DOF)
        a[:6] = 0.0

        # Add damping to arm joints
        a[6:] *= 0.95  # 5% damping

        # Integrate velocity
        self.v += a * dt
        self.v[:6] = 0.0  # Lock base velocity

        # Integrate position
        self.q = pin.integrate(model, self.q, self.v * dt)

        # ---- Ball physics (simple particle) ----
        # Apply gravity acceleration
        self.ball_vel += gravity * dt

        # Update position
        self.ball_pos += self.ball_vel * dt

        # Ground collision (simple)
        BALL_RADIUS = 0.03  # 3cm radius
        if self.ball_pos[2] < BALL_RADIUS:
            self.ball_pos[2] = BALL_RADIUS
            self.ball_vel[2] = -self.ball_vel[2] * 0.6  # Bounce with energy loss
            if abs(self.ball_vel[2]) < 0.002:  # Stop if moving < 2mm/s
                self.ball_vel = np.zeros(3)  # Fully stop the ball

        # Update displays
        self.update_ball_position()

    def run(self):
        """Main control loop."""
        t = self.term
        logger.info("Entering main control loop")

        try:
            with t.fullscreen(), t.cbreak():
                logger.info("Fullscreen terminal mode active")

                while self.running:
                    try:
                        # Measure frame start time
                        frame_start = time.time()

                        # Update physics simulation
                        if self.last_frame_time is not None:
                            dt = frame_start - self.last_frame_time
                            self.update_physics(dt)

                        # Update visualizer
                        self.update_display()

                        # Draw UI
                        self.draw_ui()

                        # Calculate elapsed time for this frame so far
                        frame_elapsed = time.time() - frame_start

                        # Calculate dynamic timeout to maintain target rate
                        target_frame_time = 1.0 / self.target_hz
                        timeout = max(0.0, target_frame_time - frame_elapsed)

                        # Get key with dynamic timeout (0 if we're behind schedule)
                        key = t.inkey(timeout=timeout)

                        # Track frame timing (after inkey completes)
                        if self.last_frame_time is not None:
                            dt = frame_start - self.last_frame_time
                            self.frame_times.append(dt)
                        self.last_frame_time = frame_start
                        self.frame_count += 1

                        if not key:
                            continue

                        logger.debug(f"Key pressed: {repr(key)} (name={key.name if hasattr(key, 'name') else None})")

                        # Process key
                        if key.name == 'KEY_ESCAPE' or key.lower() == 'q':
                            logger.info("Exit key pressed")
                            self.running = False
                            self.add_command("Exiting...")

                        elif key.lower() == ' ':
                            self.paused = not self.paused
                            self.add_command(f"{'Paused' if self.paused else 'Resumed'}")

                        elif key.lower() == 'c':
                            self.enable_collision_detection = not self.enable_collision_detection
                            status = "enabled" if self.enable_collision_detection else "disabled"
                            self.add_command(f"Collision detection {status}")

                        elif key.lower() == 'v':
                            self.show_collision_geom = not self.show_collision_geom
                            # Update visualization to show/hide collision geometries
                            if self.enable_collision_detection:
                                self.update_collision_visualization()
                            status = "shown" if self.show_collision_geom else "hidden"
                            self.add_command(f"Collision geometries {status}")

                        elif key.lower() == 'e':
                            self.prevent_environment_collision = not self.prevent_environment_collision
                            status = "ON" if self.prevent_environment_collision else "OFF"
                            self.add_command(f"Environment collision prevention: {status}")

                        elif key.lower() == 'h':
                            logger.info("Home position requested")
                            self.home_position()

                        elif key.lower() == 'r':
                            logger.info("Reset simulation requested")
                            self.reset_simulation()

                        elif key.lower() == 'g':
                            logger.info("Toggle gravity requested")
                            self.toggle_gravity()

                        # F-keys for saving states (F1-F9)
                        elif key.name and key.name.startswith('KEY_F') and len(key.name) == 6:
                            try:
                                f_num = int(key.name[5])  # Extract number from KEY_F1, KEY_F2, etc.
                                if 1 <= f_num <= 9:
                                    logger.info(f"Saving state to slot {f_num}")
                                    self.save_state(f_num)
                            except (ValueError, IndexError):
                                pass

                        # Number keys for loading states (1-9)
                        elif key in '0123456789':
                            num = int(key)
                            # 1-6 for joint selection (maps to indices 0-5)
                            # 0, 7-9 for loading states
                            if 1 <= num <= 6:
                                # Joint selection (1-6 maps to indices 0-5)
                                self.selected_joint = num - 1
                                joint_name = JOINT_NAMES[self.selected_joint]
                                self.add_command(f"Selected joint {num}: {joint_name}")
                            elif num == 0 or num >= 7:
                                # Load state from slot 0, 7-9
                                logger.info(f"Loading state from slot {num}")
                                self.load_state(num)

                        elif key.name == 'KEY_LEFT' or key.lower() == 'a':
                            self.selected_joint = (self.selected_joint - 1) % len(JOINT_NAMES)
                            self.add_command(f"Selected: {JOINT_NAMES[self.selected_joint]}")

                        elif key.name == 'KEY_RIGHT' or key.lower() == 'd':
                            self.selected_joint = (self.selected_joint + 1) % len(JOINT_NAMES)
                            self.add_command(f"Selected: {JOINT_NAMES[self.selected_joint]}")

                        elif key.name == 'KEY_UP' or key.lower() == 'w':
                            step = LARGE_STEP_SIZE if key.name == 'KEY_SUP' else STEP_SIZE
                            old_pos = self.get_joint_position(self.selected_joint)
                            new_pos = old_pos + step
                            joint_name = JOINT_NAMES[self.selected_joint]

                            if self.set_joint_position_safe(self.selected_joint, new_pos):
                                self.update_display()
                                self.add_command(f"{joint_name}: {old_pos:+.2f}deg -> {new_pos:+.2f}deg")
                            else:
                                self.add_command(f"{joint_name}: BLOCKED - collision would occur at {new_pos:+.2f}deg")

                        elif key.name == 'KEY_DOWN' or key.lower() == 's':
                            step = LARGE_STEP_SIZE if key.name == 'KEY_SDOWN' else STEP_SIZE
                            old_pos = self.get_joint_position(self.selected_joint)
                            new_pos = old_pos - step
                            joint_name = JOINT_NAMES[self.selected_joint]

                            if self.set_joint_position_safe(self.selected_joint, new_pos):
                                self.update_display()
                                self.add_command(f"{joint_name}: {old_pos:+.2f}deg -> {new_pos:+.2f}deg")
                            else:
                                self.add_command(f"{joint_name}: BLOCKED - collision would occur at {new_pos:+.2f}deg")

                        elif key.lower() in ['+', '=']:
                            old_pos = self.get_joint_position(self.selected_joint)
                            new_pos = old_pos + LARGE_STEP_SIZE
                            joint_name = JOINT_NAMES[self.selected_joint]

                            if self.set_joint_position_safe(self.selected_joint, new_pos):
                                self.update_display()
                                self.add_command(f"{joint_name}: {old_pos:+.2f}deg -> {new_pos:+.2f}deg (large)")
                            else:
                                self.add_command(f"{joint_name}: BLOCKED - collision would occur at {new_pos:+.2f}deg")

                        elif key.lower() in ['-', '_']:
                            old_pos = self.get_joint_position(self.selected_joint)
                            new_pos = old_pos - LARGE_STEP_SIZE
                            joint_name = JOINT_NAMES[self.selected_joint]

                            if self.set_joint_position_safe(self.selected_joint, new_pos):
                                self.update_display()
                                self.add_command(f"{joint_name}: {old_pos:+.2f}deg -> {new_pos:+.2f}deg (large)")
                            else:
                                self.add_command(f"{joint_name}: BLOCKED - collision would occur at {new_pos:+.2f}deg")

                    except Exception as e:
                        # Log exceptions in the inner loop but continue running
                        logger.error(f"Error in control loop: {e}", exc_info=True)
                        self.add_command(f"Error: {str(e)[:40]}")
                        time.sleep(0.1)  # Brief pause to prevent spam

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
            self.add_command("Interrupted by user")

        except Exception as e:
            logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
            print(t.clear)
            print(f"\nFatal error: {e}")
            print(f"Check {LOG_FILE} for details")
            traceback.print_exc()

        finally:
            logger.info("Simulation ending")
            print(t.clear)
            print("Simulation ended!")
            print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    # Arguments already parsed at top of file
    enable_collision = args.collision

    print("Starting SO101 simulation...")
    print(f"URDF: {args.urdf}")
    print(f"Collision detection: {'ENABLED' if enable_collision else 'DISABLED'}")
    print(f"Collision geometries: {collision_model.ngeoms}")
    print("Open the visualizer at: http://127.0.0.1:7000/static/")
    print("Press any key to start...")

    sim = SO101Simulation(enable_collision=enable_collision)
    sim.run()
    print("Quit cleanly")
