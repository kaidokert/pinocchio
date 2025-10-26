# SO101 Robot Arm Simulation with Keyboard Control
# Uses blessed for terminal UI and Pinocchio for dynamics

import logging
import os
import sys
import time
import numpy as np
from pathlib import Path
from blessed import Terminal

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Configuration
STEP_SIZE = 2.0  # degrees per keypress
LARGE_STEP_SIZE = 10.0  # degrees with Shift
DT = 0.01  # simulation timestep

# Get model paths from environment
model_path = Path(os.environ.get("EXAMPLE_ROBOT_DATA_MODEL_DIR"))
mesh_dir = model_path.parent.parent
urdf_model_path = model_path / "so_arm_description/urdf/so101.urdf"

# Load robot model
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    str(urdf_model_path), str(mesh_dir), pin.JointModelFreeFlyer()
)

# Joint names (excluding the 6 DOF free flyer = indices 7 onwards)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_INDICES = list(range(7, 7 + len(JOINT_NAMES)))  # Indices in configuration vector


class SO101Simulation:
    def __init__(self):
        self.term = Terminal()

        # Initialize visualizer
        self.viz = MeshcatVisualizer(model, collision_model, visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

        # Add ground plane
        self.add_ground_plane()

        # Add test ball
        self.ball_pos = np.array([0.3, 0.3, 0.5])  # Start 50cm above ground
        self.ball_vel = np.zeros(3)
        self.add_test_ball()

        # Create data object for dynamics
        self.data = pin.Data(model)

        # Robot state
        self.q = pin.neutral(model)  # Configuration
        self.v = np.zeros(model.nv)  # Velocity

        # Set initial base position (sitting on ground)
        self.q[2] = 0.0  # Z position at ground level

        # Control state
        self.running = True
        self.selected_joint = 0
        self.last_command = "Ready"
        self.command_history = []
        self.paused = False

        # Display initial state
        self.viz.display(self.q)

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

    def add_test_ball(self):
        """Add a test ball to demonstrate gravity."""
        import meshcat.geometry as g
        import meshcat.transformations as tf

        # Create a red sphere
        self.viz.viewer["test_ball"].set_object(
            g.Sphere(0.05),  # 5cm radius
            g.MeshLambertMaterial(color=0xff0000)  # Red
        )
        self.update_ball_position()

    def update_ball_position(self):
        """Update test ball position in viewer."""
        import meshcat.transformations as tf
        self.viz.viewer["test_ball"].set_transform(
            tf.translation_matrix(self.ball_pos)
        )

    def get_joint_position(self, joint_idx):
        """Get position of a joint in degrees."""
        config_idx = JOINT_INDICES[joint_idx]
        return np.degrees(self.q[config_idx])

    def set_joint_position(self, joint_idx, value_deg):
        """Set position of a joint in degrees."""
        config_idx = JOINT_INDICES[joint_idx]
        self.q[config_idx] = np.radians(value_deg)

    def draw_bar(self, value, width=20, min_val=-180, max_val=180):
        """Draw a horizontal bar representing a value in degrees."""
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        filled = int(normalized * width)
        bar = "█" * filled + "░" * (width - filled)
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
        lines.append("")

        # Joint positions with bars
        lines.append(t.bold + "  JOINT POSITIONS:" + t.normal)
        lines.append("")

        for i, name in enumerate(JOINT_NAMES):
            pos = self.get_joint_position(i)

            # Highlight selected joint
            if i == self.selected_joint:
                color = t.black_on_green
                marker = " ◄"
            else:
                color = t.normal
                marker = "  "

            # Joint name and number
            lines.append(f"  {color}[{i}] {name:20s}{t.normal}{marker}")

            # Value and bar
            bar = self.draw_bar(pos, width=20, min_val=-180, max_val=180)
            value_color = t.green if -180 <= pos <= 180 else t.red
            lines.append(f"      {value_color}{pos:+7.2f}°{t.normal} │{bar}│")
            lines.append("")

        # Controls
        lines.append("  " + "─" * 60)
        lines.append("")
        lines.append(t.bold + "  CONTROLS:" + t.normal)
        lines.append(f"    {t.yellow}0-5{t.normal}         Select joint")
        lines.append(f"    {t.yellow}↑/W{t.normal}         Increase position (+{STEP_SIZE}°)")
        lines.append(f"    {t.yellow}↓/S{t.normal}         Decrease position (-{STEP_SIZE}°)")
        lines.append(f"    {t.yellow}←/A{t.normal}         Previous joint")
        lines.append(f"    {t.yellow}→/D{t.normal}         Next joint")
        lines.append(f"    {t.yellow}+/-{t.normal}         Large step (±{LARGE_STEP_SIZE}°)")
        lines.append(f"    {t.yellow}SPACE{t.normal}       Pause/Resume simulation")
        lines.append(f"    {t.yellow}H{t.normal}           Home position (all zeros)")
        lines.append(f"    {t.yellow}R{t.normal}           Reset simulation (robot + ball)")
        lines.append(f"    {t.yellow}G{t.normal}           Apply gravity (drop arm)")
        lines.append(f"    {t.yellow}Q/ESC{t.normal}       Quit")
        lines.append("")

        # Status line
        lines.append("  " + "─" * 60)
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

    def home_position(self):
        """Move all joints to zero position."""
        for i in range(len(JOINT_NAMES)):
            self.set_joint_position(i, 0.0)
        self.v = np.zeros(model.nv)
        self.update_display()
        self.add_command("Moved to home position")

    def reset_simulation(self):
        """Reset entire simulation to initial state."""
        # Reset robot configuration
        self.q = pin.neutral(model)
        self.v = np.zeros(model.nv)
        self.q[2] = 0.0  # Base on ground

        # Reset ball
        self.ball_pos = np.array([0.3, 0.3, 0.5])
        self.ball_vel = np.zeros(3)

        # Update displays
        self.update_display()
        self.update_ball_position()
        self.add_command("Simulation reset")

    def apply_gravity(self):
        """Let arm drop under gravity for a moment."""
        self.add_command("Applying gravity...")

        # Reset ball position for new drop
        self.ball_pos = np.array([0.3, 0.3, 0.5])
        self.ball_vel = np.zeros(3)

        # Gravity constant
        gravity = np.array([0, 0, -9.81])  # m/s^2

        # Proper gravity simulation using forward dynamics
        # Note: First 6 DOF are the free-flyer base
        tau = np.zeros(model.nv)  # No control torques

        for _ in range(100):  # 1 second of simulation
            # ---- Robot dynamics ----
            # Compute forward dynamics (ABA algorithm)
            a = pin.aba(model, self.data, self.q, self.v, tau)

            # Lock the base position (first 6 DOF)
            a[:6] = 0.0

            # Add damping to arm joints
            a[6:] *= 0.95  # 5% damping

            # Integrate velocity
            self.v += a * DT
            self.v[:6] = 0.0  # Lock base velocity

            # Integrate position
            self.q = pin.integrate(model, self.q, self.v * DT)

            # ---- Ball physics (simple particle) ----
            # Apply gravity acceleration
            self.ball_vel += gravity * DT

            # Update position
            self.ball_pos += self.ball_vel * DT

            # Ground collision (simple)
            if self.ball_pos[2] < 0.05:  # Ball radius
                self.ball_pos[2] = 0.05
                self.ball_vel[2] = -self.ball_vel[2] * 0.6  # Bounce with energy loss
                if abs(self.ball_vel[2]) < 0.1:  # Stop if moving slowly
                    self.ball_vel[2] = 0

            # Update displays
            self.update_display()
            self.update_ball_position()
            time.sleep(DT)

        self.v = np.zeros(model.nv)  # Stop motion
        self.ball_vel = np.zeros(3)  # Stop ball
        self.add_command("Gravity applied")

    def run(self):
        """Main control loop."""
        t = self.term

        try:
            with t.fullscreen(), t.cbreak():
                while self.running:
                    # Draw UI
                    self.draw_ui()

                    # Get key with timeout
                    key = t.inkey(timeout=0.05)

                    if not key:
                        continue

                    # Process key
                    if key.name == 'KEY_ESCAPE' or key.lower() == 'q':
                        self.running = False
                        self.add_command("Exiting...")

                    elif key.lower() == ' ':
                        self.paused = not self.paused
                        self.add_command(f"{'Paused' if self.paused else 'Resumed'}")

                    elif key.lower() == 'h':
                        self.home_position()

                    elif key.lower() == 'r':
                        self.reset_simulation()

                    elif key.lower() == 'g':
                        self.apply_gravity()

                    elif key in '012345':
                        self.selected_joint = int(key)
                        joint_name = JOINT_NAMES[self.selected_joint]
                        self.add_command(f"Selected joint {self.selected_joint}: {joint_name}")

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
                        self.set_joint_position(self.selected_joint, new_pos)
                        self.update_display()
                        joint_name = JOINT_NAMES[self.selected_joint]
                        self.add_command(f"{joint_name}: {old_pos:+.2f}° → {new_pos:+.2f}°")

                    elif key.name == 'KEY_DOWN' or key.lower() == 's':
                        step = LARGE_STEP_SIZE if key.name == 'KEY_SDOWN' else STEP_SIZE
                        old_pos = self.get_joint_position(self.selected_joint)
                        new_pos = old_pos - step
                        self.set_joint_position(self.selected_joint, new_pos)
                        self.update_display()
                        joint_name = JOINT_NAMES[self.selected_joint]
                        self.add_command(f"{joint_name}: {old_pos:+.2f}° → {new_pos:+.2f}°")

                    elif key.lower() in ['+', '=']:
                        old_pos = self.get_joint_position(self.selected_joint)
                        new_pos = old_pos + LARGE_STEP_SIZE
                        self.set_joint_position(self.selected_joint, new_pos)
                        self.update_display()
                        joint_name = JOINT_NAMES[self.selected_joint]
                        self.add_command(f"{joint_name}: {old_pos:+.2f}° → {new_pos:+.2f}° (large)")

                    elif key.lower() in ['-', '_']:
                        old_pos = self.get_joint_position(self.selected_joint)
                        new_pos = old_pos - LARGE_STEP_SIZE
                        self.set_joint_position(self.selected_joint, new_pos)
                        self.update_display()
                        joint_name = JOINT_NAMES[self.selected_joint]
                        self.add_command(f"{joint_name}: {old_pos:+.2f}° → {new_pos:+.2f}° (large)")

        except KeyboardInterrupt:
            self.add_command("Interrupted by user")

        finally:
            print(t.clear)
            print("Simulation ended!")


if __name__ == "__main__":
    print("Starting SO101 simulation...")
    print("Open the visualizer at: http://127.0.0.1:7000/static/")
    print("Press any key to start...")

    sim = SO101Simulation()
    sim.run()
