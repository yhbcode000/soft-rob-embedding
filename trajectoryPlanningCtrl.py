import mujoco
import numpy as np
import time
import logging
from mujoco import viewer
import hydra
from omegaconf import DictConfig

# Hydra configuration
@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    # Logging setup
    logging.basicConfig(
        level=cfg.logging.level,
        filename=cfg.logging.filename,
        filemode=cfg.logging.filemode,
        format=cfg.logging.format
    )
    logger = logging.getLogger()

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(cfg.simulation.model_path)
    data = mujoco.MjData(model)

    # Parameters for the logarithmic spiral
    a = 1.0  # Scaling parameter
    b = 0.22  # Controls taper angle
    Dq = np.pi / 6  # Step size for discretization in radians

    # Target object properties
    target_position = np.array(cfg.simulation.target)  # [x, y, z]
    target_radius = 0.05  # Radius in meters

    # Actuation constants
    F_base = 0.5  # Base force (normalized for [0, 1])
    F_delta_max = 0.25  # Maximum change in force (normalized for [0, 1])
    step_size = cfg.simulation.timestep

    # Function: Generate a logarithmic spiral trajectory
    def generate_spiral_trajectory(start_pos, target_pos, a, b, steps=100):
        def spiral(theta):
            return a * np.exp(b * theta)

        # Calculate trajectory points
        angles = np.linspace(0, 2 * np.pi, steps)
        trajectory = np.array([
            [spiral(theta) * np.cos(theta), spiral(theta) * np.sin(theta), 0]
            for theta in angles
        ])
        # Align trajectory with the target
        trajectory[:, 2] = np.linspace(start_pos[2], target_pos[2], len(trajectory))
        trajectory += start_pos  # Translate trajectory to start position
        return trajectory

    # Function: Detect contact based on force thresholds
    def detect_contact(motor_current, threshold=cfg.simulation.tolerance):
        return motor_current > threshold

    # Generate trajectory
    trajectory = generate_spiral_trajectory(data.geom_xpos[-1], target_position, a, b)

    # Function: Adjust cable forces dynamically
    def calculate_cable_forces(current_pos, target_pos, base_force, delta_max):
        direction_vector = target_pos - current_pos
        distance = np.linalg.norm(direction_vector[:2])
        angle_to_target = np.arctan2(direction_vector[1], direction_vector[0])

        left_cable_force = np.clip(base_force + delta_max * np.cos(angle_to_target), 0, 1)
        right_cable_force = np.clip(base_force + delta_max * np.sin(angle_to_target), 0, 1)

        return left_cable_force, right_cable_force

    # Visualization setup
    viewer_handle = None
    if cfg.simulation.show_visualization:
        viewer_handle = viewer.launch_passive(model, data)

    best_distance = np.inf

    # Simulation loop
    for step in range(1000):
        current_pos = data.qpos[:3]
        current_target = trajectory[min(step, len(trajectory) - 1)]

        # Calculate cable forces dynamically
        left_force, right_force = calculate_cable_forces(current_pos, current_target, F_base, F_delta_max)

        # Apply forces to the robot
        data.ctrl[0] = left_force
        data.ctrl[1] = right_force

        # Step the simulation
        mujoco.mj_step(model, data)

        # Detect contact with the target
        # contact = detect_contact(data.sensor["motor_current"])
        # if contact:
        #     logger.info(f"Contact detected at step {step}: Position {current_pos}")
        #     break

        # Log state periodically
        if step % 10 == 0:
            logger.info(f"Step {step}: Position {current_pos}, Target {current_target}, Left Force {left_force}, Right Force {right_force}")

        # Update viewer if enabled
        if viewer_handle:
            viewer_handle.sync()
            
        best_distance = min(best_distance, np.linalg.norm(target_position - current_pos))
        
        time.sleep(cfg.simulation.timestep)

    # Final analysis
    logger.info("Simulation completed. Reviewing results...")

if __name__ == "__main__":
    main()
