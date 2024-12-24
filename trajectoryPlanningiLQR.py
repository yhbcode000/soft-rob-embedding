import hydra
import numpy as np
import time
import logging
import mujoco
from mujoco import MjModel, MjData, mj_step, viewer
from dataclasses import dataclass

@dataclass
class SpiralState:
    qpos: np.ndarray
    qvel: np.ndarray
    cable_forces: np.ndarray

    @staticmethod
    def from_mjdata(data: MjData, num_cables: int) -> 'SpiralState':
        return SpiralState(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            cable_forces=data.ctrl[:num_cables].copy()
        )

    def apply_to_mjdata(self, data: MjData):
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.ctrl[:len(self.cable_forces)] = self.cable_forces

class SpiRobTrajectoryPlanner:
    def __init__(self, model: MjModel, data: MjData, initial_state: SpiralState, num_cables: int,
                 horizon: int, ilqr_iterations: int, alpha: float, tolerance: float = 0.1):
        self.model = model
        self.data = data
        self.initial_state = initial_state
        self.num_cables = num_cables
        self.horizon = horizon
        self.ilqr_iterations = ilqr_iterations
        self.alpha = alpha
        self.tolerance = tolerance

        # Initialize control sequence for cables
        self.u_sequence = np.zeros((horizon, num_cables))

    def set_simulation_state(self, state: SpiralState):
        state.apply_to_mjdata(self.data)
        mujoco.mj_forward(self.model, self.data)

    def forward_simulation(self, u_sequence: np.ndarray):
        states = []
        self.set_simulation_state(self.initial_state)
        states.append(SpiralState.from_mjdata(self.data, self.num_cables))

        for t in range(self.horizon):
            self.data.ctrl[:self.num_cables] = u_sequence[t]
            mj_step(self.model, self.data)
            states.append(SpiralState.from_mjdata(self.data, self.num_cables))
            logging.debug(f"Step {t}, State: {states[-1]}, Control: {u_sequence[t]}")

        return states

    def cost_function(self, state: SpiralState, u: np.ndarray, target_xyz: np.ndarray):
        self.set_simulation_state(state)
        mujoco.mj_forward(self.model, self.data)
        end_effector_pos = self.data.geom_xpos[-1]
        pos_error = np.linalg.norm(end_effector_pos - target_xyz)
        control_cost = np.linalg.norm(u)
        logging.debug(f"Cost function - Position Error: {pos_error}, Control Cost: {control_cost}")
        return 0.5 * (pos_error**2) + 0.01 * (control_cost**2)

    def backward_pass(self, states, u_seq, target_xyz):
        u_new = u_seq.copy()
        for t in reversed(range(self.horizon)):
            grad_u = np.zeros_like(u_seq[t])
            base_cost = self.cost_function(states[t], u_seq[t], target_xyz)

            for i in range(len(u_seq[t])):
                u_perturb = u_seq[t].copy()
                u_perturb[i] += 1e-4
                perturbed_cost = self.cost_function(states[t], u_perturb, target_xyz)
                grad_u[i] = (perturbed_cost - base_cost) / 1e-4

            u_new[t] -= self.alpha * grad_u
            u_new[t] = np.clip(u_new[t], 0.0, 1.0)
            logging.debug(f"Backward pass - Step {t}, Gradients: {grad_u}, Updated Control: {u_new[t]}")

        return u_new

    def plan(self, target_xyz: np.ndarray):
        for iteration in range(self.ilqr_iterations):
            states = self.forward_simulation(self.u_sequence)

            total_cost = sum(
                self.cost_function(states[t], self.u_sequence[t], target_xyz)
                for t in range(self.horizon)
            )
            logging.info(f"[iLQR] Iteration {iteration} - Cost: {total_cost:.4f}")

            u_updated = self.backward_pass(states, self.u_sequence, target_xyz)
            self.u_sequence = u_updated

        return self.u_sequence

    def play_sequence(self, u_sequence, viewer_handle=None):
        while True:
            command = input("Press Enter to play the sequence or type 'exit' to quit: ")
            if command.lower() == 'exit':
                print("Exiting play sequence mode.")
                break
            self.set_simulation_state(self.initial_state)
            for t in range(self.horizon):
                self.data.ctrl[:self.num_cables] = u_sequence[t]
                mj_step(self.model, self.data)
                if viewer_handle:
                    viewer_handle.sync()
                logging.debug(f"Playback - Step {t}, Control: {u_sequence[t]}")
                time.sleep(self.model.opt.timestep)


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        filename=cfg.logging.filename,
        filemode=cfg.logging.filemode,
        format=cfg.logging.format
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting the SpiRob trajectory planning.")

    # Load MuJoCo model and data
    model = MjModel.from_xml_path(cfg.simulation.model_path)
    data = MjData(model)
    model.opt.timestep = cfg.simulation.timestep

    # Initial state
    initial_state = SpiralState.from_mjdata(data, num_cables=2)

    # Planner configuration
    planner = SpiRobTrajectoryPlanner(
        model=model,
        data=data,
        initial_state=initial_state,
        num_cables=2,
        horizon=cfg.simulation.ilqr_horizon,
        ilqr_iterations=cfg.simulation.ilqr_iterations,
        alpha=cfg.simulation.ilqr_alpha,
        tolerance=cfg.simulation.tolerance
    )

    # Target position
    target_xyz = np.array(cfg.simulation.target)

    # Plan the trajectory
    best_u_sequence = planner.plan(target_xyz)

    # Launch viewer for visualization (optional)
    viewer_handle = None
    if cfg.simulation.show_visualization:
        viewer_handle = viewer.launch_passive(model, data)

    # Play the planned sequence
    planner.play_sequence(best_u_sequence, viewer_handle=viewer_handle)

    logger.info("SpiRob trajectory planning complete.")

if __name__ == "__main__":
    main()