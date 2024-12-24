import os
import time
import numpy as np
import logging
from heapq import heappush, heappop
from dataclasses import dataclass, field
from typing import List, Optional
from mujoco import MjModel, MjData, mj_step, viewer
import hydra
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ------------------------------------------------------------------------
# Data Classes for State and Node
# ------------------------------------------------------------------------
@dataclass
class State:
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray

    @staticmethod
    def from_mjdata(data: MjData) -> 'State':
        return State(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            qacc=data.qacc.copy()
        )

    def apply_to_mjdata(self, data: MjData):
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.qacc[:] = self.qacc

@dataclass(order=True)
class Node:
    final_distance: float
    actions: List[np.ndarray] = field(compare=False)  # The entire action sequence
    depth: int = 0                                   # The depth in the search tree

# ------------------------------------------------------------------------
# Main Trajectory Planning Class
# ------------------------------------------------------------------------
class TrajectoryPlanning:
    def __init__(self, model: MjModel, data: MjData, initial_state: State, action_space: List[np.ndarray]):
        self.model = model
        self.data = data
        self.initial_state = initial_state
        self.action_space = action_space

    def set_simulation_state(self, state: State):
        """
        Set the simulation to a specific MuJoCo state.
        """
        state.apply_to_mjdata(self.data)
        mj_step(self.model, self.data)

    def simulate_sequence(self, actions: List[np.ndarray], target: np.ndarray, viewer_handle=None) -> float:
        """
        Simulate a sequence of actions and return the final distance to the target.
        """
        self.set_simulation_state(self.initial_state)

        for ctrl in actions:
            self.data.ctrl[:] = ctrl
            mj_step(self.model, self.data)
        if viewer_handle:
            viewer_handle.sync()

        final_position = self.data.geom_xpos[-1]
        return np.linalg.norm(final_position - target)

    def brute_force_search(self, target: np.ndarray, max_depth: int, tolerance: float = 0.02, viewer_handle=None) -> Node:
        """
        Priority-based BFS to find the optimal trajectory to reach the target using a heap.
        """
        root_node = Node(final_distance=float('inf'), actions=[], depth=0)
        priority_queue = []  # Priority queue based on distance
        heappush(priority_queue, root_node)
        best_node = root_node
        result_found = False

        def process_node(node: Node):
            nonlocal best_node, result_found

            # Evaluate the node if it hasn't been processed
            if node.final_distance == float('inf'):
                node.final_distance = self.simulate_sequence(node.actions, target, viewer_handle)

            # Update the best node if this node is better
            if node.final_distance < best_node.final_distance:
                best_node = node
                logging.info(f"{node.depth}. New best node found with distance: {node.final_distance:.4f}")
                if node.final_distance < tolerance:
                    result_found = True
                    logging.info("Target reached within tolerance.")
                    return

            # Expand child nodes if within max depth
            if node.depth < max_depth:
                for action in self.action_space:
                    actions = node.actions + [action]
                    node = Node(
                        final_distance=float('inf'),
                        actions=actions,
                        depth=node.depth + 1
                    )
                    heappush(priority_queue, node)

        # TODO currently mujoco is not support multi thread
        # # Multi-threaded node processing
        # max_threads = multiprocessing.cpu_count()
        # with ThreadPoolExecutor(max_threads) as executor:
        #     while priority_queue and not result_found:
        #         futures = [executor.submit(process_node, heappop(priority_queue)) 
        #                    for _ in range(min(len(priority_queue), max_threads))]
        #         for future in as_completed(futures):
        #             future.result()
        #             if result_found:
        #                 break
        
        while priority_queue and not result_found:
            process_node(heappop(priority_queue))
            
        return best_node
            

    def play_sequence(self, best_node: Node, target: np.ndarray, viewer_handle = None, timestep: Optional[float] = 0.01):
        """
        Play the best sequence from the initial state and visualize it.
        """
        while True:
            command = input("Press Enter to play the sequence or type 'exit' to quit: ")
            if command.lower() == 'exit':
                print("Exiting play sequence mode.")
                break
            
            logging.info("Replaying the best sequence.")
            self.set_simulation_state(self.initial_state)
            self.data.ctrl = np.zeros_like(self.data.ctrl)
            if viewer_handle:
                viewer_handle.sync()

            for ctrl in best_node.actions:
                self.data.ctrl[:] = ctrl
                mj_step(self.model, self.data)
                if viewer_handle:
                    viewer_handle.sync()
                time.sleep(timestep)

            final_position = self.data.geom_xpos[-1]
            final_distance = np.linalg.norm(final_position - target)
            logging.info(f"Final distance: {final_distance:.4f}")

# ------------------------------------------------------------------------
# Hydra-based Main Function
# ------------------------------------------------------------------------
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for trajectory planning with parallel BFS.
    """
    logging.basicConfig(
        level=cfg.logging.level,
        format=cfg.logging.format,
        filename=cfg.logging.filename,
        filemode=cfg.logging.filemode
    )

    model = MjModel.from_xml_path(cfg.simulation.model_path)
    data = MjData(model)

    if cfg.simulation.timestep is not None:
        model.opt.timestep = cfg.simulation.timestep

    initial_state = State.from_mjdata(data)

    action_space = [
        np.array([0.0, 0.0]),   # 1) No tension (fully relaxed)
        # np.array([0.5, 0.0]),   # 2) Half tension on cable1, none on cable2
        # np.array([0.0, 0.5]),   # 3) None on cable1, half tension on cable2
        np.array([0.5, 0.5]),   # 4) Moderate equal tension on both cables
        np.array([1.0, 0.0]),   # 5) Full tension on cable1, none on cable2
        np.array([0.0, 1.0]),   # 6) None on cable1, full tension on cable2
        np.array([0.3, 0.7]),   # 7) Imbalanced tension favoring cable2
        np.array([0.7, 0.3])    # 8) Imbalanced tension favoring cable1
    ]

    planner = TrajectoryPlanning(model, data, initial_state, action_space)

    target = np.array(cfg.simulation.target)
    max_depth = cfg.simulation.max_depth
    tolerance = cfg.simulation.tolerance

    viewer_handle = None
    if cfg.simulation.show_visualization:
        viewer_handle = viewer.launch_passive(model, data)

    best_node = planner.brute_force_search(target, max_depth, tolerance, viewer_handle)

    logging.info(f"Best distance found: {best_node.final_distance:.4f}")
    logging.info(f"Sequence: {best_node.actions}")

    planner.play_sequence(best_node, target, viewer_handle, cfg.simulation.timestep)

if __name__ == "__main__":
    main()
