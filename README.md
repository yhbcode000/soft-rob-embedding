# Soft-Rob-Embedding

Unifying the representation of robot statuses and actions with natural language embeddings by integrating robotics with large language models. This project aligns robotic states and actions into a shared vector space with natural language, enabling intuitive, human-friendly interaction paradigms. By leveraging the semantic richness of language models, the system can interpret robotic states and generate actions similarly to processing natural language, facilitating more intuitive and efficient robotic planning.

---

## Features 🚀

- **Unified Embeddings**: Combine robot states and actions with natural language for seamless planning.
- **Soft Robot Trajectory Planning**: Enable efficient control and natural movement for soft robots.

---

## Challenges 🤔

**Defining Actions in Non-Mimic Robots**

When planning robot actions in the action space, it is necessary to predefine semantically meaningful actions. This is relatively straightforward for humanoid and robot dogs but becomes challenging for non-mimic robots, complex robots, and tangibles (e.g., soft robots). This project aims to overcome this challenge by unifying robot states and language embeddings to define actions in abstract, LLM-understandable representations.

**Trajectory Planning in Soft Robots**

In typical robotic systems, trajectories can be planned by interpolating between two states and following the trajectory. However, in soft robots, trajectory planning is complicated by the state's dependency on historical movements. This requires projecting the control (whether predefined or abstract) and time-space into the robot's state space. To address this, the project leverages reinforcement learning (RL) to achieve a suboptimal solver, reducing computational complexity to O(n).

---

## FAQ ❓

**How does this differ from end-to-end robot control models?**

Soft robots have significantly higher degrees of freedom (DoF) than rigid robots, making direct simulation and RL application nearly impossible. Common approaches simplify soft robots by treating them as rigid robots or relying on predefined actions to manage complexity. While effective, these approaches limit the possibilities of soft robotics. By expanding the action space with trajectory planning models, we aim to realize the unique potential of soft robots, enabling conditional reflection and richer interactions.

**What is the next step for this project?**

Utilizing multi-agent knowledge emergence to develop action embeddings. Exploring Human-Robot Interaction (HRI) scenarios to test system integration with human users. Supporting subsystems such as intent recognition will also be investigated to unify multi-agent and HRI frameworks.

---

## Wiki 📖

Comprehensive documentation, including system architecture, design principles, and usage guidelines, is available on the project [Wiki](https://github.com/yhbcode000/soft-rob-embedding/wiki).

---

## Contribution 🙌

For collaboration or inquiries, please contact the authors. Reach out to either the paper authors or the repository maintainers.

Wang, Z., Freris, N. M., & Wei, X. "SpiRobs: Logarithmic spiral-shaped robots for versatile grasping across scales," Device, 2024. [DOI:10.1016/j.device.2024.100646](https://linkinghub.elsevier.com/retrieve/pii/S2666998624006033)

---

## License 📝

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
