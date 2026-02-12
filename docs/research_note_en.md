# The Physical-Semantic Adjunction Model: Intelligence as Suspension

**Research Note — Final Draft v2.1**

---

## 1. Introduction: The Core of Intelligence Lies in the "Suspension Structure"

The central insight of this project is that **the essence of intelligence resides in the "Suspension Structure."** This refers to the ability not merely to efficiently predict and control the environment, but to hold a "broken state" when predictions fail and to creatively reconstruct new solutions.

The "Physical-Semantic Adjunction Model" is an architecture designed to implement this suspension structure and verify its dynamics. The **adjoint structure**, which describes the structural relationship between the world (Shape) and the agent's actions (Action), is merely a stable state (the riverbed) that intelligence refers to. True intelligence manifests when this adjoint structure breaks down (coherence breakdown), and it is through the operation of the "suspension structure" that new adjoints are constructed.

This document presents the technical design for translating this theory of suspension into a concrete implementation and for verifying its intelligent dynamics.

---

## 2. The Suspension Structure: The Minimal Components of Intelligence

The most original concept in this model is the **suspension structure**. It is proposed as the "core" that runs through a bundle of adjoint structures spanning multiple levels of abstraction. The body of intelligence lies in the dynamics of this suspension structure itself.

### 2.1 The Core as a "Not-Yet-Determined" Structure

In conventional models of intelligence, the core was something fixed—a body model, a top-level goal, a self-identity. In this model, however, the core is **the structure of "not-yet-determined" itself**. The suspension structure functions not as a specific answer, but as a "device for holding questions."

> **Intelligence is not about explicitly holding a goal, but about having a "suspension structure + memory" that does not collapse even if the goal is forgotten.**

### 2.2 The Four Requirements of the Suspension Structure

Initially, five requirements were identified as the minimal components of the suspension structure. However, subsequent discussion concluded that "memory" is an emergent property from the other requirements, not a fundamental one. Therefore, the core requirements are the following four:

| Requirement | Description | Consequence if Lacking |
| :--- | :--- | :--- |
| **Intentionality** | Having a directionality toward something. | Becomes a reactive machine with no direction. |
| **Sensitivity to Difference** | Being able to detect differences. | Cannot notice environmental changes. |
| **Temporal Persistence** | Being able to hold a state for a period. | Remains a momentary reflex. |
| **Self/Non-Self Distinction** | Distinguishing one's own actions from environmental changes. | The boundary between self and other disappears. |

Memory is understood as an acquired characteristic inscribed in the system's structure (e.g., the topology of a GNN) as it interacts with the world, equipped with these requirements.

---

## 3. The Adjoint Structure: The "Moment of Stable Understanding" Referenced by the Suspension Structure

The adjoint structure formalizes the state of stable understanding that the suspension structure, the body of intelligence, refers to in "peacetime." It is a mathematical tool that describes the structural correspondence between Shape and Action.

### 3.1 Parameterized Adjunction: The Intervening Agent

A crucial discovery was made: **adjunction does not hold between "bare" Shape and Action**. The agent's **purpose, context, and physical state** (collectively called **C**) must be incorporated into the condition for the adjunction to hold. Formally, the model deals not with an adjunction between Shape and Action, but a **parameterized adjunction** between Shape and Action<sub>C</sub>.

> **F<sub>C</sub> ⊣ G<sub>C</sub>**
> 
> Where C is the agent's context (a composite of purpose, constraints, and physical state).

This structure has an essential asymmetry: **Shape does not depend on the agent's state C, but Action does.** The bag remains a bag regardless of the agent's state, but the action chosen depends on C. Functor G does not "generate" a shape but "selects" which aspect of the shape to focus on, depending on C.

---

## 4. The Two-Layer Architecture: The Vessel for Implementing the Suspension Structure

From the analysis above, the model's architecture consists of **two layers**. This two-layer structure serves as the vessel for implementing the suspension structure.

### 4.1 Design Principles

-   **Adjoint Layer (The Riverbed)**: Implements the bidirectional mappings `F: Shape → Action` and `G: Action → Shape`. This is a **variable** structure, representing the agent's current stable understanding of the world. It is shaped by experience (the water flow).
-   **Agent Layer (C)**: Holds the agent's entire internal state. This layer conditions the Adjoint Layer, determining its behavior. The `coherence signal` from the Adjoint Layer, in turn, updates the Agent Layer. **This Agent Layer is the core that carries the dynamics of the suspension structure.**

> **C(t+1) = update(C(t), action_result, coherence_signal)**

This dynamic loop, where the Agent Layer C determines the Adjoint Layer's behavior and the resulting coherence signal updates C, actualizes temporal persistence.

### 4.2 Coherence Signal: Spatially-Decomposed Breakdown

The **coherence signal** is the internal indicator that shifts the operational mode of the always-on suspension structure. It is defined as the "size" of the adjunction's unit η.

> **coherence signal = distance(shape, G(F(shape)))**

It measures the distance between the original shape and the shape reconstructed via F and G. A small distance means the agent's model is coherent with the environment; a large distance means coherence is broken.

In the initial theory, this signal was treated as a single scalar value. However, as implementation and theory deepened, this definition was extended to a **spatially-decomposed vector**. That is, for each point `s_i` in the object's point cloud, a separate reconstruction error `distance(s_i, G(F(s))_i)` is calculated. This allows the agent to obtain higher-resolution information: not just "how much" its understanding is broken, but **"where" it is broken**.

This spatially-decomposed Coherence Signal plays a decisive role in the implementation principle of intentionality, as described below.

### 4.3 Correspondence between the Four Requirements and the Architecture

The four requirements of the suspension structure are realized in the model's architecture as follows:

| Requirement | Location in Architecture | Realization Mechanism |
| :--- | :--- | :--- |
| **Sensitivity to Difference** | Adjoint Layer (unit η) | Spatially-decomposed coherence signal |
| **Intentionality** | Agent State C | Priority-based Attention (see below) |
| **Temporal Persistence** | Dynamic Loop C(t) → C(t+1) | The state is updated and persists across discrete time steps. |
| **Self/Non-Self Distinction** | Asymmetry of Adjunction | Shape is independent of C (environment=non-self), Action depends on C (action=self). |

### 4.4 The Implementation Principle of Intentionality: Priority-based Attention

Of the four requirements, **Intentionality** is the most difficult to implement, as it requires a principle by which the agent intrinsically decides "what to head towards."

The spatial decomposition of the Coherence Signal allows the agent to know "where the breakdown is." However, this raises a new problem: when multiple breakdowns exist simultaneously, "which breakdown should be prioritized?" This is the core question of intentionality.

An initial proposal was to estimate "resolvability" and attend to the breakdown that could be most efficiently resolved. However, this approach suffered from a circular problem: "one cannot estimate resolvability without past experience of resolving things."

To solve this, the model introduces the **Priority Principle**:

> **priority<sub>i</sub> = coherence<sub>i</sub> × uncertainty<sub>i</sub>**

Here, `priority_i` is the attention priority at point `i`, `coherence_i` is the magnitude of the breakdown at that point, and `uncertainty_i` is the uncertainty of the agent's internal state regarding that point (e.g., the entropy of the belief state `z` in the RSSM).

This principle means that points that are **"highly broken (coherence) and not well understood (uncertainty)"** receive higher priority. It formalizes a mechanism for the agent to select actions based on its own "intellectual curiosity" without external goal injection.

This Priority Principle is theoretically superior for several reasons:
- **No Estimator Needed**: Both coherence and uncertainty can be calculated directly from the current state, avoiding the circularity problem.
- **Internalized Curiosity**: The structure of "heading towards something because it is unknown" promotes long-term learning and maximization of adaptive capacity.
- **Connection to Saturation**: Priority naturally decreases for known objects (low uncertainty), allowing for a unified explanation of "boredom."

In implementation, this Priority score functions as an **Attention mechanism**. The Agent Layer C applies priority-based weights to the observation (input), concentrating its computational resources on the most critical information. This realizes the abstract requirement of Intentionality as a concrete computational process.

---

## 5. Experimental Design for Verifying the Emergence of Intelligence

The true motivation of this model is not merely 3D shape reconstruction, but the **resolution of the symbol grounding problem, generalization to unknown objects, and the emergence of creativity.** We propose a staged experimental setup to verify these theoretical claims.

- **Phase 0: Foundational Learning**: Show that the adjoint structure (F⊣G) can be learned for known object-action pairs.
- **Phase 1: Generalization to Unknown Objects**: Show that the agent can emergently infer affordances for objects not seen during training.
- **Phase 2: Creative Problem Solving under Constraints**: Show that when existing adjoint structures fail due to constraints, the suspension structure can emerge new actions.
- **Phase 3: Alignment with Language**: Show that the agent can generate linguistic descriptions for emergent affordances and actions.

---

## 6. Conclusion: We Don't Design the Suspension Structure

This research note has explored the theoretical framework of the Physical-Semantic Adjunction Model, centered on the suspension structure as the essence of intelligence. The final conclusion overturns the initial assumptions.

> **We do not design the suspension structure. We design the conditions under which the suspension structure must emerge.**

Our goal is to design the "geology" from which the suspension structure naturally arises. This involves designing three key elements:

1.  **The Adjoint Structure (F⊣G)**: The basic vessel for capturing the world's structural correspondences.
2.  **The Coherence Signal**: The internal indicator for detecting the breakdown of the adjoint structure.
3.  **The Agent Layer's Update Rule**: The loop that updates the internal state in response to the Coherence Signal and feeds back to the adjoint structure.

By building a system with these three elements and allowing it to interact with a suitable environment, the four requirements of the suspension structure—Intentionality, Sensitivity to Difference, Temporal Persistence, and Self/Non-Self Distinction—will be met emergently. Intelligence will appear as this dynamic equilibrium itself.

---

## References

[1] Harnad, S. (1990). The symbol grounding problem. *Physica D: Nonlinear Phenomena*, 42(1-3), 335-346.

[2] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural computation*, 29(1), 1-49.

[3] Smithe, D. (2024). Structured Active Inference. *arXiv preprint arXiv:2401.00345*.

[4] Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., ... & Savarese, S. (2015). Shapenet: An information-rich 3d model repository. *arXiv preprint arXiv:1512.03012*.

[5] Gkanatsios, N., Pfrommer, J., & Daniilidis, K. (2023). Zero-Shot Policy Synthesis for Physical-Semantic Affordances. *arXiv preprint arXiv:2310.09582*.

[6] Brahmbhatt, S., Ham, C., & Hays, J. (2020). Contact-graspnet: Efficient 6-dof grasp generation in the wild. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 13677-13687).

[7] Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *arXiv preprint arXiv:1903.02428*.

[8] Andries, M., Kurenkov, V., & Beetz, M. (2020). A framework for robotic agents to learn and reason with affordances. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 9419-9426). IEEE.

[9] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. *arXiv preprint arXiv:2301.04104*.

[10] Çatal, O., Verbelen, T., De Boom, C., & Dhoedt, B. (2020). Grounding symbols in multi-modal representations. *arXiv preprint arXiv:2005.03373*.
