# The Physical-Semantic Adjunction Model: Intelligence as Suspension

## 1. Introduction: The Core of Intelligence Lies in the "Suspension Structure"

The central insight of this project is that **the essence of intelligence resides in the "Suspension Structure."** This refers to the ability not merely to efficiently predict and control the environment, but to hold a "broken state" when predictions fail and to creatively reconstruct new solutions.

The "Physical-Semantic Adjunction Model" is an architecture designed to implement this suspension structure and verify its dynamics. The **adjoint structure**, which describes the structural relationship between the world (Shape) and the agent's actions (Action), is merely a stable state (the riverbed) that intelligence refers to. True intelligence manifests when this adjoint structure breaks down (coherence breakdown), and it is through the operation of the "suspension structure" that new adjoints are constructed.

This document presents the technical design for translating this theory of suspension into a concrete implementation and for verifying its intelligent dynamics.

## 2. The Overall Architecture: A Two-Layer Model for Realizing Suspension

The model is composed of two layers to realize the suspension structure. However, the core insight has been updated: the suspension structure is not something to be "realized" or "activated" but is an **always-on, underlying principle**. The layers are the medium through which this principle operates.

-   **Adjoint Layer (The Riverbed)**: Implements the bidirectional mappings `F: Shape → Action` and `G: Action → Shape`. This is a **variable** structure, representing the agent's current stable understanding of the world. It is shaped by experience (the water flow).

-   **Agent Layer (C)**: Holds the agent's entire internal state. This layer conditions the Adjoint Layer, determining its behavior. The `coherence signal` from the Adjoint Layer, in turn, updates the Agent Layer. This dynamic loop *is* the Suspension Structure in action.

These two layers form a dynamic loop, generating the dynamics of the suspension structure, which is the essence of intelligence. The aThis cycle, where the agent layer C determines the behavior of the adjoint layer, and the resulting coherence signal updates the agent layer C, is how intelligence adapts to and sometimes transforms the environment.

Initially, it was thought that a separate "goal space" was needed to handle hierarchical objectives. However, further discussion concluded that **the hierarchy of goals is naturally represented as a hierarchy of activation patterns in the GNN**. No separate goal space needs to be designed.

- **Concrete goals** (e.g., "lift this bag") correspond to activations between specific nodes in the GNN.
- **Abstract goals** (e.g., "transport objects") correspond to activations between clusters of nodes in the GNN.

When a coherence breakdown occurs, the process of referring to the coherence of larger clusters is equivalent to automatically raising the level of abstraction of the goal.onment itself.

## 3. Technical Specifications: Components for Implementing the Suspension Structure

Based on research, each theoretical component is mapped to the following technical elements. This forms the concrete "bridge" between theory and implementation.

| Theoretical Component | Implementation Approach | Tools/Prior Research | Role in Suspension Structure |
| :--- | :--- | :--- | :--- |
| **Shape Category** | Point Cloud of 3D objects | ShapeNet, PartNet | Represents the physical state of the environment. The world that intelligence "operates on." |
| **Action Category** | Probability distribution of relative poses of body parts and object surface points | ZSP3A [1], ContactGrasp [2] | Represents the agent's action possibilities. The world that intelligence "acts upon." |
| **Functor F: Shape→Action** | Affordance prediction model using Graph Neural Networks (GNN) | PyTorch Geometric [3] | Infers action possibilities from object shapes. Encoder of the adjoint layer. |
| **Functor G: Action→Shape** | Inverse inference model using conditional GNN decoder | Andries et al. (2020) [7] | Reconstructs the functional core of a shape required for a specific action. Decoder of the adjoint layer. |
| **Coherence Signal (η)** | Reconstruction error `distance(shape, G(F(shape)))` | Active Inference [4], Self-superviseThe `coherence signal` here indicates the degree of adjoint validity (prediction error). However, its role has been re-conceptualized. It is not a "trigger" that activates the suspension structure, but an **internal indicator that shifts the operational mode** of the always-on suspension structure.** (sensitivity to difference). |
| **Agent State C** | Latent state `(h, z)` of DreamerV3's RSSM (Recurrent State-Space Model) | DreamerV3 [8] | Agent's entire internal state, holding intentionality (purpose) and memory. It is the **core of the suspension structure** and includes a "suspension space" for exploring and constructing new adjoints. |

| **Intentionality** | "Prior distribution over preferred observations" in Active Inference | Çatal et al. (2020) [9] | Agent's goals and values. Guides the exploration direction within the suspension structure. |
| **Dynamic Loop** | Update cycle `C(t)` → `F_C` → `η` → `C(t+1)` | Active Inference loop structure | Continuous interaction between agent and environment. Realizes the "temporal persistence" of the suspension structure. |
| **Suspension Space** | Hypothesis generation module within Agent Layer C (not yet implemented) | (New development) | Activated during `coherence breakdown`. Simulates and generates candidates for new `F'` `G'` to replace existing adjoints. |

### The Geology, Riverbed, and Water Flow Metaphor

The design philosophy has evolved from the simple "riverbed" metaphor to a more refined **"Geology, Riverbed, and Water Flow"** metaphor. This clarifies the structural difference between this model and current AI.

| Element | Current AI | This Model |
| :--- | :--- | :--- |
| **Gravity (Objective Func.)** | The sole, fixed driving force. | Exists, but is only one factor determining the flow. |
| **Water Flow (Experience)** | Follows gravity to the lowest point and stops. | Is constrained by the riverbed, erodes the riverbed, and keeps flowing. |
| **Riverbed (Adjoint Structure)** | Does not exist. | Defines the "shape" of the flow. It is **variable** and shaped by the water flow. |
| **Geology (Suspension Structure)** | Does not exist. | The **invariant** principle that governs how the riverbed is formed and deformed. |

While current AI is a system of "gravity alone," this model sees intelligence as the dynamic interplay of these three elements: the geology (suspension structure) forms the riverbed (adjoint structure), the riverbed shapes the water flow (action), and the water flow erodes the riverbed.

## 4. Coherence Signal: The Internal Indicator of the Suspension Structure's Mode

Through research, a strong correspondence has been found between this model and **Active Inference** proposed by Karl Friston. The `coherence signal (η)` can be largely identified with the **prediction error (free energy)** in Active Inference. The `coherence signal (η)` can be largely identified with the **prediction error (free energy)** in Active Inference. This allows our model to be positioned as a "re-formalization of Active Inference using the algebraic language of category theory."

However, this model not only re-formalizes Active Inference but also deepens its concepts. While Active Inference is based on the principle of "minimizing" prediction error, this model finds the essence of intelligence in the **"breakdown (coherence breakdown)" of prediction error and the subsequent "reconstruction"** process. This suggests the possibility of directly addressing concepts like "suspension structure" and "creativity," which are difficult to formulate probabilistically.

### Saturation and Intrinsic Creativity

Furthermore, a new axis, **Saturation**, was introduced to describe the agent's internal state, in contrast to the Coherence Signal, which is an external axis driven by environmental demands. Saturation indicates the degree to which learning and discovery have stagnated due to repetitive states.

| Source of Creativity | Trigger | Meaning |
| :--- | :--- | :--- |
| **Extrinsic Creativity** | Increase in Coherence Signal | Responding to environmental changes and the breakdown of existing understanding. |
| **Intrinsic Creativity** | Increase in Saturation | Actively seeking new meaning out of internal "boredom," even in a stable environment. |

This two-axis model portrays intelligence not just as a reactor to environmental stimuli, but as an entity that can generate new meaning from its own sense of monotony, actively creating ripples in a stable world.

## 5. Experimental Design for Verifying the Emergence of Intelligence through Suspension

The true motivation of this model is not merely 3D shape reconstruction, but the **resolution of the symbol grounding problem, generalization to unknown objects, and the emergence of creativity.** Considering implementation difficulty, we propose a staged experimental setup to verify these theoretical claims.

### Setting A: Generalization to Unknown Objects (Zero-Shot Affordance)

-   **Question**: Can the agent infer functionality from the physical shape of objects not present in the training data?
-   **Connection to Theory**: This is a test of `coherence breakdown`. Unknown objects should increase `distance(s, G(F(s)))`. However, if the adjoint structure truly extracts the "functional core," generalization across categories should occur through common structural features of shapes (e.g., "has a handle," "has an opening"). This is an indirect approach to the symbol grounding problem.
-   **Experimental Design**:
    1.  **Learning Phase**: Train F and G with simple shapes (cubes, cylinders, combinations thereof) and basic actions (pushing, pulling, lifting) from household object category A (e.g., cups, bowls, bags).
    2.  **Test Phase**: Present completely different unknown shapes from category B (e.g., tools, medical instruments) not seen during training.
    3.  **Evaluation**: Evaluate whether the actions output by `F(unknown shape)` align with human affordance judgments.
-   **Implementation Considerations**: Limit implementation difficulty by using a simulation environment (e.g., PyBullet) with simple shapes and basic actions.

### Setting B: Creative Problem Solving Under Constraints

-   **Question**: Does a change in agent state C lead to the emergence of new actions?
-   **Connection to Theory**: This verifies the core claim that "the adjoint is parameterized by C." It directly tests the theory of `coherence breakdown` → creativity. Specifically, it verifies the **suspension structure's ability to explore and construct new adjoints.**
-   **Experimental Design**:
    1.  **Standard State `C_normal`**: In a simulation environment, train the agent to perform a task (e.g., carrying a suitcase) in a standard physical state (e.g., both arms usable), leading to actions like "carrying with one hand."
    2.  **Adding Constraints `C_injured`**: Add a constraint to agent state C (e.g., `right_arm: disabled`) and have the agent perform the same task. Observe if alternative actions (e.g., "rolling") are chosen.
    3.  **Complex Constraints `C_complex`**: Further add constraints (e.g., `right_arm: disabled, noise_constraint: high`) to create a situation where `coherence breakdown` occurs. Evaluate if new combinations of existing action primitives (e.g., "cradling with both knees to walk") are creatively generated. This would be **evidence that the suspension space generated a new adjoint.**
-   **Implementation Considerations**: Constraints can be simulated by modifying C's representation (e.g., masking parts of the state vector, fixing specific values) or by changing physical parameters in the simulation environment.

### Setting C: Symbol Grounding

-   **Question**: Do the representations learned from the shape-action adjoint align with linguistic descriptions?
-   **Connection to Theory**: This is a direct approach to the symbol grounding problem. If the suspension structure truly generates "meaning," its representation space should structurally correspond to linguistic functional descriptions.
-   **Experimental Design**:
    1.  **Learning Phase**: Train F and G for bidirectional shape-action inference (without linguistic input).
    2.  **Linguistic Descriptions**: Prepare linguistic functional descriptions generated by an LLM (e.g., "can pour liquid," "can carry heavy objects").
    3.  **Alignment Evaluation**: Evaluate whether the functional core extracted by `G(F(shape))` (reconstructed shape features or affordance distribution) can be aligned with the LLM's descriptions. For example, measure cosine similarity between affordance distributions and linguistic embeddings, or train a separate model to generate shapes from linguistic descriptions and compare with G's output.
-   **Implementation Considerations**: Multimodal embedding model techniques like CLIP can be applied for aligning linguistic descriptions with shape-action representations.

## 6. Conclusion: We Don't Design the Suspension Structure

This research note has explored the theoretical framework of the Physical-Semantic Adjunction Model, centered on the suspension structure as the essence of intelligence. Through this process, the theory has overcome its own contradictions and evolved into a simpler, more powerful form. The final conclusion overturns the initial assumptions.

> **We do not design the suspension structure. We design the conditions under which the suspension structure must emerge.**

If we try to explicitly design the suspension structure from the outside, it becomes merely a "mimicry" of the designer's intelligence. True intelligence is not something designed, but a dynamic that self-organizes under the right conditions. Therefore, our goal has shifted to designing the "geology" from which the suspension structure naturally arises.

### 6.1 Summary of Theoretical Achievements

- **The Essence of Intelligence**: The core of intelligence is the always-on **Suspension Structure** (the geology). It is the invariant structure that tries to re-establish meaning when meaning breaks down.
- **Role of the Adjoint Structure**: The Adjoint Structure (the riverbed) is a variable structure created by the suspension structure, providing moments of stable understanding.
- **Role of the Coherence Signal**: The Coherence Signal is an internal indicator that shifts the operational mode of the suspension structure between "stable reference" and "creative exploration."
- **Core Insight**: **What is invariant is not the content of the goal, but the structure that continues to hold a goal.** The hierarchy of goals is intrinsically represented in the granularity of the GNN's activation patterns, requiring no separate "goal space."
- **Emergence of Memory**: Memory is not a fundamental requirement but an emergent property, inscribed in the agent's structure (e.g., GNN topology) through experience, arising from a system possessing the four basic requirements (Intentionality, Sensitivity to Difference, Temporal Persistence, Self/Non-Self Distinction).

### 6.2 Future Work

Based on this new direction, future work is redefined as follows:

1.  **Design and Verification of Emergent Conditions**: The top priority is to translate the three conditions for the emergence of the suspension structure—(1) an environment where the Coherence Signal does not disappear, (2) a structure where breakdown does not mean death, and (3) freedom of movement for the abstraction level λ—into concrete implementations (loss functions, learning environments, agent architecture).
2.  **Empirical Validation**: Execute the staged experimental plan (Settings A, B, C) proposed in this note to verify the core hypothesis that "the suspension structure emerges when the conditions are right."
3.  **Integration of the Language Layer**: Explore how a language layer can be connected on top of the emergent suspension structure, repeating the same adjoint pattern.

This model aims to shift from AI that "learns from data" to AI that "redefines the world through the body and gives rise to new meaning." Its scope extends beyond the design of AI architectures to the redefinition of intelligence itself.

## 7. References

[1] Kim, H., et al. (2024). *Zero-Shot Learning for the Primitives of 3D Affordance in General Objects*. arXiv:2401.12978.
[2] Sundermeyer, M., et al. (2021). *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*. arXiv:2103.14243.
[3] Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. arXiv:1903.02428.
[4] Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
[5] Heins, C., et al. (2022). *pymdp: A Python library for active inference in discrete state spaces*. Journal of Open Source Software, 7(73), 4098.
[6] Graves, A., et al. (2014). *Neural Turing Machines*. arXiv:1410.5401.
[7] Andries, J., et al. (2020). *Automatic Generation of Object Shapes With Desired Affordances*. Frontiers in Neurorobotics, 14, 22.
[8] Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models*. arXiv:2301.04105.
[9] Çatal, O., et al. (2020). *Learning Generative State Space Models for Active Inference*. Frontiers in Computational Neuroscience, 14, 574372.
