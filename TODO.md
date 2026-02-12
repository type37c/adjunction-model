# TODO List: Prototype for Theoretical Validation

This document outlines the development tasks for building a prototype to validate the core theoretical claims of the Physical-Semantic Adjunction Model. The primary goal is to demonstrate the model's ability to address the symbol grounding problem, generalize to unknown objects, and exhibit creative problem-solving under constraint.

---

## Phase 0: Foundational Implementation (Replicating Knowns)

This phase focuses on establishing the basic architecture and ensuring that the core components (F, G, C) can be trained and function as expected on known data. This is a prerequisite for the more ambitious theoretical validation.

### Core Components

-   [ ] **Environment Setup**: Install PyTorch, PyTorch Geometric, and `inferactively-pymdp`. (See `ARCHITECTURE.md` Section 5.1 for core libraries.)
-   [ ] **Data Loader**: Implement a data loader for a simplified 3D dataset (e.g., basic geometric primitives like cubes, cylinders, spheres) with associated affordance labels and basic actions (push, pull, lift). This will be used for initial F and G training. (Reference: `3D AffordanceNet` [1] and `Contact-GraspNet` [2] repositories for data handling examples.)
-   [ ] **Adjoint Layer (F)**: Implement the GNN-based encoder for the `Shape -> Action` mapping. (Reference: `PointNet++` or `DGCNN` implementations in PyTorch Geometric, `3D AffordanceNet` [1] and `Contact-GraspNet` [2] for architecture details.)
-   [ ] **Adjoint Layer (G)**: Implement the conditional decoder for the `Action -> Shape` mapping. (Reference: General conditional VAE decoder patterns, `Andries et al. (2020)` [3] for conceptual approach.)
-   [ ] **Agent Layer (C)**: Implement the RSSM structure (`h` and `z` states) and the GRU-based update rule. (Reference: `DreamerV3` [4] implementations for RSSM architecture.)
-   [ ] **Action Selection**: Implement the Expected Free Energy (EFE) calculation for policy selection. (Reference: `Çatal et al. (2020)` [5] for EFE calculation, `inferactively-pymdp` for general Active Inference loop structure.)
-   [ ] **Loss Function**: Implement the composite loss function (`L_recon`, `L_affordance`, `L_vfe`). (Reference: `ARCHITECTURE.md` Section 7 for formulas, `DreamerV3` [4] and `Çatal et al. (2020)` [5] for VFE terms.)
-   [ ] **Training Loop**: Integrate all components into a single, end-to-end training script for Phase 0. (This is a new development task, integrating F, G, C, and losses as described in `ARCHITECTURE.md` Section 5.2.)

## Phase 1: Theoretical Validation - Setting A (Zero-Shot Affordance)

This phase focuses on validating the model's ability to generalize to unknown objects and infer affordances without prior exposure.

-   [ ] **Simulated Environment Setup**: Set up a basic physics simulation environment (e.g., PyBullet) capable of generating simple geometric primitives and simulating basic interactions (push, pull, lift).
-   [ ] **Data Generation for Training**: Generate a dataset of simple geometric primitives (cubes, cylinders, combinations) for Category A objects with associated affordances and actions within the simulation.
-   [ ] **Data Generation for Testing**: Generate a dataset of novel geometric primitives or Category B objects (e.g., tools) not seen during training, with ground-truth affordance labels for evaluation.
-   [ ] **Evaluation Script for Setting A**: Develop a script to: 
    -   Load the trained Phase 0 model.
    -   Present unknown objects to the model.
    -   Record `F(unknown_shape)` outputs (affordance predictions).
    -   Qualitatively and quantitatively evaluate the consistency of predicted affordances with human judgment.
    -   Analyze `G(F(unknown_shape))` for functional relevance.

## Phase 2: Theoretical Validation - Setting B (Creative Problem Solving Under Constraint)

This phase focuses on validating the emergence of novel behaviors when the agent's internal state (C) is constrained.

-   [ ] **Task Definition**: Define a simple manipulation task in the simulated environment (e.g., 
carrying a suitcase).
-   [ ] **Agent State (C) Manipulation**: Implement mechanisms to modify the agent state C to simulate constraints (e.g., `right_arm: disabled`, `noise_constraint: high`). This might involve masking parts of the state vector or altering physical parameters in the simulation.
-   [ ] **Evaluation Script for Setting B**: Develop a script to:
    -   Train the model for the defined task under `C_normal`.
    -   Introduce `C_injured` and observe changes in action selection.
    -   Introduce `C_complex` to induce `coherence breakdown` and observe for emergent, novel behaviors.
    -   Qualitatively analyze the generated actions for creativity and problem-solving.

## Phase 3: Theoretical Validation - Setting C (Symbol Grounding with Language)

This phase aims to align the learned Shape-Action representations with linguistic descriptions.

-   [ ] **Language Description Generation**: Integrate an LLM to generate functional descriptions for objects and actions.
-   [ ] **Alignment Mechanism**: Implement a method to align the model's internal representations (affordance distributions, functional skeletons) with language embeddings (e.g., using CLIP-like architectures or cosine similarity).
-   [ ] **Evaluation Script for Setting C**: Develop a script to:
    -   Evaluate the structural correspondence between the model's learned representations and linguistic descriptions.
    -   Potentially train a separate model to generate shapes from linguistic descriptions and compare with G's output.

## Undecided / Needs Further Discussion

-   [ ] **Suspension Structure Emergence Verification**: Based on the new principle ("we don't design the suspension structure, we design the conditions for its emergence"), this task is redefined. The goal is to design and verify the conditions that force the suspension structure to emerge.
    -   [ ] **Condition 1 (Non-vanishing Coherence Signal)**: Design a learning environment or loss function where the coherence signal is never permanently zero, forcing the agent to continuously adapt.
    -   [ ] **Condition 2 (Breakdown is not Fatal)**: Ensure the agent's architecture can withstand and recover from coherence breakdowns without catastrophic failure (e.g., through robust state management in Layer C).
    -   [ ] **Condition 3 (Free Movement of Abstraction)**: Implement a mechanism that allows the agent to shift its level of abstraction (λ) in response to coherence signals, enabling it to re-frame problems at different granularities.

## References

[1] Kim, H., et al. (2024). *Zero-Shot Learning for the Primitives of 3D Affordance in General Objects*. arXiv:2401.12978.
[2] Sundermeyer, M., et al. (2021). *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*. arXiv:2103.14243.
[3] Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. arXiv:1903.02428.
[4] Hafner, D., et al. (2023). *Mastering Diverse Domains through World Models*. arXiv:2301.04104.
[5] Çatal, O., et al. (2020). *Learning Generative State Space Models for Active Inference*. arXiv:2006.06520.
[6] Andries, J., et al. (2020). *Automatic Generation of Object Shapes With Desired Affordances*. Frontiers in Neurorobotics, 14, 22. doi: 10.3389/fnbot.2020.00022
