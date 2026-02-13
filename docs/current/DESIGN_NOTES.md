# Implementation Design Notes

This document outlines the key design decisions and architectural choices made during the implementation of the Phase 0-1 Minimum Viable Prototype (MVP). It is intended to provide context for future developers (AI or human) on *why* the code is structured the way it is.

## 1. Overall Strategy: MVP First

**Decision**: Prioritize implementing a minimal, end-to-end prototype that can validate the core theoretical hypothesis, rather than implementing all features from `ARCHITECTURE.md` at once.

**Rationale**:
- **Risk Mitigation**: The most significant risk in this project is the potential gap between the abstract theory and empirical reality. By building an MVP to test the coherence signal hypothesis first, we could quickly confirm if the foundational concept was sound before investing significant time in more complex components like the Agent Layer C (DreamerV3).
- **Faster Iteration**: A smaller codebase is easier to debug and modify. This allowed for rapid iteration and problem-solving, as seen with the `torch-geometric` and data loader issues.
- **Focus**: It forced a focus on the absolute essentials: the adjunction `F ⊣ G` and the `coherence_signal`.

## 2. Data: Synthetic Dataset Generator

**Decision**: Implement a synthetic data generator (`src/data/synthetic_dataset.py`) instead of immediately integrating the full 3D AffordanceNet dataset.

**Rationale**:
- **Development Velocity**: Large datasets like AffordanceNet involve significant download and preprocessing time. A synthetic generator provides an immediate, lightweight source of data, decoupling data preparation from model development.
- **Controlled Environment**: It allows for a highly controlled experimental setup. We can precisely define the "known" distribution (cubes, cylinders, spheres) and test against a clearly "novel" distribution (torus), making the experimental results easier to interpret.
- **Reproducibility**: The generator is deterministic (given a seed), ensuring that any agent can reproduce the exact training and test data.

## 3. Model Architecture (F & G)

### Functor F (Shape → Action)

**Decision**: Use a PointNet++ style architecture with `EdgeConv` layers from PyTorch Geometric.

**Rationale**:
- **Local Feature Extraction**: `EdgeConv` is effective at capturing local geometric features by constructing a k-NN graph at each layer. This aligns with the theoretical need to understand local shape properties to predict affordances.
- **Standard Practice**: It's a well-established and robust architecture for point cloud processing, reducing the risk of using a novel or untested GNN design.

### Functor G (Action → Shape)

**Decision**: Implement a conditional decoder that uses the predicted affordance vector to modulate the shape generation process.

**Rationale**:
- **Conditioning**: The core idea of `G` is to reconstruct the "functional core" of a shape *given* an affordance. The conditioning mechanism (concatenating the affordance vector to the point features) is a simple and direct way to implement this dependency.
- **Symmetry with F**: `F` maps from shape to affordance; `G` maps from affordance back to shape. The architecture reflects this duality.

## 4. Coherence Signal: Chamfer Distance

**Decision**: Use Chamfer distance as the metric for `distance(shape, G(F(shape)))`.

**Rationale**:
- **Permutation Invariance**: Chamfer distance is invariant to the order of points in a point cloud, which is a critical property for this task.
- **Differentiability**: It is fully differentiable, allowing it to be used directly as a loss component for end-to-end training.
- **Computational Efficiency**: While not as precise as Earth Mover's Distance (EMD), it is significantly more computationally efficient and sufficient for measuring the dissimilarity between two point clouds in this context.

## 5. Training Loop (Phase 1)

**Decision**: The loss function was defined as `L = L_recon + λ_aff * L_aff + λ_coh * L_coherence`.

**Rationale**:
- **`L_recon` (Reconstruction Loss)**: This is the `coherence_signal` itself. Minimizing this term forces the model to learn a good adjunction, where `G(F(shape))` is close to the original `shape`.
- **`L_aff` (Affordance Loss)**: A standard binary cross-entropy loss to ensure that `F` learns to predict meaningful affordances. Without this, the model could collapse to a trivial solution where `F` and `G` are identity functions.
- **`L_coherence` (Coherence Regularization)**: This term, `-log(coherence_signal)`, prevents the model from collapsing to a state where the coherence signal is always zero. It encourages the signal to be informative.

## 6. Directory Structure

**Decision**: Organize the code into a `src` directory with sub-packages for `data`, `models`, and `training`.

**Rationale**:
- **Modularity**: Separates concerns, making the codebase easier to navigate and maintain. `data` handles data loading, `models` contains the core network architectures, and `training` handles the learning process.
- **Python Packaging**: Using `__init__.py` files in each directory allows the code to be treated as a proper Python package. This enables clean imports (e.g., `from src.models import ...`) and avoids `sys.path` manipulation in production code (though it was used in test scripts for simplicity).
- **Scalability**: This structure is scalable. When we add the Agent Layer C, it can be placed in `src/models/agent_c.py` or a new `src/agent` directory without disrupting the existing layout.
