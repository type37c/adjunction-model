# Adjunction Model Architecture (v2.0)

**Last Updated**: 2026-02-14

## 1. Core Philosophy: Suspension Structure and Slack

This model moves beyond traditional AI by focusing not on what the agent *knows*, but on how it *copes with not knowing*. The central concept is the **suspension structure**, an always-on process that governs how the agent's understanding of the world (the **adjunction F⊣G**) is formed, maintained, and repaired.

> The key insight is that intelligence is not the stability of the adjunction (the 'riverbed'), but the underlying principle that governs its formation (the 'geology').

This is achieved through the preservation and utilization of **slack**, a quantifiable measure of the "looseness" in the agent's world model. There are two types of slack:

1.  **η (Unit Slack)**: `||Shape - G(F(Shape))||²`
    -   Measures the **shape reconstruction error**. A high η means the agent's action-based understanding of a shape does not fully capture its geometry.
    -   Represents **physical or perceptual slack**.

2.  **ε (Counit Slack)**: `||Affordance - F(G(Affordance))||²`
    -   Measures the **affordance re-encoding error**. A high ε means the agent's reconstructed shape implies different actions than the original.
    -   Represents **semantic or conceptual slack**.

Our recent experiments have shown that **η and ε are strongly coupled** (Spearman correlation 0.92) and that removing reconstruction loss causes **total slack to grow by over 750%**, providing the agent with the flexibility needed for complex behavior.

## 2. Three-Phase Training Process

The model is trained in three distinct phases to manage the complex interplay between task learning and slack preservation.

![Three-Phase Training Process](docs/assets/three_phase_training.png)
*Figure 1: The three-phase training process, showing the active and frozen components in each phase.*

### Phase 1: Pre-training F⊣G with Reconstruction Loss

-   **Goal**: Learn a basic, stable adjunction F⊣G.
-   **Components**: Only F (Encoder) and G (Decoder) are active.
-   **Loss Function**: `L = L_aff + L_recon`
    -   `L_aff`: Affordance prediction loss (supervised).
    -   `L_recon`: Shape reconstruction loss (self-supervised).
-   **Outcome**: A "tight" adjunction where both η and ε are minimized. This provides a good initialization but lacks flexibility.

### Phase 2: Slack Preservation and Agent Co-training

-   **Goal**: Preserve and amplify slack (η, ε) while co-training Agent C.
-   **Components**: F, G, and Agent C are all active.
-   **Loss Function**: `L = L_aff + L_kl + L_coherence`
    -   `L_recon` is **removed** to allow slack to grow.
    -   `L_kl`: KL divergence loss to regularize Agent C's policy.
    -   `L_coherence`: A regularization term that encourages η to be non-zero.
-   **Outcome**: A "loose" adjunction with high slack, and an agent that begins to learn how to utilize it. This is the most critical phase for enabling complex behavior.

### Phase 3: Fine-tuning Agent C

-   **Goal**: Fine-tune Agent C's policy for a specific task.
-   **Components**: F and G are frozen; only Agent C is active.
-   **Loss Function**: `L = L_task + L_kl`
    -   `L_task`: A task-specific reward signal.
-   **Outcome**: A specialized agent that can exploit the preserved slack to solve complex problems efficiently.

## 3. Component Architecture

### 3.1. Adjoint Layer (F⊣G)

| Component | Model | Input | Output | Key Function |
| :--- | :--- | :--- | :--- | :--- |
| **F (Encoder)** | PointNet++ | Point Cloud (N, 3) | Affordance Dist. (B, A) | Shape → Action |
| **G (Decoder)** | FoldingNet | Affordance Dist. (B, A) | Point Cloud (N, 3) | Action → Shape |

-   **F (Functor F)**: A GNN-based encoder that maps a point cloud to a distribution over affordances. It learns the "meaning" of shapes in terms of action possibilities.
-   **G (Functor G)**: A conditional decoder that reconstructs a point cloud from an affordance distribution. It learns the "shape" of actions.

### 3.2. Agent Layer (C)

| Component | Model | Input | Output | Key Function |
| :--- | :--- | :--- | :--- | :--- |
| **Agent C** | RSSM | Observation, Prev. State | Action, Next State | Policy & World Model |

-   **Recurrent State-Space Model (RSSM)**: Agent C is implemented as an RSSM, which combines a deterministic GRU with a stochastic latent variable. This allows the agent to maintain a belief about the state of the world and plan future actions.
-   **Intrinsic Motivation**: The agent is driven by a combination of extrinsic task rewards and intrinsic motivation signals derived from the slack variables (η, ε) and coherence.

## 4. Key Implementation Details

-   **Data Format**: The model uses **graph-based data representations** (PyTorch Geometric `Data` objects) internally to handle variable-sized point clouds and batching.
-   **Slack Calculation**: η and ε are calculated at every forward pass and can be used as part of the loss function or as intrinsic rewards for Agent C.
-   **Model Versioning**: The current implementation is `ConditionalAdjunctionModelV4`, which includes the full three-phase logic and slack calculations.

## 5. Theoretical Justification

This architecture is a direct implementation of the **suspension structure** concept. 

-   **Phase 1** creates a stable but rigid "riverbed".
-   **Phase 2** introduces the "geology" by removing the reconstruction constraint, allowing the riverbed to become flexible and dynamic.
-   **Phase 3** lets the agent navigate this dynamic landscape to achieve its goals.

The discovery that **slack is used for exploitation, not exploration** (Slack-KL correlation -0.99) is a major finding. It suggests that the agent is not simply wandering aimlessly in its expanded space of possibilities, but is using this freedom to find more efficient and robust solutions to problems. This is the hallmark of a **competence-driven** system.

## 6. Temporal Suspension Experiment

Phase 2 Slack validated *static* suspension (Level 0). The **Temporal Suspension Experiment** extends this to *temporal* suspension (Level 1), where the agent must decide *when* to act.

### 6.1. Dataset: Progressive Shape Revelation

A 3D shape is revealed over T time steps. Early steps contain only an **ambiguous core** (points that could belong to any shape); later steps add **shape-specific** points.

| Step | Points | Ambiguity | Content |
| :--- | :--- | :--- | :--- |
| 0 | ~30 | 0.90 | Mostly ambiguous core |
| 3 | ~190 | 0.51 | Mixed core + shape-specific |
| 7 | 512 | 0.00 | Full shape revealed |

### 6.2. Agent Decision: Act or Wait

A **ConfidenceGate** maps the agent's context vector to a scalar confidence c(t) in [0, 1]. The agent "acts" (classifies the shape) at the first step where c(t) > threshold.

### 6.3. Comparison: Slack vs Tight

| Condition | L_recon | L_coherence | Expected Behaviour |
| :--- | :--- | :--- | :--- |
| **Slack** (Phase 2) | Removed | Active | Waits longer, higher accuracy |
| **Tight** (Phase 1) | Active | Removed | Acts early, lower accuracy |

### 6.4. Key Metrics

- **η(t) trajectory**: How unit slack evolves as the shape is revealed
- **Action timing**: Mean step at which the agent commits
- **Classification accuracy**: Correctness at the chosen action step

### 6.5. Implementation

| File | Purpose |
| :--- | :--- |
| `src/data/temporal_dataset.py` | Progressive revelation dataset |
| `experiments/temporal_suspension_experiment.py` | Main experiment script |
| `experiments/analyze_temporal_suspension.py` | Analysis and visualisation |
| `tests/test_temporal_suspension.py` | Integration tests (6 tests) |
