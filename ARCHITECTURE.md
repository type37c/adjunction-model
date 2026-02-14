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
-   **Model Versioning**: The current implementation is `AdjunctionModel`, which includes slack calculations and the adjunction structure.

## 5. Theoretical Justification

This architecture is a direct implementation of the **suspension structure** concept. 

-   **Phase 1** creates a stable but rigid "riverbed".
-   **Phase 2** introduces the "geology" by removing the reconstruction constraint, allowing the riverbed to become flexible and dynamic.
-   **Phase 3** lets the agent navigate this dynamic landscape to achieve its goals.

The discovery that **slack is used for exploitation, not exploration** (Slack-KL correlation -0.99) is a major finding. It suggests that the agent is not simply wandering aimlessly in its expanded space of possibilities, but is using this freedom to find more efficient and robust solutions to problems. This is the hallmark of a **competence-driven** system.

## 6. Purpose-Emergent Active Assembly

The Phase 2 Slack experiments validated the foundational hypothesis of slack preservation. The **Purpose-Emergent Active Assembly** experiment extends this framework to test whether purpose (directional intent) can emerge from slack without explicit target assignment.

### 6.1. Concept: Purposeless Assembly

Unlike traditional supervised learning where the agent is told what to achieve, this experiment explores emergent goal formation:

- Points start randomly scattered in 3D space.
- Multiple reference shapes (sphere, cube, cylinder) are available but the agent is **NOT told which to target**.
- The purpose loss is `min_shape CD(assembled, reference)` — the agent is rewarded for approaching **any coherent shape**.
- The key hypothesis: the agent should spontaneously "choose" a reference shape and move toward it.

### 6.2. Dataset: PurposelessAssemblyDataset

The dataset generates episodes without explicit target assignments:

- **Initial Points**: Randomly scattered points in 3D space (uniform distribution).
- **Reference Shapes**: Sphere, cube, and cylinder are pre-computed and shared across all episodes.
- **No Target Labels**: Unlike supervised tasks, no target shape is assigned per episode.
- **Action Space**: The agent produces per-point displacement vectors to assemble the cloud.

### 6.3. Agent Action: Displacement Vectors

A **DisplacementHead** architecture maps the agent's context (from the adjunction model) and per-point affordance features to a displacement vector Δx ∈ R³ per point:

- **Input**: Affordance embeddings + agent context vector
- **Output**: Per-point displacement Δx
- **Update Rule**: x(t+1) = x(t) + Δx(t)

The agent iteratively refines the point cloud assembly over multiple steps.

### 6.4. Loss Function

The training loss combines multiple objectives:

**L = λ_purpose · L_purpose + λ_coherence · L_coherence + λ_kl · L_kl**

Where:
- **L_purpose**: `min_shape CD(assembled, ref_shape)` — minimum Chamfer Distance across all reference shapes
- **L_coherence**: Coherence regularization to prevent slack collapse
- **L_kl**: KL divergence regularization for the agent's policy

The purpose loss creates a reward landscape where the agent benefits from approaching any coherent structure, without being told which one.

### 6.5. Key Hypothesis

**Purpose emerges from slack without explicit target assignment.**

The agent should spontaneously exhibit goal-directed behavior by:
1. Exploring the space of possible assemblies early in training
2. Converging toward specific reference shapes as training progresses
3. Demonstrating consistent "choices" of target shapes within episodes

This would validate that intentional structure (purpose) is not imposed from outside but emerges from the slack in the agent's world model.

### 6.6. Implementation

| File | Purpose |
| :--- | :--- |
| `src/data/purposeless_dataset.py` | Purposeless assembly dataset (scattered points, no target) |
| `experiments/purpose_emergent_experiment.py` | Main experiment with purpose loss and displacement head |
| `experiments/analyze_purpose_emergent.py` | Analysis and comparison plots |
| `tests/test_purpose_emergent.py` | Integration tests for dataset and model |
