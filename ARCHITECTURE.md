# Architecture Design

This document outlines the concrete architecture for the prototype implementation of the Physical-Semantic Adjunction Model.

## 1. Overall Architecture

The model is a **2-Layer Architecture** built on PyTorch and PyTorch Geometric, integrating three main components:

1.  **Agent Layer (C)**: Manages the agent's internal state, memory, and decision-making. Based on Active Inference principles.
2.  **Adjoint Layer (F, G)**: Manages the bidirectional mapping between physical shapes and potential actions (affordances). Based on Geometric Deep Learning.
3.  **Dynamic Loop**: Connects the two layers, where the Agent Layer's actions influence the Adjoint Layer's context, and the Adjoint Layer's outputs inform the Agent Layer's beliefs.

```mermaid
graph TD
    subgraph Agent Layer (C)
        direction LR
        C_t["C(t) = (h_t, z_t)"]
        Action["Action Selection(a_t)"]
        C_t -- "Beliefs" --> Action
    end

    subgraph Adjoint Layer (F, G)
        direction TB
        F["F: Shape -> Action"] -- "Affordance Probs" --> G["G: Action -> Shape"]
    end

    subgraph Environment
        Shape["Shape (Point Cloud)"]
    end

    C_t_plus_1["C(t+1) = (h_{t+1}, z_{t+1})"]

    Shape -- "Observation" --> F
    G -- "Reconstruction & Coherence Signal" --> C_t_plus_1
    Action -- "Action (a_t)" --> C_t_plus_1
    C_t_plus_1 -- "Recurrence" --> C_t

```

## 2. Component Design

### 2.1. Agent Layer (C)

-   **Borrowed From**: **DreamerV3's RSSM (Recurrent State-Space Model)** for its proven performance and structure.
-   **State Representation `C = (h, z)`**:
    -   **`h` (Deterministic State)**: A GRU's hidden state. Represents the compressed history of past experiences. This implements the **Memory** requirement of the suspension structure.
    -   **`z` (Stochastic State)**: A categorical distribution over a set of discrete latent variables. Represents the agent's current belief about the world. This implements the **Perception/Interpretation** aspect.
-   **Update Rule `C(t) -> C(t+1)`**:
    1.  **Memory Update**: `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})`
    2.  **Belief Update**: `z_t ~ q(z_t | h_t, o_t)`, where `o_t` is the observation from the Adjoint Layer (including the coherence signal). This is the inference step.
    3.  **Action Selection**: Select action `a_t` that minimizes the **Expected Free Energy (EFE)**, following the formulation from **Çatal et al. (2020)**. This implements **Intentionality**.

### 2.2. Adjoint Layer (F, G)

-   **Functor F (Shape -> Action)**:
    -   **Borrowed From**: **3D AffordanceNet / Contact-GraspNet**.
    -   **Implementation**: A PointNet++ or DGCNN backbone that takes a point cloud (`Shape`) as input.
    -   **Output**: A probability distribution over a discrete set of affordances (e.g., `p(graspable), p(sittable), ...`).

-   **Functor G (Action -> Shape)**:
    -   **Borrowed From**: **Andries et al. (2020)** and standard **Conditional VAE Decoders**.
    -   **Implementation**: A conditional decoder that takes the affordance distribution from F as input.
    -   **Output**: A reconstructed point cloud representing the "functional skeleton" of the input shape that corresponds to the given affordances.

## 3. Loss Function

The model is trained end-to-end with a composite loss function:

1.  **Reconstruction Loss (Coherence Signal)**: `L_recon = ||Shape - G(F(Shape))||`. This is the primary measure for the **Sensitivity to Difference** requirement of the suspension structure.
2.  **Affordance Loss**: `L_affordance = CrossEntropy(F(Shape), GroundTruthAffordance)`. This supervises the F functor.
3.  **Variational Free Energy**: `L_vfe`. This is the core loss from Active Inference, driving both belief inference and action selection. It is composed of KL-divergence terms from the RSSM and the EFE for policy selection.

**Total Loss**: `L_total = w1*L_recon + w2*L_affordance + w3*L_vfe` (where w1, w2, w3 are weighting factors).
