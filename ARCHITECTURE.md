# Architecture Design

This document describes the concrete architecture for the prototype implementation of the Physical-Semantic Adjunction Model. It is written so that a developer with no prior context can understand the design intent and begin implementation by reading this file alone.

## 1. Theoretical Background

This model addresses the **symbol grounding problem** in AI: current systems manipulate symbols (words, labels) without understanding their physical meaning. The core idea is that **meaning arises from the structural correspondence between physical shapes and the actions they afford**. A cup's shape means "drinkable" not because of a label, but because its geometry enables the action of drinking.

However, the true essence of intelligence in this model is not the stable **adjunction (F⊣G)** itself, but the **suspension structure** that dynamically maintains and repairs this adjunction. The adjunction represents a moment of stable understanding, a 'riverbed' where meaning flows. But when this stable state breaks down (a `coherence breakdown`), the agent's intelligence is revealed in its ability to hold this broken state in 'suspension' and actively seek to re-establish a new, coherent understanding. This dynamic process of breakdown, suspension, and repair is the core of creative problem-solving and true intelligence.

The adjunction is formalized using category theory — a pair of structure-preserving maps F and G between two domains (Shape and Action) that are not inverses of each other, but are "optimally related" in a precise mathematical sense. This adjunction is **conditioned on an agent's internal state C** (goals, memory, physical constraints), meaning the same shape affords different actions depending on who is interacting with it. The agent's internal state C is where the 'suspension structure' resides, driving the search for new adjunctions when old ones fail.

## 2. Two-Layer Architecture

The model has two layers connected by a dynamic loop:

**Adjoint Layer (F, G)**: Handles the structural relationship between shapes and actions. This is where the adjunction lives.

**Agent Layer (C)**: Manages the agent's internal state — memory, beliefs, and goals. This is what parameterizes the adjunction.

```
┌─────────────────────────────────────────────────────────┐
│  Agent Layer (C)                                        │
│  C(t) = (h_t, z_t)                                     │
│  h_t: deterministic memory (GRU hidden state)           │
│  z_t: stochastic belief (categorical distribution)      │
│                                                         │
│  Update: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})          │
│          z_t ~ q(z_t | h_t, o_t)                        │
│          a_t = argmin Expected Free Energy              │
│                                                         │
│  Implements: Memory, Intentionality, Temporal Duration  │
├─────────────────────────────────────────────────────────┤
│  Adjoint Layer (F, G)                                   │
│                                                         │
│  Point Cloud ──► [F: GNN Encoder] ──► Affordance Dist.  │
│                                           │              │
│                                           ▼              │
│  Reconstructed  ◄── [G: Conditional  ◄── Affordance     │
│  Point Cloud         Decoder]             Dist.          │
│                                                         │
│  Coherence Signal = ||Original - Reconstructed||        │
│                                                         │
│  Implements: Sensitivity to Difference, Self/Non-Self   │
└─────────────────────────────────────────────────────────┘
```

## 3. Adjoint Layer: F and G

### 3.1. Why GNN?

The input data is **3D point clouds** — sets of (x, y, z) coordinates representing object surfaces. A point cloud is naturally a graph: each point is a node, and edges connect nearby points (k-nearest neighbors). Graph Neural Networks (GNNs) are the standard architecture for learning on such structures, because they can capture local geometric patterns (edges, surfaces, holes) through message passing between neighboring nodes.

We use **PointNet++** or **DGCNN** as the specific GNN variant. Both are well-established for point cloud processing and have open-source implementations in PyTorch Geometric.

### 3.2. F (Shape → Action): The GNN Encoder

F takes a point cloud as input and outputs a probability distribution over affordance categories.

| Property | Specification |
|---|---|
| Input | Point cloud: N points × 3 coordinates |
| Architecture | PointNet++ (or DGCNN) backbone |
| Output | Affordance probability vector: e.g., `[p(graspable), p(sittable), p(supportable), ...]` |
| Supervision | **Directly supervised** by ground-truth affordance labels from the 3D AffordanceNet dataset (18 affordance categories, per-point annotations) |
| Borrowed From | 3D AffordanceNet [1], Contact-GraspNet [2] |

F is **not** a generic encoder. Its output space is forced to correspond to meaningful action categories by the supervised affordance labels. This is critical: the bottleneck between F and G is not an arbitrary latent space, but the **space of possible actions**.

### 3.3. G (Action → Shape): The Conditional Decoder

G takes the affordance distribution produced by F and reconstructs the point cloud.

| Property | Specification |
|---|---|
| Input | Affordance probability vector (output of F) |
| Architecture | Conditional decoder (MLP or FoldingNet-style) |
| Output | Reconstructed point cloud: N points × 3 coordinates |
| Supervision | Implicitly supervised through the reconstruction loss |
| Borrowed From | Andries et al. (2020) [3], standard conditional VAE decoders |

G does not reconstruct the original shape perfectly. It reconstructs the **functional skeleton** — the geometric features that are relevant to the affordances. A cup and a bowl may produce similar reconstructions if their affordance profiles are similar, because G can only "see" the shape through the lens of what actions it enables.

### 3.4. F + G Together: An Autoencoder Through Action Space

F and G together form a structure similar to an autoencoder:

```
Point Cloud → [F: Encoder] → Affordance Distribution → [G: Decoder] → Reconstructed Point Cloud
```

However, this differs from a standard autoencoder in two critical ways:

**First, the bottleneck is semantically meaningful.** In a standard autoencoder, the latent space is an arbitrary compressed representation of the input. Here, the bottleneck is the **action space** — a space of affordance probabilities with ground-truth supervision. Shape information can only pass through the bottleneck by being translated into action possibilities. This is the computational realization of the adjunction: shapes and actions are in different spaces, but structurally coupled.

**Second, the reconstruction error has theoretical significance.** In a standard autoencoder, reconstruction error is just a training signal. Here, `||Shape - G(F(Shape))||` is the **coherence signal** — it measures how well the agent's action-based understanding of a shape captures the shape's actual geometry. A high coherence signal means the agent's action model is missing something about the shape. This corresponds to the unit η of the adjunction in category theory.

### 3.5. How C Conditions the Adjoint Layer

The agent state C parameterizes the adjunction. In implementation, this means C is provided as an additional input to both F and G (e.g., concatenated to the feature vectors). When C changes (different goals, different memories), F and G produce different outputs for the same shape.

This realizes the theoretical concept of **conditional adjunction**: the adjunction F⊣G only holds relative to a specific agent state C. It also realizes the original vision of a "dynamic GNN that deforms according to coherence" — the GNN's weights are fixed, but its behavior changes because C changes.

## 4. Agent Layer: C

### 4.1. State Representation

The agent state is borrowed from **DreamerV3's RSSM (Recurrent State-Space Model)** [4], the most empirically validated world model architecture (tested on 150+ tasks).

| Component | Type | Role | Suspension Requirement |
|---|---|---|---|
| `h_t` | GRU hidden state (deterministic) | Compressed history of all past experiences | **Memory** |
| `z_t` | Categorical distribution (stochastic) | Current belief about the state of the world | Perception / Interpretation |
| Preferred observation prior | Fixed or learned distribution | What the agent "wants" to observe | **Intentionality** |

### 4.2. Update Rule

The update `C(t) → C(t+1)` follows three steps, borrowed from **Çatal et al. (2020)** [5] and DreamerV3:

**Step 1 — Memory Update**: `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})`. The GRU integrates the previous belief, the previous action, and the previous memory into a new memory state. This implements **temporal duration** — the model carries forward information across time steps.

**Step 2 — Belief Inference**: `z_t ~ q(z_t | h_t, o_t)`. Given the new memory and the current observation (which includes the coherence signal from the Adjoint Layer), the model infers a new belief about the world.

**Step 3 — Action Selection**: The agent evaluates candidate action sequences by imagining their consequences using the world model, and selects the sequence that minimizes **Expected Free Energy (EFE)**. EFE balances two drives: (a) epistemic value (reducing uncertainty about the world) and (b) pragmatic value (achieving preferred observations). This implements **intentionality**.

### 4.3. Self / Non-Self Distinction

This requirement is satisfied **structurally** without a dedicated mechanism. In the adjunction, Shape (the external world) does not depend on C, but Action (the agent's interpretation) does depend on C. This asymmetry means the model inherently distinguishes between "what is out there" (Shape, independent of the agent) and "what I can do with it" (Action, dependent on the agent). No additional implementation is needed.

## 5. Implementation Strategy

This section outlines the strategy for leveraging existing codebases and libraries to accelerate the prototype development, while focusing new development efforts on the core theoretical contributions of the Physical-Semantic Adjunction Model.

### 5.1. Core Libraries and Frameworks

-   **PyTorch**: The primary deep learning framework for model implementation.
-   **PyTorch Geometric**: Essential for implementing Graph Neural Networks (GNNs) for point cloud processing (F and G).
-   **`inferactively-pymdp`**: A Python library for Active Inference. While designed for discrete state spaces, its core inference loop structure and Free Energy minimization principles will be adapted for the continuous state space of Agent Layer (C).

### 5.2. Code Reuse and New Development Boundaries

We will adopt a strategy of **maximal reuse for established components** and **focused new development for theoretically novel parts**.

| Component | Reuse Strategy | Specific References / Codebases |
|---|---|---|
| **F (Shape → Action) GNN Encoder** | **High Reuse**: Leverage existing GNN architectures and implementations for point cloud feature extraction. | `PointNet++` or `DGCNN` implementations within PyTorch Geometric. Data loading and preprocessing from `3D AffordanceNet` [1] and `Contact-GraspNet` [2] repositories. |
| **G (Action → Shape) Conditional Decoder** | **Moderate Reuse**: Standard conditional VAE decoder architectures can be adapted. The specific input/output mapping will be new. | General conditional VAE decoder patterns. `Andries et al. (2020)` [3] for the conceptual approach of generating shapes from affordances. |
| **Agent Layer (C) State Representation** | **High Reuse**: Directly adopt the RSSM architecture. | `DreamerV3` [4] implementations (e.g., from official or community PyTorch ports). |
| **Agent Layer (C) Update Rule** | **High Reuse**: Adapt the three-step update rule. | `Çatal et al. (2020)` [5] for EFE action selection, and `DreamerV3` [4] for RSSM update mechanics. |
| **Loss Function Components** | **High Reuse**: Standard reconstruction losses, cross-entropy for classification, and variational free energy terms. | PyTorch's `nn.MSELoss`, `nn.CrossEntropyLoss`. Variational Free Energy terms as defined in `DreamerV3` [4] and `Çatal et al. (2020)` [5]. |
| **Integration of F, G, C (Adjunction Loop)** | **New Development**: The specific way F, G, and C interact to form the conditional adjunction and dynamic loop is novel. | This is the core theoretical contribution and will be implemented from scratch, guided by the `ARCHITECTURE.md` and `interim_design.md`. |
| **Coherence Signal Calculation** | **New Development**: While conceptually a reconstruction loss, its interpretation as the unit η and its role in the agent's observation `o_t` is specific to this model. | Implemented as `||Shape - G(F(Shape))||` within the main model integration. |

### 5.3. Development Environment

-   **Python 3.9+**
-   **`pip`** for package management.
-   **`conda` or `venv`** for environment isolation (recommended).

---

## 6. Experimental Design and Evaluation Metrics

This section outlines the experimental designs to validate the core theoretical claims of the Physical-Semantic Adjunction Model: solving the symbol grounding problem, generalization to unknown objects, and the emergence of creativity under constraint.

### 6.1. Core Theoretical Claims to Validate

1.  **Symbol Grounding**: The adjoint structure between Shape and Action constructs physical "meaning" without relying on linguistic symbols.
2.  **Context-Dependent Meaning**: This meaning changes contextually based on the agent's internal state (C).
3.  **Creativity from Coherence Breakdown**: Creative problem-solving emerges at moments of coherence breakdown.

### 6.2. Experimental Settings

#### Setting A: Generalization to Unknown Objects (Zero-Shot Affordance)

-   **Question**: Can the model infer functionality from physical shape for objects not seen during training?
-   **Theoretical Connection**: This tests `coherence breakdown`. Unknown objects should increase `distance(s, G(F(s)))`. If the adjoint structure truly extracts "functional skeletons," then cross-category generalization should occur through shared structural features (e.g., "has a handle," "has an opening"). This is an indirect approach to the symbol grounding problem.
-   **Experimental Design**:
    1.  **Training Phase**: Train F and G with simple shapes (cubes, cylinders, combinations) and basic actions (push, pull, lift) from everyday object category A (e.g., cups, bowls, bags).
    2.  **Test Phase**: Present entirely different category B objects (e.g., tools, medical instruments) or novel combinations of basic shapes not seen during training.
    3.  **Evaluation**: Assess if `F(unknown_shape)` outputs affordances consistent with human judgment. Metrics include precision/recall of predicted affordances, and qualitative analysis of `G(F(unknown_shape))` for functional relevance.
-   **Implementation Note**: Use a simulated environment (e.g., PyBullet) to simplify shape and action complexity.

#### Setting B: Creative Problem Solving Under Constraint

-   **Question**: Can changes in agent state C lead to the emergence of novel behaviors?
-   **Theoretical Connection**: This validates the core claim that the adjoint is parameterized by C, and `coherence breakdown` leads to creativity.
-   **Experimental Design**:
    1.  **Standard State `C_normal`**: Train the agent to perform a task (e.g., carrying a suitcase) in a simulated environment with a standard body state (e.g., both arms functional). Observe the chosen action (e.g., "carry with one hand").
    2.  **Constraint `C_injured`**: Introduce a constraint to agent state C (e.g., `right_arm: disabled`) and observe if an alternative action (e.g., "roll") is chosen for the same task.
    3.  **Complex Constraint `C_complex`**: Introduce further constraints (e.g., `right_arm: disabled, noise_constraint: high`) to induce `coherence breakdown`. Evaluate if novel combinations of existing action primitives (e.g., "cradle with both knees") emerge creatively.
-   **Implementation Note**: Constraints can be simulated by modifying C's representation (e.g., masking parts of the state vector) or physical parameters in the simulation.

#### Setting C: Symbol Grounding with Language

-   **Question**: Do the representations learned from the Shape-Action adjoint align with linguistic descriptions?
-   **Theoretical Connection**: This is a direct approach to the symbol grounding problem. If the adjoint truly captures "function," its representation space should structurally correspond to linguistic functional descriptions.
-   **Experimental Design**:
    1.  **Learning Phase**: Train F and G for bidirectional Shape-Action inference (without language input).
    2.  **Language Description**: Prepare functional descriptions generated by an LLM (e.g., "can pour liquid," "can carry heavy items").
    3.  **Alignment Evaluation**: Assess if the functional skeleton extracted by `G(F(shape))` (reconstructed shape features or affordance distribution) can be aligned with LLM descriptions. This could involve measuring cosine similarity between affordance distributions and language embeddings, or training a separate model to generate shapes from linguistic descriptions and comparing with G's output.
-   **Implementation Note**: Techniques from multimodal embedding models (e.g., CLIP) can be adapted for alignment.

---

## 7. Loss Function

| Loss | Formula | Purpose | Weight |
|---|---|---|---|
| Reconstruction (Coherence Signal) | `L_recon = \|\|Shape - G(F(Shape))\|\|` | Measures adjunction coherence; implements sensitivity to difference | `w1` |
| Affordance | `L_aff = CrossEntropy(F(Shape), label)` | Supervises F to produce meaningful affordance predictions | `w2` |
| Variational Free Energy | `L_vfe = KL[q(z\|h,o) \|\| p(z\|h)] - log p(o\|z)` | Drives belief inference and world model learning | `w3` |

**Total**: `L = w1 * L_recon + w2 * L_aff + w3 * L_vfe`

The weights `w1, w2, w3` are hyperparameters to be tuned. A reasonable starting point is `w1=1.0, w2=1.0, w3=0.1` (prioritizing the adjoint layer's learning in early training).

## 6. Data

**Dataset**: 3D AffordanceNet [1] — 23,000 3D shapes across 23 object categories, with per-point affordance annotations across 18 affordance types (e.g., grasp, sit, support, contain, wrap-grasp, open, lay, press, ...).

This dataset provides both the input to F (point clouds) and the supervision for F (affordance labels). The reconstruction target for G is the original point cloud.

## 7. File Structure (Planned)

```
adjunction-model/
├── README.md
├── ARCHITECTURE.md          # This file
├── TODO.md                  # Development status
├── docs/                    # Research notes and design documents
├── src/
│   ├── data/
│   │   └── affordance_loader.py    # Data loader for 3D AffordanceNet
│   ├── models/
│   │   ├── encoder_f.py            # F: PointNet++ encoder
│   │   ├── decoder_g.py            # G: Conditional decoder
│   │   ├── agent_c.py              # C: RSSM + EFE action selection
│   │   └── adjunction_model.py     # Full model integrating F, G, C
│   ├── losses/
│   │   └── composite_loss.py       # L_recon + L_aff + L_vfe
│   ├── train.py                    # Training loop
│   └── evaluate.py                 # Evaluation and coherence signal analysis
└── configs/
    └── default.yaml                # Hyperparameters
```

## 8. Key References

1. 3D AffordanceNet: https://github.com/Gorilla-Lab-SCUT/AffordanceNet
2. Contact-GraspNet (PyTorch): https://github.com/elchun/contact_graspnet_pytorch
3. Andries et al. (2020), "Automatic Generation of Object Shapes With Desired Affordances": https://doi.org/10.3389/fnbot.2020.00022
4. DreamerV3 (Hafner et al., 2023), "Mastering Diverse Domains through World Models": https://arxiv.org/abs/2301.04104
5. Çatal et al. (2020), "Learning Generative State Space Models for Active Inference": https://doi.org/10.3389/fncom.2020.574372
6. Ha & Schmidhuber (2018), "World Models": https://worldmodels.github.io/
7. Smithe (2024), "Structured Active Inference": https://arxiv.org/abs/2406.07577
