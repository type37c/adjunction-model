# Architecture Design

This document describes the concrete architecture for the prototype implementation of the Physical-Semantic Adjunction Model. It is written so that a developer with no prior context can understand the design intent and begin implementation by reading this file alone.

## 1. Theoretical Background

This model addresses the **symbol grounding problem** in AI: current systems manipulate symbols (words, labels) without understanding their physical meaning. The core idea is that **meaning arises from the structural correspondence between physical shapes and the actions they afford**. A cup's shape means "drinkable" not because of a label, but because its geometry enables the action of drinking.

This correspondence is formalized using **adjunction** from category theory — a pair of structure-preserving maps F and G between two domains (Shape and Action) that are not inverses of each other, but are "optimally related" in a precise mathematical sense. The adjunction is **conditioned on an agent's internal state C** (goals, memory, physical constraints), meaning the same shape affords different actions depending on who is interacting with it.

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

## 5. Loss Function

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
