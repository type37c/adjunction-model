# Physical-Semantic Adjunction Model

**物理的意味的随伴モデル：身体性・意味理解・創造性の圏論的統合**

A novel AI architecture that uses **adjunctions** from category theory to construct bidirectional semantic relationships between Shape and Action, mediated by an agent's embodiment.

---

## Overview

This project proposes the **Physical-Semantic Adjunction Model**, which addresses the symbol grounding problem and the lack of embodiment in current AI systems. The core idea is that the relationship between an object's shape and the actions it affords can be formally described as a **parameterized adjunction** F<sub>C</sub> ⊣ G<sub>C</sub>, where C represents the agent's full internal state (purpose, memory, physical constraints).

### Key Concepts

- **Conditional Adjunction**: The adjunction between Shape and Action only holds when parameterized by the agent's context C
- **Coherence Signal**: An internal stability metric derived from the unit η of the adjunction, measuring `distance(shape, G(F(shape)))`
- **Suspension Structure**: The minimal substrate of intelligence, defined by five requirements (intentionality, sensitivity to difference, temporal persistence, self/non-self distinction, memory)
- **Two-Layer Architecture**: An Adjoint Layer (Shape ⇄ Action via dynamic GNN) coupled with an Agent Layer (C, containing purpose and memory)
- **Creativity as Diagram Repair**: Coherence breakdown triggers creative problem-solving, formalized as the repair of a commutative diagram

### Relationship to Active Inference

The model has a strong structural correspondence with Karl Friston's Active Inference (coherence signal ≈ free energy, F⊣G loop ≈ action-perception loop), but extends it with concepts that probabilistic frameworks cannot directly handle: the suspension structure, creativity from coherence breakdown, and the asymmetry of the adjunction.

---

## Repository Structure

```
adjunction-model/
├── README.md
├── docs/
│   ├── research_note_ja.md    # 研究ノート（日本語版）
│   ├── research_note_en.md    # Research Note (English)
│   ├── interim_design.md      # 中間設計文書（理論→実装の橋渡し）
│   └── research_notes.md      # リサーチノート（先行研究調査）
└── src/                       # (coming soon) prototype implementation
```

## Documents

| Document | Description |
| :--- | :--- |
| [Research Note (日本語)](docs/research_note_ja.md) | 理論の全体像を整合的にまとめた研究ノート |
| [Research Note (English)](docs/research_note_en.md) | Full research note in English |
| [Interim Design](docs/interim_design.md) | Technical specification bridging theory and implementation |
| [Research Notes](docs/research_notes.md) | Survey of prior work (Categorical Deep Learning, Structured Active Inference, Affordance Learning, etc.) |

---

## Development Guidelines

All development, especially by AI agents, MUST follow the rules outlined in [AGENT_GUIDELINES.md](AGENT_GUIDELINES.md). This file contains the core principles, workflow, and best practices for ensuring consistent and high-quality development across multiple sessions.

---

## Status

This project is in the **theoretical formulation** stage. The next step is a minimal viable experiment using PyTorch Geometric to verify that an adjoint structure can learn bidirectional inference between shape and action by minimizing reconstruction error.

## License

TBD
