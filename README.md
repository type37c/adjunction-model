# Physical-Semantic Adjunction Model

**物理的意味的随伴モデル：身体性・意味理解・創造性の圏論的統合**

A novel AI architecture that uses **adjunctions** from category theory to construct bidirectional semantic relationships between Shape and Action, mediated by an agent's embodiment.

---

## Overview

This project proposes the **Physical-Semantic Adjunction Model**, which addresses the symbol grounding problem and the lack of embodiment in current AI systems. The core idea is that the relationship between an object's shape and the actions it affords can be formally described as a **parameterized adjunction** F<sub>C</sub> ⊣ G<sub>C</sub>, where C represents the agent's full internal state (purpose, memory, physical constraints).

### Key Concepts (Updated)

- **Adjunction F ⊣ G**: The core structure linking `Shape` and `Action`.
- **Coherence Signal η**: `distance(shape, G(F(shape)))`. A low signal indicates a stable understanding, while a high signal indicates a breakdown, triggering a mode shift.
- **Suspension Structure**: The substrate of intelligence, which is not designed directly but emerges from specific conditions. It operates in different modes (e.g., stable reference, creative exploration) based on the coherence signal.
- **Emergence over Design**: The principle that the suspension structure itself should not be hardcoded. Instead, we design the conditions for its emergence.

### Relationship to Active Inference

The model has a strong structural correspondence with Karl Friston's Active Inference (coherence signal ≈ free energy, F⊣G loop ≈ action-perception loop), but extends it with concepts that probabilistic frameworks cannot directly handle: the suspension structure, creativity from coherence breakdown, and the asymmetry of the adjunction.

---

## Repository Structure

```
adjunction-model/
├── README.md
├── requirements.txt
├── docs/
│   ├── QUICKSTART.md          # --> For new developers: Start here!
│   ├── DESIGN_NOTES.md        # --> Key implementation decisions (the "why")
│   ├── DEBUGGING_GUIDE.md     # --> Common bugs and how to fix them
│   ├── research_note_ja.md    # Core theoretical document (Japanese)
│   └── research_note_en.md    # Core theoretical document (English)
├── src/
│   ├── data/                  # Data loading and generation
│   ├── models/                # Core model components (F, G, Adjunction)
│   └── training/              # Training loops and logic
└── experiments/
    └── test_coherence_signal.py # MVP experiment script
```

## Documents

| Document | Description |
| :--- | :--- |
| [Research Note (日本語)](docs/research_note_ja.md) | 理論の全体像を整合的にまとめた研究ノート |
| [Research Note (English)](docs/research_note_en.md) | Full research note in English |


---

## Development Guidelines

All development, especially by AI agents, MUST follow the rules outlined in [AGENT_GUIDELINES.md](AGENT_GUIDELINES.md). This file contains the core principles, workflow, and best practices for ensuring consistent and high-quality development across multiple sessions.

---

## Status: Phase 0-1 Complete

**The project has moved from theoretical formulation to empirical validation.**

A Minimum Viable Prototype (MVP) has been successfully implemented and tested. The core hypothesis—that the `coherence_signal` increases for novel shapes—has been experimentally confirmed.

- **Current State**: The `F ⊣ G` adjunction structure is implemented and learns effectively.
- **Next Step**: Proceed to **Phase 2**, which involves implementing the **Agent Layer C** (based on DreamerV3's RSSM) to enable online adaptation and more complex behaviors based on the coherence signal.

## For New Developers & Agents

To get started, please follow the guides in the `docs` directory in this order:

1.  **[QUICKSTART.md](docs/QUICKSTART.md)**: Set up your environment and run the main experiment.
2.  **[DESIGN_NOTES.md](docs/DESIGN_NOTES.md)**: Understand the key design decisions behind the current implementation.
3.  **[DEBUGGING_GUIDE.md](docs/DEBUGGING_GUIDE.md)**: Familiarize yourself with common issues and their solutions.

These documents are designed to quickly bring any agent up to speed on the project's status, architecture, and development workflow.

## License

TBD
