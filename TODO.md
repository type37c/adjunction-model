# TODO List

This file tracks the development status and next steps for the `adjunction-model` project.

## Phase 1: Prototype Implementation

### Core Components
- [ ] **Environment Setup**: Install PyTorch, PyTorch Geometric, and `inferactively-pymdp`.
- [ ] **Data Loader**: Implement a data loader for the `3D AffordanceNet` dataset, including point cloud preprocessing.
- [ ] **Adjoint Layer (F)**: Implement the PointNet++ based encoder for the `Shape -> Action` mapping.
- [ ] **Adjoint Layer (G)**: Implement the conditional decoder for the `Action -> Shape` mapping.
- [ ] **Agent Layer (C)**: Implement the RSSM structure (`h` and `z` states) and the GRU-based update rule.
- [ ] **Action Selection**: Implement the Expected Free Energy (EFE) calculation for policy selection, based on Çatal et al. (2020).
- [ ] **Loss Function**: Implement the composite loss function (`L_recon`, `L_affordance`, `L_vfe`).
- [ ] **Training Loop**: Integrate all components into a single, end-to-end training script.

### Undecided / Needs Further Discussion

- [ ] **Suspension Structure Verification**: How do we experimentally verify that the integrated model exhibits the properties of a "suspension structure"?
    - How to measure the presence of the 5 requirements (sensitivity, intentionality, duration, self/non-self, memory) in the running model?
    - How to design an experiment to test the hypothesis that `coherence breakdown` leads to creative problem-solving?
    - *Decision*: This is an **experimental design problem**, to be addressed **after** the prototype is functional.

## Phase 2: Experimentation & Analysis

- [ ] **Minimal Verification Task**: Run the prototype on the `3D AffordanceNet` dataset and confirm that it learns.
- [ ] **Coherence Signal Analysis**: Plot the `L_recon` (coherence signal) over time. Does it correlate with task difficulty or novelty?
- [ ] **Ablation Studies**: Systematically remove components (e.g., the EFE part of the loss) to verify that each part contributes as expected.
- [ ] **Suspension Structure Experiments**: Design and run experiments to test the hypotheses outlined in the "Undecided" section above.

## Phase 3: Documentation & Publication

- [ ] **Refine Research Notes**: Update the `research_note_ja.md` and `research_note_en.md` with implementation details and experimental results.
- [ ] **Prepare Manuscript**: Draft a paper for submission to a relevant conference (e.g., NeurIPS, ICLR, ALife) or journal.
