# TODO: Adjunction Model Development

## Phase 1: Adjoint Layer (F⊣G) - COMPLETED ✓

- [x] Implement Functor F (Shape → Action)
- [x] Implement Functor G (Action → Shape)
- [x] Implement Coherence Signal (spatial)
- [x] Basic training loop for F⊣G
- [x] Synthetic dataset for MVP

## Phase 2: Agent Layer C - REVISED (NEW)

### v1-v4: Initial Implementations - COMPLETED ✓
- [x] RSSM, Priority Attention, Valence Memory, Intrinsic Motivation
- **NOTE**: These were built on a flawed training process. See `docs/phase2_slack_implementation_analysis_2026_02_14.md`.

### **Phase 2 Slack: η/ε Preservation Training (CURRENT FOCUS)**
- **GOAL**: Train F⊣G and Agent C simultaneously **without reconstruction loss** to preserve η and ε as "slack".
- [ ] **Implement Phase 2 Slack Trainer** - COMPLETED ✓ (2026-02-14)
  - [x] Train F⊣G and Agent C from scratch.
  - [x] Use Affordance Loss (`L_aff`) as the primary driver.
  - [x] **Remove Reconstruction Loss (`L_recon`)** to stop minimizing η.
  - [x] Add Coherence Regularization (`L_coherence`) to prevent η from collapsing to zero.
  - [x] Implement Counit `ε` calculation and tracking.
- [ ] **Debug and Validate Forward Pass** - COMPLETED ✓ (2026-02-14)
  - [x] Fixed multiple tensor shape and dimension mismatches.
  - [x] Created `docs/tensor_shape_specification_2026_02_14.md` to prevent recurrence.
- [ ] **Run Small-Scale Validation Experiment** (IN PROGRESS)
  - [ ] Run for 5-10 epochs to confirm training stability and metric collection.
  - [ ] Analyze η and ε behavior. Do they remain non-zero?
- [ ] **Run Full-Scale Slack Experiment** (NEXT)
  - [ ] Train for 100+ episodes.
  - [ ] Observe for emergent suspension structures.

## Phase 3: Language Grounding - NOT STARTED

- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

## Current Issues & Next Steps

### HIGH PRIORITY

1. **Complete Phase 2 Slack Validation** (IN PROGRESS)
   - [ ] Run the small-scale experiment (5 epochs, 20 shapes).
   - [ ] Analyze the results from `results/phase2_slack/`.
   - [ ] Confirm that η and ε are being preserved and have meaningful values.
   - [ ] Commit all related code and documentation changes.

2. **Full-Scale η/ε Preservation Experiment** (NEXT)
   - [ ] Run `phase2_slack_experiment` for 100-200 epochs.
   - [ ] **Primary Goal**: Observe if suspension structures emerge. Look for dynamic, non-monotonic changes in confidence, η, and ε.
   - [ ] Analyze the correlation between η, ε, and agent behavior.

### MEDIUM PRIORITY

3. **Redefine Curiosity Reward**
   - **Problem**: Current Curiosity (confidence-based) is structurally flawed because F/G are no longer pre-trained to minimize reconstruction error. The agent cannot improve its "understanding" (reduce η) in the same way.
   - **New Hypothesis**: Curiosity could be redefined as the motivation to explore states that lead to a **reduction in the sum of slack (η + ε)**. This would represent the agent actively trying to make its world model more coherent and less ambiguous, but only when it chooses to.
   - [ ] Design and implement Curiosity v6 based on this new hypothesis.

4. **Update `research_note_ja.md`**
   - [ ] Add a new section detailing the failure of the previous training paradigm and the introduction of the Phase 2 Slack model.
   - [ ] Explain the theoretical importance of preserving η and ε.
   - [ ] Document the results of the upcoming slack experiments.

## Theoretical Questions to Investigate

1. **Does Preserving η/ε Lead to Suspension Structures?** (THE CORE QUESTION)
   - Will the agent learn to modulate its engagement with the world, sometimes acting to reduce slack (η+ε) and other times leveraging it?

2. **What is the emergent relationship between η and ε?**
   - Are they correlated? Anti-correlated? Does the agent learn to trade one for the other?

3. **What is the new role of Intrinsic Motivation?**
   - **Competence**: May still relate to successfully predicting affordances (`L_aff`).
   - **Novelty**: KL divergence of the RSSM remains relevant.
   - **Curiosity**: Needs redefinition (see above).

---

**Last Updated**: 2026-02-14
**Status**: Refactoring complete. Validating new Phase 2 Slack training process.
