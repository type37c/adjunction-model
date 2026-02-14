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

### **Phase 2 Slack: η/ε Preservation Training - COMPLETED ✓ (2026-02-14)**
- **GOAL**: Train F⊣G and Agent C simultaneously **without reconstruction loss** to preserve η and ε as "slack".
- [x] **Implement Phase 2 Slack Trainer** - COMPLETED ✓
  - [x] Train F⊣G and Agent C from scratch.
  - [x] Use Affordance Loss (`L_aff`) as the primary driver.
  - [x] **Remove Reconstruction Loss (`L_recon`)** to stop minimizing η.
  - [x] Add Coherence Regularization (`L_coherence`) to prevent η from collapsing to zero.
  - [x] Implement Counit `ε` calculation and tracking.
- [x] **Debug and Validate Forward Pass** - COMPLETED ✓
  - [x] Fixed multiple tensor shape and dimension mismatches.
  - [x] Created `docs/tensor_shape_specification_2026_02_14.md` to prevent recurrence.
- [x] **Run Full-Scale Slack Experiment (Local CPU)** - COMPLETED ✓ (2026-02-14)
  - [x] 50 epochs, 100 shapes, batch size 8
  - [x] Results saved to `experiments/phase2_slack_results/`
  - [x] Detailed analysis completed
  - [x] Visualization generated
- [x] **Analyze Slack Experiment Results** - COMPLETED ✓ (2026-02-14)
  - [x] η preserved: 0.502-0.520 (mean: 0.5153) ✓
  - [x] ε dynamic: 0.096-1.044 (mean: 0.8238) ✓
  - [x] Strong η-ε correlation: r = 0.835 (p < 0.001) ✓
  - [x] Affordance Loss reduced by 96.78% ✓
  - [x] **CONCLUSION**: Phase 2 Slack successfully preserves suspension structure

### **Curiosity v6: η-based Intrinsic Motivation - DESIGNED ✓ (2026-02-14)**
- **GOAL**: Redefine curiosity based on experimental results
- [x] **Design Curiosity v6** - COMPLETED ✓
  - [x] Definition: `R_curiosity = η(t-1) - η(t)` (η reduction)
  - [x] Rationale: η-ε strong correlation (r=0.835) → η-only is sufficient
  - [x] Architecture: AgentLayerC_v6 with simplified intrinsic reward
  - [x] Document: `docs/docs/docs/CURIOSITY_V6_DESIGN.md`
- [ ] **Implement Curiosity v6** (NEXT STEP)
  - [ ] Create `src/models/agent_layer_v6.py`
  - [ ] Create `src/models/conditional_adjunction_v6.py`
  - [ ] Create `src/training/train_phase2_curiosity_v6.py`
  - [ ] Run 50-epoch experiment
  - [ ] Compare with Phase 2 Slack (no curiosity baseline)

### **Temporal Suspension Theory - DOCUMENTED ✓ (2026-02-14)**
- **GOAL**: Extend static suspension to temporal/sequential contexts
- [x] **Document Temporal Suspension Theory** - COMPLETED ✓
  - [x] Map 4 requirements (Deferral, Difference, Trace, Iterability) to time evolution
  - [x] Identify limitation: Current experiment covers only Level 0 (static suspension)
  - [x] Propose next experiment: Sequence-based Suspension (Level 1)
  - [x] Document: `docs/docs/docs/01_temporal_suspension.md`

### **Temporal Suspension Experiment (Active Assembly) - IMPLEMENTED ✓ (2026-02-14)**
- **GOAL**: Validate temporal suspension (Level 1) via active point-cloud assembly
- **DESIGN**: Agent produces per-point displacement vectors to assemble target shapes.
  Action and understanding are inseparable — the agent must *discover* the target shape
  from hints and progressively revealed points, then commit to large displacements.
- [x] **Implement Active Assembly Dataset** - COMPLETED ✓
  - [x] `src/data/temporal_dataset.py`: Scattered initial points + target shapes
  - [x] Progressive revelation schedule (T steps, ambiguity 0.90 → 0.00)
  - [x] Hint points near target surface for implicit shape discovery
  - [x] Collate function for variable-length temporal batches
- [x] **Implement Temporal Suspension Experiment** - COMPLETED ✓
  - [x] `experiments/temporal_suspension_experiment.py`: Main experiment script
  - [x] DisplacementHead: context + per-point affordances → Δx ∈ R³
  - [x] chamfer_distance_graph: Differentiable, batch-aware Chamfer Distance
  - [x] TemporalSuspensionTrainer: Slack vs Tight mode comparison
  - [x] Fixed per-point tensor size mismatch across temporal steps
- [x] **Implement Analysis Script** - COMPLETED ✓
  - [x] `experiments/analyze_temporal_suspension.py`: ‖Δx(t)‖ dynamics, CD(t),
        η(t)/ε(t) evolution, training curves, comprehensive 8-panel figure
- [x] **Integration Tests** - COMPLETED ✓
  - [x] `tests/test_temporal_suspension.py`: 7 tests, all passing
- [ ] **Run Experiment** (NEXT: pending design review)
  - [ ] Run 50-epoch experiment (slack mode)
  - [ ] Run 50-epoch experiment (tight mode)
  - [ ] Analyze results: ‖Δx(t)‖ trajectory, CD(t), η(t)
  - [ ] Compare slack vs tight: Does slack model show small→large displacement pattern?
  - [ ] Document findings

## Phase 3: Language Grounding - NOT STARTED

- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

## Current Issues & Next Steps

### HIGH PRIORITY

1. **Implement Curiosity v6** (IMMEDIATE NEXT STEP)
   - **Goal**: Test η-based curiosity in Phase 2 Slack framework
   - **Steps**:
     - [ ] Create `agent_layer_v6.py` (η-only intrinsic reward)
     - [ ] Create `conditional_adjunction_v6.py` (use v6 agent)
     - [ ] Create `train_phase2_curiosity_v6.py` (training script)
     - [ ] Run 50-epoch experiment (CPU, 2-3 hours)
     - [ ] Analyze: Does valence correlate with η reduction?
     - [ ] Compare with Phase 2 Slack baseline
   - **Expected**: Curiosity-driven learning improves over baseline

2. **Run Temporal Suspension Experiment — Active Assembly** (AFTER DESIGN REVIEW)
   - **Goal**: Validate temporal suspension via active point-cloud assembly
   - **Status**: Implementation complete (v2: active assembly), awaiting design review
   - **Steps**:
     - [ ] Review experiment design (displacement head, Chamfer Distance, loss)
     - [ ] Run: `python experiments/temporal_suspension_experiment.py`
     - [ ] Analyze: `python experiments/analyze_temporal_suspension.py`
     - [ ] Compare slack vs tight displacement patterns
   - **Expected**: Slack model shows small→large ‖Δx‖ pattern, lower final CD

### MEDIUM PRIORITY

3. **Update `research_note_ja.md`**
   - [ ] Add Phase 2 Slack results section
   - [ ] Document η-ε correlation (r=0.835)
   - [ ] Explain Curiosity v6 design rationale
   - [ ] Add temporal suspension theory
   - [ ] Include "one character" insight

4. **Kaggle Notebook Cleanup** (OPTIONAL)
   - **Status**: Local CPU experiment succeeded, Kaggle not critical
   - [ ] Fix cell order for "Run All" compatibility (if needed in future)
   - [ ] Test GPU execution (if GPU experiments become necessary)

### LOW PRIORITY

5. **Consider Alternative GPU Options**
   - **Current**: Local CPU is working well for 50-epoch experiments
   - **Future**: If 100+ epochs or larger datasets are needed
   - [ ] Research CLI-based GPU rental options (Vast.ai, RunPod)
   - [ ] Test workflow on alternative platform

## Theoretical Questions to Investigate

1. **Does Preserving η/ε Lead to Suspension Structures?** ✓ ANSWERED
   - **YES**: Phase 2 Slack successfully preserves η (0.50-0.52) and ε (0.10-1.04)
   - **NEXT**: Does curiosity-driven learning enhance suspension structure?

2. **What is the emergent relationship between η and ε?** ✓ ANSWERED
   - **Strong positive correlation**: r = 0.835 (p < 0.001)
   - **Implication**: Controlling η also controls ε
   - **Curiosity design**: η-only is sufficient (Curiosity v6)

3. **What is the new role of Intrinsic Motivation?** ✓ PARTIALLY ANSWERED
   - **Curiosity**: Redefined as η reduction (`R_curiosity = η(t-1) - η(t)`)
   - **Competence**: Affordance prediction accuracy (unchanged)
   - **Novelty**: KL divergence of RSSM (unchanged)
   - **NEXT**: Validate Curiosity v6 in experiments

4. **Does temporal suspension require sequences?** ✓ BEING TESTED
   - **Insight**: "One character doesn't trigger action" → context is needed
   - **Hypothesis**: Static suspension (Level 0) ≠ Temporal suspension (Level 1)
   - **Approach**: Active assembly — agent must discover target shape from hints
   - **NEXT**: Run experiment and compare slack vs tight displacement patterns

## Recent Progress (2026-02-14)

### Completed Today
- ✅ **Phase 2 Slack Experiment Completed** (50 epochs, local CPU)
  - η preserved: 0.502-0.520 (does not collapse to 0)
  - ε dynamic: 0.096-1.044 (observable and meaningful)
  - η-ε correlation: r = 0.835 (strong positive)
  - Affordance Loss: 96.78% reduction (learning successful)
- ✅ **Detailed Analysis and Visualization**
  - Created `detailed_analysis.png` with 8 subplots
  - Generated `analysis_report.txt` with statistics
  - Saved `correlation_analysis.json` for future reference
  - Documented `KEY_FINDINGS.md` with theoretical implications
- ✅ **Curiosity v6 Design Completed**
  - Definition: η-based curiosity (`R_curiosity = η(t-1) - η(t)`)
  - Rationale: η-ε correlation → η-only sufficient
  - Architecture: AgentLayerC_v6 with simplified intrinsic reward
  - Document: `CURIOSITY_V6_DESIGN.md`
- ✅ **Temporal Suspension Theory Documented**
  - Mapped 4 requirements to time evolution
  - Identified static vs. temporal suspension distinction
  - Proposed sequence-based experiment (Level 1)
  - Document: `01_temporal_suspension.md`
- ✅ **Committed to GitHub**
  - Experimental results: `experiments/phase2_slack_results/`
  - Theory documents: `docs/docs/docs/`
  - Commit: f0f8f1e
- ✅ **Temporal Suspension Experiment v2 (Active Assembly) Implemented**
  - Redesigned from passive classification to active point-cloud assembly
  - DisplacementHead replaces ConfidenceGate + ShapeClassifier
  - Chamfer Distance as primary objective
  - 7 integration tests, all passing

### Key Insights
- **"One character doesn't trigger action"**: Suspension requires context/sequences
- **Static ≠ Temporal**: Current experiment is Level 0 (static), need Level 1 (temporal)
- **η-ε coupling**: Strong correlation (r=0.835) simplifies curiosity design

### Next Session Goals
1. Review temporal suspension experiment design (active assembly v2)
2. Run temporal suspension experiment (slack + tight, ~1-2 hours CPU)
3. Analyze results: ‖Δx(t)‖ pattern, CD convergence, η(t) trajectory
4. Compare slack vs tight displacement dynamics
5. Implement Curiosity v6 (if time permits)

---

**Last Updated**: 2026-02-14  
**Status**: Phase 2 Slack completed. Temporal suspension experiment v2 (active assembly) implemented (pending execution). Curiosity v6 designed.
