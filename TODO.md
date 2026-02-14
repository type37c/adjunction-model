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

### **Purpose-Emergent Active Assembly - IMPLEMENTED ✓ (2026-02-14)**
- **GOAL**: Test whether purpose (directional intent) emerges from slack without explicit target assignment
- **DESIGN**: Agent assembles scattered points toward reference shapes (sphere, cube, cylinder) **without being told which to target**. Purpose loss is `min_shape CD(assembled, reference)`.
- [x] **Implement Purposeless Assembly Dataset** - COMPLETED ✓
  - [x] `src/data/purposeless_dataset.py`: Scattered initial points, no target assignment
  - [x] Reference shapes (sphere, cube, cylinder) pre-computed and shared
  - [x] Collate function for batch processing
- [x] **Implement Purpose-Emergent Experiment** - COMPLETED ✓
  - [x] `experiments/purpose_emergent_experiment.py`: Main experiment script
  - [x] DisplacementHead: context + per-point affordances → Δx ∈ R³
  - [x] Purpose loss: `min_shape CD(assembled, ref_shape)`
  - [x] PurposeEmergentTrainer: Baseline vs purpose-emergent conditions
- [x] **Implement Analysis Script** - COMPLETED ✓
  - [x] `experiments/analyze_purpose_emergent.py`: Purpose loss trajectories, shape selection analysis
- [x] **Integration Tests** - COMPLETED ✓
  - [x] `tests/test_purpose_emergent.py`: Dataset and model tests
- [ ] **Run Full Experiment** (NEXT PRIORITY)
  - [ ] Run extended training (100+ epochs)
  - [ ] Analyze results: Does agent spontaneously choose target shapes?
  - [ ] Track purpose loss convergence
  - [ ] Document findings

## Phase 3: Language Grounding - NOT STARTED

- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

## Current Issues & Next Steps

### HIGH PRIORITY

1. **Analyze Purpose-Emergent Experiment Results** (IMMEDIATE NEXT STEP)
   - **Goal**: Understand whether purpose emerges from slack without target assignment
   - **Steps**:
     - [ ] Run extended training (100+ epochs) for both conditions
     - [ ] Analyze purpose loss convergence patterns
     - [ ] Track which reference shapes agent converges toward
     - [ ] Compare purpose-emergent vs baseline conditions
     - [ ] Measure shape selection consistency across episodes
   - **Expected**: Agent spontaneously chooses and commits to specific shapes

2. **Scale Purpose-Emergent Experiments**
   - **Goal**: Test robustness and generalization
   - **Steps**:
     - [ ] Increase number of reference shapes (add torus, pyramid, etc.)
     - [ ] Test with larger point clouds (512, 1024 points)
     - [ ] Vary slack preservation parameters (λ_coherence)
     - [ ] Document scaling behavior
   - **Expected**: Purpose emergence scales with model capacity

### MEDIUM PRIORITY

3. **Implement Curiosity v6** (RESEARCH PRIORITY)
   - **Goal**: Test η-based curiosity in Phase 2 Slack framework
   - **Steps**:
     - [ ] Create `agent_layer_v6.py` (η-only intrinsic reward)
     - [ ] Create `conditional_adjunction_v6.py` (use v6 agent)
     - [ ] Create `train_phase2_curiosity_v6.py` (training script)
     - [ ] Run 50-epoch experiment (CPU, 2-3 hours)
     - [ ] Analyze: Does valence correlate with η reduction?
     - [ ] Compare with Phase 2 Slack baseline
   - **Expected**: Curiosity-driven learning improves over baseline

4. **Update `research_note_ja.md`**
   - [ ] Add Phase 2 Slack results section
   - [ ] Document η-ε correlation (r=0.835)
   - [ ] Add Purpose-Emergent experiment design
   - [ ] Explain emergent goal formation hypothesis
   - [ ] Include key insights on slack and purpose

### LOW PRIORITY

5. **Kaggle Notebook Cleanup** (OPTIONAL)
   - **Status**: Local CPU experiment succeeded, Kaggle not critical
   - [ ] Fix cell order for "Run All" compatibility (if needed in future)
   - [ ] Test GPU execution (if GPU experiments become necessary)

6. **Consider Alternative GPU Options**
   - **Current**: Local CPU is working well for current experiments
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

4. **Can purpose emerge without explicit targets?** ⏳ BEING TESTED
   - **Hypothesis**: Purpose (directional intent) emerges from slack alone
   - **Approach**: Purpose-Emergent Active Assembly — agent rewarded for approaching ANY coherent shape
   - **Key Question**: Does agent spontaneously choose and commit to specific shapes?
   - **NEXT**: Run extended experiments and analyze shape selection patterns

## Recent Progress (2026-02-14)

### Completed
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
- ✅ **Purpose-Emergent Active Assembly Implemented**
  - Dataset: Scattered points, no target assignment
  - Purpose loss: `min_shape CD(assembled, reference)`
  - Tests: Integration tests passing
  - Status: Ready for extended experiments

### Key Insights
- **η-ε coupling**: Strong correlation (r=0.835) simplifies curiosity design
- **Slack preservation**: Removing reconstruction loss allows slack to grow while task performance improves
- **Purpose hypothesis**: Testing whether goal-directed behavior emerges without explicit targets

### Next Session Goals
1. Run extended Purpose-Emergent experiments (100+ epochs)
2. Analyze shape selection patterns and convergence
3. Compare purpose-emergent vs baseline conditions
4. Document findings on emergent goal formation
5. Implement Curiosity v6 (if time permits)

---

**Last Updated**: 2026-02-14  
**Status**: Phase 2 Slack completed. Purpose-Emergent Active Assembly implemented (pending extended experiments). Curiosity v6 designed.
