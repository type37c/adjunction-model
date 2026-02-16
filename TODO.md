# TODO: Adjunction Model Development

## Experiment Directory Restructuring - COMPLETED ✓ (2026-02-16)

Following the 2026-02-16 discussion and dev_instructions.pdf, experiments have been reorganized to clearly separate **loss function-driven** and **intrinsic reward-driven** approaches.

**New Structure**:
- `experiments/slack_affordance_loss/` - Phase 2 Slack (Affordance Loss driven) ✓
- `experiments/intrinsic_reward_baseline/` - Pure intrinsic motivation (2/13 reproduction) ✓
- `experiments/intrinsic_reward_curiosity/` - Intrinsic + Curiosity v6 (planned)
- `experiments/temporal_suspension/` - Temporal suspension structure (planned)
- `experiments/purpose_emergent/` - Purpose-emergent assembly ✓

**Archived**:
- `phase2_valence_experiment/` → `docs/archive/2026-02-16/` (superseded by new approach)

---

## Current Priority: Intrinsic Reward Baseline Experiment

### **Intrinsic Reward Baseline - IMPLEMENTED ✓ (2026-02-16)**

**Goal**: Reproduce the 2/13 experiment in a clean codebase to verify that intrinsic motivation alone (without external objectives) can drive structure formation.

**Status**: ✅ Implementation complete, ready to run

**Implementation**:
- [x] Create experiment directory structure
- [x] Create `config.yaml` with 2/13 parameters
- [x] Implement `IntrinsicRewardTrainer` (F/G frozen, no Affordance Loss)
- [x] Create `run.py` (training script)
- [x] Create `analyze.py` (analysis and visualization)
- [x] Create `README.md` (experiment documentation)

**Next Steps**:
- [ ] Run experiment (100 epochs, ~2-3 hours on CPU)
- [ ] Analyze results and compare with 2/13
- [ ] Document findings
- [ ] Update theory documents based on results

**Expected Results** (from 2/13):
- Intrinsic reward: +1584%
- Valence growth: +15% (accelerating)
- Coherence: stable (~0.43)
- Competence contribution: 59.5%

---

## Phase 1: Adjoint Layer (F⊣G) - COMPLETED ✓

- [x] Implement Functor F (Shape → Action)
- [x] Implement Functor G (Action → Shape)
- [x] Implement Coherence Signal (spatial)
- [x] Basic training loop for F⊣G
- [x] Synthetic dataset for MVP

## Phase 2: Agent Layer C - REVISED

### **slack_affordance_loss/ Experiment - COMPLETED ✓ (2026-02-14)**

**Formerly**: Phase 2 Slack

**Goal**: Train F⊣G and Agent C simultaneously **without reconstruction loss** to preserve η and ε as "slack", using Affordance Loss as the driving force.

**Status**: ✅ Completed

**Key Findings**:
- η preserved: 0.502-0.520 (mean: 0.5153) ✓
- ε dynamic: 0.096-1.044 (mean: 0.8238) ✓
- Strong η-ε correlation: r = 0.835 (p < 0.001) ✓
- Affordance Loss reduced by 96.78% ✓
- **CONCLUSION**: Slack can be preserved with external objectives (Affordance Loss)

**Limitation**: This is an extension of existing AI paradigms (external objective-driven). The unique claim of this project is that **intrinsic motivation alone** can drive structure formation.

### **intrinsic_reward_baseline/ Experiment - IMPLEMENTED ✓ (2026-02-16)**

See "Current Priority" section above.

### **intrinsic_reward_curiosity/ Experiment - PLANNED**

**Goal**: Test Curiosity v6 (η-based) on top of the intrinsic reward baseline.

**Status**: 📋 Planned (after intrinsic_reward_baseline results)

**Design**:
- F/G frozen
- Intrinsic rewards: Competence + Novelty + Curiosity
- Curiosity v6 definition: `R_curiosity = η(t-1) - η(t)`
- Rationale: η-ε correlation (r=0.835) suggests η-only curiosity is sufficient

**Next Steps**:
- [ ] Implement after intrinsic_reward_baseline is validated
- [ ] Compare with baseline (Curiosity disabled)

## Phase 3: Language Grounding - NOT STARTED

- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

---

## Theoretical Questions

### 1. **Does Preserving η/ε Lead to Suspension Structures?** ✓ ANSWERED

**YES**: `slack_affordance_loss/` experiment successfully preserves η (0.50-0.52) and ε (0.10-1.04).

**NEXT**: Does intrinsic motivation alone (without Affordance Loss) also preserve slack?

### 2. **What is the emergent relationship between η and ε?** ✓ ANSWERED

**Strong positive correlation**: r = 0.835 (p < 0.001)

**Implication**: Controlling η also controls ε → η-only curiosity is sufficient (Curiosity v6)

### 3. **Can intrinsic motivation alone drive structure formation?** ⏳ BEING TESTED

**Hypothesis**: Internal発的報酬 (Competence + Novelty) alone can drive Agent C's learning without external objectives.

**Approach**: `intrinsic_reward_baseline/` experiment (2/13 reproduction)

**Key Question**: Can we reproduce 2/13 results (+1584% intrinsic reward, +15% valence growth)?

### 4. **Can purpose emerge without explicit targets?** ⏳ PLANNED

**Hypothesis**: Purpose (directional intent) emerges from slack alone.

**Approach**: `purpose_emergent/` experiment — agent rewarded for approaching ANY coherent shape

**Key Question**: Does agent spontaneously choose and commit to specific shapes?

**Status**: Implemented, pending extended experiments

---

## Recent Progress (2026-02-16)

### Completed
- ✅ **Experiment Directory Restructuring**
  - Separated loss function-driven vs intrinsic reward-driven experiments
  - Created clear naming and documentation structure
  - Archived obsolete experiments
- ✅ **Intrinsic Reward Baseline Implementation**
  - Implemented `IntrinsicRewardTrainer` (F/G frozen, no Affordance Loss)
  - Created complete experiment package (config, run, analyze, README)
  - Ready to reproduce 2/13 results
- ✅ **Documentation Updates**
  - Created `experiments/README.md` (experiment overview)
  - Created experiment templates in `_templates/`
  - Updated individual experiment READMEs

### Key Insights from 2026-02-16 Discussion
- **Core hypothesis**: Intrinsic motivation alone can drive structure formation (not just external objectives)
- **2/13 success factors**: Competence reward ("attending to breakdowns") was the primary driving force (59.5%)
- **Design principle**: Separate loss function-driven and intrinsic reward-driven experiments completely
- **Priority shift**: Focus on intrinsic reward experiments as the unique contribution of this project

### Next Session Goals
1. **Run intrinsic_reward_baseline experiment** (100 epochs, ~2-3 hours)
2. **Analyze and compare with 2/13 results**
3. **Document findings** and update theory documents
4. **Decide next steps** based on results:
   - If successful: Implement intrinsic_reward_curiosity
   - If unsuccessful: Debug and iterate

---

**Last Updated**: 2026-02-16  
**Status**: Experiment directory restructured. Intrinsic reward baseline implemented and ready to run.
