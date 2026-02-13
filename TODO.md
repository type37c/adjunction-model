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
- [x] **Implement Phase 2 Slack Trainer** - COMPLETED ✓ (2026-02-14)
  - [x] Train F⊣G and Agent C from scratch.
  - [x] Use Affordance Loss (`L_aff`) as the primary driver.
  - [x] **Remove Reconstruction Loss (`L_recon`)** to stop minimizing η.
  - [x] Add Coherence Regularization (`L_coherence`) to prevent η from collapsing to zero.
  - [x] Implement Counit `ε` calculation and tracking.
- [x] **Debug and Validate Forward Pass** - COMPLETED ✓ (2026-02-14)
  - [x] Fixed multiple tensor shape and dimension mismatches.
  - [x] Created `docs/tensor_shape_specification_2026_02_14.md` to prevent recurrence.
- [x] **Create Kaggle Notebook for GPU Experiment** - COMPLETED ✓ (2026-02-14)
  - [x] Fixed imports (ConditionalAdjunctionV4, SyntheticAffordanceDataset)
  - [x] Integrated working code with collate_fn
  - [x] Added visualization and analysis
  - [x] Pushed to GitHub: `kaggle/phase2_slack_gpu_experiment.ipynb`
- [ ] **Run Full-Scale Slack Experiment on Kaggle** (NEXT - BLOCKED)
  - **ISSUE**: `torch_geometric` import error when using "Run All"
  - **CAUSE**: Dependencies install in Cell 2, but imports happen in Cell 4
  - **WORKAROUND**: Run cells sequentially (1→2→3→4...) instead of "Run All"
  - **TODO**: Reorder cells to fix "Run All" compatibility
  - [ ] Fix cell order in notebook
  - [ ] Test "Run All" works correctly
  - [ ] Run 100 epochs on Kaggle GPU T4 x2
  - [ ] Download and analyze results

## Phase 3: Language Grounding - NOT STARTED

- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

## Current Issues & Next Steps

### HIGH PRIORITY

1. **Fix Kaggle Notebook Cell Order** (IMMEDIATE NEXT STEP)
   - **Problem**: "Run All" fails because imports happen before dependencies install
   - **Solution**: Reorder cells so installation happens before imports
   - **Steps**:
     - [ ] Move dependency installation (Cell 2) before imports
     - [ ] Test that "Run All" works without errors
     - [ ] Push updated notebook to GitHub
     - [ ] Verify on Kaggle

2. **Run Full-Scale Kaggle GPU Experiment** (AFTER FIX)
   - [ ] Ensure GPU T4 x2 is enabled in Kaggle settings
   - [ ] Run "Run All" (after cell order fix)
   - [ ] Wait ~30-40 minutes for 100 epochs
   - [ ] Download results:
     - `metrics.json` - Training metrics
     - `model_final.pt` - Trained model
     - `training_results.png` - Visualizations
   - [ ] Analyze η/ε behavior:
     - Does η remain non-zero (preserved as slack)?
     - Is ε observable and meaningful?
     - Do they show dynamic, non-monotonic changes?

3. **Analyze Slack Experiment Results**
   - [ ] Plot η and ε over time
   - [ ] Check for correlation/anti-correlation between η and ε
   - [ ] Look for signs of suspension structure emergence
   - [ ] Compare with theoretical predictions
   - [ ] Document findings in `docs/phase2_slack_results_2026_02.md`

### MEDIUM PRIORITY

4. **Redefine Curiosity Reward**
   - **Problem**: Current Curiosity (confidence-based) is structurally flawed because F/G are no longer pre-trained to minimize reconstruction error. The agent cannot improve its "understanding" (reduce η) in the same way.
   - **New Hypothesis**: Curiosity could be redefined as the motivation to explore states that lead to a **reduction in the sum of slack (η + ε)**. This would represent the agent actively trying to make its world model more coherent and less ambiguous, but only when it chooses to.
   - [ ] Design and implement Curiosity v6 based on this new hypothesis.

5. **Update `research_note_ja.md`**
   - [ ] Add a new section detailing the failure of the previous training paradigm and the introduction of the Phase 2 Slack model.
   - [ ] Explain the theoretical importance of preserving η and ε.
   - [ ] Document the results of the upcoming slack experiments.

### LOW PRIORITY

6. **Consider Alternative GPU Options**
   - **Current**: Kaggle (free, but UI-heavy, no CLI)
   - **Alternatives to explore**:
     - Google Colab (similar to Kaggle)
     - Paperspace Gradient (CLI available)
     - Vast.ai (CLI-friendly, pay-per-use)
     - RunPod (CLI-friendly, pay-per-use)
   - [ ] Research CLI-based GPU rental options
   - [ ] Test workflow on alternative platform

## Theoretical Questions to Investigate

1. **Does Preserving η/ε Lead to Suspension Structures?** (THE CORE QUESTION)
   - Will the agent learn to modulate its engagement with the world, sometimes acting to reduce slack (η+ε) and other times leveraging it?

2. **What is the emergent relationship between η and ε?**
   - Are they correlated? Anti-correlated? Does the agent learn to trade one for the other?

3. **What is the new role of Intrinsic Motivation?**
   - **Competence**: May still relate to successfully predicting affordances (`L_aff`).
   - **Novelty**: KL divergence of the RSSM remains relevant.
   - **Curiosity**: Needs redefinition (see above).

## Recent Progress (2026-02-14)

### Completed Today
- ✅ Identified and fixed Kaggle experiment failure (ModuleNotFoundError: 'src')
- ✅ Corrected imports in notebook (ConditionalAdjunctionV4, SyntheticAffordanceDataset)
- ✅ Integrated working training code with proper collate_fn
- ✅ Added comprehensive visualization and analysis cells
- ✅ Pushed corrected notebook to GitHub
- ✅ Created detailed Kaggle setup guide (`KAGGLE_SETUP_GUIDE.md`)

### Discovered Issues
- ⚠️ "Run All" fails due to cell execution order (install → import)
- ⚠️ Need to reorder cells for seamless execution

### Next Session Goals
1. Fix cell order in Kaggle notebook
2. Run full 100-epoch experiment on GPU
3. Analyze results and look for suspension structure emergence

---

**Last Updated**: 2026-02-14
**Status**: Kaggle notebook corrected and pushed. Ready for cell reordering and full experiment run.
