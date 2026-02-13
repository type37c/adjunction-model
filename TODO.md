# TODO: Adjunction Model Development

## Phase 1: Adjoint Layer (F⊣G) - COMPLETED ✓

- [x] Implement Functor F (Shape → Action)
- [x] Implement Functor G (Action → Shape)
- [x] Implement Coherence Signal (spatial)
- [x] Basic training loop for F⊣G
- [x] Synthetic dataset for MVP

## Phase 2: Agent Layer C - COMPLETED ✓

### v1: Basic RSSM - COMPLETED ✓
- [x] RSSM (Recurrent State Space Model) implementation
- [x] Conditional Adjunction (F_C, G_C with FiLM)
- [x] Basic integration test

### v2: Priority-based Attention - COMPLETED ✓
- [x] Spatial coherence signal computation
- [x] Priority = coherence × uncertainty
- [x] Attention mechanism based on priority
- [x] FiLM modulation with context from Agent C

### v3: Purpose Space P (Valence Memory) - COMPLETED ✓
- [x] Valence Memory implementation
- [x] Priority = coherence × uncertainty × valence
- [x] Valence updated by coherence changes

### v4: Intrinsic Motivation - COMPLETED ✓ (NEW)
- [x] Intrinsic reward computation (curiosity + competence + novelty)
- [x] Valence Memory v2 (intrinsic reward-based)
- [x] Agent Layer C v4 with intrinsic motivation
- [x] Value function for estimating future rewards
- [x] TD learning for value function
- [x] Value-based autonomous training loop
- [x] Conditional Adjunction Model v4
- [x] Validation tests

## Phase 3: Language Grounding - NOT STARTED

### 3.1: Purpose Space P Formalization - COMPLETED ✓ (REVISED)
- [x] Formalize Purpose Space P as intrinsic motivation
- [x] Define relationship: body adjunction → P → language
- [x] Design P as emergent from Agent C's internal state

### 3.2: Language Integration - NOT STARTED
- [ ] Design language embedding space
- [ ] Implement language → P mapping
- [ ] Implement P → language mapping
- [ ] Test language grounding with simple phrases

## Current Issues & Next Steps

### HIGH PRIORITY

1. **Full-scale training experiments** (NEW)
   - Run value-based training on larger dataset
   - Analyze Agent C's emergent behavior
   - Visualize value function and intrinsic rewards over time
   - Compare with supervised training baseline

2. **Update research_note_ja.md** (NEW)
   - Section 3.5: Purpose Space P (revised with intrinsic motivation)
   - Section 5.5: Language grounding (revised with P as intermediary)

3. **Create discussion log entry** (NEW)
   - Document the development of intrinsic motivation
   - Explain the shift from coherence-based to reward-based valence

### MEDIUM PRIORITY

4. **Suspension structure analysis** (REVISED)
   - Investigate whether suspension structure emerges with intrinsic motivation
   - Analyze Agent C's behavior when facing breakdowns
   - Measure: Does Agent C engage with breakdowns or avoid them?

5. **Visualization tools** (NEW)
   - Plot intrinsic rewards over time
   - Visualize value function landscape
   - Show Agent C's state trajectory in latent space

### LOW PRIORITY

6. **Code cleanup and documentation**
   - Add docstrings to new modules
   - Create usage examples
   - Update README with new components

7. **Performance optimization**
   - Profile training loop
   - Optimize value function updates
   - Consider parallel episode processing

## Theoretical Questions to Investigate

1. **Intrinsic motivation vs. suspension structure**
   - Does intrinsic motivation lead to suspension structure emergence?
   - Or does it prevent it by making Agent C "too purposeful"?

2. **Value function convergence**
   - Under what conditions does V(state) converge?
   - What is the optimal balance between value learning and Agent C learning?

3. **F/G adaptation**
   - Should F/G be completely frozen, or allow slow adaptation?
   - What is the effect of different F/G learning rates?

4. **Language grounding with P**
   - How to integrate language with intrinsic motivation?
   - Should language provide additional rewards, or just be grounded in P?

## Experimental Roadmap

### Experiment 1: Baseline Comparison (IMMEDIATE)
- Compare value-based training vs. supervised training
- Metrics: intrinsic reward, coherence, reconstruction error
- Hypothesis: Value-based prevents coherence collapse

### Experiment 2: Intrinsic Reward Ablation (SHORT-TERM)
- Test with only curiosity, only competence, only novelty
- Find optimal α, β, γ weights
- Hypothesis: Competence (β) is most important for suspension structure

### Experiment 3: Value Function Analysis (SHORT-TERM)
- Visualize value landscape in state space
- Identify high-value and low-value regions
- Hypothesis: High-value states correspond to "productive breakdowns"

### Experiment 4: Long-term Behavior (MEDIUM-TERM)
- Train for many episodes (100+)
- Observe emergent patterns in Agent C's behavior
- Hypothesis: Agent C develops "strategies" for different shape types

### Experiment 5: Real-world Shapes (LONG-TERM)
- Test on 3D AffordanceNet dataset
- Evaluate generalization to unseen shapes
- Hypothesis: Intrinsic motivation improves generalization

## Notes

### Development Summary (2026-02-13)

Major milestone achieved: **Purpose Space P with Intrinsic Motivation**

Key accomplishments:
1. Identified critical issue: Agent C needs purpose to preserve itself
2. Implemented intrinsic rewards (curiosity, competence, novelty)
3. Created value function and TD learning
4. Developed value-based autonomous training
5. All validation tests passed

This represents a fundamental shift from supervised learning to autonomous learning driven by intrinsic motivation.

See `docs/development_summary_2026_02_13.md` for full details.

---

**Last Updated**: 2026-02-13
**Status**: Phase 2 Complete (v4), Phase 3 Ready to Start
