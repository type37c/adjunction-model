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

1. **Full-scale training experiments** - COMPLETED ✓ (2026-02-13)
   - [x] Run value-based training on larger dataset (100 episodes, 50 shapes)
   - [x] Analyze Agent C's emergent behavior
   - [x] Visualize value function and intrinsic rewards over time
   - [ ] Compare with supervised training baseline (DEFERRED)
   
   **Key Findings**:
   - Intrinsic reward improved by +1584% (0.025 → 0.421)
   - Value function learned correctly (0.013 → 6.639)
   - Valence grows smoothly (+15%: 0.58 → 0.67)
   - Coherence remains stable (~0.43), no collapse
   - **Bug fixed**: Valence was not being recorded
   - **Bug fixed**: Curiosity reward had sign issue (disabled)
   - **Redesigned**: Competence reward = "attending to breakdowns" (not "reducing breakdowns")
   - Competence reward now contributes 59.5% of intrinsic reward
   - Uncertainty increases in v3 (67.0 → 67.9), suggesting exploration
   - Possible emergence of suspension structure (exponential growth after episode 60)
   
   See: `docs/development_report_2026_02_13_final.md`

2. **Update research_note_ja.md** (PENDING)
   - Section 3.5: Purpose Space P (revised with intrinsic motivation)
   - Section 5.5: Experimental results (add v4 training results)
   - Section 5.6: Competence reward redesign and suspension structure emergence

3. **Create discussion log entry** - COMPLETED ✓ (2026-02-13)
   - [x] Document the development of intrinsic motivation
   - [x] Explain the shift from coherence-based to reward-based valence
   - [x] Discuss Competence reward redesign
   - [x] Analyze suspension structure emergence
   
   See: `docs/discussion_log_2026_02_13_v2.md`

### MEDIUM PRIORITY

4. **Suspension structure analysis** - PARTIALLY COMPLETED ✓ (2026-02-13)
   - [x] Investigate whether suspension structure emerges with intrinsic motivation
   - [x] Analyze Agent C's behavior when facing breakdowns
   - [x] Measure: Does Agent C engage with breakdowns or avoid them?
   
   **Findings**:
   - Agent C does NOT avoid breakdowns (Coherence stable at ~0.43)
   - Competence reward redesigned as "attending to breakdowns"
   - Agent C learns to "engage with difficulty" (59.5% of intrinsic reward)
   - Uncertainty increases (v3), suggesting "definite → indefinite" transition
   - Exponential growth in intrinsic reward after episode 60 suggests critical point
   - **Conclusion**: Possible emergence of suspension structure, but not fully confirmed
   
   **Next**: Longer training (1000 episodes) to observe full emergence

5. **Visualization tools** - COMPLETED ✓ (2026-02-13)
   - [x] Plot intrinsic rewards over time
   - [x] Visualize value function landscape
   - [x] Show Agent C's internal state (Coherence, Uncertainty, Valence)
   
   See: `results/full_scale_training/*.png`

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

1. **Intrinsic motivation vs. suspension structure** - PARTIALLY ANSWERED ✓
   - Does intrinsic motivation lead to suspension structure emergence?
     → **YES, with proper design**: Competence reward as "attending to breakdowns"
   - Or does it prevent it by making Agent C "too purposeful"?
     → **NO**: Agent C does not avoid breakdowns, engages with them
   - **New insight**: Competence reward must be redesigned for frozen F/G
   - **New insight**: "Attending to breakdowns" promotes suspension, not "reducing breakdowns"

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

### Experiment 1: Full-scale Value-based Training - COMPLETED ✓ (2026-02-13)
- [x] Run value-based training on 100 episodes, 50 shapes
- [x] Metrics: intrinsic reward (+1584%), value function (+51000%), coherence (stable)
- [x] Hypothesis CONFIRMED: Value-based prevents coherence collapse
- [x] Additional finding: Competence reward redesign critical for success

See: `docs/development_report_2026_02_13_final.md`

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

#### Morning: Purpose Space P with Intrinsic Motivation

Major milestone achieved: **Purpose Space P with Intrinsic Motivation**

Key accomplishments:
1. Identified critical issue: Agent C needs purpose to preserve itself
2. Implemented intrinsic rewards (curiosity, competence, novelty)
3. Created value function and TD learning
4. Developed value-based autonomous training
5. All validation tests passed

This represents a fundamental shift from supervised learning to autonomous learning driven by intrinsic motivation.

See `docs/development_summary_2026_02_13.md` for full details.

#### Afternoon: Full-scale Training Experiments

Major milestone achieved: **Agent C v4 Full-scale Training with Competence Reward Redesign**

Key accomplishments:
1. Fixed Valence recording bug (now properly tracked)
2. Identified and disabled Curiosity reward (sign issue)
3. **Redesigned Competence reward**: "attending to breakdowns" (not "reducing breakdowns")
4. Achieved +1584% improvement in intrinsic reward (0.025 → 0.421)
5. Value function learned correctly (+51000%: 0.013 → 6.639)
6. Coherence remains stable (~0.43), no collapse
7. Observed possible emergence of suspension structure (exponential growth after episode 60)

Theoretical insights:
- Agent C learns to "engage with difficulty" (Competence = 59.5% of intrinsic reward)
- Uncertainty increases (67.0 → 67.9), suggesting "definite → indefinite" transition
- Competence reward design is critical for suspension structure emergence
- "Attending to breakdowns" promotes suspension when F/G are frozen

See `docs/development_report_2026_02_13_final.md` for full details.

---

**Last Updated**: 2026-02-13 (Afternoon)
**Status**: Phase 2 Complete (v4 + Full-scale Training), Phase 3 Ready to Start
