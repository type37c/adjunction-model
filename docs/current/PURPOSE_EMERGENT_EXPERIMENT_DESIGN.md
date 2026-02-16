# Purpose-Emergent Active Assembly: Experiment Design

**Design Note**

**Date**: 2026-02-15

---

## 1. Core Idea: Emergence over Instruction

This experiment is designed to test a fundamental hypothesis: **Can purpose emerge from an agent's interaction with its environment, rather than being explicitly instructed?**

The core idea is to shift from evaluating the distance to a single, predefined target to evaluating the **existence of a force directed towards *any* coherent shape**.

| | Previous Implementation | Purpose-Emergent Design |
|:---|:---|:---|
| **Target** | One (predefined) | **Any** (chosen by the agent) |
| **Evaluation** | Distance to the target | **Existence of a "force of intention"** |
| **Suspension** | "I can't move because I lack information" | **"I haven't decided which one to choose yet"** |
| **Resolution** | Information increases, revealing the target | **The agent commits to a choice** |

---

## 2. Architecture: Purpose-Emergent Active Assembly

### 2.1. Dataset: No Target, All Points from the Start

We replace the previous dataset with `PurposelessAssemblyDataset`.

- **No `target_points`**: The dataset provides only the initial scattered points.
- **No Progressive Revelation**: All points are presented from step 0. Ambiguity is not in the environment but in the agent's internal state (η).
- **Reference Shapes**: A dictionary of reference shapes (cube, sphere, etc.) is provided, not for training, but as a basis for measuring the "force of intention."

### 2.2. Loss Function: The Force of Intention

The key to enabling purpose emergence is the `L_purpose`:

```python
L_purpose = torch.stack([chamfer_distance(pos, shape) for shape in reference_shapes]).min()
```

This `min over shapes` Chamfer Distance allows the agent the freedom to choose any target. It penalizes states that are dissimilar to *all* known shapes but rewards states that are close to *any* of them.

### 2.3. Training Loop

The training loop proceeds for a fixed number of steps (`num_steps`).

1.  **Displacement**: The agent's `DisplacementHead` generates a displacement vector for each point.
2.  **Curiosity Reward**: A curiosity reward `R_curiosity = η(t-1) - η(t)` is calculated. A decrease in η (moving towards a coherent shape) is rewarded.
3.  **Loss Calculation**: The total loss `L = L_purpose + λ_coherence * L_coherence + λ_kl * L_kl` is calculated **only at the final step**. This gives the agent the freedom to decide *when* to commit.

### 2.4. Inertia of Purpose: The Role of Valence Memory (Revised on 2026-02-16)

The mechanism for creating an "inertia of purpose" has been refined based on the latest theoretical insights.

**Previous Idea (Problematic)**:
- The `valence` was updated based on `R_curiosity = η(t-1) - η(t)`.
- This reward was then used in a `Priority = coherence × uncertainty × valence` calculation.
- This approach was an over-design, as it hard-coded the attention mechanism.

**New Principle: Emergence of Attention Mechanism**

1.  **Feedback Loop**: The agent's action causes a change in Slack (`Δη`). This change is not a reward but a **consequence**.
2.  **Valence as Memory**: `valence_memory` records the association between the action taken and the resulting `Δη`.
3.  **Internal State Update**: The `valence` (the memory of action-consequence pairs) is fed into the agent's RSSM (`h_t`).
4.  **Learned Attention**: The agent's RSSM learns to use this `valence` memory, along with current observations of `coherence` and `uncertainty`, to decide where to allocate attention.
5.  **Self-Organized Commitment**: A positive feedback loop is established not by an explicit reward, but by the agent learning that certain actions lead to stable and predictable Slack dynamics. The agent learns to commit to a shape because doing so creates a more manageable internal state.

This new design is a more faithful implementation of our core principle: **"We do not design the suspension structure; we design the conditions for its emergence."** We are no longer designing the attention mechanism (the Priority formula); we are designing the conditions (the feedback loop and memory structure) under which the attention mechanism itself can emerge.

---

## 3. Observational Metrics: Evaluating the Emergence

We will track the following metrics to evaluate the emergence of purpose:

| Metric | Expected Behavior | Meaning |
|:---|:---|:---|
| `eta_trajectory` | High → Low (Monotonic decrease) | The process of suspension resolving |
| `purpose_switches` | Fluctuates initially → Stabilizes | Selection and stabilization of purpose |
| `displacement_trajectory` | Small → Large | Cautious exploration → Commitment |
| `commitment_step` | Around steps 3-5 | The timing of suspension resolution |

---

## 4. Conclusion: Designing the "Geology"

This experimental design shifts from "constructing" suspension by providing a target to designing the "geology"—the conditions under which suspension can **emerge**.

- The environment only says, "Become something coherent."
- The agent itself chooses a purpose and commits to it.
- Valence memory creates inertia, and the decrease in η visualizes the commitment.

This design is the most faithful implementation of our core theoretical ideas.
