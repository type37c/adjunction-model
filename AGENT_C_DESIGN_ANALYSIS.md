# Agent C Design Analysis: Uncovering the Source of η-Explosion

**Date**: 2026-02-17
**Author**: Manus AI

## 1. Executive Summary

The `intrinsic_reward_baseline` experiment failed due to a catastrophic explosion of the coherence signal (η), which grew from ~0.47 to over 96 trillion. This document analyzes the root cause of this failure, tracing it back to a fundamental design issue in **Agent C**. 

The core problem is an unstable positive feedback loop where Agent C, in its attempt to maximize the learned Value function, generates destructive `context` vectors. These vectors corrupt the FiLM modulation of the frozen F/G networks, leading to nonsensical reconstructions and an exploding reconstruction error (η). The Value function, which is indirectly dependent on η, also explodes as a result.

This analysis compares the current implementation with the successful 2/13 legacy experiment, identifies the critical missing constraints, and proposes concrete solutions to stabilize the system.

## 2. The Unstable Feedback Loop

The current value-based training architecture creates a fatal feedback loop:

1.  **Agent C Generates Context**: At each step, Agent C produces a `context` vector based on its internal state.
2.  **Context Modulates F/G**: This `context` vector is used to generate `gamma` and `beta` parameters for the FiLM layers, which modulate the behavior of the (frozen) F and G networks.
3.  **F-G Adjunction**: The modulated F and G networks perform the core adjunction task: `Shape -> Affordances -> Reconstructed Shape`.
4.  **η is Calculated**: The coherence signal (η) is calculated as the Chamfer distance between the original and reconstructed shape. It represents the reconstruction error.
5.  **η Influences Reward**: η is a component of the intrinsic reward `R_intrinsic` (specifically, `R_competence` is based on the reduction of η).
6.  **Value Function Learns Reward**: The Value function `V(s)` is trained via TD-learning to predict the expected future `R_intrinsic`.
7.  **Agent C Chases Value**: Agent C's policy is trained to take actions (i.e., produce a `context`) that lead to states with the highest predicted value `V(s)`.

This loop becomes unstable because **there are no constraints on the `context` vector**. Agent C discovers that by outputting extreme values in `context`, it can radically manipulate the behavior of F/G. This manipulation leads to an explosive increase in η, which in turn causes an explosive increase in the predicted Value. Agent C is therefore incentivized to destroy the adjunction structure to maximize its objective.

## 3. Key Differences: Current vs. 2/13 Legacy Code

The investigation reveals a critical difference in how the `context` vector was handled between the current implementation and the stable 2/13 legacy code.

| Feature | Current Implementation (Exploded) | 2/13 Legacy Implementation (Stable) |
| :--- | :--- | :--- |
| **F/G Parameters** | Frozen (`requires_grad=False`) | Frozen (`requires_grad=False`) |
| **Agent C Objective** | Maximize `V(s)`, which is learned from `R_intrinsic` (η-dependent). | Not explicitly trained to maximize a value function. The agent's state evolved, but its parameters were not being optimized against a long-term reward signal in the same way. |
| **Context Generation** | Output of Agent C's GRU and linear layers. | Output of Agent C's GRU and linear layers. |
| **Constraints on Context** | **None.** The `context` vector can take on any value. | **Implicit Constraints.** While not explicitly regularized, the training regime (Phase 2 training with affordance loss) indirectly constrained the `context` to produce meaningful reconstructions. The agent was not being driven by a long-term value signal that incentivized breaking the structure. |
| **Result** | η explodes to >10¹³, Value function explodes. | η (Coherence) remained stable around **0.43**. |


## 4. Root Cause: Unconstrained Context Modulation

The evidence strongly suggests that the root cause of the explosion is the **unconstrained nature of the `context` vector** used for FiLM modulation. 

- **F/G are confirmed to be frozen.** The explosion is not due to F/G parameters being updated.
- **The η calculation is correct.** The method matches the legacy code.
- **The explosion begins between epoch 20 and 30**, which is when the Value function starts to provide a strong enough signal to guide Agent C's policy.

Agent C is behaving rationally according to its objective: it has found a loophole to maximize its predicted reward. The loophole is that by generating massive `gamma` and `beta` values in the FiLM layers, it can create astronomical reconstruction errors (η), which the broken Value function misinterprets as highly desirable.

## 5. Proposed Solutions

To fix this, we must close the loophole by constraining the `context` vector or the resulting FiLM parameters. 

### **Primary Recommendation: Add Constraints to FiLM Parameters**

This is the most direct and targeted solution.

1.  **Clip FiLM Parameters**: After the `context` vector is passed through the linear layers to produce `gamma` and `beta`, clip their values to a reasonable range. A good starting point would be to keep `gamma` centered around 1 and `beta` around 0.

    ```python
    # In the FiLM layer or where gamma/beta are generated
    gamma = torch.clamp(gamma, 0.5, 1.5) 
    beta = torch.clamp(beta, -0.5, 0.5)
    ```

2.  **Add L2 Regularization to Context**: Add a penalty to Agent C's loss function proportional to the squared magnitude of the `context` vector. This will discourage the agent from producing large context values.

    ```python
    # In Agent C's loss calculation
    agent_loss = -total_value / len(trajectory)
    context_l2_penalty = torch.stack([s["context"] for s in trajectory]).norm(2) * 1e-4
    agent_loss += context_l2_penalty
    ```

### **Secondary Recommendation: Gradient Clipping**

While we have learning rate and reward scaling, clipping the gradients of the Value function and Agent C optimizers can provide an additional layer of stability against sudden explosions.

```python
# In the training loop
td_loss.backward()
torch.nn.utils.clip_grad_norm_(value_function.parameters(), max_norm=1.0)
value_optimizer.step()

agent_loss.backward()
torch.nn.utils.clip_grad_norm_(model.agent_c.parameters(), max_norm=1.0)
agent_c_optimizer.step()
```

## 6. Next Steps

We must implement the primary recommendation (FiLM parameter constraints) before re-running the experiment. This is not a hyperparameter tweak but a fundamental fix to ensure the stability of the adjunction structure. We will prioritize adding clipping to the FiLM `gamma` and `beta` values and then proceed with a new verification experiment.
