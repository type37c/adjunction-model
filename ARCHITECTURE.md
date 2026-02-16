# Adjunction Model Architecture (v2.1)

**Last Updated**: 2026-02-17

## 1. Core Philosophy: Intelligence as Slack Management

This model redefines intelligence not as perfect prediction, but as **effective slack management**. The central concept is the **suspension structure**, an always-on process that governs how the agent's understanding of the world (the **adjunction F⊣G**) is formed and maintained.

This is achieved through the preservation and utilization of **slack**, a quantifiable measure of the "looseness" in the agent's world model:

-   **η (Shape Slack)**: `||Shape - G(F(Shape))||²` — Represents **perceptual uncertainty**.
-   **ε (Affordance Slack)**: `||Affordance - F(G(Affordance))||²` — Represents **semantic ambiguity**.

Our foundational experiments have shown that **η and ε are strongly coupled** (Spearman correlation 0.92) and that removing reconstruction loss causes **total slack to grow by over 750%**, providing the agent with the flexibility needed for complex, competence-driven behavior.

## 2. Current Paradigm: Value-Based Intrinsic Reinforcement Learning

The current architecture (`v2.1`) focuses on reproducing the seminal 2/13 experiment, where goal-directed behavior emerges from pure intrinsic motivation. This approach abandons supervised learning for Agent C in favor of a **value-based reinforcement learning** framework, which is essential for modeling temporal concepts like **suspension (保留)**.

> **Core Insight**: Suspension is fundamentally about delaying immediate action for future benefit. This requires a future-oriented learning mechanism, which is precisely what value-based RL (specifically TD learning) provides. A simple supervised model that only predicts the *current* state cannot learn to plan for the *future*.

### Three-Phase Training Cycle (per Epoch)

The training process for the `ValueBasedTrainer` is a faithful reproduction of the successful 2/13 experiment. It consists of a three-phase cycle that repeats every epoch:

![Value-Based Training Cycle](docs/assets/value_based_training_cycle.png)
*Figure 1: The three-phase training cycle for the Intrinsic Reward Baseline experiment.*

**F and G are frozen throughout this entire process.**

#### Phase 1: Trajectory Collection (No Gradient)
-   **Goal**: Collect experience by letting the agent interact with the environment.
-   **Process**: The agent, using its current policy, performs actions for a set number of episodes. All states, actions, and resulting intrinsic rewards are stored in a replay buffer.
-   **Components**: `AdjunctionModel` (in `eval` mode), `AgentC` (policy), Environment.

#### Phase 2: Value Function Update (TD Learning)
-   **Goal**: Teach the `ValueFunction` to predict the expected future sum of rewards.
-   **Process**: The `TDLearner` samples trajectories from the replay buffer and updates the `ValueFunction` using the Bellman equation: `V(s_t) ← R_t + γ * V(s_{t+1})`.
-   **Components**: `ValueFunction`, `TDLearner`.
-   **Loss**: Temporal Difference (TD) Error.

#### Phase 3: Agent C Update (Value Maximization)
-   **Goal**: Improve Agent C's policy to take actions that lead to high-value states.
-   **Process**: Agent C is updated to maximize the value predicted by the now-fixed `ValueFunction`. The agent learns to select actions that it believes will yield the highest long-term intrinsic reward.
-   **Components**: `AgentC`.
-   **Loss**: `-V(s_t)` — The loss is the *negative* of the value, so minimizing the loss maximizes the value.

This cycle allows the agent to learn complex, temporally-extended behaviors by explicitly modeling the future value of its actions.

## 3. Component Architecture

| Component | Model | Input | Output | Key Function |
| :--- | :--- | :--- | :--- | :--- |
| **F (Encoder)** | PointNet++ | Point Cloud | Affordance Dist. | Shape → Action Possibilities |
| **G (Decoder)** | FoldingNet | Affordance Dist. | Point Cloud | Action Possibilities → Shape |
| **Agent C** | RSSM | Observation | Action | Policy: Selects actions to maximize future value |
| **Value Func.** | MLP | State | Scalar Value | Value Estimation: Predicts expected future reward |

### 3.1. Adjoint Layer (F⊣G)
-   The world model, responsible for interpreting sensory input (point clouds) into a latent space of affordances and vice-versa. In the current experiments, F and G are **pre-trained and frozen**.

### 3.2. Agent & Value Layer (C, V)
-   **Agent C (RSSM)**: The agent's policy network. It takes the current observation from the environment and decides on an action. Its goal is to steer the environment into states that the `ValueFunction` deems valuable.
-   **ValueFunction (V)**: A simple MLP that learns to map a given state `s_t` to a scalar value representing the expected discounted sum of future intrinsic rewards. It provides the crucial long-term perspective that Agent C uses to make decisions.

## 4. Theoretical Justification: Valence and Temporal Dynamics

The move to a value-based architecture is strongly motivated by our latest theoretical insights documented in `docs/theory/priority_and_valence_reconsidered.md`.

-   **Priority is an emergent property, not a calculation**: The old model's hardcoded `Priority = Coherence × Uncertainty × Valence` was flawed. The agent must *learn* how to balance these factors.
-   **Valence is Memory**: Coherence and Uncertainty describe the *current* state. Valence, however, is the only axis that spans time. It represents the agent's accumulated confidence and is intrinsically linked to the *expected future outcome* of its actions.
-   **TD Learning as the vehicle for Valence**: The `ValueFunction` is the concrete implementation of this temporal, memory-based aspect. It learns the agent's "optimism" or "pessimism" about future states, which is the essence of valence.

This architecture allows us to test the hypothesis that **an agent driven to maximize its future competence (high value) will spontaneously learn to organize its world**, thus reproducing the powerful results of the 2/13 experiment in a more theoretically sound manner.

## 5. Future Experiments (Post-Baseline)

Once the intrinsic reward baseline is firmly established, we will proceed to more complex experiments:

-   **Purpose-Emergent Active Assembly**: Investigating if an agent can develop a preference for a specific shape (e.g., a sphere) without being explicitly told to, driven by a `min_shape(ChamferDistance)` loss.
-   **Temporal Suspension**: Designing experiments where the agent must explicitly delay gratification (e.g., ignore a small, immediate reward for a much larger future reward) to test the limits of the value function's temporal planning capabilities.
