# Adjunction Model: Suspension, Slack, and Competence

**A novel AI architecture that uses category theory to model how agents learn, adapt, and create through the preservation and exploitation of "slack" in their world models.**

---

## 1. Core Idea: Intelligence as Slack Management

This project redefines AI intelligence not as perfect prediction, but as **effective slack management**. The core idea is that an agent's understanding of the world is a "loose" and flexible structure (an **adjunction F⊣G**), and intelligence arises from the ability to preserve and exploit the "slack" in this structure.

-   **η (Shape Slack)**: The error in reconstructing a shape from its affordances. Represents **perceptual uncertainty**.
-   **ε (Affordance Slack)**: The error in re-encoding the affordances of a reconstructed shape. Represents **semantic ambiguity**.

Our latest experiments show that by removing reconstruction loss, the model learns to **preserve and amplify** this slack, which is then used by the agent to improve its competence.

## 2. Major Achievements (Feb 2026)

This project has moved from theoretical formulation to groundbreaking empirical validation.

### ✅ Intrinsic Reward Baseline Experiment (2/13 Reproduction)

Our current flagship experiment reproduces the key findings of 2/13, where an agent learns to form structures driven purely by intrinsic motivation, without any external objective.

- **F/G Frozen**: The world model (F and G) is fixed.
- **No Affordance Loss**: The agent is not told what to do.
- **Value-Based Training**: The agent learns to maximize the expected future sum of intrinsic rewards (competence and novelty).

**Key Finding (from 2/13)**: The agent spontaneously learns to focus on areas of high coherence (破綻への注目), leading to a **+1584%** increase in intrinsic rewards and an accelerating growth in `valence` (the agent's confidence in its own competence).

This experiment validates the core hypothesis: **goal-directed behavior can emerge from the agent's own internal drive to improve its competence.**

### ✅ Phase 2 Slack Experiment: Foundational Results

Our foundational slack experiments (now in `experiments/slack_affordance_loss/`) have validated core hypotheses about suspension structure:

1.  **Slack is Preserved and Amplified**: Total slack (η + ε) grew by **+757%** while task performance (affordance loss) improved by **93%**.
2.  **η and ε are Strongly Coupled**: A Spearman correlation of **0.92** proves that shape slack and affordance slack are not independent but co-evolve.
3.  **Slack is Used for Exploitation**: A near-perfect negative correlation of **-0.99** between slack and exploration (KL divergence) reveals that the agent uses slack to **improve task efficiency**, not to explore.

> For a full breakdown, see the **[Comprehensive Analysis Report](results/phase2_slack/COMPREHENSIVE_ANALYSIS_REPORT.md)**.

---

## 3. Getting Started

To get started with the project, please see the documents in the `docs/current/` directory:

1.  **[QUICKSTART.md](docs/current/QUICKSTART.md)**: Set up your environment and run the main experiments.
2.  **[ARCHITECTURE.md](ARCHITECTURE.md)**: Understand the v2.0 architecture and three-phase training.
3.  **[EXPERIMENTAL_RESULTS.md](docs/current/EXPERIMENTAL_RESULTS.md)**: Review the key findings from our experiments.

### Documentation Structure

```
adjunction-model/
├── README.md
├── ARCHITECTURE.md
├── AGENT_GUIDELINES.md
├── TODO.md
├── docs/
│   ├── current/         # -> Up-to-date guides and specifications
│   ├── theory/          # -> Core theoretical background
│   └── archive/         # -> Historical logs and dated analyses
├── src/
│   ├── models/          # -> F, G, Agent C, ValueFunction, etc.
│   └── training/        # -> ValueBasedAgentTrainer, Phase2SlackTrainer
├── experiments/
│   ├── slack_affordance_loss/   # -> Phase 2 Slack (L_aff driven)
│   ├── intrinsic_reward_baseline/ # -> 2/13 Reproduction (Value driven)
│   ├── purpose_emergent/        # -> Purpose-Emergent Assembly
│   └── ...
└── results/
    └── ...
```

---

## 4. Development Guidelines

All development, especially by AI agents, MUST follow the rules outlined in **[AGENT_GUIDELINES.md](AGENT_GUIDELINES.md)**. This file contains the core principles for ensuring consistent and high-quality development.

## 5. License

MIT License

TBD
TBD

TBD
