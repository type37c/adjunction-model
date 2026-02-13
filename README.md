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

### ✅ Phase 2 Slack Experiment: Complete Success

We have successfully validated the **Phase 2 Slack Hypothesis**, demonstrating that:

1.  **Slack is Preserved and Amplified**: Total slack (η + ε) grew by **+757%** while task performance (affordance loss) improved by **93%**.
2.  **η and ε are Strongly Coupled**: A Spearman correlation of **0.92** proves that shape slack and affordance slack are not independent but co-evolve.
3.  **Slack is Used for Exploitation**: A near-perfect negative correlation of **-0.99** between slack and exploration (KL divergence) reveals that the agent uses slack to **improve task efficiency**, not to explore.
4.  **Suspension Structure Emerges**: We found **strong evidence (2/3 score)** of suspension structure, including non-monotonic behavior and active slack modulation.

> For a full breakdown, see the **[Comprehensive Analysis Report](results/phase2_slack/COMPREHENSIVE_ANALYSIS_REPORT.md)**.

### ✅ Architecture v2.0: Three-Phase Training

The model now uses a three-phase training process to manage the interplay between learning and slack.

![Three-Phase Training Process](docs/assets/three_phase_training.png)

> See the full **[ARCHITECTURE.md](ARCHITECTURE.md)** for details.

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
│   ├── models/          # -> F, G, Agent C, and AdjunctionModelV4
│   └── training/        # -> Phase 1, 2, and 3 trainers
├── experiments/
│   ├── phase2_slack_experiment.py
│   └── analyze_phase2_slack.py
└── results/
    └── phase2_slack/    # -> All data, logs, and reports from the experiment
```

---

## 4. Development Guidelines

All development, especially by AI agents, MUST follow the rules outlined in **[AGENT_GUIDELINES.md](AGENT_GUIDELINES.md)**. This file contains the core principles for ensuring consistent and high-quality development.

## 5. License

TBD
