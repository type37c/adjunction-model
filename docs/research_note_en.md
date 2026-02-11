_# The Physical-Semantic Adjunction Model: A Categorical Integration of Embodiment, Meaning, and Creativity_

**Research Note — Draft v1.0**

---

## 1. Introduction: What Current AI Lacks

While Large Language Models (LLMs) have achieved human-like natural language processing capabilities by learning statistical co-occurrences from vast text data, their abilities are fundamentally limited. An LLM knows the probabilistic association between the words "bag" and "carry," but it does not know the physical effort of lifting an object against gravity or the tactile sensation of a bag about to tear. In other words, the **lack of embodiment** and the associated **symbol grounding problem** remain unsolved.

This limitation becomes critical when dealing with novel situations. Current AI cannot intuitively infer what a new object can be used for—its **affordance**—when it encounters one not present in its training data. It can only search and imitate past data, lacking the ability to generate new meaning.

This research note proposes a new approach to this problem: the **Physical-Semantic Adjunction Model**. This model employs the mathematical structure of **adjunction** from category theory to construct a bidirectional semantic relationship between Shape and Action in real-time, mediated by the agent's embodiment.

---

## 2. Core Theory: The Adjoint Structure of Shape and Action

### 2.1 Basic Structure

The model divides the world into two categories, **Shape** and **Action**, and establishes a pair of adjoint functors between them.

The functor **F: Shape → Action** infers possible actions from an object's shape. For example, upon seeing a long, thin metal rod with a bent tip, it derives potential actions like "to hook" or "to pry open."

The functor **G: Action → Shape** calculates the necessary shape conditions to realize a certain action. For instance, from the action "to cut," it derives "a shape with a sharp edge."

When these two functors form an adjunction **F ⊣ G**, the following natural isomorphism of Hom-sets holds:

> **Hom<sub>Action</sub>(F(s), a) ≅ Hom<sub>Shape</sub>(s, G(a))**

This isomorphism mathematically guarantees the semantic correspondence between shape and action, where the left side represents the paths from a shape 's' to an action 'a' via F, and the right side represents the degree to which the shape 's' is the shape G(a) that enables the action 'a'.

### 2.2 Natural Transformations of the Adjunction

An adjoint pair is accompanied by two natural transformations.

The **Unit η: Id → G∘F** describes the relationship between the original shape and the composite map G(F(shape)), which transforms a shape into an action via F and then back into a shape via G. Through thought experiments, this composite map was confirmed to function as a projection that extracts the **functional core** of the original shape. For example, G(F(bag)) strips away the bag's color, material, and brand logo, returning only its functional essence: "a shape with an opening, an internal space, and a handle."

The **Counit ε: F∘G → Id** describes the properties of the composite map that transforms an action into a shape via G and then back into an action via F. F(G(cut)) returns not just "to cut" but a bundle of related actions like "to shave" and "to pierce." Thus, the counit functions as a **semantic extension of action**.

### 2.3 Conditional Adjunction: The Agent's Mediation

A crucial discovery was made through thought experiments: **the adjunction does not hold between "bare" Shape and Action**.

Consider the scenario of "carrying something in a bag." If the action "to carry" is too abstract, the set of shapes returned by G(carry) explodes; a flat board, a bucket, or a palm can all "carry." However, by adding the condition "to carry with minimal effort," G(carry with minimal effort) converges to "a shape with an opening, a handle, and the capacity to contain objects," allowing G(F(bag)) to correctly return the bag's functional core.

This discovery implies that the agent's **purpose, context, and physical state** (collectively denoted as **C**) are integral to the very conditions for the adjunction's existence. Formally, the model deals not with an adjunction between Shape and Action, but with a **parameterized adjunction** between **Shape and Action<sub>C</sub>**.

> **F<sub>C</sub> ⊣ G<sub>C</sub>**
>
> where C is the agent's context (a composite of purpose, constraints, and physical state).

### 2.4 Asymmetry of the Adjunction

This structure possesses an essential asymmetry: **Shape does not depend on the agent's state C, but Action does**.

Whether the agent is tired or in a hurry, a bag remains a bag, and a switch remains a switch. Shape is an invariant of the environment. In contrast, for the same suitcase, a healthy agent might choose to "lift and carry with one hand," while an agent with an injured right arm might choose to "roll with the left hand." A change in C alters the output of F.

Furthermore, G does not "generate" a shape but **selects** "which aspect" of a shape to see. A shape is a polyhedron that simultaneously embodies multiple functional aspects, and G determines which face of this polyhedron to focus on, according to the agent's state C. For the same bag, a tired agent focuses on its "handle" and "lightness," while an agent in a hurry focuses on its "capacity" and "opening size."

### 2.5 Preservation of Abstraction

The functor G **preserves the level of abstraction** in its translation from action descriptions to shape descriptions.

| Action Description | Shape Description returned by G | Abstraction Level |
| :--- | :--- | :--- |
| "Depress the switch's protrusion by 0.5cm with a finger" | "A push-button mechanism movable by 0.5cm" | Low (Concrete) |
| "Turn on the light" | "A controllable light source" | Medium |
| "Brighten the room" | "Something that emits light" | High (Abstract) |

These are not separate events but descriptions of the same causal chain at different levels of abstraction. G does not transform the level of abstraction but preserves it. A change in the agent's state C alters the level of abstraction λ at which the world is perceived, which in turn changes the granularity of the inputs and outputs of F and G. An independent adjunction F<sub>λ</sub> ⊣ G<sub>λ</sub> exists for each level of abstraction λ, and C determines which λ to select.

---

## 3. The Coherence Signal: The Inevitability of Using Adjunction

### 3.1 The Riverbed Analogy

The design philosophy of this model is encapsulated in the **"riverbed"** analogy. A riverbed is a structure that exists before water flows, constraining the flow while also being gradually eroded and reshaped by it.

Translating this analogy into an AI architecture implies a demand for a mechanism that allows the agent to inspect its own internal structure for stability *before* acting. The reward signal in reinforcement learning is inherently post-hoc; it only recognizes a "collapse" after an action has failed. The riverbed philosophy requires a mechanism to internally detect stability beforehand.

### 3.2 Defining the Coherence Signal

The **coherence signal** answers this demand. It is defined as the "size" of the adjunction's unit η.

> **coherence signal = distance(shape, G(F(shape)))**

It measures the distance between the original shape and the shape reconstructed via F and G. If this distance is small, the agent's current internal model is coherent with the environment (close to commutative). If the distance is large, coherence has broken down (non-commutative).

Crucially, **this signal cannot be defined without the adjoint structure**. In a mere pair of bidirectional maps (where F and G are independent), there is no standard for judging whether G(F(x)) has "returned to the original." The existence of the unit η provides a natural metric for measuring "how much it has not returned." This is one of the strongest justifications for the substantive, non-ornamental role of the adjoint structure in this model.

### 3.3 Coherence Breakdown and Creativity

The moment of commutativity collapse—the moment the coherence signal spikes—is when **creative problem-solving** is demanded of the agent.

For instance, consider carrying a suitcase in a quiet environment with an injured right arm. The usual alternative, "to drag," is unavailable due to the noise it creates. The physical state filter and the context filter interfere, breaking commutativity. At this point, the agent must synthesize existing action primitives in a new way (e.g., "cradling it with both knees to walk") to resolve the non-commutativity.

This structure suggests the possibility of **formalizing creativity as a process of repairing a commutative diagram**.

### 3.4 Revising the Position of the Coherence Signal

An important revision is necessary here. In the initial discussions, the coherence signal was placed at the center of the model. However, a more precise analysis revealed that the coherence signal is a means to implement **one of the five requirements of the suspension structure (sensitivity to difference)**, not the center of the entire model (this will be detailed in Section 4).

The necessity of using adjunction rests not on the single pillar of the coherence signal, but on a broader foundation: the fact that four of the five requirements of the suspension structure are naturally derived from the adjoint structure.

---

## 4. The Suspension Structure: The Minimal Components of Intelligence

### 4.1 The Suspension Structure as the Core

The most original concept in this model is the **suspension structure**. It was proposed as the "core" that penetrates the bundle of adjoint structures across multiple levels of abstraction.

In conventional models of intelligence, the core has been something fixed—a body model, a top-level goal, a self-identity. In this model, however, the core is **the very structure of "not-yet-decidedness."** The suspension structure functions not as a specific answer, but as a "device for holding a question."

Just as a universal construction (like a limit or colimit) in category theory is defined not as a specific object but as a condition—"the unique object satisfying these properties"—the suspension structure is defined as the condition of "something that makes all levels of abstraction coherent," which is then instantiated according to the situation.

> **Intelligence is not the explicit maintenance of a purpose, but the possession of a "suspension structure + memory" that does not collapse even when the purpose is forgotten.**

### 4.2 The Five Requirements of the Suspension Structure

The following five requirements have been identified as the minimal components of the suspension structure. Intelligence cannot be established if even one is missing.

| Requirement | Description | Consequence if Lacking |
| :--- | :--- | :--- |
| **Intentionality** | Having a direction toward something | Becomes a reactive machine without direction |
| **Sensitivity to Difference** | Being able to detect differences | Cannot notice environmental changes |
| **Temporal Persistence** | Being able to maintain a state for a period | Remains at the level of instantaneous reflexes |
| **Self/Non-self Distinction** | Being able to distinguish one's own actions from environmental changes | The boundary between self and other dissolves |
| **Memory** | Being able to retain and refer to past experiences | Learning does not accumulate |

### 4.3 Correspondence with the Adjoint Structure

These five requirements are realized in the model's architecture as follows:

| Suspension Structure Requirement | Location in Architecture | Realization Mechanism |
| :--- | :--- | :--- |
| **Sensitivity to Difference** | Adjoint Structure (unit η) | coherence signal = distance(shape, G(F(shape))) |
| **Intentionality** | Agent State C | A goal vector within C acts as a parameter for the adjunction |
| **Temporal Persistence** | Dynamic Loop C(t) → C(t+1) | The state is updated and persists across discrete time steps |
| **Self/Non-self Distinction** | Asymmetry of the Adjunction | Shape is independent of C (environment=non-self), Action is dependent on C (action=self) |
| **Memory** | Agent State C | Stored as a memory module within C |

Notably, four of the five requirements (sensitivity to difference, intentionality, temporal persistence, and self/non-self distinction) are naturally derived from the adjoint structure. Only memory is located outside the adjunction, but it is stored within the agent state C. **The content of memory is determined by the adjunction (what to remember = experiences with high coherence signal), while the retention of memory is handled by C.**

---

## 5. The Two-Layer Architecture

### 5.1 Design Principles

Based on the analysis above, the model's architecture consists of **two layers**.

The **Adjoint Layer** describes the structural relationship between Shape and Action. From this layer emerge sensitivity to difference and the self/non-self distinction. A dynamic Graph Neural Network (GNN) is envisioned as the implementation basis for this layer. Nodes represent functional primitives (e.g., "to grasp," "to wrap," "to support"), edges represent composability, and the graph's topology and edge weights continuously deform in response to the coherence signal.

The **Agent Layer (C)** maintains the entire internal state of the agent, which parameterizes the adjoint layer. Intentionality (purpose) and memory are stored here. C is not a fixed parameter but a state variable that dynamically changes according to the following update rule:

> **C(t+1) = update(C(t), action_result, coherence_signal)**

Temporal persistence is achieved through the dynamic loop that couples these two layers. The agent layer C determines the behavior of the adjoint layer, and the result of the adjoint layer's interaction with the world (the coherence signal) updates the agent layer C. This is the riverbed analogy itself—the flow of water (action) erodes the shape of the riverbed (graph structure), and the shape of the riverbed constrains the flow of water.

### 5.2 Implementation Concept for the Adjoint Layer with a Dynamic GNN

A dynamic GNN with the following structure is conceived as the implementation basis for the adjoint layer.

As a **static layer**, nodes represent functional primitives, and edges represent composability. As an **adjoint structure**, F (Shape→Action) is implemented as message passing on the graph, and G (Action→Shape) as a graph-wide readout (in reverse). As the **continuously changing part**, the edge weights (ease of composition), node activation thresholds, and the graph topology itself (addition of new primitives) are updated in response to the coherence signal.

### 5.3 Extension to a Language Faculty

The model's structure naturally accommodates the addition of language capabilities as a **hierarchical layering of the same adjoint pattern**.

While the current structure is an adjunction of Shape ⇄ Action, the extended version becomes a chain of two adjunctions: Shape ⇄ Action ⇄ Language. Language is another adjoint layer that "names" the adjunction of shape and action. This extension requires no new principles; the same pattern of adjunction + coherence is merely repeated at each layer.

---

## 6. Correspondence with Active Inference

### 6.1 Structural Correspondence

A strong structural correspondence exists between this model and Karl Friston's Active Inference [1].

| Concept in This Model | Concept in Active Inference |
| :--- | :--- |
| Coherence signal (η) | Free energy (prediction error) |
| Dynamic loop of F⊣G | Action-Perception loop |
| Agent state C | Parameters of the generative model |
| Coherence breakdown | Surprise (unexpected observation) |

The correspondence between the coherence signal and free energy is particularly essential. G(F(shape)) is "the state of the world as predicted by the agent from its current internal state C and the shape," and distance(shape, G(F(shape))) is the discrepancy between that prediction and reality, i.e., the prediction error.

### 6.2 The Uniqueness of This Model

However, this model is not merely a reformulation of Active Inference. The following concepts cannot be directly addressed within the framework of Active Inference.

The **suspension structure** is qualitatively different from the "generative model" in Active Inference. While a generative model is a probabilistic description of the world, the suspension structure is a formalization of indeterminacy itself—the structure of "not-yet-decidedness."

The claim that **coherence breakdown gives rise to creativity** has a different orientation from the principle of surprise minimization in Active Inference. In Active Inference, surprise (prediction error) is something to be minimized, whereas in this model, coherence breakdown is actively positioned as a trigger for creative problem-solving.

The **asymmetry of the adjunction**—that Shape is independent of C while Action is dependent on C—is not explicitly handled in the symmetric framework of Active Inference.

### 6.3 Positioning

This model can be positioned as **"a reformulation of Active Inference in the algebraic language of category theory, further extended with the concepts of suspension structure and creativity."** While Active Inference is based on Bayesian probability theory, this model focuses on the structure of the adjoint functors themselves. This allows for a more direct treatment of concepts that are difficult to formalize probabilistically, such as suspension, indeterminacy, and creativity.

Notably, **Structured Active Inference** by Smithe (2024) [2] is a study that significantly generalizes Active Inference using categorical systems theory and is the closest prior work to this model. In particular, the concept of mode-dependence, where available actions depend on the current state, directly corresponds to the parameter C of the parameterized adjunction in this model.

---

## 7. Outlook for Implementation

### 7.1 Table of Technological Correspondences

| Theoretical Component | Implementation Strategy | Tools & Prior Work |
| :--- | :--- | :--- |
| Shape Category | 3D object point clouds or meshes | ShapeNet [3], PartNet |
| Action Category | Probability distribution of relationships between body parts and object surface points | ZSP3A [4], Contact-GraspNet [5] |
| Functor F | Affordance prediction model using GNN | PyTorch Geometric [6] |
| Functor G | Inverse inference model using GNN | To be newly developed |
| Coherence signal | Reconstruction error: distance(shape, G(F(shape))) | Self-supervised learning |
| Agent state C | Latent state vector of an RNN or Transformer + external memory | Neural Turing Machine [7] |

### 7.2 A Minimal Viable Task for Verification

To provide an initial proof of concept for this grand theory, the following minimal experiment is proposed.

First, select 3D models from a specific category (e.g., chairs, cups) from ShapeNet and create paired data of shapes and grasping poses, referencing methods from Contact-GraspNet [5] and ZSP3A [4]. Second, implement GNN-based F and G using PyTorch Geometric. Third, train F and G simultaneously in a self-supervised learning framework with the loss function `loss = distance(shape_contact_region, G(F(shape)))`.

A successful experiment would be the first demonstration that **an adjoint structure can learn bidirectional inference between shape and action by minimizing reconstruction error**.

---

## 8. Conclusion and Future Work

This research note has constructed the theoretical framework of the Physical-Semantic Adjunction Model through thought experiments and progressive refinement. The main achievements are summarized below.

**Theoretical contributions** include the discovery of "conditional adjunction," where the adjoint structure is parameterized by the agent's context C; the formulation of an internal stability detection mechanism via the coherence signal; the identification of the five requirements of the suspension structure and the confirmation that four of them are naturally derived from the adjoint structure; and the framework for positioning coherence breakdown as a condition for creativity.

Regarding **the relationship with prior work**, a strong structural correspondence with Active Inference was confirmed, while the unique contributions of the suspension structure, creativity, and the asymmetry of the adjunction were clarified.

**Future work** includes the following. First, a rigorous verification of whether the properties of the unit η (e.g., commutativity as a natural transformation) are formally satisfied in this model. Second, a precise formulation of the conditions under which the suspension structure is "instantiated." Third, the design of the time scale for the graph deformation in the dynamic GNN. Fourth, the clarification of the relationship between the addition of a language layer and existing LLMs. Fifth, the empirical validation of the theory by executing the minimal viable task.

This model aims to shift from an AI that "learns from data" to an AI that "redefines the world through its body." Its scope may extend beyond the design of AI architectures to a redefinition of intelligence itself.

---

## References

[1]: Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

[2]: Smithe, T. S. C. (2024). *Structured Active Inference (Chapters 1-3)*. arXiv:2406.07577.

[3]: Chang, A. X., et al. (2015). *ShapeNet: An Information-Rich 3D Model Repository*. arXiv:1512.03012.

[4]: Kim, H., et al. (2024). *Zero-Shot Learning for the Primitives of 3D Affordance in General Objects*. arXiv:2401.12978.

[5]: Sundermeyer, M., et al. (2021). *Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes*. arXiv:2103.14243.

[6]: Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. arXiv:1903.02428.

[7]: Graves, A., et al. (2014). *Neural Turing Machines*. arXiv:1410.5401.

[8]: Gavranović, B., et al. (2024). *Categorical Deep Learning: An Algebraic Theory of Architectures*. ICML 2024. arXiv:2402.15332.
