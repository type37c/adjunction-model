## Q2: What is the nature of the asymmetry in the adjunction?

**Date**: 2026-02-15

### The Question

What can be said about the asymmetry of the adjunction `F ⊣ G`?

### The Insight

There is a fundamental asymmetry in the relationship between Shape, Action, and the agent's internal state (C).

1.  **Shape is invariant to C**: A suitcase remains a suitcase regardless of whether the agent is tired or in a hurry. Shape is an invariant property of the environment.

2.  **Action is dependent on C**: The same suitcase might be interpreted as "liftable with one hand" by an energetic agent, but as "rollable with the left hand" by an agent with an injured right arm. The output of `F(Shape, C)` changes as C changes.

3.  **G's role is selection, not generation**: The functor `G` does not "generate" a shape from scratch. Instead, it "selects which aspect" of a shape to focus on. A shape is a multifaceted structure that simultaneously contains multiple functional aspects. G determines which facet to attend to based on the agent's state C. For the same bag, a tired agent might focus on the "handle" and "lightness," while an agent in a hurry might focus on the "capacity" and "opening size."

This led to the refinement of the functors to explicitly include C: `F(Shape, C)` and `G(Action, C)`, clarifying that the agent's internal state modulates the adjunction, making it a dynamic, context-sensitive relationship rather than a static one.
