## Q6: How does the architecture handle contextual affordances?

**Date**: 2026-02-15

### The Question

How does the architecture perceive a contextual affordance like "a cup on the edge of a table is likely to fall"? The current Functor F only takes the shape of the cup as input, not its relationship to the table.

### The Insight

This question revealed a critical distinction between object-centric affordances and scene-level affordances.

1.  **F's Limitation**: The current F is designed to process individual object shapes. It can infer "graspable" from a cup's geometry, but not "likely to fall," as that requires the context of the table.

2.  **Agent C's Role**: Agent C, through its GNN, processes the entire scene's spatial relationships. It *can* see that the cup is on the edge of the table. Therefore, the affordance "likely to fall" is not an output of F, but an **emergent property computed by Agent C** based on the outputs of F and the global scene context.

3.  **Language as a Context-Builder**: This led to a refined understanding of language's role. Language is not directly connected to F or G. Instead, it is a tool used by Agent C to **construct the context** that is fed into F. The input to F is not just `Shape`, but `F(Scene_Representation)`. Language helps build that `Scene_Representation`.
    *   `F(cup_shape)` → "graspable"
    *   `F(cup_shape, context="on edge of table")` → "graspable," "likely to fall"

4.  **Refined Model**: Language is a tool within Agent C's purpose space (P) that allows it to decide *how to query the world*. It determines *what context* to provide to the F functor, thereby changing the perceived affordances. This connects the abstract, symbolic nature of language to the concrete, physical nature of affordances without requiring a direct, rigid adjunction.
