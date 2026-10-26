# FP32 vector recurrent state candidate

One V-element scratch vector is first the memory projection and then the
readout accumulator. Every K term is accumulated in ascending row order; the
outer update writes every state element. Normalization, exp, immutable inputs,
initial-state copy, four token frames, complete delta/state/output histories,
final state and protected timing functions remain unchanged.

The candidate uses FP32 FMA vector instructions with the fixed original error
contract. It is an optimization of the repository's independent scalar CSL
reference, not a production Cerebras inference implementation. Require matched
six-case validation, fresh paired seed and complete two-shape coverage before a
cross-shape retention claim. See candidate.json for falsification gates.
