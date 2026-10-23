# Native FP32 recurrent refinement

Interleave four independent FP32 V-lane partial accumulators for memory projection and readout, then a three-add tree. Every K product is still evaluated; changed FP32 association must pass the unchanged local/full-chain bounds. Extra scratch is only3*V FP32 values, with code/descriptor growth subject to the original memory gate.

The starting reference already uses FP32 vector state arithmetic. Every token,
full state, intermediate, canonical input and protected timer/history function
is preserved. See candidate.json for the preregistered falsification gates.
