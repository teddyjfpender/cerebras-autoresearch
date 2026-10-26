# Native FP32 recurrent refinement

Convert each canonical BF16 query/key once into FP32 work arrays, reuse exact converted values in the unchanged scalar squared-norm reductions, then apply the original query/key scaling with vector FP32 multiplies. Remove redundant conversions and scalar scaling loops.

The starting reference already uses FP32 vector state arithmetic. Every token,
full state, intermediate, canonical input and protected timer/history function
is preserved. See candidate.json for the preregistered falsification gates.
