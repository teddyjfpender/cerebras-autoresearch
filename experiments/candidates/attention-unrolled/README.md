# Bounded QK and PV unrolling

Adds vector query scaling and compile-time bounded QK/PV updates to descriptor reuse and bank-offset accumulation. All causal terms still execute. Compile-time assertions restrict D <= 128 and S <= 32. Screening remained below the strict 5.01x target.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
