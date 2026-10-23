# Vector polynomial softmax

Evaluates the degree-ten exponential polynomial with DSD vector operations. The initial positive-case screens were slower than DSR normalization; no retained gain is claimed.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
