# SDK reciprocal normalization

Uses the SDK FP32 reciprocal in place of general division. It did not improve on vector-max screening; no retained gain is claimed.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
