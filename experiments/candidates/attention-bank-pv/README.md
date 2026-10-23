# Bank-separated weighted-value partials

Groups KV slots into four strided FP32 accumulators, then adds their partial vectors. It slightly improved the coarse 4-PE screen but added overhead at small shard sizes; no retained gain is claimed.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
