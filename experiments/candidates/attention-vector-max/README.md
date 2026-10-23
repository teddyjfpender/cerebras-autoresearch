# Vector shard-maximum merge

Adds a bank-offset vector merge of gathered head maxima, constant head scaling, and bounded short-loop unrolling. The strict 5.01x challenge rejected its 4.97x screen.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
