# Bank offset with polynomial softmax

Combines the bank-offset scratch accumulator with vector polynomial softmax. Screening did not beat the exact-SDK-exp bank candidate; no retained gain is claimed.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
