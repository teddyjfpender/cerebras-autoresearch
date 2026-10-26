# Bank-offset QK accumulator

Reuses DSRs and vector normalization, with an aligned QK scratch vector offset by one FP32 word. Screening improved the original 4-PE profile to about 2.14x, and the finer 32-PE profile to about 4x. These isolated-stage results are screens, not fresh confirmations.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.
