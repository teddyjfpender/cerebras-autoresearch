# FP32 positive reciprocal normalization

Confirmed at 5.103x on decode-attention.d64-ctx128-latency32: 220,273 to 43,165 sequence cycles at both seeds, with six cases per run. Program-object memory is 30,480 versus 29,216 bytes/PE. Both arms use 32 compute PEs. This is a simulator operator result, not hardware/model TPS.

The current baseline adds unchanged arithmetic hooks for normalization and
maxima merging. Source/evaluator identities separate it from earlier screens;
only matched full runs and fresh confirmation can retain a gain. All new
proposals keep the FP32 output bound and exact KV-cache check.

The positive stable-softmax denominator is finite and at least one. The
bit-linear reciprocal seed has at most 1/8 relative error before refinement;
four FP32 Newton steps follow. An exhaustive NumPy check over all 8,388,608
normalized mantissas observes at most 1.3642e-7 relative error. That host check
is separate from the actual device output checks and raw replay.

See [confirmed evidence](../../../results/reports/2026-10-23-attention-5x.json)
and [24-case raw replay](../../../results/reports/2026-10-23-attention-5x-replay.json).
