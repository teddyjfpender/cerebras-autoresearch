# Decoder vector projections, RoPE frequencies and softmax reuse

Experimental candidate, with no measured performance claim yet.

Keep vectorized projections, replace D=8 RoPE powers with fixed FP32 parameter frequencies, and normalize each attention score once before its reuse across value channels. This removes redundant data-independent powers and repeated divisions while keeping all tokens, nonlinearities and cache states.

The D=8 frequency table follows the fixed RoPE base and head dimension; other
head dimensions retain the original formula. Position-dependent rotations stay
inside the timed token function. Attention probabilities are computed once per
slot/head and reused across value channels. Every cache snapshot remains.

Reject any full output/state failure at either context, changed precision/PE/timing contract, less than 1.05x baseline speedup, memory growth above 1.10x or failed fresh paired confirmation.
