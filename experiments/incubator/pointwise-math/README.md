# FP32 SiLU experiment

Status: the SDK NumPy primitive probe passed; CSL performance and complete-layer
qualification are pending. No standalone performance score is claimed.

The candidate uses a degree-eight Taylor polynomial for `exp(-g/16)` followed
by four squarings, for `|g| <= 8`. Other inputs call the SDK exponential. Every
operation executes at runtime in FP32. The SiLU formula and complete block
accuracy contract are fixed; coefficients are independent of benchmark inputs.

`check_silu_fp32.py` compares 20,001 grid points and 10,000 fixed-seed random
points with FP64 SiLU. It models both fused and unfused Horner arithmetic. Both
passed the primitive target `2e-6 + 2e-6*abs(SiLU)`; the largest absolute error
was `7.035e-7`. This is a numerical probe, not a proof for every FP32 value.

The actual CSL candidates are `mlp-silu-polynomial` and
`decoder-silu-polynomial`. They must pass the original complete six-case oracle,
program-memory and cycle gates, and fresh paired confirmation. The complete
MLP/layer result decides whether this primitive idea is useful. Compare with the
existing confirmed implementation before selecting a successor.
