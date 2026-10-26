# Resident SwiGLU chain

The staged contract, oracle, driver and baseline have been integrated under
[kernels/composed/resident-swiglu](../../../kernels/composed/resident-swiglu/README.md),
with the canonical [benchmark contract](../../../docs/benchmarks/resident-swiglu.md).
Both initial cells now passed full numerical/core baseline qualification. This directory preserves design staging
and independent primitive probes; it does not supply a qualified kernel score.

`primitive_silu_probe.py` is a math-only FP32 polynomial screen across every
finite FP16 gate in[-4,4], with a prescribed0.00025 relative-error gate. Degrees9
and10 passed that host arithmetic screen; degree8 and below failed. This is not
an SDK arithmetic proof or a complete-chain validation. Any future candidate
must retain the canonical precision/bounds and pass full fresh paired evidence.
