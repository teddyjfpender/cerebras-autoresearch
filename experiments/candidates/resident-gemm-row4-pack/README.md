# resident-gemm-row4-pack

Expand the four independent row FMAs at each K step for Mt=4 while preserving a generic fallback. Reduce loop/index instructions atop contiguous column accumulation. Full timed output packing and all weight-state snapshots remain, so repeated-frame correctness can falsify transfer of the cold-kernel idea.

The starting source is the registered resident four-frame GEMM baseline. This is an unmeasured candidate. The oracle, precision, PE budget, sequence/timer controls and snapshots stay fixed. Run the tiny oracle before widening to the two FFN shapes. Score each shape separately and require fresh confirmation before retention.

Falsification: Reject any six-case output/state failure, omitted FP16 K term, changed timer/repeat/PE contract or shutdown failure. Retain only after at least 1.05x matched cycle speedup, at most 1.10x program-memory ratio, and fresh paired confirmation. Cross-shape claims require every resident GEMM cell and the suite worst-cell guard.
