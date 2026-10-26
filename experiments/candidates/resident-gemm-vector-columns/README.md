# resident-gemm-vector-columns

Replace the short-M callback map with Nt-wide FMAs directly into strided column-major output. Preserve every FP16 K term, four resident frames, two-hop communication, all resets and full snapshots. Longer vectors may reduce dispatch cost; output-bank stride may negate it.

The starting source is the registered resident four-frame GEMM baseline. This is an unmeasured candidate. The oracle, precision, PE budget, sequence/timer controls and snapshots stay fixed. Run the tiny oracle before widening to the two FFN shapes. Score each shape separately and require fresh confirmation before retention.

Falsification: Reject any six-case output/state failure, omitted FP16 K term, changed timer/repeat/PE contract or shutdown failure. Retain only after at least 1.05x matched cycle speedup, at most 1.10x program-memory ratio, and fresh paired confirmation. Cross-shape claims require every resident GEMM cell and the suite worst-cell guard.
