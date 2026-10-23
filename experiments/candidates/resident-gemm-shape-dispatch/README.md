# Aspect-ratio dispatch

Use contiguous Nt-wide row accumulation with hoisted weight DSR only for expansion tiles with Nt>=32 and Nt>Kt; use the original Mt-wide column map for shorter or contraction tiles. Freeze the threshold before additional shapes. Combine both with frame residency justified by complete weight restoration, while preserving exact FP16 term order, outputs, all snapshots and fixed controls.

Reject any output/state/shutdown failure or missing arithmetic term. Each cell needs full six-case and fresh paired seeds. Profile retention requires1.05x speedup and1.10x memory; suite retention additionally requires complete coverage, geometric mean1.05x and fixed worst-cell regression guards. Do not claim every cell individually retained if only the suite passes.
