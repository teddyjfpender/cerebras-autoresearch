# bf16-swiglu-vector-scratch-tail5

Extend the vector/scratch dispatcher with a degree-5 FP32 Taylor exp tail for 4<abs(g)<=8, four repeated squares and stable SiLU division. Keep degree-six central SiLU and native exp outside abs(g)>8. An exhaustive separate BF16 gate screen passed with one quarter of the unchanged local allowance reserved; require full device/fresh validation. This targets costly positive-tail exp fallbacks without weakening precision or allocating new scratch.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims. Reject any alias used while communication is live or any scratch extent exceeding W1; compare complete working-state histories at both seeds.
