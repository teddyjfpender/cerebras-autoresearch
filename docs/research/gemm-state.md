# Repeated GEMM state: integration finding

At the pinned WaferLLM revision, the initial independent check of the timed
warmup/repeat sequence failed at `P=4, M=K=N=16`, FP16. The program compiled and
ran, but the collected final output violated the fixed numerical bound.

| Observation | Evidence |
| --- | --- |
| Warmup 1, timed repeats 5 | `runs/v2/20261022T104201Z-meshgemm.tiny-4ae03887/result.json` |
| Positive-input maximum absolute error | 1.2340296134352684 |
| Maximum error / allowed bound | 35.73948480273888 |
| Raw failing inputs/output/timestamps | That run's `work/out/case-positive.npz` |
| One cold invocation, unchanged bound | `runs/v2/20261022T104827Z-meshgemm.cold-tiny-41411450/result.json` |
| Full six-case cold check | `runs/v2/20261022T105637Z-meshgemm.cold-tiny-ea193fc5/result.json` |

The original host launcher checks the output of an initial one-shot GEMM before
performing its warmed timing sequence. That earlier output check does not
establish correctness of the final warmed invocation. The failure above is an
observation in this SDK/simulator configuration, not a proof of its root cause.

The cold profiles therefore measure one complete invocation from canonical
resident input state. They are not a faster version of the repeated workload
and make no warmed-throughput claim. Profile names and timing protocols expose
this distinction.

## Next research gate

Add a separate reusable-state benchmark: distinct inputs across repetitions,
complete output/state collection after each, fixed precision bounds, and a
fresh sequence confirmation. Audit buffer rotations, shifts, initialization,
and repeat counters. Restore semantically correct reuse before treating its
cycle measurements as a steady-state GEMM score.
