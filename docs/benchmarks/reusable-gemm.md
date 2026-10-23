# Resident reusable GEMM

The adapter is registered under its own evaluator track. Device compilation and
six-case qualification are in progress. Configuration and Python packing tests
are not execution proof.

## Workload

A fixed 4x4 compute grid evaluates four independently supplied resident FP16
matrices against one shared resident FP16 weight matrix:

`Y[t] = X[t] @ W`, for `t = 0..3`.

| Profile | T | M | K | N | Compute PEs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reusable-gemm.tiny-sequence` | 4 | 16 | 16 | 16 | 16 |
| `reusable-gemm.prefill-ffn-up-scaled` | 4 | 16 | 80 | 272 | 16 |
| `reusable-gemm.prefill-ffn-down-scaled` | 4 | 16 | 272 | 80 | 16 |

All frames and canonical weights are loaded before one `sequence_host` launch.
There are no host reloads between frames. The reference restores working tiles
and shift counters before each frame. It preserves the pinned two-hop matrix
communication and FP16 accumulation, then captures every output and the entire
working W0 tile at every boundary.

The source derives from WaferLLM MeshGEMM at
`fd1c2daae37cd68706c03fc8009887ecee9900f8`, with `LICENSE.upstream` included.
The original pinned checkout and cold one-invocation profiles stay separate.

## Numerical and state contract

`resident_gemm_sequence_fp16_v1` uses FP64 products of exactly quantized inputs.
For each output, let `l = K + 4`, `u = 2^-11` and
`gamma = l*u / (1-l*u)`. The fixed error bound is:

`abs(actual - expected) <= gamma * (abs(X) @ abs(W)) + l*2^-24`.

Profiles require `l*u < 1`. No candidate can weaken this bound or silently change
precision, output coverage, timing or the PE budget.

State checks require exact FP16 bits:

- Every canonical input frame and the canonical weights survive unchanged.
- Every frame's working-W0 snapshot equals its canonical packed tile.
- All four matrix outputs are checked independently, including early frames.

The working-weight requirement makes boundary restoration observable. A solver
may remove a reset/copy only if every output and boundary state still passes.
The NumPy packing implementation is separate from the upstream helper. Independent
global-index tests cover all input/output frames and the P=4 weight permutation.

## Timing and metrics

The timer covers the complete four-frame sequence: working-buffer restoration,
shift-counter reset, all computation/communication, every output store and every
working-weight snapshot. Initial H2D transfers and post-run D2H checks are outside.
`sequence_host`, `save_frame`, timestamp conversion, `init_task` and exit/
repeat controls are protected against candidate edits.

The primary score is `max_local_cycles`, with 0 warmups and 1 timed sequence.
`cycles_per_gemm` divides that score by four. Work units are `2*T*M*K*N` arithmetic
operations. Program-object memory, PE-cycle cost, imbalance, numerical errors and
SDK step wall time are also recorded.

These instrumented sequence scores have their own profile identities. They are
not directly comparable to a cold GEMM score, steady-state hardware throughput,
Qwen TPS or the hosted inference service.

## Workflow after integration

```sh
make check
make sdk-test
tools/sdk-run python3 research.py run reusable-gemm.tiny-sequence \
  --variant baseline --seed 20261027 --max-seconds 1200
python3 research.py candidate-init reusable-gemm.tiny-sequence YOUR_CANDIDATE
# Complete candidate.json and edit the candidate's CSL.
tools/sdk-run python3 research.py challenge reusable-gemm.tiny-sequence \
  --variant candidate --candidate YOUR_CANDIDATE --seed 20261027 --max-seconds 2400
```

Start with the tiny full oracle, then qualify the two FFN shapes. Retention
requires all six numerical families, a matched source/harness/SDK/evaluator,
at least 1.05x speedup, at most 1.10x program-memory growth and a fresh paired
seed. Every terminal attempt and decision enters the append-only TSV; preserve
raw output/state/timestamps and verify artifacts before widening a claim.


## Readback transport

After the timer ends, the driver retrieves all outputs, working-weight history
and canonical inputs. A full-grid history read of 87,040 half values stalled in
the simulator during scaled-profile bringup and hit the existing 1,200-second
step limit. That attempt is failed and remains in the journal; diagnostic output
inspection does not qualify it.

The readback driver now groups rows into replies of at most 32,768 transport
words. This is a conservative software bound, not a claimed hardware/API limit.
Every PE word is reconstructed in its original order, and the entire state
oracle is unchanged. Because the trusted driver changed, qualification requires
new evaluator-bound baselines. The timed CSL program is unchanged.
