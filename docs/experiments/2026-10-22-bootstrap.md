# Experiment ledger

| Date | Change | Baseline run | Candidate run | Result |
| --- | --- | --- | --- | --- |
| 2026-10-11 | API concurrency 1 to 2 planned | Blocked: no `CEREBRAS_API_KEY` | Blocked: no `CEREBRAS_API_KEY` | No measured TPS increase |

## Kernel simulator route reopened — 2026-10-22

- Verified official SDK 2.10 examples at `06ffd5961e8d35b6324e0ebf6595fe4a5a87c2e1`.
- Inspected WaferLLM at `fd1c2daae37cd68706c03fc8009887ecee9900f8`:
  modifiable inference kernels and local simulator launchers are available.
- Current objective is simulator kernel research; API concurrency is secondary.
- SDK 2.10.1 is installed and all bundled checksums passed. SIF SHA256:
  `fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d`.
- Local environment: Lima 2.2.1, QEMU 11.1.2, Ubuntu 22.04 x86_64,
  SingularityCE 4.0.2, WSE-3 simulator, 4 virtual CPUs and 8 GiB RAM.
- The official numerical smoke passed:
  `runs/20261022T091254Z-sdk-smoke-3322/result.json`.

## Initial kernel search

All cells use FP16, a 4×4 PE grid, group size 2, one warmup and ten timed
repeats. The metric below is the **upstream maximum simulated cycle span**.
It includes local computation and reduction, using upstream timestamp alignment;
it has not been calibrated as a hardware makespan. Host wall time is not scored.

| Shape M×N | Variant | Cycles | Validation / decision | Run under `runs/` |
| --- | --- | ---: | --- | --- |
| 64×64 | Baseline, original upstream driver | 264.7 | Upstream positive-input check only | `20261022T091422Z-meshgemv-3546` |
| 64×64 | Baseline, independent oracle | 264.7 | All six cases passed | `20261022T091855Z-meshgemv-3927` |
| 64×64 | Bank alignment, prototype 1 | — | Compiler rejected global pointer cast | `20261022T092044Z-meshgemv-4167` |
| 64×64 | Bank alignment, prototype 2 | — | Compiler rejected struct-field tensor expression | `20261022T092402Z-meshgemv-4298` |
| 64×64 | Bank alignment | 264.7 | Six cases passed; rejected, no gain | `20261022T092525Z-meshgemv-4426` |
| 80×272 | Baseline | 648.7 | All six cases passed | `20261022T092800Z-meshgemv-4668` |
| 80×272 | Bank alignment | 659.7 | Six cases passed; rejected, slower | `20261022T093144Z-meshgemv-4961` |
| 64×64 | Unroll, screen | 241.3 | Positive only; provisional | `20261022T093610Z-meshgemv-5199` |
| 80×272 | Unroll, screen | 622.3 | Positive only; below 1.05× retention gate | `20261022T094054Z-meshgemv-5497` |
| 64×64 | Unroll, first full attempt | — | Six checks printed PASS, then shutdown hung; run failed | `20261022T094131Z-meshgemv-5648` |
| 64×64 | Fresh baseline | 264.7 | Six cases passed and simulator exited | `20261022T094556Z-meshgemv-6083` |
| 64×64 | Unroll, full retry | — | Numerical checks passed; shutdown fault rejected by log guard | `20261022T094713Z-meshgemv-6325` |
| 64×64 | Alignment, automatic challenge | 264.7 | Screen had no gain; automatically rejected | `20261022T094542Z-challenge-6083` |
| 64×64 | Isolated-case baseline, five host threads | — | Last case passed numerically, then shutdown fault; failed | `20261022T095135Z-meshgemv-6802` |
| 64×64 | Baseline, one host thread | 264.7 | Six isolated cases passed and exited cleanly | `20261022T095833Z-meshgemv-7563` |
| 64×64 | Unroll, one host thread | 241.3 | Six isolated cases passed; initial matched pair | `20261022T100338Z-meshgemv-7563` |
| 64×64 | Baseline, fresh seed 20261023 | 264.7 | Six isolated cases passed and exited cleanly | `20261022T100715Z-meshgemv-7563` |
| 64×64 | Unroll, fresh seed 20261023 | 241.3 | Six isolated cases passed; gain confirmed for this cell | `20261022T101100Z-meshgemv-7563` |

The 80×272 cell preserves the Qwen FFN projection aspect ratio at 1/64 scale.
It is a small kernel problem. No full Qwen decoder, model TPS gain, or hosted
service speedup has been established.

### Setup faults retained in the ledger

- Git status checks timed out before compilation; one captured example is:
  `runs/20261022T094318Z-challenge-5998/result.json`.
  Pristine-source verification now checks pinned Git blobs directly, avoiding
  cross-platform index refreshes on the shared filesystem.
- The failed simulator shutdown emitted repeated asynchronous-stop messages.
  Its original log grew to several GiB; bounded head/tail diagnostics and the
  original byte count are preserved in that run's `sim.shutdown-*` files.
  The duplicate log was truncated after killing the process. New commands
  reject and kill runs exceeding 64 MiB per log or 600 seconds per step.
- Numerical outputs and raw timestamps are preserved separately as `.npz`
  artifacts. The comparison command verifies their hashes. A shutdown failure
  cannot pass the full-run gate even if every numerical check printed PASS.
- The current full suite launches one fresh simulator process per case and
  requires each to exit cleanly. This tests an alternative to reusing a session
  across all six inputs. Both arms use identical process boundaries; results
  from the old driver cannot pass comparison with the new driver.
- Case isolation alone did not prevent a baseline shutdown failure. The SDK
  supports `simfab_numthreads`; the current driver explicitly uses one host
  simulator thread, with tracing disabled. This leaves the 4×4 simulated PE
  grid fixed. Both arms must use the same driver and simulator configuration.
- For the inspected physical PE `(4,1)`, `cs_readelf --msize` reported resident
  memory of 6,624 bytes for the baseline and 6,720 bytes for unroll: +96 bytes.
  Raw reports are saved as `representative-pe-memory.txt` in the two runs above.
  This is one inspected PE, not a complete fabric allocation report.

### Retention rule

During exploratory setup, the local gate was set to 1.05× cycle speedup for
each shape, six numerical cases, unchanged toolchain/workload/oracle, and a
fresh paired confirmation with the next seed. This is a kernel experiment
rule. Screens never qualify, and a gain in one cell does not establish a gain
in another. A model TPS claim still requires Qwen-specific integration.

## Confirmed outcome

`runs/20261022T095821Z-challenge-7563/result.json` reports
`confirmed_kernel_improvement`: unroll reduced the maximum simulated span
from **264.7 to 241.3 cycles**, or **8.84% fewer cycles**, for the 64×64 cell.
Both paired seeds passed all six numerical cases. All 25 simulator processes
in that challenge exited cleanly with one host simulator thread.

Retain `unroll` for this cell. The 80×272 screen did not meet the gate, and bank
alignment was rejected. No full Qwen TPS improvement has been established.

A compact tracked snapshot, including source/verifier hashes and raw result
digests, is in [../../results/archive/2026-10-22.json](../../results/archive/2026-10-22.json).
The SDK binaries, compiler outputs, raw arrays, and logs remain local and ignored.
