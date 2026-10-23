# Incubator: resident reusable MeshGEMM

**Current status:** the baseline adapter and all three profiles are registered.
Canonical code is in `kernels/dense/reusable-gemm/`; the contract is in
`docs/benchmarks/reusable-gemm.md`. Device qualification is in progress.
The design snapshots and unmeasured candidate ideas below remain historical
preparation material. The reviewed evaluator patch has been applied.

This is the next important benchmark for the twelve-hour session. The original
warmed sequence fails correctness, so it cannot supply a throughput score.
The source is derived from WaferLLM MeshGEMM at
`fd1c2daae37cd68706c03fc8009887ecee9900f8`; its Apache license is included in
`LICENSE.upstream`. Changes are marked in `src/meshgemm.csl`.

## Concrete hypotheses

- `x_shift_step` is initialized in `init_task` and decremented by initial shifts.
  The original repeat entry does not restore it. Thus later repetitions take
  different buffer-swap branches from the first. This is an observed source
  fact and a plausible contributor, not yet a proven complete root cause.
- After P weight exchanges, the working W0 tile may return to its canonical
  permutation. Verify every element after every frame before relying on that
  property to remove weight reloads. Pointer and queue state also matter.
- Conservative restoration of both working buffers can establish a correct
  reference. An optimized reusable implementation could avoid demonstrably
  redundant copies/clears while preserving every output and boundary state.

## Proposed contract

One program and one host launch compute four different resident X matrices with
one shared resident W, at FP16 and a fixed 4x4 PE grid. Canonical input frames
and weights are loaded once before timing. Timed work includes canonical
working-buffer restoration, counter reset, matrix communication/computation,
every output store and every working-weight snapshot. No host reloads or
warmup answer caching are allowed inside or between frames.

The independent oracle must check all four matrix outputs against FP64 products
of quantized inputs under the existing FP16 forward-error bound; verify all
canonical inputs remain bit-exact; and verify every working-W0 snapshot returns
to the specified canonical tile. That last requirement is an explicit proposed
state contract, subject to actual SDK investigation before catalog integration.
A failed hypothesis must remain failed; do not reinterpret its data as proof.

The timer covers the whole sequence, including state instrumentation. Report
sequence cycles and average cycles/GEMM, memory, PE budget, numerical error and
SDK iteration cost. These will not be directly comparable to cold GEMM or a
production throughput path without instrumentation.

Start with M=K=N=16, T=4. Then qualify an FFN-prefill shape. Expose candidate
workspaces only after a correct independent baseline, and require six numerical
families plus fresh paired confirmation before retaining a performance change.

Development stays outside the trusted evaluator while the existing eight-profile
campaign is frozen. Integration requires its own adapter, contract version,
new matched baselines and evidence journal. The four existing tracks stay active.

Pure packing checks passed under SDK NumPy (3 tests): independent global-index
checks cover all input frames, the P=4 weight permutation and all output frames.
These checks launch no simulator and do not qualify the CSL implementation.

## Staged evaluator checks

The proposed adapter, three profiles (tiny and the two scaled FFN-prefill
shapes), oracle and recovery/harness fixes are assembled in the ignored
`runs/staging/reusable-evaluator/` tree. The live evaluator remains unchanged.
That isolated tree passed 69 host tests, nine SDK numerical replay checks and
five SDK packing/state-oracle checks. No device program was launched by those
checks; compilation and numerical qualification remain required.

The host oracle now packs directly with NumPy global-index transformations,
without calling the pinned upstream packing helper. Independent index tests
cover all four frames, the known P=4 weight permutation and every output element.
Negative tests cover the last frame, an intermediate working-weight snapshot
and signed-zero corruption of canonical input. Complete output and state stores
use timed vector copies in the proposed baseline.

Further compute/state attribution and a possible working-buffer storage reuse
are described in [optimization notes](optimization-notes.md). These are
unmeasured hypotheses, subject to the unchanged sequence/state contract.


## Portable next evaluator (not applied)

`next-evaluator.patch` preserves all 26 reviewed integration paths, including
the adapter/oracle, three profiles, baseline CSL, replay support and evidence
integrity fixes. `next-evaluator-manifest.json` binds every old/new file hash
and the F1/F2 evaluator identities. The patch has passed `git apply --check`
against the frozen F1 tree. The staged tree passes 97 host tests (one Linux-only
Mac skip); these do not qualify a device program.

Do not apply while a frozen F1 campaign is active or intended to resume.
Changing the trusted evaluator requires new matched baselines. After final F1
checkpointing, verify every manifest hash, hold the normal simulator lease,
apply and verify the new evaluator identity. Run host and SDK contract checks
before tiny six-case device qualification. Preserve any compiler/state failure;
do not weaken the output or boundary-state contract to pass.

After the registered baseline qualifies, `create_candidates.py` creates three
separate compute/reuse hypotheses, using the committed corrected cold map-K
source. `create_bank_candidate.py` adds the separate bank-layout hypothesis.
Run them from the repository root after integration. They intentionally refuse
an absent adapter and existing candidate names. They have been dry-run in the
isolated staged tree, with protected-control checks; SDK compilation, full
numerical validation, timing and fresh pairing remain outstanding.
