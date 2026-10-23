# Resident GEMM research session

Authorized window: **2026-10-23 07:45:41 UTC to 20:45:41 UTC** (13 hours).
The live checkpoint is `runs/campaigns/20261023-resident-gemm-thirteen-hour/session.json`.

First finish all three resident GEMM baselines, six independent cases each, with
full output/state/timestamp replay and artifact checks. Then screen candidate
workspaces, fully validate eligible changes, and require fresh paired seeds
before retention. Every terminal V2 attempt and decision is appended by the
runner to `results/results.tsv`; partial or cancelled runs cannot qualify.

The selected host route is versioned in `config/sdk-platform.json`. The same-
binary platform screen preserved identical numerical/state bytes but found up
to four cycles of cross-host difference. It failed the old timing-equivalence
gate, so new baselines are required on the distinct selected evaluator. Host
turnaround improvements are separate from kernel-cycle improvements. See
`results/reports/2026-10-23-sdk-rosetta-platform-screen.json` for the diagnostic.

Initial approaches are direct column vectors, contiguous row accumulation with
full vector packing, four-row expansion, and resident boundary-weight reuse.
Each workspace has concrete hypothesis/falsification and attribution fields.
The timer, workload, FP16 precision, PE budget, output collection and exact
per-frame working-weight state requirements stay fixed. Cross-shape claims need
complete suite coverage and worst-cell guards. No model or hardware TPS follows.

At the deadline, stop starting work, preserve any interrupted attempt, reconcile
and verify the TSV, and export compact evidence for completed results. Historical
receipts and raw evidence remain immutable.

The first optimization campaign completed with one confirmed profile gain:
FFN-up 95,611 -> 75,962 cycles, +3.54% program memory, all six cases at both
seeds. Tiny and FFN-down rejected all four initial approaches for insufficient
speed. Every terminal attempt/decision is journaled (216 events at this checkpoint).


At the 13:30 UTC checkpoint, Resident GEMM's full three-shape suite is confirmed
at1.146754x conservative geometric mean, FP16 resident SwiGLU's two shapes at
1.415571x, and BF16 resident SwiGLU's separate four-shape suite at1.589946x.
Their72/48/96 paired cases passed complete numerical/core/timer replay. Every
terminal attempt and decision is journaled;402 events passed artifact and Git
prefix verification. These simulator tracks have distinct contracts/evaluators.

The next registered challenge is resident recurrent gated-delta attention:
scaled K16/K32, V16, three value heads per key head, four tokens, FP32 state and
BF16 Q/K/V/output. Full state/prefix histories, nonzero initial state and all
physical immutable inputs are checked. Device baselines and optimization remain
to be run. A native K128/V128 one-group follow-up is staged separately. The
original deadline remains20:45:41 UTC; no model or hardware TPS is measured.
