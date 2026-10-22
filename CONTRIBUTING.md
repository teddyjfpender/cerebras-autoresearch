# Contributing experiments

Use Python 3.10+ for the host control plane. Run from the checkout with
`research.py`, or `python3 -m pip install -e .` for the console entry point.
Install development tooling with `python3 -m pip install -r requirements-dev.txt`
or use the pinned `uvx` commands in `make check`.

Before finishing a change:

1. Run `make check` (or `make RUFF=ruff check` after a pip install).
2. For numerical/timing driver changes, run `make sdk-test` and a representative
   full SDK benchmark; rebaseline the affected comparison arms.
3. For a new adapter, run qualification on the relevant family. Record failures
   and distinguish configured, screened, numerically validated, and confirmed.
4. Inspect receipts, numerical error bounds, resource observations, and source
   identities before retaining an optimization.
5. Export a compact result report and reference raw artifact paths. Keep SDK
   binaries, credentials, weights, and generated simulator logs out of Git.
6. Run `python3 research.py ledger verify --artifacts`. Check the automatic TSV
   event and derived board; include appended rows when committing an experiment.
   CI rejects changes to the ledger prefix from the base commit. Read the
   [ledger contract](docs/experiments/ledger.md) before importing or recovering runs.

Keep kernel hypotheses in `experiments/candidates/NAME/candidate.json`, with
editable CSL beside them. Add independent correctness evidence before widening
precision, shape, layout, or model claims. Baseline checkouts in `.deps/` remain
pristine. Existing raw records and archived snapshots preserve their original
contract; do not revise them to match a new score.

The SDK-free CI workflow requires no credentials or container. Local qualification
requires the licensed SDK and may take tens of minutes under QEMU. `--max-seconds`
bounds a run/suite/challenge; partial receipts survive budget exhaustion.

Read the [CSL bringup findings](docs/development/csl-pitfalls.md) when debugging
export declarations, descriptor pointer types or state across repeated calls.
Correct a frozen candidate in a new workspace, preserving the failed receipt.
