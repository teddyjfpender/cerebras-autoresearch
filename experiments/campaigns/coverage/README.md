# Remaining independent baseline coverage

This bounded baseline-only plan covers five independent shapes that remain
unqualified in the current historical catalog coverage report. Each cell must
pass all six cases. Empty candidate lists mean no optimization is attempted;
`no_confirmed_gain` in a completed cell does not mean its baseline failed.
Inspect the referenced baseline receipt for `validated`.

Run this after the active optimization and new reusable-GEMM investigation,
with a new checkpoint under the current evaluator. The plan has not run and
its presence does not increase measured catalog coverage. Preserve failures
and partial receipts; a cancelled or budget-exhausted suite is incomplete.

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/coverage/remaining-independent-plan.json \
  --state runs/campaigns/remaining-independent/result.json --max-seconds 5400
```

Do not run another simulator/journal writer concurrently. Export a terminal
campaign report and verify artifacts before updating coverage totals.
