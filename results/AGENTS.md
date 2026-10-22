# Evidence handling

Historical snapshots are immutable records of their original contracts. Do not
rewrite their cycle metric or numerical results during a repository migration.
Add new reports with new paths. Raw outputs and SDK binaries stay outside Git.

Clearly distinguish configured, screened, validated, reference-only, diagnostic,
failed, and confirmed. Include raw receipt paths and digests for retained results.
No simulator result alone establishes model TPS, hardware energy, or hosted speed.

`results.tsv` is an append-only event journal, not a list of winners. The runner
and decision CLI write it. Do not alter or reorder accepted rows. Never convert
screens, references, failures, or provisional measurements into confirmations.
`leaderboard.json` is derived exclusively from the TSV; regenerate through
`research.py ledger board`, then verify. Git prefix checks anchor accepted rows.
The local SHA chain is unsigned and does not authenticate a public judge.
