# Trusted benchmark layer

Keep profiles, numerical inputs/oracles, timers, precision, and PE budgets fixed
within a paired attempt. Candidate edits belong in `experiments/candidates/`.
Do not edit trusted drivers to make a candidate pass. A justified contract change
needs documentation, a new fingerprint/profile, new baselines, and gate tests.

Reference and diagnostic tiers cannot pass the independent-oracle promotion
gate. Missing timing remains missing; never substitute host wall time.

Do not alter timestamp/repetition control or move requested work outside the
timed region. Preserve raw output, expected output, bounds, timestamps, and clean
exit requirements. Test failed-output preservation and artifact corruption.

Manifests are configuration, not proof of execution. Update qualification
evidence with actual receipt paths and explicit measured coverage.
