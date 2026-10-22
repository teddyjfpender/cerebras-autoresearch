# decoder-vector-projections

Initial profile: `decoder.h16-f32-ctx4`. This parameterized candidate may be
measured on other compatible profiles; it has no cross-shape performance claim.

Vector accumulation across output channels reduces scalar projection loop overhead; retain RMSNorm, nonlinear operations, all token outputs and cache snapshots. Additional descriptors may increase memory.

Reject any full-output/state numerical failure, changed timed work or PE budget, speedup below 1.05x, memory growth above 1.10x, or failed fresh-seed confirmation.

All numerical drivers and time/sequence controls remain fixed. See the
[four-track plan](../../campaigns/inference-blocks/README.md) for a finite
campaign with the normal full-case and fresh-confirmation gates.

Status: experimental implementation; qualification and optimization evidence
are reported separately in `results/reports/`.
