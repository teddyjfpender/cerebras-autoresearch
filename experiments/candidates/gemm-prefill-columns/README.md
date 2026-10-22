# gemm-prefill-columns

Initial profile: `meshgemm.cold-prefill-ffn-up-scaled`. This parameterized candidate may be
measured on other compatible profiles; it has no cross-shape performance claim.

Use direct column FMA dispatch instead of a map callback on larger FFN tiles while preserving DSR progression and all communication. The larger tile may amortize dispatch differently than tiny GEMM.

Reject numerical failures, speedup below 1.05x, memory growth above 1.10x or failed fresh paired confirmation.

All numerical drivers and time/sequence controls remain fixed. See the
[four-track plan](../../campaigns/inference-blocks/README.md) for a finite
campaign with the normal full-case and fresh-confirmation gates.

Status: experimental implementation; qualification and optimization evidence
are reported separately in `results/reports/`.

Source is derived from [WaferLLM](https://github.com/MeshInfra/WaferLLM), pinned
at `fd1c2daae37cd68706c03fc8009887ecee9900f8`; see `LICENSE.upstream`.
