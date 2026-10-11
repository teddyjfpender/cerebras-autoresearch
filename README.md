# Cerebras Qwen 3.8 Inference API autoresearch

This repository measures the served `qwen-3.8-27b` model through the
[Cerebras Inference API](https://inference-docs.cerebras.ai/api-reference/chat-completions)
and searches for higher **client-observed aggregate output TPS** by changing
request concurrency. It separately records server decode TPS from the API's
`usage.completion_tokens` and `time_info.completion_time` fields.

The [Cerebras CSL SDK](https://sdk.cerebras.ai/) runs standalone wafer scale
programs. It does not let this repository replace kernels inside Cerebras's
hosted Qwen inference service. This repo therefore makes no claim to increase
the model's single-request decode TPS. API request scheduling can increase burst
aggregate TPS if account rate limits and server capacity permit it.

## Run

Set `CEREBRAS_API_KEY` in the process environment. Keep it out of this repo.
The benchmark uses the same two code-generation prompts, model ID, seed,
reasoning mode, and completion cap in both arms. Response text is not stored.

```sh
python3 tools/inference_benchmark.py preflight
python3 tools/inference_benchmark.py search
```

`search` runs a serial baseline followed by a concurrency-2 candidate, then
prints the comparison path. For a manual experiment:

```sh
python3 tools/inference_benchmark.py run baseline --requests 2 --concurrency 1
python3 tools/inference_benchmark.py run candidate --requests 2 --concurrency 2
python3 tools/inference_benchmark.py compare runs/<baseline-id>/result.json runs/<candidate-id>/result.json
```

Each `run` prints the result path. Increase `--requests` after checking your
[account limits](https://inference-docs.cerebras.ai/support/rate-limits). The
[Qwen 3.8 model page](https://inference-docs.cerebras.ai/models/qwen-3.8-27b)
currently lists five requests per minute for Free Trial and 300 per minute
for Developer accounts. The default two requests per arm stay within one
free-tier minute, but are only a smoke test. At least five requests per arm
are required by the comparison gate, so space the runs to respect your limit.
For `search` with five requests per arm on the Free Trial tier, pass
`--cooldown-seconds 65` between arms.
Use `--max-completion-tokens` to control request cost and quota reservation.

The result file records the prompt hash, request parameters, server decode TPS,
client burst aggregate TPS, per-request timings, output hashes, and model
fingerprints. A comparison can support a **client scheduling** result; it cannot
prove a Cerebras kernel or model speedup. Sustained TPS requires a longer run
that includes rate-limit refill and 429 behavior.

## Scope and provenance

The API model ID, timing fields, and authentication come from the official
[Chat Completions reference](https://inference-docs.cerebras.ai/api-reference/chat-completions).
The fixed prompt set is in `model/api-prompts.json`. [AGENTS.md](AGENTS.md)
defines the experiment protocol. No SDK or production serving code is included.
