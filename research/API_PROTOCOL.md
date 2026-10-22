# Optional hosted API telemetry protocol

Objective: increase **client-observed burst aggregate output TPS** for the
hosted Cerebras `qwen-3.8-27b` API, using the same requests and prompt set.
Record the API's server decode TPS separately. Do not describe a concurrency
gain as a faster model kernel or single-request decode rate.

1. Run `python3 tools/inference_benchmark.py preflight`. A Cerebras API key in
   `CEREBRAS_API_KEY` is required. Do not commit keys or response text.
2. Read the live account limits. Run a serial baseline with a fixed request
   count and token cap. Save its result path.
3. Change only the client scheduling parameter, beginning with concurrency 2.
   Use the same request count, prompts, seed, model, and completion cap.
4. Compare runs with `tools/inference_benchmark.py compare BASE CANDIDATE`.
   Reject HTTP errors, missing timing metadata, or changed request settings.
5. Retain a setting only after at least five successful requests per arm and
   at least 15% higher measured client burst aggregate TPS. Re-run the winning
   setting with fresh requests. Record paths and limits in `EXPERIMENTS.md`.
6. To claim sustained TPS, test over a window that includes rate-limit refill;
   short bursts do not establish sustained capacity.

The public CSL SDK cannot patch Cerebras's hosted inference runtime. Do not
edit or benchmark standalone CSL examples as evidence of hosted Qwen TPS.
