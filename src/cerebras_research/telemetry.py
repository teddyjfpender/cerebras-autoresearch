#!/usr/bin/env python3
"""Measure Qwen 3.8 27B client burst throughput through Cerebras Inference API."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "benchmarks/telemetry/prompts.json"
ENDPOINT = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "qwen-3.8-27b"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    return {
        "api_key_available": bool(os.environ.get("CEREBRAS_API_KEY")),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "prompts_sha256": sha256(PROMPTS),
    }


def request_completion(prompt, max_tokens, timeout):
    key = os.environ["CEREBRAS_API_KEY"]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "none",
        "temperature": 0,
        "seed": 7,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        # Error text can contain request details; never write it to experiment logs.
        raise RuntimeError(f"Cerebras API returned HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Cerebras API request failed: {type(error).__name__}") from error
    elapsed = time.monotonic() - start
    usage = data.get("usage") or {}
    timing = data.get("time_info") or {}
    completion_tokens = usage.get("completion_tokens")
    completion_time = timing.get("completion_time")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        raise RuntimeError("API response omitted positive usage.completion_tokens")
    if not isinstance(completion_time, (int, float)) or completion_time <= 0:
        raise RuntimeError("API response omitted positive time_info.completion_time")
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    content = (choice.get("message") or {}).get("content") or ""
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "server_completion_seconds": completion_time,
        "server_queue_seconds": timing.get("queue_time"),
        "server_prompt_seconds": timing.get("prompt_time"),
        "server_total_seconds": timing.get("total_time"),
        "client_e2e_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_characters": len(content),
        "system_fingerprint": data.get("system_fingerprint"),
    }


def run(label, requests, concurrency, max_tokens, timeout):
    if not os.environ.get("CEREBRAS_API_KEY"):
        raise RuntimeError("CEREBRAS_API_KEY is not set; provide it through the environment")
    prompts = json.loads(PROMPTS.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(x, str) and x for x in prompts)
    ):
        raise ValueError("Prompt file must be a nonempty JSON list of nonempty strings")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-api-{label}-{os.getpid()}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    result = {
        "run_id": run_id,
        "label": label,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "prompts_sha256": sha256(PROMPTS),
        "request_count": requests,
        "concurrency": concurrency,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": "none",
        "temperature": 0,
        "seed": 7,
        "status": "running",
        "responses": [],
    }
    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(request_completion, prompts[i % len(prompts)], max_tokens, timeout)
                for i in range(requests)
            ]
            for future in futures:
                result["responses"].append(future.result())
        result["client_span_seconds"] = time.monotonic() - started
        tokens = sum(x["completion_tokens"] for x in result["responses"])
        server_seconds = sum(x["server_completion_seconds"] for x in result["responses"])
        result["total_completion_tokens"] = tokens
        result["server_decode_tps"] = tokens / server_seconds
        result["client_burst_output_tps"] = tokens / result["client_span_seconds"]
        result["median_client_e2e_seconds"] = statistics.median(
            x["client_e2e_seconds"] for x in result["responses"]
        )
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return run_dir / "result.json"


def compare(baseline_path, candidate_path):
    base = json.loads(baseline_path.read_text())
    test = json.loads(candidate_path.read_text())
    if base["status"] != "complete" or test["status"] != "complete":
        raise ValueError("Both API runs must be complete")
    if base["label"] != "baseline" or test["label"] != "candidate":
        raise ValueError("Expected baseline and candidate labels")
    for field in (
        "model",
        "endpoint",
        "prompts_sha256",
        "request_count",
        "max_completion_tokens",
        "reasoning_effort",
        "temperature",
        "seed",
    ):
        if base[field] != test[field]:
            raise ValueError(f"Incompatible API run field: {field}")
    server_speedup = test["server_decode_tps"] / base["server_decode_tps"]
    client_speedup = test["client_burst_output_tps"] / base["client_burst_output_tps"]
    fingerprints = sorted(
        {str(x.get("system_fingerprint")) for run in (base, test) for x in run["responses"]}
    )
    same_outputs = [x.get("content_sha256") for x in base["responses"]] == [
        x.get("content_sha256") for x in test["responses"]
    ]
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_concurrency": base["concurrency"],
        "candidate_concurrency": test["concurrency"],
        "server_decode_tps_speedup": server_speedup,
        "client_burst_output_tps_speedup": client_speedup,
        "system_fingerprints": fingerprints,
        "identical_output_hashes": same_outputs,
        "minimum_sample_count_met": base["request_count"] >= 5,
        "client_speedup_gate": (
            base["request_count"] >= 5
            and test["concurrency"] > base["concurrency"]
            and client_speedup >= 1.15
        ),
        "model_decode_tps_gain_claim_supported": False,
    }


def search(requests, candidate_concurrency, max_tokens, timeout, cooldown_seconds):
    baseline = run("baseline", requests, 1, max_tokens, timeout)
    if cooldown_seconds:
        time.sleep(cooldown_seconds)
    candidate = run("candidate", requests, candidate_concurrency, max_tokens, timeout)
    result = compare(baseline, candidate)
    summary = candidate.parent / "comparison.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("label", choices=("baseline", "candidate"))
    run_parser.add_argument("--requests", type=int, default=2)
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--max-completion-tokens", type=int, default=256)
    run_parser.add_argument("--timeout", type=int, default=60)
    search_parser = sub.add_parser("search", help="Run serial baseline and concurrent candidate")
    search_parser.add_argument("--requests", type=int, default=2)
    search_parser.add_argument("--candidate-concurrency", type=int, default=2)
    search_parser.add_argument("--max-completion-tokens", type=int, default=256)
    search_parser.add_argument("--timeout", type=int, default=60)
    search_parser.add_argument("--cooldown-seconds", type=int, default=0)
    score = sub.add_parser("compare")
    score.add_argument("baseline", type=Path)
    score.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        info = preflight()
        print(json.dumps(info, indent=2))
        return 0 if info["api_key_available"] else 2
    if args.command == "compare":
        print(json.dumps(compare(args.baseline, args.candidate), indent=2))
        return 0
    if args.command == "search":
        if (
            args.requests < 2
            or args.candidate_concurrency < 2
            or args.candidate_concurrency > args.requests
            or args.max_completion_tokens < 1
            or args.timeout < 1
            or args.cooldown_seconds < 0
        ):
            parser.error("search needs at least two requests and candidate concurrency 2..requests")
        print(
            search(
                args.requests,
                args.candidate_concurrency,
                args.max_completion_tokens,
                args.timeout,
                args.cooldown_seconds,
            )
        )
        return 0
    if (
        args.requests < 1
        or args.concurrency < 1
        or args.concurrency > args.requests
        or args.max_completion_tokens < 1
        or args.timeout < 1
    ):
        parser.error("requests, concurrency, max-completion-tokens, and timeout must be positive")
    print(
        run(args.label, args.requests, args.concurrency, args.max_completion_tokens, args.timeout)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
