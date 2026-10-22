"""Capture the evaluator and declared research context before an SDK attempt."""

import json
import os
import subprocess

from .paths import ROOT, digest, identity


def capture(profile, variant, candidate=None):
    files = sorted((ROOT / "src/cerebras_research").glob("*.py")) + sorted(
        (ROOT / "benchmarks/drivers").glob("*.py")
    )
    evaluator = {str(p.relative_to(ROOT)): digest(p) for p in files}

    def git(*args):
        return subprocess.check_output(
            ["git", "--no-optional-locks", *args], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=15
        )

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        commit, dirty = "unknown", True
    hypotheses = {
        "baseline": "Measure the pinned reference workload.",
        "unroll": "Remove local FMA map callback overhead through literal expansion.",
        "bank-alignment": "Reduce local bank conflicts through aligned, padded storage.",
        "fused": "Remove intermediate vector stores through per-output epilogue fusion.",
    }
    research = {
        "approach": profile["adapter"] + "." + variant,
        "candidate": "",
        "hypothesis": hypotheses.get(variant, ""),
        "falsification": "Reject numerical or footprint failures, absent cycle gain, or memory-budget overrun.",
    }
    manifest = {}
    if candidate:
        manifest = json.loads((candidate / "candidate.json").read_text())
        for name in ("approach", "hypothesis", "falsification"):
            value = manifest.get(name, "")
            if not isinstance(value, str) or not value.strip() or value.startswith("TODO"):
                raise ValueError(f"Candidate needs a concrete {name} in candidate.json")
            research[name] = value
        research.update(
            candidate=candidate.name, manifest_sha256=digest(candidate / "candidate.json")
        )
    for name, variable in (("author", "AUTHOR"), ("model", "MODEL"), ("agent_harness", "HARNESS")):
        research[name] = (
            os.environ.get("CEREBRAS_RESEARCH_" + variable) or manifest.get(name) or "unknown"
        )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "evaluator_sha256": identity(evaluator),
        "evaluator_files": evaluator,
    }, research
