"""Append-only local TSV journal and derived, explicitly scoped research views.

SHA-256 chaining detects accidental corruption. It is not judge authentication.
Git prefix checks anchor accepted history; raw receipt replay checks evidence.
"""

import argparse
import contextlib
import fcntl
import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path

from .paths import ROOT, contained, digest, identity, tree_digest, write_json

FIELDS = (
    "sequence",
    "event_id",
    "recorded_utc",
    "kind",
    "origin",
    "outcome",
    "reported_outcome",
    "evidence_audit",
    "decision",
    "profile",
    "family",
    "approach",
    "candidate",
    "variant",
    "seed",
    "case_count",
    "primary_metric",
    "primary_value",
    "memory_bytes",
    "max_error_fraction",
    "speedup",
    "memory_ratio",
    "git_commit",
    "git_dirty",
    "evaluator_sha256",
    "profile_sha256",
    "baseline_tree_sha256",
    "source_sha256",
    "harness_sha256",
    "sdk_sha256",
    "author",
    "model",
    "agent_harness",
    "hypothesis",
    "falsification",
    "note",
    "receipt",
    "receipt_sha256",
    "previous_sha256",
    "row_sha256",
)
HEADER = "\t".join(FIELDS) + "\n"
ZERO = "0" * 64
LIVE = {"running", "preparing"}


def cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite ledger number")
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def read_rows(path=None):
    path = Path(path or ROOT / "results/results.tsv")
    if not path.exists():
        return []
    raw = path.read_text()
    if not raw.startswith(HEADER) or not raw.endswith("\n"):
        raise ValueError("Ledger header or final newline changed")
    rows, seen, previous = [], set(), ZERO
    for number, line in enumerate(raw[len(HEADER) :].splitlines(), 1):
        cells = line.split("\t")
        if len(cells) != len(FIELDS):
            raise ValueError(f"Malformed TSV row {number}")
        row = dict(zip(FIELDS, cells))
        payload = {k: row[k] for k in FIELDS if k != "row_sha256"}
        if row["sequence"] != str(number) or row["previous_sha256"] != previous:
            raise ValueError(f"Ledger sequence/chain changed at row {number}")
        if identity(payload) != row["row_sha256"] or row["event_id"] in seen:
            raise ValueError(f"Ledger digest/duplicate event at row {number}")
        if row["event_id"] != identity({"kind": row["kind"], "receipt": row["receipt"]}):
            raise ValueError(f"Ledger event identity changed at row {number}")
        seen.add(row["event_id"])
        previous = row["row_sha256"]
        rows.append(row)
    return rows


def kind_of(data):
    if data.get("record_type") in (
        "comparison",
        "suite_comparison",
        "confirmation",
        "suite_confirmation",
    ):
        return data["record_type"]
    if isinstance(data.get("profile"), dict):
        return "run"
    if isinstance(data.get("runs"), list):
        return "challenge"
    if isinstance(data.get("runs"), dict):
        return "suite"
    raise ValueError("Unrecognized experiment receipt")


def resolve(path):
    return contained(ROOT, path)


def replay(data, kind):
    """Recompute decisions using the same gates; never trust a 'confirmed' label."""
    from .scoring import compare, confirm

    inputs = data.get("inputs", {})
    budgets = data.get("budgets", {})
    if kind == "comparison":
        actual = compare(resolve(inputs["baseline"]), resolve(inputs["candidate"]), **budgets)
    elif kind == "confirmation" or data.get("status") == "confirmed_improvement":
        initial, fresh = data["initial"], data["confirmation"]
        if not budgets:
            budgets = {k: initial[k] for k in ("minimum_speedup", "max_memory_growth")}
        paths = (
            (
                inputs[k]
                for k in (
                    "initial_baseline",
                    "initial_candidate",
                    "fresh_baseline",
                    "fresh_candidate",
                )
            )
            if inputs
            else (pair[key] for pair in (initial, fresh) for key in ("baseline", "candidate"))
        )
        actual = confirm(*(resolve(path) for path in paths), **budgets)
    elif kind in ("suite_comparison", "suite_confirmation"):
        from .cli import compare_suites, confirm_suites

        args = argparse.Namespace(**budgets)
        if kind == "suite_comparison":
            actual = compare_suites(resolve(inputs["baseline"]), resolve(inputs["candidate"]), args)
        else:
            actual = confirm_suites(
                *(
                    resolve(inputs[k])
                    for k in (
                        "initial_baseline",
                        "initial_candidate",
                        "fresh_baseline",
                        "fresh_candidate",
                    )
                ),
                args,
            )
    else:
        return

    def portable(value, key=""):
        if isinstance(value, dict):
            return {k: portable(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if key in ("baseline", "candidate") and isinstance(value, str):
            return str(resolve(value).relative_to(ROOT))
        return value

    if any(portable(value, key) != portable(data.get(key), key) for key, value in actual.items()):
        raise ValueError("Saved decision differs from evidence replay")


def audit_run(path, data):
    from .scoring import load_validated, verify_provenance

    if data["status"] == "validated":
        load_validated(path)
        return
    if data["status"] not in ("failed", "cancelled"):
        verify_provenance(data, path.parent)
    # Partial failed runs can lack a prepared tree or harness. They remain
    # failures and cannot contribute a leaderboard point.
    if data.get("source_sha256") and data["source_sha256"] != tree_digest(path.parent / "work"):
        raise ValueError("Run source changed")
    for name, sha in data.get("harness_files", {}).items():
        if digest(contained(path.parent / "work", name)) != sha:
            raise ValueError("Run harness changed")
    for name, sha in data.get("artifacts", {}).items():
        if digest(contained(path.parent, name)) != sha:
            raise ValueError("Run artifact changed")


def receipt_row(path, origin):
    path = resolve(path)
    data = json.loads(path.read_text())
    if data.get("schema_version") != 2 or data.get("status") in LIVE:
        return None
    kind = kind_of(data)
    evidence, audit_error = "verified", ""
    if kind == "run":
        try:
            audit_run(path, data)
        except (ValueError, OSError, KeyError) as error:
            if origin != "historical_import":
                raise
            evidence, audit_error = "invalid", str(error)
    else:
        replay(data, kind)
    measured = data
    if kind in ("comparison", "confirmation") or data.get("status") == "confirmed_improvement":
        pointer = data["candidate"] if kind == "comparison" else data["confirmation"]["candidate"]
        measured = json.loads(resolve(pointer).read_text())
    profile = measured.get("profile", {})
    profile = profile if isinstance(profile, dict) else {"id": profile}
    research, provenance = measured.get("research", {}), measured.get("provenance", {})
    metric = profile.get("metric")
    metrics = measured.get("metrics", {})
    outcome = data.get("status") or data.get("decision", "measured")
    if kind == "suite_comparison":
        outcome = "needs_fresh_confirmation" if data["eligible"] else "rejected"
    decision = data.get("decision", "")
    if not decision:
        decision = {
            "validated": "baseline"
            if data.get("variant") == "baseline"
            else "measured_unconfirmed",
            "screen_passed": "screen_only",
            "reference_validated": "reference_only",
            "diagnostic_completed": "diagnostic_only",
            "confirmed_improvement": "confirmed",
            "confirmed_suite_improvement": "confirmed_suite",
        }.get(outcome, outcome)
    speedup, memory_ratio = data.get("speedup"), data.get("memory_ratio")
    if outcome == "confirmed_improvement":
        speedup = min(data[k]["speedup"] for k in ("initial", "confirmation"))
        memory_ratio = max(data[k]["memory_ratio"] for k in ("initial", "confirmation"))
    relative = str(path.relative_to(ROOT))
    fields = {
        "event_id": identity({"kind": kind, "receipt": relative}),
        "kind": kind,
        "origin": origin,
        "outcome": outcome,
        "reported_outcome": outcome,
        "evidence_audit": evidence,
        "decision": decision,
        "profile": profile.get("id", ""),
        "family": profile.get("family", ""),
        "approach": research.get(
            "approach", profile.get("adapter", "unknown") + "." + measured.get("variant", "unknown")
        ),
        "candidate": research.get("candidate", ""),
        "variant": measured.get("variant", ""),
        "seed": measured.get("seed", ""),
        "case_count": len(measured.get("cases", [])),
        "primary_metric": metric,
        "primary_value": metrics.get(metric),
        "memory_bytes": metrics.get("max_program_object_bytes_per_pe"),
        "max_error_fraction": metrics.get("max_error_fraction"),
        "speedup": speedup,
        "memory_ratio": memory_ratio,
        **{k: provenance.get(k, "") for k in ("git_commit", "git_dirty", "evaluator_sha256")},
        **{
            k: measured.get(k, "")
            for k in ("profile_sha256", "baseline_tree_sha256", "source_sha256", "harness_sha256")
        },
        "sdk_sha256": measured.get("sdk", {}).get("sif_sha256", ""),
        **{k: research.get(k, "unknown") for k in ("author", "model", "agent_harness")},
        "hypothesis": research.get("hypothesis", ""),
        "falsification": research.get("falsification", ""),
        "note": audit_error or data.get("error", ""),
        "receipt": relative,
        "receipt_sha256": digest(path),
    }
    if audit_error:
        fields.update(outcome="evidence_invalid", decision="unverified_history")
    return {k: cell(v) for k, v in fields.items()}


def track(row):
    return tuple(
        row[k]
        for k in (
            "profile_sha256",
            "baseline_tree_sha256",
            "harness_sha256",
            "sdk_sha256",
            "evaluator_sha256",
        )
    )


def board(rows):
    measured = [
        r
        for r in rows
        if r["kind"] == "run"
        and r["outcome"] == "validated"
        and r["case_count"] == "6"
        and r["primary_metric"] == "max_local_cycles"
    ]
    best = {}
    for row in measured:
        key = (*track(row), row["approach"])
        if key not in best or float(row["primary_value"]) < float(best[key]["primary_value"]):
            best[key] = row
    front = [
        row
        for row in measured
        if not any(
            track(other) == track(row)
            and float(other["primary_value"]) <= float(row["primary_value"])
            and float(other["memory_bytes"]) <= float(row["memory_bytes"])
            and (
                float(other["primary_value"]) < float(row["primary_value"])
                or float(other["memory_bytes"]) < float(row["memory_bytes"])
            )
            for other in measured
        )
    ]
    keys = (
        "event_id",
        "profile",
        "approach",
        "variant",
        "primary_value",
        "memory_bytes",
        "source_sha256",
        "evaluator_sha256",
        "receipt",
        "decision",
    )

    def compact(row):
        return {k: row[k] for k in keys}

    return {
        "schema_version": 1,
        "ledger_rows": len(rows),
        "ledger_head": rows[-1]["row_sha256"] if rows else ZERO,
        "authentication": "unsigned local SHA-256 chain; Git history anchors accepted prefixes",
        "scope": "measured views are provisional; compare only within identical contract/toolchain tracks",
        "measured_best_by_approach": [compact(r) for r in best.values()],
        "measured_cycle_memory_front": [compact(r) for r in front],
        "confirmed_improvements": [compact(r) for r in rows if r["decision"] == "confirmed"],
        "confirmed_suite_events": [
            r["event_id"] for r in rows if r["decision"] == "confirmed_suite"
        ],
        "qwen_tps_result": False,
    }


@contextlib.contextmanager
def lock():
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/.ledger.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def append_receipts(paths, origin="live"):
    ledger = ROOT / "results/results.tsv"
    with lock():
        rows = read_rows(ledger)
        existing = {r["event_id"]: r for r in rows}
        added, skipped = 0, []
        for path in paths:
            fields = receipt_row(path, origin)
            if fields is None:
                skipped.append(str(Path(path).relative_to(ROOT)))
                continue
            old = existing.get(fields["event_id"])
            if old:
                if old["receipt_sha256"] != fields["receipt_sha256"]:
                    raise ValueError(
                        "An already-recorded receipt changed; preserve it and create a new attempt"
                    )
                continue
            row = dict.fromkeys(FIELDS, "")
            row.update(
                fields,
                sequence=str(len(rows) + 1),
                recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                previous_sha256=rows[-1]["row_sha256"] if rows else ZERO,
            )
            row["row_sha256"] = identity({k: row[k] for k in FIELDS if k != "row_sha256"})
            rows.append(row)
            existing[row["event_id"]] = row
            added += 1
        ledger.parent.mkdir(parents=True, exist_ok=True)
        temporary = ledger.with_name(".ledger-" + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("w") as handle:
                handle.write(HEADER + "".join("\t".join(r[k] for k in FIELDS) + "\n" for r in rows))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(ledger)
        finally:
            temporary.unlink(missing_ok=True)
        write_json(ROOT / "results/leaderboard.json", board(rows))
    return {
        "added": added,
        "rows": len(rows),
        "ledger": str(ledger.relative_to(ROOT)),
        "skipped_receipts": skipped,
    }


def verify(artifacts=False, base_ref=None):
    path = ROOT / "results/results.tsv"
    rows = read_rows(path)
    expected_board = board(rows)
    if json.loads((ROOT / "results/leaderboard.json").read_text()) != expected_board:
        raise ValueError("Derived leaderboard is stale or changed; run 'ledger board'")
    if base_ref:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", base_ref + "^{commit}"], cwd=ROOT, text=True
        ).strip()
        found = subprocess.check_output(
            ["git", "ls-tree", "--name-only", commit, "--", "results/results.tsv"], cwd=ROOT
        )
        if found:
            old = subprocess.check_output(
                ["git", "show", commit + ":results/results.tsv"], cwd=ROOT
            )
            if not path.read_bytes().startswith(old):
                raise ValueError(
                    "Ledger history was edited, reordered, or truncated relative to Git base"
                )
    if artifacts:
        for row in rows:
            source = receipt_row(row["receipt"], row["origin"])
            for key, value in source.items():
                if row[key] != value:
                    raise ValueError(f"Ledger differs from receipt: {row['receipt']}/{key}")
    return {
        "status": "verified",
        "rows": len(rows),
        "head": expected_board["ledger_head"],
        "artifacts_checked": artifacts,
        "git_prefix_checked": bool(base_ref),
        "authenticated_judge": False,
    }
