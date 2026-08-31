from __future__ import annotations

import copy
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = runpy.run_path(
    str(REPO_ROOT / "scripts" / "analyze_dfr_pubmed.py")
)


def test_semantic_status_hash_ignores_paths_and_execution_times() -> None:
    status = {
        "schema_version": 1,
        "experiment": "dfr-pubmed-frozen-representation-pilot",
        "code_version": "abc123",
        "scope": {"dataset": "pubmed", "expected_runs": 90},
        "resource_limits": {"processes": 1},
        "completed_runs": ["run-b", "run-a"],
        "failed_run": None,
        "in_progress_run": None,
        "paths": {"dfr_output_root": "C:/private/location"},
        "started_utc": "2026-07-29T13:00:00+02:00",
        "completed_utc": "2026-07-29T16:00:00+02:00",
    }
    relocated = copy.deepcopy(status)
    relocated["paths"] = {"dfr_output_root": "/different/machine"}
    relocated["started_utc"] = "2030-01-01T00:00:00Z"
    relocated["completed_utc"] = "2030-01-01T03:00:00Z"
    relocated["completed_runs"].reverse()

    fingerprint = NAMESPACE["semantic_status_sha256"]
    assert fingerprint(status) == fingerprint(relocated)


def test_semantic_status_hash_changes_with_completed_run_set() -> None:
    status = {
        "schema_version": 1,
        "experiment": "dfr-pubmed-frozen-representation-pilot",
        "code_version": "abc123",
        "scope": {"dataset": "pubmed", "expected_runs": 90},
        "resource_limits": {"processes": 1},
        "completed_runs": ["run-a"],
        "failed_run": None,
        "in_progress_run": None,
    }
    changed = copy.deepcopy(status)
    changed["completed_runs"].append("run-b")

    fingerprint = NAMESPACE["semantic_status_sha256"]
    assert fingerprint(status) != fingerprint(changed)
