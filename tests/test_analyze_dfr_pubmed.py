from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
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


def test_fresh_canonical_metrics_need_no_migration_marker(tmp_path: Path) -> None:
    model, preparation, split, seed = "gcn", "erm", "geomgcn-0", 1
    run_id = NAMESPACE["expected_canonical_id"](model, preparation, split, seed)
    root = tmp_path / run_id
    root.mkdir()
    arrays = {
        "labels": np.array([0, 1, 0, 1]),
        "predictions": np.array([0, 0, 0, 1]),
        "test_idx": np.arange(4),
        "merged_bin": np.array([0, 0, 1, 1]),
        "tail_mask": np.array([True, True, False, False]),
    }
    checksums = {
        key: NAMESPACE["array_sha256"](arrays[name])
        for key, name in (("labels", "labels"), ("merged_bin", "merged_bin"),
                          ("tail_mask", "tail_mask"), (f"{split}.test", "test_idx"))
    }
    manifest = {"graph": {"num_classes": 2}, "array_checksums_sha256": checksums}
    config = dict(run_id=run_id, dataset="pubmed", model=model, preparation=preparation,
                  split=split, seed=seed, code_version="fresh", fixed_input_checksums=checksums)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    np.savez(root / "run_data.npz", **arrays)
    metrics, _ = NAMESPACE["recompute_scores"](arrays, run_id=run_id)
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    kwargs = dict(model=model, preparation=preparation, split=split, seed=seed)
    loaded = NAMESPACE["load_canonical_run"](
        tmp_path, manifest, input_digest=hashlib.sha256(), **kwargs
    )
    assert loaded.metrics == metrics
    metrics["tail_macro_f1"] += 0.1
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from recomputed"):
        NAMESPACE["load_canonical_run"](
            tmp_path, manifest, input_digest=hashlib.sha256(), **kwargs
        )
