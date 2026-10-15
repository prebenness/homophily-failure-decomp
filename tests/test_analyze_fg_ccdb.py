from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = runpy.run_path(str(ROOT / "scripts" / "analyze_fg_ccdb.py"))


def test_checkpoint_scores_do_not_require_training_history(tmp_path: Path) -> None:
    model, preparation, split, seed = "gcn", "erm", "geomgcn-0", 1
    run_id = NAMESPACE["expected_id"](model, preparation, split, seed, fg=False)
    root = tmp_path / run_id
    root.mkdir()
    arrays = dict(
        labels=np.array([0, 1, 0, 1]), predictions=np.array([0, 0, 0, 1]),
        test_idx=np.arange(4), train_idx=np.array([0, 1]),
        merged_bin=np.array([0, 0, 1, 1]),
        tail_mask=np.array([True, True, False, False]),
    )
    checksums = {
        key: NAMESPACE["array_sha256"](arrays[name])
        for key, name in (("labels", "labels"), ("merged_bin", "merged_bin"),
                          ("tail_mask", "tail_mask"), (f"{split}.test", "test_idx"))
    }
    manifest = {"array_checksums_sha256": checksums,
                "local_homophily": {"merged_bins": [{"interval": "low"}, {"interval": "high"}]}}
    config = dict(run_id=run_id, dataset="pubmed", model=model, preparation=preparation,
                  split=split, seed=seed, code_version="fresh", fixed_input_checksums=checksums)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "checkpoint.pt").write_bytes(b"unused by analysis")
    np.savez(root / "run_data.npz", **arrays)
    scores = NAMESPACE["gap_audit_metrics"](
        y_true=arrays["labels"], y_pred=arrays["predictions"], test_idx=arrays["test_idx"],
        merged_bin=arrays["merged_bin"], tail_mask=arrays["tail_mask"],
        bin_intervals=["low", "high"], standard_metric="accuracy",
    )
    (root / "metrics.json").write_text(json.dumps(scores), encoding="utf-8")
    row, _ = NAMESPACE["score_run"](
        root, manifest=manifest, model=model, preparation=preparation,
        split=split, seed=seed, fg=False,
    )
    assert row["tail_macro_f1"] == scores["tail_macro_f1"]
    assert row["best_epoch"] is None and row["train_seconds"] is None
