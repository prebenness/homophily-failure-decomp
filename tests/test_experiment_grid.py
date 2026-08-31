from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hfd.experiment_grid import (
    RunState,
    execution_plan,
    grid_jsonl_text,
    inspect_runs,
    read_experiment_grid,
    select_runs,
    train_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "manifests" / "main-runs.jsonl"
SPEC_PATH = REPO_ROOT / "configs" / "experiment.yaml"
REQUIRED_OUTPUTS = (
    "config.json",
    "metrics.json",
    "history.json",
    "checkpoint.pt",
    "run_data.npz",
)


def test_committed_grid_is_complete_and_deterministic() -> None:
    rows = read_experiment_grid(GRID_PATH)

    assert len(rows) == 4200
    assert grid_jsonl_text(rows) == GRID_PATH.read_text(encoding="utf-8")
    assert Counter(row.dataset for row in rows) == {
        "pubmed": 840,
        "ogbn-arxiv": 840,
        "roman-empire": 840,
        "amazon-ratings": 840,
        "arxiv-year": 840,
    }
    assert Counter(row.model for row in rows) == {
        "mlp": 600,
        "gcn": 600,
        "graphsage-mean": 600,
        "h2gcn": 600,
        "gpr-gnn": 600,
        "linkx": 600,
        "acm-gcn": 600,
    }
    assert Counter(row.preparation for row in rows) == {
        "erm": 1050,
        "last": 1050,
        "head": 1050,
        "full": 1050,
    }
    assert rows[0].run_id == "pubmed_mlp_erm_geomgcn-0_seed-1"
    assert rows[-1].run_id == "arxiv-year_acm-gcn_full_lim-4_seed-6"
    assert all(
        (row.preparation in {"last", "head"}) == bool(row.parent_run_id)
        for row in rows
    )


def test_recovery_selection_includes_matching_erm_parent() -> None:
    rows = read_experiment_grid(GRID_PATH)
    selected = select_runs(
        rows,
        datasets=["pubmed"],
        models=["gcn"],
        preparations=["head"],
        splits=["geomgcn-0"],
        seeds=[1],
    )

    assert [row.preparation for row in selected] == ["erm", "head"]
    assert selected[1].parent_run_id == selected[0].run_id


def _write_run(root: Path, row, *, config_overrides=None, omit=()) -> None:
    run_dir = root / row.run_id
    run_dir.mkdir()
    config = {
        "run_id": row.run_id,
        "dataset": row.dataset,
        "model": row.model,
        "preparation": row.preparation,
        "split": row.split,
        "seed": row.seed,
        "parent_run_id": row.parent_run_id or None,
    }
    config.update(config_overrides or {})
    for filename in REQUIRED_OUTPUTS:
        if filename in omit:
            continue
        path = run_dir / filename
        if filename == "config.json":
            path.write_text(json.dumps(config), encoding="utf-8")
        elif filename == "metrics.json":
            path.write_text(json.dumps({"checkpoint": {}}), encoding="utf-8")
        elif filename == "history.json":
            path.write_text(json.dumps([{"epoch": 0}]), encoding="utf-8")
        elif filename == "checkpoint.pt":
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("archive/data.pkl", b"state")
                archive.writestr("archive/version", b"3")
        elif filename == "run_data.npz":
            np.savez(
                path,
                logits=np.zeros((1, 1)),
                predictions=np.zeros(1, dtype=np.int64),
                labels=np.zeros(1, dtype=np.int64),
                train_idx=np.zeros(1, dtype=np.int64),
                val_idx=np.zeros(1, dtype=np.int64),
                test_idx=np.zeros(1, dtype=np.int64),
                merged_bin=np.zeros(1, dtype=np.int64),
                tail_mask=np.ones(1, dtype=np.bool_),
            )
        else:
            raise AssertionError(filename)


def test_status_and_resume_plan_are_dependency_safe(tmp_path: Path) -> None:
    all_rows = read_experiment_grid(GRID_PATH)
    rows = select_runs(
        all_rows,
        datasets=["pubmed"],
        models=["gcn"],
        preparations=["head"],
        splits=["geomgcn-0"],
        seeds=[1],
    )
    _write_run(tmp_path, rows[0])

    inspections = inspect_runs(rows, tmp_path, REQUIRED_OUTPUTS)
    assert [item.state for item in inspections] == [
        RunState.COMPLETE,
        RunState.MISSING,
    ]
    assert execution_plan(rows, inspections) == [rows[1]]


def test_incomplete_parent_is_scheduled_before_recovery(tmp_path: Path) -> None:
    all_rows = read_experiment_grid(GRID_PATH)
    rows = select_runs(
        all_rows,
        datasets=["pubmed"],
        models=["gcn"],
        preparations=["last"],
        splits=["geomgcn-0"],
        seeds=[1],
    )
    _write_run(tmp_path, rows[0], omit=("checkpoint.pt",))

    inspections = inspect_runs(rows, tmp_path, REQUIRED_OUTPUTS)
    assert [item.state for item in inspections] == [
        RunState.INCOMPLETE,
        RunState.MISSING,
    ]
    assert execution_plan(rows, inspections) == rows


def test_replacing_parent_also_replaces_complete_recovery_child(
    tmp_path: Path,
) -> None:
    all_rows = read_experiment_grid(GRID_PATH)
    rows = select_runs(
        all_rows,
        datasets=["pubmed"],
        models=["gcn"],
        preparations=["head"],
        splits=["geomgcn-0"],
        seeds=[1],
    )
    _write_run(tmp_path, rows[0], omit=("checkpoint.pt",))
    _write_run(tmp_path, rows[1])

    inspections = inspect_runs(rows, tmp_path, REQUIRED_OUTPUTS)
    assert [item.state for item in inspections] == [
        RunState.INCOMPLETE,
        RunState.COMPLETE,
    ]
    assert execution_plan(rows, inspections) == rows


def test_invalid_existing_run_requires_explicit_rerun(tmp_path: Path) -> None:
    row = read_experiment_grid(GRID_PATH)[0]
    _write_run(tmp_path, row, config_overrides={"seed": 99})
    inspections = inspect_runs([row], tmp_path, REQUIRED_OUTPUTS)

    assert inspections[0].state is RunState.INVALID
    with pytest.raises(RuntimeError, match="existing output is invalid"):
        execution_plan([row], inspections)
    assert execution_plan([row], inspections, rerun_invalid=True) == [row]


def test_training_command_preserves_atomic_run_fields() -> None:
    row = read_experiment_grid(GRID_PATH)[0]
    command = train_command(
        "python",
        Path("scripts/train_run.py"),
        row,
        output_root=Path("artifacts/runs"),
        device="cuda:1",
        require_clean=True,
    )

    assert command == [
        "python",
        str(Path("scripts/train_run.py")),
        "--dataset",
        "pubmed",
        "--model",
        "mlp",
        "--preparation",
        "erm",
        "--split",
        "geomgcn-0",
        "--seed",
        "1",
        "--output-root",
        str(Path("artifacts/runs")),
        "--device",
        "cuda:1",
        "--require-clean",
    ]


def test_training_command_can_pin_an_extension_registry_and_manifest() -> None:
    row = read_experiment_grid(GRID_PATH)[0]
    command = train_command(
        "python",
        Path("scripts/train_run.py"),
        row,
        output_root=Path("artifacts/runs"),
        dataset_config=Path("configs/datasets-additional.yaml"),
        experiment_manifest=Path("manifests/additional-dataset-runs.jsonl"),
    )

    assert command[-4:] == [
        "--config",
        str(Path("configs/datasets-additional.yaml")),
        "--experiment-manifest",
        str(Path("manifests/additional-dataset-runs.jsonl")),
    ]
