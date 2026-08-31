import json
import runpy
import zipfile
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = runpy.run_path(str(REPO_ROOT / "scripts" / "verify_raw_store.py"))


def _sha(array: np.ndarray) -> str:
    return NAMESPACE["array_sha256"](array)


def make_store(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    panel_path = tmp_path / "panel.jsonl"
    manifest_dir = tmp_path / "datasets"
    manifest_dir.mkdir()

    run_id = "toy_gcn_erm_split-0_seed-1"
    panel = {
        "run_order": 1,
        "run_id": run_id,
        "dataset": "toy",
        "model": "gcn",
        "preparation": "erm",
        "split": "split-0",
        "seed": 1,
        "parent_run_id": None,
    }
    panel_path.write_text(json.dumps(panel) + "\n", encoding="utf-8")

    labels = np.array([0, 1, 0], dtype=np.int64)
    train_idx = np.array([0], dtype=np.int64)
    val_idx = np.array([1], dtype=np.int64)
    test_idx = np.array([2], dtype=np.int64)
    merged_bin = np.array([0, 1, 0], dtype=np.int64)
    tail_mask = np.array([True, False, True], dtype=np.bool_)
    checksums = {
        "labels": _sha(labels),
        "split-0.train": _sha(train_idx),
        "split-0.val": _sha(val_idx),
        "split-0.test": _sha(test_idx),
        "merged_bin": _sha(merged_bin),
        "tail_mask": _sha(tail_mask),
    }
    dataset_manifest = {
        "dataset": "toy",
        "metric": "accuracy",
        "training_seeds": [1],
        "graph": {"num_nodes": 3, "num_classes": 2},
        "splits": [{"split_id": "split-0"}],
        "local_homophily": {
            "merged_bins": [
                {"interval": "[0,0.5)"},
                {"interval": "[0.5,1]"},
            ]
        },
        "array_checksums_sha256": checksums,
    }
    (manifest_dir / "toy.json").write_text(
        json.dumps(dataset_manifest), encoding="utf-8"
    )
    model_config = {
        "model": "gcn",
        "z_dim": 4,
        "head_dropout": 0.0,
        "encoder": {"hidden_dim": 4},
        "train": {
            "optimizer": "adamw",
            "lr": 0.01,
            "weight_decay": 0.0,
            "max_epochs": 1,
        },
    }
    (manifest_dir / "gcn.yaml").write_text(
        json.dumps(model_config),
        encoding="utf-8",
    )

    run_dir = raw_root / run_id
    run_dir.mkdir()
    config = {
        **{key: value for key, value in panel.items() if key != "run_order"},
        "code_version": "1" * 40,
        "versions": {
            "python": "3.11.15",
            "torch": "2.6.0+cu124",
            "numpy": "2.2.6",
        },
        "fixed_input_checksums": checksums,
        "model_config": model_config,
        "train_config": model_config["train"],
        "recovery_config": None,
    }
    logits = np.array([[2.0, 1.0], [0.1, 0.9], [3.0, 1.0]], dtype=np.float32)
    predictions = logits.argmax(axis=-1).astype(np.int64)
    metrics = NAMESPACE["gap_audit_metrics"](
        y_true=labels,
        y_pred=predictions,
        test_idx=test_idx,
        merged_bin=merged_bin,
        tail_mask=tail_mask,
        bin_intervals=["[0,0.5)", "[0.5,1]"],
        standard_metric="accuracy",
    )
    metrics["checkpoint"] = {
        "parent_run_id": None,
        "best_epoch": 1,
        "epochs_run": 1,
        "train_seconds": 1.0,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps([{"epoch": 1}]), encoding="utf-8"
    )
    with zipfile.ZipFile(run_dir / "checkpoint.pt", "w") as archive:
        archive.writestr("archive/data.pkl", b"state")
        archive.writestr("archive/version", b"3")
    np.savez_compressed(
        run_dir / "run_data.npz",
        logits=logits,
        predictions=predictions,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        merged_bin=merged_bin,
        tail_mask=tail_mask,
    )
    return raw_root, panel_path, manifest_dir


def add_reference_environment_provenance(
    raw_root: Path,
    manifest_dir: Path,
    *,
    experiment_manifest_sha256: str,
) -> None:
    graph_checksums = {"edge_index": "2" * 64}
    manifest_path = manifest_dir / "toy.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph_checksums_sha256"] = graph_checksums
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    run_id = "toy_gcn_erm_split-0_seed-1"
    config_path = raw_root / run_id / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "git_dirty": False,
            "command": ["python", "scripts/train_run.py"],
            "experiment_manifest_sha256": experiment_manifest_sha256,
            "parent_checkpoint_sha256": None,
            "accelerator": "cpu",
            "graph_checksums": graph_checksums,
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_metadata_and_array_verification_pass(tmp_path: Path) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)

    metadata = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="metadata",
        progress_every=0,
    )
    arrays = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="arrays",
        progress_every=0,
    )

    assert metadata["status"] == "pass"
    assert metadata["expected_runs"] == 1
    assert arrays["status"] == "pass"


def test_strict_environment_uses_selected_non_main_panel_checksum(
    tmp_path: Path,
) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)
    add_reference_environment_provenance(
        raw_root,
        manifest_dir,
        experiment_manifest_sha256=NAMESPACE["file_sha256"](panel_path),
    )

    report = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="metadata",
        progress_every=0,
        require_reference_environment=True,
    )

    assert panel_path != REPO_ROOT / "manifests" / "main-runs.jsonl"
    assert report["status"] == "pass"


def test_strict_environment_rejects_checksum_from_different_panel(
    tmp_path: Path,
) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)
    main_panel_path = REPO_ROOT / "manifests" / "main-runs.jsonl"
    add_reference_environment_provenance(
        raw_root,
        manifest_dir,
        experiment_manifest_sha256=NAMESPACE["file_sha256"](main_panel_path),
    )

    report = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="metadata",
        progress_every=0,
        require_reference_environment=True,
    )

    assert report["status"] == "fail"
    assert any(
        "experiment manifest checksum differs" in error
        for error in report["errors"]
    )


def test_array_verification_detects_registered_input_change(tmp_path: Path) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)
    run_dir = raw_root / "toy_gcn_erm_split-0_seed-1"
    with np.load(run_dir / "run_data.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["labels"] = np.array([1, 1, 0], dtype=np.int64)
    np.savez_compressed(run_dir / "run_data.npz", **arrays)

    report = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="arrays",
        progress_every=0,
    )

    assert report["status"] == "fail"
    assert any("labels checksum differs" in error for error in report["errors"])


def test_metadata_verification_rejects_corrupt_checkpoint(
    tmp_path: Path,
) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)
    checkpoint = (
        raw_root / "toy_gcn_erm_split-0_seed-1" / "checkpoint.pt"
    )
    checkpoint.write_bytes(b"not a checkpoint")

    report = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="metadata",
        progress_every=0,
    )

    assert report["status"] == "fail"
    assert any(
        "not a readable PyTorch ZIP archive" in error
        for error in report["errors"]
    )


def test_metadata_verification_rejects_model_config_drift(
    tmp_path: Path,
) -> None:
    raw_root, panel_path, manifest_dir = make_store(tmp_path)
    config_path = (
        raw_root / "toy_gcn_erm_split-0_seed-1" / "config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_config"]["train"]["lr"] = 0.02
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = NAMESPACE["validate_store"](
        raw_root=raw_root,
        panel_path=panel_path,
        dataset_manifest_dir=manifest_dir,
        model_config_dir=manifest_dir,
        level="metadata",
        progress_every=0,
    )

    assert report["status"] == "fail"
    assert any(
        "config.model_config differs" in error for error in report["errors"]
    )


def test_panel_requires_dependency_safe_order(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.jsonl"
    row = {
        "run_order": 1,
        "run_id": "toy_gcn_last_split-0_seed-1",
        "dataset": "toy",
        "model": "gcn",
        "preparation": "last",
        "split": "split-0",
        "seed": 1,
        "parent_run_id": "toy_gcn_erm_split-0_seed-1",
    }
    panel_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(NAMESPACE["ContractError"], match="must occur earlier"):
        NAMESPACE["load_panel"](panel_path)
