from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_training_compute.py"


def load_module():
    module_name = "analyze_training_compute_modes"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def actor_manifest() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "manifests"
            / "additional-datasets"
            / "actor.json"
        ).read_text(encoding="utf-8")
    )


def actor_runs(module) -> list:
    manifest = actor_manifest()
    return [
        module.RunTime(
            dataset="actor",
            model="mlp",
            preparation=preparation,
            split=str(split["split_id"]),
            seed=int(seed),
            seconds=1.0,
            gpu_name="NVIDIA A100-SXM4-80GB",
        )
        for preparation in module.PREPARATION_ORDER
        for split in manifest["splits"]
        for seed in manifest["training_seeds"]
    ]


def write_run(
    root: Path,
    run_id: str,
    *,
    dataset: str,
    gpu_name: str = "NVIDIA A100-SXM4-80GB",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model": "mlp",
                "preparation": "erm",
                "split": "split-0",
                "seed": 1,
                "device": "cuda",
                "accelerator": {"gpu_name": gpu_name},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"checkpoint": {"train_seconds": 1.0}}),
        encoding="utf-8",
    )


def test_compute_modes_use_safe_output_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_module()

    assert module.DEFAULT_OUTPUT_DIRS[module.AnalysisDatasetMode.ORIGINAL] == (
        REPO_ROOT / "results" / "main" / "compute"
    )
    assert module.DEFAULT_OUTPUT_DIRS[module.AnalysisDatasetMode.ADDITIONAL] == (
        REPO_ROOT / "results" / "additional" / "compute"
    )
    assert module.DEFAULT_OUTPUT_DIRS[module.AnalysisDatasetMode.EXTENDED] == (
        REPO_ROOT / "results" / "extended" / "compute"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--raw-dir", str(tmp_path)],
    )
    args = module.parse_args()
    assert args.dataset_mode is module.AnalysisDatasetMode.ORIGINAL
    assert args.raw_root == tmp_path
    assert args.output_dir == (
        REPO_ROOT / "results" / "main" / "compute"
    )


def test_compute_requires_exact_registered_split_seed_pairs() -> None:
    module = load_module()
    manifest = actor_manifest()
    runs = actor_runs(module)

    module.validate_run_panel(
        runs,
        {"actor": manifest},
        ("actor",),
        model_order=("mlp",),
    )

    last = runs[-1]
    mismatched = [
        *runs[:-1],
        module.RunTime(
            dataset=last.dataset,
            model=last.model,
            preparation=last.preparation,
            split="unexpected-split",
            seed=last.seed,
            seconds=last.seconds,
            gpu_name=last.gpu_name,
        ),
    ]
    with pytest.raises(ValueError, match="run panel mismatch"):
        module.validate_run_panel(
            mismatched,
            {"actor": manifest},
            ("actor",),
            model_order=("mlp",),
        )


def test_compute_collection_rejects_wrong_store_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    module = load_module()
    original_root = tmp_path / "original"
    additional_root = tmp_path / "additional"
    original_root.mkdir()
    additional_root.mkdir()
    write_run(original_root, "duplicate", dataset="pubmed")
    write_run(additional_root, "duplicate", dataset="actor")

    specs = module.raw_root_specs(
        "extended", original_root, additional_root
    )
    with pytest.raises(ValueError, match="Duplicate run_id"):
        module.collect_runs(specs, ("pubmed", "actor"))

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    write_run(wrong_root, "actor-run", dataset="actor")
    with pytest.raises(ValueError, match="wrong raw store"):
        module.collect_runs(
            module.raw_root_specs("original", wrong_root, None),
            ("pubmed",),
        )


def test_additional_compute_inventory_uses_recorded_gpu_names(
    tmp_path: Path,
) -> None:
    module = load_module()
    runs = [
        module.RunTime(
            dataset="actor",
            model="mlp",
            preparation=preparation,
            split="split-0",
            seed=1,
            seconds=3_600.0,
            gpu_name="AMD Instinct MI210",
        )
        for preparation in module.PREPARATION_ORDER
    ]

    module.analyse(
        runs,
        tmp_path,
        dataset_mode="additional",
        included_datasets=("actor",),
    )

    with (tmp_path / "hardware_inventory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        hardware = list(csv.DictReader(handle))
    manifest = json.loads(
        (tmp_path / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert hardware == [
        {"gpu_model": "AMD Instinct MI210", "memory_gb": ""}
    ]
    assert manifest["dataset_mode"] == "additional"
    assert manifest["datasets"] == ["actor"]
    assert manifest["run_count"] == 4
    assert manifest["runs_per_preparation"] == 1


def test_extended_compute_keeps_original_and_additional_hardware_provenance(
    tmp_path: Path,
) -> None:
    module = load_module()
    runs = [
        module.RunTime(
            dataset="pubmed",
            model="mlp",
            preparation=preparation,
            seconds=3_600.0,
            gpu_name="UNRELIABLE ORIGINAL RECORD",
        )
        for preparation in module.PREPARATION_ORDER
    ] + [
        module.RunTime(
            dataset="actor",
            model="mlp",
            preparation=preparation,
            seconds=3_600.0,
            gpu_name="AMD Instinct MI210",
        )
        for preparation in module.PREPARATION_ORDER
    ]

    module.analyse(
        runs,
        tmp_path,
        dataset_mode="extended",
        included_datasets=("pubmed", "actor"),
    )

    with (tmp_path / "hardware_inventory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        hardware = list(csv.DictReader(handle))
    assert hardware == [
        {"gpu_model": "NVIDIA A100", "memory_gb": "80"},
        {
            "gpu_model": "NVIDIA GeForce RTX 3080 Laptop GPU",
            "memory_gb": "16",
        },
        {"gpu_model": "AMD Instinct MI210", "memory_gb": ""},
    ]
    assert all(
        row["gpu_model"] != "UNRELIABLE ORIGINAL RECORD" for row in hardware
    )
