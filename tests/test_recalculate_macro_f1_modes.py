from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "recalculate_macro_f1.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "recalculate_macro_f1_modes", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run_config(
    root: Path,
    run_id: str,
    *,
    dataset: str,
    model: str = "mlp",
    preparation: str = "erm",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "preparation": preparation,
            }
        ),
        encoding="utf-8",
    )


def test_modes_select_expected_datasets_and_safe_output_roots() -> None:
    module = load_module()

    original_order, original = module.selected_datasets("original")
    additional_order, actor_only = module.selected_datasets(
        "additional", ("blogcatalog",)
    )
    extended_order, extended = module.selected_datasets("extended")

    assert original == original_order == (
        "pubmed",
        "ogbn-arxiv",
        "roman-empire",
        "amazon-ratings",
        "arxiv-year",
    )
    assert additional_order == ("actor", "blogcatalog")
    assert actor_only == ("actor",)
    assert extended == extended_order == (*original_order, *additional_order)
    assert module.DEFAULT_OUTPUT_ROOTS[module.AnalysisDatasetMode.ORIGINAL] == (
        REPO_ROOT / "results" / "main"
    )
    assert module.DEFAULT_OUTPUT_ROOTS[module.AnalysisDatasetMode.ADDITIONAL] == (
        REPO_ROOT / "results" / "additional"
    )
    assert module.DEFAULT_OUTPUT_ROOTS[module.AnalysisDatasetMode.EXTENDED] == (
        REPO_ROOT / "results" / "extended"
    )


def test_exclusions_must_belong_to_the_selected_mode() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="outside 'original' mode"):
        module.selected_datasets("original", ("actor",))
    with pytest.raises(ValueError, match="At least one dataset"):
        module.selected_datasets("additional", ("actor", "blogcatalog"))


def test_run_collection_rejects_dataset_in_wrong_store(tmp_path: Path) -> None:
    module = load_module()
    original_root = tmp_path / "original"
    original_root.mkdir()
    write_run_config(original_root, "actor-run", dataset="actor")

    specs = module.raw_root_specs("original", original_root, None)
    with pytest.raises(RuntimeError, match="wrong raw store"):
        module.collect_selected_runs(specs, ("pubmed",))


def test_run_collection_rejects_duplicate_ids_across_stores(
    tmp_path: Path,
) -> None:
    module = load_module()
    original_root = tmp_path / "original"
    additional_root = tmp_path / "additional"
    original_root.mkdir()
    additional_root.mkdir()
    write_run_config(original_root, "duplicate", dataset="pubmed")
    write_run_config(additional_root, "duplicate", dataset="actor")

    specs = module.raw_root_specs(
        "extended", original_root, additional_root
    )
    with pytest.raises(RuntimeError, match="Duplicate run_id"):
        module.collect_selected_runs(specs, ("pubmed", "actor"))


def test_each_mode_requires_its_own_run_store(tmp_path: Path) -> None:
    module = load_module()

    with pytest.raises(ValueError, match="--raw-root"):
        module.raw_root_specs("original", None, None)
    with pytest.raises(ValueError, match="--additional-raw-root"):
        module.raw_root_specs("additional", tmp_path, None)
    with pytest.raises(ValueError, match="--additional-raw-root"):
        module.raw_root_specs("extended", tmp_path, None)


def test_original_aggregate_builder_reproduces_tracked_csvs(
    tmp_path: Path,
) -> None:
    module = load_module()
    run_rows = []
    for path in sorted((REPO_ROOT / "results" / "main" / "runs").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record["metrics"]
        run_rows.append(
            {
                "run_id": record["run_id"],
                "dataset": record["dataset"],
                "model": record["model"],
                "preparation": record["preparation"],
                "split": record["split"],
                "seed": int(record["seed"]),
                "whole_test_macro_f1": metrics["whole_test"]["macro_f1"],
                "tail_macro_f1": metrics["tail"]["macro_f1"],
                "tail_gap": metrics["tail_gap"],
            }
        )
    manifests = {
        dataset: json.loads(
            module.DATASET_REGISTRY[dataset]
            .manifest_path(REPO_ROOT)
            .read_text(encoding="utf-8")
        )
        for dataset in module.ORIGINAL_DATASET_ORDER
    }
    dataset_index = {
        dataset: index
        for index, dataset in enumerate(module.ORIGINAL_DATASET_ORDER)
    }
    model_index = {
        model: index for index, model in enumerate(module.MODEL_ORDER)
    }
    preparation_index = {
        preparation: index
        for index, preparation in enumerate(module.PREPARATION_ORDER)
    }
    run_rows.sort(
        key=lambda row: (
            dataset_index[row["dataset"]],
            model_index[row["model"]],
            preparation_index[row["preparation"]],
            row["split"],
            row["seed"],
        )
    )

    module.build_aggregates(
        run_rows,
        manifests,
        tmp_path,
        module.ORIGINAL_DATASET_ORDER,
    )

    for filename in (
        "prepared_scores.csv",
        "paired_gains.csv",
        "erm_tail_gaps.csv",
        "recovered_shares.csv",
    ):
        assert (tmp_path / "aggregates" / filename).read_bytes() == (
            REPO_ROOT / "results" / "main" / "aggregates" / filename
        ).read_bytes()
