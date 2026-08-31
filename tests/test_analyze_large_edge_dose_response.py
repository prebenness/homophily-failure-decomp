from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_large_edge_dose_response.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

MODELS = NAMESPACE["MODELS"]
P_PERCENTAGES = NAMESPACE["P_PERCENTAGES"]
GRAPH_REPLICATES = NAMESPACE["GRAPH_REPLICATES"]
DESIGN_ID = NAMESPACE["DESIGN_ID"]
CODE_VERSION = "a" * 40

DATASET_SPECS = {
    "arxiv-year": {
        "identities": [
            (f"lim-{split_index}", seed)
            for split_index in range(5)
            for seed in range(1, 7)
        ],
        "tail_side": "high-h_v",
        "desired_same": False,
        "direction": "decrease-local-homophily",
        "generator": "singleton",
        "seed_base": 2026081100,
    },
    "ogbn-arxiv": {
        "identities": [("ogb-time", seed) for seed in range(1, 31)],
        "tail_side": "low-h_v",
        "desired_same": True,
        "direction": "increase-local-homophily",
        "generator": "coordinated",
        "seed_base": 2026082100,
    },
}
TAIL_COUNT = 20


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_sha256(
    dataset: str, model: str, split: str, seed: int
) -> str:
    return hashlib.sha256(
        f"{dataset}/{model}/{split}/{seed}".encode()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(actual_root: Path, dataset: str) -> tuple[Path, str]:
    spec = DATASET_SPECS[dataset]
    dataset_root = actual_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    path = dataset_root / "experiment_manifest.json"
    manifest = {
        "schema_version": 1,
        "design_id": DESIGN_ID,
        "code_version": CODE_VERSION,
        "dataset": dataset,
        "models": list(MODELS),
        "p_percentages": list(P_PERCENTAGES),
        "graph_replicates": len(GRAPH_REPLICATES),
        "graph_replicate_indices": list(GRAPH_REPLICATES),
        "replicate_seed_base": spec["seed_base"],
        "replicate_seed_rule": "replicate_seed_base + replicate_index",
        "tail_definition": f"frozen registered {dataset} tail",
        "tail_side": spec["tail_side"],
        "tail_count": TAIL_COUNT,
        "desired_same": spec["desired_same"],
        "treatment_direction": spec["direction"],
        "generator": spec["generator"],
        "target_ranking": "one frozen permutation per graph replicate",
        "target_sets": "nested prefixes of the replicate target ranking",
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "graph_start_at_each_p": f"registered original {dataset} graph",
        "treatment": "one degree-preserving directed swap per target",
        "control": "one degree-preserving relation-matched swap per target",
        "shared_graph_pairs": "the same pair is evaluated by both models",
        "checkpoint_panel": "registered split/seed identities",
        "checkpoint_identities": [
            {"split": split, "seed": seed}
            for split, seed in spec["identities"]
        ],
        "n_checkpoints_per_model": 30,
        "metric_population": "fixed-full-tail-test",
        "evaluation_subset": "complete fixed tail-test subset",
        "metric": "macro-F1 over classes present in the subset",
        "baseline": "registered original graph evaluated once",
        "actual_generation_attempts": 1,
        "pair_output_files": [
            "graphs.npz",
            "swaps.json",
            "checkpoint_metrics.csv",
            "predictions.npz",
            "metadata.json",
        ],
        "baseline_output_files": [
            "checkpoint_metrics.csv",
            "predictions.npz",
            "metadata.json",
        ],
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return dataset_root, file_sha256(path)


def write_baseline(
    dataset_root: Path, dataset: str, manifest_sha256: str
) -> None:
    spec = DATASET_SPECS[dataset]
    rows: list[dict] = []
    for model_index, model in enumerate(MODELS):
        for split, seed in spec["identities"]:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "split": split,
                    "seed": seed,
                    "n_tail_test": 100,
                    "tail_test_class_composition": '{"0":50,"1":50}',
                    "tail_test_macro_f1": 0.4 + 0.1 * model_index,
                    "tail_test_accuracy": 0.41 + 0.1 * model_index,
                    "n_full_graph_prediction_differences_from_parent": 0,
                    "n_tail_test_prediction_differences_from_parent": 0,
                    "full_graph_prediction_difference_nodes_from_parent": "[]",
                    "full_graph_logit_reproduction_checked": False,
                    "max_abs_logit_difference_from_parent": 0.0,
                    "mean_abs_logit_difference_from_parent": 0.0,
                    "checkpoint_sha256": checkpoint_sha256(
                        dataset, model, split, seed
                    ),
                }
            )
    baseline_dir = dataset_root / "baseline"
    metrics_path = baseline_dir / "checkpoint_metrics.csv"
    write_csv(metrics_path, rows)
    predictions_path = baseline_dir / "predictions.npz"
    np.savez_compressed(predictions_path, predictions=np.asarray([1]))
    metadata = {
        "schema_version": 1,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": dataset,
        "models": list(MODELS),
        "metric_population": "fixed-full-tail-test",
        "n_checkpoints_per_model": 30,
        "n_checkpoint_rows": 60,
        "all_original_predictions_reproduced": True,
        "all_original_tail_test_predictions_reproduced": True,
        "n_checkpoints_with_full_graph_prediction_differences": 0,
        "total_full_graph_prediction_differences": 0,
        "max_full_graph_prediction_differences_per_checkpoint": 0,
        "full_graph_prediction_differences_by_checkpoint": [],
        "checkpoint_identities": [
            {"split": split, "seed": seed}
            for split, seed in spec["identities"]
        ],
        "code_version": CODE_VERSION,
        "experiment_manifest_sha256": manifest_sha256,
        "output_sha256": {
            "checkpoint_metrics.csv": file_sha256(metrics_path),
            "predictions.npz": file_sha256(predictions_path),
        },
    }
    (baseline_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def graph_hash(dataset: str, label: str) -> str:
    return hashlib.sha256(f"{dataset}/{label}".encode()).hexdigest()


def write_dose_cell(
    dataset_root: Path,
    dataset: str,
    manifest_sha256: str,
    *,
    replicate_index: int,
    p_percent: int,
) -> None:
    spec = DATASET_SPECS[dataset]
    cell = (
        dataset_root
        / f"replicate-{replicate_index:04d}"
        / f"pct-{p_percent:03d}"
    )
    cell.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model_index, model in enumerate(MODELS):
        for split, seed in spec["identities"]:
            baseline = 0.4 + 0.1 * model_index
            treatment = baseline + p_percent / 1000 + replicate_index / 10000
            control = baseline + replicate_index / 10000
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "replicate_index": replicate_index,
                    "replicate_seed": spec["seed_base"] + replicate_index,
                    "p_percent": p_percent,
                    "n_selected_tail_nodes": (TAIL_COUNT * p_percent) // 100,
                    "split": split,
                    "seed": seed,
                    "n_tail_test": 100,
                    "tail_test_class_composition": '{"0":50,"1":50}',
                    "treatment_macro_f1": treatment,
                    "control_macro_f1": control,
                    "treatment_minus_control_macro_f1": (
                        treatment - control
                    ),
                    "treatment_accuracy": treatment,
                    "control_accuracy": control,
                    "treatment_minus_control_accuracy": treatment - control,
                    "checkpoint_sha256": checkpoint_sha256(
                        dataset, model, split, seed
                    ),
                }
            )
    metrics_path = cell / "checkpoint_metrics.csv"
    write_csv(metrics_path, rows)

    ranking = np.roll(
        np.arange(100, 100 + TAIL_COUNT, dtype=np.int64),
        replicate_index - 1,
    )
    target_count = (TAIL_COUNT * p_percent) // 100
    targets = ranking[:target_count]
    graphs_path = cell / "graphs.npz"
    np.savez_compressed(
        graphs_path,
        treatment_edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        control_edge_index=np.asarray([[0, 2], [2, 0]], dtype=np.int64),
        targets=targets,
        target_order=targets[::-1],
        tail_ranking=ranking,
    )
    swaps_path = cell / "swaps.json"
    swaps_path.write_text("{}", encoding="utf-8")
    predictions_path = cell / "predictions.npz"
    np.savez_compressed(predictions_path, predictions=np.asarray([1]))

    original_h = 0.2 if spec["desired_same"] else 0.8
    treatment_change = 0.1 if spec["desired_same"] else -0.1
    audit = {
        "edge_count_preserved": True,
        "treatment_degrees_preserved": True,
        "control_degrees_preserved": True,
        "treatment_homophily_step_exact": True,
        "control_target_homophily_unchanged": True,
        "treatment_unselected_tail_homophily_unchanged": True,
        "control_full_tail_homophily_unchanged": True,
        "each_target_processed_once": True,
        "treatment_control_swap_counts_match": True,
        "num_targets": target_count,
        "original_mean_target_h": original_h,
        "treatment_mean_target_h": original_h + treatment_change,
        "control_mean_target_h": original_h,
        "mean_treatment_change": treatment_change,
        "mean_control_change": 0.0,
        "original_edge_sha256": graph_hash(dataset, "original"),
        "treatment_edge_sha256": graph_hash(
            dataset, f"{replicate_index}/{p_percent}/treatment"
        ),
        "control_edge_sha256": graph_hash(
            dataset, f"{replicate_index}/{p_percent}/control"
        ),
    }
    metadata = {
        "schema_version": 1,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": dataset,
        "models": list(MODELS),
        "tail_side": spec["tail_side"],
        "desired_same": spec["desired_same"],
        "treatment_direction": spec["direction"],
        "generator": spec["generator"],
        "replicate_index": replicate_index,
        "replicate_seed": spec["seed_base"] + replicate_index,
        "p_percent": p_percent,
        "tail_count": TAIL_COUNT,
        "requested_targets": target_count,
        "selected_targets": target_count,
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "target_selection": (
            "prefix of the replicate's frozen random tail-node ranking"
        ),
        "graph_start": f"registered original {dataset} graph",
        "generation_attempt": 0,
        "generation_retry_failures": [],
        "audit": audit,
        "checkpoint_identities": [
            {"split": split, "seed": seed}
            for split, seed in spec["identities"]
        ],
        "n_checkpoints_per_model": 30,
        "n_checkpoint_rows": 60,
        "metric_population": "fixed-full-tail-test",
        "code_version": CODE_VERSION,
        "seed_base": spec["seed_base"],
        "experiment_manifest_sha256": manifest_sha256,
        "output_sha256": {
            "graphs.npz": file_sha256(graphs_path),
            "swaps.json": file_sha256(swaps_path),
            "checkpoint_metrics.csv": file_sha256(metrics_path),
            "predictions.npz": file_sha256(predictions_path),
        },
    }
    (cell / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def write_panel(
    actual_root: Path,
    dataset: str,
    *,
    omit: tuple[int, int] | None = None,
) -> Path:
    dataset_root, manifest_sha256 = write_manifest(actual_root, dataset)
    write_baseline(dataset_root, dataset, manifest_sha256)
    for replicate_index in GRAPH_REPLICATES:
        for p_percent in P_PERCENTAGES[1:]:
            if (replicate_index, p_percent) == omit:
                continue
            write_dose_cell(
                dataset_root,
                dataset,
                manifest_sha256,
                replicate_index=replicate_index,
                p_percent=p_percent,
            )
    return dataset_root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("dataset", ["arxiv-year", "ogbn-arxiv"])
def test_analyze_accepts_exact_dataset_identity_panel_and_writes_outputs(
    tmp_path: Path, dataset: str
) -> None:
    actual_root = tmp_path / "actual"
    write_panel(actual_root, dataset)
    output_root = tmp_path / "results" / dataset

    outputs = NAMESPACE["analyze"](
        actual_root, output_root, dataset=dataset
    )

    assert set(outputs) == {
        "replicate_summary",
        "dose_summary",
        "figure",
        "analysis_manifest",
    }
    assert all(path.is_file() for path in outputs.values())
    assert outputs["figure"].name == f"{dataset}_edge_dose_response.png"
    assert len(read_csv(outputs["replicate_summary"])) == 420
    dose_rows = read_csv(outputs["dose_summary"])
    assert len(dose_rows) == 84
    assert plt.imread(outputs["figure"]).size > 0
    p_zero = [
        row
        for row in dose_rows
        if row["model"] == "gcn"
        and row["condition"] == "treatment"
        and int(row["p_percent"]) == 0
    ]
    assert float(p_zero[0]["mean_macro_f1"]) == pytest.approx(0.4)
    assert float(
        p_zero[0]["sd_macro_f1_across_graph_replicates"]
    ) == pytest.approx(0.0)
    manifest = json.loads(
        outputs["analysis_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["design"]["dataset"] == dataset
    assert len(manifest["design"]["checkpoint_identities"]) == 30
    assert manifest["validation"]["complete_nonzero_graph_cells"] == 100
    assert manifest["validation"]["nonzero_condition_scores"] == 12000


@pytest.mark.parametrize(
    ("dataset", "treatment_h", "change", "wanted"),
    [
        ("arxiv-year", 0.9, 0.1, "decrease"),
        ("ogbn-arxiv", 0.1, -0.1, "increase"),
    ],
)
def test_wrong_treatment_direction_is_rejected(
    tmp_path: Path,
    dataset: str,
    treatment_h: float,
    change: float,
    wanted: str,
) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, dataset)
    metadata_path = (
        dataset_root / "replicate-0001" / "pct-005" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["audit"]["treatment_mean_target_h"] = treatment_h
    metadata["audit"]["mean_treatment_change"] = change
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        RuntimeError, match=f"must {wanted} target homophily"
    ):
        NAMESPACE["load_inputs"](actual_root, dataset=dataset)


def test_missing_coordinated_audit_is_rejected(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, "ogbn-arxiv")
    metadata_path = (
        dataset_root / "replicate-0001" / "pct-005" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["audit"]["control_full_tail_homophily_unchanged"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="graph audits failed"):
        NAMESPACE["load_inputs"](actual_root, dataset="ogbn-arxiv")


def test_incomplete_five_by_twenty_grid_is_rejected(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    write_panel(actual_root, "ogbn-arxiv", omit=(5, 100))

    with pytest.raises(RuntimeError, match="Dose grid differs"):
        NAMESPACE["load_inputs"](actual_root, dataset="ogbn-arxiv")


def test_checkpoint_identity_drift_is_rejected(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, "ogbn-arxiv")
    cell = dataset_root / "replicate-0001" / "pct-005"
    metrics_path = cell / "checkpoint_metrics.csv"
    rows = read_csv(metrics_path)
    rows[-1]["seed"] = rows[0]["seed"]
    write_csv(metrics_path, rows)
    metadata_path = cell / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_sha256"]["checkpoint_metrics.csv"] = file_sha256(
        metrics_path
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="identities must be unique"):
        NAMESPACE["load_inputs"](actual_root, dataset="ogbn-arxiv")


def test_non_prefix_target_set_is_rejected(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, "arxiv-year")
    cell = dataset_root / "replicate-0001" / "pct-005"
    graphs_path = cell / "graphs.npz"
    with np.load(graphs_path, allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["targets"] = values["tail_ranking"][1:2]
    values["target_order"] = values["targets"].copy()
    np.savez_compressed(graphs_path, **values)
    metadata_path = cell / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_sha256"]["graphs.npz"] = file_sha256(graphs_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not the ranking prefix"):
        NAMESPACE["load_inputs"](actual_root, dataset="arxiv-year")


def test_different_p100_target_population_is_rejected(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, "ogbn-arxiv")
    for p_percent in P_PERCENTAGES[1:]:
        cell = dataset_root / "replicate-0005" / f"pct-{p_percent:03d}"
        graphs_path = cell / "graphs.npz"
        with np.load(graphs_path, allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        ranking = values["tail_ranking"] + 1000
        target_count = (TAIL_COUNT * p_percent) // 100
        values["tail_ranking"] = ranking
        values["targets"] = ranking[:target_count]
        values["target_order"] = values["targets"][::-1]
        np.savez_compressed(graphs_path, **values)
        metadata_path = cell / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["output_sha256"]["graphs.npz"] = file_sha256(graphs_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="p=100 target population differs"):
        NAMESPACE["load_inputs"](actual_root, dataset="ogbn-arxiv")


def test_raw_outputs_are_hashed_once_during_validation(
    tmp_path: Path,
) -> None:
    actual_root = tmp_path / "actual"
    dataset_root = write_panel(actual_root, "arxiv-year")
    observed: dict[Path, int] = defaultdict(int)
    original = NAMESPACE["load_inputs"].__globals__["file_sha256"]

    def counting_hash(path: Path) -> str:
        observed[path.resolve()] += 1
        return original(path)

    NAMESPACE["load_inputs"].__globals__["file_sha256"] = counting_hash
    try:
        NAMESPACE["load_inputs"](actual_root, dataset="arxiv-year")
    finally:
        NAMESPACE["load_inputs"].__globals__["file_sha256"] = original

    representative = (
        dataset_root
        / "replicate-0001"
        / "pct-005"
        / "predictions.npz"
    ).resolve()
    assert observed[representative] == 1


def test_cli_requires_dataset_and_explicit_roots() -> None:
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"]([])
