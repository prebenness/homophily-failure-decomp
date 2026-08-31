"""Build the RQ2 within-class and class-composition analyses.

The direct RQ2 calculation compares tail and non-tail accuracy within each
true class represented on both sides. The class contrasts receive equal
weight within a run, and the run-level contrast is

    mean over classes(non-tail accuracy - tail accuracy).

The accompanying class-composition analysis includes every true class present
in the tail. It evaluates the whole test over those classes, then reweights the
tail's true-class composition to their proportions in the whole test set.
Accuracy and macro-F1 are calculated for the whole test and the reweighted
tail. The run-level contrast is

    whole-test score - class-reweighted tail score,

so a positive value means that the tail remains harder after adjusting for
true-class composition.  Cell summaries are the mean of the 30 run-level
contrasts with paired percentile-bootstrap intervals.

The raw archive is read-only.  This script validates the registered run panel
and fixed node-level inputs, then writes reproducible CSV and LaTeX outputs
below ``results/main/rq2``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from hfd.analysis_datasets import (
    ADDITIONAL_DATASET_ORDER,
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    RegisteredRun,
    load_registered_runs,
    validate_registered_run_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = list(ORIGINAL_DATASET_ORDER)
MODEL_ORDER = [
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "linkx",
    "acm-gcn",
]
PAPER_MODEL_ORDER = [
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
]
MODEL_DISPLAY = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "linkx": "LINKX",
    "acm-gcn": "ACM-GCN",
}
PREPARATION_ORDER = ["erm", "last", "head", "full"]
EXPECTED_RUNS_PER_CELL = 30
COMPARISON_TOLERANCE = 1e-12
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_716
ANALYSIS_VERSION = "rq2-within-class-and-composition-adjusted-v4"
DEFAULT_OUTPUT_DIRS = {
    AnalysisDatasetMode.ORIGINAL: REPO_ROOT / "results" / "main" / "rq2",
    AnalysisDatasetMode.ADDITIONAL: (
        REPO_ROOT / "results" / "additional" / "rq2"
    ),
}
DEFAULT_GAP_TABLES = {
    AnalysisDatasetMode.ORIGINAL: (
        REPO_ROOT / "results" / "main" / "aggregates" / "erm_tail_gaps.csv"
    ),
    AnalysisDatasetMode.ADDITIONAL: (
        REPO_ROOT
        / "results"
        / "additional"
        / "aggregates"
        / "erm_tail_gaps.csv"
    ),
}
DEFAULT_ACCURACY_GAP_TABLES = {
    AnalysisDatasetMode.ORIGINAL: (
        REPO_ROOT
        / "results"
        / "main"
        / "rq1"
        / "accuracy"
        / "whole_vs_tail_accuracy.csv"
    ),
    AnalysisDatasetMode.ADDITIONAL: (
        REPO_ROOT
        / "results"
        / "additional"
        / "rq1"
        / "accuracy"
        / "whole_vs_tail_accuracy.csv"
    ),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sample_std(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(values.std(ddof=1))


def metrics_from_joint_confusion(
    joint: np.ndarray,
    class_ids: np.ndarray,
) -> tuple[float, float]:
    """Return accuracy and macro-F1 for a weighted joint confusion matrix.

    Rows correspond to the included true classes in ``class_ids``.  Columns
    correspond to every registered prediction label.  Prediction labels
    outside ``class_ids`` therefore contribute false negatives, while true
    classes outside ``class_ids`` are excluded from the population.
    """
    joint = np.asarray(joint, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if joint.ndim != 2 or class_ids.ndim != 1:
        raise ValueError("Joint confusion and class ids have invalid dimensions")
    if joint.shape[0] != class_ids.size or class_ids.size == 0:
        raise ValueError("Joint confusion must have one non-empty row per class")
    if np.any(class_ids < 0) or np.any(class_ids >= joint.shape[1]):
        raise ValueError("An included class is outside the prediction columns")
    if np.any(joint < 0.0) or not np.all(np.isfinite(joint)):
        raise ValueError("Joint confusion contains an invalid weight")
    if not np.isclose(joint.sum(), 1.0, rtol=0.0, atol=COMPARISON_TOLERANCE):
        raise ValueError("Joint confusion weights must sum to one")

    true_positives = joint[np.arange(class_ids.size), class_ids]
    accuracy_value = float(true_positives.sum())
    f1_values = []
    for row_index, class_id in enumerate(class_ids):
        true_positive = float(joint[row_index, class_id])
        false_negative = float(joint[row_index].sum() - true_positive)
        false_positive = float(joint[:, class_id].sum() - true_positive)
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_values.append(
            0.0 if denominator == 0.0 else 2.0 * true_positive / denominator
        )
    return accuracy_value, float(np.mean(f1_values))


def class_adjusted_metrics(
    class_ids: np.ndarray,
    whole_prediction_counts: np.ndarray,
    tail_prediction_counts: np.ndarray,
) -> dict:
    """Reweight tail confusion rows to the whole-test class composition."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    whole_counts = np.asarray(whole_prediction_counts, dtype=np.float64)
    tail_counts = np.asarray(tail_prediction_counts, dtype=np.float64)
    if whole_counts.shape != tail_counts.shape or whole_counts.ndim != 2:
        raise ValueError("Whole and tail prediction counts must have equal matrices")
    if whole_counts.shape[0] != class_ids.size or class_ids.size == 0:
        raise ValueError("Prediction-count matrices require a non-empty class set")
    if np.any(whole_counts < 0.0) or np.any(tail_counts < 0.0):
        raise ValueError("Prediction counts cannot be negative")
    whole_support = whole_counts.sum(axis=1)
    tail_support = tail_counts.sum(axis=1)
    if np.any(whole_support <= 0.0) or np.any(tail_support <= 0.0):
        raise ValueError("Every class requires whole and tail support")

    reference_weights = whole_support / whole_support.sum()
    tail_rows = tail_counts / tail_support[:, None]
    whole_joint = whole_counts / whole_counts.sum()
    adjusted_tail_joint = reference_weights[:, None] * tail_rows
    whole_accuracy, whole_macro_f1 = metrics_from_joint_confusion(
        whole_joint, class_ids
    )
    tail_accuracy, tail_macro_f1 = metrics_from_joint_confusion(
        adjusted_tail_joint, class_ids
    )

    return {
        "reference_weights": reference_weights,
        "whole_accuracy": whole_accuracy,
        "adjusted_tail_accuracy": tail_accuracy,
        "adjusted_accuracy_gap": whole_accuracy - tail_accuracy,
        "whole_macro_f1": whole_macro_f1,
        "adjusted_tail_macro_f1": tail_macro_f1,
        "adjusted_macro_f1_gap": whole_macro_f1 - tail_macro_f1,
    }


def rq1_matched_bootstrap_interval(
    values: np.ndarray,
    dataset: str,
    model: str,
) -> tuple[float, float]:
    """Use the complete-run resamples used by the RQ1 tail-gap tables."""
    key = f"{BOOTSTRAP_SEED}:{dataset}:{model}:erm-tail-gap".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values.size,
        size=(BOOTSTRAP_RESAMPLES, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    return float(lower), float(upper)


def within_class_bootstrap_seed(
    dataset: str,
    model: str,
    preparation: str,
) -> int:
    """Return the seed used by the original within-class accuracy table."""
    key = (
        f"{BOOTSTRAP_SEED}:{dataset}:{model}:{preparation}:"
        "class-macro-accuracy"
    ).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def within_class_bootstrap_interval(
    values: np.ndarray,
    dataset: str,
    model: str,
    preparation: str,
) -> tuple[float, float]:
    """Reproduce the complete-run bootstrap used by the original RQ2 table."""
    seed = within_class_bootstrap_seed(dataset, model, preparation)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values.size,
        size=(BOOTSTRAP_RESAMPLES, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    return float(lower), float(upper)


def mean_within_class_accuracy_contrast(rows: list[dict]) -> float:
    """Give every supported class equal weight in one run."""
    if not rows:
        raise ValueError("A within-class contrast requires at least one class")
    values = np.asarray(
        [row["contrast_non_tail_minus_tail"] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Within-class accuracy contrasts must be finite")
    return float(values.mean())


def within_class_supported_rows(rows: Iterable[dict]) -> list[dict]:
    """Retain classes represented among both tail and non-tail test nodes."""
    return sorted(
        (
            row
            for row in rows
            if int(row["n_tail_test"]) > 0
            and int(row["n_non_tail_test"]) > 0
        ),
        key=lambda row: int(row["class"]),
    )


def validate_fixed_checksums(
    config: dict, manifest: dict, run_id: str
) -> None:
    recorded = config["fixed_input_checksums"]
    expected = manifest["array_checksums_sha256"]
    split = config["split"]
    keys = ("labels", "merged_bin", "tail_mask", f"{split}.test")
    for key in keys:
        if recorded.get(key) != expected.get(key):
            raise RuntimeError(
                f"{run_id}: fixed-input checksum differs from manifest for {key}"
            )


def check_array(
    array: np.ndarray,
    expected_checksum: str,
    run_id: str,
    name: str,
) -> None:
    if array_sha256(array) != expected_checksum:
        raise RuntimeError(f"{run_id}: checksum mismatch for {name}")


def recompute_run_rows(
    run_data_path: Path,
    config: dict,
    manifest: dict,
) -> tuple[list[dict], tuple[tuple[int, int, int], ...], str]:
    run_id = config["run_id"]
    split = config["split"]
    expected_checksums = manifest["array_checksums_sha256"]
    with np.load(run_data_path, allow_pickle=False) as saved:
        required = {"labels", "predictions", "test_idx", "merged_bin", "tail_mask"}
        if not required.issubset(saved.files):
            raise RuntimeError(
                f"{run_id}: run_data.npz is missing "
                f"{sorted(required - set(saved.files))}"
            )
        labels = np.asarray(saved["labels"], dtype=np.int64)
        predictions = np.asarray(saved["predictions"], dtype=np.int64)
        test_idx = np.asarray(saved["test_idx"], dtype=np.int64)
        merged_bin = np.asarray(saved["merged_bin"], dtype=np.int64)
        tail_mask = np.asarray(saved["tail_mask"], dtype=np.bool_)

    if not (
        labels.shape == predictions.shape == merged_bin.shape == tail_mask.shape
    ):
        raise RuntimeError(f"{run_id}: full-node arrays have inconsistent shapes")
    if test_idx.ndim != 1 or np.unique(test_idx).size != test_idx.size:
        raise RuntimeError(f"{run_id}: test indices are not unique one-dimensional ids")
    if np.any(test_idx < 0) or np.any(test_idx >= labels.size):
        raise RuntimeError(f"{run_id}: test index is outside the node array")
    n_classes = int(manifest["graph"]["num_classes"])
    if np.any(predictions < 0) or np.any(predictions >= n_classes):
        raise RuntimeError(f"{run_id}: prediction is outside the registered class set")

    check_array(labels, expected_checksums["labels"], run_id, "labels")
    check_array(
        test_idx,
        expected_checksums[f"{split}.test"],
        run_id,
        "test_idx",
    )
    check_array(
        merged_bin,
        expected_checksums["merged_bin"],
        run_id,
        "merged_bin",
    )
    check_array(
        tail_mask,
        expected_checksums["tail_mask"],
        run_id,
        "tail_mask",
    )

    tail_test = test_idx[tail_mask[test_idx]]
    non_tail_test = test_idx[
        (merged_bin[test_idx] >= 0) & ~tail_mask[test_idx]
    ]
    result = []
    support_signature = []
    tail_classes, tail_counts = np.unique(labels[tail_test], return_counts=True)
    for class_id_raw, n_tail_raw in zip(tail_classes, tail_counts):
        class_id = int(class_id_raw)
        n_tail = int(n_tail_raw)
        tail_class = tail_test[labels[tail_test] == class_id]
        whole_class = test_idx[labels[test_idx] == class_id]
        non_tail_class = non_tail_test[labels[non_tail_test] == class_id]
        n_non_tail = int(non_tail_class.size)
        tail_accuracy = float(
            np.mean(predictions[tail_class] == labels[tail_class])
        )
        non_tail_accuracy = (
            float(np.mean(predictions[non_tail_class] == labels[non_tail_class]))
            if n_non_tail > 0
            else None
        )
        accuracy_contrast = (
            non_tail_accuracy - tail_accuracy
            if non_tail_accuracy is not None
            else None
        )
        whole_prediction_counts = np.bincount(
            predictions[whole_class], minlength=n_classes
        ).astype(np.int64)
        tail_prediction_counts = np.bincount(
            predictions[tail_class], minlength=n_classes
        ).astype(np.int64)
        if int(whole_prediction_counts.sum()) != int(whole_class.size):
            raise RuntimeError(f"{run_id}: whole prediction counts do not sum")
        if int(tail_prediction_counts.sum()) != n_tail:
            raise RuntimeError(f"{run_id}: tail prediction counts do not sum")
        support_signature.append((class_id, n_tail, n_non_tail))
        result.append(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "preparation": config["preparation"],
                "split": split,
                "seed": int(config["seed"]),
                "class": class_id,
                "tail_test_nodes_all_classes": int(tail_test.size),
                "whole_test_nodes_all_classes": int(test_idx.size),
                "registered_classes": n_classes,
                "n_whole_test": int(whole_class.size),
                "n_tail_test": n_tail,
                "n_non_tail_test": n_non_tail,
                "whole_prediction_counts": tuple(
                    int(value) for value in whole_prediction_counts
                ),
                "tail_prediction_counts": tuple(
                    int(value) for value in tail_prediction_counts
                ),
                "tail_accuracy": tail_accuracy,
                "non_tail_accuracy": non_tail_accuracy,
                "contrast_non_tail_minus_tail": accuracy_contrast,
            }
        )
    return result, tuple(support_signature), array_sha256(predictions)


def read_gap_status(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["model"])
            mean_gap = float(row["mean_tail_gap"])
            ci_lower = float(row["bootstrap_ci_lower"])
            ci_upper = float(row["bootstrap_ci_upper"])
            result[key] = {
                "erm_mean_tail_gap": mean_gap,
                "erm_tail_gap_ci_lower": ci_lower,
                "erm_tail_gap_ci_upper": ci_upper,
                "erm_tail_gap_ci_excludes_zero": (
                    ci_lower > 0.0 or ci_upper < 0.0
                ),
                "positive_nonzero_erm_tail_gap": mean_gap > 0.0 and ci_lower > 0.0,
            }
    expected = {
        (dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER
    }
    if set(result) != expected:
        raise RuntimeError(
            f"ERM gap table panel mismatch: missing={sorted(expected - set(result))}, "
            f"unexpected={sorted(set(result) - expected)}"
        )
    return result


def read_accuracy_gap_status(path: Path) -> dict[tuple[str, str], dict]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["model"])
            mean_gap = float(row["mean_gap_whole_minus_tail"])
            ci_lower = float(row["bootstrap_ci_lower_gap"])
            ci_upper = float(row["bootstrap_ci_upper_gap"])
            result[key] = {
                "erm_accuracy_mean_gap": mean_gap,
                "erm_accuracy_gap_ci_lower": ci_lower,
                "erm_accuracy_gap_ci_upper": ci_upper,
                "erm_accuracy_gap_ci_excludes_zero": (
                    ci_lower > 0.0 or ci_upper < 0.0
                ),
            }
    expected = {
        (dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER
    }
    if set(result) != expected:
        raise RuntimeError(
            "ERM accuracy-gap table panel mismatch: "
            f"missing={sorted(expected - set(result))}, "
            f"unexpected={sorted(set(result) - expected)}"
        )
    return result


def discover_and_validate_runs(
    raw_root: Path,
    manifests: dict[str, dict],
    registrations: tuple[RegisteredRun, ...] | None = None,
) -> tuple[list[dict], dict]:
    run_rows = []
    panels: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    fixed_support: dict[
        tuple[str, str], tuple[tuple[int, int, int], ...]
    ] = {}
    seen_run_ids = set()
    input_digest = hashlib.sha256()
    training_versions = set()
    all_configs: list[dict] = []

    for run_dir in sorted(raw_root.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir() or run_dir.name == "historical-records":
            continue
        config_path = run_dir / "config.json"
        run_data_path = run_dir / "run_data.npz"
        if not config_path.is_file() or not run_data_path.is_file():
            raise FileNotFoundError(f"Incomplete raw run directory: {run_dir}")
        config = read_json(config_path)
        all_configs.append(config)
        run_id = config["run_id"]
        if run_id != run_dir.name:
            raise RuntimeError(f"Run-id and directory-name mismatch: {run_dir}")
        if run_id in seen_run_ids:
            raise RuntimeError(f"Duplicate run id: {run_id}")
        seen_run_ids.add(run_id)

        dataset = config["dataset"]
        model = config["model"]
        preparation = config["preparation"]
        if dataset not in manifests:
            raise RuntimeError(f"Unexpected dataset in raw archive: {dataset}")
        if model not in MODEL_ORDER:
            raise RuntimeError(f"Unexpected model in raw archive: {model}")
        if preparation not in PREPARATION_ORDER:
            raise RuntimeError(
                f"Unexpected preparation in raw archive: {preparation}"
            )
        validate_fixed_checksums(config, manifests[dataset], run_id)
        rows, support_signature, prediction_checksum = recompute_run_rows(
            run_data_path, config, manifests[dataset]
        )

        support_key = (dataset, config["split"])
        previous = fixed_support.setdefault(support_key, support_signature)
        if previous != support_signature:
            raise RuntimeError(
                f"{run_id}: class support differs within fixed split {support_key}"
            )
        run_rows.extend(rows)
        pair = (config["split"], int(config["seed"]))
        panel_key = (dataset, model, preparation)
        if pair in panels[panel_key]:
            raise RuntimeError(f"{panel_key}: duplicate split-seed pair {pair}")
        panels[panel_key].add(pair)
        training_versions.add(config["code_version"])
        input_digest.update(run_id.encode("utf-8"))
        input_digest.update(file_sha256(config_path).encode("ascii"))
        input_digest.update(prediction_checksum.encode("ascii"))

    if registrations is not None:
        validate_registered_run_configs(
            all_configs,
            registrations,
            context="RQ2 input",
        )

    for dataset in DATASET_ORDER:
        manifest = manifests[dataset]
        expected_pairs = {
            (split["split_id"], int(seed))
            for split in manifest["splits"]
            for seed in manifest["training_seeds"]
        }
        if len(expected_pairs) != EXPECTED_RUNS_PER_CELL:
            raise RuntimeError(
                f"{dataset}: manifest has {len(expected_pairs)} run pairs; "
                f"expected {EXPECTED_RUNS_PER_CELL}"
            )
        for model in MODEL_ORDER:
            for preparation in PREPARATION_ORDER:
                key = (dataset, model, preparation)
                observed = panels.get(key, set())
                if observed != expected_pairs:
                    raise RuntimeError(
                        f"{key}: run panel mismatch; "
                        f"missing={sorted(expected_pairs - observed)}, "
                        f"unexpected={sorted(observed - expected_pairs)}"
                    )

    expected_run_count = (
        len(DATASET_ORDER)
        * len(MODEL_ORDER)
        * len(PREPARATION_ORDER)
        * EXPECTED_RUNS_PER_CELL
    )
    if len(seen_run_ids) != expected_run_count:
        raise RuntimeError(
            f"Found {len(seen_run_ids)} runs; expected {expected_run_count}"
        )
    audit = {
        "run_count": len(seen_run_ids),
        "run_level_class_rows": len(run_rows),
        "combined_config_and_predictions_sha256": input_digest.hexdigest(),
        "training_code_versions": sorted(training_versions),
    }
    return run_rows, audit


def build_within_class_accuracy_run_contrasts(
    run_rows: list[dict],
    manifests: dict[str, dict],
    gap_status: dict[tuple[str, str], dict],
) -> list[dict]:
    """Build the direct non-tail-minus-tail RQ2 contrast for every ERM run."""
    grouped: dict[tuple[str, str, str, str, int], list[dict]] = defaultdict(list)
    for row in run_rows:
        if row["preparation"] != "erm":
            continue
        grouped[
            (
                row["dataset"],
                row["model"],
                row["preparation"],
                row["split"],
                row["seed"],
            )
        ].append(row)

    result = []
    for key, rows in grouped.items():
        dataset, model, preparation, split, seed = key
        rows = within_class_supported_rows(rows)
        if not rows:
            raise RuntimeError(f"{key}: no class has support on both sides")
        class_ids = [int(row["class"]) for row in rows]
        if len(class_ids) != len(set(class_ids)):
            raise RuntimeError(f"{key}: duplicate class in within-class aggregation")
        tail_totals = {int(row["tail_test_nodes_all_classes"]) for row in rows}
        if len(tail_totals) != 1:
            raise RuntimeError(f"{key}: inconsistent total tail-node count")
        total_tail_nodes = tail_totals.pop()
        supported_tail_nodes = sum(int(row["n_tail_test"]) for row in rows)
        result.append(
            {
                "dataset_order": DATASET_ORDER.index(dataset),
                "dataset": dataset,
                "dataset_direction": manifests[dataset]["local_homophily"][
                    "direction"
                ],
                "tail_side": manifests[dataset]["local_homophily"]["tail_side"],
                "model_order": MODEL_ORDER.index(model),
                "model": model,
                "preparation_order": PREPARATION_ORDER.index(preparation),
                "preparation": preparation,
                "split": split,
                "seed": int(seed),
                "supported_classes": len(rows),
                "supported_class_ids": ";".join(str(value) for value in class_ids),
                "supported_tail_nodes": supported_tail_nodes,
                "total_tail_nodes": total_tail_nodes,
                "supported_tail_node_fraction": (
                    supported_tail_nodes / total_tail_nodes
                ),
                "class_macro_accuracy_contrast": (
                    mean_within_class_accuracy_contrast(rows)
                ),
                **gap_status[(dataset, model)],
            }
        )

    expected_rows = (
        len(DATASET_ORDER)
        * len(MODEL_ORDER)
        * EXPECTED_RUNS_PER_CELL
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Found {len(result)} within-class run contrasts; "
            f"expected {expected_rows}"
        )
    return sorted(
        result,
        key=lambda row: (
            row["dataset_order"],
            row["model_order"],
            row["preparation_order"],
            row["split"],
            row["seed"],
        ),
    )


def build_within_class_accuracy_cell_summaries(
    run_rows: list[dict],
) -> list[dict]:
    """Summarise the 30 direct RQ2 run contrasts in each ERM cell."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["model"], row["preparation"])].append(row)

    result = []
    for key, rows in grouped.items():
        dataset, model, preparation = key
        rows = sorted(rows, key=lambda row: (row["split"], row["seed"]))
        if len(rows) != EXPECTED_RUNS_PER_CELL:
            raise RuntimeError(
                f"{key}: found {len(rows)} within-class run contrasts; "
                f"expected {EXPECTED_RUNS_PER_CELL}"
            )
        values = np.asarray(
            [row["class_macro_accuracy_contrast"] for row in rows],
            dtype=np.float64,
        )
        lower, upper = within_class_bootstrap_interval(
            values, dataset, model, preparation
        )
        supported_classes = np.asarray(
            [row["supported_classes"] for row in rows], dtype=np.int64
        )
        tail_coverage = np.asarray(
            [row["supported_tail_node_fraction"] for row in rows],
            dtype=np.float64,
        )
        result.append(
            {
                "dataset_order": rows[0]["dataset_order"],
                "dataset": dataset,
                "dataset_direction": rows[0]["dataset_direction"],
                "tail_side": rows[0]["tail_side"],
                "model_order": rows[0]["model_order"],
                "model": model,
                "preparation_order": rows[0]["preparation_order"],
                "preparation": preparation,
                "n_runs": len(rows),
                "mean_supported_classes": float(supported_classes.mean()),
                "min_supported_classes": int(supported_classes.min()),
                "max_supported_classes": int(supported_classes.max()),
                "mean_supported_tail_node_fraction": float(
                    tail_coverage.mean()
                ),
                "mean_class_macro_accuracy_contrast": float(values.mean()),
                "sample_std_class_macro_accuracy_contrast": sample_std(values),
                "bootstrap_ci_lower_class_macro_accuracy": lower,
                "bootstrap_ci_upper_class_macro_accuracy": upper,
                "class_macro_accuracy_ci_excludes_zero": (
                    lower > 0.0 or upper < 0.0
                ),
                "erm_mean_tail_gap": rows[0]["erm_mean_tail_gap"],
                "erm_tail_gap_ci_lower": rows[0]["erm_tail_gap_ci_lower"],
                "erm_tail_gap_ci_upper": rows[0]["erm_tail_gap_ci_upper"],
                "erm_tail_gap_ci_excludes_zero": rows[0][
                    "erm_tail_gap_ci_excludes_zero"
                ],
                "positive_nonzero_erm_tail_gap": rows[0][
                    "positive_nonzero_erm_tail_gap"
                ],
            }
        )

    expected_rows = len(DATASET_ORDER) * len(MODEL_ORDER)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Found {len(result)} within-class cell summaries; "
            f"expected {expected_rows}"
        )
    return sorted(
        result,
        key=lambda row: (
            row["dataset_order"],
            row["model_order"],
            row["preparation_order"],
        ),
    )


def build_run_level_contrasts(
    run_rows: list[dict],
    manifests: dict[str, dict],
    gap_status: dict[tuple[str, str], dict],
    accuracy_gap_status: dict[tuple[str, str], dict],
) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, int], list[dict]] = defaultdict(list)
    for row in run_rows:
        if row["preparation"] != "erm":
            continue
        grouped[
            (
                row["dataset"],
                row["model"],
                row["preparation"],
                row["split"],
                row["seed"],
            )
        ].append(row)

    result = []
    for key, rows in grouped.items():
        dataset, model, preparation, split, seed = key
        rows = sorted(rows, key=lambda row: int(row["class"]))
        class_ids = [int(row["class"]) for row in rows]
        if len(class_ids) != len(set(class_ids)):
            raise RuntimeError(f"{key}: duplicate class in run-level aggregation")
        registered_classes = {int(row["registered_classes"]) for row in rows}
        if len(registered_classes) != 1:
            raise RuntimeError(f"{key}: inconsistent registered class count")
        n_classes = registered_classes.pop()
        tail_totals = {int(row["tail_test_nodes_all_classes"]) for row in rows}
        if len(tail_totals) != 1:
            raise RuntimeError(f"{key}: inconsistent total tail-node count")
        total_tail_nodes = tail_totals.pop()
        tail_nodes = sum(int(row["n_tail_test"]) for row in rows)
        if tail_nodes != total_tail_nodes:
            raise RuntimeError(f"{key}: tail-present class rows omit tail nodes")
        whole_counts = np.asarray(
            [row["whole_prediction_counts"] for row in rows],
            dtype=np.int64,
        )
        tail_counts = np.asarray(
            [row["tail_prediction_counts"] for row in rows],
            dtype=np.int64,
        )
        if whole_counts.shape != (len(rows), n_classes):
            raise RuntimeError(f"{key}: invalid whole prediction-count matrix")
        if tail_counts.shape != (len(rows), n_classes):
            raise RuntimeError(f"{key}: invalid tail prediction-count matrix")
        adjusted = class_adjusted_metrics(
            np.asarray(class_ids, dtype=np.int64),
            whole_counts,
            tail_counts,
        )
        reference_weights = adjusted.pop("reference_weights")
        result.append(
            {
                "dataset_order": DATASET_ORDER.index(dataset),
                "dataset": dataset,
                "dataset_direction": manifests[dataset]["local_homophily"][
                    "direction"
                ],
                "tail_side": manifests[dataset]["local_homophily"]["tail_side"],
                "model_order": MODEL_ORDER.index(model),
                "model": model,
                "preparation_order": PREPARATION_ORDER.index(preparation),
                "preparation": preparation,
                "split": split,
                "seed": int(seed),
                "tail_classes": len(rows),
                "tail_class_ids": ";".join(str(value) for value in class_ids),
                "whole_reference_weights": ";".join(
                    f"{value:.17g}" for value in reference_weights
                ),
                **adjusted,
                **gap_status[(dataset, model)],
                **accuracy_gap_status[(dataset, model)],
            }
        )
    expected_rows = (
        len(DATASET_ORDER)
        * len(MODEL_ORDER)
        * EXPECTED_RUNS_PER_CELL
    )
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Found {len(result)} run-level contrasts; expected {expected_rows}"
        )
    return sorted(
        result,
        key=lambda row: (
            row["dataset_order"],
            row["model_order"],
            row["preparation_order"],
            row["split"],
            row["seed"],
        ),
    )


def build_cell_summaries(run_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["dataset"], row["model"], row["preparation"])].append(row)

    result = []
    for key, rows in grouped.items():
        dataset, model, preparation = key
        rows = sorted(rows, key=lambda row: (row["split"], row["seed"]))
        if len(rows) != EXPECTED_RUNS_PER_CELL:
            raise RuntimeError(
                f"{key}: found {len(rows)} run-level contrasts; "
                f"expected {EXPECTED_RUNS_PER_CELL}"
            )
        accuracy = np.array(
            [row["adjusted_accuracy_gap"] for row in rows],
            dtype=np.float64,
        )
        f1 = np.array(
            [row["adjusted_macro_f1_gap"] for row in rows],
            dtype=np.float64,
        )
        accuracy_lower, accuracy_upper = rq1_matched_bootstrap_interval(
            accuracy, dataset, model
        )
        f1_lower, f1_upper = rq1_matched_bootstrap_interval(
            f1, dataset, model
        )
        result.append(
            {
                "dataset_order": rows[0]["dataset_order"],
                "dataset": dataset,
                "dataset_direction": rows[0]["dataset_direction"],
                "tail_side": rows[0]["tail_side"],
                "model_order": rows[0]["model_order"],
                "model": model,
                "preparation_order": rows[0]["preparation_order"],
                "preparation": preparation,
                "n_runs": len(rows),
                "mean_whole_accuracy": float(
                    np.mean([row["whole_accuracy"] for row in rows])
                ),
                "mean_adjusted_tail_accuracy": float(
                    np.mean([row["adjusted_tail_accuracy"] for row in rows])
                ),
                "mean_adjusted_accuracy_gap": float(accuracy.mean()),
                "sample_std_adjusted_accuracy_gap": sample_std(accuracy),
                "bootstrap_ci_lower_adjusted_accuracy_gap": accuracy_lower,
                "bootstrap_ci_upper_adjusted_accuracy_gap": accuracy_upper,
                "adjusted_accuracy_gap_ci_excludes_zero": (
                    accuracy_lower > 0.0 or accuracy_upper < 0.0
                ),
                "mean_whole_macro_f1": float(
                    np.mean([row["whole_macro_f1"] for row in rows])
                ),
                "mean_adjusted_tail_macro_f1": float(
                    np.mean([row["adjusted_tail_macro_f1"] for row in rows])
                ),
                "mean_adjusted_macro_f1_gap": float(f1.mean()),
                "sample_std_adjusted_macro_f1_gap": sample_std(f1),
                "bootstrap_ci_lower_adjusted_macro_f1_gap": f1_lower,
                "bootstrap_ci_upper_adjusted_macro_f1_gap": f1_upper,
                "adjusted_macro_f1_gap_ci_excludes_zero": (
                    f1_lower > 0.0 or f1_upper < 0.0
                ),
                "erm_mean_tail_gap": rows[0]["erm_mean_tail_gap"],
                "erm_tail_gap_ci_lower": rows[0]["erm_tail_gap_ci_lower"],
                "erm_tail_gap_ci_upper": rows[0]["erm_tail_gap_ci_upper"],
                "erm_tail_gap_ci_excludes_zero": rows[0][
                    "erm_tail_gap_ci_excludes_zero"
                ],
                "positive_nonzero_erm_tail_gap": rows[0][
                    "positive_nonzero_erm_tail_gap"
                ],
                "erm_accuracy_mean_gap": rows[0]["erm_accuracy_mean_gap"],
                "erm_accuracy_gap_ci_lower": rows[0][
                    "erm_accuracy_gap_ci_lower"
                ],
                "erm_accuracy_gap_ci_upper": rows[0][
                    "erm_accuracy_gap_ci_upper"
                ],
                "erm_accuracy_gap_ci_excludes_zero": rows[0][
                    "erm_accuracy_gap_ci_excludes_zero"
                ],
            }
        )
    expected_rows = len(DATASET_ORDER) * len(MODEL_ORDER)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Found {len(result)} cell summaries; expected {expected_rows}"
        )
    return sorted(
        result,
        key=lambda row: (
            row["dataset_order"],
            row["model_order"],
            row["preparation_order"],
        ),
    )


def latex_number(value: float, signed: bool = False) -> str:
    rounded = round(float(value), 3)
    if rounded == 0.0:
        return ".000" if tuple(DATASET_ORDER) == ADDITIONAL_DATASET_ORDER else "0.000"
    formatted = f"{rounded:+.3f}" if signed else f"{rounded:.3f}"
    if tuple(DATASET_ORDER) == ADDITIONAL_DATASET_ORDER:
        return (
            formatted.replace("+0.", "+.")
            .replace("-0.", "-.")
            .replace("0.", ".")
        )
    return formatted


def latex_cell(row: dict, metric: str) -> str:
    add_dagger = True
    if metric == "within_accuracy":
        mean = row["mean_class_macro_accuracy_contrast"]
        lower = row["bootstrap_ci_lower_class_macro_accuracy"]
        upper = row["bootstrap_ci_upper_class_macro_accuracy"]
        excludes_zero = row["class_macro_accuracy_ci_excludes_zero"]
        unadjusted_excludes_zero = True
        add_dagger = False
    elif metric == "accuracy":
        mean = row["mean_adjusted_accuracy_gap"]
        lower = row["bootstrap_ci_lower_adjusted_accuracy_gap"]
        upper = row["bootstrap_ci_upper_adjusted_accuracy_gap"]
        excludes_zero = row["adjusted_accuracy_gap_ci_excludes_zero"]
        unadjusted_excludes_zero = row[
            "erm_accuracy_gap_ci_excludes_zero"
        ]
    elif metric == "f1":
        mean = row["mean_adjusted_macro_f1_gap"]
        lower = row["bootstrap_ci_lower_adjusted_macro_f1_gap"]
        upper = row["bootstrap_ci_upper_adjusted_macro_f1_gap"]
        excludes_zero = row["adjusted_macro_f1_gap_ci_excludes_zero"]
        unadjusted_excludes_zero = row["erm_tail_gap_ci_excludes_zero"]
    else:
        raise ValueError(f"Unknown table metric: {metric}")
    macro = "gaincell" if excludes_zero else "uncertaingaincell"
    mean_text = latex_number(mean, signed=True)
    if add_dagger and not unadjusted_excludes_zero:
        mean_text += r"^{\dagger}"
    interval = f"[{latex_number(lower)},{latex_number(upper)}]"
    return f"\\{macro}{{{mean_text}}}{{{interval}}}"


def latex_table(cell_rows: list[dict], metric: str) -> list[str]:
    by_cell = {
        (row["dataset"], row["model"]): row
        for row in cell_rows
        if row["preparation"] == "erm"
    }
    expected = {
        (dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER
    }
    if set(by_cell) != expected:
        raise RuntimeError(f"Incomplete ERM table panel for {metric}")
    is_original = tuple(DATASET_ORDER) == tuple(ORIGINAL_DATASET_ORDER)
    if metric == "within_accuracy":
        caption = (
            "ERM within-class accuracy contrast, non-tail minus tail. "
            "Within each run, class contrasts are averaged equally over true "
            "classes represented in both the tail and non-tail subsets; "
            "cells report the mean "
            "of the 30 run-level contrasts and the 95\\% percentile-bootstrap "
            "interval. Positive values indicate lower tail accuracy."
        )
        label = "tab:rq2-class-macro-accuracy"
        if not is_original:
            caption = (
                "Per-class non-tail accuracy minus tail accuracy, averaged "
                "equally over classes. Each cell reports the mean over the "
                "30 runs and the 95\\% percentile-bootstrap interval. Grey "
                "cells denote confidence intervals that contain zero."
            )
    elif metric == "accuracy":
        caption = (
            "Class-composition-adjusted ERM accuracy gap, whole test minus "
            "tail. The tail is reweighted to the whole-test proportions of "
            "true classes present in the tail; the whole-test score uses the "
            "same class set. Entries are the "
            "mean and 95\\% bootstrap interval across 30 runs. Positive "
            "values indicate lower tail accuracy; grey cells include zero; "
            "$\\dagger$ marks cells whose unadjusted RQ1 accuracy interval "
            "includes zero."
        )
        label = "tab:rq2-adjusted-accuracy"
        if not is_original:
            caption = (
                "Class-adjusted accuracy tail gaps for all models on all "
                "datasets. Entries are the mean and 95\\% bootstrap interval "
                "across 30 runs. Grey cells include zero, and $\\dagger$ "
                "marks cells whose unadjusted accuracy tail gap includes zero."
            )
    elif metric == "f1":
        caption = (
            "Class-composition-adjusted ERM macro-F1 gap, whole test minus "
            "tail. The tail is reweighted to the whole-test proportions of "
            "true classes present in the tail; the whole-test score uses the "
            "same class set. Entries are the "
            "mean and 95\\% bootstrap interval across 30 runs. Positive "
            "values indicate lower tail macro-F1; grey cells include zero; "
            "$\\dagger$ marks cells whose unadjusted RQ1 macro-F1 interval "
            "includes zero."
        )
        label = "tab:rq2-adjusted-macro-f1"
        if not is_original:
            caption = (
                "Class-adjusted macro-F1 tail gaps for all models on all "
                "datasets. Entries are the mean and 95\\% bootstrap interval "
                "across 30 runs. Grey cells include zero, and $\\dagger$ "
                "marks cells whose unadjusted macro-F1 tail gap includes zero."
            )
    else:
        raise ValueError(f"Unknown table metric: {metric}")

    if is_original:
        column_spec = r"{@{\extracolsep{\fill}}lccccc@{}}"
        group_lines = [
            r"    & \multicolumn{2}{c}{Low-homophily tail}",
            r"    & \multicolumn{3}{c}{High-homophily tail} \\",
            r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
            (
                r"    Model & PubMed & ogbn-arxiv & Roman-empire "
                r"& Amazon-ratings & arxiv-year \\"
            ),
        ]
    else:
        label = label.replace("tab:", "tab:additional-", 1)
        column_spec = r"{@{\extracolsep{\fill}}lcc@{}}"
        group_lines = [
            r"    & \multicolumn{2}{c}{High-homophily tail} \\",
            r"    \cmidrule(lr){2-3}",
            r"    Model & Actor & BlogCatalog \\",
        ]

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \scriptsize",
        r"  \begin{tabular*}{\linewidth}" + column_spec,
        r"    \toprule",
        *group_lines,
        r"    \midrule",
    ]
    table_model_order = MODEL_ORDER if is_original else PAPER_MODEL_ORDER
    for model in table_model_order:
        cells = [
            latex_cell(by_cell[(dataset, model)], metric)
            for dataset in DATASET_ORDER
        ]
        lines.append(
            f"    {MODEL_DISPLAY[model]:<16} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"\end{table}",
        ]
    )
    return lines


def write_appendix_tex(
    path: Path,
    adjusted_cell_rows: list[dict],
    within_class_cell_rows: list[dict],
) -> None:
    within_cells = {
        (row["dataset"], row["model"]): row
        for row in within_class_cell_rows
        if row["preparation"] == "erm"
    }
    expected_cells = {
        (dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER
    }
    if set(within_cells) != expected_cells:
        raise RuntimeError("Incomplete within-class accuracy table")
    erm_cells = {
        (row["dataset"], row["model"]): row
        for row in adjusted_cell_rows
        if row["preparation"] == "erm"
    }
    significant_accuracy_cells = {
        key for key, row in erm_cells.items()
        if row["erm_accuracy_gap_ci_excludes_zero"]
    }
    significant_f1_cells = {
        key for key, row in erm_cells.items()
        if row["erm_tail_gap_ci_excludes_zero"]
    }
    expected_f1_nonsignificant = {
        ("roman-empire", "mlp"),
        ("arxiv-year", "graphsage-mean"),
        ("arxiv-year", "h2gcn"),
    }
    is_original = tuple(DATASET_ORDER) == tuple(ORIGINAL_DATASET_ORDER)
    if is_original:
        if significant_accuracy_cells != set(erm_cells):
            raise RuntimeError("Unexpected nonsignificant RQ1 accuracy cell")
        if set(erm_cells) - significant_f1_cells != expected_f1_nonsignificant:
            raise RuntimeError(
                "RQ1 macro-F1 significance markers no longer match the "
                "appendix explanation"
            )
    prefix = "" if is_original else "additional-"
    total_cells = len(expected_cells)
    rq2_cells = sum(
        bool(row["positive_nonzero_erm_tail_gap"])
        for row in within_cells.values()
    )
    lines = [
        r"\section{RQ2: Within-Class and Class-Composition Analyses}",
        f"\\label{{app:{prefix}rq2-within-class}}",
        "",
        r"\paragraph{Within-class accuracy.}",
        (
            f"Table~\\ref{{tab:{prefix}rq2-class-macro-accuracy}} directly "
            "evaluates "
            "the comparison stated in RQ2. For each run, we include each true "
            "class represented among both tail and non-tail test nodes. Within "
            "each included class, we calculate accuracy separately for "
            "tail and non-tail test nodes and subtract tail accuracy from "
            "non-tail accuracy. We then average these class contrasts equally. "
            "The resulting run-level value is the mean class contrast."
        ),
        (
            "Each cell is the mean of the 30 run-level values, with a 95\\% "
            "percentile-bootstrap interval obtained by resampling complete "
            "runs. Positive values indicate lower tail accuracy. Grey shading "
            f"marks an interval containing zero. The table reports all "
            f"{total_cells} model--dataset cells. The RQ2 comparison concerns "
            f"the {rq2_cells} cells whose RQ1 macro-F1 gap in "
            f"Table~\\ref{{tab:{prefix}erm-tail-gap}} is "
            "positive and has an interval wholly above zero."
        ),
        "",
    ]
    lines.extend(latex_table(within_class_cell_rows, "within_accuracy"))
    lines.extend(
        [
            "",
            r"\paragraph{Class-composition-adjusted RQ1 gaps.}",
            f"Tables~\\ref{{tab:{prefix}rq2-adjusted-accuracy}} and "
            f"\\ref{{tab:{prefix}rq2-adjusted-macro-f1}} give "
            "class-composition-adjusted versions of the two RQ1 gap "
            f"calculations. All {total_cells} ERM model--dataset cells use "
            "the same procedure.",
            (
            "For run $r$, let $W_r$ be the whole test set, $T_r$ the tail, "
            "and $C_r$ the true classes present in the tail. For $c\\in C_r$, "
            "we assign the reference weight "
            "$w_{c,r}=n^{W_r}_{c,r}/\\sum_{k\\in C_r}n^{W_r}_{k,r}$: the "
            "class's proportion in the whole test set, renormalised over "
            "$C_r$."
            ),
            (
            "For each class in $C_r$, we divide its tail confusion-matrix row "
            "by the number of tail nodes in that class, giving the conditional "
            "distribution of predicted labels. We multiply this row by "
            "$w_{c,r}$, thereby replacing the tail's class proportions with "
            "those of the whole test set. The whole-test score is calculated "
            "over the same classes; its class proportions already equal "
            "$w_{c,r}$. "
            "True classes outside $C_r$ are outside the adjusted population. "
            "Predictions outside $C_r$ remain errors for included classes. "
            "Macro-F1 is the equal-weighted mean of the one-vs-rest F1 values "
            "for the classes in $C_r$."
            ),
            (
            "We calculate accuracy and macro-F1 for the whole test and the "
            "class-reweighted tail. The run-level value is the whole-test "
            "score minus the reweighted tail score. Each table cell "
            "is the mean of the 30 run-level differences. Its 95\\% confidence "
            "interval is obtained by resampling the 30 complete runs as in "
            "Appendix~\\ref{app:significance}."
            ),
            (
            "Positive values mean that the tail scores lower after adjusting "
            "for true-class composition; negative values mean that it scores "
            "higher. Grey shading marks an adjusted interval containing zero. "
            "A $\\dagger$ marks a cell whose corresponding unadjusted RQ1 "
            f"interval in Table~\\ref{{tab:{prefix}erm-tail-gap-accuracy}} or "
            f"Table~\\ref{{tab:{prefix}erm-tail-gap}} contains zero."
            ),
            (
            "The adjusted estimates concern classes present in the tail. "
            "When $C_r$ omits a class included by RQ1, the difference between "
            "an adjusted cell and its RQ1 counterpart contains two changes: "
            "restriction to $C_r$ and reweighting of the tail to the "
            "whole-test class proportions."
            ),
            "",
        ]
    )
    lines.extend(latex_table(adjusted_cell_rows, "accuracy"))
    lines.append("")
    lines.extend(latex_table(adjusted_cell_rows, "f1"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty RQ2 summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def recorded_path(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute one."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def relative_paths(paths: Iterable[Path]) -> list[str]:
    return [recorded_path(path) for path in paths]


def write_readme(
    path: Path,
    dataset_mode: AnalysisDatasetMode = AnalysisDatasetMode.ORIGINAL,
) -> None:
    content = """# RQ2 within-class and class-composition analyses

This directory contains the direct within-class accuracy comparison stated in
RQ2 and class-composition-adjusted versions of the two RQ1
whole-test-minus-tail gaps. The direct comparison includes classes represented
among both tail and non-tail test nodes. The class-composition adjustment
includes every class present in the tail.

Outputs:

- `within_class_accuracy_run_contrasts.csv`: one row per ERM split-seed run,
  containing the equal-class mean of non-tail accuracy minus tail accuracy;
- `within_class_accuracy_cell_summary.csv`: the 30-run means and
  paired-bootstrap intervals used in the direct RQ2 table;
- `class_adjusted_run_metrics.csv`: one row per ERM split-seed run, containing
  the whole-test and class-composition-adjusted tail scores and their gaps;
- `class_adjusted_cell_summary.csv`: the 30-run means and paired-bootstrap
  intervals for every ERM dataset-model cell;
- `appendix_tables.tex`: the direct RQ2 table followed by the two adjusted RQ1
  tables, generated from the cell summaries.

The direct RQ2 contrast is the unweighted mean across included true classes of
non-tail accuracy minus tail accuracy. A positive value means that tail
nodes have lower accuracy within class. The table contains all 35 cells; the
RQ2 condition selects cells whose unadjusted RQ1 macro-F1 gap is positive and
has an interval wholly above zero.

For the class-composition adjustment, the tail confusion matrix is reweighted
to the whole-test class proportions over the tail-present classes. The
whole-test score is calculated over the same class set. These gaps are the
whole-test score minus the class-reweighted tail score. Predictions outside
the retained class set remain errors for retained true classes.

The adjusted tables retain all cells. A dagger is based on the corresponding
unadjusted RQ1 table: `results/main/rq1/accuracy/whole_vs_tail_accuracy.csv`
for adjusted accuracy and `results/main/aggregates/erm_tail_gaps.csv` for
adjusted macro-F1.

Regenerate the table from the canonical raw archive:

```powershell
python scripts/analyze_rq2_within_class.py `
  --raw-root PATH-TO-CANONICAL-RUNS
```

The raw archive is never modified. `analysis_manifest.json` records the
calculation, bootstrap definition, panel checks, combined input hash,
software versions, and output hashes.
"""
    if dataset_mode is AnalysisDatasetMode.ADDITIONAL:
        content = (
            content.replace("all 35 cells", "all 14 cells")
            .replace("results/main/", "results/additional/")
            .replace(
                "--raw-root PATH-TO-CANONICAL-RUNS",
                "--dataset-mode additional `\n  --additional-raw-root PATH-TO-ADDITIONAL-RUNS",
            )
        )
    path.write_text(content, encoding="utf-8")


def write_manifest(
    path: Path,
    raw_root: Path,
    gap_table: Path,
    accuracy_gap_table: Path,
    audit: dict,
    run_contrasts: list[dict],
    cell_summaries: list[dict],
    within_class_run_contrasts: list[dict],
    within_class_cell_summaries: list[dict],
    outputs: list[Path],
    dataset_mode: AnalysisDatasetMode = AnalysisDatasetMode.ORIGINAL,
) -> None:
    payload = {
        "schema_version": 4,
        "analysis_version": ANALYSIS_VERSION,
        "analysis": (
            "RQ2 within-class accuracy and class-composition-adjusted "
            "whole-test versus tail gaps"
        ),
        "definitions": {
            "unit": "one split-seed prediction set",
            "within_class_eligible_class": (
                "true class with at least one tail and one non-tail test node "
                "in that run"
            ),
            "class_adjusted_tail_class": (
                "true class with at least one tail test node in that run"
            ),
            "within_class_accuracy_contrast": (
                "unweighted mean over eligible true classes of non-tail "
                "accuracy minus tail accuracy"
            ),
            "within_class_positive_contrast": (
                "tail accuracy is lower within the average eligible class"
            ),
            "reference_class_distribution": (
                "whole-test true-class proportions renormalised over "
                "tail-present classes"
            ),
            "tail_conditional_confusion_rows": (
                "tail prediction-label distribution within each tail-present "
                "true class"
            ),
            "class_reweighted_tail_confusion": (
                "tail conditional confusion rows multiplied by the whole-test "
                "reference class proportions"
            ),
            "whole_test_score": (
                "ordinary score over whole-test nodes whose true class is "
                "present in the tail"
            ),
            "excluded_true_classes": (
                "classes absent from the tail are outside both evaluated "
                "populations"
            ),
            "predictions_outside_tail_classes": (
                "retained as false negatives for tail-present true classes"
            ),
            "class_adjusted_contrast": (
                "whole-test score minus class-reweighted tail score"
            ),
            "class_adjusted_positive_contrast": (
                "tail performance is lower after class-composition adjustment"
            ),
            "cell_aggregation": (
                "unweighted arithmetic mean of the 30 run-level contrasts"
            ),
            "sample_standard_deviation": "ddof=1",
            "bootstrap_intervals": {
                "common": {
                    "method": "two-sided percentile bootstrap of complete runs",
                    "level": 0.95,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "global_seed": BOOTSTRAP_SEED,
                    "quantile_method": "linear",
                },
                "within_class_accuracy_seed_derivation": (
                    "SHA256(global seed, dataset, model, preparation, "
                    "class-macro-accuracy), matching the original RQ2 table"
                ),
                "class_adjusted_seed_derivation": (
                    "SHA256(global seed, dataset, model, erm-tail-gap), "
                    "matching RQ1"
                ),
            },
            "rq2_condition": (
                "ERM macro-F1 tail-gap mean is positive and its 95% "
                "paired-bootstrap interval lies wholly above zero"
            ),
        },
        "scope": {
            "dataset_mode": dataset_mode.value,
            "datasets": DATASET_ORDER,
            "models": MODEL_ORDER,
            "validated_preparations": PREPARATION_ORDER,
            "adjusted_preparations": ["erm"],
            "expected_runs_per_cell": EXPECTED_RUNS_PER_CELL,
            "adjusted_run_rows": len(run_contrasts),
            "adjusted_cell_rows": len(cell_summaries),
            "within_class_run_rows": len(within_class_run_contrasts),
            "within_class_cell_rows": len(within_class_cell_summaries),
            "registered_run_panel_exact_match": True,
            **audit,
        },
        "inputs": {
            "raw_store": (
                "config.json and run_data.npz from the external run store, "
                "validated against "
                + (
                    "manifests/main-runs.jsonl"
                    if dataset_mode is AnalysisDatasetMode.ORIGINAL
                    else "manifests/additional-dataset-runs.jsonl"
                )
                + " and the dataset manifests"
            ),
            "erm_macro_f1_gap_table": {
                "path": recorded_path(gap_table),
                "sha256": file_sha256(gap_table),
            },
            "erm_accuracy_gap_table": {
                "path": recorded_path(accuracy_gap_table),
                "sha256": file_sha256(accuracy_gap_table),
            },
        },
        "analysis_script": {
            "path": "scripts/analyze_rq2_within_class.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "outputs": {
            relative: file_sha256(output)
            for relative, output in zip(relative_paths(outputs), outputs)
            if output != path
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=(
            AnalysisDatasetMode.ORIGINAL.value,
            AnalysisDatasetMode.ADDITIONAL.value,
        ),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="canonical directory containing one subdirectory per raw run",
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="raw run store for Actor and BlogCatalog",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument(
        "--gap-table",
        type=Path,
    )
    parser.add_argument(
        "--accuracy-gap-table",
        type=Path,
    )
    args = parser.parse_args(argv)
    mode = AnalysisDatasetMode(args.dataset_mode)
    if mode is AnalysisDatasetMode.ORIGINAL:
        if args.raw_root is None:
            parser.error("--raw-root is required for 'original' mode")
        args.input_root = args.raw_root
    else:
        if args.additional_raw_root is None:
            parser.error("--additional-raw-root is required for 'additional' mode")
        args.input_root = args.additional_raw_root
    args.dataset_mode = mode
    args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[mode]
    args.gap_table = args.gap_table or DEFAULT_GAP_TABLES[mode]
    args.accuracy_gap_table = (
        args.accuracy_gap_table or DEFAULT_ACCURACY_GAP_TABLES[mode]
    )
    return args


def main() -> None:
    global DATASET_ORDER

    args = parse_args()
    DATASET_ORDER = list(
        ORIGINAL_DATASET_ORDER
        if args.dataset_mode is AnalysisDatasetMode.ORIGINAL
        else ADDITIONAL_DATASET_ORDER
    )
    raw_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    gap_table = args.gap_table.expanduser().resolve()
    accuracy_gap_table = args.accuracy_gap_table.expanduser().resolve()
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw run directory does not exist: {raw_root}")
    if not gap_table.is_file():
        raise FileNotFoundError(f"ERM tail-gap table does not exist: {gap_table}")
    if not accuracy_gap_table.is_file():
        raise FileNotFoundError(
            f"ERM accuracy-gap table does not exist: {accuracy_gap_table}"
        )

    manifests = {
        dataset: read_json(DATASET_REGISTRY[dataset].manifest_path(REPO_ROOT))
        for dataset in DATASET_ORDER
    }
    gap_status = read_gap_status(gap_table)
    accuracy_gap_status = read_accuracy_gap_status(accuracy_gap_table)
    registrations = load_registered_runs(
        REPO_ROOT, args.dataset_mode, DATASET_ORDER
    )
    run_rows, audit = discover_and_validate_runs(
        raw_root, manifests, registrations
    )
    within_class_run_contrasts = build_within_class_accuracy_run_contrasts(
        run_rows, manifests, gap_status
    )
    within_class_cell_summaries = (
        build_within_class_accuracy_cell_summaries(within_class_run_contrasts)
    )
    run_contrasts = build_run_level_contrasts(
        run_rows, manifests, gap_status, accuracy_gap_status
    )
    cell_summaries = build_cell_summaries(run_contrasts)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_contrast_path = output_dir / "class_adjusted_run_metrics.csv"
    cell_summary_path = output_dir / "class_adjusted_cell_summary.csv"
    within_class_run_path = (
        output_dir / "within_class_accuracy_run_contrasts.csv"
    )
    within_class_cell_path = (
        output_dir / "within_class_accuracy_cell_summary.csv"
    )
    appendix_tex_path = output_dir / "appendix_tables.tex"
    readme_path = output_dir / "README.md"
    manifest_path = output_dir / "analysis_manifest.json"
    write_csv(within_class_run_path, within_class_run_contrasts)
    write_csv(within_class_cell_path, within_class_cell_summaries)
    write_csv(run_contrast_path, run_contrasts)
    write_csv(cell_summary_path, cell_summaries)
    write_appendix_tex(
        appendix_tex_path, cell_summaries, within_class_cell_summaries
    )
    write_readme(readme_path, args.dataset_mode)
    outputs = [
        within_class_run_path,
        within_class_cell_path,
        run_contrast_path,
        cell_summary_path,
        appendix_tex_path,
        readme_path,
        manifest_path,
    ]
    write_manifest(
        manifest_path,
        raw_root,
        gap_table,
        accuracy_gap_table,
        audit,
        run_contrasts,
        cell_summaries,
        within_class_run_contrasts,
        within_class_cell_summaries,
        outputs,
        args.dataset_mode,
    )

    print(
        f"Validated {audit['run_count']} runs and "
        f"{audit['run_level_class_rows']} tail-present run-class rows."
    )
    print(f"Wrote {len(run_contrasts)} adjusted ERM run metrics.")
    print(f"Wrote {len(cell_summaries)} adjusted ERM cell summaries.")
    print(
        f"Wrote {len(within_class_run_contrasts)} within-class ERM run contrasts."
    )
    print(
        f"Wrote {len(within_class_cell_summaries)} within-class ERM cell summaries."
    )
    for output in outputs:
        print(output.resolve().relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
