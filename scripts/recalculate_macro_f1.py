"""Recalculate macro-F1 from saved predictions using true-label support.

The raw run archives are read-only. Corrected per-run JSON records and compact
cell-level CSV summaries are written below the mode-specific result root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hfd.analysis_datasets import (
    ADDITIONAL_DATASET_ORDER,
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
    load_registered_runs,
    validate_registered_run_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
# Retain this public constant for callers that import the legacy five-dataset
# order. Mode-aware code obtains its order from ``dataset_order`` below.
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
MODEL_GROUP = {
    "mlp": "graphless",
    "gcn": "standard-message-passing",
    "graphsage-mean": "standard-message-passing",
    "h2gcn": "heterophily-designed-aggregation",
    "gpr-gnn": "heterophily-designed-aggregation",
    "linkx": "adjacency-feature",
    "acm-gcn": "heterophily-designed-aggregation",
}
PREPARATION_ORDER = ["erm", "last", "head", "full"]
EXPECTED_RUNS_PER_CELL = 30
ANALYSIS_VERSION = "true-label-support-v1"
METRIC_DEFINITION_COMMIT = "91b2518af84cf7150d6f498360f5ab00ac5dc079"
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_716
CHANGE_TOLERANCE = 1e-12
DEFAULT_OUTPUT_ROOTS = {
    AnalysisDatasetMode.ORIGINAL: REPO_ROOT / "results" / "main",
    AnalysisDatasetMode.ADDITIONAL: REPO_ROOT / "results" / "additional",
    AnalysisDatasetMode.EXTENDED: REPO_ROOT / "results" / "extended",
}
SCRIPT_SHA256 = hashlib.sha256(
    Path(__file__)
    .read_text(encoding="utf-8")
    .replace("\r\n", "\n")
    .encode("utf-8")
).hexdigest()


def selected_datasets(
    mode: AnalysisDatasetMode | str,
    excluded_datasets: list[str] | tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the full mode order and the non-excluded analysis datasets."""

    mode_order = dataset_order(mode)
    invalid = sorted(set(excluded_datasets) - set(mode_order))
    if invalid:
        raise ValueError(
            f"Cannot exclude datasets outside {AnalysisDatasetMode(mode).value!r} "
            f"mode: {invalid}"
        )
    selected = tuple(
        dataset for dataset in mode_order if dataset not in excluded_datasets
    )
    if not selected:
        raise ValueError("At least one dataset must remain after exclusions")
    return mode_order, selected


def raw_root_specs(
    mode: AnalysisDatasetMode | str,
    raw_root: Path | None,
    additional_raw_root: Path | None,
) -> tuple[tuple[str, Path, frozenset[str]], ...]:
    """Resolve the run stores required by one dataset mode."""

    mode = AnalysisDatasetMode(mode)
    specs: list[tuple[str, Path, frozenset[str]]] = []
    if mode in {AnalysisDatasetMode.ORIGINAL, AnalysisDatasetMode.EXTENDED}:
        if raw_root is None:
            raise ValueError(f"--raw-root is required for {mode.value!r} mode")
        specs.append(("original", raw_root, frozenset(ORIGINAL_DATASET_ORDER)))
    if mode in {AnalysisDatasetMode.ADDITIONAL, AnalysisDatasetMode.EXTENDED}:
        if additional_raw_root is None:
            raise ValueError(
                f"--additional-raw-root is required for {mode.value!r} mode"
            )
        specs.append(
            (
                "additional",
                additional_raw_root,
                frozenset(ADDITIONAL_DATASET_ORDER),
            )
        )
    return tuple(specs)


def collect_selected_runs(
    root_specs: tuple[tuple[str, Path, frozenset[str]], ...],
    included_datasets: tuple[str, ...],
) -> list[tuple[Path, dict]]:
    """Load selected run configs while enforcing store ownership and IDs."""

    included = set(included_datasets)
    selected_runs: list[tuple[Path, dict]] = []
    seen_run_ids: set[str] = set()
    for store_name, root, allowed_datasets in root_specs:
        if not root.is_dir():
            raise FileNotFoundError(f"Raw run store not found: {root}")
        run_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name != "historical-records"
        )
        unexpected_directories = [
            path.name for path in run_dirs if not (path / "config.json").is_file()
        ]
        if unexpected_directories:
            raise RuntimeError(
                f"Unexpected non-run directories in {store_name} raw root: "
                f"{unexpected_directories}"
            )
        for run_dir in run_dirs:
            config = read_json(run_dir / "config.json")
            dataset = str(config.get("dataset"))
            if dataset not in allowed_datasets:
                raise RuntimeError(
                    f"{run_dir.name}: dataset {dataset!r} is in the wrong raw "
                    f"store ({store_name})"
                )
            run_id = str(config.get("run_id"))
            if run_id != run_dir.name:
                raise RuntimeError(f"{run_dir.name}: directory and run_id differ")
            if run_id in seen_run_ids:
                raise RuntimeError(f"Duplicate run_id across raw stores: {run_id}")
            seen_run_ids.add(run_id)
            if dataset not in included:
                continue
            if config.get("model") not in MODEL_ORDER:
                raise RuntimeError(f"Unexpected model: {config.get('model')}")
            if config.get("preparation") not in PREPARATION_ORDER:
                raise RuntimeError(
                    f"Unexpected preparation: {config.get('preparation')}"
                )
            selected_runs.append((run_dir, config))
    return selected_runs


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> float:
    """Match sklearn macro-F1 with explicit labels and zero_division=0."""
    class_scores = []
    for label in labels:
        true_label = y_true == label
        predicted_label = y_pred == label
        true_positive = int(np.count_nonzero(true_label & predicted_label))
        false_positive = int(np.count_nonzero(~true_label & predicted_label))
        false_negative = int(np.count_nonzero(true_label & ~predicted_label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(class_scores))


def bootstrap_seed(dataset: str, model: str, preparation: str) -> int:
    key = f"{BOOTSTRAP_SEED}:{dataset}:{model}:{preparation}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def paired_bootstrap(
    gains: np.ndarray, dataset: str, model: str, preparation: str
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(bootstrap_seed(dataset, model, preparation))
    sample_indices = rng.integers(
        0, gains.size, size=(BOOTSTRAP_RESAMPLES, gains.size)
    )
    bootstrap_means = gains[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    smaller_tail_count = min(
        int(np.count_nonzero(bootstrap_means <= 0.0)),
        int(np.count_nonzero(bootstrap_means >= 0.0)),
    )
    p_value = min(
        1.0,
        2.0 * smaller_tail_count / BOOTSTRAP_RESAMPLES,
    )
    return float(lower), float(upper), float(p_value), smaller_tail_count


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def check_array(
    array: np.ndarray, expected_checksum: str, run_id: str, name: str
) -> str:
    actual = array_sha256(array)
    if actual != expected_checksum:
        raise RuntimeError(f"{run_id}: checksum mismatch for {name}")
    return actual


def calculate_run(
    run_dir: Path, config: dict, manifest: dict, output_root: Path
) -> tuple[dict, dict]:
    run_id = config["run_id"]
    split = config["split"]
    expected = manifest["array_checksums_sha256"]
    recorded = config["fixed_input_checksums"]
    required_checksums = {
        "labels": "labels",
        "test_idx": f"{split}.test",
        "merged_bin": "merged_bin",
        "tail_mask": "tail_mask",
    }
    for manifest_key in required_checksums.values():
        if recorded.get(manifest_key) != expected[manifest_key]:
            raise RuntimeError(
                f"{run_id}: config checksum differs from the registered manifest "
                f"for {manifest_key}"
            )

    with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        predictions = saved["predictions"]
        test_idx = saved["test_idx"]
        merged_bin = saved["merged_bin"]
        tail_mask = saved["tail_mask"]

    checksums = {
        "labels": check_array(labels, expected["labels"], run_id, "labels"),
        "test_idx": check_array(
            test_idx, expected[f"{split}.test"], run_id, "test_idx"
        ),
        "merged_bin": check_array(
            merged_bin, expected["merged_bin"], run_id, "merged_bin"
        ),
        "tail_mask": check_array(
            tail_mask, expected["tail_mask"], run_id, "tail_mask"
        ),
        "predictions": array_sha256(predictions),
    }

    dataset_labels = np.unique(labels[labels >= 0])
    expected_n_classes = int(manifest["graph"]["num_classes"])
    expected_labels = np.arange(expected_n_classes, dtype=dataset_labels.dtype)
    if not np.array_equal(dataset_labels, expected_labels):
        raise RuntimeError(
            f"{run_id}: dataset labels are {dataset_labels.tolist()}; expected "
            f"{expected_labels.tolist()}"
        )

    test_labels = labels[test_idx]
    test_predictions = predictions[test_idx]
    whole_true_labels = np.unique(test_labels)
    whole_f1 = macro_f1(test_labels, test_predictions, whole_true_labels)

    tail_test_idx = test_idx[tail_mask[test_idx]]
    if tail_test_idx.size == 0:
        raise RuntimeError(f"{run_id}: empty test tail")
    tail_true_labels = np.unique(labels[tail_test_idx])
    tail_f1 = macro_f1(
        labels[tail_test_idx], predictions[tail_test_idx], tail_true_labels
    )

    bins = []
    manifest_bins = manifest["local_homophily"]["merged_bins"]
    for bin_index, bin_definition in enumerate(manifest_bins):
        bin_test_idx = test_idx[merged_bin[test_idx] == bin_index]
        bin_true_labels = np.unique(labels[bin_test_idx])
        bin_f1 = (
            macro_f1(
                labels[bin_test_idx], predictions[bin_test_idx], bin_true_labels
            )
            if bin_test_idx.size
            else None
        )
        bins.append(
            {
                "bin_index": bin_index,
                "interval": bin_definition["interval"],
                "is_tail": bool(bin_definition["is_tail"]),
                "n_test": int(bin_test_idx.size),
                "true_labels": [int(label) for label in bin_true_labels],
                "macro_f1": bin_f1,
            }
        )
    scored_bins = [row for row in bins if row["macro_f1"] is not None]
    worst_bin = min(scored_bins, key=lambda row: row["macro_f1"])

    old_metrics = read_json(run_dir / "metrics.json")
    old_bins = old_metrics["per_bin"]
    if len(old_bins) != len(bins):
        raise RuntimeError(f"{run_id}: stored and registered bin counts differ")
    old_comparison = {
        "whole_abs_change": abs(
            whole_f1 - float(old_metrics["whole_test_macro_f1"])
        ),
        "tail_abs_change": abs(tail_f1 - float(old_metrics["tail_macro_f1"])),
        "bin_abs_changes": [
            abs(float(new["macro_f1"]) - float(old["macro_f1"]))
            for new, old in zip(bins, old_bins)
            if new["macro_f1"] is not None
        ],
    }

    record = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_script_sha256": SCRIPT_SHA256,
        "metric_definition_commit": METRIC_DEFINITION_COMMIT,
        "run_id": run_id,
        "dataset": config["dataset"],
        "model": config["model"],
        "preparation": config["preparation"],
        "split": split,
        "seed": int(config["seed"]),
        "training_code_version": config["code_version"],
        "metric_definition": {
            "name": "macro-F1",
            "labels": "classes present in the evaluated subset's true labels",
            "zero_division": 0,
            "dataset_labels": [int(label) for label in dataset_labels],
        },
        "input_checksums_sha256": checksums,
        "metrics": {
            "whole_test": {
                "n_test": int(test_idx.size),
                "true_labels": [int(label) for label in whole_true_labels],
                "macro_f1": whole_f1,
            },
            "tail": {
                "n_test": int(tail_test_idx.size),
                "true_labels": [int(label) for label in tail_true_labels],
                "macro_f1": tail_f1,
            },
            "tail_gap": whole_f1 - tail_f1,
            "per_bin": bins,
            "worst_bin": {
                "bin_index": worst_bin["bin_index"],
                "interval": worst_bin["interval"],
                "macro_f1": worst_bin["macro_f1"],
            },
        },
    }
    output_path = (
        output_root
        / "runs"
        / config["dataset"]
        / config["model"]
        / f"{run_id}.json"
    )
    write_json(output_path, record)

    summary_row = {
        "run_id": run_id,
        "dataset": config["dataset"],
        "model": config["model"],
        "preparation": config["preparation"],
        "split": split,
        "seed": int(config["seed"]),
        "whole_test_macro_f1": whole_f1,
        "tail_macro_f1": tail_f1,
        "tail_gap": whole_f1 - tail_f1,
    }
    return summary_row, old_comparison


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_aggregates(
    run_rows: list[dict],
    manifests: dict[str, dict],
    output_root: Path,
    ordered_datasets: tuple[str, ...] | list[str] = tuple(DATASET_ORDER),
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_run: dict[tuple[str, str, str, int], dict[str, float]] = defaultdict(dict)
    for row in run_rows:
        by_cell[(row["dataset"], row["model"], row["preparation"])].append(row)
        by_run[(row["dataset"], row["model"], row["split"], row["seed"])][
            row["preparation"]
        ] = row["tail_macro_f1"]

    prepared_rows = []
    gain_rows = []
    gap_rows = []
    datasets = [dataset for dataset in ordered_datasets if dataset in manifests]
    for dataset in datasets:
        dataset_index = ordered_datasets.index(dataset)
        manifest = manifests[dataset]
        tail_side = manifest["local_homophily"]["tail_side"]
        expected_pairs = {
            (split["split_id"], int(seed))
            for split in manifest["splits"]
            for seed in manifest["training_seeds"]
        }
        if len(expected_pairs) != EXPECTED_RUNS_PER_CELL:
            raise RuntimeError(
                f"{dataset}: registered split/seed panel has "
                f"{len(expected_pairs)} pairs; expected {EXPECTED_RUNS_PER_CELL}"
            )
        expected_run_count = len(expected_pairs)
        for model_index, model in enumerate(MODEL_ORDER):
            for preparation_index, preparation in enumerate(PREPARATION_ORDER):
                rows = by_cell[(dataset, model, preparation)]
                if len(rows) != expected_run_count:
                    raise RuntimeError(
                        f"{dataset}/{model}/{preparation}: found {len(rows)} runs; "
                        f"expected {expected_run_count}"
                    )
                observed_pairs = {(row["split"], row["seed"]) for row in rows}
                if observed_pairs != expected_pairs:
                    missing = sorted(expected_pairs - observed_pairs)
                    unexpected = sorted(observed_pairs - expected_pairs)
                    raise RuntimeError(
                        f"{dataset}/{model}/{preparation}: run panel mismatch; "
                        f"missing={missing}, unexpected={unexpected}"
                    )
                scores = np.array(
                    [row["tail_macro_f1"] for row in rows], dtype=np.float64
                )
                prepared_rows.append(
                    {
                        "dataset_order": dataset_index,
                        "dataset": dataset,
                        "tail_side": tail_side,
                        "model_order": model_index,
                        "model": model,
                        "model_group": MODEL_GROUP[model],
                        "preparation_order": preparation_index,
                        "preparation": preparation,
                        "n_runs": scores.size,
                        "mean_tail_macro_f1": float(scores.mean()),
                        "sample_std_tail_macro_f1": float(scores.std(ddof=1)),
                    }
                )

            erm_rows = sorted(
                by_cell[(dataset, model, "erm")],
                key=lambda row: (row["split"], row["seed"]),
            )
            gaps = np.array(
                [row["tail_gap"] for row in erm_rows], dtype=np.float64
            )
            lower, upper, p_value, smaller_tail_count = paired_bootstrap(
                gaps, dataset, model, "erm-tail-gap"
            )
            gap_rows.append(
                {
                    "dataset_order": dataset_index,
                    "dataset": dataset,
                    "tail_side": tail_side,
                    "model_order": model_index,
                    "model": model,
                    "model_group": MODEL_GROUP[model],
                    "n_pairs": gaps.size,
                    "mean_tail_gap": float(gaps.mean()),
                    "sample_std_tail_gap": float(gaps.std(ddof=1)),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_smaller_tail_count": smaller_tail_count,
                    "bootstrap_p_value": p_value,
                    "ci_excludes_zero": lower > 0.0 or upper < 0.0,
                }
            )

            for preparation_index, preparation in enumerate(
                PREPARATION_ORDER[1:], start=1
            ):
                paired = []
                for key, values in sorted(by_run.items()):
                    run_dataset, run_model, _, _ = key
                    if run_dataset != dataset or run_model != model:
                        continue
                    if set(values) != set(PREPARATION_ORDER):
                        raise RuntimeError(f"{key}: incomplete preparation set")
                    paired.append(values[preparation] - values["erm"])
                if len(paired) != expected_run_count:
                    raise RuntimeError(
                        f"{dataset}/{model}/{preparation}: found {len(paired)} "
                        f"paired gains; expected {expected_run_count}"
                    )
                gains = np.array(paired, dtype=np.float64)
                lower, upper, p_value, smaller_tail_count = paired_bootstrap(
                    gains, dataset, model, preparation
                )
                gain_rows.append(
                    {
                        "dataset_order": dataset_index,
                        "dataset": dataset,
                        "tail_side": tail_side,
                        "model_order": model_index,
                        "model": model,
                        "model_group": MODEL_GROUP[model],
                        "preparation_order": preparation_index,
                        "preparation": preparation,
                        "n_pairs": gains.size,
                        "mean_paired_gain": float(gains.mean()),
                        "sample_std_paired_gain": float(gains.std(ddof=1)),
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                        "bootstrap_smaller_tail_count": smaller_tail_count,
                        "bootstrap_p_value": p_value,
                        "ci_excludes_zero": lower > 0.0 or upper < 0.0,
                    }
                )

    gain_by_cell = {
        (row["dataset"], row["model"], row["preparation"]): row
        for row in gain_rows
    }
    recovered_share_rows = []
    for dataset in datasets:
        dataset_index = ordered_datasets.index(dataset)
        tail_side = manifests[dataset]["local_homophily"]["tail_side"]
        for model_index, model in enumerate(MODEL_ORDER):
            full = gain_by_cell[(dataset, model, "full")]
            denominator = float(full["mean_paired_gain"])
            qualifies = denominator >= 0.03 and bool(full["ci_excludes_zero"])
            for preparation_index, preparation in enumerate(("last", "head"), start=1):
                numerator = float(
                    gain_by_cell[(dataset, model, preparation)]["mean_paired_gain"]
                )
                recovered_share_rows.append(
                    {
                        "dataset_order": dataset_index,
                        "dataset": dataset,
                        "tail_side": tail_side,
                        "model_order": model_index,
                        "model": model,
                        "model_group": MODEL_GROUP[model],
                        "preparation_order": preparation_index,
                        "preparation": preparation,
                        "mean_paired_gain": numerator,
                        "full_mean_paired_gain": denominator,
                        "full_bootstrap_ci_lower": full["bootstrap_ci_lower"],
                        "full_bootstrap_ci_upper": full["bootstrap_ci_upper"],
                        "qualifies": qualifies,
                        "recovered_share": numerator / denominator if qualifies else None,
                    }
                )

    write_csv(
        output_root / "aggregates" / "prepared_scores.csv",
        list(prepared_rows[0]),
        prepared_rows,
    )
    write_csv(
        output_root / "aggregates" / "paired_gains.csv",
        list(gain_rows[0]),
        gain_rows,
    )
    write_csv(
        output_root / "aggregates" / "erm_tail_gaps.csv",
        list(gap_rows[0]),
        gap_rows,
    )
    write_csv(
        output_root / "aggregates" / "recovered_shares.csv",
        list(recovered_share_rows[0]),
        recovered_share_rows,
    )
    return prepared_rows, gain_rows, gap_rows, recovered_share_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(mode.value for mode in AnalysisDatasetMode),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="raw run store for the original five datasets",
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="separate raw run store for Actor and BlogCatalog",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "corrected-result root; defaults to results/main, "
            "results/additional, or results/extended according to dataset mode"
        ),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        choices=tuple(DATASET_REGISTRY),
    )
    args = parser.parse_args()

    mode = AnalysisDatasetMode(args.dataset_mode)
    try:
        mode_order, included_datasets = selected_datasets(
            mode, tuple(args.exclude_dataset)
        )
        root_specs = raw_root_specs(
            mode, args.raw_root, args.additional_raw_root
        )
    except ValueError as error:
        parser.error(str(error))
    output_root = args.output_root or DEFAULT_OUTPUT_ROOTS[mode]
    manifests = {
        dataset: read_json(DATASET_REGISTRY[dataset].manifest_path(REPO_ROOT))
        for dataset in included_datasets
    }

    selected_runs = collect_selected_runs(root_specs, included_datasets)
    registrations = load_registered_runs(
        REPO_ROOT, mode, included_datasets
    )
    validate_registered_run_configs(
        (config for _, config in selected_runs),
        registrations,
        context=f"{mode.value} macro-F1 input",
    )

    unexpected_pair_counts = {
        dataset: len(manifest["splits"]) * len(manifest["training_seeds"])
        for dataset, manifest in manifests.items()
        if len(manifest["splits"]) * len(manifest["training_seeds"])
        != EXPECTED_RUNS_PER_CELL
    }
    if unexpected_pair_counts:
        raise RuntimeError(
            "Registered split/seed panels must contain exactly "
            f"{EXPECTED_RUNS_PER_CELL} pairs: {unexpected_pair_counts}"
        )
    expected_runs = (
        len(included_datasets)
        * len(MODEL_ORDER)
        * len(PREPARATION_ORDER)
        * EXPECTED_RUNS_PER_CELL
    )
    if len(selected_runs) != expected_runs:
        raise RuntimeError(f"Found {len(selected_runs)} runs; expected {expected_runs}")

    expected_output_paths = {
        output_root
        / "runs"
        / config["dataset"]
        / config["model"]
        / f"{config['run_id']}.json"
        for _, config in selected_runs
    }
    existing_output_paths = set((output_root / "runs").rglob("*.json"))
    stale_output_paths = sorted(existing_output_paths - expected_output_paths)
    if stale_output_paths:
        preview = ", ".join(str(path) for path in stale_output_paths[:5])
        raise RuntimeError(
            f"Found {len(stale_output_paths)} stale generated run files. "
            f"Remove them before regenerating. First files: {preview}"
        )

    run_rows = []
    comparisons = []
    for run_dir, config in selected_runs:
        row, comparison = calculate_run(
            run_dir, config, manifests[config["dataset"]], output_root
        )
        run_rows.append(row)
        comparisons.append(comparison)

    dataset_index = {value: index for index, value in enumerate(mode_order)}
    model_index = {value: index for index, value in enumerate(MODEL_ORDER)}
    preparation_index = {
        value: index for index, value in enumerate(PREPARATION_ORDER)
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
    prepared_rows, gain_rows, gap_rows, recovered_share_rows = build_aggregates(
        run_rows, manifests, output_root, mode_order
    )

    bin_changes = [
        change for comparison in comparisons for change in comparison["bin_abs_changes"]
    ]
    audit = {
        "comparison_tolerance": CHANGE_TOLERANCE,
        "runs_with_changed_whole_test_macro_f1": sum(
            comparison["whole_abs_change"] > CHANGE_TOLERANCE
            for comparison in comparisons
        ),
        "runs_with_changed_tail_macro_f1": sum(
            comparison["tail_abs_change"] > CHANGE_TOLERANCE
            for comparison in comparisons
        ),
        "per_bin_rows": len(bin_changes),
        "per_bin_rows_changed": sum(
            change > CHANGE_TOLERANCE for change in bin_changes
        ),
        "maximum_whole_test_abs_change": max(
            comparison["whole_abs_change"] for comparison in comparisons
        ),
        "maximum_tail_abs_change": max(
            comparison["tail_abs_change"] for comparison in comparisons
        ),
        "maximum_per_bin_abs_change": max(bin_changes),
    }
    analysis_manifest = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_code": {
            "script": "scripts/recalculate_macro_f1.py",
            "script_sha256": SCRIPT_SHA256,
            "script_hash_normalisation": "UTF-8 text with LF line endings",
            "numpy_version": np.__version__,
        },
        "metric_definition_commit": METRIC_DEFINITION_COMMIT,
        "metric": {
            "name": "macro-F1",
            "label_set": (
                "classes present in each evaluated subset's true labels, "
                "fixed before predictions are evaluated"
            ),
            "zero_division": 0,
        },
        "scope": {
            "dataset_mode": mode.value,
            "included_datasets": list(included_datasets),
            "excluded_datasets": list(args.exclude_dataset),
            "dataset_order": list(mode_order),
            "model_order": MODEL_ORDER,
            "preparation_order": PREPARATION_ORDER,
            "runs": len(run_rows),
            "prepared_score_cells": len(prepared_rows),
            "paired_gain_cells": len(gain_rows),
            "erm_tail_gap_cells": len(gap_rows),
            "recovered_share_rows": len(recovered_share_rows),
            "registered_run_panel_exact_match": True,
        },
        "aggregates": {
            "standard_deviation": "sample standard deviation (ddof=1)",
            "paired_bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "interval": "two-sided percentile 95%",
                "quantile_method": "linear",
                "p_value": (
                    "two-sided percentile-bootstrap tail area: min(1, "
                    "2 * min(count(mean <= 0), count(mean >= 0)) / resamples)"
                ),
                "zero_tail_reporting": "p < 2 / resamples",
                "decision_threshold": 0.05,
                "global_seed": BOOTSTRAP_SEED,
                "seed_derivation": "SHA256(global seed, dataset, model, preparation)",
                "multiple_comparison_correction": None,
            },
        },
        "comparison_with_stored_raw_metrics": audit,
        "training_code_versions": sorted(
            {str(config["code_version"]) for _, config in selected_runs}
        ),
    }
    write_json(output_root / "analysis_manifest.json", analysis_manifest)
    print(
        f"Wrote {len(run_rows)} corrected runs, {len(prepared_rows)} prepared-score "
        f"cells, {len(gain_rows)} paired-gain cells, {len(gap_rows)} ERM-gap "
        f"cells, and {len(recovered_share_rows)} recovered-share rows to "
        f"{output_root}"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
