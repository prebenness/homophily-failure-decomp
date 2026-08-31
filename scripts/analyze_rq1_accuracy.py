"""Compute whole-test and tail accuracy for the complete ERM run panel.

For every split-seed run, accuracy is calculated separately on the complete
test set and on the registered tail subset. Cell means give all 30 runs equal
weight. The raw run archive is read-only.

Example:
    python scripts/analyze_rq1_accuracy.py \
        --raw-root PATH-TO-CANONICAL-RUNS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
DATASET_ORDER = ORIGINAL_DATASET_ORDER
DATASET_LABELS = {
    "pubmed": "PubMed",
    "ogbn-arxiv": "ogbn-arxiv",
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "arxiv-year": "arxiv-year",
    "actor": "Actor",
    "blogcatalog": "BlogCatalog",
}
MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "linkx",
    "acm-gcn",
)
PAPER_MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR-GNN",
    "linkx": "LINKX",
    "acm-gcn": "ACM-GCN",
}
ANALYSIS_VERSION = "rq1-whole-tail-accuracy-v2"
FLOAT_TOLERANCE = 1e-12
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_716
DEFAULT_OUTPUT_ROOTS = {
    AnalysisDatasetMode.ORIGINAL: (
        REPO_ROOT / "results" / "main" / "rq1" / "accuracy"
    ),
    AnalysisDatasetMode.ADDITIONAL: (
        REPO_ROOT / "results" / "additional" / "rq1" / "accuracy"
    ),
}
SCRIPT_SHA256 = hashlib.sha256(
    Path(__file__)
    .read_text(encoding="utf-8")
    .replace("\r\n", "\n")
    .encode("utf-8")
).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_seed(dataset: str, model: str) -> int:
    """Use the same cell-specific resampling seed as the macro-F1 gap table."""
    key = f"{BOOTSTRAP_SEED}:{dataset}:{model}:erm-tail-gap".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def paired_bootstrap(
    gaps: np.ndarray, dataset: str, model: str
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(bootstrap_seed(dataset, model))
    sample_indices = rng.integers(
        0, gaps.size, size=(BOOTSTRAP_RESAMPLES, gaps.size)
    )
    bootstrap_means = gaps[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    smaller_tail_count = min(
        int(np.count_nonzero(bootstrap_means <= 0.0)),
        int(np.count_nonzero(bootstrap_means >= 0.0)),
    )
    p_value = min(
        1.0, 2.0 * smaller_tail_count / BOOTSTRAP_RESAMPLES
    )
    return float(lower), float(upper), float(p_value), smaller_tail_count


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        raise ValueError("Accuracy is undefined for an empty subset")
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have the same shape")
    return float(np.mean(y_true == y_pred))


def compute_run_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    test_idx: np.ndarray,
    tail_mask: np.ndarray,
) -> dict:
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional")
    if labels.shape != predictions.shape or labels.shape != tail_mask.shape:
        raise ValueError("Full-node arrays must have the same shape")
    if test_idx.ndim != 1:
        raise ValueError("test_idx must be one-dimensional")
    if test_idx.size == 0:
        raise ValueError("The test set is empty")
    if np.any(test_idx < 0) or np.any(test_idx >= labels.size):
        raise ValueError("test_idx contains an out-of-range node index")

    tail_test_idx = test_idx[tail_mask[test_idx]]
    if tail_test_idx.size == 0:
        raise ValueError("The test tail is empty")

    whole_accuracy = accuracy(labels[test_idx], predictions[test_idx])
    tail_accuracy = accuracy(
        labels[tail_test_idx], predictions[tail_test_idx]
    )
    return {
        "n_test": int(test_idx.size),
        "n_tail_test": int(tail_test_idx.size),
        "whole_accuracy": whole_accuracy,
        "tail_accuracy": tail_accuracy,
        "gap_whole_minus_tail": whole_accuracy - tail_accuracy,
    }


def registered_tail_bin(manifest: dict) -> tuple[int, dict]:
    tail_bins = [
        (index, bin_definition)
        for index, bin_definition in enumerate(
            manifest["local_homophily"]["merged_bins"]
        )
        if bin_definition["is_tail"]
    ]
    if len(tail_bins) != 1:
        raise RuntimeError(
            f"{manifest['dataset']}: expected one registered tail bin; "
            f"found {len(tail_bins)}"
        )
    tail_index, tail_definition = tail_bins[0]
    registered_interval = manifest["local_homophily"]["tail"]["interval"]
    if tail_definition["interval"] != registered_interval:
        raise RuntimeError(
            f"{manifest['dataset']}: tail interval differs across manifest fields"
        )
    return tail_index, tail_definition


def check_array(
    array: np.ndarray, expected_checksum: str, run_id: str, name: str
) -> str:
    actual_checksum = array_sha256(array)
    if actual_checksum != expected_checksum:
        raise RuntimeError(f"{run_id}: checksum mismatch for {name}")
    return actual_checksum


def calculate_run(run_dir: Path, config: dict, manifest: dict) -> dict:
    run_id = config["run_id"]
    split = config["split"]
    expected_checksums = manifest["array_checksums_sha256"]
    recorded_checksums = config["fixed_input_checksums"]
    required_manifest_keys = (
        "labels",
        f"{split}.test",
        "merged_bin",
        "tail_mask",
    )
    for key in required_manifest_keys:
        if recorded_checksums.get(key) != expected_checksums[key]:
            raise RuntimeError(
                f"{run_id}: config checksum differs from the registered "
                f"manifest for {key}"
            )

    with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        predictions = saved["predictions"]
        test_idx = saved["test_idx"]
        merged_bin = saved["merged_bin"]
        tail_mask = saved["tail_mask"]

    input_checksums = {
        "labels": check_array(
            labels, expected_checksums["labels"], run_id, "labels"
        ),
        "test_idx": check_array(
            test_idx,
            expected_checksums[f"{split}.test"],
            run_id,
            "test_idx",
        ),
        "merged_bin": check_array(
            merged_bin,
            expected_checksums["merged_bin"],
            run_id,
            "merged_bin",
        ),
        "tail_mask": check_array(
            tail_mask,
            expected_checksums["tail_mask"],
            run_id,
            "tail_mask",
        ),
        "predictions": array_sha256(predictions),
    }

    tail_bin_index, tail_bin_definition = registered_tail_bin(manifest)
    if not np.array_equal(tail_mask, merged_bin == tail_bin_index):
        raise RuntimeError(
            f"{run_id}: saved tail mask does not equal the registered tail bin"
        )

    calculated = compute_run_accuracy(
        labels, predictions, test_idx, tail_mask
    )

    stored_metrics = read_json(run_dir / "metrics.json")
    if stored_metrics.get("standard_metric") != "accuracy":
        raise RuntimeError(f"{run_id}: stored standard metric is not accuracy")
    stored_bins = stored_metrics["per_bin"]
    if len(stored_bins) != len(
        manifest["local_homophily"]["merged_bins"]
    ):
        raise RuntimeError(f"{run_id}: stored and registered bin counts differ")
    stored_tail = stored_bins[tail_bin_index]
    if stored_tail["bin"] != tail_bin_definition["interval"]:
        raise RuntimeError(f"{run_id}: stored tail interval is incorrect")

    whole_abs_error = abs(
        calculated["whole_accuracy"]
        - float(stored_metrics["whole_test_standard"])
    )
    tail_abs_error = abs(
        calculated["tail_accuracy"] - float(stored_tail["accuracy"])
    )
    if whole_abs_error > FLOAT_TOLERANCE or tail_abs_error > FLOAT_TOLERANCE:
        raise RuntimeError(
            f"{run_id}: recomputed accuracy differs from stored metrics"
        )

    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "run_id": run_id,
        "split": split,
        "seed": int(config["seed"]),
        "tail_side": manifest["local_homophily"]["tail_side"],
        "tail_interval": tail_bin_definition["interval"],
        **calculated,
        "whole_stored_abs_error": whole_abs_error,
        "tail_stored_abs_error": tail_abs_error,
        "labels_sha256": input_checksums["labels"],
        "test_idx_sha256": input_checksums["test_idx"],
        "tail_mask_sha256": input_checksums["tail_mask"],
        "predictions_sha256": input_checksums["predictions"],
        "training_code_version": config["code_version"],
    }


def discover_erm_runs(
    raw_root: Path,
    manifests: dict[str, dict],
    registrations: tuple[RegisteredRun, ...] | None = None,
) -> list[tuple[Path, dict]]:
    run_dirs = sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name != "historical-records"
    )
    unexpected_directories = [
        path.name for path in run_dirs if not (path / "config.json").is_file()
    ]
    if unexpected_directories:
        raise RuntimeError(
            "Unexpected non-run directories in raw root: "
            f"{unexpected_directories}"
        )

    selected: list[tuple[Path, dict]] = []
    all_configs: list[dict] = []
    seen_run_ids: set[str] = set()
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        all_configs.append(config)
        if config["dataset"] not in manifests or config["preparation"] != "erm":
            continue
        run_id = config["run_id"]
        if run_id != run_dir.name:
            raise RuntimeError(f"{run_dir.name}: directory and run_id differ")
        if run_id in seen_run_ids:
            raise RuntimeError(f"Duplicate run_id: {run_id}")
        if config["model"] not in MODEL_ORDER:
            raise RuntimeError(f"{run_id}: unexpected model {config['model']}")
        seen_run_ids.add(run_id)
        selected.append((run_dir, config))
    if registrations is not None:
        validate_registered_run_configs(
            all_configs,
            registrations,
            context="RQ1 accuracy input",
        )
    return selected


def validate_panel(rows: list[dict], manifests: dict[str, dict]) -> None:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["dataset"], row["model"])].append(row)

    expected_cells = {
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    }
    if set(by_cell) != expected_cells:
        missing = sorted(expected_cells - set(by_cell))
        unexpected = sorted(set(by_cell) - expected_cells)
        raise RuntimeError(
            f"Run-cell mismatch; missing={missing}, unexpected={unexpected}"
        )

    for dataset, model in sorted(expected_cells):
        expected_pairs = {
            (split["split_id"], int(seed))
            for split in manifests[dataset]["splits"]
            for seed in manifests[dataset]["training_seeds"]
        }
        cell_rows = by_cell[(dataset, model)]
        observed_pairs = {(row["split"], row["seed"]) for row in cell_rows}
        if len(cell_rows) != len(expected_pairs) or observed_pairs != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs)
            unexpected = sorted(observed_pairs - expected_pairs)
            raise RuntimeError(
                f"{dataset}/{model}: run panel mismatch; found "
                f"{len(cell_rows)} rows, expected {len(expected_pairs)}; "
                f"missing={missing}, unexpected={unexpected}"
            )


def summarise_runs(rows: list[dict]) -> list[dict]:
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["dataset"], row["model"])].append(row)

    summaries = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            cell_rows = sorted(
                by_cell[(dataset, model)],
                key=lambda row: (row["split"], row["seed"]),
            )
            whole = np.asarray(
                [row["whole_accuracy"] for row in cell_rows], dtype=np.float64
            )
            tail = np.asarray(
                [row["tail_accuracy"] for row in cell_rows], dtype=np.float64
            )
            gap = whole - tail
            lower, upper, p_value, smaller_tail_count = paired_bootstrap(
                gap, dataset, model
            )
            summaries.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "tail_side": cell_rows[0]["tail_side"],
                    "tail_interval": cell_rows[0]["tail_interval"],
                    "n_runs": len(cell_rows),
                    "mean_whole_accuracy": float(whole.mean()),
                    "sample_std_whole_accuracy": float(whole.std(ddof=1)),
                    "mean_tail_accuracy": float(tail.mean()),
                    "sample_std_tail_accuracy": float(tail.std(ddof=1)),
                    "mean_gap_whole_minus_tail": float(gap.mean()),
                    "sample_std_gap_whole_minus_tail": float(gap.std(ddof=1)),
                    "bootstrap_ci_lower_gap": lower,
                    "bootstrap_ci_upper_gap": upper,
                    "bootstrap_smaller_tail_count": smaller_tail_count,
                    "bootstrap_p_value": p_value,
                    "gap_ci_excludes_zero": lower > 0.0 or upper < 0.0,
                    "mean_n_test": float(
                        np.mean([row["n_test"] for row in cell_rows])
                    ),
                    "mean_n_tail_test": float(
                        np.mean([row["n_tail_test"] for row in cell_rows])
                    ),
                }
            )
    return summaries


def make_wide_table(summaries: list[dict]) -> list[dict]:
    lookup = {
        (row["dataset"], row["model"]): row for row in summaries
    }
    wide_rows = []
    for model in MODEL_ORDER:
        wide_row: dict[str, str | float] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
        }
        for dataset in DATASET_ORDER:
            summary = lookup[(dataset, model)]
            wide_row[f"{dataset}_whole_accuracy"] = summary[
                "mean_whole_accuracy"
            ]
            wide_row[f"{dataset}_tail_accuracy"] = summary[
                "mean_tail_accuracy"
            ]
        wide_rows.append(wide_row)
    return wide_rows


def format_signed(value: float) -> str:
    formatted = f"{value:+.3f}"
    if formatted in {"+0.000", "-0.000"}:
        return ".000" if tuple(DATASET_ORDER) == ADDITIONAL_DATASET_ORDER else "0.000"
    if tuple(DATASET_ORDER) == ADDITIONAL_DATASET_ORDER:
        return formatted.replace("+0.", "+.").replace("-0.", "-.")
    return formatted


def format_interval_endpoint(value: float) -> str:
    formatted = f"{value:.3f}"
    if tuple(DATASET_ORDER) == ADDITIONAL_DATASET_ORDER:
        if formatted in {"0.000", "-0.000"}:
            return ".000"
        return formatted.replace("-0.", "-.").replace("0.", ".")
    return "0.000" if formatted == "-0.000" else formatted


def latex_gap_cell(summary: dict) -> str:
    macro = (
        "gaincell"
        if summary["gap_ci_excludes_zero"]
        else "uncertaingaincell"
    )
    mean = format_signed(float(summary["mean_gap_whole_minus_tail"]))
    lower = format_interval_endpoint(float(summary["bootstrap_ci_lower_gap"]))
    upper = format_interval_endpoint(float(summary["bootstrap_ci_upper_gap"]))
    return f"\\{macro}{{{mean}}}{{[{lower},{upper}]}}"


def build_appendix_table(summaries: list[dict]) -> str:
    lookup = {
        (row["dataset"], row["model"]): row for row in summaries
    }
    paper_model_labels = {
        "mlp": "MLP",
        "gcn": "GCN",
        "graphsage-mean": "GraphSAGE-mean",
        "h2gcn": "H2GCN-2",
        "gpr-gnn": "GPR-GNN",
        "linkx": "LINKX",
        "acm-gcn": "ACM-GCN",
    }
    is_original = tuple(DATASET_ORDER) == tuple(ORIGINAL_DATASET_ORDER)
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
        column_spec = r"{@{\extracolsep{\fill}}lcc@{}}"
        group_lines = [
            r"    & \multicolumn{2}{c}{High-homophily tail} \\",
            r"    \cmidrule(lr){2-3}",
            r"    Model & Actor & BlogCatalog \\",
        ]
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        (
            (
                r"  \caption{ERM tail gap, whole-test accuracy minus tail "
                r"accuracy (mean difference and 95\% confidence interval). "
                r"Cells whose 95\% confidence interval contains 0 are greyed "
                r"out.}"
            )
            if is_original
            else (
                r"  \caption{ERM accuracy gap, whole-test accuracy minus tail "
                r"accuracy. Each cell gives the mean paired difference over "
                r"30 matched split--seed runs and its 95\% "
                r"percentile-bootstrap interval. Positive values indicate "
                r"lower tail accuracy; grey cells contain zero.}"
            )
        ),
        (
            r"  \label{tab:erm-tail-gap-accuracy}"
            if is_original
            else r"  \label{tab:additional-erm-tail-gap-accuracy}"
        ),
        r"  \scriptsize",
        r"  \begin{tabular*}{\linewidth}" + column_spec,
        r"    \toprule",
        *group_lines,
        r"    \midrule",
    ]
    table_model_order = MODEL_ORDER if is_original else PAPER_MODEL_ORDER
    for model in table_model_order:
        cells = [
            latex_gap_cell(lookup[(dataset, model)])
            for dataset in DATASET_ORDER
        ]
        lines.append(
            f"    {paper_model_labels[model]:<16} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialised = list(rows)
    if not materialised:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialised[0]))
        writer.writeheader()
        writer.writerows(materialised)


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
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="raw run store for Actor and BlogCatalog",
    )
    parser.add_argument(
        "--output-root",
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
    args.output_root = args.output_root or DEFAULT_OUTPUT_ROOTS[mode]
    return args


def main() -> None:
    global DATASET_ORDER

    args = parse_args()
    DATASET_ORDER = (
        ORIGINAL_DATASET_ORDER
        if args.dataset_mode is AnalysisDatasetMode.ORIGINAL
        else ADDITIONAL_DATASET_ORDER
    )

    manifests = {
        dataset: read_json(DATASET_REGISTRY[dataset].manifest_path(REPO_ROOT))
        for dataset in DATASET_ORDER
    }
    registrations = load_registered_runs(
        REPO_ROOT, args.dataset_mode, DATASET_ORDER
    )
    selected_runs = discover_erm_runs(
        args.input_root, manifests, registrations
    )
    expected_run_count = sum(
        len(manifests[dataset]["splits"])
        * len(manifests[dataset]["training_seeds"])
        * len(MODEL_ORDER)
        for dataset in DATASET_ORDER
    )
    if len(selected_runs) != expected_run_count:
        raise RuntimeError(
            f"Found {len(selected_runs)} ERM runs; expected {expected_run_count}"
        )

    run_rows = [
        calculate_run(run_dir, config, manifests[config["dataset"]])
        for run_dir, config in selected_runs
    ]
    validate_panel(run_rows, manifests)

    dataset_index = {value: index for index, value in enumerate(DATASET_ORDER)}
    model_index = {value: index for index, value in enumerate(MODEL_ORDER)}
    run_rows.sort(
        key=lambda row: (
            dataset_index[row["dataset"]],
            model_index[row["model"]],
            row["split"],
            row["seed"],
        )
    )
    summaries = summarise_runs(run_rows)
    wide_rows = make_wide_table(summaries)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "run_accuracy.csv"
    summary_path = output_root / "whole_vs_tail_accuracy.csv"
    wide_path = output_root / "whole_vs_tail_accuracy_wide.csv"
    latex_path = output_root / "appendix_table.tex"
    write_csv(run_path, run_rows)
    write_csv(summary_path, summaries)
    write_csv(wide_path, wide_rows)
    latex_path.write_text(build_appendix_table(summaries), encoding="utf-8")

    max_whole_error = max(row["whole_stored_abs_error"] for row in run_rows)
    max_tail_error = max(row["tail_stored_abs_error"] for row in run_rows)
    output_checksums = {
        path.name: file_sha256(path)
        for path in (run_path, summary_path, wide_path, latex_path)
    }
    analysis_manifest = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_script": "scripts/analyze_rq1_accuracy.py",
        "analysis_script_sha256": SCRIPT_SHA256,
        "scope": {
            "dataset_mode": args.dataset_mode.value,
            "preparation": "erm",
            "datasets": list(DATASET_ORDER),
            "models": list(MODEL_ORDER),
            "n_runs": len(run_rows),
            "n_cells": len(summaries),
            "registered_run_panel_exact_match": True,
        },
        "metric_definition": {
            "whole_accuracy": (
                "fraction of all test_idx nodes predicted correctly within "
                "each run"
            ),
            "tail_accuracy": (
                "fraction of test_idx nodes in the manifest-registered tail "
                "predicted correctly within each run"
            ),
            "gap": "whole_accuracy minus tail_accuracy within each run",
        },
        "aggregation": (
            "Unweighted arithmetic mean of the 30 split-seed run values in "
            "each dataset-model cell"
        ),
        "paired_bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "global_seed": BOOTSTRAP_SEED,
            "cell_seed_key": (
                "sha256(global_seed:dataset:model:erm-tail-gap), first 8 "
                "bytes interpreted little-endian"
            ),
            "resampled_values": "30 paired whole-minus-tail run gaps",
            "interval": "2.5th and 97.5th percentiles",
            "quantile_method": "linear",
            "two_sided_p_value": (
                "2 * min(fraction of bootstrap means <= 0, fraction >= 0), "
                "capped at 1"
            ),
            "multiple_comparison_correction": None,
        },
        "tail_definitions": {
            dataset: {
                "side": manifests[dataset]["local_homophily"]["tail_side"],
                "interval": manifests[dataset]["local_homophily"]["tail"][
                    "interval"
                ],
            }
            for dataset in DATASET_ORDER
        },
        "validation": {
            "complete_registered_split_seed_panel": True,
            "fixed_arrays_match_manifest_sha256": True,
            "maximum_whole_stored_absolute_error": max_whole_error,
            "maximum_tail_stored_absolute_error": max_tail_error,
            "stored_comparison_tolerance": FLOAT_TOLERANCE,
        },
        "training_code_versions": sorted(
            {row["training_code_version"] for row in run_rows}
        ),
        "output_sha256": output_checksums,
    }
    write_json(output_root / "analysis_manifest.json", analysis_manifest)
    print(
        f"Wrote {len(run_rows)} run rows and {len(summaries)} cell summaries "
        f"to {output_root}"
    )
    print(
        "Maximum absolute difference from stored accuracy: "
        f"whole={max_whole_error:.3g}, tail={max_tail_error:.3g}"
    )


if __name__ == "__main__":
    main()
