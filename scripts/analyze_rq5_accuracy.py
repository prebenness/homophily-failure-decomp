"""Compute and plot RQ5 tail and whole-test accuracy by preparation.

Accuracy is recomputed from each run's saved labels and predictions. The
calculated values are checked against the canonical raw metrics before cell
means are plotted.

Example:
    python scripts/analyze_rq5_accuracy.py \
        --raw-root PATH-TO-CANONICAL-RUNS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_rq1_accuracy import calculate_run, file_sha256, read_json
from analyze_rq5_preparations import (
    DATASET_LABELS,
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    PLOT_MODEL_ORDER,
    PREPARATION_LABELS,
    PREPARATION_ORDER,
    REPO_ROOT,
    TAIL_LABELS,
    configure_style,
    relative_path,
    y_axis_spec,
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


ANALYSIS_VERSION = "rq5-tail-whole-accuracy-v1"
EXPECTED_RUNS_PER_CELL = 30
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AccuracySummary:
    dataset: str
    model: str
    preparation: str
    n_runs: int
    mean_tail_accuracy: float
    sample_std_tail_accuracy: float
    mean_whole_test_accuracy: float
    sample_std_whole_test_accuracy: float
    mean_n_tail_test: float
    mean_n_test: float


RUN_FIELDNAMES = (
    "dataset",
    "model",
    "preparation",
    "run_id",
    "split",
    "seed",
    "tail_side",
    "tail_interval",
    "n_test",
    "n_tail_test",
    "whole_accuracy",
    "tail_accuracy",
    "gap_whole_minus_tail",
    "whole_stored_abs_error",
    "tail_stored_abs_error",
    "labels_sha256",
    "test_idx_sha256",
    "tail_mask_sha256",
    "predictions_sha256",
    "training_code_version",
    "raw_config_sha256",
    "raw_metrics_sha256",
    "corrected_record_sha256",
)


def discover_runs(raw_root: Path) -> list[tuple[Path, dict]]:
    run_dirs = sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name != "historical-records"
    )
    unexpected = [
        path.name for path in run_dirs if not (path / "config.json").is_file()
    ]
    if unexpected:
        raise RuntimeError(
            f"Unexpected non-run directories in raw root: {unexpected}"
        )

    selected: list[tuple[Path, dict]] = []
    seen_run_ids: set[str] = set()
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        dataset = str(config.get("dataset"))
        model = str(config.get("model"))
        preparation = str(config.get("preparation"))
        if dataset not in DATASET_ORDER:
            continue
        if model not in MODEL_ORDER:
            raise RuntimeError(
                f"{run_dir.name}: unexpected model {model!r}"
            )
        if preparation not in PREPARATION_ORDER:
            raise RuntimeError(
                f"{run_dir.name}: unexpected preparation {preparation!r}"
            )

        run_id = str(config.get("run_id"))
        if run_id != run_dir.name:
            raise RuntimeError(f"{run_dir.name}: directory and run_id differ")
        if run_id in seen_run_ids:
            raise RuntimeError(f"Duplicate run_id: {run_id}")
        seen_run_ids.add(run_id)
        selected.append((run_dir, config))
    return selected


def validate_corrected_record(config: dict) -> tuple[Path, str]:
    record_path = (
        REPO_ROOT
        / "results"
        / "main"
        / "runs"
        / str(config["dataset"])
        / str(config["model"])
        / f"{config['run_id']}.json"
    )
    if not record_path.is_file():
        raise FileNotFoundError(
            f"Corrected repository record not found: {record_path}"
        )
    record = read_json(record_path)
    for field in (
        "run_id",
        "dataset",
        "model",
        "preparation",
        "split",
        "seed",
    ):
        if record.get(field) != config.get(field):
            raise RuntimeError(
                f"{config['run_id']}: raw and corrected {field} differ"
            )
    return record_path, file_sha256(record_path)


def calculate_all_runs(
    raw_root: Path,
    manifests: dict[str, dict],
) -> list[dict]:
    selected = discover_runs(raw_root)
    expected_count = (
        len(DATASET_ORDER)
        * len(MODEL_ORDER)
        * len(PREPARATION_ORDER)
        * EXPECTED_RUNS_PER_CELL
    )
    if len(selected) != expected_count:
        raise RuntimeError(
            f"Found {len(selected)} active runs; expected {expected_count}"
        )

    rows: list[dict] = []
    for run_dir, config in selected:
        metrics_path = run_dir / "metrics.json"
        metrics = read_json(metrics_path)
        canonicalization = metrics.get("canonicalization", {})
        if canonicalization.get("status") != "canonical active result":
            raise RuntimeError(
                f"{config['run_id']}: raw metrics are not canonical active"
            )
        record_path, record_sha256 = validate_corrected_record(config)
        calculated = calculate_run(
            run_dir,
            config,
            manifests[str(config["dataset"])],
        )
        row = {
            "dataset": calculated["dataset"],
            "model": calculated["model"],
            "preparation": str(config["preparation"]),
            **{
                key: value
                for key, value in calculated.items()
                if key not in {"dataset", "model"}
            },
            "raw_config_sha256": file_sha256(run_dir / "config.json"),
            "raw_metrics_sha256": file_sha256(metrics_path),
            "corrected_record_sha256": record_sha256,
        }
        if not np.isfinite(
            [row["whole_accuracy"], row["tail_accuracy"]]
        ).all():
            raise RuntimeError(
                f"{config['run_id']}: non-finite accuracy value"
            )
        if not (
            0.0 <= float(row["whole_accuracy"]) <= 1.0
            and 0.0 <= float(row["tail_accuracy"]) <= 1.0
        ):
            raise RuntimeError(
                f"{config['run_id']}: accuracy outside [0, 1]"
            )
        rows.append(row)

    dataset_index = {value: index for index, value in enumerate(DATASET_ORDER)}
    model_index = {value: index for index, value in enumerate(MODEL_ORDER)}
    preparation_index = {
        value: index for index, value in enumerate(PREPARATION_ORDER)
    }
    rows.sort(
        key=lambda row: (
            dataset_index[row["dataset"]],
            model_index[row["model"]],
            preparation_index[row["preparation"]],
            row["split"],
            row["seed"],
        )
    )
    validate_panel(rows, manifests)
    return rows


def validate_panel(rows: list[dict], manifests: dict[str, dict]) -> None:
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[
            (row["dataset"], row["model"], row["preparation"])
        ].append(row)

    expected_cells = {
        (dataset, model, preparation)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for preparation in PREPARATION_ORDER
    }
    if set(by_cell) != expected_cells:
        missing = sorted(expected_cells - set(by_cell))
        unexpected = sorted(set(by_cell) - expected_cells)
        raise RuntimeError(
            f"Run-cell mismatch; missing={missing}, unexpected={unexpected}"
        )

    for dataset, model, preparation in sorted(expected_cells):
        expected_pairs = {
            (split["split_id"], int(seed))
            for split in manifests[dataset]["splits"]
            for seed in manifests[dataset]["training_seeds"]
        }
        cell_rows = by_cell[(dataset, model, preparation)]
        observed_pairs = {
            (row["split"], int(row["seed"])) for row in cell_rows
        }
        if (
            len(cell_rows) != EXPECTED_RUNS_PER_CELL
            or observed_pairs != expected_pairs
        ):
            missing = sorted(expected_pairs - observed_pairs)
            unexpected = sorted(observed_pairs - expected_pairs)
            raise RuntimeError(
                f"{dataset}/{model}/{preparation}: run panel mismatch; "
                f"found {len(cell_rows)} rows, expected "
                f"{EXPECTED_RUNS_PER_CELL}; missing={missing}, "
                f"unexpected={unexpected}"
            )


def summarise_runs(rows: list[dict]) -> list[AccuracySummary]:
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[
            (row["dataset"], row["model"], row["preparation"])
        ].append(row)

    summaries: list[AccuracySummary] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            for preparation in PREPARATION_ORDER:
                cell_rows = by_cell[(dataset, model, preparation)]
                tail = np.asarray(
                    [row["tail_accuracy"] for row in cell_rows],
                    dtype=np.float64,
                )
                whole = np.asarray(
                    [row["whole_accuracy"] for row in cell_rows],
                    dtype=np.float64,
                )
                summaries.append(
                    AccuracySummary(
                        dataset=dataset,
                        model=model,
                        preparation=preparation,
                        n_runs=len(cell_rows),
                        mean_tail_accuracy=float(tail.mean()),
                        sample_std_tail_accuracy=float(tail.std(ddof=1)),
                        mean_whole_test_accuracy=float(whole.mean()),
                        sample_std_whole_test_accuracy=float(
                            whole.std(ddof=1)
                        ),
                        mean_n_tail_test=float(
                            np.mean(
                                [row["n_tail_test"] for row in cell_rows]
                            )
                        ),
                        mean_n_test=float(
                            np.mean([row["n_test"] for row in cell_rows])
                        ),
                    )
                )
    return summaries


def validate_erm_reference(
    summaries: list[AccuracySummary],
    reference_path: Path,
) -> float:
    with reference_path.open(encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference = {
        (row["dataset"], row["model"]): row for row in reference_rows
    }
    differences: list[float] = []
    for summary in summaries:
        if summary.preparation != "erm":
            continue
        key = (summary.dataset, summary.model)
        if key not in reference:
            raise RuntimeError(f"Missing ERM accuracy reference for {key}")
        expected = reference[key]
        differences.extend(
            (
                abs(
                    summary.mean_tail_accuracy
                    - float(expected["mean_tail_accuracy"])
                ),
                abs(
                    summary.mean_whole_test_accuracy
                    - float(expected["mean_whole_accuracy"])
                ),
            )
        )
    maximum = max(differences, default=0.0)
    if maximum > FLOAT_TOLERANCE:
        raise RuntimeError(
            "RQ5 ERM accuracy means differ from the audited RQ1 values; "
            f"maximum absolute difference={maximum}"
        )
    return maximum


def plot_dataset(
    dataset: str,
    summaries: list[AccuracySummary],
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    configure_style()
    by_cell = {
        (row.model, row.preparation): row
        for row in summaries
        if row.dataset == dataset and row.model in PLOT_MODEL_ORDER
    }
    expected = len(PLOT_MODEL_ORDER) * len(PREPARATION_ORDER)
    if len(by_cell) != expected:
        raise RuntimeError(
            f"{dataset}: found {len(by_cell)} accuracy cells; "
            f"expected {expected}"
        )

    metric_colours = {
        "tail": "#2f6f8f",
        "whole_test": "#a65e3b",
    }
    metric_dashes = {
        "tail": (0, (2.4, 1.8)),
        "whole_test": (0, (4.4, 2.0)),
    }
    preparation_offsets = np.linspace(-0.18, 0.18, len(PREPARATION_ORDER))
    model_positions = np.arange(len(PLOT_MODEL_ORDER), dtype=float)
    figure, axis = plt.subplots(figsize=(5.5, 3.0))

    for model_index, model in enumerate(PLOT_MODEL_ORDER):
        x_positions = model_index + preparation_offsets
        tail_values = [
            by_cell[(model, preparation)].mean_tail_accuracy
            for preparation in PREPARATION_ORDER
        ]
        whole_test_values = [
            by_cell[(model, preparation)].mean_whole_test_accuracy
            for preparation in PREPARATION_ORDER
        ]
        axis.plot(
            x_positions,
            tail_values,
            color=metric_colours["tail"],
            linewidth=0.8,
            linestyle=metric_dashes["tail"],
            marker="o",
            markersize=3.8,
            markerfacecolor=metric_colours["tail"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            dash_capstyle="round",
            zorder=2,
        )
        axis.plot(
            x_positions,
            whole_test_values,
            color=metric_colours["whole_test"],
            linewidth=0.8,
            linestyle=metric_dashes["whole_test"],
            marker="o",
            markersize=4.5,
            markerfacecolor="none",
            markeredgecolor=metric_colours["whole_test"],
            markeredgewidth=1.0,
            dash_capstyle="round",
            zorder=3,
        )

    plotted_values = [
        value
        for row in by_cell.values()
        for value in (
            row.mean_tail_accuracy,
            row.mean_whole_test_accuracy,
        )
    ]
    y_lower, y_upper, y_ticks = y_axis_spec(plotted_values)
    axis.set_xlim(-0.55, len(PLOT_MODEL_ORDER) - 0.45)
    axis.set_ylim(y_lower, y_upper)
    axis.set_xticks(model_positions)
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in PLOT_MODEL_ORDER]
    )
    axis.set_yticks(y_ticks)
    axis.set_ylabel("Accuracy")
    axis.set_xlabel("")
    axis.set_title(
        f"{DATASET_LABELS[dataset]}: {TAIL_LABELS[dataset]}",
        loc="left",
        pad=8,
        fontsize=9.0,
    )
    axis.grid(axis="y", color="0.90", linewidth=0.5)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0, pad=5)
    axis.tick_params(axis="y", direction="out", length=2.5, width=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    metric_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle=metric_dashes["tail"],
            color=metric_colours["tail"],
            markerfacecolor=metric_colours["tail"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=3.8,
            linewidth=0.8,
            label="Tail",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle=metric_dashes["whole_test"],
            color=metric_colours["whole_test"],
            markerfacecolor="none",
            markeredgecolor=metric_colours["whole_test"],
            markeredgewidth=1.0,
            markersize=4.5,
            linewidth=0.8,
            label="Whole test",
        ),
    ]
    legend_axis = figure.add_axes([0.68, 0.88, 0.305, 0.10], zorder=10)
    legend_axis.set_axis_off()
    legend = legend_axis.legend(
        handles=metric_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.3,
        borderaxespad=0.0,
        fontsize=7.5,
    )

    order_note = figure.text(
        0.5,
        0.018,
        (
            r"Preparation order within each model: "
            r"$A_{\mathrm{ERM}}$  $\cdot$  $A_{\mathrm{last}}$  "
            r"$\cdot$  $A_{\mathrm{head}}$  $\cdot$  "
            r"$A_{\mathrm{full}}$"
        ),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="0.36",
    )
    figure.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.19)
    metadata = {
        "Title": (
            f"RQ5 {DATASET_LABELS[dataset]} tail and whole-test accuracy "
            "by graph model and preparation"
        ),
        "Author": "homophily-failure-decomp",
        "Subject": "RQ5",
        "CreationDate": None,
        "ModDate": None,
    }
    outputs: list[Path] = []
    for output_format in formats:
        output = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=(legend, order_note),
            pad_inches=0.10,
            metadata=metadata,
        )
        outputs.append(output)
    plt.close(figure)
    return outputs


def write_run_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in RUN_FIELDNAMES} for row in rows
        )


def write_summary_csv(
    path: Path,
    summaries: list[AccuracySummary],
) -> None:
    fieldnames = (
        "dataset_order",
        "dataset",
        "dataset_label",
        "model_order",
        "model",
        "model_label",
        "preparation_order",
        "preparation",
        "preparation_label",
        "n_runs",
        "mean_tail_accuracy",
        "sample_std_tail_accuracy",
        "mean_whole_test_accuracy",
        "sample_std_whole_test_accuracy",
        "mean_n_tail_test",
        "mean_n_test",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            if row.model not in PLOT_MODEL_ORDER:
                continue
            writer.writerow(
                {
                    "dataset_order": DATASET_ORDER.index(row.dataset),
                    "dataset": row.dataset,
                    "dataset_label": DATASET_LABELS[row.dataset],
                    "model_order": PLOT_MODEL_ORDER.index(row.model),
                    "model": row.model,
                    "model_label": MODEL_LABELS[row.model].replace("\n", ""),
                    "preparation_order": PREPARATION_ORDER.index(
                        row.preparation
                    ),
                    "preparation": row.preparation,
                    "preparation_label": PREPARATION_LABELS[
                        row.preparation
                    ],
                    "n_runs": row.n_runs,
                    "mean_tail_accuracy": f"{row.mean_tail_accuracy:.10f}",
                    "sample_std_tail_accuracy": (
                        f"{row.sample_std_tail_accuracy:.10f}"
                    ),
                    "mean_whole_test_accuracy": (
                        f"{row.mean_whole_test_accuracy:.10f}"
                    ),
                    "sample_std_whole_test_accuracy": (
                        f"{row.sample_std_whole_test_accuracy:.10f}"
                    ),
                    "mean_n_tail_test": f"{row.mean_n_tail_test:.10f}",
                    "mean_n_test": f"{row.mean_n_test:.10f}",
                }
            )


def write_readme(path: Path) -> None:
    path.write_text(
        """# RQ5 accuracy figures

`preparation_accuracy_run_values.csv` contains the recomputed tail and
whole-test accuracy for all 4,200 runs. Each value is calculated from the
saved labels and predictions and checked against the canonical raw metrics.

`preparation_tail_vs_whole_accuracy.csv` reports the means plotted in the
five figures. It contains the six graph models and omits MLP.

Regenerate the CSV files and the PDF/PNG figures with:

```powershell
python scripts/analyze_rq5_accuracy.py `
  --raw-root C:\\path\\to\\heterophilic-graph-paper
```

The four preparations are ordered from left to right within each model:
`A_erm`, `A_last`, `A_head`, and `A_full`. Dashed blue lines with filled
circles show tail accuracy. Dashed rust lines with open circles show
whole-test accuracy. MLP is retained in the run-level calculations but
omitted from the figures because it is not a graph model.

Figure text uses the same font stack and dimensions as the corresponding RQ5
macro-F1 figures.
""",
        encoding="utf-8",
    )


def run_panel_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for field in (
            "run_id",
            "predictions_sha256",
            "raw_config_sha256",
            "raw_metrics_sha256",
            "corrected_record_sha256",
        ):
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def write_manifest(
    path: Path,
    *,
    rows: list[dict],
    summaries: list[AccuracySummary],
    outputs: list[Path],
    erm_reference_maximum_difference: float,
) -> None:
    manifests = {
        dataset: read_json(
            REPO_ROOT / "manifests" / "datasets" / f"{dataset}.json"
        )
        for dataset in DATASET_ORDER
    }
    payload = {
        "schema_version": 1,
        "analysis": "rq5-tail-and-whole-test-accuracy-by-preparation",
        "analysis_version": ANALYSIS_VERSION,
        "scope": {
            "datasets": list(DATASET_ORDER),
            "models": list(MODEL_ORDER),
            "plotted_models": list(PLOT_MODEL_ORDER),
            "preparations": list(PREPARATION_ORDER),
            "n_runs": len(rows),
            "n_cells": len(summaries),
            "runs_per_cell": EXPECTED_RUNS_PER_CELL,
        },
        "definitions": {
            "whole_test_accuracy": (
                "fraction of all test_idx nodes predicted correctly "
                "within each run"
            ),
            "tail_accuracy": (
                "fraction of test_idx nodes in the manifest-registered "
                "tail predicted correctly within each run"
            ),
            "cell_mean": "unweighted arithmetic mean over 30 runs",
        },
        "tail_definitions": {
            dataset: {
                "side": manifests[dataset]["local_homophily"][
                    "tail_side"
                ],
                "interval": manifests[dataset]["local_homophily"]["tail"][
                    "interval"
                ],
            }
            for dataset in DATASET_ORDER
        },
        "validation": {
            "complete_registered_split_seed_panel": True,
            "raw_and_corrected_run_identifiers_match": True,
            "fixed_arrays_match_manifest_sha256": True,
            "recomputed_values_match_stored_accuracy": True,
            "maximum_whole_stored_absolute_error": max(
                float(row["whole_stored_abs_error"]) for row in rows
            ),
            "maximum_tail_stored_absolute_error": max(
                float(row["tail_stored_abs_error"]) for row in rows
            ),
            "stored_comparison_tolerance": FLOAT_TOLERANCE,
            "erm_means_match_rq1_accuracy_audit": True,
            "maximum_erm_reference_absolute_difference": (
                erm_reference_maximum_difference
            ),
        },
        "input_run_panel_sha256": run_panel_sha256(rows),
        "analysis_scripts": {
            "scripts/analyze_rq5_accuracy.py": file_sha256(Path(__file__)),
            "scripts/analyze_rq1_accuracy.py": file_sha256(
                REPO_ROOT / "scripts" / "analyze_rq1_accuracy.py"
            ),
            "scripts/analyze_rq5_preparations.py": file_sha256(
                REPO_ROOT / "scripts" / "analyze_rq5_preparations.py"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "outputs": [
            {
                "path": relative_path(output),
                "sha256": file_sha256(output),
            }
            for output in outputs
            if output.exists() and output != path
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="canonical raw run archive containing run_data.npz",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "rq5",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "figures" / "rq5",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")
    args.raw_root = args.raw_root.resolve()
    args.analysis_dir = args.analysis_dir.resolve()
    args.figure_dir = args.figure_dir.resolve()
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"Raw run archive not found: {args.raw_root}")

    manifests = {
        dataset: read_json(
            REPO_ROOT / "manifests" / "datasets" / f"{dataset}.json"
        )
        for dataset in DATASET_ORDER
    }
    rows = calculate_all_runs(args.raw_root, manifests)
    summaries = summarise_runs(rows)
    erm_reference_maximum_difference = validate_erm_reference(
        summaries,
        REPO_ROOT
        / "results"
        / "main"
        / "rq1"
        / "accuracy"
        / "whole_vs_tail_accuracy.csv",
    )

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    run_csv = args.analysis_dir / "preparation_accuracy_run_values.csv"
    summary_csv = (
        args.analysis_dir / "preparation_tail_vs_whole_accuracy.csv"
    )
    readme = args.analysis_dir / "ACCURACY.md"
    manifest_path = args.analysis_dir / "accuracy_analysis_manifest.json"
    write_run_csv(run_csv, rows)
    write_summary_csv(summary_csv, summaries)
    write_readme(readme)

    figure_outputs: list[Path] = []
    for dataset in DATASET_ORDER:
        figure_outputs.extend(
            plot_dataset(
                dataset,
                summaries,
                args.figure_dir
                / f"rq5_{dataset}_tail_vs_whole_accuracy_by_preparation",
                formats=tuple(args.formats),
                dpi=args.dpi,
            )
        )

    outputs = [run_csv, summary_csv, readme, *figure_outputs]
    write_manifest(
        manifest_path,
        rows=rows,
        summaries=summaries,
        outputs=outputs,
        erm_reference_maximum_difference=(
            erm_reference_maximum_difference
        ),
    )
    print(
        f"Analysed {len(rows)} runs in {len(summaries)} cells and wrote "
        f"{len(figure_outputs)} accuracy figure files."
    )
    for output in [*outputs, manifest_path]:
        print(relative_path(output))


if __name__ == "__main__":
    main()
