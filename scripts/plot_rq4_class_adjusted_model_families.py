"""Plot the class-composition-adjusted analogue of the RQ4 score figure.

The input is the run-level output of ``analyze_rq2_within_class.py``. For each
dataset, model, split, and seed, the tail confusion matrix has been reweighted
to the whole-test proportions of true classes present in the tail. The
whole-test score uses the same class set. This script plots the resulting
whole-test and tail macro-F1 means. Bootstrap intervals remain available in
the companion CSV.
The resulting figure is the class-adjusted sensitivity display used in the
RQ4 appendix.

Example:
    python scripts/plot_rq4_class_adjusted_model_families.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

from hfd.analysis_datasets import (
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    dataset_order,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_rq4_model_families import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    EXPECTED_RUNS,
    SCORE_SCOPES,
    ScoreSummary,
    bootstrap_seed,
    configure_dataset_mode as configure_rq4_dataset_mode,
    file_sha256,
    percentile_bootstrap_mean,
    relative_path,
    write_score_csv,
)
from plot_compact_main_figures import (  # noqa: E402
    MODEL_FAMILIES,
    MODEL_ORDER,
    plot_merged_figure,
)


DATASET_ORDER = ORIGINAL_DATASET_ORDER
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "acm-gcn": "ACM-GCN",
    "linkx": "LINKX",
}


def score_axis_upper_limit(
    mode: AnalysisDatasetMode | str,
) -> float:
    """Leave marker headroom when additional-dataset scores approach one."""

    selected_mode = AnalysisDatasetMode(mode)
    return 1.06 if selected_mode is AnalysisDatasetMode.ADDITIONAL else 1.0


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel and keep imported RQ4 writers in sync."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)
    configure_rq4_dataset_mode(mode)


@dataclass(frozen=True)
class AdjustedRunScore:
    split: str
    seed: int
    whole_macro_f1: float
    tail_macro_f1: float
    tail_gap: float
    tail_class_ids: tuple[int, ...]
    whole_reference_weights: tuple[float, ...]


def parse_semicolon_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(";") if item)


def parse_semicolon_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(";") if item)


def validate_adjusted_record(
    record: AdjustedRunScore,
    context: str,
) -> None:
    scores = np.asarray(
        [record.whole_macro_f1, record.tail_macro_f1],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{context}: non-finite macro-F1 component")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise RuntimeError(f"{context}: macro-F1 component lies outside [0, 1]")
    if not np.isclose(
        record.whole_macro_f1 - record.tail_macro_f1,
        record.tail_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{context}: adjusted gap differs from whole-test minus tail macro-F1"
        )
    if not record.tail_class_ids:
        raise RuntimeError(f"{context}: adjusted population is empty")
    if len(record.tail_class_ids) != len(record.whole_reference_weights):
        raise RuntimeError(
            f"{context}: tail class IDs and reference weights differ in length"
        )
    if not np.isclose(
        sum(record.whole_reference_weights),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{context}: whole-test reference weights do not sum to one"
        )
    if any(weight < 0.0 for weight in record.whole_reference_weights):
        raise RuntimeError(f"{context}: negative whole-test reference weight")
def load_adjusted_run_scores(
    path: Path,
    *,
    datasets: tuple[str, ...] = DATASET_ORDER,
    models: tuple[str, ...] = MODEL_ORDER,
    expected_runs: int = EXPECTED_RUNS,
) -> dict[
    tuple[str, str],
    dict[tuple[str, int], AdjustedRunScore],
]:
    if not path.is_file():
        raise FileNotFoundError(f"Adjusted run-metric table not found: {path}")

    required = {
        "dataset",
        "model",
        "preparation",
        "split",
        "seed",
        "tail_classes",
        "tail_class_ids",
        "whole_reference_weights",
        "whole_macro_f1",
        "adjusted_tail_macro_f1",
        "adjusted_macro_f1_gap",
    }
    records: dict[
        tuple[str, str],
        dict[tuple[str, int], AdjustedRunScore],
    ] = {
        (dataset, model): {}
        for dataset in datasets
        for model in models
    }

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise RuntimeError(f"{path}: missing adjusted-run columns {missing}")

        for row in reader:
            dataset = row["dataset"]
            model = row["model"]
            if row["preparation"] != "erm":
                continue
            if dataset not in datasets or model not in models:
                continue

            split = row["split"]
            seed = int(row["seed"])
            key = (split, seed)
            cell = records[(dataset, model)]
            if key in cell:
                raise RuntimeError(
                    f"Duplicate adjusted ERM split-seed pair for "
                    f"{dataset}/{model}: {key}"
                )

            class_ids = parse_semicolon_ints(row["tail_class_ids"])
            reference_weights = parse_semicolon_floats(
                row["whole_reference_weights"]
            )
            expected_classes = int(row["tail_classes"])
            if len(class_ids) != expected_classes:
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "tail-class count does not match the class IDs"
                )
            if len(reference_weights) != expected_classes:
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "tail-class count does not match the reference weights"
                )

            record = AdjustedRunScore(
                split=split,
                seed=seed,
                whole_macro_f1=float(row["whole_macro_f1"]),
                tail_macro_f1=float(row["adjusted_tail_macro_f1"]),
                tail_gap=float(row["adjusted_macro_f1_gap"]),
                tail_class_ids=class_ids,
                whole_reference_weights=reference_weights,
            )
            validate_adjusted_record(
                record,
                f"{dataset}/{model}/{split}/seed-{seed}",
            )
            cell[key] = record

    for dataset in datasets:
        for model in models:
            n_runs = len(records[(dataset, model)])
            if n_runs != expected_runs:
                raise RuntimeError(
                    f"{dataset}/{model}: found {n_runs} adjusted ERM runs; "
                    f"expected {expected_runs}"
                )
    return records


def validate_common_adjustment_support(
    all_runs: dict[
        tuple[str, str],
        dict[tuple[str, int], AdjustedRunScore],
    ],
    *,
    datasets: tuple[str, ...] = DATASET_ORDER,
    models: tuple[str, ...] = MODEL_ORDER,
) -> None:
    for dataset in datasets:
        reference_model = models[0]
        reference_runs = all_runs[(dataset, reference_model)]
        reference_keys = set(reference_runs)
        for model in models[1:]:
            model_runs = all_runs[(dataset, model)]
            if set(model_runs) != reference_keys:
                raise RuntimeError(
                    f"{dataset}/{model}: split-seed panel differs from "
                    f"{reference_model}"
                )
            for key in sorted(reference_keys):
                reference = reference_runs[key]
                candidate = model_runs[key]
                if (
                    candidate.tail_class_ids
                    != reference.tail_class_ids
                ):
                    raise RuntimeError(
                        f"{dataset}/{model}/{key}: tail class IDs differ "
                        f"from {reference_model}"
                    )
                if not np.allclose(
                    candidate.whole_reference_weights,
                    reference.whole_reference_weights,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"{dataset}/{model}/{key}: reference weights differ "
                        f"from {reference_model}"
                    )
def summarise_adjusted_scores(
    all_runs: dict[
        tuple[str, str],
        dict[tuple[str, int], AdjustedRunScore],
    ],
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> list[ScoreSummary]:
    summaries: list[ScoreSummary] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            records = all_runs[(dataset, model)]
            keys = sorted(records)
            for scope in SCORE_SCOPES:
                attribute = (
                    "whole_macro_f1"
                    if scope == "whole_test"
                    else "tail_macro_f1"
                )
                values = np.asarray(
                    [getattr(records[key], attribute) for key in keys],
                    dtype=np.float64,
                )
                lower, upper, _ = percentile_bootstrap_mean(
                    values,
                    seed=bootstrap_seed(
                        base_seed,
                        dataset,
                        model,
                        scope,
                        "class-adjusted-absolute-macro-f1",
                    ),
                    resamples=bootstrap_resamples,
                )
                summaries.append(
                    ScoreSummary(
                        dataset=dataset,
                        model=model,
                        scope=scope,
                        n_runs=values.size,
                        mean_macro_f1=float(values.mean()),
                        sample_std_macro_f1=float(values.std(ddof=1)),
                        ci_lower=lower,
                        ci_upper=upper,
                        run_scores=tuple(float(value) for value in values),
                    )
                )
    return summaries


def support_summary(
    all_runs: dict[
        tuple[str, str],
        dict[tuple[str, int], AdjustedRunScore],
    ],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for dataset in DATASET_ORDER:
        runs = all_runs[(dataset, MODEL_ORDER[0])]
        tail_classes = np.asarray(
            [len(run.tail_class_ids) for run in runs.values()],
            dtype=np.int64,
        )
        summary[dataset] = {
            "minimum_tail_classes": int(tail_classes.min()),
            "maximum_tail_classes": int(tail_classes.max()),
        }
    return summary


def mean_score_cells(
    summaries: list[ScoreSummary],
) -> dict[tuple[str, str], tuple[float, float]]:
    means = {
        (summary.dataset, summary.model, summary.scope): (
            summary.mean_macro_f1
        )
        for summary in summaries
    }
    expected = {
        (dataset, model, scope)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for scope in SCORE_SCOPES
    }
    if set(means) != expected:
        missing = sorted(expected - set(means))
        unexpected = sorted(set(means) - expected)
        raise RuntimeError(
            "Class-adjusted score panel does not match Figure 5: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        (dataset, model): (
            means[(dataset, model, "whole_test")],
            means[(dataset, model, "tail")],
        )
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    }


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    all_runs: dict[
        tuple[str, str],
        dict[tuple[str, int], AdjustedRunScore],
    ],
    outputs: list[Path],
) -> None:
    payload = {
        "analysis": "RQ4 class-composition-adjusted score figure",
        "dataset_mode": args.dataset_mode.value,
        "datasets": list(DATASET_ORDER),
        "estimand": {
            "reference_distribution": (
                "whole-test proportions of true classes present in the tail"
            ),
            "whole_component": (
                "ordinary whole-test macro-F1 over tail-present true classes"
            ),
            "tail_component": "class-adjusted tail macro-F1",
            "gap": "whole component minus tail component",
        },
        "model_families": {
            family: [
                MODEL_LABELS[model]
                for model in MODEL_ORDER
                if MODEL_FAMILIES[model] == family
            ]
            for family in (
                "mlp",
                "standard",
                "heterophily-designed",
                "non-aggregation",
            )
        },
        "model_order": [MODEL_LABELS[model] for model in MODEL_ORDER],
        "visual_encoding": {
            "whole_test": "filled diamond",
            "tail": "filled circle",
            "connector": "solid line for each dataset-model cell",
            "model_family": "same colour mapping as main-text Figure 5",
            "confidence_intervals_shown": False,
            "y_axis_range": [
                0.0,
                score_axis_upper_limit(args.dataset_mode),
            ],
        },
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "base_seed": args.bootstrap_seed,
            "unit": "complete split-seed run",
            "interval": "95% percentile",
        },
        "input": {
            "path": relative_path(args.class_adjusted_run_metrics),
            "sha256": file_sha256(args.class_adjusted_run_metrics),
        },
        "support": support_summary(all_runs),
        "outputs": [
            {
                "path": relative_path(output),
                "sha256": file_sha256(output),
            }
            for output in outputs
            if output != path
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(mode.value for mode in AnalysisDatasetMode),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--class-adjusted-run-metrics",
        type=Path,
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    args = parser.parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    result_root = REPO_ROOT / "results" / (
        "main" if mode is AnalysisDatasetMode.ORIGINAL else mode.value
    )
    args.dataset_mode = mode
    args.class_adjusted_run_metrics = (
        args.class_adjusted_run_metrics
        or result_root / "rq2" / "class_adjusted_run_metrics.csv"
    )
    args.analysis_dir = args.analysis_dir or result_root / "rq4"
    args.figure_dir = args.figure_dir or result_root / "figures" / "rq4"
    return args


def main() -> None:
    args = parse_args()
    configure_dataset_mode(args.dataset_mode)
    if args.bootstrap_resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")

    args.class_adjusted_run_metrics = (
        args.class_adjusted_run_metrics.resolve()
    )
    args.analysis_dir = args.analysis_dir.resolve()
    args.figure_dir = args.figure_dir.resolve()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_adjusted_run_scores(
        args.class_adjusted_run_metrics, datasets=DATASET_ORDER
    )
    validate_common_adjustment_support(all_runs, datasets=DATASET_ORDER)
    summaries = summarise_adjusted_scores(
        all_runs,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )

    score_csv = (
        args.analysis_dir / "class_adjusted_erm_whole_tail_macro_f1.csv"
    )
    figure_stem = (
        args.figure_dir
        / "rq4_class_adjusted_whole_test_and_tail_macro_f1_by_model_family"
    )
    manifest = args.analysis_dir / "class_adjusted_figure_manifest.json"

    write_score_csv(
        score_csv,
        summaries,
        model_order=MODEL_ORDER,
        model_labels=MODEL_LABELS,
        model_families=MODEL_FAMILIES,
    )
    figure_outputs = plot_merged_figure(
        mean_score_cells(summaries),
        figure_stem,
        models=MODEL_ORDER,
        formats=tuple(args.formats),
        dpi=args.dpi,
        datasets=DATASET_ORDER,
        y_upper=score_axis_upper_limit(args.dataset_mode),
    )
    outputs = [*figure_outputs, score_csv, manifest]
    write_manifest(
        manifest,
        args=args,
        all_runs=all_runs,
        outputs=outputs,
    )

    print(
        f"Analysed {len(summaries)} class-adjusted score cells from "
        f"{sum(len(cell) for cell in all_runs.values())} ERM runs."
    )
    for output in outputs:
        print(relative_path(output))


if __name__ == "__main__":
    main()
