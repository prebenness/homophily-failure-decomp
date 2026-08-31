"""Plot RQ5 tail and non-tail macro-F1 by preparation.

The compact figure preserves the layout and visual encoding of the earlier
tail-versus-whole-test figure, but replaces the whole-test series with
macro-F1 on the complement of the registered tail.

The non-tail score is recalculated from the node-level predictions in the
canonical raw archive because macro-F1 is not additive across local-homophily
bins.

Example:
    python scripts/plot_rq5_tail_vs_non_tail.py \
        --raw-root PATH-TO-CANONICAL-RUNS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from hfd.analysis_datasets import (
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
    load_registered_runs,
    run_panel_paths,
)

DATASET_ORDER = ORIGINAL_DATASET_ORDER
DATASET_LABELS = {
    name: metadata.display_label for name, metadata in DATASET_REGISTRY.items()
}
MODEL_ORDER = (
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage-mean": "SAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR",
    "acm-gcn": "ACM",
    "linkx": "LINKX",
}
PREPARATION_ORDER = ("erm", "last", "head", "full")
PREPARATION_LABELS = {
    "erm": r"$A_{\mathrm{erm}}$",
    "last": r"$A_{\mathrm{last}}$",
    "head": r"$A_{\mathrm{head}}$",
    "full": r"$A_{\mathrm{full}}$",
}
EXPECTED_RUNS = 30
FIGURE_SIZE_IN = (5.5, 2.95)
PANEL_TITLE_SIZE_PT = 8.3
X_TICK_SIZE_PT = 5.5
Y_TICK_SIZE_PT = 7.2
KEY_TEXT_SIZE_PT = 7.2
KEY_X_TICK_SIZE_PT = 8.5
SHARED_LABEL_SIZE_PT = 8.25
DATA_LINE_WIDTH_PT = 0.60
NON_TAIL_MARKER_SIZE_PT = 2.4
TAIL_MARKER_SIZE_PT = 2.3
KEY_LINE_WIDTH_PT = 0.70
KEY_MARKER_SIZE_PT = 3.0
TAIL_COLOUR = "#0173B2"
NON_TAIL_COLOUR = "#DE8F05"
ANALYSIS_VERSION = "rq5-tail-vs-non-tail-macro-f1-v1"


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel used by this script's analysis and plot."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Match macro-F1 with explicit true-label support and zero_division=0."""
    class_scores: list[float] = []
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


def calculate_run_scores(run_dir: Path, config: dict) -> dict[str, object]:
    run_id = str(config["run_id"])
    with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        predictions = saved["predictions"]
        test_idx = saved["test_idx"]
        tail_mask = saved["tail_mask"]

    tail_idx = test_idx[tail_mask[test_idx]]
    non_tail_idx = test_idx[~tail_mask[test_idx]]
    if tail_idx.size == 0 or non_tail_idx.size == 0:
        raise RuntimeError(
            f"{run_id}: tail and non-tail test subsets must both be non-empty"
        )

    tail_labels = np.unique(labels[tail_idx])
    non_tail_labels = np.unique(labels[non_tail_idx])
    return {
        "run_id": run_id,
        "dataset": str(config["dataset"]),
        "model": str(config["model"]),
        "preparation": str(config["preparation"]),
        "split": str(config["split"]),
        "seed": int(config["seed"]),
        "n_tail_test": int(tail_idx.size),
        "n_non_tail_test": int(non_tail_idx.size),
        "tail_macro_f1": macro_f1(
            labels[tail_idx],
            predictions[tail_idx],
            tail_labels,
        ),
        "non_tail_macro_f1": macro_f1(
            labels[non_tail_idx],
            predictions[non_tail_idx],
            non_tail_labels,
        ),
    }


def validate_registered_row_ids(
    rows: list[dict[str, object]],
    dataset_mode: AnalysisDatasetMode | str,
) -> None:
    """Require the selected graph-model rows to match the frozen registry."""

    mode = AnalysisDatasetMode(dataset_mode)
    registrations = load_registered_runs(REPO_ROOT, mode, DATASET_ORDER)
    expected_ids = {
        run.run_id
        for run in registrations
        if run.model in MODEL_ORDER and run.preparation in PREPARATION_ORDER
    }
    observed_ids = [str(row["run_id"]) for row in rows]
    observed_id_set = set(observed_ids)
    missing = sorted(expected_ids - observed_id_set)
    unexpected = sorted(observed_id_set - expected_ids)
    duplicate_count = len(observed_ids) - len(observed_id_set)
    if missing or unexpected or duplicate_count:
        raise RuntimeError(
            "RQ5 raw runs: registered run ID mismatch; "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"unexpected={unexpected[:10]} ({len(unexpected)} total), "
            f"duplicates={duplicate_count}"
        )


def collect_run_scores(
    raw_root: Path,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(raw_root.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir() or run_dir.name == "historical-records":
            continue
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Unexpected directory in raw archive: {run_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("dataset") not in DATASET_ORDER:
            continue
        if config.get("model") not in MODEL_ORDER:
            continue
        if config.get("preparation") not in PREPARATION_ORDER:
            continue
        if config.get("run_id") != run_dir.name:
            raise RuntimeError(f"{run_dir}: directory and run_id differ")
        rows.append(calculate_run_scores(run_dir, config))

    expected = (
        len(DATASET_ORDER)
        * len(MODEL_ORDER)
        * len(PREPARATION_ORDER)
        * EXPECTED_RUNS
    )
    if len(rows) != expected:
        raise RuntimeError(f"Found {len(rows)} graph-model runs; expected {expected}")
    if AnalysisDatasetMode(dataset_mode) is AnalysisDatasetMode.ADDITIONAL:
        validate_registered_row_ids(rows, dataset_mode)
    return rows


def summarise(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["model"]),
            str(row["preparation"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            reference_panel: set[tuple[str, int]] | None = None
            for preparation in PREPARATION_ORDER:
                cell = grouped[(dataset, model, preparation)]
                if len(cell) != EXPECTED_RUNS:
                    raise RuntimeError(
                        f"{dataset}/{model}/{preparation}: found {len(cell)} "
                        f"runs; expected {EXPECTED_RUNS}"
                    )
                panel = {
                    (str(row["split"]), int(row["seed"]))
                    for row in cell
                }
                if reference_panel is None:
                    reference_panel = panel
                elif panel != reference_panel:
                    raise RuntimeError(
                        f"{dataset}/{model}/{preparation}: split-seed panel "
                        "does not match ERM"
                    )
                tail = np.asarray(
                    [float(row["tail_macro_f1"]) for row in cell],
                    dtype=np.float64,
                )
                non_tail = np.asarray(
                    [float(row["non_tail_macro_f1"]) for row in cell],
                    dtype=np.float64,
                )
                summaries.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "preparation": preparation,
                        "n_runs": len(cell),
                        "mean_tail_macro_f1": float(tail.mean()),
                        "mean_non_tail_macro_f1": float(non_tail.mean()),
                    }
                )
    return summaries


def configure_style() -> None:
    for candidate_text in font_manager.findSystemFonts(fontext="ttf"):
        candidate = Path(candidate_text)
        if candidate.stem.casefold() == "arial":
            font_manager.fontManager.addfont(candidate_text)
            break
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "stixsans",
            "font.size": 7.5,
            "axes.labelsize": SHARED_LABEL_SIZE_PT,
            "axes.titlesize": PANEL_TITLE_SIZE_PT,
            "xtick.labelsize": X_TICK_SIZE_PT,
            "ytick.labelsize": Y_TICK_SIZE_PT,
            "axes.linewidth": 0.62,
            "axes.edgecolor": "0.16",
            "axes.labelcolor": "0.12",
            "text.color": "0.12",
            "xtick.color": "0.16",
            "ytick.color": "0.16",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def y_axis_spec(
    values: list[float],
    *,
    upper_cap: float = 1.0,
) -> tuple[float, float, np.ndarray]:
    lower_data = min(values)
    upper_data = max(values)
    data_span = upper_data - lower_data
    padding = max(0.01, 0.04 * data_span)
    rounding = 0.01
    lower = max(
        0.0,
        float(np.floor((lower_data - padding) / rounding) * rounding),
    )
    upper = min(
        upper_cap,
        float(np.ceil((upper_data + padding) / rounding) * rounding),
    )
    if upper_cap > 1.0 and upper_data >= 0.98:
        upper = upper_cap
    plotted_span = upper - lower
    tick_step = 0.05 if plotted_span <= 0.40 else 0.10
    first_tick = np.ceil(lower / tick_step) * tick_step
    last_tick = np.floor(upper / tick_step) * tick_step
    ticks = np.arange(first_tick, last_tick + tick_step / 2.0, tick_step)
    return lower, upper, ticks


def build_figure(
    summaries: list[dict[str, object]],
) -> plt.Figure:
    configure_style()
    by_cell = {
        (
            str(row["dataset"]),
            str(row["model"]),
            str(row["preparation"]),
        ): row
        for row in summaries
    }
    expected = (
        len(DATASET_ORDER) * len(MODEL_ORDER) * len(PREPARATION_ORDER)
    )
    if len(by_cell) != expected:
        raise RuntimeError(
            f"Found {len(by_cell)} compact-figure cells; expected {expected}"
        )

    if DATASET_ORDER == ORIGINAL_DATASET_ORDER:
        panel_positions = {
            "pubmed": (0, 0),
            "ogbn-arxiv": (0, 1),
            "roman-empire": (1, 0),
            "amazon-ratings": (1, 1),
            "arxiv-year": (1, 2),
        }
        key_position = (0, 2)
        figure_size = FIGURE_SIZE_IN
        n_rows, n_columns = 2, 3
    else:
        panel_positions = {
            dataset: (0, index)
            for index, dataset in enumerate(DATASET_ORDER)
        }
        key_position = (0, len(DATASET_ORDER))
        figure_size = (FIGURE_SIZE_IN[0], 1.65)
        n_rows, n_columns = 1, len(DATASET_ORDER) + 1
    preparation_offsets = np.linspace(-0.16, 0.16, len(PREPARATION_ORDER))
    model_positions = np.arange(len(MODEL_ORDER), dtype=float) * 0.70
    figure, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    axes = np.asarray(axes, dtype=object).reshape(n_rows, n_columns)

    for dataset, (row_index, column_index) in panel_positions.items():
        axis = axes[row_index, column_index]
        plotted_values: list[float] = []
        for model_index, model in enumerate(MODEL_ORDER):
            x_positions = model_positions[model_index] + preparation_offsets
            tail_values = [
                float(
                    by_cell[
                        (dataset, model, preparation)
                    ]["mean_tail_macro_f1"]
                )
                for preparation in PREPARATION_ORDER
            ]
            non_tail_values = [
                float(
                    by_cell[
                        (dataset, model, preparation)
                    ]["mean_non_tail_macro_f1"]
                )
                for preparation in PREPARATION_ORDER
            ]
            plotted_values.extend(tail_values)
            plotted_values.extend(non_tail_values)
            axis.plot(
                x_positions,
                tail_values,
                color=TAIL_COLOUR,
                linewidth=DATA_LINE_WIDTH_PT,
                linestyle="-",
                marker="o",
                markersize=TAIL_MARKER_SIZE_PT,
                markerfacecolor=TAIL_COLOUR,
                markeredgecolor="white",
                markeredgewidth=0.25,
                solid_capstyle="round",
                zorder=3,
            )
            axis.plot(
                x_positions,
                non_tail_values,
                color=NON_TAIL_COLOUR,
                linewidth=DATA_LINE_WIDTH_PT,
                linestyle="-",
                marker="s",
                markersize=NON_TAIL_MARKER_SIZE_PT,
                markerfacecolor=NON_TAIL_COLOUR,
                markeredgecolor="white",
                markeredgewidth=0.25,
                solid_capstyle="round",
                zorder=2,
            )

        y_lower, y_upper, y_ticks = y_axis_spec(
            plotted_values,
            upper_cap=(
                1.06
                if DATASET_ORDER != ORIGINAL_DATASET_ORDER
                else 1.0
            ),
        )
        axis.set_xlim(model_positions[0] - 0.23, model_positions[-1] + 0.23)
        axis.set_ylim(y_lower, y_upper)
        axis.set_xticks(model_positions)
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
        axis.set_yticks(y_ticks)
        tick_step = (
            float(y_ticks[1] - y_ticks[0])
            if len(y_ticks) > 1
            else 0.10
        )
        axis.yaxis.set_major_formatter(
            FormatStrFormatter(
                "%.2f" if np.isclose(tick_step, 0.05) else "%.1f"
            )
        )
        axis.set_title(
            DATASET_LABELS[dataset],
            loc="left",
            pad=4,
            fontsize=PANEL_TITLE_SIZE_PT,
        )
        axis.grid(axis="y", color="0.88", linewidth=0.35)
        axis.grid(axis="x", visible=False)
        axis.tick_params(
            axis="x",
            length=0,
            pad=3,
            labelsize=X_TICK_SIZE_PT,
        )
        axis.tick_params(
            axis="y",
            direction="out",
            length=2.2,
            width=0.55,
            pad=2.2,
            labelsize=Y_TICK_SIZE_PT,
        )

    key_axis = axes[key_position]
    key_x = np.arange(len(PREPARATION_ORDER), dtype=float)
    key_tail = np.array([0.33, 0.50, 0.40, 0.48])
    key_non_tail = np.array([0.76, 0.61, 0.71, 0.64])
    key_axis.set_xlim(-0.35, 5.00)
    key_axis.set_ylim(0.18, 0.82)
    key_axis.set_xticks(key_x)
    key_axis.set_xticklabels(
        [PREPARATION_LABELS[preparation] for preparation in PREPARATION_ORDER]
    )
    key_axis.set_yticks([0.2, 0.4, 0.6, 0.8])
    key_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    key_axis.set_title(
        "Example for one model",
        loc="left",
        pad=4,
        fontsize=PANEL_TITLE_SIZE_PT,
    )
    key_axis.grid(axis="y", color="0.88", linewidth=0.35)
    key_axis.grid(axis="x", visible=False)
    for x_value, tail_value, non_tail_value in zip(
        key_x, key_tail, key_non_tail, strict=True
    ):
        key_axis.vlines(
            x_value,
            0.18,
            max(tail_value, non_tail_value) + 0.01,
            color="0.82",
            linewidth=0.35,
            zorder=0,
        )
    key_axis.plot(
        key_x,
        key_non_tail,
        color=NON_TAIL_COLOUR,
        linewidth=KEY_LINE_WIDTH_PT,
        linestyle="-",
        marker="s",
        markersize=KEY_MARKER_SIZE_PT,
        markerfacecolor=NON_TAIL_COLOUR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        solid_capstyle="round",
        zorder=2,
    )
    key_axis.plot(
        key_x,
        key_tail,
        color=TAIL_COLOUR,
        linewidth=KEY_LINE_WIDTH_PT,
        linestyle="-",
        marker="o",
        markersize=KEY_MARKER_SIZE_PT,
        markerfacecolor=TAIL_COLOUR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        solid_capstyle="round",
        zorder=3,
    )
    key_axis.text(
        3.25,
        key_non_tail[-1],
        "Non-tail",
        va="center",
        fontsize=KEY_TEXT_SIZE_PT,
    )
    key_axis.text(
        3.25,
        key_tail[-1],
        "Tail",
        va="center",
        fontsize=KEY_TEXT_SIZE_PT,
    )
    key_axis.tick_params(
        axis="x",
        length=0,
        pad=3,
        labelsize=KEY_X_TICK_SIZE_PT,
    )
    key_axis.tick_params(
        axis="y",
        direction="out",
        length=2.2,
        width=0.55,
        pad=2.2,
        labelsize=Y_TICK_SIZE_PT,
    )

    figure.text(
        0.017,
        0.52 if n_rows == 2 else 0.50,
        "Macro-F1",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=SHARED_LABEL_SIZE_PT,
    )
    figure.subplots_adjust(
        left=0.072,
        right=0.995,
        top=0.945 if n_rows == 2 else 0.90,
        bottom=0.105 if n_rows == 2 else 0.18,
        wspace=0.16,
        hspace=0.30 if n_rows == 2 else 0.0,
    )
    return figure


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
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
        required=True,
        help="canonical raw run archive containing run_data.npz",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    result_root = REPO_ROOT / "results" / (
        "main" if mode is AnalysisDatasetMode.ORIGINAL else mode.value
    )
    args.dataset_mode = mode
    args.analysis_dir = args.analysis_dir or result_root / "rq5"
    args.figure_dir = args.figure_dir or result_root / "figures" / "rq5"
    return args


def main() -> None:
    args = parse_args()
    configure_dataset_mode(args.dataset_mode)
    raw_root = args.raw_root.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw run archive not found: {raw_root}")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_rows = collect_run_scores(raw_root, args.dataset_mode)
    summaries = summarise(run_rows)

    run_csv = (
        analysis_dir / "preparation_tail_vs_non_tail_macro_f1_run_values.csv"
    )
    summary_csv = (
        analysis_dir / "preparation_tail_vs_non_tail_macro_f1.csv"
    )
    write_csv(
        run_csv,
        run_rows,
        [
            "run_id",
            "dataset",
            "model",
            "preparation",
            "split",
            "seed",
            "n_tail_test",
            "n_non_tail_test",
            "tail_macro_f1",
            "non_tail_macro_f1",
        ],
    )
    write_csv(
        summary_csv,
        summaries,
        [
            "dataset",
            "model",
            "preparation",
            "n_runs",
            "mean_tail_macro_f1",
            "mean_non_tail_macro_f1",
        ],
    )

    figure = build_figure(summaries)
    output_stem = figure_dir / "rq5_tail_vs_non_tail_macro_f1_compact"
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    metadata = {
        "Title": (
            "RQ5 tail and non-tail macro-F1 by dataset, graph model, "
            "and preparation"
        ),
        "Author": "homophily-failure-decomp",
        "Subject": "RQ5",
    }
    figure.savefig(png_path, dpi=args.dpi, metadata=metadata)
    figure.savefig(
        pdf_path,
        dpi=args.dpi,
        metadata={
            **metadata,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)

    manifest_path = analysis_dir / "tail_vs_non_tail_analysis_manifest.json"
    experiment_manifest = run_panel_paths(REPO_ROOT, args.dataset_mode)[0]
    dataset_manifests = [
        DATASET_REGISTRY[dataset].manifest_path(REPO_ROOT)
        for dataset in DATASET_ORDER
    ]
    outputs = [png_path, pdf_path, run_csv, summary_csv]
    manifest = {
        "schema_version": 1,
        "analysis": "rq5-tail-vs-non-tail-macro-f1",
        "analysis_version": ANALYSIS_VERSION,
        "dataset_mode": args.dataset_mode.value,
        "definitions": {
            "tail": "test nodes in the registered minority-pattern tail bin",
            "non_tail": "all test nodes outside the registered tail bin",
            "subset_macro_f1": (
                "macro-F1 over classes present in the subset's true labels, "
                "fixed before scoring, with zero_division=0"
            ),
            "cell_summary": (
                "arithmetic mean over 30 registered split-seed runs"
            ),
        },
        "raw_store": (
            "external run store validated against "
            f"{provenance_path(experiment_manifest)}"
        ),
        "validated_run_count": len(run_rows),
        "datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "preparations": list(PREPARATION_ORDER),
        "registered_inputs": {
            "experiment_manifest": {
                "path": provenance_path(experiment_manifest),
                "sha256": file_sha256(experiment_manifest),
            },
            "dataset_manifests": [
                {
                    "path": provenance_path(path),
                    "sha256": file_sha256(path),
                }
                for path in dataset_manifests
            ],
        },
        "outputs": [
            {
                "path": provenance_path(path),
                "sha256": file_sha256(path),
            }
            for path in outputs
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "rendering": {
            "dpi": args.dpi,
            "font_family": matplotlib.rcParams["font.family"],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Calculated {len(run_rows)} run scores and {len(summaries)} means.")
    for path in (*outputs, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
