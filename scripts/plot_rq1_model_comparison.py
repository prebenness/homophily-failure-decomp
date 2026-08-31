"""Plot test accuracy against continuous h_v for several models.

The figure shows one mean-accuracy curve per model and a shared support panel.
Accuracy is calculated separately within each run and h_v interval. For
datasets with several fixed splits, training seeds are averaged within split
before the split-level means are averaged. Split-aware bootstrap intervals are
written to CSV but omitted from the multi-model figure to keep it legible.

Example:
    python scripts/plot_rq1_model_comparison.py \
        --raw-root PATH-TO-CANONICAL-RUNS \
        --dataset amazon-ratings
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(SCRIPT_REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, NullLocator

from plot_rq1_continuous_homophily import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    REPO_ROOT,
    AnalysisData,
    analyse_runs,
    discover_runs,
    file_sha256,
    load_json,
    parse_interval,
    relative_output_paths,
    save_figure,
    validate_frozen_arrays,
)


MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
# Preserve the established model colours while changing presentation order.
MODEL_COLOUR_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "linkx",
    "acm-gcn",
)
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR-GNN",
    "acm-gcn": "ACM-GCN",
    "linkx": "LINKX",
}
DATASET_LABELS = {
    "pubmed": "PubMed",
    "ogbn-arxiv": "ogbn-arxiv",
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "arxiv-year": "arxiv-year",
}
MODEL_MARKERS = {
    "mlp": "o",
    "gcn": "s",
    "graphsage-mean": "^",
    "h2gcn": "D",
    "gpr-gnn": "P",
    "acm-gcn": "v",
    "linkx": "X",
}
DEFAULT_COMPARISON_N_BINS = 10
DISPLAY_OMISSION_REASON = "not_all_independent_accuracy_units_have_test_nodes"


def configure_style() -> list[tuple[float, float, float]]:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        font_scale=1.0,
        rc={
            "mathtext.fontset": "stixsans",
            "font.size": 10.0,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    return list(sns.color_palette("colorblind", n_colors=len(MODEL_ORDER)))


def validate_shared_support(
    data_by_model: dict[str, AnalysisData],
    models: list[str],
) -> AnalysisData:
    reference = data_by_model[models[0]]
    for model in models[1:]:
        candidate = data_by_model[model]
        if not np.array_equal(candidate.edges, reference.edges):
            raise RuntimeError(f"{model}: h_v bin edges differ across models")
        if not np.array_equal(candidate.run_support, reference.run_support):
            raise RuntimeError(f"{model}: test support differs across models")
    return reference


def complete_unit_display_mask(data: AnalysisData) -> np.ndarray:
    """Show only bins represented in every independent accuracy unit."""
    if data.accuracy_units.ndim != 2:
        raise ValueError("Accuracy units must be a two-dimensional array")
    if data.accuracy_units.shape[1] != data.centres.size:
        raise ValueError("Accuracy units do not match the h_v bins")
    return np.all(np.isfinite(data.accuracy_units), axis=0)


def positive_support_display_mask(data: AnalysisData) -> np.ndarray:
    """Show a support bar whenever its mean test-node count is positive."""
    if data.mean_run_support.shape != data.centres.shape:
        raise ValueError("Mean support does not match the h_v bins")
    return np.isfinite(data.mean_run_support) & (data.mean_run_support > 0.0)


def format_log_count(value: float, _position: float | None = None) -> str:
    """Format base-ten support ticks compactly."""
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.1g}"


def plot_support_panel(
    support_axis: plt.Axes,
    reference: AnalysisData,
    support_display_mask: np.ndarray,
) -> np.ndarray:
    """Plot positive, displayed support bars on a logarithmic scale."""
    support_display_mask = np.asarray(support_display_mask, dtype=np.bool_)
    if support_display_mask.shape != reference.centres.shape:
        raise ValueError("Support display mask does not match the h_v bins")
    visible = (
        support_display_mask
        & np.isfinite(reference.mean_run_support)
        & (reference.mean_run_support > 0.0)
    )
    if not np.any(visible):
        raise ValueError("The support panel requires at least one displayed bin")

    bar_width = 0.82 * (reference.edges[1] - reference.edges[0])
    displayed_support = reference.mean_run_support[visible]
    support_axis.bar(
        reference.centres[visible],
        displayed_support,
        width=bar_width,
        color="0.76",
        linewidth=0,
        zorder=2,
    )
    lower_decade = 10.0 ** np.floor(np.log10(displayed_support.min()))
    upper_limit = float(displayed_support.max()) * 1.12
    highest_visible_decade = int(np.floor(np.log10(upper_limit)))
    lowest_visible_decade = int(np.log10(lower_decade))
    major_ticks = 10.0 ** np.arange(
        lowest_visible_decade,
        highest_visible_decade + 1,
        dtype=np.float64,
    )
    support_axis.set_yscale("log")
    support_axis.set_ylim(lower_decade, upper_limit)
    support_axis.set_ylabel("Nodes / run", fontsize=9.5)
    support_axis.set_xlim(0.0, 1.0)
    support_axis.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    support_axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    support_axis.set_xlabel(r"Local homophily $h_v$")
    support_axis.tick_params(axis="x", direction="out", length=3)
    support_axis.set_yticks(major_ticks)
    support_axis.yaxis.set_major_formatter(FuncFormatter(format_log_count))
    support_axis.yaxis.set_minor_locator(NullLocator())
    support_axis.grid(False)
    return visible


def omitted_bin_records(
    reference: AnalysisData,
    accuracy_display_mask: np.ndarray,
) -> list[dict]:
    """Describe display-only omissions while retaining their raw estimates."""
    unit_support = np.isfinite(reference.accuracy_units).sum(axis=0)
    support_display_mask = positive_support_display_mask(reference)
    result = []
    for index in np.flatnonzero(~accuracy_display_mask):
        result.append(
            {
                "interval": (
                    f"[{reference.edges[index]:.6g}, "
                    f"{reference.edges[index + 1]:.6g}"
                    f"{']' if index == reference.centres.size - 1 else ')'}"
                ),
                "mean_test_nodes_per_run": float(
                    reference.mean_run_support[index]
                ),
                "independent_units_with_test_nodes": int(unit_support[index]),
                "independent_unit_count": int(reference.accuracy_units.shape[0]),
                "omits_accuracy_point": True,
                "omits_support_bar": bool(not support_display_mask[index]),
                "reason": DISPLAY_OMISSION_REASON,
            }
        )
    return result


def plot_confidence_band(
    axis: plt.Axes,
    data: AnalysisData,
    accuracy_display_mask: np.ndarray,
    colour: tuple[float, float, float],
    alpha: float = 0.11,
) -> None:
    """Draw a low-opacity split-bootstrap interval behind one model curve."""
    visible = (
        np.asarray(accuracy_display_mask, dtype=np.bool_)
        & np.isfinite(data.accuracy_ci_low)
        & np.isfinite(data.accuracy_ci_high)
    )
    axis.fill_between(
        data.centres,
        data.accuracy_ci_low,
        data.accuracy_ci_high,
        where=visible,
        color=colour,
        alpha=alpha,
        linewidth=0,
        interpolate=False,
        zorder=2,
    )


def plot_comparison(
    data_by_model: dict[str, AnalysisData],
    models: list[str],
    tail_interval: tuple[float, float],
    dataset: str,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    show_confidence_bands: bool = False,
) -> list[Path]:
    colours = configure_style()
    reference = validate_shared_support(data_by_model, models)
    accuracy_display_mask = complete_unit_display_mask(reference)
    support_display_mask = positive_support_display_mask(reference)
    colour_by_model = {
        model: colours[MODEL_COLOUR_ORDER.index(model)] for model in models
    }

    figure, (axis, support_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.5, 4.0),
        gridspec_kw={"height_ratios": (4.1, 1.1), "hspace": 0.07},
    )

    tail_low, tail_high = tail_interval
    boundary = tail_high if tail_low == 0.0 else tail_low
    axis.axvspan(
        tail_low,
        tail_high,
        color="0.35",
        alpha=0.065,
        linewidth=0,
        zorder=0,
    )
    for current_axis in (axis, support_axis):
        if 0.0 < boundary < 1.0:
            current_axis.axvline(
                boundary,
                color="0.50",
                linewidth=0.8,
                linestyle=(0, (3, 3)),
                zorder=1,
            )
    axis.text(
        tail_low + 0.012,
        0.975,
        "Tail",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9.5,
        color="0.35",
    )

    for model in models:
        data = data_by_model[model]
        if show_confidence_bands:
            plot_confidence_band(
                axis,
                data,
                accuracy_display_mask,
                colour_by_model[model],
            )
        displayed_accuracy = data.mean_accuracy.copy()
        displayed_accuracy[~accuracy_display_mask] = np.nan
        axis.plot(
            data.centres,
            displayed_accuracy,
            label=MODEL_LABELS[model],
            color=colour_by_model[model],
            linewidth=1.7,
            marker=MODEL_MARKERS[model],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )

    axis.set_ylim(0.0, 1.02)
    axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", which="both", labelbottom=False, length=0)
    axis.grid(axis="y", color="0.88", linewidth=0.55)
    axis.grid(axis="x", visible=False)
    handles, labels = axis.get_legend_handles_labels()
    first_row_count = (len(handles) + 1) // 2
    legend_rows = (
        (handles[:first_row_count], labels[:first_row_count], 0.925),
        (handles[first_row_count:], labels[first_row_count:], 0.875),
    )
    for row_handles, row_labels, anchor_y in legend_rows:
        if not row_handles:
            continue
        figure.legend(
            row_handles,
            row_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            ncol=len(row_handles),
            frameon=False,
            handlelength=1.45,
            columnspacing=1.05,
            handletextpad=0.4,
            borderaxespad=0.0,
            fontsize=9.5,
        )

    plot_support_panel(support_axis, reference, support_display_mask)
    sns.despine(ax=axis)
    sns.despine(ax=support_axis)

    display_dataset = DATASET_LABELS.get(dataset, dataset)
    figure.suptitle(
        display_dataset,
        x=0.5,
        y=0.978,
        ha="center",
        fontsize=13.0,
        fontweight="normal",
    )
    figure.subplots_adjust(left=0.105, right=0.985, top=0.815, bottom=0.13)

    outputs = save_figure(
        figure,
        output_stem,
        formats,
        dpi,
        f"{display_dataset}: test accuracy by local homophily",
    )
    plt.close(figure)
    return outputs


def write_binned_csv(
    path: Path,
    data_by_model: dict[str, AnalysisData],
    models: list[str],
) -> None:
    reference = validate_shared_support(data_by_model, models)
    accuracy_display_mask = complete_unit_display_mask(reference)
    support_display_mask = positive_support_display_mask(reference)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "model_label",
                "bin_left",
                "bin_right",
                "bin_centre",
                "mean_test_accuracy",
                "split_bootstrap_95_ci_low",
                "split_bootstrap_95_ci_high",
                "mean_test_nodes_per_run",
                "runs_with_test_nodes",
                "bootstrap_units_with_test_nodes",
                "accuracy_displayed_in_figure",
                "accuracy_display_omission_reason",
                "support_bar_displayed_in_figure",
            ]
        )
        for model in models:
            data = data_by_model[model]
            for index in range(data.centres.size):
                writer.writerow(
                    [
                        model,
                        MODEL_LABELS[model],
                        f"{data.edges[index]:.6f}",
                        f"{data.edges[index + 1]:.6f}",
                        f"{data.centres[index]:.6f}",
                        f"{data.mean_accuracy[index]:.10f}",
                        f"{data.accuracy_ci_low[index]:.10f}",
                        f"{data.accuracy_ci_high[index]:.10f}",
                        f"{data.mean_run_support[index]:.10f}",
                        int(np.isfinite(data.run_accuracy[:, index]).sum()),
                        int(np.isfinite(data.accuracy_units[:, index]).sum()),
                        bool(accuracy_display_mask[index]),
                        (
                            ""
                            if accuracy_display_mask[index]
                            else DISPLAY_OMISSION_REASON
                        ),
                        bool(support_display_mask[index]),
                    ]
                )


def write_analysis_manifest(
    path: Path,
    args: argparse.Namespace,
    dataset_manifest: dict,
    records_by_model: dict[str, list],
    data_by_model: dict[str, AnalysisData],
    outputs: list[Path],
) -> None:
    reference = validate_shared_support(data_by_model, args.models)
    payload = {
        "schema_version": 3,
        "analysis": "rq1-continuous-local-homophily-model-comparison",
        "dataset": args.dataset,
        "models": args.models,
        "model_labels": {model: MODEL_LABELS[model] for model in args.models},
        "preparation": args.preparation,
        "runs_per_model": {
            model: len(records_by_model[model]) for model in args.models
        },
        "run_ids": {
            model: [record.run_id for record in records_by_model[model]]
            for model in args.models
        },
        "run_data_sha256": {
            model: {
                record.run_id: record.run_data_sha256
                for record in records_by_model[model]
            }
            for model in args.models
        },
        "fixed_input_checksums_sha256": {
            "h": dataset_manifest["array_checksums_sha256"]["h"],
            "labels": dataset_manifest["array_checksums_sha256"]["labels"],
        },
        "definitions": {
            "h_v_bins": {
                "count": args.bins,
                "range": [0.0, 1.0],
                "convention": (
                    "equal-width [left, right) intervals; final interval includes 1"
                ),
            },
            "accuracy": (
                "correct predictions divided by test nodes within each run and h_v "
                "bin; when multiple fixed splits exist, seed runs are averaged "
                "within split before an unweighted mean across splits"
            ),
            "accuracy_interval": {
                "method": "two-sided percentile bootstrap of independent accuracy units",
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "quantile_method": "linear",
                "unit_count": int(reference.accuracy_units.shape[0]),
                "units": reference.accuracy_unit_definition,
                "displayed_in_figure": bool(args.show_confidence_bands),
                "stored_in": "binned CSV",
                "figure_style": (
                    "semi-transparent model-coloured envelope"
                    if args.show_confidence_bands
                    else None
                ),
            },
            "bin_display": {
                "accuracy_rule": (
                    "display a bin only when every independent accuracy unit "
                    "contains at least one test node"
                ),
                "support_bar_rule": (
                    "display a support bar whenever the mean test-node count "
                    "per run is positive; omit zero-support bars"
                ),
                "raw_values_retained_in": "binned CSV",
                "omitted_bins": omitted_bin_records(
                    reference, complete_unit_display_mask(reference)
                ),
            },
            "support_bars": {
                "quantity": "mean test-node count per run and h_v bin",
                "scale": "logarithmic",
                "base": 10,
                "minor_ticks": False,
            },
            "tail_interval": dataset_manifest["local_homophily"]["tail"]["interval"],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_scripts": {
            "comparison": {
                "path": "scripts/plot_rq1_model_comparison.py",
                "sha256": file_sha256(Path(__file__)),
            },
            "shared_calculations": {
                "path": "scripts/plot_rq1_continuous_homophily.py",
                "sha256": file_sha256(
                    REPO_ROOT / "scripts" / "plot_rq1_continuous_homophily.py"
                ),
            },
        },
        "outputs": relative_output_paths(outputs),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs",
        help="directory containing one subdirectory per completed run",
    )
    parser.add_argument("--dataset", default="amazon-ratings")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    parser.add_argument(
        "--preparation",
        default="erm",
        choices=("erm", "last", "head", "full"),
    )
    parser.add_argument(
        "--array-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "manifests",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "manifests" / "datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "figures" / "rq1",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_COMPARISON_N_BINS,
        help=(
            "number of equal-width h_v bins "
            f"(default: {DEFAULT_COMPARISON_N_BINS})"
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-confidence-bands",
        action="store_true",
        help=(
            "draw semi-transparent split-bootstrap 95% confidence envelopes "
            "behind the model curves"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("Each model may be listed only once")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")
    args.raw_root = args.raw_root.resolve()
    args.array_root = args.array_root.resolve()
    args.manifest_root = args.manifest_root.resolve()
    args.output_dir = args.output_dir.resolve()

    manifest_path = args.manifest_root / f"{args.dataset}.json"
    frozen_path = args.array_root / f"{args.dataset}.npz"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    if not frozen_path.is_file():
        raise FileNotFoundError(
            f"Frozen arrays not found: {frozen_path}. Run "
            f"scripts/compute_dataset_stats.py --datasets {args.dataset} first."
        )
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"Raw run directory not found: {args.raw_root}")

    dataset_manifest = load_json(manifest_path)
    if dataset_manifest.get("dataset") != args.dataset:
        raise RuntimeError(f"Manifest dataset mismatch in {manifest_path}")
    with np.load(frozen_path) as frozen_arrays:
        h, frozen_labels = validate_frozen_arrays(frozen_arrays, dataset_manifest)

    records_by_model = {}
    data_by_model = {}
    for model in args.models:
        records = discover_runs(
            args.raw_root,
            args.dataset,
            model,
            args.preparation,
            dataset_manifest,
        )
        records_by_model[model] = records
        data_by_model[model] = analyse_runs(
            records,
            h,
            frozen_labels,
            dataset_manifest,
            args.bins,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
    validate_shared_support(data_by_model, args.models)

    tail_interval = parse_interval(
        dataset_manifest["local_homophily"]["tail"]["interval"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = (
        "all-models"
        if args.models == list(MODEL_ORDER)
        else "-".join(args.models)
    )
    stem = f"rq1_{args.dataset}_{args.preparation}_{model_slug}"
    figure_stem = args.output_dir / f"{stem}_accuracy_by_local_homophily"
    binned_csv = args.output_dir / f"{stem}_binned_accuracy.csv"
    analysis_manifest = args.output_dir / f"{stem}_analysis_manifest.json"

    outputs = plot_comparison(
        data_by_model,
        args.models,
        tail_interval,
        args.dataset,
        figure_stem,
        tuple(args.formats),
        args.dpi,
        args.show_confidence_bands,
    )
    write_binned_csv(binned_csv, data_by_model, args.models)
    outputs.extend((binned_csv, analysis_manifest))
    write_analysis_manifest(
        analysis_manifest,
        args,
        dataset_manifest,
        records_by_model,
        data_by_model,
        outputs,
    )

    total_runs = sum(len(records) for records in records_by_model.values())
    print(f"Analysed {total_runs} runs across {len(args.models)} models.")
    for output in outputs:
        print(relative_output_paths([output])[0])


if __name__ == "__main__":
    main()
