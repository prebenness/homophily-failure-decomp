"""Plot the five RQ1 continuous-homophily results as one composite figure.

The top row contains the two homophilic datasets and a shared legend. The
bottom row contains the three heterophilic datasets. Each dataset panel shows
the selected test metric, split-bootstrap 95% confidence bands, and mean
test-node support per run on a logarithmic scale.

Example:
    python scripts/plot_rq1_composite.py \
        --raw-root PATH-TO-CANONICAL-RUNS \
        --bins 12
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfd.analysis_datasets import (  # noqa: E402
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
    load_registered_runs,
)

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
    validate_frozen_arrays,
)
from plot_rq1_model_comparison import (
    DATASET_LABELS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_COLOUR_ORDER,
    MODEL_ORDER,
    configure_style,
    plot_support_panel,
    positive_support_display_mask,
    validate_shared_support,
)


DEFAULT_COMPOSITE_N_BINS = 12
METRIC_CHOICES = ("accuracy", "macro-f1")
DEFAULT_COMPOSITE_METRIC = "accuracy"
COMPOSITE_CONFIDENCE_ALPHA = 0.20
COMPOSITE_SUPPORT_FLOOR = 1.0
PAPER_TEXT_WIDTH_IN = 5.5
COMPOSITE_HEIGHT_IN = 3.35
ACCURACY_PANEL_HEIGHT_RATIO = 2.35
SUPPORT_PANEL_HEIGHT_RATIO = 1.00
MINIMUM_TEXT_SIZE_PT = 8.0
DATASET_TITLE_SIZE_PT = 9.3
AXIS_LABEL_SIZE_PT = 8.5
TICK_LABEL_SIZE_PT = 8.0
LEGEND_TEXT_SIZE_PT = 8.0
LEGEND_TITLE_SIZE_PT = 8.5
SHARED_X_LABEL_SIZE_PT = 9.0
MODEL_LINE_WIDTH_PT = 1.0
PLOT_MARKER_SIZE_PT = 2.4
LEGEND_LINE_WIDTH_PT = 1.1
LEGEND_MARKER_SIZE_PT = 3.4
PRIMARY_LEGEND_ANCHOR_Y = 1.10
SECONDARY_LEGEND_TITLE = "Other"
SECONDARY_LEGEND_ANCHOR_Y = 0.05
DATASET_LAYOUT = {
    "pubmed": (0, 0),
    "ogbn-arxiv": (0, 1),
    "roman-empire": (1, 0),
    "amazon-ratings": (1, 1),
    "arxiv-year": (1, 2),
}
DATASET_ORDER = tuple(DATASET_LAYOUT)
ADDITIONAL_DATASET_LAYOUT = {
    "actor": (0, 0),
    "blogcatalog": (0, 1),
}
ADDITIONAL_DATASET_ORDER = tuple(ADDITIONAL_DATASET_LAYOUT)
ORIGINAL_LEGEND_CELL = (0, 2)
ADDITIONAL_LEGEND_CELL = (0, 2)
ADDITIONAL_COMPOSITE_HEIGHT_IN = 1.95


def panel_spec(
    mode: AnalysisDatasetMode | str,
) -> tuple[tuple[str, ...], dict[str, tuple[int, int]], tuple[int, int], int, float]:
    selected_mode = AnalysisDatasetMode(mode)
    if selected_mode is AnalysisDatasetMode.ORIGINAL:
        return (
            DATASET_ORDER,
            DATASET_LAYOUT,
            ORIGINAL_LEGEND_CELL,
            2,
            COMPOSITE_HEIGHT_IN,
        )
    if selected_mode is AnalysisDatasetMode.ADDITIONAL:
        return (
            ADDITIONAL_DATASET_ORDER,
            ADDITIONAL_DATASET_LAYOUT,
            ADDITIONAL_LEGEND_CELL,
            1,
            ADDITIONAL_COMPOSITE_HEIGHT_IN,
        )
    raise ValueError("The RQ1 appendix figure supports original or additional mode")


def dataset_label(dataset: str) -> str:
    registered = DATASET_REGISTRY.get(dataset)
    if registered is not None:
        return registered.display_label
    return DATASET_LABELS.get(dataset, dataset)


def metric_slug(metric: str) -> str:
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Unsupported metric: {metric}")
    return metric.replace("-", "_")


def metric_label(metric: str) -> str:
    return {"accuracy": "Accuracy", "macro-f1": "Macro-F1"}[metric]


def metric_metadata_label(metric: str) -> str:
    return {"accuracy": "accuracy", "macro-f1": "macro-F1"}[metric]


def score_axis_upper_limit(
    mode: AnalysisDatasetMode | str,
) -> float:
    """Leave marker and confidence-band headroom at a perfect score."""

    selected_mode = AnalysisDatasetMode(mode)
    return 1.06 if selected_mode is AnalysisDatasetMode.ADDITIONAL else 1.0


def validate_registered_record_ids(
    records: list,
    dataset: str,
    model: str,
    preparation: str,
    dataset_mode: AnalysisDatasetMode | str,
) -> None:
    """Require a selected raw-run cell to match its frozen registry exactly."""

    mode = AnalysisDatasetMode(dataset_mode)
    registrations = (
        run
        for run in load_registered_runs(REPO_ROOT, mode, (dataset,))
        if run.model == model and run.preparation == preparation
    )
    expected_ids = {run.run_id for run in registrations}
    observed_ids = [record.run_id for record in records]
    observed_id_set = set(observed_ids)
    missing = sorted(expected_ids - observed_id_set)
    unexpected = sorted(observed_id_set - expected_ids)
    duplicate_count = len(observed_ids) - len(observed_id_set)
    if missing or unexpected or duplicate_count:
        raise RuntimeError(
            f"{dataset}/{model}/{preparation}: registered run ID mismatch; "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"unexpected={unexpected[:10]} ({len(unexpected)} total), "
            f"duplicates={duplicate_count}"
        )


def composite_output_stem(
    preparation: str,
    n_bins: int,
    metric: str,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> str:
    mode = AnalysisDatasetMode(dataset_mode)
    panel_name = (
        "five-datasets"
        if mode is AnalysisDatasetMode.ORIGINAL
        else "additional-datasets"
    )
    return (
        f"rq1_{preparation}_{panel_name}_all-models_"
        f"{n_bins}-bin_{metric_slug(metric)}_by_local_homophily_with-ci"
    )


def metric_units(data: AnalysisData, metric: str) -> np.ndarray:
    return getattr(data, f"{metric_slug(metric)}_units")


def metric_mean(data: AnalysisData, metric: str) -> np.ndarray:
    return getattr(data, f"mean_{metric_slug(metric)}")


def metric_interval(
    data: AnalysisData,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    slug = metric_slug(metric)
    return getattr(data, f"{slug}_ci_low"), getattr(data, f"{slug}_ci_high")


def complete_metric_unit_display_mask(
    data: AnalysisData,
    metric: str,
) -> np.ndarray:
    """Show bins represented in every independent unit for the metric."""
    units = metric_units(data, metric)
    if units.ndim != 2 or units.shape[1] != data.centres.size:
        raise ValueError(f"{metric_label(metric)} units do not match the h_v bins")
    return np.all(np.isfinite(units), axis=0)


def plot_metric_confidence_band(
    axis: plt.Axes,
    data: AnalysisData,
    display_mask: np.ndarray,
    colour: tuple[float, float, float],
    metric: str,
) -> None:
    low, high = metric_interval(data, metric)
    visible = (
        np.asarray(display_mask, dtype=np.bool_)
        & np.isfinite(low)
        & np.isfinite(high)
    )
    axis.fill_between(
        data.centres,
        low,
        high,
        where=visible,
        color=colour,
        alpha=COMPOSITE_CONFIDENCE_ALPHA,
        linewidth=0,
        interpolate=False,
        zorder=2,
    )


def composite_support_display_mask(reference: AnalysisData) -> np.ndarray:
    """Show support bars only from one mean test node per run upwards."""
    return positive_support_display_mask(reference) & (
        reference.mean_run_support >= COMPOSITE_SUPPORT_FLOOR
    )


def composite_omitted_bin_records(
    reference: AnalysisData,
    metric: str,
) -> list[dict]:
    """Describe metric omissions under the composite support-floor rule."""
    units = metric_units(reference, metric)
    display_mask = complete_metric_unit_display_mask(reference, metric)
    unit_support = np.isfinite(units).sum(axis=0)
    point_key = f"omits_{metric_slug(metric)}_point"
    records = []
    for index in np.flatnonzero(~display_mask):
        omission_reason = (
            "not_all_independent_accuracy_units_have_test_nodes"
            if metric == "accuracy"
            else "not_all_independent_macro_f1_units_have_test_nodes"
        )
        record = {
            "interval": (
                f"[{reference.edges[index]:.6g}, "
                f"{reference.edges[index + 1]:.6g}"
                f"{']' if index == reference.centres.size - 1 else ')'}"
            ),
            "mean_test_nodes_per_run": float(reference.mean_run_support[index]),
            "independent_units_with_test_nodes": int(unit_support[index]),
            "independent_unit_count": int(units.shape[0]),
            point_key: True,
            "reason": omission_reason,
        }
        omits_support = (
            record["mean_test_nodes_per_run"] < COMPOSITE_SUPPORT_FLOOR
        )
        record["omits_support_bar"] = omits_support
        if omits_support:
            record["support_omission_reason"] = (
                "mean_test_nodes_per_run_below_composite_support_floor"
            )
        records.append(record)
    return records


def add_tail_region(
    accuracy_axis: plt.Axes,
    support_axis: plt.Axes,
    tail_interval: tuple[float, float],
) -> None:
    tail_low, tail_high = tail_interval
    boundary = tail_high if tail_low == 0.0 else tail_low
    for axis in (accuracy_axis, support_axis):
        axis.axvspan(
            tail_low,
            tail_high,
            color="0.35",
            alpha=0.065,
            linewidth=0,
            zorder=0,
        )
        if 0.0 < boundary < 1.0:
            axis.axvline(
                boundary,
                color="0.50",
                linewidth=0.8,
                linestyle=(0, (3, 3)),
                zorder=1,
            )


def add_dataset_panel(
    figure: plt.Figure,
    slot,
    dataset: str,
    data_by_model: dict[str, AnalysisData],
    models: list[str],
    tail_interval: tuple[float, float],
    colour_by_model: dict[str, tuple[float, float, float]],
    show_y_labels: bool,
    metric: str,
    score_y_upper: float = 1.0,
) -> tuple[plt.Axes, plt.Axes]:
    inner = slot.subgridspec(
        2,
        1,
        height_ratios=(
            ACCURACY_PANEL_HEIGHT_RATIO,
            SUPPORT_PANEL_HEIGHT_RATIO,
        ),
        hspace=0.035,
    )
    accuracy_axis = figure.add_subplot(inner[0])
    support_axis = figure.add_subplot(inner[1], sharex=accuracy_axis)

    reference = validate_shared_support(data_by_model, models)
    metric_display_mask = complete_metric_unit_display_mask(reference, metric)
    support_display_mask = composite_support_display_mask(reference)
    add_tail_region(accuracy_axis, support_axis, tail_interval)

    for model in models:
        data = data_by_model[model]
        plot_metric_confidence_band(
            accuracy_axis,
            data,
            metric_display_mask,
            colour_by_model[model],
            metric,
        )
        displayed_metric = metric_mean(data, metric).copy()
        displayed_metric[~metric_display_mask] = np.nan
        accuracy_axis.plot(
            data.centres,
            displayed_metric,
            color=colour_by_model[model],
            linewidth=MODEL_LINE_WIDTH_PT,
            marker=MODEL_MARKERS[model],
            markersize=PLOT_MARKER_SIZE_PT,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=3,
        )

    accuracy_axis.set_title(
        dataset_label(dataset),
        fontsize=DATASET_TITLE_SIZE_PT,
        fontweight="normal",
        pad=3,
    )
    accuracy_axis.set_ylim(0.0, score_y_upper)
    accuracy_axis.set_yticks([0.0, 0.5, 1.0])
    accuracy_axis.set_yticklabels(["0", "0.5", "1"])
    accuracy_axis.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
        length=0,
    )
    accuracy_axis.tick_params(
        axis="y",
        labelsize=TICK_LABEL_SIZE_PT,
        length=2.5,
        width=0.7,
    )
    accuracy_axis.grid(axis="y", color="0.88", linewidth=0.45)
    accuracy_axis.grid(axis="x", visible=False)
    if show_y_labels:
        accuracy_axis.set_ylabel(
            metric_label(metric),
            fontsize=AXIS_LABEL_SIZE_PT,
            labelpad=3,
        )
    else:
        accuracy_axis.set_ylabel("")
        accuracy_axis.tick_params(axis="y", labelleft=False)

    plot_support_panel(support_axis, reference, support_display_mask)
    current_support_low, current_support_high = support_axis.get_ylim()
    support_axis.set_ylim(
        max(COMPOSITE_SUPPORT_FLOOR, current_support_low),
        current_support_high,
    )
    support_axis.set_xlabel("")
    boundary = tail_interval[1] if tail_interval[0] == 0.0 else tail_interval[0]
    support_axis.set_xticks([0.0, boundary, 1.0])
    support_axis.set_xticklabels(
        ["0", f"{boundary:g}", "1"],
        fontsize=TICK_LABEL_SIZE_PT,
    )
    visible_log_ticks = [
        tick
        for tick in support_axis.get_yticks()
        if support_axis.get_ylim()[0] <= tick <= support_axis.get_ylim()[1]
    ]
    if len(visible_log_ticks) > 3:
        middle_index = len(visible_log_ticks) // 2
        visible_log_ticks = [
            visible_log_ticks[0],
            visible_log_ticks[middle_index],
            visible_log_ticks[-1],
        ]
        support_axis.set_yticks(visible_log_ticks)
    support_axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE_PT,
        length=2.5,
        width=0.7,
        pad=2,
    )
    if show_y_labels:
        support_axis.set_ylabel(
            "Nodes / run",
            fontsize=AXIS_LABEL_SIZE_PT,
            labelpad=3,
        )
    else:
        support_axis.set_ylabel("")
    sns.despine(ax=accuracy_axis)
    sns.despine(ax=support_axis)
    return accuracy_axis, support_axis


def add_shared_legend(
    legend_axis: plt.Axes,
    models: list[str],
    colour_by_model: dict[str, tuple[float, float, float]],
) -> None:
    legend_axis.axis("off")
    model_handles = [
        Line2D(
            [0],
            [0],
            color=colour_by_model[model],
            linewidth=LEGEND_LINE_WIDTH_PT,
            marker=MODEL_MARKERS[model],
            markersize=LEGEND_MARKER_SIZE_PT,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]
    model_legend = legend_axis.legend(
        handles=model_handles,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(0.0, PRIMARY_LEGEND_ANCHOR_Y),
        ncol=2,
        frameon=False,
        handlelength=1.20,
        handletextpad=0.40,
        columnspacing=0.70,
        labelspacing=0.24,
        borderaxespad=0.0,
        fontsize=LEGEND_TEXT_SIZE_PT,
        title_fontsize=LEGEND_TITLE_SIZE_PT,
    )
    legend_axis.add_artist(model_legend)

    encoding_handles = [
        Patch(
            facecolor=colour_by_model[models[0]],
            alpha=COMPOSITE_CONFIDENCE_ALPHA,
            edgecolor="none",
            label="95% confidence band",
        ),
        Patch(
            facecolor=(0.35, 0.35, 0.35, 0.10),
            edgecolor=(0.50, 0.50, 0.50, 1.0),
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            label="Tail region",
        ),
        Patch(
            facecolor="0.76",
            edgecolor="none",
            label="Nodes / run (log)",
        ),
    ]
    legend_axis.legend(
        handles=encoding_handles,
        title=SECONDARY_LEGEND_TITLE,
        loc="lower left",
        bbox_to_anchor=(0.0, SECONDARY_LEGEND_ANCHOR_Y),
        ncol=1,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.55,
        labelspacing=0.30,
        borderaxespad=0.0,
        fontsize=LEGEND_TEXT_SIZE_PT,
        title_fontsize=LEGEND_TITLE_SIZE_PT,
    )


def save_composite_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    metric: str,
    dataset_count_label: str = "five",
) -> list[Path]:
    """Save the exact paper-width canvas without subsequent font scaling."""
    metadata = {
        "Title": (
            f"RQ1: test {metric_metadata_label(metric)} by local homophily "
            f"across {dataset_count_label} "
            "datasets"
        ),
        "Author": "homophily-failure-decomp",
        "Subject": "RQ1 continuous local-homophily analysis",
    }
    pdf_metadata = {
        **metadata,
        "CreationDate": None,
        "ModDate": None,
    }
    outputs = []
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            path,
            dpi=dpi,
            metadata=(
                pdf_metadata if output_format.lower() == "pdf" else metadata
            ),
        )
        outputs.append(path)
    return outputs


def plot_composite(
    data_by_dataset: dict[str, dict[str, AnalysisData]],
    tail_by_dataset: dict[str, tuple[float, float]],
    models: list[str],
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    metric: str = "accuracy",
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> list[Path]:
    (
        selected_datasets,
        dataset_layout,
        legend_cell,
        grid_rows,
        canvas_height,
    ) = panel_spec(dataset_mode)
    score_y_upper = score_axis_upper_limit(dataset_mode)
    colours = configure_style()
    colour_by_model = {
        model: colours[MODEL_COLOUR_ORDER.index(model)] for model in models
    }
    figure = plt.figure(figsize=(PAPER_TEXT_WIDTH_IN, canvas_height))
    outer = figure.add_gridspec(
        grid_rows,
        3,
        width_ratios=(1.0, 1.0, 1.15),
        left=0.085,
        right=0.985,
        top=0.950 if grid_rows == 2 else 0.925,
        bottom=0.095 if grid_rows == 2 else 0.170,
        wspace=0.22,
        hspace=0.32 if grid_rows == 2 else 0.0,
    )

    for dataset in selected_datasets:
        row, column = dataset_layout[dataset]
        add_dataset_panel(
            figure,
            outer[row, column],
            dataset,
            data_by_dataset[dataset],
            models,
            tail_by_dataset[dataset],
            colour_by_model,
            show_y_labels=(column == 0),
            metric=metric,
            score_y_upper=score_y_upper,
        )

    legend_axis = figure.add_subplot(outer[legend_cell])
    add_shared_legend(legend_axis, models, colour_by_model)
    figure.supxlabel(
        r"Local homophily $h_v$",
        x=0.525,
        y=0.018 if grid_rows == 2 else 0.035,
        fontsize=SHARED_X_LABEL_SIZE_PT,
    )

    outputs = save_composite_figure(
        figure,
        output_stem,
        formats,
        dpi,
        metric,
        "five" if len(selected_datasets) == 5 else "two",
    )
    plt.close(figure)
    return outputs


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    manifests: dict[str, dict],
    records: dict[str, dict[str, list]],
    data_by_dataset: dict[str, dict[str, AnalysisData]],
    outputs: list[Path],
) -> None:
    (
        selected_datasets,
        dataset_layout,
        legend_cell,
        _grid_rows,
        canvas_height,
    ) = panel_spec(args.dataset_mode)
    metric_key = metric_slug(args.metric)
    if args.metric == "accuracy":
        metric_definition = (
            "correct predictions divided by test nodes within each run and h_v "
            "bin"
        )
    else:
        metric_definition = (
            "macro-F1 within each run and h_v bin, averaged over the classes "
            "present in that bin's true labels, fixed before predictions are "
            "evaluated, with zero_division=0"
        )
    metric_definition += (
        "; when multiple fixed splits exist, seed runs are averaged within "
        "split before an unweighted mean across splits"
    )
    payload = {
        "schema_version": 1,
        "analysis": "rq1-continuous-local-homophily-composite",
        "dataset_mode": args.dataset_mode,
        "datasets": list(selected_datasets),
        "layout": {
            dataset: {"row": row, "column": column}
            for dataset, (row, column) in dataset_layout.items()
        },
        "reserved_legend_cell": {
            "row": legend_cell[0],
            "column": legend_cell[1],
        },
        "models": args.models,
        "preparation": args.preparation,
        "metric": args.metric,
        "definitions": {
            "h_v_bins": {
                "count": args.bins,
                "range": [0.0, 1.0],
                "convention": (
                    "equal-width [left, right) intervals; final interval includes 1"
                ),
            },
            metric_key: metric_definition,
            "confidence_bands": {
                "method": (
                    "two-sided percentile bootstrap of independent "
                    f"{metric_label(args.metric)} units"
                ),
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "quantile_method": "linear",
                "units_by_dataset": {
                    dataset: getattr(
                        data_by_dataset[dataset][args.models[0]],
                        f"{metric_key}_unit_definition",
                    )
                    for dataset in selected_datasets
                },
                "style": "semi-transparent model-coloured envelope",
                "alpha": COMPOSITE_CONFIDENCE_ALPHA,
            },
            "publication_geometry": {
                "target_paper_width_inches": PAPER_TEXT_WIDTH_IN,
                "canvas_height_inches": canvas_height,
                f"{metric_key}_panel_height_ratio": (
                    ACCURACY_PANEL_HEIGHT_RATIO
                ),
                "support_panel_height_ratio": SUPPORT_PANEL_HEIGHT_RATIO,
                "minimum_text_size_points": MINIMUM_TEXT_SIZE_PT,
                "designed_for_unscaled_inclusion": True,
            },
            "visual_design": {
                f"{metric_key}_axis_range": [
                    0.0,
                    score_axis_upper_limit(args.dataset_mode),
                ],
                "plot_marker_size_points": PLOT_MARKER_SIZE_PT,
                "legend_marker_size_points": LEGEND_MARKER_SIZE_PT,
                "model_line_width_points": MODEL_LINE_WIDTH_PT,
            },
            f"{metric_key}_display_rule": (
                "display a bin only when every independent metric unit "
                "contains at least one test node"
            ),
            "support_bars": {
                "quantity": "mean test-node count per run and h_v bin",
                "scale": "logarithmic",
                "minimum_displayed_mean_test_nodes_per_run": (
                    COMPOSITE_SUPPORT_FLOOR
                ),
                "display_rule": (
                    "display mean counts at or above one test node per run; "
                    "omit bars below the displayed logarithmic range"
                ),
            },
            "tail_intervals": {
                dataset: manifests[dataset]["local_homophily"]["tail"]["interval"]
                for dataset in selected_datasets
            },
            f"omitted_{metric_key}_bins": {
                dataset: composite_omitted_bin_records(
                    validate_shared_support(
                        data_by_dataset[dataset], args.models
                    ),
                    args.metric,
                )
                for dataset in selected_datasets
            },
        },
        "runs_per_dataset_model": {
            dataset: {
                model: len(records[dataset][model]) for model in args.models
            }
            for dataset in selected_datasets
        },
        "run_ids": {
            dataset: {
                model: [
                    record.run_id for record in records[dataset][model]
                ]
                for model in args.models
            }
            for dataset in selected_datasets
        },
        "fixed_input_checksums_sha256": {
            dataset: {
                "h": manifests[dataset]["array_checksums_sha256"]["h"],
                "labels": manifests[dataset]["array_checksums_sha256"]["labels"],
            }
            for dataset in selected_datasets
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_scripts": {
            "composite": {
                "path": "scripts/plot_rq1_composite.py",
                "sha256": file_sha256(Path(__file__)),
            },
            "model_comparison": {
                "path": "scripts/plot_rq1_model_comparison.py",
                "sha256": file_sha256(
                    REPO_ROOT / "scripts" / "plot_rq1_model_comparison.py"
                ),
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
        default=REPO_ROOT / "artifacts" / "runs",
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="required raw-run root in additional mode",
    )
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
        "--metric",
        choices=METRIC_CHOICES,
        default=DEFAULT_COMPOSITE_METRIC,
        help="performance metric plotted in each local-homophily bin",
    )
    parser.add_argument(
        "--array-root",
        type=Path,
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_COMPOSITE_N_BINS)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    selected_datasets, *_ = panel_spec(mode)
    if len(set(args.models)) != len(args.models):
        raise ValueError("Each model may be listed only once")
    if args.bins < 2:
        raise ValueError("At least two h_v bins are required")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")

    if mode is AnalysisDatasetMode.ADDITIONAL:
        if args.additional_raw_root is None:
            raise ValueError(
                "--additional-raw-root is required in additional mode"
            )
        args.raw_root = args.additional_raw_root
    args.array_root = args.array_root or (
        REPO_ROOT
        / (
            "artifacts/manifests"
            if mode is AnalysisDatasetMode.ORIGINAL
            else "artifacts/additional-dataset-manifests"
        )
    )
    args.manifest_root = args.manifest_root or (
        REPO_ROOT
        / (
            "manifests/datasets"
            if mode is AnalysisDatasetMode.ORIGINAL
            else "manifests/additional-datasets"
        )
    )
    args.output_dir = args.output_dir or (
        REPO_ROOT
        / (
            "results/main/figures/rq1"
            if mode is AnalysisDatasetMode.ORIGINAL
            else "results/additional/figures/rq1"
        )
    )

    args.raw_root = args.raw_root.resolve()
    args.array_root = args.array_root.resolve()
    args.manifest_root = args.manifest_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"Raw run directory not found: {args.raw_root}")

    manifests = {}
    records = {}
    data_by_dataset = {}
    tail_by_dataset = {}
    for dataset in selected_datasets:
        manifest_path = args.manifest_root / f"{dataset}.json"
        frozen_path = args.array_root / f"{dataset}.npz"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Frozen arrays not found: {frozen_path}")

        manifest = load_json(manifest_path)
        if manifest.get("dataset") != dataset:
            raise RuntimeError(f"Manifest dataset mismatch in {manifest_path}")
        with np.load(frozen_path) as frozen_arrays:
            h, labels = validate_frozen_arrays(frozen_arrays, manifest)

        manifests[dataset] = manifest
        records[dataset] = {}
        data_by_dataset[dataset] = {}
        for model in args.models:
            model_records = discover_runs(
                args.raw_root,
                dataset,
                model,
                args.preparation,
                manifest,
            )
            if mode is AnalysisDatasetMode.ADDITIONAL:
                validate_registered_record_ids(
                    model_records,
                    dataset,
                    model,
                    args.preparation,
                    mode,
                )
            records[dataset][model] = model_records
            data_by_dataset[dataset][model] = analyse_runs(
                model_records,
                h,
                labels,
                manifest,
                args.bins,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
        validate_shared_support(data_by_dataset[dataset], args.models)
        tail_by_dataset[dataset] = parse_interval(
            manifest["local_homophily"]["tail"]["interval"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = composite_output_stem(
        args.preparation,
        args.bins,
        args.metric,
        mode,
    )
    figure_stem = args.output_dir / stem
    manifest_path = args.output_dir / f"{stem}_analysis_manifest.json"
    outputs = plot_composite(
        data_by_dataset,
        tail_by_dataset,
        args.models,
        figure_stem,
        tuple(args.formats),
        args.dpi,
        args.metric,
        mode,
    )
    outputs.append(manifest_path)
    write_manifest(
        manifest_path,
        args,
        manifests,
        records,
        data_by_dataset,
        outputs,
    )

    total_runs = sum(
        len(records[dataset][model])
        for dataset in selected_datasets
        for model in args.models
    )
    print(
        f"Analysed {total_runs} runs across {len(selected_datasets)} datasets "
        f"and {len(args.models)} models."
    )
    for output in outputs:
        print(relative_output_paths([output])[0])


if __name__ == "__main__":
    main()
