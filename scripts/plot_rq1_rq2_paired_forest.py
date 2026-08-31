"""Plot RQ1 and class-adjusted RQ2 macro-F1 tail gaps together.

The figure follows the five-panel visual style of the paper's continuous RQ1
figure. Within each dataset panel, every model shows:

* the unadjusted ERM tail gap from RQ1;
* the class-composition-adjusted ERM tail gap from the RQ2 analysis;
* the corresponding 95% paired-bootstrap confidence intervals.

The two point estimates share one vertical tail-gap scale and are joined by a
faint line. Filled markers denote intervals that exclude zero; unfilled markers
denote intervals that contain zero. The connector is descriptive only and is
not a confidence interval for the change between estimators.

Example:
    python scripts/plot_rq1_rq2_paired_forest.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from hfd.analysis_datasets import (  # noqa: E402
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
)


DATASET_ORDER = (
    "pubmed",
    "ogbn-arxiv",
    "roman-empire",
    "amazon-ratings",
    "arxiv-year",
)
DATASET_LAYOUT = {
    "pubmed": (0, 0),
    "ogbn-arxiv": (0, 1),
    "roman-empire": (1, 0),
    "amazon-ratings": (1, 1),
    "arxiv-year": (1, 2),
}
LEGEND_CELL = (0, 2)
ADDITIONAL_DATASET_ORDER = dataset_order(AnalysisDatasetMode.ADDITIONAL)
ADDITIONAL_DATASET_LAYOUT = {
    "actor": (0, 0),
    "blogcatalog": (0, 1),
}
ADDITIONAL_LEGEND_CELL = (0, 2)
DATASET_LABELS = {
    "pubmed": "PubMed",
    "ogbn-arxiv": "ogbn-arxiv",
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "arxiv-year": "arxiv-year",
    "actor": "Actor",
    "blogcatalog": "BlogCatalog",
}
TAIL_LABELS = {
    "pubmed": r"low-$h_v$ tail",
    "ogbn-arxiv": r"low-$h_v$ tail",
    "roman-empire": r"high-$h_v$ tail",
    "amazon-ratings": r"high-$h_v$ tail",
    "arxiv-year": r"high-$h_v$ tail",
    "actor": r"high-$h_v$ tail",
    "blogcatalog": r"high-$h_v$ tail",
}
EXPECTED_TAIL_SIDES = {
    "pubmed": "low-h_v",
    "ogbn-arxiv": "low-h_v",
    "roman-empire": "high-h_v",
    "amazon-ratings": "high-h_v",
    "arxiv-year": "high-h_v",
    "actor": "high-h_v",
    "blogcatalog": "high-h_v",
}
MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
SOURCE_MODEL_ORDER = (
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
MODEL_SHORT_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "SAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR",
    "acm-gcn": "ACM",
    "linkx": "LINKX",
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
CONDITION_ORDER = ("unadjusted", "class_adjusted")
CONDITION_LABELS = {
    "unadjusted": "Unadjusted",
    "class_adjusted": "Class-adjusted",
}
CONDITION_POSITIONS = {
    "unadjusted": 0.0,
    "class_adjusted": 1.0,
}

EXPECTED_CELLS = len(DATASET_ORDER) * len(MODEL_ORDER)
FIGURE_WIDTH_IN = 5.5
FIGURE_HEIGHT_IN = 3.15
ADDITIONAL_FIGURE_HEIGHT_IN = 1.80
MINIMUM_TEXT_SIZE_PT = 8.0
MODEL_TEXT_SIZE_PT = 8.0
DATASET_TEXT_SIZE_PT = 9.3
LEGEND_TEXT_SIZE_PT = 7.5
LEGEND_TITLE_SIZE_PT = 8.5
AXIS_LABEL_SIZE_PT = 8.5
TICK_TEXT_SIZE_PT = 8.0
MODEL_JITTER_SPAN = 0.20
MARKER_SIZE_PT = 3.5
CI_LINE_WIDTH_PT = 0.75
CONNECTOR_LINE_WIDTH_PT = 0.55
CONNECTOR_ALPHA = 0.25
ZERO_LINE_WIDTH_PT = 1.25
ZERO_LINE_COLOUR = "0.25"
GAP_LIMIT = 0.53
GAP_TICKS = (-0.5, -0.25, 0.0, 0.25, 0.5)
DATASET_GAP_LIMITS = {
    "pubmed": (-0.03, 0.41),
    "ogbn-arxiv": (-0.03, 0.41),
    "roman-empire": (-0.22, 0.21),
    "amazon-ratings": (-0.53, 0.03),
    "arxiv-year": (-0.04, 0.08),
}
DATASET_GAP_TICKS = {
    "pubmed": (0.0, 0.1, 0.2, 0.3, 0.4),
    "ogbn-arxiv": (0.0, 0.1, 0.2, 0.3, 0.4),
    "roman-empire": (-0.2, -0.1, 0.0, 0.1, 0.2),
    "amazon-ratings": (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0),
    "arxiv-year": (-0.04, 0.0, 0.04, 0.08),
}
Y_SCALE_CHOICES = ("shared", "dataset")
DEFAULT_FIGURE_STEM = (
    "rq1_rq2_unadjusted_and_class_adjusted_macro_f1_tail_gaps_unequal_y"
)
ADDITIONAL_FIGURE_STEM = (
    "rq1_rq2_additional_datasets_unadjusted_and_class_adjusted_"
    "macro_f1_tail_gaps_unequal_y"
)


def panel_spec(
    mode: AnalysisDatasetMode | str,
) -> tuple[
    tuple[str, ...],
    dict[str, tuple[int, int]],
    tuple[int, int],
    int,
    float,
]:
    selected_mode = AnalysisDatasetMode(mode)
    if selected_mode is AnalysisDatasetMode.ORIGINAL:
        return DATASET_ORDER, DATASET_LAYOUT, LEGEND_CELL, 2, FIGURE_HEIGHT_IN
    if selected_mode is AnalysisDatasetMode.ADDITIONAL:
        return (
            ADDITIONAL_DATASET_ORDER,
            ADDITIONAL_DATASET_LAYOUT,
            ADDITIONAL_LEGEND_CELL,
            1,
            ADDITIONAL_FIGURE_HEIGHT_IN,
        )
    raise ValueError(
        "The paired RQ1/RQ2 appendix figure supports original or additional mode"
    )


@dataclass(frozen=True)
class IntervalEstimate:
    mean: float
    ci_lower: float
    ci_upper: float
    ci_excludes_zero: bool


@dataclass(frozen=True)
class PairedGapCell:
    dataset: str
    dataset_order: int
    tail_side: str
    model: str
    model_order: int
    n_runs: int
    unadjusted: IntervalEstimate
    class_adjusted: IntervalEstimate


def parse_bool(value: str, *, context: str) -> bool:
    normalised = value.strip().lower()
    if normalised == "true":
        return True
    if normalised == "false":
        return False
    raise ValueError(f"{context}: expected True or False, found {value!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interval_from_row(
    row: dict[str, str],
    *,
    mean_field: str,
    lower_field: str,
    upper_field: str,
    excludes_field: str,
    context: str,
) -> IntervalEstimate:
    estimate = IntervalEstimate(
        mean=float(row[mean_field]),
        ci_lower=float(row[lower_field]),
        ci_upper=float(row[upper_field]),
        ci_excludes_zero=parse_bool(
            row[excludes_field],
            context=f"{context}:{excludes_field}",
        ),
    )
    if not (
        estimate.ci_lower <= estimate.mean <= estimate.ci_upper
    ):
        raise RuntimeError(
            f"{context}: confidence interval does not contain its mean"
        )
    calculated_excludes = (
        estimate.ci_lower > 0.0 or estimate.ci_upper < 0.0
    )
    if calculated_excludes != estimate.ci_excludes_zero:
        raise RuntimeError(
            f"{context}: stored zero-exclusion flag disagrees with interval"
        )
    return estimate


def load_unadjusted(
    path: Path,
) -> dict[tuple[str, str], tuple[int, int, str, int, IntervalEstimate]]:
    records: dict[
        tuple[str, str],
        tuple[int, int, str, int, IntervalEstimate],
    ] = {}
    for row in read_csv_rows(path):
        dataset = row["dataset"]
        model = row["model"]
        context = f"{path}:{dataset}:{model}"
        key = (dataset, model)
        if key in records:
            raise RuntimeError(f"{context}: duplicate cell")
        estimate = interval_from_row(
            row,
            mean_field="mean_tail_gap",
            lower_field="bootstrap_ci_lower",
            upper_field="bootstrap_ci_upper",
            excludes_field="ci_excludes_zero",
            context=context,
        )
        records[key] = (
            int(row["dataset_order"]),
            int(row["model_order"]),
            row["tail_side"],
            int(row["n_pairs"]),
            estimate,
        )
    return records


def load_adjusted(
    path: Path,
) -> dict[
    tuple[str, str],
    tuple[int, int, str, int, IntervalEstimate, IntervalEstimate],
]:
    records: dict[
        tuple[str, str],
        tuple[
            int,
            int,
            str,
            int,
            IntervalEstimate,
            IntervalEstimate,
        ],
    ] = {}
    for row in read_csv_rows(path):
        dataset = row["dataset"]
        model = row["model"]
        context = f"{path}:{dataset}:{model}"
        key = (dataset, model)
        if key in records:
            raise RuntimeError(f"{context}: duplicate cell")
        adjusted = interval_from_row(
            row,
            mean_field="mean_adjusted_macro_f1_gap",
            lower_field="bootstrap_ci_lower_adjusted_macro_f1_gap",
            upper_field="bootstrap_ci_upper_adjusted_macro_f1_gap",
            excludes_field="adjusted_macro_f1_gap_ci_excludes_zero",
            context=context,
        )
        embedded_unadjusted = interval_from_row(
            row,
            mean_field="erm_mean_tail_gap",
            lower_field="erm_tail_gap_ci_lower",
            upper_field="erm_tail_gap_ci_upper",
            excludes_field="erm_tail_gap_ci_excludes_zero",
            context=f"{context}:embedded-rq1",
        )
        records[key] = (
            int(row["dataset_order"]),
            int(row["model_order"]),
            row["tail_side"],
            int(row["n_runs"]),
            adjusted,
            embedded_unadjusted,
        )
    return records


def validate_expected_panel(
    keys: set[tuple[str, str]],
    *,
    context: str,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> None:
    expected = {
        (dataset, model)
        for dataset in datasets
        for model in MODEL_ORDER
    }
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{context}: incomplete panel; missing={missing}, "
            f"unexpected={unexpected}"
        )


def estimates_match(
    first: IntervalEstimate,
    second: IntervalEstimate,
    *,
    atol: float = 1e-12,
) -> bool:
    return (
        np.isclose(first.mean, second.mean, rtol=0.0, atol=atol)
        and np.isclose(
            first.ci_lower,
            second.ci_lower,
            rtol=0.0,
            atol=atol,
        )
        and np.isclose(
            first.ci_upper,
            second.ci_upper,
            rtol=0.0,
            atol=atol,
        )
        and first.ci_excludes_zero == second.ci_excludes_zero
    )


def load_paired_cells(
    unadjusted_path: Path,
    adjusted_path: Path,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> list[PairedGapCell]:
    unadjusted = load_unadjusted(unadjusted_path)
    adjusted = load_adjusted(adjusted_path)
    validate_expected_panel(
        set(unadjusted), context=str(unadjusted_path), datasets=datasets
    )
    validate_expected_panel(
        set(adjusted), context=str(adjusted_path), datasets=datasets
    )

    cells = []
    for dataset_order, dataset in enumerate(datasets):
        for model_order, model in enumerate(MODEL_ORDER):
            source_model_order = SOURCE_MODEL_ORDER.index(model)
            key = (dataset, model)
            (
                unadjusted_dataset_order,
                unadjusted_model_order,
                unadjusted_tail_side,
                unadjusted_n,
                unadjusted_estimate,
            ) = unadjusted[key]
            (
                adjusted_dataset_order,
                adjusted_model_order,
                adjusted_tail_side,
                adjusted_n,
                adjusted_estimate,
                embedded_unadjusted,
            ) = adjusted[key]

            if (
                unadjusted_dataset_order != dataset_order
                or adjusted_dataset_order != dataset_order
            ):
                raise RuntimeError(f"{key}: dataset-order mismatch")
            if (
                unadjusted_model_order != source_model_order
                or adjusted_model_order != source_model_order
            ):
                raise RuntimeError(f"{key}: source model-order mismatch")
            expected_tail_side = DATASET_REGISTRY[dataset].tail_side
            if (
                unadjusted_tail_side != expected_tail_side
                or adjusted_tail_side != expected_tail_side
            ):
                raise RuntimeError(f"{key}: tail-side mismatch")
            if unadjusted_n != 30 or adjusted_n != 30:
                raise RuntimeError(f"{key}: expected 30 runs")
            if not estimates_match(unadjusted_estimate, embedded_unadjusted):
                raise RuntimeError(
                    f"{key}: RQ2 source does not embed the current RQ1 result"
                )

            cells.append(
                PairedGapCell(
                    dataset=dataset,
                    dataset_order=dataset_order,
                    tail_side=expected_tail_side,
                    model=model,
                    model_order=model_order,
                    n_runs=30,
                    unadjusted=unadjusted_estimate,
                    class_adjusted=adjusted_estimate,
                )
            )

    expected_cells = len(datasets) * len(MODEL_ORDER)
    if len(cells) != expected_cells:
        raise RuntimeError(
            f"Expected {expected_cells} paired cells, found {len(cells)}"
        )
    return cells


def configure_style() -> dict[str, tuple[float, float, float]]:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        font_scale=1.0,
        rc={
            "mathtext.fontset": "stixsans",
            "font.size": MINIMUM_TEXT_SIZE_PT,
            "axes.labelsize": AXIS_LABEL_SIZE_PT,
            "xtick.labelsize": TICK_TEXT_SIZE_PT,
            "ytick.labelsize": TICK_TEXT_SIZE_PT,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.18",
            "axes.labelcolor": "0.12",
            "text.color": "0.12",
            "xtick.color": "0.18",
            "ytick.color": "0.18",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    colours = list(
        sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))
    )
    return {
        model: colours[SOURCE_MODEL_ORDER.index(model)]
        for model in MODEL_ORDER
    }


def gap_tick(value: float, _position: float) -> str:
    if np.isclose(value, 0.0):
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def model_jitters() -> dict[str, float]:
    offsets = np.linspace(
        -MODEL_JITTER_SPAN / 2.0,
        MODEL_JITTER_SPAN / 2.0,
        len(MODEL_ORDER),
    )
    return {
        model: float(offset)
        for model, offset in zip(MODEL_ORDER, offsets)
    }


def y_axis_specification(
    dataset: str,
    y_scale: str,
    cells: list[PairedGapCell] | None = None,
) -> tuple[tuple[float, float], tuple[float, ...]]:
    if y_scale == "shared":
        return (-GAP_LIMIT, GAP_LIMIT), GAP_TICKS
    if y_scale == "dataset":
        if dataset in DATASET_GAP_LIMITS:
            return DATASET_GAP_LIMITS[dataset], DATASET_GAP_TICKS[dataset]
        if cells is None:
            raise ValueError(
                f"{dataset}: cells are required to derive an appendix y-axis"
            )
        bounds = [
            bound
            for cell in cells
            if cell.dataset == dataset
            for estimate in (cell.unadjusted, cell.class_adjusted)
            for bound in (estimate.ci_lower, estimate.ci_upper)
        ]
        if not bounds:
            raise ValueError(f"{dataset}: no intervals available for y-axis")
        low = min(min(bounds), 0.0)
        high = max(max(bounds), 0.0)
        span = max(high - low, 0.02)
        target_step = span / 4.0
        magnitude = 10.0 ** np.floor(np.log10(target_step))
        step = next(
            multiplier * magnitude
            for multiplier in (1.0, 2.0, 2.5, 5.0, 10.0)
            if multiplier * magnitude >= target_step
        )
        tick_low = np.floor(low / step) * step
        tick_high = np.ceil(high / step) * step
        if tick_low >= 0.0:
            tick_low = -step
        if tick_high <= 0.0:
            tick_high = step
        ticks = np.arange(tick_low, tick_high + 0.5 * step, step)
        limits = (float(tick_low - 0.15 * step), float(tick_high + 0.15 * step))
        return limits, tuple(float(tick) for tick in ticks)
    raise ValueError(
        f"Unknown y-axis scale {y_scale!r}; expected one of {Y_SCALE_CHOICES}"
    )


def validate_y_axis_specifications(
    cells: list[PairedGapCell],
    y_scale: str,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> None:
    for dataset in datasets:
        limits, ticks = y_axis_specification(dataset, y_scale, cells)
        lower_limit, upper_limit = limits
        dataset_bounds = [
            bound
            for cell in cells
            if cell.dataset == dataset
            for estimate in (cell.unadjusted, cell.class_adjusted)
            for bound in (estimate.ci_lower, estimate.ci_upper)
        ]
        if min(dataset_bounds) <= lower_limit:
            raise ValueError(
                f"{dataset}: lower interval bound falls outside the "
                f"{y_scale} y-axis limits"
            )
        if max(dataset_bounds) >= upper_limit:
            raise ValueError(
                f"{dataset}: upper interval bound falls outside the "
                f"{y_scale} y-axis limits"
            )
        if not lower_limit < 0.0 < upper_limit:
            raise ValueError(f"{dataset}: y-axis limits must straddle zero")
        if not any(np.isclose(tick, 0.0) for tick in ticks):
            raise ValueError(f"{dataset}: y-axis ticks must include zero")


def draw_estimate(
    axis: plt.Axes,
    estimate: IntervalEstimate,
    *,
    x: float,
    model: str,
    colour: tuple[float, float, float],
) -> None:
    facecolour = colour if estimate.ci_excludes_zero else "white"
    edgecolour = "white" if estimate.ci_excludes_zero else colour
    edgewidth = 0.35 if estimate.ci_excludes_zero else 0.80
    axis.errorbar(
        x,
        estimate.mean,
        yerr=np.asarray(
            [
                [estimate.mean - estimate.ci_lower],
                [estimate.ci_upper - estimate.mean],
            ],
            dtype=np.float64,
        ),
        fmt=MODEL_MARKERS[model],
        linestyle="None",
        color=colour,
        ecolor=colour,
        elinewidth=CI_LINE_WIDTH_PT,
        capsize=1.7,
        capthick=0.65,
        markersize=MARKER_SIZE_PT,
        markerfacecolor=facecolour,
        markeredgecolor=edgecolour,
        markeredgewidth=edgewidth,
        zorder=3,
    )


def add_dataset_panel(
    axis: plt.Axes,
    dataset: str,
    by_cell: dict[tuple[str, str], PairedGapCell],
    model_colours: dict[str, tuple[float, float, float]],
    jitters: dict[str, float],
    *,
    y_scale: str,
    y_specification: tuple[tuple[float, float], tuple[float, ...]] | None = None,
    show_y_axis_label: bool,
    show_y_tick_labels: bool,
) -> None:
    for model in MODEL_ORDER:
        cell = by_cell[(dataset, model)]
        colour = model_colours[model]
        jitter = jitters[model]
        unadjusted_x = CONDITION_POSITIONS["unadjusted"] + jitter
        adjusted_x = CONDITION_POSITIONS["class_adjusted"] + jitter
        axis.plot(
            [unadjusted_x, adjusted_x],
            [cell.unadjusted.mean, cell.class_adjusted.mean],
            color=colour,
            linewidth=CONNECTOR_LINE_WIDTH_PT,
            solid_capstyle="round",
            alpha=CONNECTOR_ALPHA,
            zorder=1.5,
        )
        draw_estimate(
            axis,
            cell.unadjusted,
            x=unadjusted_x,
            model=model,
            colour=colour,
        )
        draw_estimate(
            axis,
            cell.class_adjusted,
            x=adjusted_x,
            model=model,
            colour=colour,
        )

    axis.set_xlim(
        (-0.30, 1.30) if y_scale == "dataset" else (-0.35, 1.35)
    )
    y_limits, y_ticks = y_specification or y_axis_specification(
        dataset,
        y_scale,
        list(by_cell.values()),
    )
    axis.set_ylim(*y_limits)
    axis.set_xticks(
        [
            CONDITION_POSITIONS["unadjusted"],
            CONDITION_POSITIONS["class_adjusted"],
        ]
    )
    axis.set_xticklabels(
        [
            CONDITION_LABELS["unadjusted"],
            CONDITION_LABELS["class_adjusted"],
        ]
    )
    axis.set_yticks(y_ticks)
    axis.yaxis.set_major_formatter(FuncFormatter(gap_tick))
    axis.set_title(
        DATASET_LABELS[dataset],
        fontsize=DATASET_TEXT_SIZE_PT,
        fontweight="normal",
        pad=3,
    )
    axis.grid(axis="y", color="0.91", linewidth=0.35, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.axhline(
        0.0,
        color=ZERO_LINE_COLOUR,
        linewidth=ZERO_LINE_WIDTH_PT,
        linestyle=(0, (4, 2.2)),
        zorder=2.2,
    )
    axis.tick_params(axis="x", length=0, pad=4)
    axis.tick_params(
        axis="y",
        direction="out",
        length=2.5,
        width=0.65,
    )
    if show_y_axis_label:
        axis.set_ylabel("Tail gap")
    if not show_y_tick_labels:
        axis.tick_params(axis="y", labelleft=False)
    sns.despine(ax=axis)


def model_legend_handles(
    model_colours: dict[str, tuple[float, float, float]],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            linestyle="-",
            linewidth=CONNECTOR_LINE_WIDTH_PT,
            color=(*model_colours[model], CONNECTOR_ALPHA),
            markerfacecolor=model_colours[model],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=3.8,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]


def other_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="|",
            linestyle="-",
            linewidth=CI_LINE_WIDTH_PT,
            color="0.38",
            markersize=6.0,
            markeredgewidth=0.70,
            label="95% CI",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="0.38",
            markerfacecolor="white",
            markeredgecolor="0.38",
            markeredgewidth=0.80,
            markersize=3.8,
            label="Unfilled: CI includes 0",
        ),
        Line2D(
            [0],
            [0],
            linestyle=(0, (4, 2.2)),
            linewidth=ZERO_LINE_WIDTH_PT,
            color=ZERO_LINE_COLOUR,
            label="Zero gap",
        ),
    ]


def add_shared_legend(
    legend_axis: plt.Axes,
    model_colours: dict[str, tuple[float, float, float]],
    y_scale: str,
) -> None:
    legend_axis.axis("off")
    model_legend = legend_axis.legend(
        handles=model_legend_handles(model_colours),
        title="Models",
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.04),
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
    legend_axis.legend(
        handles=other_legend_handles(),
        title="Other",
        loc="lower left",
        bbox_to_anchor=(-0.03, 0.00),
        ncol=1,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.48,
        labelspacing=0.25,
        borderaxespad=0.0,
        fontsize=LEGEND_TEXT_SIZE_PT,
        title_fontsize=LEGEND_TITLE_SIZE_PT,
    )


def plot_paired_forest(
    cells: list[PairedGapCell],
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
    y_scale: str,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> list[Path]:
    (
        selected_datasets,
        dataset_layout,
        legend_cell,
        grid_rows,
        figure_height,
    ) = panel_spec(dataset_mode)
    validate_y_axis_specifications(cells, y_scale, selected_datasets)
    model_colours = configure_style()
    by_cell = {(cell.dataset, cell.model): cell for cell in cells}
    jitters = model_jitters()

    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, figure_height))
    grid = figure.add_gridspec(
        grid_rows,
        3,
        width_ratios=(1.0, 1.0, 1.15) if y_scale == "dataset" else None,
        left=0.085,
        right=0.985,
        top=0.950 if grid_rows == 2 else 0.925,
        bottom=0.105 if grid_rows == 2 else 0.190,
        wspace=0.28 if y_scale == "dataset" else 0.20,
        hspace=(0.34 if y_scale == "dataset" else 0.30) if grid_rows == 2 else 0.0,
    )
    for dataset in selected_datasets:
        row, column = dataset_layout[dataset]
        axis = figure.add_subplot(grid[row, column])
        add_dataset_panel(
            axis,
            dataset,
            by_cell,
            model_colours,
            jitters,
            y_scale=y_scale,
            y_specification=y_axis_specification(dataset, y_scale, cells),
            show_y_axis_label=(column == 0),
            show_y_tick_labels=(
                y_scale == "dataset" or column == 0
            ),
        )
    add_shared_legend(
        figure.add_subplot(grid[legend_cell]),
        model_colours,
        y_scale,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": (
            "RQ1 and RQ2 unadjusted and class-adjusted macro-F1 tail gaps"
        ),
        "Author": "homophily-failure-decomp",
        "Subject": (
            "RQ1 and RQ2 paired slope panels with "
            f"{'panel-specific' if y_scale == 'dataset' else 'shared'} "
            "y-axis scaling"
        ),
    }
    pdf_metadata = {
        **metadata,
        "CreationDate": None,
        "ModDate": None,
    }
    outputs = []
    for output_format in formats:
        output_path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            output_path,
            dpi=dpi,
            metadata=(
                pdf_metadata if output_format.lower() == "pdf" else metadata
            ),
            facecolor="white",
        )
        outputs.append(output_path)
    plt.close(figure)
    return outputs


def write_plot_csv(path: Path, cells: list[PairedGapCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset_order",
                "dataset",
                "tail_side",
                "model_order",
                "model",
                "n_runs",
                "unadjusted_mean_gap",
                "unadjusted_ci_lower",
                "unadjusted_ci_upper",
                "unadjusted_ci_excludes_zero",
                "class_adjusted_mean_gap",
                "class_adjusted_ci_lower",
                "class_adjusted_ci_upper",
                "class_adjusted_ci_excludes_zero",
            ]
        )
        for cell in cells:
            writer.writerow(
                [
                    cell.dataset_order,
                    cell.dataset,
                    cell.tail_side,
                    cell.model_order,
                    cell.model,
                    cell.n_runs,
                    f"{cell.unadjusted.mean:.10f}",
                    f"{cell.unadjusted.ci_lower:.10f}",
                    f"{cell.unadjusted.ci_upper:.10f}",
                    cell.unadjusted.ci_excludes_zero,
                    f"{cell.class_adjusted.mean:.10f}",
                    f"{cell.class_adjusted.ci_lower:.10f}",
                    f"{cell.class_adjusted.ci_upper:.10f}",
                    cell.class_adjusted.ci_excludes_zero,
                ]
            )


def write_readme(
    path: Path,
    figure_stem: str,
    y_scale: str,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> None:
    selected_datasets, *_ = panel_spec(dataset_mode)
    scale_description = (
        "panel-specific vertical scales, all of which include zero"
        if y_scale == "dataset"
        else "one shared vertical tail-gap scale"
    )
    scale_argument = "" if y_scale == "dataset" else " --y-scale shared"
    mode_argument = (
        ""
        if AnalysisDatasetMode(dataset_mode) is AnalysisDatasetMode.ORIGINAL
        else " --dataset-mode additional"
    )
    text = f"""# RQ1 and RQ2 grouped tail-gap plot

`{figure_stem}.pdf` and `{figure_stem}.png` show the {len(selected_datasets) * len(MODEL_ORDER)} ERM
dataset-model cells in {len(selected_datasets)} dataset panels using {scale_description}.
Within each panel, each coloured model line joins its unadjusted RQ1
macro-F1 tail gap to its class-composition-adjusted RQ2 macro-F1 tail gap.
Endpoints carry 95% paired-bootstrap confidence intervals.

Filled markers denote intervals that exclude zero. Open markers denote
intervals that contain zero. The faint connector joins the two point estimates
only; it is not an interval for their difference.

Regenerate from the tracked summary tables with:

```powershell
python scripts/plot_rq1_rq2_paired_forest.py{mode_argument}{scale_argument}
```
"""
    path.write_text(text, encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    cells: list[PairedGapCell],
    unadjusted_path: Path,
    adjusted_path: Path,
    output_paths: list[Path],
    plot_csv_path: Path,
    readme_path: Path,
    args: argparse.Namespace,
) -> None:
    (
        selected_datasets,
        dataset_layout,
        legend_cell,
        grid_rows,
        figure_height,
    ) = panel_spec(args.dataset_mode)
    y_axis_specifications = {
        dataset: y_axis_specification(dataset, args.y_scale, cells)
        for dataset in selected_datasets
    }
    payload = {
        "schema_version": 1,
        "analysis": "rq1-rq2-grouped-macro-f1-tail-gap-plot",
        "dataset_mode": args.dataset_mode,
        "definitions": {
            "unadjusted_gap": (
                "whole-test macro-F1 minus registered-tail macro-F1"
            ),
            "class_adjusted_gap": (
                "ordinary whole-test macro-F1 over tail-present classes "
                "minus tail macro-F1 after reweighting to the whole-test "
                "class composition"
            ),
            "intervals": (
                "95% percentile bootstrap intervals over 30 complete "
                "split-seed runs, read from the source analyses"
            ),
            "open_marker": "the corresponding interval contains zero",
            "filled_marker": "the corresponding interval excludes zero",
            "connector": (
                "descriptive line joining the two point estimates; not an "
                "interval for their difference"
            ),
        },
        "panel": {
            "dataset_order": list(selected_datasets),
            "model_order": list(MODEL_ORDER),
            "n_cells": len(cells),
            "n_unadjusted_intervals_including_zero": sum(
                not cell.unadjusted.ci_excludes_zero for cell in cells
            ),
            "n_class_adjusted_intervals_including_zero": sum(
                not cell.class_adjusted.ci_excludes_zero for cell in cells
            ),
        },
        "figure_design": {
            "width_inches": FIGURE_WIDTH_IN,
            "height_inches": figure_height,
            "minimum_text_size_points": MINIMUM_TEXT_SIZE_PT,
            "composition": (
                f"{grid_rows}-by-three grid with {len(selected_datasets)} "
                "dataset slope panels and one legend cell"
            ),
            "styling_reference": (
                "paper Figure 3, RQ1 continuous local-homophily composite"
            ),
            "grid_shape": [grid_rows, 3],
            "dataset_layout": {
                dataset: list(dataset_layout[dataset])
                for dataset in selected_datasets
            },
            "legend_cell": list(legend_cell),
            "y_scale_mode": args.y_scale,
            "y_limits": {
                dataset: list(y_axis_specifications[dataset][0])
                for dataset in selected_datasets
            },
            "y_ticks": {
                dataset: list(y_axis_specifications[dataset][1])
                for dataset in selected_datasets
            },
            "shared_scale": args.y_scale == "shared",
            "condition_positions": CONDITION_POSITIONS,
            "model_jitter_span": MODEL_JITTER_SPAN,
            "model_jitters": model_jitters(),
            "model_palette": "seaborn colorblind",
            "model_markers": MODEL_MARKERS,
            "marker_size_points": MARKER_SIZE_PT,
            "ci_line_width_points": CI_LINE_WIDTH_PT,
            "connector_line_width_points": CONNECTOR_LINE_WIDTH_PT,
            "connector_alpha": CONNECTOR_ALPHA,
            "zero_line_width_points": ZERO_LINE_WIDTH_PT,
            "zero_line_colour": ZERO_LINE_COLOUR,
        },
        "inputs": {
            "unadjusted": {
                "path": relative_path(unadjusted_path),
                "sha256": file_sha256(unadjusted_path),
            },
            "class_adjusted": {
                "path": relative_path(adjusted_path),
                "sha256": file_sha256(adjusted_path),
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "script": {
            "path": relative_path(Path(__file__)),
            "sha256": file_sha256(Path(__file__)),
        },
        "command": {
            "dataset_mode": args.dataset_mode,
            "formats": list(args.formats),
            "dpi": args.dpi,
            "y_scale": args.y_scale,
        },
        "outputs": {
            relative_path(output_path): file_sha256(output_path)
            for output_path in output_paths
        }
        | {
            relative_path(plot_csv_path): file_sha256(plot_csv_path),
            relative_path(readme_path): file_sha256(readme_path),
        },
        "cells": [
            {
                **{
                    key: value
                    for key, value in asdict(cell).items()
                    if key not in {"unadjusted", "class_adjusted"}
                },
                "unadjusted": asdict(cell.unadjusted),
                "class_adjusted": asdict(cell.class_adjusted),
            }
            for cell in cells
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


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
        "--unadjusted-csv",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "aggregates"
        / "erm_tail_gaps.csv",
    )
    parser.add_argument(
        "--class-adjusted-csv",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "rq2"
        / "class_adjusted_cell_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "main"
        / "figures"
        / "rq1-rq2"
        / "unequal-y",
    )
    parser.add_argument(
        "--figure-stem",
        default=DEFAULT_FIGURE_STEM,
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    parser.add_argument(
        "--y-scale",
        choices=Y_SCALE_CHOICES,
        default="dataset",
        help=(
            "Use one shared y-axis or panel-specific dataset scales. "
            "Both modes include zero in every panel."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    selected_datasets, *_ = panel_spec(mode)
    original_unadjusted = (
        REPO_ROOT / "results/main/aggregates/erm_tail_gaps.csv"
    )
    original_adjusted = (
        REPO_ROOT / "results/main/rq2/class_adjusted_cell_summary.csv"
    )
    original_output = REPO_ROOT / "results/main/figures/rq1-rq2/unequal-y"
    if mode is AnalysisDatasetMode.ADDITIONAL:
        if args.unadjusted_csv == original_unadjusted:
            args.unadjusted_csv = (
                REPO_ROOT / "results/additional/aggregates/erm_tail_gaps.csv"
            )
        if args.class_adjusted_csv == original_adjusted:
            args.class_adjusted_csv = (
                REPO_ROOT
                / "results/additional/rq2/class_adjusted_cell_summary.csv"
            )
        if args.output_dir == original_output:
            args.output_dir = (
                REPO_ROOT / "results/additional/figures/rq1-rq2/unequal-y"
            )
        if args.figure_stem == DEFAULT_FIGURE_STEM:
            args.figure_stem = ADDITIONAL_FIGURE_STEM
    cells = load_paired_cells(
        args.unadjusted_csv,
        args.class_adjusted_csv,
        selected_datasets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / args.figure_stem
    output_paths = plot_paired_forest(
        cells,
        output_stem,
        formats=tuple(args.formats),
        dpi=args.dpi,
        y_scale=args.y_scale,
        dataset_mode=mode,
    )
    plot_csv_path = args.output_dir / "paired_tail_gap_values.csv"
    readme_path = args.output_dir / "README.md"
    manifest_path = args.output_dir / "analysis_manifest.json"
    write_plot_csv(plot_csv_path, cells)
    write_readme(readme_path, args.figure_stem, args.y_scale, mode)
    write_manifest(
        manifest_path,
        cells=cells,
        unadjusted_path=args.unadjusted_csv,
        adjusted_path=args.class_adjusted_csv,
        output_paths=output_paths,
        plot_csv_path=plot_csv_path,
        readme_path=readme_path,
        args=args,
    )
    for path in output_paths + [
        plot_csv_path,
        readme_path,
        manifest_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
