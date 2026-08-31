"""Summarise and plot absolute RQ5 tail scores for all preparations.

For every dataset--model--preparation cell, this script reads the corrected
run-level records, calculates the arithmetic mean tail macro-F1 over the 30
registered split--seed runs, and obtains a 95% percentile-bootstrap confidence
interval by resampling complete runs.

It writes two paper-ready figures per dataset. The tail-only figure aligns the
four preparations vertically within each of the six graph models. The
comparison figure offsets A_erm, A_last, A_head, and A_full from left to right
and overlays tail and whole-test macro-F1. It also writes one compact comparison
figure containing all five datasets for use in the main paper.

Example:
    python scripts/analyze_rq5_preparations.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns

from hfd.analysis_datasets import (
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
)

DATASET_ORDER = ORIGINAL_DATASET_ORDER
DATASET_LABELS = {
    name: metadata.display_label for name, metadata in DATASET_REGISTRY.items()
}
TAIL_LABELS = {
    name: (
        r"low-$h_v$ tail"
        if metadata.tail_side == "low-h_v"
        else r"high-$h_v$ tail"
    )
    for name, metadata in DATASET_REGISTRY.items()
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
PLOT_MODEL_ORDER = (
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
    "graphsage-mean": "GraphSAGE-\nmean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "linkx": "LINKX",
    "acm-gcn": "ACM-GCN",
}
COMPACT_MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage-mean": "SAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR",
    "linkx": "LINKX",
    "acm-gcn": "ACM",
}
PREPARATION_ORDER = ("erm", "last", "head", "full")
PREPARATION_LABELS = {
    "erm": r"$A_{\mathrm{ERM}}$",
    "last": r"$A_{\mathrm{last}}$",
    "head": r"$A_{\mathrm{head}}$",
    "full": r"$A_{\mathrm{full}}$",
}
EXPECTED_RUNS = 30
ANALYSIS_VERSION = "true-label-support-v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20_260_723
COMPACT_FIGURE_SIZE_IN = (5.5, 2.95)
COMPACT_PANEL_TITLE_SIZE_PT = 8.3
COMPACT_X_TICK_SIZE_PT = 6.0
COMPACT_Y_TICK_SIZE_PT = 7.2
COMPACT_KEY_TEXT_SIZE_PT = 7.2
COMPACT_KEY_X_TICK_SIZE_PT = 8.5
COMPACT_SHARED_LABEL_SIZE_PT = 8.25
COMPACT_DATA_LINE_WIDTH_PT = 0.60
COMPACT_WHOLE_MARKER_SIZE_PT = 2.4
COMPACT_TAIL_MARKER_SIZE_PT = 2.3
COMPACT_KEY_LINE_WIDTH_PT = 0.70
COMPACT_KEY_MARKER_SIZE_PT = 3.0


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel used by this script's analysis and plots."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)


@dataclass(frozen=True)
class RunScore:
    run_id: str
    split: str
    seed: int
    tail_macro_f1: float
    whole_test_macro_f1: float
    path: Path
    sha256: str


@dataclass(frozen=True)
class ScoreSummary:
    dataset: str
    model: str
    preparation: str
    n_runs: int
    mean_tail_macro_f1: float
    sample_std_tail_macro_f1: float
    mean_whole_test_macro_f1: float
    sample_std_whole_test_macro_f1: float
    ci_lower: float
    ci_upper: float
    run_scores: tuple[float, ...]


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


def bootstrap_seed(base_seed: int, *parts: str) -> int:
    key = ":".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def percentile_bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        values.size,
        size=(resamples, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
        method="linear",
    )
    return float(lower), float(upper)


def load_cell_runs(
    run_root: Path,
    dataset: str,
    model: str,
) -> dict[str, dict[tuple[str, int], RunScore]]:
    model_dir = run_root / dataset / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {model_dir}")

    records = {preparation: {} for preparation in PREPARATION_ORDER}
    for path in sorted(model_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        preparation = str(payload.get("preparation"))
        if preparation not in records:
            continue
        if payload.get("dataset") != dataset:
            raise RuntimeError(f"Dataset mismatch in {path}")
        if payload.get("model") != model:
            raise RuntimeError(f"Model mismatch in {path}")
        if payload.get("analysis_version") != ANALYSIS_VERSION:
            raise RuntimeError(
                f"{path}: expected analysis_version={ANALYSIS_VERSION!r}"
            )

        split = str(payload["split"])
        seed = int(payload["seed"])
        key = (split, seed)
        if key in records[preparation]:
            raise RuntimeError(
                f"Duplicate {preparation} split-seed pair for "
                f"{dataset}/{model}: {key}"
            )
        records[preparation][key] = RunScore(
            run_id=str(payload["run_id"]),
            split=split,
            seed=seed,
            tail_macro_f1=float(payload["metrics"]["tail"]["macro_f1"]),
            whole_test_macro_f1=float(
                payload["metrics"]["whole_test"]["macro_f1"]
            ),
            path=path,
            sha256=file_sha256(path),
        )

    reference_keys: set[tuple[str, int]] | None = None
    for preparation in PREPARATION_ORDER:
        preparation_records = records[preparation]
        if len(preparation_records) != EXPECTED_RUNS:
            raise RuntimeError(
                f"{dataset}/{model}/{preparation}: found "
                f"{len(preparation_records)} corrected runs; "
                f"expected {EXPECTED_RUNS}"
            )
        keys = set(preparation_records)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            missing = sorted(reference_keys - keys)
            unexpected = sorted(keys - reference_keys)
            raise RuntimeError(
                f"{dataset}/{model}/{preparation}: split-seed panel differs; "
                f"missing={missing}, unexpected={unexpected}"
            )

    return records


def summarise_cell(
    dataset: str,
    model: str,
    preparation: str,
    records: dict[tuple[str, int], RunScore],
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> ScoreSummary:
    tail_values = np.asarray(
        [
            records[key].tail_macro_f1
            for key in sorted(records)
        ],
        dtype=np.float64,
    )
    whole_test_values = np.asarray(
        [
            records[key].whole_test_macro_f1
            for key in sorted(records)
        ],
        dtype=np.float64,
    )
    lower, upper = percentile_bootstrap_mean(
        tail_values,
        seed=bootstrap_seed(
            base_seed,
            dataset,
            model,
            preparation,
            "absolute-tail-macro-f1",
        ),
        resamples=bootstrap_resamples,
    )
    return ScoreSummary(
        dataset=dataset,
        model=model,
        preparation=preparation,
        n_runs=tail_values.size,
        mean_tail_macro_f1=float(tail_values.mean()),
        sample_std_tail_macro_f1=float(tail_values.std(ddof=1)),
        mean_whole_test_macro_f1=float(whole_test_values.mean()),
        sample_std_whole_test_macro_f1=float(
            whole_test_values.std(ddof=1)
        ),
        ci_lower=lower,
        ci_upper=upper,
        run_scores=tuple(float(value) for value in tail_values),
    )


def analyse(
    run_root: Path,
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> tuple[
    list[ScoreSummary],
    dict[
        tuple[str, str],
        dict[str, dict[tuple[str, int], RunScore]],
    ],
]:
    summaries: list[ScoreSummary] = []
    all_runs: dict[
        tuple[str, str],
        dict[str, dict[tuple[str, int], RunScore]],
    ] = {}

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            records = load_cell_runs(run_root, dataset, model)
            all_runs[(dataset, model)] = records
            for preparation in PREPARATION_ORDER:
                summaries.append(
                    summarise_cell(
                        dataset,
                        model,
                        preparation,
                        records[preparation],
                        bootstrap_resamples=bootstrap_resamples,
                        base_seed=base_seed,
                    )
                )

    return summaries, all_runs


def configure_style() -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, str],
]:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="serif",
        font_scale=1.0,
        rc={
            "font.serif": [
                "TeX Gyre Termes",
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.65,
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
    blue_map = plt.get_cmap("Blues")
    palette = [
        blue_map(position)
        for position in np.linspace(0.44, 0.78, len(PREPARATION_ORDER))
    ]
    colours = {
        preparation: palette[index]
        for index, preparation in enumerate(PREPARATION_ORDER)
    }
    markers = {
        "erm": "o",
        "last": "s",
        "head": "D",
        "full": "^",
    }
    return colours, markers


def configure_compact_style() -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Configure the compact main-text figure at its final paper dimensions."""
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        font_scale=1.0,
        rc={
            "mathtext.fontset": "stixsans",
            "font.size": 7.5,
            "axes.labelsize": COMPACT_SHARED_LABEL_SIZE_PT,
            "axes.titlesize": COMPACT_PANEL_TITLE_SIZE_PT,
            "xtick.labelsize": COMPACT_X_TICK_SIZE_PT,
            "ytick.labelsize": COMPACT_Y_TICK_SIZE_PT,
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
        },
    )
    blue, orange = sns.color_palette("colorblind", n_colors=2)
    return blue, orange


def y_axis_spec(
    values: list[float],
) -> tuple[float, float, np.ndarray]:
    if not values:
        raise ValueError("At least one plotted value is required")
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
        1.0,
        float(np.ceil((upper_data + padding) / rounding) * rounding),
    )
    plotted_span = upper - lower
    tick_step = 0.05 if plotted_span <= 0.40 else 0.10
    first_tick = np.ceil(lower / tick_step) * tick_step
    last_tick = np.floor(upper / tick_step) * tick_step
    ticks = np.arange(first_tick, last_tick + tick_step / 2.0, tick_step)
    return lower, upper, ticks


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    dataset: str,
    formats: tuple[str, ...],
    dpi: int,
    metadata_title: str | None = None,
    tight: bool = True,
) -> list[Path]:
    metadata = {
        "Title": metadata_title
        or (
            f"RQ5 {DATASET_LABELS[dataset]} tail macro-F1 "
            "by model and preparation"
        ),
        "Author": "homophily-failure-decomp",
        "Subject": "RQ5",
    }
    outputs: list[Path] = []
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        output_metadata = dict(metadata)
        if output_format == "pdf":
            output_metadata.update(
                {
                    "CreationDate": None,
                    "ModDate": None,
                }
            )
        layout_options = (
            {"bbox_inches": "tight", "pad_inches": 0.10}
            if tight
            else {}
        )
        figure.savefig(
            path,
            dpi=dpi,
            metadata=output_metadata,
            **layout_options,
        )
        outputs.append(path)
    return outputs


def plot_dataset(
    dataset: str,
    summaries: list[ScoreSummary],
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    colours, markers = configure_style()
    by_cell = {
        (row.model, row.preparation): row
        for row in summaries
        if row.dataset == dataset and row.model in PLOT_MODEL_ORDER
    }
    expected = len(PLOT_MODEL_ORDER) * len(PREPARATION_ORDER)
    if len(by_cell) != expected:
        raise RuntimeError(
            f"{dataset}: found {len(by_cell)} plotted cells; expected {expected}"
        )

    figure, axis = plt.subplots(figsize=(7.2, 3.75))
    model_positions = np.arange(len(PLOT_MODEL_ORDER), dtype=float)

    for model_index, model in enumerate(PLOT_MODEL_ORDER):
        means = [
            by_cell[(model, preparation)].mean_tail_macro_f1
            for preparation in PREPARATION_ORDER
        ]
        axis.plot(
            [model_index, model_index],
            [min(means), max(means)],
            color=colours["full"],
            alpha=0.35,
            linewidth=0.9,
            linestyle=(0, (3.0, 2.2)),
            zorder=1,
        )

    for preparation in PREPARATION_ORDER:
        colour = colours[preparation]
        marker = markers[preparation]
        for model_index, model in enumerate(PLOT_MODEL_ORDER):
            summary = by_cell[(model, preparation)]
            x = model_index
            axis.plot(
                x,
                summary.mean_tail_macro_f1,
                marker=marker,
                linestyle="None",
                markersize=5.2,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.55,
                color=colour,
                zorder=3,
            )

    dataset_summaries = [
        row
        for row in summaries
        if row.dataset == dataset and row.model in PLOT_MODEL_ORDER
    ]
    y_lower, y_upper, y_ticks = y_axis_spec(
        [row.mean_tail_macro_f1 for row in dataset_summaries]
    )
    axis.set_xlim(-0.55, len(PLOT_MODEL_ORDER) - 0.45)
    axis.set_ylim(y_lower, y_upper)
    axis.set_xticks(model_positions)
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in PLOT_MODEL_ORDER]
    )
    axis.set_yticks(y_ticks)
    axis.set_ylabel("Tail macro-F1")
    axis.set_xlabel("Model")
    axis.set_title(DATASET_LABELS[dataset], pad=24)
    axis.grid(axis="y", color="0.86", linewidth=0.65)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(axis="y", direction="out", length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[preparation],
            linestyle="None",
            color=colours[preparation],
            markerfacecolor=colours[preparation],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.5,
            label=PREPARATION_LABELS[preparation],
        )
        for preparation in PREPARATION_ORDER
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.55,
        borderaxespad=0.0,
        fontsize=8.0,
    )

    figure.subplots_adjust(left=0.10, right=0.985, top=0.81, bottom=0.18)
    outputs = save_figure(
        figure,
        output_stem,
        dataset=dataset,
        formats=formats,
        dpi=dpi,
    )
    plt.close(figure)
    return outputs


def plot_tail_vs_whole_dataset(
    dataset: str,
    summaries: list[ScoreSummary],
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
            f"{dataset}: found {len(by_cell)} comparison cells; "
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
            by_cell[(model, preparation)].mean_tail_macro_f1
            for preparation in PREPARATION_ORDER
        ]
        whole_test_values = [
            by_cell[
                (model, preparation)
            ].mean_whole_test_macro_f1
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

    dataset_summaries = [
        row
        for row in summaries
        if row.dataset == dataset and row.model in PLOT_MODEL_ORDER
    ]
    plotted_values = [
        value
        for row in dataset_summaries
        for value in (
            row.mean_tail_macro_f1,
            row.mean_whole_test_macro_f1,
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
    axis.set_ylabel("Macro-F1")
    axis.set_xlabel("")
    axis.set_title(
        DATASET_LABELS[dataset],
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
    axis.legend(
        handles=metric_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.008),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.3,
        borderaxespad=0.0,
        fontsize=7.5,
    )

    figure.text(
        0.5,
        0.018,
        (
            r"$A_{\mathrm{ERM}}\quad A_{\mathrm{last}}\quad "
            r"A_{\mathrm{head}}\quad A_{\mathrm{full}}$"
        ),
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="0.36",
    )
    figure.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.19)
    outputs = save_figure(
        figure,
        output_stem,
        dataset=dataset,
        formats=formats,
        dpi=dpi,
        metadata_title=(
            f"RQ5 {DATASET_LABELS[dataset]} tail and whole-test macro-F1 "
            "by graph model and preparation"
        ),
    )
    plt.close(figure)
    return outputs


def build_tail_vs_whole_compact_figure(
    summaries: list[ScoreSummary],
) -> plt.Figure:
    blue, orange = configure_compact_style()
    by_cell = {
        (row.dataset, row.model, row.preparation): row
        for row in summaries
        if row.model in PLOT_MODEL_ORDER
    }
    expected = (
        len(DATASET_ORDER) * len(PLOT_MODEL_ORDER) * len(PREPARATION_ORDER)
    )
    if len(by_cell) != expected:
        raise RuntimeError(
            f"Found {len(by_cell)} compact-figure cells; expected {expected}"
        )

    metric_colours = {
        "tail": blue,
        "whole_test": orange,
    }
    if DATASET_ORDER == ORIGINAL_DATASET_ORDER:
        panel_positions = {
            "pubmed": (0, 0),
            "ogbn-arxiv": (0, 1),
            "roman-empire": (1, 0),
            "amazon-ratings": (1, 1),
            "arxiv-year": (1, 2),
        }
        key_position = (0, 2)
        figure_size = COMPACT_FIGURE_SIZE_IN
        n_rows, n_columns = 2, 3
    else:
        panel_positions = {
            dataset: (0, index)
            for index, dataset in enumerate(DATASET_ORDER)
        }
        key_position = (0, len(DATASET_ORDER))
        figure_size = (COMPACT_FIGURE_SIZE_IN[0], 1.65)
        n_rows, n_columns = 1, len(DATASET_ORDER) + 1
    preparation_offsets = np.linspace(-0.16, 0.16, len(PREPARATION_ORDER))
    model_positions = np.arange(len(PLOT_MODEL_ORDER), dtype=float) * 0.70
    figure, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    axes = np.asarray(axes, dtype=object).reshape(n_rows, n_columns)

    for dataset, (row_index, column_index) in panel_positions.items():
        axis = axes[row_index, column_index]
        plotted_values: list[float] = []
        for model_index, model in enumerate(PLOT_MODEL_ORDER):
            x_positions = model_positions[model_index] + preparation_offsets
            tail_values = [
                by_cell[(dataset, model, preparation)].mean_tail_macro_f1
                for preparation in PREPARATION_ORDER
            ]
            whole_test_values = [
                by_cell[
                    (dataset, model, preparation)
                ].mean_whole_test_macro_f1
                for preparation in PREPARATION_ORDER
            ]
            plotted_values.extend(tail_values)
            plotted_values.extend(whole_test_values)
            axis.plot(
                x_positions,
                tail_values,
                color=metric_colours["tail"],
                linewidth=COMPACT_DATA_LINE_WIDTH_PT,
                linestyle="-",
                marker="o",
                markersize=COMPACT_TAIL_MARKER_SIZE_PT,
                markerfacecolor=metric_colours["tail"],
                markeredgecolor="white",
                markeredgewidth=0.25,
                solid_capstyle="round",
                zorder=3,
            )
            axis.plot(
                x_positions,
                whole_test_values,
                color=metric_colours["whole_test"],
                linewidth=COMPACT_DATA_LINE_WIDTH_PT,
                linestyle="-",
                marker="s",
                markersize=COMPACT_WHOLE_MARKER_SIZE_PT,
                markerfacecolor=metric_colours["whole_test"],
                markeredgecolor="white",
                markeredgewidth=0.25,
                solid_capstyle="round",
                zorder=2,
            )

        y_lower, y_upper, y_ticks = y_axis_spec(plotted_values)
        axis.set_xlim(model_positions[0] - 0.23, model_positions[-1] + 0.23)
        axis.set_ylim(y_lower, y_upper)
        axis.set_xticks(model_positions)
        axis.set_xticklabels(
            [COMPACT_MODEL_LABELS[model] for model in PLOT_MODEL_ORDER]
        )
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
            fontsize=COMPACT_PANEL_TITLE_SIZE_PT,
        )
        axis.grid(axis="y", color="0.88", linewidth=0.35)
        axis.grid(axis="x", visible=False)
        axis.tick_params(
            axis="x",
            length=0,
            pad=3,
            labelsize=COMPACT_X_TICK_SIZE_PT,
        )
        axis.tick_params(
            axis="y",
            direction="out",
            length=2.2,
            width=0.55,
            pad=2.2,
            labelsize=COMPACT_Y_TICK_SIZE_PT,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    key_axis = axes[key_position]
    key_x = np.arange(len(PREPARATION_ORDER), dtype=float)
    key_tail = np.array([0.33, 0.50, 0.40, 0.48])
    key_whole = np.array([0.76, 0.61, 0.71, 0.64])
    key_axis.set_xlim(-0.35, 5.00)
    key_axis.set_ylim(0.18, 0.82)
    key_axis.set_xticks(key_x)
    key_axis.set_xticklabels(
        [
            r"$A_{\mathrm{erm}}$",
            r"$A_{\mathrm{last}}$",
            r"$A_{\mathrm{head}}$",
            r"$A_{\mathrm{full}}$",
        ]
    )
    key_axis.set_yticks([0.2, 0.4, 0.6, 0.8])
    key_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    key_axis.set_title(
        "Example for one model",
        loc="left",
        pad=4,
        fontsize=COMPACT_PANEL_TITLE_SIZE_PT,
    )
    key_axis.grid(axis="y", color="0.88", linewidth=0.35)
    key_axis.grid(axis="x", visible=False)
    for x_value, tail_value, whole_value in zip(
        key_x, key_tail, key_whole, strict=True
    ):
        key_axis.vlines(
            x_value,
            0.18,
            max(tail_value, whole_value) + 0.01,
            color="0.82",
            linewidth=0.35,
            zorder=0,
        )
    key_axis.plot(
        key_x,
        key_whole,
        color=metric_colours["whole_test"],
        linewidth=COMPACT_KEY_LINE_WIDTH_PT,
        linestyle="-",
        marker="s",
        markersize=COMPACT_KEY_MARKER_SIZE_PT,
        markerfacecolor=metric_colours["whole_test"],
        markeredgecolor="white",
        markeredgewidth=0.35,
        solid_capstyle="round",
        zorder=2,
    )
    key_axis.plot(
        key_x,
        key_tail,
        color=metric_colours["tail"],
        linewidth=COMPACT_KEY_LINE_WIDTH_PT,
        linestyle="-",
        marker="o",
        markersize=COMPACT_KEY_MARKER_SIZE_PT,
        markerfacecolor=metric_colours["tail"],
        markeredgecolor="white",
        markeredgewidth=0.35,
        solid_capstyle="round",
        zorder=3,
    )
    key_axis.text(
        3.25,
        key_whole[-1],
        "Whole test",
        va="center",
        fontsize=COMPACT_KEY_TEXT_SIZE_PT,
    )
    key_axis.text(
        3.25,
        key_tail[-1],
        "Tail",
        va="center",
        fontsize=COMPACT_KEY_TEXT_SIZE_PT,
    )
    key_axis.tick_params(
        axis="x",
        length=0,
        pad=3,
        labelsize=COMPACT_KEY_X_TICK_SIZE_PT,
    )
    key_axis.tick_params(
        axis="y",
        direction="out",
        length=2.2,
        width=0.55,
        pad=2.2,
        labelsize=COMPACT_Y_TICK_SIZE_PT,
    )
    key_axis.spines["top"].set_visible(False)
    key_axis.spines["right"].set_visible(False)

    figure.text(
        0.017,
        0.52 if n_rows == 2 else 0.50,
        "Macro-F1",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=COMPACT_SHARED_LABEL_SIZE_PT,
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


def plot_tail_vs_whole_compact(
    summaries: list[ScoreSummary],
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    figure = build_tail_vs_whole_compact_figure(summaries)
    outputs = save_figure(
        figure,
        output_stem,
        dataset=DATASET_ORDER[0],
        formats=formats,
        dpi=dpi,
        metadata_title=(
            "RQ5 tail and whole-test macro-F1 by dataset, graph model, "
            "and preparation"
        ),
        tight=False,
    )
    plt.close(figure)
    return outputs


def write_summary_csv(
    path: Path,
    summaries: list[ScoreSummary],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
                "mean_tail_macro_f1",
                "sample_std_tail_macro_f1",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    DATASET_ORDER.index(row.dataset),
                    row.dataset,
                    DATASET_LABELS[row.dataset],
                    MODEL_ORDER.index(row.model),
                    row.model,
                    MODEL_LABELS[row.model].replace("\n", ""),
                    PREPARATION_ORDER.index(row.preparation),
                    row.preparation,
                    PREPARATION_LABELS[row.preparation],
                    row.n_runs,
                    f"{row.mean_tail_macro_f1:.10f}",
                    f"{row.sample_std_tail_macro_f1:.10f}",
                    f"{row.ci_lower:.10f}",
                    f"{row.ci_upper:.10f}",
                ]
            )


def write_tail_vs_whole_csv(
    path: Path,
    summaries: list[ScoreSummary],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
                "mean_tail_macro_f1",
                "mean_whole_test_macro_f1",
            ]
        )
        for row in summaries:
            if row.model not in PLOT_MODEL_ORDER:
                continue
            writer.writerow(
                [
                    DATASET_ORDER.index(row.dataset),
                    row.dataset,
                    DATASET_LABELS[row.dataset],
                    PLOT_MODEL_ORDER.index(row.model),
                    row.model,
                    MODEL_LABELS[row.model].replace("\n", ""),
                    PREPARATION_ORDER.index(row.preparation),
                    row.preparation,
                    PREPARATION_LABELS[row.preparation],
                    row.n_runs,
                    f"{row.mean_tail_macro_f1:.10f}",
                    f"{row.mean_whole_test_macro_f1:.10f}",
                ]
            )


def combined_run_digest(
    records: dict[str, dict[tuple[str, int], RunScore]],
) -> str:
    digest = hashlib.sha256()
    for preparation in PREPARATION_ORDER:
        for key, record in sorted(records[preparation].items()):
            digest.update(preparation.encode("utf-8"))
            digest.update(repr(key).encode("utf-8"))
            digest.update(relative_path(record.path).encode("utf-8"))
            digest.update(record.sha256.encode("ascii"))
    return digest.hexdigest()


def write_readme(path: Path) -> None:
    content = """# RQ5 absolute preparation scores

`preparation_tail_macro_f1.csv` reports mean tail macro-F1 and its 95%
percentile-bootstrap interval for every dataset, model, and preparation.
Each interval resamples the 30 complete split--seed runs.

`preparation_tail_vs_whole_macro_f1.csv` reports the tail and whole-test
macro-F1 means plotted in the comparison figures. It contains the six graph
models and omits MLP.

Regenerate both sets of PDF/PNG figures and their tables with:

```powershell
python scripts/analyze_rq5_preparations.py
```

The tail-only figures show one point for `A_erm`, `A_last`, `A_head`, and
`A_full` at each graph model. Preparation points share an x-coordinate within
a model. The dashed vertical segment joins the lowest and highest preparation
means.

The comparison figures place the four preparations from left to right within
each model and connect them with dashed visual guides. Blue filled circles
show tail macro-F1; rust open circles show whole-test macro-F1. Changes in
the orange series relative to `A_erm` show whole-test performance gained or
sacrificed. The vertical separation between the two series is not itself a
measure of sacrificed performance.

`rq5_tail_vs_whole_macro_f1_compact` combines the five datasets in a 2-by-3
layout. The top row contains the homophilic datasets and the key; the bottom
row contains the heterophilic datasets. It uses short model names, solid lines,
blue circles for tail macro-F1, and rust squares for whole-test macro-F1. Each
dataset panel uses its own y-axis range.

MLP remains in the tail summary table but is omitted from all figures because
it is not a graph model.

The five separate dataset figures prefer TeX Gyre Termes, with Times New
Roman, STIXGeneral, and DejaVu Serif as fallbacks. The compact main-text figure
uses Arial, STIX Sans mathematics, and the seaborn colourblind blue and orange
used by Figure 3. It is generated at its final 5.5-by-2.95-inch paper size.
"""
    if DATASET_ORDER != ORIGINAL_DATASET_ORDER:
        content = content.replace(
            "python scripts/analyze_rq5_preparations.py",
            (
                "python scripts/analyze_rq5_preparations.py "
                "--dataset-mode additional"
            ),
        ).replace(
            "combines the five datasets in a 2-by-3\n"
            "layout. The top row contains the homophilic datasets and the key; "
            "the bottom\nrow contains the heterophilic datasets.",
            "combines Actor and BlogCatalog with the key in a single row.",
        ).replace(
            "The five separate dataset figures",
            "The two separate dataset figures",
        ).replace(
            "5.5-by-2.95-inch",
            "5.5-by-1.65-inch",
        )
    path.write_text(content, encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    summaries: list[ScoreSummary],
    all_runs: dict[
        tuple[str, str],
        dict[str, dict[tuple[str, int], RunScore]],
    ],
    outputs: list[Path],
) -> None:
    payload = {
        "schema_version": 1,
        "analysis": (
            "rq5-absolute-tail-and-whole-test-macro-f1-by-preparation"
        ),
        "dataset_mode": args.dataset_mode.value,
        "analysis_version": ANALYSIS_VERSION,
        "datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "plotted_models": list(PLOT_MODEL_ORDER),
        "preparations": list(PREPARATION_ORDER),
        "definitions": {
            "scores": {
                "tail": (
                    "macro-F1 on the registered minority-pattern tail"
                ),
                "whole_test": "macro-F1 on the complete test set",
            },
            "cell_mean": "arithmetic mean over 30 split-seed runs",
            "tail_interval": {
                "method": (
                    "non-parametric percentile bootstrap over complete runs"
                ),
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "base_seed": args.bootstrap_seed,
                "seed_derivation": (
                    "SHA256(base seed, dataset, model, preparation, "
                    "absolute-tail-macro-f1)"
                ),
                "quantile_method": "linear",
            },
        },
        "score_summaries": [
            {
                "dataset": row.dataset,
                "model": row.model,
                "preparation": row.preparation,
                "n_runs": row.n_runs,
                "mean_tail_macro_f1": row.mean_tail_macro_f1,
                "mean_whole_test_macro_f1": (
                    row.mean_whole_test_macro_f1
                ),
                "tail_bootstrap_ci": [row.ci_lower, row.ci_upper],
            }
            for row in summaries
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_script": {
            "path": "scripts/analyze_rq5_preparations.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "input_run_sets": {
            f"{dataset}/{model}": {
                "preparations": len(PREPARATION_ORDER),
                "runs_per_preparation": EXPECTED_RUNS,
                "combined_sha256": combined_run_digest(records),
            }
            for (dataset, model), records in sorted(all_runs.items())
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
        "--dataset-mode",
        choices=(
            AnalysisDatasetMode.ORIGINAL.value,
            AnalysisDatasetMode.ADDITIONAL.value,
        ),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="directory containing corrected per-run JSON files",
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
    args.run_root = args.run_root or result_root / "runs"
    args.analysis_dir = args.analysis_dir or result_root / "rq5"
    args.figure_dir = args.figure_dir or result_root / "figures" / "rq5"
    return args


def main() -> None:
    args = parse_args()
    configure_dataset_mode(args.dataset_mode)
    if args.bootstrap_resamples < 100:
        raise ValueError("At least 100 bootstrap resamples are required")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")

    args.run_root = args.run_root.resolve()
    args.analysis_dir = args.analysis_dir.resolve()
    args.figure_dir = args.figure_dir.resolve()
    if not args.run_root.is_dir():
        raise FileNotFoundError(
            f"Corrected run directory not found: {args.run_root}"
        )

    summaries, all_runs = analyse(
        args.run_root,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.analysis_dir / "preparation_tail_macro_f1.csv"
    comparison_csv = (
        args.analysis_dir / "preparation_tail_vs_whole_macro_f1.csv"
    )
    readme = args.analysis_dir / "README.md"
    manifest = args.analysis_dir / "analysis_manifest.json"
    write_summary_csv(summary_csv, summaries)
    write_tail_vs_whole_csv(comparison_csv, summaries)
    write_readme(readme)

    figure_outputs: list[Path] = []
    for dataset in DATASET_ORDER:
        figure_outputs.extend(
            plot_dataset(
                dataset,
                summaries,
                args.figure_dir
                / f"rq5_{dataset}_tail_macro_f1_by_preparation",
                formats=tuple(args.formats),
                dpi=args.dpi,
            )
        )
        figure_outputs.extend(
            plot_tail_vs_whole_dataset(
                dataset,
                summaries,
                args.figure_dir
                / f"rq5_{dataset}_tail_vs_whole_macro_f1_by_preparation",
                formats=tuple(args.formats),
                dpi=args.dpi,
            )
        )

    figure_outputs.extend(
        plot_tail_vs_whole_compact(
            summaries,
            args.figure_dir / "rq5_tail_vs_whole_macro_f1_compact",
            formats=tuple(args.formats),
            dpi=args.dpi,
        )
    )

    outputs = [summary_csv, comparison_csv, readme, *figure_outputs]
    write_manifest(
        manifest,
        args,
        summaries,
        all_runs,
        outputs,
    )

    print(
        f"Analysed {len(summaries)} RQ5 preparation-score cells "
        f"and wrote {len(figure_outputs)} figure files."
    )
    for output in [*outputs, manifest]:
        print(relative_path(output))


if __name__ == "__main__":
    main()
