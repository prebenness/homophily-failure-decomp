#!/usr/bin/env python3
"""Plot the complete validated Appendix J edge-perturbation matrix."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402


P_PERCENTAGES = tuple(range(0, 101, 5))
ROMAN_PERCENTAGES = tuple(range(0, 71, 5))
CONDITIONS = ("treatment", "control")
MODELS = (
    ("gcn", "GCN"),
    ("graphsage-mean", "GraphSAGE-mean"),
    ("h2gcn", "H2GCN-2"),
    ("gpr-gnn", "GPR-GNN"),
    ("acm-gcn", "ACM-GCN"),
    ("linkx", "LINKX"),
)
DATASETS = (
    ("pubmed", "PubMed"),
    ("ogbn-arxiv", "ogbn-arxiv"),
    ("roman-empire", "Roman-empire"),
    ("amazon-ratings", "Amazon-ratings"),
    ("arxiv-year", "arxiv-year"),
)

DEFAULT_SUMMARY = (
    REPO_ROOT
    / "results"
    / "edge_perturbation_dose_response_full"
    / "dose_response_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "edge_perturbation_dose_response_composite"
)
DEFAULT_PAPER_OUTPUT = (
    REPO_ROOT / "paper" / "figures" / "edge_perturbation_dose_response.pdf"
)
OUTPUT_STEM = "edge_perturbation_dose_response"
MAIN_TEXT_STEM = "edge_perturbation_dose_response_gcn_graphsage"

STYLE = {
    "treatment": {
        "colour": "#DE8F05",
        "linestyle": "-",
        "marker": "o",
        "label": "Treatment",
    },
    "control": {
        "colour": "#0173B2",
        "linestyle": "--",
        "marker": "s",
        "label": "Control",
    },
}


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    model: str
    p_percent: float
    condition: str
    n_graph_replicates: int
    n_checkpoints_per_graph: int
    mean_macro_f1: float
    sd_macro_f1: float


def _register_arial_font() -> None:
    for candidate_text in font_manager.findSystemFonts(fontext="ttf"):
        candidate = Path(candidate_text)
        if candidate.stem.casefold() == "arial":
            font_manager.fontManager.addfont(candidate_text)
            return


def _expected_percentages(dataset: str) -> tuple[int, ...]:
    return ROMAN_PERCENTAGES if dataset == "roman-empire" else P_PERCENTAGES


def load_summary(path: Path = DEFAULT_SUMMARY) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    required = (
        "dataset",
        "model",
        "p_percent",
        "condition",
        "n_graph_replicates",
        "n_checkpoints_per_graph",
        "mean_macro_f1",
        "sd_macro_f1_across_graph_replicates",
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != required:
            raise ValueError(
                f"{path} must contain exactly the compact summary columns"
            )
        for raw in reader:
            rows.append(
                SummaryRow(
                    dataset=raw["dataset"],
                    model=raw["model"],
                    p_percent=int(raw["p_percent"]),
                    condition=raw["condition"],
                    n_graph_replicates=int(raw["n_graph_replicates"]),
                    n_checkpoints_per_graph=int(raw["n_checkpoints_per_graph"]),
                    mean_macro_f1=float(raw["mean_macro_f1"]),
                    sd_macro_f1=float(
                        raw["sd_macro_f1_across_graph_replicates"]
                    ),
                )
            )

    expected_keys = {
        (dataset, model, percentage, condition)
        for dataset, _ in DATASETS
        for model, _ in MODELS
        for percentage in _expected_percentages(dataset)
        for condition in CONDITIONS
    }
    observed_keys = {
        (row.dataset, row.model, row.p_percent, row.condition) for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(
            f"{path} does not contain the complete unique 6x5 dose-response matrix"
        )
    if any(row.n_graph_replicates != 5 for row in rows):
        raise ValueError(f"{path} does not report five graph replicates")
    if any(row.n_checkpoints_per_graph != 30 for row in rows):
        raise ValueError(f"{path} does not report 30 checkpoints per graph")
    if any(
        not math.isfinite(row.mean_macro_f1)
        or not 0.0 <= row.mean_macro_f1 <= 1.0
        for row in rows
    ):
        raise ValueError(f"{path} contains an invalid macro-F1 mean")
    if any(
        not math.isfinite(row.sd_macro_f1) or row.sd_macro_f1 < 0.0
        for row in rows
    ):
        raise ValueError(f"{path} contains an invalid standard deviation")
    return rows


def load_all_summaries(path: Path = DEFAULT_SUMMARY) -> list[SummaryRow]:
    """Compatibility wrapper for callers of the former loader."""

    return load_summary(path)


def _series(
    rows_by_cell: dict[tuple[str, str], list[SummaryRow]],
    dataset: str,
    model: str,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        (
            row
            for row in rows_by_cell[(dataset, model)]
            if row.condition == condition
        ),
        key=lambda row: row.p_percent,
    )
    return (
        np.asarray([row.p_percent for row in selected], dtype=float),
        np.asarray([row.mean_macro_f1 for row in selected], dtype=float),
        np.asarray([row.sd_macro_f1 for row in selected], dtype=float),
    )


def _cell_limits(rows: Sequence[SummaryRow]) -> tuple[float, float]:
    low = min(row.mean_macro_f1 - row.sd_macro_f1 for row in rows)
    high = max(row.mean_macro_f1 + row.sd_macro_f1 for row in rows)
    padding = max((high - low) * 0.07, 0.005)
    return low - padding, high + padding


def build_figure(
    rows: Sequence[SummaryRow],
    *,
    x_limits_by_dataset: Mapping[str, float] | None = None,
    xlabel: str = "Tail nodes rewired (%)",
    main_text: bool = False,
) -> plt.Figure:
    """Build the established portrait matrix, optionally with per-dataset doses.

    The default preserves the original shared 0--100 percent axes. Reverse
    experiments can supply their actual endpoints without rescaling doses to
    the eligible population. In that mode only axes in one dataset column
    share their x scale.
    """
    if x_limits_by_dataset is not None:
        if set(x_limits_by_dataset) != {dataset for dataset, _ in DATASETS}:
            raise ValueError("per-dataset x limits must cover exactly five datasets")
        if any(not math.isfinite(value) or value <= 0 or value > 100 for value in x_limits_by_dataset.values()):
            raise ValueError("per-dataset endpoints must lie in (0, 100]")
    models = MODELS[:2] if main_text else MODELS
    rows_by_cell: dict[tuple[str, str], list[SummaryRow]] = defaultdict(list)
    for row in rows:
        rows_by_cell[(row.dataset, row.model)].append(row)

    _register_arial_font()
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.sans-serif": [
                "Arial",
                "Arimo",
                "TeX Gyre Heros",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stixsans",
            "font.size": 5.5,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 4.6,
            "ytick.labelsize": 4.5,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.48,
            "axes.edgecolor": "0.18",
            "axes.labelcolor": "0.12",
            "text.color": "0.12",
            "xtick.color": "0.18",
            "ytick.color": "0.18",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 0.62,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(5.5, 2.3 if main_text else 6.5))
        grid = figure.add_gridspec(
            nrows=len(models),
            ncols=len(DATASETS),
            width_ratios=(1.0,) * len(DATASETS),
            left=0.102,
            right=0.992,
            bottom=0.250 if main_text else 0.070,
            top=0.865 if main_text else 0.890,
            hspace=0.13,
            wspace=0.28,
        )
        axes: dict[tuple[int, int], plt.Axes] = {}

        for row_index, (model, model_label) in enumerate(models):
            first_axis: plt.Axes | None = None
            for column_index, (dataset, dataset_label) in enumerate(DATASETS):
                axis = figure.add_subplot(
                    grid[row_index, column_index],
                    sharex=first_axis if x_limits_by_dataset is None else axes.get((0, column_index)),
                )
                if first_axis is None:
                    first_axis = axis
                axes[(row_index, column_index)] = axis
                cell_rows = rows_by_cell[(dataset, model)]
                endpoint = 100.0 if x_limits_by_dataset is None else x_limits_by_dataset[dataset]
                axis.set_xlim(0, endpoint)
                axis.set_ylim(*_cell_limits(cell_rows))
                if x_limits_by_dataset is None:
                    axis.set_xticks((0, 50, 100))
                else:
                    rounding_unit = 0.5 if endpoint < 10 else 5.0
                    middle = max(rounding_unit, round(endpoint / (2 * rounding_unit)) * rounding_unit)
                    if middle >= endpoint:
                        middle = endpoint / 2
                    axis.set_xticks((0, middle, endpoint))
                    axis.set_xticklabels(("0", f"{middle:g}", f"{endpoint:.2f}".rstrip("0").rstrip(".")))
                axis.yaxis.set_major_locator(
                    MaxNLocator(nbins=2, min_n_ticks=2, steps=(1, 2, 2.5, 5, 10))
                )
                axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                axis.grid(axis="y", color="0.88", linewidth=0.25)
                axis.grid(axis="x", visible=False)
                axis.set_axisbelow(True)
                axis.tick_params(
                    axis="both",
                    direction="out",
                    length=1.4,
                    width=0.48,
                    pad=0.8,
                )
                if row_index < len(models) - 1:
                    axis.tick_params(axis="x", labelbottom=False)
                else:
                    tick_labels = axis.get_xticklabels()
                    tick_labels[0].set_horizontalalignment("left")
                    tick_labels[-1].set_horizontalalignment("right")
                if row_index == 0:
                    axis.set_title(dataset_label, pad=8.0)

                for condition in CONDITIONS:
                    x, mean, sd = _series(
                        rows_by_cell, dataset, model, condition
                    )
                    style = STYLE[condition]
                    axis.fill_between(
                        x,
                        mean - sd,
                        mean + sd,
                        color=style["colour"],
                        alpha=0.14,
                        linewidth=0,
                    )
                    axis.plot(
                        x,
                        mean,
                        color=style["colour"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=1.25,
                        markeredgecolor="white",
                        markeredgewidth=0.14,
                        markevery=2,
                        solid_capstyle="round",
                    )

            first_position = axes[(row_index, 0)].get_position()
            figure.text(
                0.045,
                (first_position.y0 + first_position.y1) / 2,
                model_label,
                ha="center",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                fontsize=5.2,
            )

        handles = [
            Line2D(
                [0],
                [0],
                color=STYLE[condition]["colour"],
                linestyle=STYLE[condition]["linestyle"],
                marker=STYLE[condition]["marker"],
                markersize=2.6,
                markeredgecolor="white",
                markeredgewidth=0.2,
                label=STYLE[condition]["label"],
            )
            for condition in CONDITIONS
        ]
        figure.legend(
            handles=handles,
            loc="lower center" if main_text else "upper center",
            bbox_to_anchor=(0.547, 0.005 if main_text else 0.994),
            ncol=2,
            frameon=False,
            handlelength=2.1,
            columnspacing=1.5,
        )
        figure.supxlabel(xlabel, y=0.115 if main_text else 0.012, fontsize=6.3)
        figure.supylabel("Tail macro-F1", x=0.012, fontsize=6.3)
        return figure


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    *,
    paper_output: Path | None = None,
    stem: str = OUTPUT_STEM,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
    }
    figure.savefig(
        outputs["pdf"],
        metadata={
            "Title": "Full edge-perturbation dose-response matrix",
            "Author": "homophily-failure-decomp",
            "Creator": "scripts/plot_edge_dose_response_composite.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(outputs["png"], dpi=300)
    if paper_output is not None:
        paper_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(outputs["pdf"], paper_output)
        outputs["paper_pdf"] = paper_output
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-output", type=Path)
    parser.add_argument(
        "--main-text", action="store_true",
        help="plot the GCN/GraphSAGE two-row subset used in the main text",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_summary(args.summary.expanduser().resolve())
    figure = build_figure(rows, main_text=args.main_text)
    stem = MAIN_TEXT_STEM if args.main_text else OUTPUT_STEM
    paper_output = args.paper_output or (REPO_ROOT / "paper" / "figures" / f"{stem}.pdf")
    outputs = save_figure(
        figure,
        args.output_dir.expanduser().resolve(),
        paper_output=paper_output.expanduser().resolve(),
        stem=stem,
    )
    plt.close(figure)
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
