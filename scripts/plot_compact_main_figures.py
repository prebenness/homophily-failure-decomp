"""Create the compact whole-test and tail macro-F1 model comparison.

The figure shows mean whole-test and tail macro-F1 for seven models:

    MLP, GCN, GraphSAGE-mean, H2GCN-2, GPR-GNN, ACM-GCN, LINKX.

Colour distinguishes MLP, standard aggregation models, and
heterophily-designed aggregation models; LINKX is shown separately as the
non-aggregation graph model.  The vertical line for each model is its signed
tail gap, while differences between lines show gap contrasts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

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
RQ3_COMPONENT_MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
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
MODEL_FAMILIES = {
    "mlp": "mlp",
    "gcn": "standard",
    "graphsage-mean": "standard",
    "h2gcn": "heterophily-designed",
    "gpr-gnn": "heterophily-designed",
    "acm-gcn": "heterophily-designed",
    "linkx": "non-aggregation",
}

# First three colours from the Seaborn colorblind palette used by Figure 4.
FAMILY_COLOURS = {
    "mlp": "#0173B2",
    "standard": "#DE8F05",
    "heterophily-designed": "#029E73",
    "non-aggregation": "#CC78BC",
}
OUTPUT_STEM = "rq3_rq4_whole_test_and_tail_macro_f1_by_model_family"


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel used by this script's loaders and plot."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Arimo",
                "TeX Gyre Heros",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "0.18",
            "axes.labelcolor": "0.12",
            "text.color": "0.12",
            "xtick.color": "0.18",
            "ytick.color": "0.18",
            "axes.unicode_minus": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def validate_cells(
    cells: dict[tuple[str, str], object],
    *,
    models: tuple[str, ...],
    context: str,
    datasets: tuple[str, ...] | None = None,
) -> None:
    datasets = DATASET_ORDER if datasets is None else datasets
    expected = {
        (dataset, model)
        for dataset in datasets
        for model in models
    }
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} is incomplete: missing={missing}, "
            f"unexpected={unexpected}"
        )


def load_whole_tail_scores(
    path: Path,
    *,
    datasets: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    datasets = DATASET_ORDER if datasets is None else datasets
    aggregation: dict[tuple[str, str], tuple[float, float]] = {}
    mlp_values: dict[str, list[tuple[float, float]]] = {
        dataset: [] for dataset in datasets
    }
    for row in read_rows(path):
        dataset = row["dataset"]
        model = row["model"]
        if dataset not in datasets:
            continue
        key = (dataset, model)
        if key in aggregation:
            raise RuntimeError(f"Duplicate component cell: {key}")
        aggregation[key] = (
            float(row["mean_aggregation_whole_macro_f1"]),
            float(row["mean_aggregation_tail_macro_f1"]),
        )
        mlp_values[dataset].append(
            (
                float(row["mean_mlp_whole_macro_f1"]),
                float(row["mean_mlp_tail_macro_f1"]),
            )
        )

    validate_cells(
        aggregation,
        models=RQ3_COMPONENT_MODEL_ORDER[1:],
        context="RQ3 component table",
        datasets=datasets,
    )

    cells = dict(aggregation)
    for dataset in datasets:
        values = np.asarray(mlp_values[dataset], dtype=float)
        if values.shape != (len(RQ3_COMPONENT_MODEL_ORDER) - 1, 2):
            raise RuntimeError(
                f"{dataset}: expected five repeated MLP component pairs; "
                f"found shape {values.shape}"
            )
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{dataset}: repeated MLP component means are inconsistent"
            )
        cells[(dataset, "mlp")] = tuple(float(value) for value in values[0])

    validate_cells(
        cells,
        models=RQ3_COMPONENT_MODEL_ORDER,
        context="Whole-test/tail score table",
        datasets=datasets,
    )
    return cells


def load_linkx_scores(
    path: Path,
    *,
    datasets: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    datasets = DATASET_ORDER if datasets is None else datasets
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    for row in read_rows(path):
        if row["model"] != "linkx" or row["preparation"] != "erm":
            continue
        dataset = row["dataset"]
        if dataset not in datasets:
            continue
        key = (dataset, "linkx")
        if key in cells:
            raise RuntimeError(f"Duplicate LINKX component cell: {key}")
        if int(row["n_runs"]) != 30:
            raise RuntimeError(
                f"{dataset}: expected 30 LINKX ERM runs; "
                f"found {row['n_runs']}"
            )
        cells[key] = (
            float(row["mean_whole_test_macro_f1"]),
            float(row["mean_tail_macro_f1"]),
        )
    validate_cells(
        cells,
        models=("linkx",),
        context="LINKX whole-test/tail score table",
        datasets=datasets,
    )
    return cells


def model_offsets(models: tuple[str, ...]) -> dict[str, float]:
    limit = 0.32 if models == MODEL_ORDER else 0.30
    offsets = np.linspace(-limit, limit, len(models))
    return {
        model: float(offset)
        for model, offset in zip(models, offsets)
    }


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0, pad=4.0)
    axis.tick_params(axis="y", direction="out", length=2.5, width=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
    metadata_title: str,
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": metadata_title,
        "Author": "homophily-failure-decomp",
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
            bbox_inches="tight",
            pad_inches=0.05,
            metadata=(
                pdf_metadata if output_format.lower() == "pdf" else metadata
            ),
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_merged_figure(
    cells: dict[tuple[str, str], tuple[float, float]],
    output_stem: Path,
    *,
    models: tuple[str, ...],
    formats: tuple[str, ...],
    dpi: int,
    datasets: tuple[str, ...] | None = None,
    y_upper: float = 1.0,
) -> list[Path]:
    datasets = DATASET_ORDER if datasets is None else datasets
    if y_upper < 1.0:
        raise ValueError("y_upper must include the full macro-F1 range")
    configure_style()
    figure, axis = plt.subplots(figsize=(5.5, 2.20))
    offsets = model_offsets(models)
    dataset_positions = np.arange(len(datasets), dtype=float)

    for dataset_index, dataset in enumerate(datasets):
        for model in models:
            whole, tail = cells[(dataset, model)]
            x = dataset_index + offsets[model]
            colour = FAMILY_COLOURS[MODEL_FAMILIES[model]]
            axis.plot(
                [x, x],
                [whole, tail],
                color=colour,
                linewidth=0.8,
                solid_capstyle="round",
                zorder=1.5,
            )
            axis.plot(
                x,
                whole,
                marker="D",
                linestyle="None",
                markersize=3.8,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=3,
            )
            axis.plot(
                x,
                tail,
                marker="o",
                linestyle="None",
                markersize=3.5,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=2.5,
            )

    axis.set_xlim(-0.52, len(datasets) - 0.48)
    axis.set_ylim(0.0, y_upper)
    axis.set_xticks(dataset_positions)
    axis.set_xticklabels(
        [DATASET_LABELS[dataset] for dataset in datasets]
    )
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_ylabel("Mean macro-F1")
    axis.set_xlabel("")
    style_axis(axis)
    family_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLOURS["mlp"],
            linewidth=1.2,
            label="MLP",
        ),
        Line2D(
            [0],
            [0],
            color=FAMILY_COLOURS["standard"],
            linewidth=1.2,
            label="Standard models",
        ),
        Line2D(
            [0],
            [0],
            color=FAMILY_COLOURS["heterophily-designed"],
            linewidth=1.2,
            label="Heterophily-designed",
        ),
    ]
    if "linkx" in models:
        family_handles.append(
            Line2D(
                [0],
                [0],
                color=FAMILY_COLOURS["non-aggregation"],
                linewidth=1.2,
                label="LINKX",
            )
        )
    scope_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="0.30",
            markerfacecolor="0.30",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=3.8,
            label="Whole test",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="0.30",
            markerfacecolor="0.30",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=3.5,
            label="Tail",
        ),
    ]
    axis.legend(
        handles=family_handles + scope_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(family_handles) + len(scope_handles),
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.90 if "linkx" in models else 0.85,
        borderaxespad=0.0,
        fontsize=6.2 if "linkx" in models else 6.7,
    )
    figure.subplots_adjust(left=0.115, right=0.995, top=0.85, bottom=0.22)
    return save_figure(
        figure,
        output_stem,
        formats=formats,
        dpi=dpi,
        metadata_title=(
            "Merged whole-test and tail macro-F1 model comparison"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(mode.value for mode in AnalysisDatasetMode),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--rq3-components",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument(
        "--linkx-scores",
        type=Path,
    )
    parser.add_argument("--output-name", default=OUTPUT_STEM)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    result_root = REPO_ROOT / "results" / (
        "main" if mode is AnalysisDatasetMode.ORIGINAL else mode.value
    )
    args.dataset_mode = mode
    args.rq3_components = (
        args.rq3_components or result_root / "rq3" / "mlp_tail_gap_components.csv"
    )
    args.output_dir = (
        args.output_dir or result_root / "figures" / "rq3-rq4"
    )
    args.linkx_scores = (
        args.linkx_scores
        or result_root / "rq5" / "preparation_tail_vs_whole_macro_f1.csv"
    )
    return args


def main() -> None:
    args = parse_args()
    configure_dataset_mode(args.dataset_mode)
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")

    whole_tail = load_whole_tail_scores(
        args.rq3_components.resolve(), datasets=DATASET_ORDER
    )
    whole_tail.update(
        load_linkx_scores(args.linkx_scores.resolve(), datasets=DATASET_ORDER)
    )
    validate_cells(
        whole_tail,
        models=MODEL_ORDER,
        context="Plotted whole-test/tail score table",
        datasets=DATASET_ORDER,
    )
    formats = tuple(args.formats)
    output_dir = args.output_dir.resolve()
    y_upper = (
        1.06
        if args.dataset_mode is AnalysisDatasetMode.ADDITIONAL
        else 1.0
    )
    outputs = plot_merged_figure(
        whole_tail,
        output_dir / args.output_name,
        models=MODEL_ORDER,
        formats=formats,
        dpi=args.dpi,
        datasets=DATASET_ORDER,
        y_upper=y_upper,
    )
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis": "rq3-rq4-compact-whole-test-and-tail-macro-f1",
        "dataset_mode": args.dataset_mode.value,
        "definition": (
            "arithmetic mean over the 30 corrected ERM split-seed runs; "
            "the vertical segment is whole-test macro-F1 minus tail macro-F1"
        ),
        "datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "input": {
            "path": provenance_path(args.rq3_components),
            "sha256": file_sha256(args.rq3_components.resolve()),
        },
        "linkx_input": {
            "path": provenance_path(args.linkx_scores),
            "sha256": file_sha256(args.linkx_scores.resolve()),
        },
        "outputs": [
            {
                "path": provenance_path(output),
                "sha256": file_sha256(output),
            }
            for output in outputs
        ],
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        "rendering": {
            "dpi": args.dpi,
            "formats": list(formats),
            "font_fallbacks": list(matplotlib.rcParams["font.sans-serif"]),
            "y_axis_range": [0.0, y_upper],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for output in [*outputs, manifest_path]:
        print(output)


if __name__ == "__main__":
    main()
