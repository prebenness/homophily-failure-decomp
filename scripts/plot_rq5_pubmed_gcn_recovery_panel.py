"""Plot the PubMed--GCN RQ5 recovery example used in Figure 1.

The figure reads the canonical RQ5 summary rather than duplicating values in
the plotting code. It shows mean tail and non-tail macro-F1 across the four
registered preparations.

Example:
    python scripts/plot_rq5_pubmed_gcn_recovery_panel.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from plot_rq5_tail_vs_non_tail import (
    NON_TAIL_COLOUR,
    PREPARATION_LABELS,
    PREPARATION_ORDER,
    TAIL_COLOUR,
    configure_style,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNS = 30
FIGURE_SIZE_IN = (3.20, 2.15)
OUTPUT_STEM = "rq5_pubmed_gcn_tail_vs_non_tail_macro_f1_panel"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_pubmed_gcn_means(summary_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["dataset"] == "pubmed" and row["model"] == "gcn"
        ]

    by_preparation = {row["preparation"]: row for row in rows}
    if set(by_preparation) != set(PREPARATION_ORDER):
        raise RuntimeError(
            "PubMed/GCN summary must contain exactly the four registered "
            f"preparations; found {sorted(by_preparation)}"
        )

    for preparation in PREPARATION_ORDER:
        n_runs = int(by_preparation[preparation]["n_runs"])
        if n_runs != EXPECTED_RUNS:
            raise RuntimeError(
                f"PubMed/GCN/{preparation}: found {n_runs} runs; "
                f"expected {EXPECTED_RUNS}"
            )

    tail = np.asarray(
        [
            float(by_preparation[preparation]["mean_tail_macro_f1"])
            for preparation in PREPARATION_ORDER
        ],
        dtype=np.float64,
    )
    non_tail = np.asarray(
        [
            float(by_preparation[preparation]["mean_non_tail_macro_f1"])
            for preparation in PREPARATION_ORDER
        ],
        dtype=np.float64,
    )
    if not (
        np.all(np.isfinite(tail))
        and np.all(np.isfinite(non_tail))
        and np.all((0.0 <= tail) & (tail <= 1.0))
        and np.all((0.0 <= non_tail) & (non_tail <= 1.0))
    ):
        raise RuntimeError("PubMed/GCN summary contains invalid macro-F1 values")
    return tail, non_tail


def build_figure(tail: np.ndarray, non_tail: np.ndarray) -> plt.Figure:
    configure_style()
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )

    x = np.arange(len(PREPARATION_ORDER), dtype=float)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE_IN)

    axis.set_xlim(-0.22, 3.64)
    axis.set_ylim(0.48, 0.96)
    axis.set_xticks(x)
    axis.set_xticklabels(
        [PREPARATION_LABELS[preparation] for preparation in PREPARATION_ORDER]
    )
    axis.set_yticks(np.arange(0.5, 1.0, 0.1))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.set_ylabel("Macro-F1")

    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.grid(axis="x", visible=False)
    for x_value, tail_value, non_tail_value in zip(
        x, tail, non_tail, strict=True
    ):
        axis.vlines(
            x_value,
            min(tail_value, non_tail_value),
            max(tail_value, non_tail_value),
            color="0.80",
            linewidth=0.55,
            zorder=1,
        )

    axis.plot(
        x,
        non_tail,
        color=NON_TAIL_COLOUR,
        linewidth=1.25,
        marker="s",
        markersize=5.0,
        markerfacecolor=NON_TAIL_COLOUR,
        markeredgecolor="white",
        markeredgewidth=0.55,
        solid_capstyle="round",
        zorder=2,
    )
    axis.plot(
        x,
        tail,
        color=TAIL_COLOUR,
        linewidth=1.25,
        marker="o",
        markersize=5.0,
        markerfacecolor=TAIL_COLOUR,
        markeredgecolor="white",
        markeredgewidth=0.55,
        solid_capstyle="round",
        zorder=3,
    )

    axis.text(
        3.13,
        non_tail[-1],
        "Non-tail",
        va="center",
        fontsize=8.2,
    )
    axis.text(
        3.13,
        tail[-1],
        "Tail",
        va="center",
        fontsize=8.2,
    )
    axis.tick_params(axis="x", length=0, pad=4)
    axis.tick_params(
        axis="y",
        direction="out",
        length=2.5,
        width=0.6,
        pad=2.5,
    )
    figure.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.19)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "main"
            / "rq5"
            / "preparation_tail_vs_non_tail_macro_f1.csv"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REPO_ROOT / "results" / "main" / "figures" / "rq5",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv.expanduser().resolve()
    figure_dir = args.figure_dir.expanduser().resolve()
    if not summary_csv.is_file():
        raise FileNotFoundError(f"RQ5 summary not found: {summary_csv}")
    if args.dpi < 72:
        raise ValueError("DPI must be at least 72")
    figure_dir.mkdir(parents=True, exist_ok=True)

    tail, non_tail = load_pubmed_gcn_means(summary_csv)
    figure = build_figure(tail, non_tail)
    output_stem = figure_dir / OUTPUT_STEM
    metadata = {
        "Title": "PubMed GCN tail and non-tail macro-F1 by preparation",
        "Author": "homophily-failure-decomp",
        "Subject": "RQ5 Figure 1 example panel",
    }
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(
        png_path,
        dpi=args.dpi,
        metadata=metadata,
        transparent=True,
    )
    figure.savefig(
        pdf_path,
        dpi=args.dpi,
        metadata={
            **metadata,
            "CreationDate": None,
            "ModDate": None,
        },
        transparent=True,
    )
    plt.close(figure)

    manifest_path = figure_dir / f"{OUTPUT_STEM}_analysis_manifest.json"
    outputs = (png_path, pdf_path)
    manifest = {
        "schema_version": 1,
        "analysis": "rq5-pubmed-gcn-recovery-example-panel",
        "definition": (
            "mean tail and non-tail macro-F1 across the 30 registered "
            "PubMed GCN split-seed runs for each preparation"
        ),
        "input": {
            "path": provenance_path(summary_csv),
            "sha256": file_sha256(summary_csv),
        },
        "outputs": [
            {
                "path": provenance_path(path),
                "sha256": file_sha256(path),
            }
            for path in outputs
        ],
        "values": {
            "preparations": list(PREPARATION_ORDER),
            "tail_macro_f1": tail.tolist(),
            "non_tail_macro_f1": non_tail.tolist(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "rendering": {"dpi": args.dpi, "transparent": True},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Tail means: {tail.tolist()}")
    print(f"Non-tail means: {non_tail.tolist()}")
    for path in (*outputs, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
