"""Generate the local-homophily-bin TikZ figure from dataset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hfd.analysis_datasets import (  # noqa: E402
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
)

DEFAULT_MANIFEST_DIR = REPO_ROOT / "manifests" / "datasets"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "main"
    / "figures"
    / "datasets"
    / "local_homophily_bin.tex"
)

DATASET_ORDER = (
    "pubmed",
    "ogbn-arxiv",
    "roman-empire",
    "amazon-ratings",
    "arxiv-year",
)
ADDITIONAL_DATASET_ORDER = dataset_order(AnalysisDatasetMode.ADDITIONAL)
DISPLAY_TITLES = {
    name: dataset.display_label for name, dataset in DATASET_REGISTRY.items()
}

HEADER = r"""\begin{figure}[t]
\centering
\usepgfplotslibrary{groupplots}
\begin{tikzpicture}
\begin{groupplot}[
    group style={
        group size=5 by 1,
        horizontal sep=0.25cm,
        ylabels at=edge left,
        yticklabels at=edge left
    },
    scale only axis,
    width=2.28cm,
    height=1.92cm,
    xmin=0, xmax=1,
    ymin=0, ymax=100,
    xtick={0,0.5,1},
    ytick={0,50,100},
    tick label style={font=\scriptsize},
    title style={font=\scriptsize},
    label style={font=\footnotesize},
    ymajorgrids=true,
    grid style={dotted,gray!40},
    axis on top,
]"""

FOOTER = r"""\end{groupplot}
\end{tikzpicture}
\caption{Distribution of nodes across local-homophily bins. Bar
widths show the bin intervals and heights show within-dataset percentages.
Dashed lines give mean \(h_v\); thick outlines mark the selected tails.}
\label{fig:local-homophily-bins-tikz}
\end{figure}"""

ADDITIONAL_HEADER = HEADER.replace(
    r"\begin{figure}[t]",
    r"\begin{figure}[H]",
).replace(
    "group size=5 by 1,",
    "group size=2 by 1,",
).replace(
    "width=2.28cm,",
    "width=5.15cm,",
).replace(
    "height=1.92cm,",
    "height=2.55cm,",
)

ADDITIONAL_FOOTER = FOOTER.replace(
    r"\label{fig:local-homophily-bins-tikz}",
    r"\label{fig:additional-local-homophily-bins-tikz}",
)


def interval_bounds(interval: str) -> tuple[float, float]:
    """Return the numeric endpoints of a registered interval."""

    try:
        lower, upper = interval[1:-1].split(",", maxsplit=1)
        return float(lower), float(upper)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid local-homophily interval: {interval!r}") from exc


def format_axis_value(value: float) -> str:
    return f"{value:g}"


def load_manifests(
    manifest_dir: Path,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> list[dict[str, Any]]:
    manifests = []
    for dataset in datasets:
        path = manifest_dir / f"{dataset}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != dataset:
            raise ValueError(
                f"{path} registers dataset {manifest.get('dataset')!r}, "
                f"expected {dataset!r}"
            )
        manifests.append(manifest)
    return manifests


def render_panel(
    manifest: dict[str, Any],
    *,
    ylabel_dataset: str = "pubmed",
    xlabel_dataset: str = "roman-empire",
) -> list[str]:
    dataset = manifest["dataset"]
    local_homophily = manifest["local_homophily"]
    mean_h = float(local_homophily["mean_h"])
    merged_bins = local_homophily["merged_bins"]
    tail_bins = [registered_bin for registered_bin in merged_bins if registered_bin["is_tail"]]
    if len(tail_bins) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one tail bin, found {len(tail_bins)}"
        )

    if dataset == ylabel_dataset:
        ylabel = r"    ylabel={Nodes (\%)}"
        if dataset == xlabel_dataset:
            ylabel += ","
        lines = [
            r"\nextgroupplot[",
            f"    title={{{DISPLAY_TITLES[dataset]}}},",
            ylabel,
        ]
        if dataset == xlabel_dataset:
            lines.append(r"    xlabel={Local homophily \(h_v\)},")
        lines.append("]")
    elif dataset == xlabel_dataset:
        lines = [
            r"\nextgroupplot[",
            f"    title={{{DISPLAY_TITLES[dataset]}}},",
            r"    xlabel={Local homophily \(h_v\)}",
            "]",
        ]
    else:
        lines = [rf"\nextgroupplot[title={{{DISPLAY_TITLES[dataset]}}}]"]

    lines.extend(
        [
            r"\addplot+[ybar interval, fill=blue!55, draw=none, mark=none]",
            "coordinates {",
        ]
    )
    for registered_bin in merged_bins:
        lower, _ = interval_bounds(registered_bin["interval"])
        percentage = 100.0 * float(registered_bin["fraction_of_eligible"])
        lines.append(
            f"    ({format_axis_value(lower)},{percentage:.1f})"
        )
    _, final_upper = interval_bounds(merged_bins[-1]["interval"])
    lines.extend(
        [
            f"    ({format_axis_value(final_upper)},0)",
            "};",
            (
                r"\addplot+[ybar interval, fill=blue!55, draw=black, "
                r"line width=1.2pt, mark=none]"
            ),
        ]
    )

    tail = tail_bins[0]
    tail_lower, tail_upper = interval_bounds(tail["interval"])
    tail_percentage = 100.0 * float(tail["fraction_of_eligible"])
    lines.extend(
        [
            "coordinates "
            f"{{({format_axis_value(tail_lower)},{tail_percentage:.1f}) "
            f"({format_axis_value(tail_upper)},0)}};",
            r"\draw[densely dashed,black,line width=0.7pt]",
            (
                f"    (axis cs:{mean_h:.3f},0) -- "
                f"(axis cs:{mean_h:.3f},100);"
            ),
        ]
    )

    anchor = "north east" if mean_h >= 0.5 else "north west"
    lines.extend(
        [
            (
                rf"\node[font=\scriptsize,anchor={anchor},"
                r"fill=white,inner sep=0.7pt]"
            ),
            f"    at (axis cs:{mean_h:.3f},98) {{{mean_h:.3f}}};",
        ]
    )
    return lines


def render_figure(
    manifests: list[dict[str, Any]],
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> str:
    if [manifest["dataset"] for manifest in manifests] != list(datasets):
        raise ValueError("Dataset manifests are not in the required panel order")

    additional = datasets == ADDITIONAL_DATASET_ORDER
    lines = (ADDITIONAL_HEADER if additional else HEADER).splitlines()
    lines.append("")
    for manifest in manifests:
        lines.extend(
            render_panel(
                manifest,
                ylabel_dataset=datasets[0],
                xlabel_dataset=(datasets[0] if additional else "roman-empire"),
            )
        )
        lines.append("")
    lines.extend((ADDITIONAL_FOOTER if additional else FOOTER).splitlines())
    return "\n".join(lines) + "\n"


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def generate_figure(
    manifest_dir: Path,
    output: Path,
    datasets: tuple[str, ...] = DATASET_ORDER,
) -> None:
    write_text_lf(
        output,
        render_figure(load_manifests(manifest_dir, datasets), datasets),
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
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    datasets = dataset_order(mode)
    manifest_dir = args.manifest_dir or (
        REPO_ROOT
        / (
            "manifests/datasets"
            if mode is AnalysisDatasetMode.ORIGINAL
            else "manifests/additional-datasets"
        )
    )
    output = args.output or (
        DEFAULT_OUTPUT
        if mode is AnalysisDatasetMode.ORIGINAL
        else REPO_ROOT
        / "results/additional/figures/datasets/local_homophily_bin.tex"
    )
    generate_figure(manifest_dir, output, datasets)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
