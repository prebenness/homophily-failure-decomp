"""Analyse RQ4 model-family tail-gap contrasts and absolute ERM scores.

RQ4 compares the heterophily-designed aggregation models H2GCN, GPR-GNN,
and ACM-GCN with the standard aggregation baselines GCN and GraphSAGE-mean.
For every dataset, designed model, and standard baseline, corrected ERM runs
are matched by split and seed before calculating

    designed-model tail gap - standard-baseline tail gap,

where a tail gap is whole-test macro-F1 minus tail macro-F1. Negative
contrasts therefore mean that the heterophily-designed model has the smaller
signed tail gap.

The script also summarises whole-test and tail macro-F1 separately for every
model. It writes the paired-contrast table used in the RQ4 appendix and a
Seaborn figure showing the absolute scores underlying those contrasts.

Example:
    python scripts/analyze_rq4_model_families.py
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
DATASET_GROUPS = {
    name: metadata.family for name, metadata in DATASET_REGISTRY.items()
}
STANDARD_MODELS = ("gcn", "graphsage-mean")
DESIGNED_MODELS = ("h2gcn", "gpr-gnn", "acm-gcn")
MODEL_ORDER = STANDARD_MODELS + DESIGNED_MODELS
MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "acm-gcn": "ACM-GCN",
}
MODEL_SHORT_LABELS = {
    "gcn": "GCN",
    "graphsage-mean": "SAGE",
    "h2gcn": "H2GCN",
    "gpr-gnn": "GPR",
    "acm-gcn": "ACM",
}
MODEL_FAMILIES = {
    "gcn": "standard",
    "graphsage-mean": "standard",
    "h2gcn": "heterophily-designed",
    "gpr-gnn": "heterophily-designed",
    "acm-gcn": "heterophily-designed",
}
FAMILY_COLOURS = {
    "standard": "#a65e3b",
    "heterophily-designed": "#2f6f8f",
}
SCORE_SCOPES = ("whole_test", "tail")
SCORE_SCOPE_LABELS = {
    "whole_test": "Whole test",
    "tail": "Tail",
}
SCOPE_MARKERS = {
    "whole_test": "D",
    "tail": "o",
}
EXPECTED_RUNS = 30
ANALYSIS_VERSION = "true-label-support-v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20_260_717


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel used by this script's analysis and outputs."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)


@dataclass(frozen=True)
class RunScore:
    run_id: str
    split: str
    seed: int
    whole_macro_f1: float
    tail_macro_f1: float
    tail_gap: float
    path: Path
    sha256: str


@dataclass(frozen=True)
class PairwiseContrastSummary:
    dataset: str
    designed_model: str
    standard_model: str
    n_pairs: int
    mean_designed_tail_gap: float
    mean_standard_tail_gap: float
    mean_contrast: float
    sample_std_contrast: float
    ci_lower: float
    ci_upper: float
    smaller_tail_count: int
    p_value: float
    run_contrasts: tuple[float, ...]


@dataclass(frozen=True)
class ScoreSummary:
    dataset: str
    model: str
    scope: str
    n_runs: int
    mean_macro_f1: float
    sample_std_macro_f1: float
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


def load_run_scores(
    run_root: Path,
    dataset: str,
    model: str,
) -> dict[tuple[str, int], RunScore]:
    model_dir = run_root / dataset / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {model_dir}")

    records: dict[tuple[str, int], RunScore] = {}
    for path in sorted(model_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("preparation") != "erm":
            continue
        if payload.get("dataset") != dataset:
            raise RuntimeError(f"Dataset mismatch in {path}")
        if payload.get("model") != model:
            raise RuntimeError(f"Model mismatch in {path}")
        if payload.get("analysis_version") != ANALYSIS_VERSION:
            raise RuntimeError(
                f"{path}: expected analysis_version={ANALYSIS_VERSION!r}"
            )

        metrics = payload["metrics"]
        whole_macro_f1 = float(metrics["whole_test"]["macro_f1"])
        tail_macro_f1 = float(metrics["tail"]["macro_f1"])
        tail_gap = float(metrics["tail_gap"])
        if not np.isclose(
            whole_macro_f1 - tail_macro_f1,
            tail_gap,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{path}: stored tail gap differs from whole-test minus tail "
                "macro-F1"
            )

        split = str(payload["split"])
        seed = int(payload["seed"])
        key = (split, seed)
        if key in records:
            raise RuntimeError(
                f"Duplicate ERM split-seed pair for {dataset}/{model}: {key}"
            )
        records[key] = RunScore(
            run_id=str(payload["run_id"]),
            split=split,
            seed=seed,
            whole_macro_f1=whole_macro_f1,
            tail_macro_f1=tail_macro_f1,
            tail_gap=tail_gap,
            path=path,
            sha256=file_sha256(path),
        )

    if len(records) != EXPECTED_RUNS:
        raise RuntimeError(
            f"{dataset}/{model}: found {len(records)} corrected ERM runs; "
            f"expected {EXPECTED_RUNS}"
        )
    return records


def bootstrap_seed(base_seed: int, *parts: str) -> int:
    key = ":".join((str(base_seed), *parts, "rq4-model-family-analysis")).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def percentile_bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float, np.ndarray]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap input must be a non-empty one-dimensional array")
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
    return float(lower), float(upper), bootstrap_means


def summarise_pairwise_contrast(
    dataset: str,
    designed_model: str,
    standard_model: str,
    designed_runs: dict[tuple[str, int], RunScore],
    standard_runs: dict[tuple[str, int], RunScore],
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> PairwiseContrastSummary:
    designed_keys = set(designed_runs)
    standard_keys = set(standard_runs)
    if designed_keys != standard_keys:
        missing = sorted(standard_keys - designed_keys)
        unexpected = sorted(designed_keys - standard_keys)
        raise RuntimeError(
            f"{dataset}/{designed_model}/{standard_model}: split-seed panels "
            f"differ; missing={missing}, unexpected={unexpected}"
        )

    keys = sorted(designed_keys)
    designed_gaps = np.asarray(
        [designed_runs[key].tail_gap for key in keys],
        dtype=np.float64,
    )
    standard_gaps = np.asarray(
        [standard_runs[key].tail_gap for key in keys],
        dtype=np.float64,
    )
    contrasts = designed_gaps - standard_gaps
    lower, upper, bootstrap_means = percentile_bootstrap_mean(
        contrasts,
        seed=bootstrap_seed(
            base_seed,
            dataset,
            designed_model,
            standard_model,
            "tail-gap-contrast",
        ),
        resamples=bootstrap_resamples,
    )
    smaller_tail_count = min(
        int(np.count_nonzero(bootstrap_means <= 0.0)),
        int(np.count_nonzero(bootstrap_means >= 0.0)),
    )
    p_value = min(
        1.0,
        2.0 * smaller_tail_count / bootstrap_resamples,
    )
    return PairwiseContrastSummary(
        dataset=dataset,
        designed_model=designed_model,
        standard_model=standard_model,
        n_pairs=contrasts.size,
        mean_designed_tail_gap=float(designed_gaps.mean()),
        mean_standard_tail_gap=float(standard_gaps.mean()),
        mean_contrast=float(contrasts.mean()),
        sample_std_contrast=float(contrasts.std(ddof=1)),
        ci_lower=lower,
        ci_upper=upper,
        smaller_tail_count=smaller_tail_count,
        p_value=float(p_value),
        run_contrasts=tuple(float(value) for value in contrasts),
    )


def score_values(
    records: dict[tuple[str, int], RunScore],
    scope: str,
) -> np.ndarray:
    if scope not in SCORE_SCOPES:
        raise ValueError(f"Unknown score scope: {scope}")
    keys = sorted(records)
    attribute = "whole_macro_f1" if scope == "whole_test" else "tail_macro_f1"
    return np.asarray(
        [getattr(records[key], attribute) for key in keys],
        dtype=np.float64,
    )


def summarise_scores(
    dataset: str,
    model: str,
    scope: str,
    records: dict[tuple[str, int], RunScore],
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> ScoreSummary:
    values = score_values(records, scope)
    lower, upper, _ = percentile_bootstrap_mean(
        values,
        seed=bootstrap_seed(
            base_seed,
            dataset,
            model,
            scope,
            "absolute-macro-f1",
        ),
        resamples=bootstrap_resamples,
    )
    return ScoreSummary(
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


def analyse(
    run_root: Path,
    *,
    bootstrap_resamples: int,
    base_seed: int,
) -> tuple[
    list[PairwiseContrastSummary],
    list[ScoreSummary],
    dict[tuple[str, str], dict[tuple[str, int], RunScore]],
]:
    all_runs: dict[
        tuple[str, str],
        dict[tuple[str, int], RunScore],
    ] = {}
    contrasts: list[PairwiseContrastSummary] = []
    score_summaries: list[ScoreSummary] = []

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            records = load_run_scores(run_root, dataset, model)
            all_runs[(dataset, model)] = records
            for scope in SCORE_SCOPES:
                score_summaries.append(
                    summarise_scores(
                        dataset,
                        model,
                        scope,
                        records,
                        bootstrap_resamples=bootstrap_resamples,
                        base_seed=base_seed,
                    )
                )

        for designed_model in DESIGNED_MODELS:
            for standard_model in STANDARD_MODELS:
                contrasts.append(
                    summarise_pairwise_contrast(
                        dataset,
                        designed_model,
                        standard_model,
                        all_runs[(dataset, designed_model)],
                        all_runs[(dataset, standard_model)],
                        bootstrap_resamples=bootstrap_resamples,
                        base_seed=base_seed,
                    )
                )

    return contrasts, score_summaries, all_runs


def configure_style() -> tuple[str, str]:
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
    return (
        FAMILY_COLOURS["standard"],
        FAMILY_COLOURS["heterophily-designed"],
    )


def model_offsets() -> dict[str, float]:
    offsets = np.linspace(-0.28, 0.28, len(MODEL_ORDER))
    return {
        model: float(offset)
        for model, offset in zip(MODEL_ORDER, offsets)
    }


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    metadata_title: str = "RQ4 whole-test and tail macro-F1 by model family",
) -> list[Path]:
    metadata = {
        "Title": metadata_title,
        "Author": "homophily-failure-decomp",
        "Subject": "RQ4",
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
            pad_inches=0.12,
            metadata=(
                pdf_metadata if output_format.lower() == "pdf" else metadata
            ),
        )
        outputs.append(path)
    return outputs


def plot_scores(
    summaries: list[ScoreSummary],
    output_stem: Path,
    *,
    formats: tuple[str, ...],
    dpi: int,
    y_label: str = "Mean macro-F1",
    metadata_title: str = "RQ4 whole-test and tail macro-F1 by model family",
) -> list[Path]:
    standard_colour, designed_colour = configure_style()
    by_cell: dict[tuple[str, str, str], ScoreSummary] = {}
    for row in summaries:
        key = (row.dataset, row.model, row.scope)
        if key in by_cell:
            raise RuntimeError(f"Duplicate RQ4 score cell: {key}")
        by_cell[key] = row
    expected = {
        (dataset, model, scope)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for scope in SCORE_SCOPES
    }
    missing = sorted(expected - set(by_cell))
    unexpected = sorted(set(by_cell) - expected)
    if missing or unexpected:
        raise RuntimeError(
            "RQ4 score panel is incomplete: "
            f"missing={missing}, unexpected={unexpected}"
        )

    figure_width = 7.2 if DATASET_ORDER == ORIGINAL_DATASET_ORDER else 5.5
    figure, axis = plt.subplots(figsize=(figure_width, 4.10))
    offsets = model_offsets()
    dataset_positions = np.arange(len(DATASET_ORDER), dtype=float)

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        for model in MODEL_ORDER:
            x = dataset_index + offsets[model]
            family = MODEL_FAMILIES[model]
            colour = FAMILY_COLOURS[family]
            whole = by_cell[(dataset, model, "whole_test")]
            tail = by_cell[(dataset, model, "tail")]

            axis.plot(
                [x, x],
                [whole.mean_macro_f1, tail.mean_macro_f1],
                color=colour,
                linewidth=0.85,
                linestyle="solid",
                solid_capstyle="round",
                zorder=1.5,
            )

            for scope, summary in (("whole_test", whole), ("tail", tail)):
                axis.plot(
                    x,
                    summary.mean_macro_f1,
                    marker=SCOPE_MARKERS[scope],
                    linestyle="None",
                    markersize=4.4 if scope == "whole_test" else 4.0,
                    markerfacecolor=colour,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    color=colour,
                    zorder=3,
                )

    axis.set_xlim(-0.55, len(DATASET_ORDER) - 0.45)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(dataset_positions)
    axis.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
    axis.set_yticks(np.arange(0.0, 1.01, 0.1))
    axis.set_ylabel(y_label)
    axis.set_xlabel("")
    axis.grid(axis="y", color="0.90", linewidth=0.5)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(axis="y", direction="out", length=2.5, width=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            color=standard_colour,
            markerfacecolor=standard_colour,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5,
            label="Standard models",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            color=designed_colour,
            markerfacecolor=designed_colour,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5,
            label="Heterophily-designed models",
        ),
    ]
    scope_handles = [
        Line2D(
            [0],
            [0],
            marker=SCOPE_MARKERS["whole_test"],
            linestyle="None",
            color="0.3",
            markerfacecolor="0.3",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.4,
            label="Whole test",
        ),
        Line2D(
            [0],
            [0],
            marker=SCOPE_MARKERS["tail"],
            linestyle="None",
            color="0.3",
            markerfacecolor="0.3",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.0,
            label="Tail",
        ),
    ]
    axis.legend(
        handles=family_handles + scope_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.35,
        borderaxespad=0.0,
        fontsize=7.25,
    )

    model_order_text = "  ·  ".join(
        MODEL_SHORT_LABELS[model]
        for model in MODEL_ORDER
    )
    figure.text(
        0.5,
        0.012,
        f"Within each dataset, left to right: {model_order_text}",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="0.38",
    )
    figure.subplots_adjust(left=0.095, right=0.985, top=0.84, bottom=0.14)
    outputs = save_figure(
        figure,
        output_stem,
        formats,
        dpi,
        metadata_title=metadata_title,
    )
    plt.close(figure)
    return outputs


def write_pairwise_csv(
    path: Path,
    summaries: list[PairwiseContrastSummary],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset_order",
                "dataset",
                "dataset_group",
                "designed_model_order",
                "designed_model",
                "designed_model_label",
                "standard_model_order",
                "standard_model",
                "standard_model_label",
                "n_pairs",
                "mean_designed_tail_gap",
                "mean_standard_tail_gap",
                "mean_tail_gap_contrast",
                "sample_std_tail_gap_contrast",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "bootstrap_smaller_tail_count",
                "bootstrap_p_value",
                "ci_excludes_zero",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    DATASET_ORDER.index(row.dataset),
                    row.dataset,
                    DATASET_GROUPS[row.dataset],
                    DESIGNED_MODELS.index(row.designed_model),
                    row.designed_model,
                    MODEL_LABELS[row.designed_model],
                    STANDARD_MODELS.index(row.standard_model),
                    row.standard_model,
                    MODEL_LABELS[row.standard_model],
                    row.n_pairs,
                    f"{row.mean_designed_tail_gap:.10f}",
                    f"{row.mean_standard_tail_gap:.10f}",
                    f"{row.mean_contrast:.10f}",
                    f"{row.sample_std_contrast:.10f}",
                    f"{row.ci_lower:.10f}",
                    f"{row.ci_upper:.10f}",
                    row.smaller_tail_count,
                    f"{row.p_value:.10f}",
                    row.ci_lower > 0.0 or row.ci_upper < 0.0,
                ]
            )


def write_score_csv(
    path: Path,
    summaries: list[ScoreSummary],
    *,
    model_order: tuple[str, ...] = MODEL_ORDER,
    model_labels: dict[str, str] = MODEL_LABELS,
    model_families: dict[str, str] = MODEL_FAMILIES,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset_order",
                "dataset",
                "dataset_group",
                "model_order",
                "model",
                "model_label",
                "model_family",
                "scope_order",
                "scope",
                "scope_label",
                "n_runs",
                "mean_macro_f1",
                "sample_std_macro_f1",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    DATASET_ORDER.index(row.dataset),
                    row.dataset,
                    DATASET_GROUPS[row.dataset],
                    model_order.index(row.model),
                    row.model,
                    model_labels[row.model],
                    model_families[row.model],
                    SCORE_SCOPES.index(row.scope),
                    row.scope,
                    SCORE_SCOPE_LABELS[row.scope],
                    row.n_runs,
                    f"{row.mean_macro_f1:.10f}",
                    f"{row.sample_std_macro_f1:.10f}",
                    f"{row.ci_lower:.10f}",
                    f"{row.ci_upper:.10f}",
                ]
            )


def latex_number(value: float) -> str:
    if abs(value) < 0.0005:
        formatted = "0.000"
    else:
        formatted = f"{value:+.3f}"
    if DATASET_ORDER != ORIGINAL_DATASET_ORDER:
        formatted = formatted.replace("+0.", "+.", 1).replace("-0.", "-.", 1)
        if formatted.startswith("0."):
            formatted = formatted[1:]
    return formatted


def latex_cell(row: PairwiseContrastSummary) -> str:
    macro = (
        "gaincell"
        if row.ci_lower > 0.0 or row.ci_upper < 0.0
        else "uncertaingaincell"
    )
    mean = latex_number(row.mean_contrast)
    interval = (
        f"[{latex_number(row.ci_lower)},{latex_number(row.ci_upper)}]"
    )
    return f"\\{macro}{{{mean}}}{{{interval}}}"


def render_appendix_table(
    summaries: list[PairwiseContrastSummary],
) -> str:
    by_cell = {
        (row.dataset, row.designed_model, row.standard_model): row
        for row in summaries
    }
    if DATASET_ORDER == ORIGINAL_DATASET_ORDER:
        label = "tab:rq4-designed-vs-standard"
        column_spec = "@{}llccccc@{}"
        header_lines = [
            "    & & \\multicolumn{2}{c}{Low-homophily tail}",
            r"    & \multicolumn{3}{c}{High-homophily tail} \\",
            "    \\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
            "    Designed model & Standard baseline & PubMed & ogbn-arxiv "
            r"& Roman-empire & Amazon-ratings & arxiv-year \\",
        ]
    else:
        label = "tab:additional-rq4-designed-vs-standard"
        column_spec = "@{}ll" + "c" * len(DATASET_ORDER) + "@{}"
        family = DATASET_GROUPS[DATASET_ORDER[0]]
        if any(DATASET_GROUPS[dataset] != family for dataset in DATASET_ORDER):
            raise ValueError(
                "Non-original RQ4 tables require one contiguous dataset family"
            )
        family_title = (
            "Low-homophily tail"
            if family == "homophilic"
            else "High-homophily tail"
        )
        end_column = 2 + len(DATASET_ORDER)
        header_lines = [
            "    & & "
            rf"\multicolumn{{{len(DATASET_ORDER)}}}{{c}}{{{family_title}}} \\",
            f"    \\cmidrule(lr){{3-{end_column}}}",
            "    Designed model & Standard baseline & "
            + " & ".join(DATASET_LABELS[dataset] for dataset in DATASET_ORDER)
            + r" \\",
        ]
    additional = DATASET_ORDER != ORIGINAL_DATASET_ORDER
    caption = (
        "  \\caption{\\textbf{RQ1c}. Macro-F1 tail-gap comparison between "
        "each heterophily-designed model and each standard aggregation "
        "baseline. Each cell is heterophily-model gap minus standard-baseline "
        "gap, with its 95\\% percentile-bootstrap interval over the 30 matched "
        "split--seed runs. Negative values indicate a smaller tail gap for "
        "the heterophily-designed model; grey cells contain zero and are not "
        "statistically significant.}"
        if additional
        else (
            "  \\caption{Paired ERM tail-gap contrast between each "
            "heterophily-designed model and each standard aggregation baseline. "
            "Each cell is designed-model gap minus standard-baseline gap, with "
            "its 95\\% percentile-bootstrap interval over 30 matched split--seed "
            "runs. Negative values indicate a smaller signed tail gap for the "
            "heterophily-designed model; grey cells contain zero.}"
        )
    )
    lines = [
        "\\begin{table}[H]",
        "  \\centering",
        caption,
        f"  \\label{{{label}}}",
        *([] if additional else ["  \\scriptsize"]),
        (
            "  \\setlength{\\tabcolsep}{5pt}"
            if additional
            else "  \\setlength{\\tabcolsep}{2.5pt}"
        ),
        f"  \\begin{{tabular}}{{{column_spec}}}",
        "    \\toprule",
        *header_lines,
        "    \\midrule",
    ]
    for designed_index, designed_model in enumerate(DESIGNED_MODELS):
        for standard_index, standard_model in enumerate(STANDARD_MODELS):
            if standard_index == 0:
                designed_label = (
                    f"\\multirow{{2}}{{*}}{{{MODEL_LABELS[designed_model]}}}"
                )
            else:
                designed_label = ""
            cells = [
                latex_cell(
                    by_cell[(dataset, designed_model, standard_model)]
                )
                for dataset in DATASET_ORDER
            ]
            lines.append(
                "    "
                + designed_label
                + " & "
                + MODEL_LABELS[standard_model]
                + " & "
                + " & ".join(cells)
                + " \\\\"
            )
        if designed_index < len(DESIGNED_MODELS) - 1:
            lines.append("    \\addlinespace[2pt]")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    contrasts: list[PairwiseContrastSummary],
    scores: list[ScoreSummary],
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunScore]],
    outputs: list[Path],
) -> None:
    input_runs = {}
    for (dataset, model), records in sorted(all_runs.items()):
        input_runs[f"{dataset}/{model}"] = {
            record.run_id: {
                "path": relative_path(record.path),
                "sha256": record.sha256,
            }
            for _, record in sorted(records.items())
        }

    payload = {
        "schema_version": 1,
        "analysis": "rq4-heterophily-designed-vs-standard-aggregation-models",
        "dataset_mode": args.dataset_mode.value,
        "datasets": list(DATASET_ORDER),
        "standard_models": list(STANDARD_MODELS),
        "heterophily_designed_models": list(DESIGNED_MODELS),
        "definitions": {
            "erm_tail_gap": "whole-test macro-F1 minus tail macro-F1",
            "pairwise_contrast": (
                "heterophily-designed-model ERM tail gap minus "
                "standard-baseline ERM tail gap"
            ),
            "negative_pairwise_contrast": (
                "heterophily-designed model has a smaller signed tail gap"
            ),
            "pairing": "same dataset, split, and seed",
            "score_summaries": (
                "separate arithmetic means of whole-test and tail macro-F1"
            ),
            "score_figure": {
                "whole_test_marker": "filled diamond",
                "tail_marker": "filled circle",
                "connector": "solid line for each dataset-model cell",
                "model_family_encoding": "colour",
                "confidence_intervals_shown": False,
                "y_axis_range": [0.0, 1.0],
            },
            "interval": {
                "method": "non-parametric percentile bootstrap over complete runs",
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "base_seed": args.bootstrap_seed,
                "quantile_method": "linear",
            },
        },
        "pairwise_contrasts": [
            {
                "dataset": row.dataset,
                "designed_model": row.designed_model,
                "standard_model": row.standard_model,
                "n_pairs": row.n_pairs,
                "mean_contrast": row.mean_contrast,
                "bootstrap_ci": [row.ci_lower, row.ci_upper],
            }
            for row in contrasts
        ],
        "score_summaries": [
            {
                "dataset": row.dataset,
                "model": row.model,
                "model_family": MODEL_FAMILIES[row.model],
                "scope": row.scope,
                "n_runs": row.n_runs,
                "mean_macro_f1": row.mean_macro_f1,
                "bootstrap_ci": [row.ci_lower, row.ci_upper],
            }
            for row in scores
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_script": {
            "path": "scripts/analyze_rq4_model_families.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "input_runs": input_runs,
        "outputs": [relative_path(output) for output in outputs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_readme(path: Path) -> None:
    content = """# RQ4 heterophily-designed versus standard aggregation models

`pairwise_tail_gap_contrasts.csv` contains the 30 direct RQ4 comparisons:
three heterophily-designed models by two standard baselines by five datasets.
Runs are matched by split and seed before calculating

```text
heterophily-designed-model tail gap - standard-baseline tail gap
```

Negative values mean that the heterophily-designed model has the smaller
signed ERM tail gap.

`erm_whole_tail_macro_f1.csv` reports the separate whole-test and tail
macro-F1 means and 95% bootstrap intervals. The RQ4 figure plots the means
without confidence intervals.
`appendix_table.tex` is the generated LaTeX table used in the paper appendix.

Regenerate every RQ4 output with:

```powershell
python scripts/analyze_rq4_model_families.py
```

The figure is written to `results/main/figures/rq4/` as PDF and PNG.
"""
    if DATASET_ORDER != ORIGINAL_DATASET_ORDER:
        content = content.replace(
            "the 30 direct RQ4 comparisons:\nthree heterophily-designed models "
            "by two standard baselines by five datasets.",
            f"the {len(DATASET_ORDER) * len(DESIGNED_MODELS) * len(STANDARD_MODELS)} "
            "direct RQ4 comparisons:\nthree heterophily-designed models by two "
            f"standard baselines by {len(DATASET_ORDER)} datasets.",
        ).replace(
            "python scripts/analyze_rq4_model_families.py",
            (
                "python scripts/analyze_rq4_model_families.py "
                "--dataset-mode additional"
            ),
        ).replace(
            "`results/main/figures/rq4/`",
            "`results/additional/figures/rq4/`",
        )
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(mode.value for mode in AnalysisDatasetMode),
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

    args.run_root = args.run_root.resolve()
    args.analysis_dir = args.analysis_dir.resolve()
    args.figure_dir = args.figure_dir.resolve()
    if not args.run_root.is_dir():
        raise FileNotFoundError(f"Corrected run directory not found: {args.run_root}")

    contrasts, scores, all_runs = analyse(
        args.run_root,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    pairwise_csv = args.analysis_dir / "pairwise_tail_gap_contrasts.csv"
    score_csv = args.analysis_dir / "erm_whole_tail_macro_f1.csv"
    appendix_table = args.analysis_dir / "appendix_table.tex"
    readme = args.analysis_dir / "README.md"
    manifest = args.analysis_dir / "analysis_manifest.json"
    figure_stem = (
        args.figure_dir / "rq4_whole_test_and_tail_macro_f1_by_model_family"
    )

    write_pairwise_csv(pairwise_csv, contrasts)
    write_score_csv(score_csv, scores)
    appendix_table.write_text(
        render_appendix_table(contrasts),
        encoding="utf-8",
    )
    write_readme(readme)
    figure_outputs = plot_scores(
        scores,
        figure_stem,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )

    outputs = [
        *figure_outputs,
        pairwise_csv,
        score_csv,
        appendix_table,
        readme,
        manifest,
    ]
    write_manifest(
        manifest,
        args,
        contrasts,
        scores,
        all_runs,
        outputs,
    )

    print(
        f"Analysed {len(contrasts)} pairwise RQ4 contrasts and "
        f"{len(scores)} absolute score cells."
    )
    for output in outputs:
        print(relative_path(output))


if __name__ == "__main__":
    main()
