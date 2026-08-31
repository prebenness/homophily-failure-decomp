"""Calculate and plot aggregation-model tail-gap contrasts against MLP.

For each dataset and aggregation model, the analysis matches the corrected ERM
runs to the MLP runs by split and seed.  The run-level contrast is

    (model whole-test macro-F1 - model tail macro-F1)
    - (MLP whole-test macro-F1 - MLP tail macro-F1).

Positive values therefore mean that the aggregation model has a larger tail
gap than MLP on the same dataset.  The figure shows the mean contrast and its
paired-bootstrap 95% interval.

The script also repeats the comparison using the class-composition-adjusted
macro-F1 gaps produced by ``scripts/analyze_rq2_within_class.py``.  The
unadjusted and adjusted contrasts use identical split-seed pairs and bootstrap
resampling indices.  It retains the unadjusted component means used by the
compact main-text model comparison.

Example:
    python scripts/plot_rq3_mlp_contrasts.py
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
MODEL_ORDER = (
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
)
MODEL_LABELS = {
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "acm-gcn": "ACM-GCN",
}
DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20_260_717
EXPECTED_RUNS = 30
ANALYSIS_VERSION = "true-label-support-v1"


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel used by this script's analysis and plots."""

    global DATASET_ORDER
    DATASET_ORDER = dataset_order(mode)


@dataclass(frozen=True)
class RunGap:
    run_id: str
    split: str
    seed: int
    whole_macro_f1: float
    tail_macro_f1: float
    tail_gap: float
    path: Path
    sha256: str
    tail_class_ids: tuple[int, ...] | None = None
    whole_reference_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ContrastSummary:
    dataset: str
    model: str
    n_pairs: int
    mean_mlp_tail_gap: float
    mean_model_tail_gap: float
    mean_contrast: float
    sample_std_contrast: float
    ci_lower: float
    ci_upper: float
    smaller_tail_count: int
    p_value: float
    run_contrasts: tuple[float, ...]


@dataclass(frozen=True)
class ComponentSummary:
    dataset: str
    model: str
    n_pairs: int
    mean_mlp_whole_macro_f1: float
    mean_mlp_tail_macro_f1: float
    mean_model_whole_macro_f1: float
    mean_model_tail_macro_f1: float
    mean_mlp_tail_gap: float
    mean_model_tail_gap: float
    mean_gap_contrast: float


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


def validate_gap_components(run: RunGap, context: str) -> None:
    components = np.asarray([run.whole_macro_f1, run.tail_macro_f1])
    if not np.all(np.isfinite(components)):
        raise RuntimeError(f"{context}: macro-F1 components must be finite")
    if np.any(components < 0.0) or np.any(components > 1.0):
        raise RuntimeError(f"{context}: macro-F1 components must lie in [0, 1]")
    expected_gap = run.whole_macro_f1 - run.tail_macro_f1
    if not np.isclose(expected_gap, run.tail_gap, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            f"{context}: stored tail gap {run.tail_gap:.12g} does not equal "
            "whole-test macro-F1 minus tail macro-F1 "
            f"({expected_gap:.12g})"
        )


def load_run_gaps(run_root: Path, dataset: str, model: str) -> dict[tuple[str, int], RunGap]:
    model_dir = run_root / dataset / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {model_dir}")

    records: dict[tuple[str, int], RunGap] = {}
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

        split = str(payload["split"])
        seed = int(payload["seed"])
        key = (split, seed)
        if key in records:
            raise RuntimeError(f"Duplicate ERM split-seed pair for {dataset}/{model}: {key}")
        metrics = payload["metrics"]
        record = RunGap(
            run_id=str(payload["run_id"]),
            split=split,
            seed=seed,
            whole_macro_f1=float(metrics["whole_test"]["macro_f1"]),
            tail_macro_f1=float(metrics["tail"]["macro_f1"]),
            tail_gap=float(metrics["tail_gap"]),
            path=path,
            sha256=file_sha256(path),
        )
        validate_gap_components(record, f"{dataset}/{model}/{split}/seed-{seed}")
        records[key] = record

    if len(records) != EXPECTED_RUNS:
        raise RuntimeError(
            f"{dataset}/{model}: found {len(records)} corrected ERM runs; "
            f"expected {EXPECTED_RUNS}"
        )
    return records


def parse_semicolon_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(";") if item != "")


def parse_semicolon_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(";") if item != "")


def load_adjusted_run_gaps(
    path: Path,
    dataset: str,
    model: str,
) -> dict[tuple[str, int], RunGap]:
    """Load class-composition-adjusted ERM macro-F1 gaps from the RQ2 output."""
    if not path.is_file():
        raise FileNotFoundError(f"Adjusted run-metric table not found: {path}")

    table_sha256 = file_sha256(path)
    records: dict[tuple[str, int], RunGap] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "model",
            "preparation",
            "split",
            "seed",
            "tail_classes",
            "tail_class_ids",
            "whole_reference_weights",
            "whole_macro_f1",
            "adjusted_tail_macro_f1",
            "adjusted_macro_f1_gap",
        }
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise RuntimeError(
                f"{path}: missing adjusted-run columns {missing_columns}"
            )

        for row in reader:
            if row["preparation"] != "erm":
                continue
            if row["dataset"] != dataset or row["model"] != model:
                continue

            split = row["split"]
            seed = int(row["seed"])
            key = (split, seed)
            if key in records:
                raise RuntimeError(
                    f"Duplicate adjusted ERM split-seed pair for "
                    f"{dataset}/{model}: {key}"
                )

            class_ids = parse_semicolon_ints(row["tail_class_ids"])
            reference_weights = parse_semicolon_floats(
                row["whole_reference_weights"]
            )
            expected_classes = int(row["tail_classes"])
            if len(class_ids) != expected_classes:
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "tail-class count does not match the class IDs"
                )
            if len(reference_weights) != expected_classes:
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "tail-class count does not match the reference weights"
                )
            if expected_classes == 0:
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "the adjusted population is empty"
                )
            if not np.isclose(sum(reference_weights), 1.0, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{dataset}/{model}/{split}/seed-{seed}: "
                    "whole-test reference weights do not sum to one"
                )

            record = RunGap(
                run_id=f"{dataset}_{model}_erm_{split}_seed-{seed}",
                split=split,
                seed=seed,
                whole_macro_f1=float(row["whole_macro_f1"]),
                tail_macro_f1=float(row["adjusted_tail_macro_f1"]),
                tail_gap=float(row["adjusted_macro_f1_gap"]),
                path=path,
                sha256=table_sha256,
                tail_class_ids=class_ids,
                whole_reference_weights=reference_weights,
            )
            validate_gap_components(
                record,
                f"{dataset}/{model}/{split}/seed-{seed}/class-adjusted",
            )
            records[key] = record

    if len(records) != EXPECTED_RUNS:
        raise RuntimeError(
            f"{dataset}/{model}: found {len(records)} adjusted ERM runs; "
            f"expected {EXPECTED_RUNS}"
        )
    return records


def validate_adjustment_support(
    dataset: str,
    model: str,
    key: tuple[str, int],
    mlp_run: RunGap,
    model_run: RunGap,
) -> None:
    """Require model and MLP adjusted gaps to use the same target population."""
    mlp_adjusted = mlp_run.tail_class_ids is not None
    model_adjusted = model_run.tail_class_ids is not None
    if mlp_adjusted != model_adjusted:
        raise RuntimeError(
            f"{dataset}/{model}/{key}: adjusted and unadjusted records were mixed"
        )
    if not mlp_adjusted:
        return

    if mlp_run.tail_class_ids != model_run.tail_class_ids:
        raise RuntimeError(
            f"{dataset}/{model}/{key}: tail class IDs differ from MLP"
        )
    if not np.allclose(
        mlp_run.whole_reference_weights,
        model_run.whole_reference_weights,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{dataset}/{model}/{key}: whole-test class weights differ from MLP"
        )
def contrast_bootstrap_seed(base_seed: int, dataset: str, model: str) -> int:
    key = f"{base_seed}:{dataset}:{model}:rq3-mlp-tail-gap-contrast".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def paired_bootstrap(
    contrasts: np.ndarray,
    dataset: str,
    model: str,
    resamples: int,
    base_seed: int,
) -> tuple[float, float, int, float]:
    rng = np.random.default_rng(contrast_bootstrap_seed(base_seed, dataset, model))
    sample_indices = rng.integers(
        0, contrasts.size, size=(resamples, contrasts.size)
    )
    bootstrap_means = contrasts[sample_indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    smaller_tail_count = min(
        int(np.count_nonzero(bootstrap_means <= 0.0)),
        int(np.count_nonzero(bootstrap_means >= 0.0)),
    )
    p_value = min(1.0, 2.0 * smaller_tail_count / resamples)
    return float(lower), float(upper), smaller_tail_count, float(p_value)


def summarise_contrast(
    dataset: str,
    model: str,
    mlp_runs: dict[tuple[str, int], RunGap],
    model_runs: dict[tuple[str, int], RunGap],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> ContrastSummary:
    mlp_keys = set(mlp_runs)
    model_keys = set(model_runs)
    if mlp_keys != model_keys:
        missing = sorted(mlp_keys - model_keys)
        unexpected = sorted(model_keys - mlp_keys)
        raise RuntimeError(
            f"{dataset}/{model}: split-seed panel differs from MLP; "
            f"missing={missing}, unexpected={unexpected}"
        )

    keys = sorted(mlp_keys)
    for key in keys:
        validate_adjustment_support(
            dataset,
            model,
            key,
            mlp_runs[key],
            model_runs[key],
        )
    mlp_gaps = np.asarray([mlp_runs[key].tail_gap for key in keys])
    model_gaps = np.asarray([model_runs[key].tail_gap for key in keys])
    contrasts = model_gaps - mlp_gaps
    lower, upper, smaller_tail_count, p_value = paired_bootstrap(
        contrasts,
        dataset,
        model,
        bootstrap_resamples,
        bootstrap_seed,
    )
    return ContrastSummary(
        dataset=dataset,
        model=model,
        n_pairs=contrasts.size,
        mean_mlp_tail_gap=float(mlp_gaps.mean()),
        mean_model_tail_gap=float(model_gaps.mean()),
        mean_contrast=float(contrasts.mean()),
        sample_std_contrast=float(contrasts.std(ddof=1)),
        ci_lower=lower,
        ci_upper=upper,
        smaller_tail_count=smaller_tail_count,
        p_value=p_value,
        run_contrasts=tuple(float(value) for value in contrasts),
    )


def summarise_components(
    dataset: str,
    model: str,
    mlp_runs: dict[tuple[str, int], RunGap],
    model_runs: dict[tuple[str, int], RunGap],
) -> ComponentSummary:
    """Calculate the four endpoint means behind a model-versus-MLP contrast."""
    mlp_keys = set(mlp_runs)
    model_keys = set(model_runs)
    if mlp_keys != model_keys:
        missing = sorted(mlp_keys - model_keys)
        unexpected = sorted(model_keys - mlp_keys)
        raise RuntimeError(
            f"{dataset}/{model}: split-seed panel differs from MLP; "
            f"missing={missing}, unexpected={unexpected}"
        )

    keys = sorted(mlp_keys)
    for key in keys:
        mlp_run = mlp_runs[key]
        model_run = model_runs[key]
        validate_adjustment_support(dataset, model, key, mlp_run, model_run)
        validate_gap_components(mlp_run, f"{dataset}/mlp/{key}")
        validate_gap_components(model_run, f"{dataset}/{model}/{key}")

    mlp_whole = np.asarray([mlp_runs[key].whole_macro_f1 for key in keys])
    mlp_tail = np.asarray([mlp_runs[key].tail_macro_f1 for key in keys])
    model_whole = np.asarray([model_runs[key].whole_macro_f1 for key in keys])
    model_tail = np.asarray([model_runs[key].tail_macro_f1 for key in keys])
    mlp_gaps = mlp_whole - mlp_tail
    model_gaps = model_whole - model_tail
    return ComponentSummary(
        dataset=dataset,
        model=model,
        n_pairs=len(keys),
        mean_mlp_whole_macro_f1=float(mlp_whole.mean()),
        mean_mlp_tail_macro_f1=float(mlp_tail.mean()),
        mean_model_whole_macro_f1=float(model_whole.mean()),
        mean_model_tail_macro_f1=float(model_tail.mean()),
        mean_mlp_tail_gap=float(mlp_gaps.mean()),
        mean_model_tail_gap=float(model_gaps.mean()),
        mean_gap_contrast=float((model_gaps - mlp_gaps).mean()),
    )


def summarise_all_components(
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunGap]],
) -> list[ComponentSummary]:
    summaries = []
    for dataset in DATASET_ORDER:
        mlp_runs = all_runs[(dataset, "mlp")]
        for model in MODEL_ORDER:
            summaries.append(
                summarise_components(
                    dataset,
                    model,
                    mlp_runs,
                    all_runs[(dataset, model)],
                )
            )
    return summaries


def analyse(
    run_root: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[
    list[ContrastSummary],
    dict[tuple[str, str], dict[tuple[str, int], RunGap]],
]:
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunGap]] = {}
    summaries: list[ContrastSummary] = []
    for dataset in DATASET_ORDER:
        mlp_runs = load_run_gaps(run_root, dataset, "mlp")
        all_runs[(dataset, "mlp")] = mlp_runs
        for model in MODEL_ORDER:
            model_runs = load_run_gaps(run_root, dataset, model)
            all_runs[(dataset, model)] = model_runs
            summaries.append(
                summarise_contrast(
                    dataset,
                    model,
                    mlp_runs,
                    model_runs,
                    bootstrap_resamples,
                    bootstrap_seed,
                )
            )
    return summaries, all_runs


def analyse_adjusted(
    adjusted_run_metrics: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[
    list[ContrastSummary],
    dict[tuple[str, str], dict[tuple[str, int], RunGap]],
]:
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunGap]] = {}
    summaries: list[ContrastSummary] = []
    for dataset in DATASET_ORDER:
        mlp_runs = load_adjusted_run_gaps(
            adjusted_run_metrics, dataset, "mlp"
        )
        all_runs[(dataset, "mlp")] = mlp_runs
        for model in MODEL_ORDER:
            model_runs = load_adjusted_run_gaps(
                adjusted_run_metrics, dataset, model
            )
            all_runs[(dataset, model)] = model_runs
            summaries.append(
                summarise_contrast(
                    dataset,
                    model,
                    mlp_runs,
                    model_runs,
                    bootstrap_resamples,
                    bootstrap_seed,
                )
            )
    return summaries, all_runs


def configure_style() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        font_scale=1.0,
        rc={
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    palette = sns.color_palette("colorblind", n_colors=2)
    return palette[0], palette[1]


def panel_grid_spec() -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Return panel rows, separating adjacent dataset families by a small gap."""

    height_ratios: list[float] = []
    panel_rows: list[int] = []
    previous_group: str | None = None
    for dataset in DATASET_ORDER:
        group = DATASET_GROUPS[dataset]
        if previous_group is not None and group != previous_group:
            height_ratios.append(0.025)
        panel_rows.append(len(height_ratios))
        height_ratios.append(1.0)
        previous_group = group
    return tuple(height_ratios), tuple(panel_rows)


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    metadata_title: str = "Aggregation-model tail-gap contrasts against MLP",
    pad_inches: float = 0.14,
) -> list[Path]:
    outputs = []
    metadata = {
        "Title": metadata_title,
        "Author": "homophily-failure-decomp",
        "Subject": "RQ3",
    }
    pdf_metadata = {
        **metadata,
        "CreationDate": None,
        "ModDate": None,
    }
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad_inches,
            metadata=(
                pdf_metadata if output_format.lower() == "pdf" else metadata
            ),
        )
        outputs.append(path)
    return outputs


def plot_contrasts(
    summaries: list[ContrastSummary],
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    plot_title: str = "Tail-gap difference from MLP",
    metadata_title: str = "Aggregation-model tail-gap contrasts against MLP",
) -> list[Path]:
    positive_colour, negative_colour = configure_style()
    summary_by_cell = {(row.dataset, row.model): row for row in summaries}

    height_ratios, panel_rows = panel_grid_spec()
    figure_height = 8.5 if DATASET_ORDER == ORIGINAL_DATASET_ORDER else max(
        4.0, 1.48 * len(DATASET_ORDER) + 1.0
    )
    figure = plt.figure(figsize=(7.0, figure_height))
    grid = figure.add_gridspec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=0.29,
    )
    axes = [figure.add_subplot(grid[row]) for row in panel_rows]
    for row in sorted(set(range(len(height_ratios))) - set(panel_rows)):
        spacer_axis = figure.add_subplot(grid[row])
        spacer_axis.set_axis_off()

    x_limits = (-0.44, 0.36)
    x_ticks = np.arange(-0.4, 0.31, 0.1)
    y_positions = np.arange(len(MODEL_ORDER), dtype=float)
    model_labels = [MODEL_LABELS[model] for model in MODEL_ORDER]

    for dataset_index, (dataset, axis) in enumerate(zip(DATASET_ORDER, axes)):
        rows = [summary_by_cell[(dataset, model)] for model in MODEL_ORDER]
        means = np.asarray([row.mean_contrast for row in rows])
        lowers = np.asarray([row.ci_lower for row in rows])
        uppers = np.asarray([row.ci_upper for row in rows])
        positive = means >= 0.0

        for mask, colour in (
            (positive, positive_colour),
            (~positive, negative_colour),
        ):
            if not np.any(mask):
                continue
            axis.hlines(
                y_positions[mask],
                lowers[mask],
                uppers[mask],
                color=colour,
                linewidth=1.65,
                zorder=2,
            )
            cap_half_height = 0.15
            axis.vlines(
                lowers[mask],
                y_positions[mask] - cap_half_height,
                y_positions[mask] + cap_half_height,
                color=colour,
                linewidth=1.15,
                zorder=2,
            )
            axis.vlines(
                uppers[mask],
                y_positions[mask] - cap_half_height,
                y_positions[mask] + cap_half_height,
                color=colour,
                linewidth=1.15,
                zorder=2,
            )
            sns.scatterplot(
                x=means[mask],
                y=y_positions[mask],
                ax=axis,
                color=colour,
                s=20,
                edgecolor="white",
                linewidth=0.5,
                legend=False,
                zorder=3,
            )

        for y, mean, lower, upper in zip(y_positions, means, lowers, uppers):
            if mean >= 0.0:
                text_x = upper + 0.009
                alignment = "left"
            else:
                text_x = lower - 0.009
                alignment = "right"
            axis.text(
                text_x,
                y,
                f"{mean:+.3f}",
                ha=alignment,
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="0.24",
                zorder=4,
            )

        axis.axvline(0.0, color="0.34", linewidth=1.0, zorder=1)
        axis.set_xlim(*x_limits)
        axis.set_xticks(x_ticks)
        axis.set_ylim(-0.55, len(MODEL_ORDER) - 0.45)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(model_labels)
        axis.invert_yaxis()
        axis.set_ylabel("")
        axis.text(
            -0.265,
            1.07,
            DATASET_LABELS[dataset],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="x", color="0.82", linewidth=0.6)
        axis.grid(axis="y", visible=False)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

        is_last = dataset_index == len(DATASET_ORDER) - 1
        if is_last:
            axis.tick_params(axis="x", direction="out", length=3.5, pad=4)
            axis.set_xlabel("")
        else:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
            axis.set_xlabel("")
            axis.spines["bottom"].set_visible(False)

    figure.subplots_adjust(left=0.23, right=0.965, top=0.885, bottom=0.07)
    for axis in axes[:-1]:
        separator_y = axis.get_position().y0
        figure.add_artist(
            plt.Line2D(
                [0.035, 0.965],
                [separator_y, separator_y],
                transform=figure.transFigure,
                color="0.74",
                linewidth=0.75,
                zorder=0,
            )
        )
    zero_x = 0.23 + (0.0 - x_limits[0]) / (x_limits[1] - x_limits[0]) * (
        0.965 - 0.23
    )
    figure.text(
        0.5975,
        0.982,
        plot_title,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="0.18",
    )
    figure.text(
        0.23,
        0.944,
        "← Smaller tail gap than MLP",
        ha="left",
        va="top",
        fontsize=7.2,
        color="0.42",
    )
    figure.text(
        zero_x,
        0.922,
        "Same as MLP",
        ha="center",
        va="top",
        fontsize=6.8,
        color="0.42",
    )
    figure.text(
        0.965,
        0.944,
        "Larger tail gap than MLP →",
        ha="right",
        va="top",
        fontsize=7.2,
        color="0.42",
    )
    outputs = save_figure(
        figure,
        output_stem,
        formats,
        dpi,
        metadata_title=metadata_title,
    )
    plt.close(figure)
    return outputs


def write_contrast_csv(
    path: Path,
    summaries: list[ContrastSummary],
    gap_name: str = "tail_gap",
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_order",
                "dataset",
                "dataset_group",
                "model_order",
                "model",
                "model_label",
                "n_pairs",
                f"mean_mlp_{gap_name}",
                f"mean_model_{gap_name}",
                f"mean_{gap_name}_contrast",
                f"sample_std_{gap_name}_contrast",
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
                    MODEL_ORDER.index(row.model),
                    row.model,
                    MODEL_LABELS[row.model],
                    row.n_pairs,
                    f"{row.mean_mlp_tail_gap:.10f}",
                    f"{row.mean_model_tail_gap:.10f}",
                    f"{row.mean_contrast:.10f}",
                    f"{row.sample_std_contrast:.10f}",
                    f"{row.ci_lower:.10f}",
                    f"{row.ci_upper:.10f}",
                    row.smaller_tail_count,
                    f"{row.p_value:.10f}",
                    row.ci_lower > 0.0 or row.ci_upper < 0.0,
                ]
            )


def write_component_csv(
    path: Path,
    summaries: list[ComponentSummary],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_order",
                "dataset",
                "dataset_group",
                "model_order",
                "model",
                "model_label",
                "n_pairs",
                "mean_mlp_whole_macro_f1",
                "mean_mlp_tail_macro_f1",
                "mean_aggregation_whole_macro_f1",
                "mean_aggregation_tail_macro_f1",
                "mean_mlp_tail_gap",
                "mean_aggregation_tail_gap",
                "mean_tail_gap_contrast",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    DATASET_ORDER.index(row.dataset),
                    row.dataset,
                    DATASET_GROUPS[row.dataset],
                    MODEL_ORDER.index(row.model),
                    row.model,
                    MODEL_LABELS[row.model],
                    row.n_pairs,
                    f"{row.mean_mlp_whole_macro_f1:.10f}",
                    f"{row.mean_mlp_tail_macro_f1:.10f}",
                    f"{row.mean_model_whole_macro_f1:.10f}",
                    f"{row.mean_model_tail_macro_f1:.10f}",
                    f"{row.mean_mlp_tail_gap:.10f}",
                    f"{row.mean_model_tail_gap:.10f}",
                    f"{row.mean_gap_contrast:.10f}",
                ]
            )


def write_component_manifest(
    path: Path,
    summaries: list[ComponentSummary],
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunGap]],
    outputs: list[Path],
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> None:
    source_inputs = {
        "corrected_erm_runs": {
            f"{dataset}/{model}": {
                record.run_id: {
                    "path": relative_path(record.path),
                    "sha256": record.sha256,
                }
                for _, record in sorted(records.items())
            }
            for (dataset, model), records in all_runs.items()
        }
    }

    payload = {
        "schema_version": 1,
        "analysis": "rq3-aggregation-and-mlp-macro-f1-components",
        "dataset_mode": AnalysisDatasetMode(dataset_mode).value,
        "datasets": list(DATASET_ORDER),
        "aggregation_models": list(MODEL_ORDER),
        "reference_model": "mlp",
        "definitions": {
            "whole_component": "whole-test macro-F1",
            "tail_component": "tail macro-F1",
            "tail_gap": "whole component minus tail component",
            "gap_contrast": (
                "aggregation-model tail gap minus MLP tail gap"
            ),
            "pairing": "same dataset, split, and seed",
            "component_summary": (
                "arithmetic mean of each component over the 30 matched runs"
            ),
        },
        "cells": [
            {
                "dataset": row.dataset,
                "model": row.model,
                "n_pairs": row.n_pairs,
                "mean_mlp_whole_macro_f1": (
                    row.mean_mlp_whole_macro_f1
                ),
                "mean_mlp_tail_macro_f1": row.mean_mlp_tail_macro_f1,
                "mean_aggregation_whole_macro_f1": (
                    row.mean_model_whole_macro_f1
                ),
                "mean_aggregation_tail_macro_f1": (
                    row.mean_model_tail_macro_f1
                ),
                "mean_mlp_tail_gap": row.mean_mlp_tail_gap,
                "mean_aggregation_tail_gap": row.mean_model_tail_gap,
                "mean_gap_contrast": row.mean_gap_contrast,
            }
            for row in summaries
        ],
        "inputs": source_inputs,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_script": {
            "path": "scripts/plot_rq3_mlp_contrasts.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "outputs": [relative_path(output) for output in outputs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    summaries: list[ContrastSummary],
    all_runs: dict[tuple[str, str], dict[tuple[str, int], RunGap]],
    outputs: list[Path],
) -> None:
    input_runs = {}
    for (dataset, model), records in all_runs.items():
        input_runs[f"{dataset}/{model}"] = {
            record.run_id: {
                "path": relative_path(record.path),
                "sha256": record.sha256,
            }
            for _, record in sorted(records.items())
        }

    payload = {
        "schema_version": 1,
        "analysis": "rq3-aggregation-tail-gap-contrast-against-mlp",
        "dataset_mode": args.dataset_mode.value,
        "datasets": list(DATASET_ORDER),
        "aggregation_models": list(MODEL_ORDER),
        "reference_model": "mlp",
        "definitions": {
            "erm_tail_gap": "whole-test macro-F1 minus tail macro-F1",
            "contrast": "aggregation-model ERM tail gap minus MLP ERM tail gap",
            "pairing": "same dataset, split, and seed",
            "positive_contrast": "aggregation model has a larger tail gap than MLP",
            "interval": {
                "method": "paired non-parametric percentile bootstrap",
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "base_seed": args.bootstrap_seed,
                "quantile_method": "linear",
            },
        },
        "cells": [
            {
                "dataset": row.dataset,
                "model": row.model,
                "n_pairs": row.n_pairs,
                "mean_contrast": row.mean_contrast,
                "bootstrap_ci": [row.ci_lower, row.ci_upper],
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
            "path": "scripts/plot_rq3_mlp_contrasts.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "input_runs": input_runs,
        "outputs": [relative_path(output) for output in outputs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_adjusted_manifest(
    path: Path,
    args: argparse.Namespace,
    summaries: list[ContrastSummary],
    outputs: list[Path],
) -> None:
    payload = {
        "schema_version": 1,
        "analysis": (
            "rq3-class-composition-adjusted-aggregation-tail-gap-contrast-"
            "against-mlp"
        ),
        "dataset_mode": args.dataset_mode.value,
        "datasets": list(DATASET_ORDER),
        "aggregation_models": list(MODEL_ORDER),
        "reference_model": "mlp",
        "definitions": {
            "tail_class": (
                "true class with at least one tail test node in the run"
            ),
            "reference_class_distribution": (
                "whole-test true-class proportions renormalised over "
                "tail-present classes"
            ),
            "class_adjustment": (
                "tail conditional confusion rows multiplied by the whole-test "
                "reference class proportions"
            ),
            "class_adjusted_macro_f1_gap": (
                "ordinary whole-test macro-F1 over tail-present true classes "
                "minus class-adjusted tail macro-F1"
            ),
            "predictions_outside_tail_classes": (
                "retained as errors for tail-present true classes"
            ),
            "contrast": (
                "aggregation-model class-adjusted macro-F1 gap minus the MLP "
                "class-adjusted macro-F1 gap"
            ),
            "pairing": (
                "same dataset, split, seed, tail-present classes, and "
                "whole-test class weights"
            ),
            "positive_contrast": (
                "aggregation model has a larger class-adjusted tail gap than MLP"
            ),
            "interval": {
                "method": "paired non-parametric percentile bootstrap",
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "base_seed": args.bootstrap_seed,
                "seed_derivation": (
                    "the same per-dataset/model seed as the unadjusted RQ3 "
                    "contrast, so the two figures use identical resampling indices"
                ),
                "quantile_method": "linear",
            },
        },
        "cells": [
            {
                "dataset": row.dataset,
                "model": row.model,
                "n_pairs": row.n_pairs,
                "mean_mlp_class_adjusted_macro_f1_gap": (
                    row.mean_mlp_tail_gap
                ),
                "mean_model_class_adjusted_macro_f1_gap": (
                    row.mean_model_tail_gap
                ),
                "mean_contrast": row.mean_contrast,
                "bootstrap_ci": [row.ci_lower, row.ci_upper],
            }
            for row in summaries
        ],
        "inputs": {
            "class_adjusted_run_metrics": {
                "path": relative_path(args.class_adjusted_run_metrics),
                "sha256": file_sha256(args.class_adjusted_run_metrics),
                "column": "adjusted_macro_f1_gap",
            },
            "rq2_analysis_manifest": {
                "path": relative_path(args.rq2_analysis_manifest),
                "sha256": file_sha256(args.rq2_analysis_manifest),
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_script": {
            "path": "scripts/plot_rq3_mlp_contrasts.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "outputs": [relative_path(output) for output in outputs],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
        help="directory containing the corrected per-run JSON files",
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
        "--class-adjusted-run-metrics",
        type=Path,
        help="RQ2 run-level class-composition-adjusted metric table",
    )
    parser.add_argument(
        "--rq2-analysis-manifest",
        type=Path,
        help="provenance manifest for the class-adjusted run metrics",
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
    args.analysis_dir = args.analysis_dir or result_root / "rq3"
    args.figure_dir = args.figure_dir or result_root / "figures" / "rq3"
    args.class_adjusted_run_metrics = (
        args.class_adjusted_run_metrics
        or result_root / "rq2" / "class_adjusted_run_metrics.csv"
    )
    args.rq2_analysis_manifest = (
        args.rq2_analysis_manifest
        or result_root / "rq2" / "analysis_manifest.json"
    )
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
    args.class_adjusted_run_metrics = args.class_adjusted_run_metrics.resolve()
    args.rq2_analysis_manifest = args.rq2_analysis_manifest.resolve()
    if not args.run_root.is_dir():
        raise FileNotFoundError(f"Corrected run directory not found: {args.run_root}")
    if not args.class_adjusted_run_metrics.is_file():
        raise FileNotFoundError(
            "Class-adjusted run-metric table not found: "
            f"{args.class_adjusted_run_metrics}"
        )
    if not args.rq2_analysis_manifest.is_file():
        raise FileNotFoundError(
            f"RQ2 analysis manifest not found: {args.rq2_analysis_manifest}"
        )

    summaries, all_runs = analyse(
        args.run_root,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    adjusted_summaries, _ = analyse_adjusted(
        args.class_adjusted_run_metrics,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    component_summaries = summarise_all_components(all_runs)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    generated_outputs: list[Path] = []
    contrast_csv = args.analysis_dir / "mlp_tail_gap_contrasts.csv"
    manifest = args.analysis_dir / "analysis_manifest.json"
    figure_stem = args.figure_dir / "rq3_aggregation_vs_mlp_tail_gap_contrast"

    write_contrast_csv(contrast_csv, summaries)
    outputs = plot_contrasts(
        summaries,
        figure_stem,
        tuple(args.formats),
        args.dpi,
    )
    outputs.extend((contrast_csv, manifest))
    write_manifest(manifest, args, summaries, all_runs, outputs)
    generated_outputs.extend(outputs)

    adjusted_contrast_csv = (
        args.analysis_dir / "class_adjusted_mlp_tail_gap_contrasts.csv"
    )
    adjusted_manifest = args.analysis_dir / "class_adjusted_analysis_manifest.json"
    adjusted_figure_stem = (
        args.figure_dir
        / "rq3_aggregation_vs_mlp_class_adjusted_tail_gap_contrast"
    )
    write_contrast_csv(
        adjusted_contrast_csv,
        adjusted_summaries,
        gap_name="class_adjusted_macro_f1_gap",
    )
    adjusted_outputs = plot_contrasts(
        adjusted_summaries,
        adjusted_figure_stem,
        tuple(args.formats),
        args.dpi,
        plot_title="Class-composition-adjusted tail-gap difference from MLP",
        metadata_title=(
            "Class-composition-adjusted aggregation-model tail-gap "
            "contrasts against MLP"
        ),
    )
    adjusted_outputs.extend((adjusted_contrast_csv, adjusted_manifest))
    write_adjusted_manifest(
        adjusted_manifest,
        args,
        adjusted_summaries,
        adjusted_outputs,
    )
    generated_outputs.extend(adjusted_outputs)

    component_csv = args.analysis_dir / "mlp_tail_gap_components.csv"
    component_manifest = (
        args.analysis_dir / "components_analysis_manifest.json"
    )
    write_component_csv(component_csv, component_summaries)
    component_outputs = [component_csv, component_manifest]
    write_component_manifest(
        component_manifest,
        component_summaries,
        all_runs,
        component_outputs,
        dataset_mode=args.dataset_mode,
    )
    generated_outputs.extend(component_outputs)

    print(
        f"Analysed {len(summaries)} unadjusted and "
        f"{len(adjusted_summaries)} class-adjusted dataset-model contrasts."
    )
    for output in generated_outputs:
        print(relative_path(output))


if __name__ == "__main__":
    main()
