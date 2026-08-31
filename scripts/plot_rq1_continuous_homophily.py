"""Plot RQ1 performance and true-class margin against continuous h_v.

The script consumes the frozen node-level homophily arrays and a complete
dataset/model/preparation cell of saved ``run_data.npz`` files. It produces:

* mean test accuracy in fixed-width h_v bins, with a split-aware bootstrap
  interval and mean test support per run;
* one true-class-margin point per node, averaged over the runs in which that
  node belongs to the test split, with unique-node support per h_v bin;
* the derived data and a machine-readable analysis manifest.

Example:
    python scripts/plot_rq1_continuous_homophily.py \
        --raw-root PATH-TO-CANONICAL-RUNS \
        --dataset amazon-ratings --model gcn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


DEFAULT_BOOTSTRAP_SEED = 20260716
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_N_BINS = 20
INTERVAL_PATTERN = re.compile(
    r"^[\[(]\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*[\])]$"
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    split: str
    seed: int
    run_data_path: Path
    run_data_sha256: str


@dataclass(frozen=True)
class AnalysisData:
    edges: np.ndarray
    centres: np.ndarray
    run_accuracy: np.ndarray
    run_macro_f1: np.ndarray
    run_support: np.ndarray
    accuracy_units: np.ndarray
    accuracy_unit_definition: str
    mean_accuracy: np.ndarray
    accuracy_ci_low: np.ndarray
    accuracy_ci_high: np.ndarray
    macro_f1_units: np.ndarray
    macro_f1_unit_definition: str
    mean_macro_f1: np.ndarray
    macro_f1_ci_low: np.ndarray
    macro_f1_ci_high: np.ndarray
    mean_run_support: np.ndarray
    node_ids: np.ndarray
    node_h: np.ndarray
    node_margin: np.ndarray
    node_test_runs: np.ndarray
    node_support: np.ndarray
    mean_node_margin: np.ndarray


def array_sha256(array: np.ndarray) -> str:
    """Return the checksum convention used by the fixed-input manifests."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_interval(interval: str) -> tuple[float, float]:
    match = INTERVAL_PATTERN.match(interval)
    if match is None:
        raise ValueError(f"Cannot parse interval {interval!r}")
    low, high = (float(value) for value in match.groups())
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"Invalid h_v interval {interval!r}")
    return low, high


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign [left, right) bins, with h_v=1 included in the final bin."""
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, -1, dtype=np.int64)
    valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    result[valid] = np.searchsorted(edges, values[valid], side="right") - 1
    result[valid & (values == edges[-1])] = len(edges) - 2
    return result


def true_class_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return p(true class) minus the greatest probability of a wrong class."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Logits and labels have incompatible shapes")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("A label lies outside the logit class dimension")

    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    row = np.arange(labels.size)
    true_probability = probabilities[row, labels]
    probabilities[row, labels] = -np.inf
    best_wrong_probability = probabilities.max(axis=1)
    return true_probability - best_wrong_probability


def macro_f1_true_label_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-F1 over classes present in the subset's true labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels have incompatible shapes")
    if y_true.ndim != 1:
        raise ValueError("Macro-F1 inputs must be one-dimensional")
    labels = np.unique(y_true)
    if labels.size == 0:
        return float("nan")

    class_scores = []
    for label in labels:
        true_label = y_true == label
        predicted_label = y_pred == label
        true_positive = int(np.count_nonzero(true_label & predicted_label))
        false_positive = int(np.count_nonzero(~true_label & predicted_label))
        false_negative = int(np.count_nonzero(true_label & ~predicted_label))
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(class_scores))


def nanmean_preserving_empty(
    values: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Take a NaN-aware mean while retaining all-empty slices as NaN."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    totals = np.where(finite, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(totals), np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def bootstrap_mean_interval(
    observations: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval for column means after resampling independent units."""
    if observations.ndim != 2 or observations.shape[0] < 2:
        raise ValueError("Bootstrap input must contain at least two units")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    sampled_rows = rng.integers(
        0, observations.shape[0], size=(resamples, observations.shape[0])
    )
    bootstrap_means = nanmean_preserving_empty(
        observations[sampled_rows], axis=1
    )
    low = np.full(observations.shape[1], np.nan, dtype=np.float64)
    high = np.full(observations.shape[1], np.nan, dtype=np.float64)
    supported = np.any(np.isfinite(bootstrap_means), axis=0)
    if np.any(supported):
        low[supported], high[supported] = np.nanquantile(
            bootstrap_means[:, supported],
            [0.025, 0.975],
            axis=0,
            method="linear",
        )
    return low, high


def make_metric_units(
    run_values: np.ndarray,
    records: list[RunRecord],
    metric_name: str,
) -> tuple[np.ndarray, str]:
    """Use fixed splits as independent units when a split panel is available."""
    split_names = sorted({record.split for record in records})
    if len(split_names) == 1:
        return (
            run_values,
            "seed-level runs, conditional on the single fixed test split",
        )

    units = []
    seed_counts = []
    for split in split_names:
        rows = [index for index, record in enumerate(records) if record.split == split]
        units.append(nanmean_preserving_empty(run_values[rows], axis=0))
        seed_counts.append(len(rows))
    if len(set(seed_counts)) != 1:
        raise RuntimeError(
            f"The split-aware {metric_name} mean requires the same number of "
            "seeds per split"
        )
    return (
        np.vstack(units),
        f"{len(split_names)} split-level means, each averaging {seed_counts[0]} seed runs",
    )


def make_accuracy_units(
    run_accuracy: np.ndarray,
    records: list[RunRecord],
) -> tuple[np.ndarray, str]:
    return make_metric_units(run_accuracy, records, "accuracy")


def make_macro_f1_units(
    run_macro_f1: np.ndarray,
    records: list[RunRecord],
) -> tuple[np.ndarray, str]:
    return make_metric_units(run_macro_f1, records, "macro-F1")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(
    raw_root: Path,
    dataset: str,
    model: str,
    preparation: str,
    manifest: dict,
) -> list[RunRecord]:
    pattern = f"{dataset}_{model}_{preparation}_*_seed-*"
    run_dirs = sorted(path for path in raw_root.glob(pattern) if path.is_dir())
    expected_runs = int(manifest["n_runs"])
    if len(run_dirs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs matching {pattern!r} in {raw_root}, "
            f"found {len(run_dirs)}"
        )

    expected_checksums = manifest["array_checksums_sha256"]
    records: list[RunRecord] = []
    observed_cells: set[tuple[str, int]] = set()
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        run_data_path = run_dir / "run_data.npz"
        if not config_path.is_file() or not run_data_path.is_file():
            raise FileNotFoundError(f"Incomplete run directory: {run_dir}")
        config = load_json(config_path)
        expected_identity = (dataset, model, preparation)
        actual_identity = (
            config.get("dataset"),
            config.get("model"),
            config.get("preparation"),
        )
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"Run identity mismatch in {config_path}: {actual_identity!r}"
            )
        if config.get("run_id") != run_dir.name:
            raise RuntimeError(f"Run id mismatch in {config_path}")
        fixed_checksums = config.get("fixed_input_checksums", {})
        for key in ("h", "labels", "tail_mask"):
            if fixed_checksums.get(key) != expected_checksums[key]:
                raise RuntimeError(
                    f"{run_dir.name}: fixed-input checksum mismatch for {key}"
                )
        split = str(config["split"])
        seed = int(config["seed"])
        cell = (split, seed)
        if cell in observed_cells:
            raise RuntimeError(f"Duplicate split/seed cell: {cell}")
        observed_cells.add(cell)
        records.append(
            RunRecord(
                run_id=run_dir.name,
                split=split,
                seed=seed,
                run_data_path=run_data_path,
                run_data_sha256=file_sha256(run_data_path),
            )
        )
    return sorted(records, key=lambda record: (record.split, record.seed))


def validate_frozen_arrays(
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
) -> tuple[np.ndarray, np.ndarray]:
    for key in ("h", "labels"):
        if key not in arrays.files:
            raise KeyError(f"Frozen array file lacks {key!r}")
        actual = array_sha256(arrays[key])
        expected = manifest["array_checksums_sha256"][key]
        if actual != expected:
            raise RuntimeError(f"Frozen-array checksum mismatch for {key}")
    h = np.asarray(arrays["h"], dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    expected_nodes = int(manifest["graph"]["num_nodes"])
    if h.shape != (expected_nodes,) or labels.shape != (expected_nodes,):
        raise RuntimeError("Frozen arrays do not match the manifest node count")
    return h, labels


def analyse_runs(
    records: list[RunRecord],
    h: np.ndarray,
    frozen_labels: np.ndarray,
    manifest: dict,
    n_bins: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> AnalysisData:
    if n_bins < 2:
        raise ValueError("At least two h_v bins are required")
    # Division gives exact decimal boundaries such as 0.6 for 20 bins. Using
    # linspace produces 0.6000000000000001 and misclassifies nodes exactly on
    # that boundary.
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    centres = (edges[:-1] + edges[1:]) / 2.0
    run_accuracy = np.full((len(records), n_bins), np.nan, dtype=np.float64)
    run_macro_f1 = np.full((len(records), n_bins), np.nan, dtype=np.float64)
    run_support = np.zeros((len(records), n_bins), dtype=np.int64)
    margin_sum = np.zeros(h.size, dtype=np.float64)
    margin_count = np.zeros(h.size, dtype=np.int64)
    expected_checksums = manifest["array_checksums_sha256"]

    for run_index, record in enumerate(records):
        with np.load(record.run_data_path) as saved:
            required = {"logits", "predictions", "labels", "test_idx"}
            missing = required.difference(saved.files)
            if missing:
                raise KeyError(f"{record.run_id} lacks arrays: {sorted(missing)}")
            logits = np.asarray(saved["logits"])
            predictions = np.asarray(saved["predictions"], dtype=np.int64)
            labels = np.asarray(saved["labels"], dtype=np.int64)
            test_idx = np.asarray(saved["test_idx"], dtype=np.int64)

        if labels.shape != frozen_labels.shape or not np.array_equal(
            labels, frozen_labels
        ):
            raise RuntimeError(f"{record.run_id}: labels differ from frozen inputs")
        if array_sha256(labels) != expected_checksums["labels"]:
            raise RuntimeError(f"{record.run_id}: label checksum mismatch")
        split_checksum_key = f"{record.split}.test"
        if split_checksum_key not in expected_checksums:
            raise RuntimeError(f"Unknown split in {record.run_id}: {record.split}")
        if array_sha256(test_idx) != expected_checksums[split_checksum_key]:
            raise RuntimeError(f"{record.run_id}: test-index checksum mismatch")
        if logits.ndim != 2 or logits.shape[0] != h.size:
            raise RuntimeError(f"{record.run_id}: invalid logit shape {logits.shape}")
        if predictions.shape != h.shape:
            raise RuntimeError(f"{record.run_id}: invalid prediction shape")
        if not np.array_equal(predictions, logits.argmax(axis=1)):
            raise RuntimeError(f"{record.run_id}: predictions do not match logits")
        if np.unique(test_idx).size != test_idx.size:
            raise RuntimeError(f"{record.run_id}: duplicate test indices")
        if np.any(test_idx < 0) or np.any(test_idx >= h.size):
            raise RuntimeError(f"{record.run_id}: out-of-range test index")

        test_bins = assign_bins(h[test_idx], edges)
        test_labels = labels[test_idx]
        test_predictions = predictions[test_idx]
        correct = test_predictions == test_labels
        valid = test_bins >= 0
        support = np.bincount(test_bins[valid], minlength=n_bins)
        correct_count = np.bincount(
            test_bins[valid], weights=correct[valid].astype(np.float64), minlength=n_bins
        )
        run_support[run_index] = support
        np.divide(
            correct_count,
            support,
            out=run_accuracy[run_index],
            where=support > 0,
        )
        for bin_index in np.flatnonzero(support > 0):
            in_bin = valid & (test_bins == bin_index)
            run_macro_f1[run_index, bin_index] = macro_f1_true_label_support(
                test_labels[in_bin],
                test_predictions[in_bin],
            )

        margins = true_class_margin(logits[test_idx], labels[test_idx])
        np.add.at(margin_sum, test_idx, margins)
        np.add.at(margin_count, test_idx, 1)

    accuracy_units, accuracy_unit_definition = make_accuracy_units(
        run_accuracy, records
    )
    mean_accuracy = nanmean_preserving_empty(accuracy_units, axis=0)
    ci_low, ci_high = bootstrap_mean_interval(
        accuracy_units, bootstrap_resamples, bootstrap_seed
    )
    macro_f1_units, macro_f1_unit_definition = make_macro_f1_units(
        run_macro_f1, records
    )
    mean_macro_f1 = nanmean_preserving_empty(macro_f1_units, axis=0)
    macro_f1_ci_low, macro_f1_ci_high = bootstrap_mean_interval(
        macro_f1_units, bootstrap_resamples, bootstrap_seed
    )
    mean_run_support = run_support.mean(axis=0)

    node_ids = np.flatnonzero((margin_count > 0) & np.isfinite(h))
    node_h = h[node_ids]
    node_margin = margin_sum[node_ids] / margin_count[node_ids]
    node_test_runs = margin_count[node_ids]
    node_bins = assign_bins(node_h, edges)
    if np.any(node_bins < 0):
        raise RuntimeError("A plotted node lies outside the h_v bin range")
    node_support = np.bincount(node_bins, minlength=n_bins)
    margin_by_bin = np.bincount(
        node_bins, weights=node_margin, minlength=n_bins
    )
    mean_node_margin = np.divide(
        margin_by_bin,
        node_support,
        out=np.full(n_bins, np.nan, dtype=np.float64),
        where=node_support > 0,
    )

    return AnalysisData(
        edges=edges,
        centres=centres,
        run_accuracy=run_accuracy,
        run_macro_f1=run_macro_f1,
        run_support=run_support,
        accuracy_units=accuracy_units,
        accuracy_unit_definition=accuracy_unit_definition,
        mean_accuracy=mean_accuracy,
        accuracy_ci_low=ci_low,
        accuracy_ci_high=ci_high,
        macro_f1_units=macro_f1_units,
        macro_f1_unit_definition=macro_f1_unit_definition,
        mean_macro_f1=mean_macro_f1,
        macro_f1_ci_low=macro_f1_ci_low,
        macro_f1_ci_high=macro_f1_ci_high,
        mean_run_support=mean_run_support,
        node_ids=node_ids,
        node_h=node_h,
        node_margin=node_margin,
        node_test_runs=node_test_runs,
        node_support=node_support,
        mean_node_margin=mean_node_margin,
    )


def configure_style() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1.15,
        rc={
            "axes.linewidth": 0.8,
            "axes.labelpad": 5.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    palette = sns.color_palette("colorblind")
    return palette[0], palette[1]


def shade_tail(
    axes: Iterable[plt.Axes],
    tail_interval: tuple[float, float],
    colour: tuple[float, float, float],
) -> None:
    low, high = tail_interval
    boundary = high if low == 0.0 else low
    for axis in axes:
        axis.axvspan(low, high, color=colour, alpha=0.10, linewidth=0, zorder=0)
        if 0.0 < boundary < 1.0:
            axis.axvline(
                boundary,
                color="0.45",
                linewidth=0.8,
                linestyle=(0, (3, 3)),
                zorder=1,
            )


def label_tail(
    axis: plt.Axes,
    tail_interval: tuple[float, float],
) -> None:
    low, high = tail_interval
    x = low + 0.012 if low > 0.0 else high - 0.012
    alignment = "left" if low > 0.0 else "right"
    axis.text(
        x,
        0.975,
        "Registered tail",
        transform=axis.get_xaxis_transform(),
        ha=alignment,
        va="top",
        fontsize=8.5,
        color="0.30",
    )


def format_common_axes(main_axis: plt.Axes, support_axis: plt.Axes) -> None:
    main_axis.set_xlim(0.0, 1.0)
    support_axis.set_xlim(0.0, 1.0)
    support_axis.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    support_axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    support_axis.set_xlabel(r"Local homophily $h_v$")
    main_axis.tick_params(axis="x", which="both", labelbottom=False, length=0)
    support_axis.tick_params(axis="x", direction="out", length=3)
    support_axis.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    support_axis.grid(False)
    sns.despine(ax=main_axis)
    sns.despine(ax=support_axis)


def plot_accuracy(
    data: AnalysisData,
    tail_interval: tuple[float, float],
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    series_colour, tail_colour = configure_style()
    figure, (axis, support_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 4.6),
        gridspec_kw={"height_ratios": (4.0, 1.05), "hspace": 0.08},
    )
    shade_tail((axis, support_axis), tail_interval, tail_colour)
    axis.fill_between(
        data.centres,
        data.accuracy_ci_low,
        data.accuracy_ci_high,
        color=series_colour,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    sns.lineplot(
        x=data.centres,
        y=data.mean_accuracy,
        ax=axis,
        color=series_colour,
        linewidth=1.8,
        marker="o",
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.65,
        sort=False,
        zorder=3,
    )
    axis.set_ylim(0.0, 1.02)
    axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axis.set_ylabel("Mean test accuracy")
    axis.grid(axis="y", color="0.88", linewidth=0.55)
    label_tail(axis, tail_interval)

    bar_width = 0.82 * (data.edges[1] - data.edges[0])
    support_axis.bar(
        data.centres,
        data.mean_run_support,
        width=bar_width,
        color="0.52",
        alpha=0.60,
        linewidth=0,
        zorder=2,
    )
    support_axis.set_ylim(0.0, float(data.mean_run_support.max()) * 1.12)
    support_axis.set_ylabel("Mean test\nnodes per run", fontsize=8.5)
    format_common_axes(axis, support_axis)
    figure.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.125)

    title = "Test accuracy by continuous local homophily"
    outputs = save_figure(figure, output_stem, formats, dpi, title)
    plt.close(figure)
    return outputs


def plot_margin(
    data: AnalysisData,
    tail_interval: tuple[float, float],
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    series_colour, tail_colour = configure_style()
    figure, (axis, support_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.0, 4.6),
        gridspec_kw={"height_ratios": (4.0, 1.05), "hspace": 0.08},
    )
    shade_tail((axis, support_axis), tail_interval, tail_colour)
    sns.scatterplot(
        x=data.node_h,
        y=data.node_margin,
        ax=axis,
        s=8,
        color=series_colour,
        alpha=0.16,
        linewidth=0,
        edgecolor=None,
        rasterized=True,
        zorder=2,
    )
    axis.axhline(0.0, color="0.25", linewidth=0.9, zorder=3)
    axis.set_ylim(-1.02, 1.02)
    axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axis.set_ylabel("Mean true-class margin")
    axis.grid(axis="y", color="0.88", linewidth=0.55)
    label_tail(axis, tail_interval)

    bar_width = 0.82 * (data.edges[1] - data.edges[0])
    support_axis.bar(
        data.centres,
        data.node_support,
        width=bar_width,
        color="0.52",
        alpha=0.60,
        linewidth=0,
        zorder=2,
    )
    support_axis.set_ylim(0.0, float(data.node_support.max()) * 1.12)
    support_axis.set_ylabel("Unique test\nnodes", fontsize=8.5)
    format_common_axes(axis, support_axis)
    figure.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.125)

    title = "True-class margin by continuous local homophily"
    outputs = save_figure(figure, output_stem, formats, dpi, title)
    plt.close(figure)
    return outputs


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    formats: tuple[str, ...],
    dpi: int,
    title: str,
) -> list[Path]:
    outputs: list[Path] = []
    metadata = {
        "Title": title,
        "Author": "homophily-failure-decomp",
        "Subject": "RQ1 continuous local-homophily analysis",
        "CreationDate": None,
        "ModDate": None,
    }
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata=metadata,
        )
        outputs.append(path)
    return outputs


def write_binned_csv(path: Path, data: AnalysisData) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bin_left",
                "bin_right",
                "bin_centre",
                "mean_test_accuracy",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
                "mean_test_nodes_per_run",
                "runs_with_test_nodes",
                "bootstrap_units_with_test_nodes",
                "unique_test_nodes",
                "mean_node_margin",
            ]
        )
        for index in range(data.centres.size):
            writer.writerow(
                [
                    f"{data.edges[index]:.6f}",
                    f"{data.edges[index + 1]:.6f}",
                    f"{data.centres[index]:.6f}",
                    f"{data.mean_accuracy[index]:.10f}",
                    f"{data.accuracy_ci_low[index]:.10f}",
                    f"{data.accuracy_ci_high[index]:.10f}",
                    f"{data.mean_run_support[index]:.10f}",
                    int(np.isfinite(data.run_accuracy[:, index]).sum()),
                    int(np.isfinite(data.accuracy_units[:, index]).sum()),
                    int(data.node_support[index]),
                    f"{data.mean_node_margin[index]:.10f}",
                ]
            )


def write_node_csv(path: Path, data: AnalysisData) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "h_v", "mean_true_class_margin", "test_runs"])
        for node_id, h_value, margin, test_runs in zip(
            data.node_ids,
            data.node_h,
            data.node_margin,
            data.node_test_runs,
        ):
            writer.writerow(
                [
                    int(node_id),
                    f"{h_value:.10f}",
                    f"{margin:.10f}",
                    int(test_runs),
                ]
            )


def relative_output_paths(paths: Iterable[Path]) -> list[str]:
    result = []
    for path in paths:
        try:
            result.append(path.resolve().relative_to(REPO_ROOT).as_posix())
        except ValueError:
            result.append(str(path.resolve()))
    return result


def write_analysis_manifest(
    path: Path,
    args: argparse.Namespace,
    manifest: dict,
    records: list[RunRecord],
    data: AnalysisData,
    outputs: list[Path],
) -> None:
    payload = {
        "schema_version": 1,
        "analysis": "rq1-continuous-local-homophily",
        "dataset": args.dataset,
        "model": args.model,
        "preparation": args.preparation,
        "run_count": len(records),
        "run_ids": [record.run_id for record in records],
        "run_data_sha256": {
            record.run_id: record.run_data_sha256 for record in records
        },
        "fixed_input_checksums_sha256": {
            "h": manifest["array_checksums_sha256"]["h"],
            "labels": manifest["array_checksums_sha256"]["labels"],
        },
        "definitions": {
            "h_v_bins": {
                "count": args.bins,
                "range": [0.0, 1.0],
                "convention": "equal-width [left, right) intervals; final interval includes 1",
            },
            "accuracy": (
                "correct predictions divided by test nodes within each run and h_v bin; "
                "when multiple fixed splits exist, seed runs are first averaged within "
                "each split and the reported curve is the mean of those split-level means"
            ),
            "accuracy_interval": {
                "method": "two-sided percentile bootstrap of independent accuracy units",
                "level": 0.95,
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "quantile_method": "linear",
                "unit_count": int(data.accuracy_units.shape[0]),
                "units": data.accuracy_unit_definition,
            },
            "true_class_margin": "softmax p(y_v) - max_{c != y_v} softmax p(c)",
            "node_margin_aggregation": (
                "arithmetic mean over all selected runs in which the node belongs "
                "to the test split"
            ),
            "accuracy_support_bars": "mean test-node count per run and h_v bin",
            "margin_support_bars": (
                "unique nodes with at least one test prediction in each h_v bin"
            ),
            "registered_tail_interval": manifest["local_homophily"]["tail"][
                "interval"
            ],
        },
        "observations": {
            "nodes_with_test_predictions": int(data.node_ids.size),
            "minimum_test_runs_per_plotted_node": int(data.node_test_runs.min()),
            "maximum_test_runs_per_plotted_node": int(data.node_test_runs.max()),
            "mean_test_runs_per_plotted_node": float(data.node_test_runs.mean()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "analysis_script": {
            "path": "scripts/plot_rq1_continuous_homophily.py",
            "sha256": file_sha256(Path(__file__)),
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
    parser.add_argument("--model", default="gcn")
    parser.add_argument(
        "--preparation",
        default="erm",
        choices=("erm", "last", "head", "full"),
    )
    parser.add_argument(
        "--array-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "manifests",
        help="directory containing frozen <dataset>.npz arrays",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "manifests" / "datasets",
        help="directory containing tracked <dataset>.json manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "main"
            / "figures"
            / "rq1"
            / "exploratory"
            / "gcn-proof-of-concept"
        ),
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
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

    manifest = load_json(manifest_path)
    if manifest.get("dataset") != args.dataset:
        raise RuntimeError(f"Manifest dataset mismatch in {manifest_path}")
    with np.load(frozen_path) as frozen_arrays:
        h, frozen_labels = validate_frozen_arrays(frozen_arrays, manifest)
    records = discover_runs(
        args.raw_root, args.dataset, args.model, args.preparation, manifest
    )
    data = analyse_runs(
        records,
        h,
        frozen_labels,
        manifest,
        args.bins,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    tail_interval = parse_interval(manifest["local_homophily"]["tail"]["interval"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rq1_{args.dataset}_{args.model}_{args.preparation}"
    accuracy_stem = args.output_dir / f"{stem}_accuracy_by_local_homophily"
    margin_stem = args.output_dir / f"{stem}_margin_by_local_homophily"
    binned_csv = args.output_dir / f"{stem}_binned_data.csv"
    node_csv = args.output_dir / f"{stem}_node_margins.csv"
    analysis_manifest = args.output_dir / f"{stem}_analysis_manifest.json"

    outputs = []
    outputs.extend(
        plot_accuracy(data, tail_interval, accuracy_stem, tuple(args.formats), args.dpi)
    )
    outputs.extend(
        plot_margin(data, tail_interval, margin_stem, tuple(args.formats), args.dpi)
    )
    write_binned_csv(binned_csv, data)
    write_node_csv(node_csv, data)
    outputs.extend((binned_csv, node_csv, analysis_manifest))
    write_analysis_manifest(
        analysis_manifest, args, manifest, records, data, outputs
    )

    print(f"Analysed {len(records)} runs and {data.node_ids.size} test nodes.")
    for output in outputs:
        print(relative_output_paths([output])[0])


if __name__ == "__main__":
    main()
