"""Aggregate sequential edge-perturbation graph pairs.

The input layout is::

    <input-root>/<dataset>/pair-XXXX/
        metadata.json
        checkpoint_metrics.csv

Only completed actual-experiment pairs are accepted.  Each dataset/model/pair
cell must contain exactly 30 unique split/seed checkpoint effects.  The
script writes the concatenated checkpoint records, one unweighted mean per
graph pair, cumulative means over graph pairs, and a four-panel convergence
figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


PAIR_DIRECTORY_PATTERN = re.compile(r"pair-(\d+)$")
EFFECT_COLUMN = "treatment_minus_control_macro_f1"
REQUIRED_COLUMNS = (
    "dataset",
    "model",
    "pair_index",
    "split",
    "seed",
    EFFECT_COLUMN,
)
EXPECTED_CHECKPOINTS = 30
FIGURE_FILENAME = "running_convergence.png"

DATASETS = ("pubmed", "roman-empire")
MODELS = ("graphsage-mean", "acm-gcn")
SPLITS = {
    "pubmed": tuple(f"geomgcn-{index}" for index in range(10)),
    "roman-empire": tuple(f"platonov-{index}" for index in range(10)),
}
SEEDS = (1, 2, 3)
SAMPLE_FRACTION = 0.5
PAIR_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "predictions.npz",
    "checkpoint_metrics.csv",
)
REQUIRED_BOOLEAN_AUDITS = (
    "edge_count_preserved",
    "treatment_degrees_preserved",
    "control_degrees_preserved",
    "treatment_homophily_step_exact",
    "control_target_homophily_unchanged",
)
GRAPH_HASH_FIELDS = (
    "original_edge_sha256",
    "treatment_edge_sha256",
    "control_edge_sha256",
)
PREFERRED_DATASET_ORDER = DATASETS
PREFERRED_MODEL_ORDER = MODELS
DISPLAY_NAMES = {
    "pubmed": "PubMed",
    "roman-empire": "Roman-empire",
    "graphsage-mean": "GraphSAGE-mean",
    "acm-gcn": "ACM-GCN",
}


@dataclass(frozen=True)
class CheckpointRecord:
    """One validated checkpoint-level treatment-control comparison."""

    row: dict[str, str]
    dataset: str
    model: str
    pair_index: int
    split: str
    seed: int
    effect: float


@dataclass(frozen=True)
class PairProvenance:
    """Validated provenance for one completed graph pair."""

    dataset: str
    pair_index: int
    metadata_sha256: str
    output_sha256: dict[str, str]
    graph_sha256: dict[str, str]
    code_version: str
    design_id: str
    seed_base: int


def _ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    preferred_index = {value: index for index, value in enumerate(preferred)}
    return sorted(
        set(values),
        key=lambda value: (preferred_index.get(value, len(preferred)), value),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def discover_pair_directories(input_root: Path) -> list[tuple[str, int, Path]]:
    """Find direct ``dataset/pair-XXXX`` inputs in deterministic order."""

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")

    discovered: list[tuple[str, int, Path]] = []
    for dataset_dir in sorted(input_root.iterdir(), key=lambda path: path.name):
        if not dataset_dir.is_dir():
            continue
        for pair_dir in sorted(dataset_dir.iterdir(), key=lambda path: path.name):
            if not pair_dir.is_dir():
                continue
            match = PAIR_DIRECTORY_PATTERN.fullmatch(pair_dir.name)
            if match is None:
                continue
            if dataset_dir.name not in DATASETS:
                raise RuntimeError(
                    f"Unexpected dataset directory containing graph pairs: "
                    f"{dataset_dir.name!r}"
                )
            discovered.append(
                (dataset_dir.name, int(match.group(1)), pair_dir)
            )

    if not discovered:
        raise RuntimeError(
            f"No dataset/pair-XXXX directories found under {input_root}"
        )
    return discovered


def _validate_metadata(
    metadata: dict,
    *,
    path: Path,
    pair_dir: Path,
    dataset: str,
    pair_index: int,
    experiment_manifest_sha256: str,
) -> PairProvenance:
    phase = metadata.get("phase")
    if phase != "actual":
        raise RuntimeError(
            f"{path}: phase must be 'actual'; got {phase!r}. "
            "Debug inputs are not valid analysis inputs."
        )
    if metadata.get("completed") is not True:
        raise RuntimeError(f"{path}: completed must be true")

    if metadata.get("dataset") != dataset:
        raise RuntimeError(
            f"{path}: dataset {metadata.get('dataset')!r} does not match "
            f"directory {dataset!r}"
        )
    recorded_pair_index = metadata.get("pair_index")
    if (
        isinstance(recorded_pair_index, bool)
        or not isinstance(recorded_pair_index, int)
    ):
        raise RuntimeError(f"{path}: pair_index must be an integer")
    if recorded_pair_index != pair_index:
        raise RuntimeError(
            f"{path}: pair_index {recorded_pair_index} does not match "
            f"directory index {pair_index}"
        )

    if metadata.get("models") != list(MODELS):
        raise RuntimeError(
            f"{path}: models must be the registered panel {list(MODELS)}"
        )
    if metadata.get("checkpoints_per_model") != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{path}: checkpoints_per_model must be {EXPECTED_CHECKPOINTS}"
        )
    expected_rows = len(MODELS) * EXPECTED_CHECKPOINTS
    if metadata.get("n_checkpoint_rows") != expected_rows:
        raise RuntimeError(f"{path}: n_checkpoint_rows must be {expected_rows}")

    sample_fraction = metadata.get("sample_fraction")
    if (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or float(sample_fraction) != SAMPLE_FRACTION
    ):
        raise RuntimeError(
            f"{path}: sample_fraction must be {SAMPLE_FRACTION}"
        )
    if metadata.get("all_original_predictions_reproduced") is not True:
        raise RuntimeError(
            f"{path}: all_original_predictions_reproduced must be true"
        )
    if (
        metadata.get("experiment_manifest_sha256")
        != experiment_manifest_sha256
    ):
        raise RuntimeError(
            f"{path}: experiment_manifest_sha256 does not match "
            "input_root/experiment_manifest.json"
        )

    code_version = metadata.get("code_version")
    design_id = metadata.get("design_id")
    seed_base = metadata.get("seed_base")
    if not isinstance(code_version, str) or not code_version:
        raise RuntimeError(f"{path}: code_version must be a non-empty string")
    if not isinstance(design_id, str) or not design_id:
        raise RuntimeError(f"{path}: design_id must be a non-empty string")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int):
        raise RuntimeError(f"{path}: seed_base must be an integer")

    audit = metadata.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{path}: audit must be an object")
    missing_audits = [
        name for name in REQUIRED_BOOLEAN_AUDITS if name not in audit
    ]
    if missing_audits:
        raise RuntimeError(
            f"{path}: audit is missing {', '.join(missing_audits)}"
        )
    failed_audits = [
        name
        for name, value in audit.items()
        if isinstance(value, bool) and value is not True
    ]
    malformed_audits = [
        name for name in REQUIRED_BOOLEAN_AUDITS if audit[name] is not True
    ]
    if malformed_audits or failed_audits:
        failed = sorted(set(malformed_audits + failed_audits))
        raise RuntimeError(
            f"{path}: graph audits must all be true; failed={failed}"
        )

    graph_sha256 = {}
    for field in GRAPH_HASH_FIELDS:
        value = audit.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(
                f"{path}: audit.{field} must be a lowercase SHA256"
            )
        graph_sha256[field] = value
    if len(set(graph_sha256.values())) != len(graph_sha256):
        raise RuntimeError(
            f"{path}: original, treatment, and control graph hashes "
            "must be distinct"
        )

    recorded_output_sha256 = metadata.get("output_sha256")
    if not isinstance(recorded_output_sha256, dict):
        raise RuntimeError(f"{path}: output_sha256 must be an object")
    verified_output_sha256 = {}
    for filename in PAIR_OUTPUT_FILES:
        output_path = pair_dir / filename
        if not output_path.is_file():
            raise RuntimeError(f"{path}: required output is missing: {filename}")
        expected_sha256 = recorded_output_sha256.get(filename)
        observed_sha256 = file_sha256(output_path)
        if expected_sha256 != observed_sha256:
            raise RuntimeError(
                f"{path}: output_sha256 mismatch for {filename}"
            )
        verified_output_sha256[filename] = observed_sha256

    return PairProvenance(
        dataset=dataset,
        pair_index=pair_index,
        metadata_sha256=file_sha256(path),
        output_sha256=verified_output_sha256,
        graph_sha256=graph_sha256,
        code_version=code_version,
        design_id=design_id,
        seed_base=seed_base,
    )


def _read_checkpoint_csv(
    path: Path,
    *,
    dataset: str,
    pair_index: int,
) -> tuple[list[CheckpointRecord], list[str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing checkpoint metrics: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            missing = [
                column for column in REQUIRED_COLUMNS if column not in fieldnames
            ]
            if missing:
                raise RuntimeError(
                    f"{path}: missing required columns {', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    if not rows:
        raise RuntimeError(f"{path}: checkpoint metrics are empty")

    records: list[CheckpointRecord] = []
    for row_number, row in enumerate(rows, start=2):
        location = f"{path}:{row_number}"
        observed_dataset = row["dataset"].strip()
        if observed_dataset != dataset:
            raise RuntimeError(
                f"{location}: dataset {observed_dataset!r} does not match "
                f"directory {dataset!r}"
            )
        model = row["model"].strip()
        split = row["split"].strip()
        if not model or not split:
            raise RuntimeError(f"{location}: model and split must be non-empty")
        if model not in MODELS:
            raise RuntimeError(
                f"{location}: model must be one of {list(MODELS)}"
            )
        try:
            observed_pair_index = int(row["pair_index"])
        except ValueError as exc:
            raise RuntimeError(
                f"{location}: pair_index must be an integer"
            ) from exc
        if observed_pair_index != pair_index:
            raise RuntimeError(
                f"{location}: pair_index {observed_pair_index} does not match "
                f"directory index {pair_index}"
            )
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise RuntimeError(f"{location}: seed must be an integer") from exc
        try:
            effect = float(row[EFFECT_COLUMN])
        except ValueError as exc:
            raise RuntimeError(
                f"{location}: {EFFECT_COLUMN} must be numeric"
            ) from exc
        if not math.isfinite(effect):
            raise RuntimeError(
                f"{location}: {EFFECT_COLUMN} must be finite"
            )
        records.append(
            CheckpointRecord(
                row=dict(row),
                dataset=observed_dataset,
                model=model,
                pair_index=pair_index,
                split=split,
                seed=seed,
                effect=effect,
            )
        )
    return records, fieldnames


def load_records(
    input_root: Path,
) -> tuple[
    list[CheckpointRecord],
    list[str],
    list[PairProvenance],
    str,
]:
    """Load, validate, and combine all completed actual graph pairs."""

    experiment_manifest_path = input_root / "experiment_manifest.json"
    if not experiment_manifest_path.is_file():
        raise RuntimeError(
            f"Missing frozen experiment manifest: {experiment_manifest_path}"
        )
    _read_json(experiment_manifest_path)
    experiment_manifest_sha256 = file_sha256(experiment_manifest_path)

    all_records: list[CheckpointRecord] = []
    pair_provenance: list[PairProvenance] = []
    output_columns = list(REQUIRED_COLUMNS)
    seen_columns = set(output_columns)

    for dataset, pair_index, pair_dir in discover_pair_directories(input_root):
        metadata_path = pair_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing pair metadata: {metadata_path}")
        metadata = _read_json(metadata_path)
        pair_provenance.append(
            _validate_metadata(
                metadata,
                path=metadata_path,
                pair_dir=pair_dir,
                dataset=dataset,
                pair_index=pair_index,
                experiment_manifest_sha256=experiment_manifest_sha256,
            )
        )
        records, fieldnames = _read_checkpoint_csv(
            pair_dir / "checkpoint_metrics.csv",
            dataset=dataset,
            pair_index=pair_index,
        )
        all_records.extend(records)
        for column in fieldnames:
            if column not in seen_columns:
                output_columns.append(column)
                seen_columns.add(column)

    validate_checkpoint_panel(all_records)
    validate_pair_provenance(pair_provenance)
    return (
        all_records,
        output_columns,
        pair_provenance,
        experiment_manifest_sha256,
    )


def validate_pair_provenance(
    pairs: Sequence[PairProvenance],
) -> None:
    signatures = {
        (
            pair.code_version,
            pair.design_id,
            pair.seed_base,
        )
        for pair in pairs
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "code_version, design_id, and seed_base must be identical "
            "across all actual pairs"
        )

    original_by_dataset: dict[str, set[str]] = defaultdict(set)
    rewired_hash_owners: dict[str, str] = {}
    for pair in pairs:
        original_by_dataset[pair.dataset].add(
            pair.graph_sha256["original_edge_sha256"]
        )
        for condition in ("treatment", "control"):
            field = f"{condition}_edge_sha256"
            graph_hash = pair.graph_sha256[field]
            owner = (
                f"{pair.dataset}/pair-{pair.pair_index:04d}/{condition}"
            )
            previous = rewired_hash_owners.get(graph_hash)
            if previous is not None:
                raise RuntimeError(
                    f"Rewired graph hash is reused by {previous} and {owner}"
                )
            rewired_hash_owners[graph_hash] = owner
    inconsistent_originals = {
        dataset: sorted(hashes)
        for dataset, hashes in original_by_dataset.items()
        if len(hashes) != 1
    }
    if inconsistent_originals:
        raise RuntimeError(
            "Original graph hash changes across pairs: "
            f"{inconsistent_originals}"
        )


def validate_checkpoint_panel(records: Sequence[CheckpointRecord]) -> None:
    """Require 30 unique split/seed effects in every graph-pair cell."""

    grouped: dict[
        tuple[str, str, int], list[CheckpointRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.model, record.pair_index)].append(record)

    for (dataset, model, pair_index), cell in sorted(grouped.items()):
        identities = [(record.split, record.seed) for record in cell]
        unique_identities = set(identities)
        if len(cell) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(
                f"{dataset}/{model}/pair-{pair_index:04d}: expected "
                f"{EXPECTED_CHECKPOINTS} checkpoint rows, found {len(cell)}"
            )
        if len(unique_identities) != EXPECTED_CHECKPOINTS:
            duplicate_count = len(cell) - len(unique_identities)
            raise RuntimeError(
                f"{dataset}/{model}/pair-{pair_index:04d}: split/seed rows "
                f"must be unique; found {duplicate_count} duplicate(s)"
            )
        expected_identities = {
            (split, seed) for split in SPLITS[dataset] for seed in SEEDS
        }
        if unique_identities != expected_identities:
            missing = sorted(expected_identities - unique_identities)
            unexpected = sorted(unique_identities - expected_identities)
            raise RuntimeError(
                f"{dataset}/{model}/pair-{pair_index:04d}: checkpoint "
                f"identities differ from the registered 10x3 panel; "
                f"missing={missing}, unexpected={unexpected}"
            )

    observed_cells = {
        (record.dataset, record.model) for record in records
    }
    expected_cells = {
        (dataset, model) for dataset in DATASETS for model in MODELS
    }
    if observed_cells != expected_cells:
        raise RuntimeError(
            "The analysis requires the exact registered dataset/model cells; "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"unexpected={sorted(observed_cells - expected_cells)}"
        )

    pair_sets = {
        (dataset, model): {
            record.pair_index
            for record in records
            if record.dataset == dataset and record.model == model
        }
        for dataset in DATASETS
        for model in MODELS
    }
    unique_pair_sets = {
        frozenset(indices) for indices in pair_sets.values()
    }
    if len(unique_pair_sets) != 1:
        raise RuntimeError(
            "Graph-pair indices must be identical across all four "
            f"dataset/model cells: {pair_sets}"
        )
    pair_indices = sorted(next(iter(unique_pair_sets)))
    expected_pair_indices = list(range(1, pair_indices[-1] + 1))
    if pair_indices != expected_pair_indices:
        raise RuntimeError(
            "Graph-pair indices must be contiguous beginning at 1; "
            f"found {pair_indices}"
        )


def build_pair_summary(
    records: Sequence[CheckpointRecord],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, int], list[CheckpointRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.model, record.pair_index)].append(record)

    datasets = _ordered(
        (record.dataset for record in records), PREFERRED_DATASET_ORDER
    )
    models = _ordered(
        (record.model for record in records), PREFERRED_MODEL_ORDER
    )
    dataset_order = {value: index for index, value in enumerate(datasets)}
    model_order = {value: index for index, value in enumerate(models)}

    rows = []
    for (dataset, model, pair_index), cell in sorted(
        grouped.items(),
        key=lambda item: (
            dataset_order[item[0][0]],
            model_order[item[0][1]],
            item[0][2],
        ),
    ):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "pair_index": pair_index,
                "n_checkpoints": len(cell),
                "mean_treatment_minus_control_macro_f1": fmean(
                    record.effect for record in cell
                ),
            }
        )
    return rows


def build_running_summary(
    pair_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)

    datasets = _ordered(
        (str(row["dataset"]) for row in pair_rows), PREFERRED_DATASET_ORDER
    )
    models = _ordered(
        (str(row["model"]) for row in pair_rows), PREFERRED_MODEL_ORDER
    )
    rows = []
    for dataset in datasets:
        for model in models:
            cell = sorted(
                grouped[(dataset, model)],
                key=lambda row: int(row["pair_index"]),
            )
            cumulative_total = 0.0
            for n_pairs, pair_row in enumerate(cell, start=1):
                pair_mean = float(
                    pair_row["mean_treatment_minus_control_macro_f1"]
                )
                cumulative_total += pair_mean
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "pair_index": int(pair_row["pair_index"]),
                        "n_pairs": n_pairs,
                        "running_mean_treatment_minus_control_macro_f1": (
                            cumulative_total / n_pairs
                        ),
                    }
                )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_convergence_figure(
    pair_rows: Sequence[dict[str, object]],
    running_rows: Sequence[dict[str, object]],
):
    """Build one panel for each dataset/model combination."""

    datasets = _ordered(
        (str(row["dataset"]) for row in pair_rows), PREFERRED_DATASET_ORDER
    )
    models = _ordered(
        (str(row["model"]) for row in pair_rows), PREFERRED_MODEL_ORDER
    )
    if len(datasets) != 2 or len(models) != 2:
        raise RuntimeError(
            "A four-panel figure requires exactly two datasets and two models"
        )

    running_lookup = {
        (str(row["dataset"]), str(row["model"]), int(row["pair_index"])): float(
            row["running_mean_treatment_minus_control_macro_f1"]
        )
        for row in running_rows
    }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.0, 5.6),
        sharey=True,
        squeeze=False,
    )
    for row_index, dataset in enumerate(datasets):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            cell = sorted(
                (
                    row
                    for row in pair_rows
                    if row["dataset"] == dataset and row["model"] == model
                ),
                key=lambda row: int(row["pair_index"]),
            )
            pair_indices = [int(row["pair_index"]) for row in cell]
            pair_means = [
                float(row["mean_treatment_minus_control_macro_f1"])
                for row in cell
            ]
            running_means = [
                running_lookup[(dataset, model, pair_index)]
                for pair_index in pair_indices
            ]
            axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
            axis.scatter(
                pair_indices,
                pair_means,
                color="#777777",
                marker="o",
                s=22,
                label="Graph-pair mean",
                zorder=2,
            )
            axis.plot(
                pair_indices,
                running_means,
                color="#2f6f8f",
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label="Running mean",
                zorder=3,
            )
            axis.set_title(
                f"{DISPLAY_NAMES.get(dataset, dataset)} · "
                f"{DISPLAY_NAMES.get(model, model)}",
                fontsize=10,
            )
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis="y", color="#dddddd", linewidth=0.6)
            if row_index == 1:
                axis.set_xlabel("Graph-pair index")
            if column_index == 0:
                axis.set_ylabel("Treatment − control macro-F1")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def build_analysis_manifest(
    *,
    records: Sequence[CheckpointRecord],
    pairs: Sequence[PairProvenance],
    experiment_manifest_sha256: str,
    generated_outputs: dict[str, Path],
) -> dict:
    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (
            DATASETS.index(pair.dataset),
            pair.pair_index,
        ),
    )
    first = ordered_pairs[0]
    pair_indices = sorted({pair.pair_index for pair in ordered_pairs})
    return {
        "schema_version": 1,
        "analysis": "sequential-edge-perturbation",
        "analysis_code": {
            "path": "scripts/analyze_edge_perturbation.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "experiment_manifest": {
            "path": "experiment_manifest.json",
            "sha256": experiment_manifest_sha256,
        },
        "design": {
            "design_id": first.design_id,
            "code_version": first.code_version,
            "seed_base": first.seed_base,
            "sample_fraction": SAMPLE_FRACTION,
        },
        "scope": {
            "datasets": list(DATASETS),
            "models": list(MODELS),
            "splits": {
                dataset: list(SPLITS[dataset]) for dataset in DATASETS
            },
            "seeds": list(SEEDS),
            "pair_indices": pair_indices,
            "pairs_per_dataset_model_cell": len(pair_indices),
            "checkpoint_rows": len(records),
        },
        "included_pairs": [
            {
                "dataset": pair.dataset,
                "pair_index": pair.pair_index,
            }
            for pair in ordered_pairs
        ],
        "input_hashes": [
            {
                "dataset": pair.dataset,
                "pair_index": pair.pair_index,
                "metadata_json_sha256": pair.metadata_sha256,
                "outputs_sha256": dict(pair.output_sha256),
                "graphs_sha256": dict(pair.graph_sha256),
            }
            for pair in ordered_pairs
        ],
        "generated_outputs_sha256": {
            path.name: file_sha256(path)
            for path in generated_outputs.values()
        },
    }


def analyze(input_root: Path, output_root: Path) -> dict[str, Path]:
    """Validate one actual experiment store and write all derived outputs."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    (
        records,
        checkpoint_columns,
        pair_provenance,
        experiment_manifest_sha256,
    ) = load_records(input_root)

    dataset_order = {
        value: index
        for index, value in enumerate(
            _ordered(
                (record.dataset for record in records),
                PREFERRED_DATASET_ORDER,
            )
        )
    }
    model_order = {
        value: index
        for index, value in enumerate(
            _ordered(
                (record.model for record in records), PREFERRED_MODEL_ORDER
            )
        )
    }
    records = sorted(
        records,
        key=lambda record: (
            dataset_order[record.dataset],
            model_order[record.model],
            record.pair_index,
            record.split,
            record.seed,
        ),
    )
    checkpoint_rows = [record.row for record in records]
    pair_rows = build_pair_summary(records)
    running_rows = build_running_summary(pair_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "checkpoint_metrics": output_root / "checkpoint_metrics.csv",
        "pair_summary": output_root / "pair_summary.csv",
        "running_summary": output_root / "running_summary.csv",
        "figure": output_root / FIGURE_FILENAME,
        "analysis_manifest": output_root / "analysis_manifest.json",
    }
    _write_csv(
        outputs["checkpoint_metrics"], checkpoint_rows, checkpoint_columns
    )
    _write_csv(
        outputs["pair_summary"],
        pair_rows,
        (
            "dataset",
            "model",
            "pair_index",
            "n_checkpoints",
            "mean_treatment_minus_control_macro_f1",
        ),
    )
    _write_csv(
        outputs["running_summary"],
        running_rows,
        (
            "dataset",
            "model",
            "pair_index",
            "n_pairs",
            "running_mean_treatment_minus_control_macro_f1",
        ),
    )
    figure = build_convergence_figure(pair_rows, running_rows)
    try:
        figure.savefig(
            outputs["figure"],
            dpi=200,
            bbox_inches="tight",
            metadata={"Creator": "scripts/analyze_edge_perturbation.py"},
        )
    finally:
        plt.close(figure)
    generated_outputs = {
        key: path
        for key, path in outputs.items()
        if key != "analysis_manifest"
    }
    _write_json(
        outputs["analysis_manifest"],
        build_analysis_manifest(
            records=records,
            pairs=pair_provenance,
            experiment_manifest_sha256=experiment_manifest_sha256,
            generated_outputs=generated_outputs,
        ),
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="root containing dataset/pair-XXXX input directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="directory for CSV summaries, convergence PNG, and analysis manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = analyze(args.input_root, args.output_root)
    print(
        f"Validated actual graph pairs and wrote {len(outputs)} outputs to "
        f"{args.output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
