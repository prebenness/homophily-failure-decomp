"""Analyse the separately registered edge-perturbation extension.

The input layout is::

    <input-root>/experiment_manifest.json
    <input-root>/<dataset>/pair-XXXX/
        metadata.json
        graphs.npz
        swaps.json
        predictions.npz
        checkpoint_metrics.csv

The experiment manifest registers an ordered list of dataset--model cells.
One model is registered per dataset.  Pair indices must be contiguous within
each dataset, while different datasets may currently contain different
numbers of retained graph pairs.
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
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


PAIR_DIRECTORY_PATTERN = re.compile(r"pair-(\d+)$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
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
SAMPLE_FRACTION = 0.5
FIGURE_FILENAME = "running_convergence.png"
PAIR_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "predictions.npz",
    "checkpoint_metrics.csv",
)
REUSED_SOURCE_FILES = ("graphs.npz", "swaps.json")
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
GRAPH_MODES = ("reuse-original", "generate")
EXPECTED_IDENTITIES = {
    "pubmed": tuple(
        (f"geomgcn-{split_index}", seed)
        for split_index in range(10)
        for seed in range(1, 4)
    ),
    "roman-empire": tuple(
        (f"platonov-{split_index}", seed)
        for split_index in range(10)
        for seed in range(1, 4)
    ),
    "amazon-ratings": tuple(
        (f"platonov-{split_index}", seed)
        for split_index in range(10)
        for seed in range(1, 4)
    ),
    "ogbn-arxiv": tuple(("ogb-time", seed) for seed in range(1, 31)),
    "arxiv-year": tuple(
        (f"lim-{split_index}", seed)
        for split_index in range(5)
        for seed in range(1, 7)
    ),
}
DISPLAY_NAMES = {
    "pubmed": "PubMed",
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "ogbn-arxiv": "ogbn-arxiv",
    "arxiv-year": "arxiv-year",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
}


@dataclass(frozen=True)
class RegisteredCell:
    """One dataset--model cell frozen in the extension manifest."""

    dataset: str
    model: str
    graph_mode: str


@dataclass(frozen=True)
class ExperimentContract:
    """Validated extension settings read from the experiment manifest."""

    cells: tuple[RegisteredCell, ...]
    design_id: str
    code_version: str
    seed_base: int
    sample_fraction: float


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
    """Validated provenance for one completed extension graph pair."""

    cell: RegisteredCell
    pair_index: int
    metadata_sha256: str
    output_sha256: dict[str, str]
    graph_sha256: dict[str, str]
    graph_source: dict


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


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


def load_experiment_contract(path: Path) -> ExperimentContract:
    """Validate the extension manifest and retain its cell order."""

    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"{path}: schema_version must be 1")

    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise RuntimeError(f"{path}: cells must be a non-empty list")

    cells: list[RegisteredCell] = []
    seen_cells: set[tuple[str, str]] = set()
    seen_datasets: set[str] = set()
    for index, raw_cell in enumerate(raw_cells):
        location = f"{path}:cells[{index}]"
        if not isinstance(raw_cell, dict):
            raise RuntimeError(f"{location}: expected an object")
        dataset = raw_cell.get("dataset")
        model = raw_cell.get("model")
        graph_mode = raw_cell.get("graph_mode")
        if not all(
            isinstance(value, str) and value
            for value in (dataset, model, graph_mode)
        ):
            raise RuntimeError(
                f"{location}: dataset, model, and graph_mode must be "
                "non-empty strings"
            )
        if dataset not in EXPECTED_IDENTITIES:
            raise RuntimeError(
                f"{location}: unsupported dataset {dataset!r}; no registered "
                "30-run identity panel is known"
            )
        if graph_mode not in GRAPH_MODES:
            raise RuntimeError(
                f"{location}: graph_mode must be one of {list(GRAPH_MODES)}"
            )
        key = (dataset, model)
        if key in seen_cells:
            raise RuntimeError(f"{location}: duplicate registered cell {key}")
        if dataset in seen_datasets:
            raise RuntimeError(
                f"{location}: only one model may be registered per dataset"
            )
        cells.append(
            RegisteredCell(
                dataset=dataset,
                model=model,
                graph_mode=graph_mode,
            )
        )
        seen_cells.add(key)
        seen_datasets.add(dataset)

    design_id = manifest.get("design_id")
    code_version = manifest.get("code_version")
    seed_base = manifest.get("seed_base")
    sample_fraction = manifest.get("sample_fraction")
    if not isinstance(design_id, str) or not design_id:
        raise RuntimeError(f"{path}: design_id must be a non-empty string")
    if not isinstance(code_version, str) or not code_version:
        raise RuntimeError(f"{path}: code_version must be a non-empty string")
    if isinstance(seed_base, bool) or not isinstance(seed_base, int):
        raise RuntimeError(f"{path}: seed_base must be an integer")
    if (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or float(sample_fraction) != SAMPLE_FRACTION
    ):
        raise RuntimeError(
            f"{path}: sample_fraction must be {SAMPLE_FRACTION}"
        )
    return ExperimentContract(
        cells=tuple(cells),
        design_id=design_id,
        code_version=code_version,
        seed_base=seed_base,
        sample_fraction=float(sample_fraction),
    )


def discover_pair_directories(
    input_root: Path,
    contract: ExperimentContract,
) -> list[tuple[RegisteredCell, int, Path]]:
    """Find each registered dataset's contiguous retained graph pairs."""

    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")

    cell_by_dataset = {cell.dataset: cell for cell in contract.cells}
    discovered_by_dataset: dict[
        str, list[tuple[RegisteredCell, int, Path]]
    ] = defaultdict(list)
    for dataset_dir in sorted(input_root.iterdir(), key=lambda item: item.name):
        if not dataset_dir.is_dir():
            continue
        pair_directories = [
            pair_dir
            for pair_dir in dataset_dir.iterdir()
            if pair_dir.is_dir()
            and PAIR_DIRECTORY_PATTERN.fullmatch(pair_dir.name) is not None
        ]
        if not pair_directories:
            continue
        cell = cell_by_dataset.get(dataset_dir.name)
        if cell is None:
            raise RuntimeError(
                "Unexpected dataset directory containing extension pairs: "
                f"{dataset_dir.name!r}"
            )
        for pair_dir in sorted(pair_directories, key=lambda item: item.name):
            match = PAIR_DIRECTORY_PATTERN.fullmatch(pair_dir.name)
            assert match is not None
            discovered_by_dataset[cell.dataset].append(
                (cell, int(match.group(1)), pair_dir)
            )

    missing = [
        cell.dataset
        for cell in contract.cells
        if not discovered_by_dataset[cell.dataset]
    ]
    if missing:
        raise RuntimeError(
            f"Registered extension cells have no completed pairs: {missing}"
        )

    discovered: list[tuple[RegisteredCell, int, Path]] = []
    for cell in contract.cells:
        dataset_pairs = sorted(
            discovered_by_dataset[cell.dataset], key=lambda item: item[1]
        )
        pair_indices = [pair_index for _, pair_index, _ in dataset_pairs]
        expected_indices = list(range(1, pair_indices[-1] + 1))
        if pair_indices != expected_indices:
            raise RuntimeError(
                f"{cell.dataset}: pair indices must be contiguous beginning "
                f"at 1; found {pair_indices}"
            )
        discovered.extend(dataset_pairs)
    return discovered


def _validate_graph_source(
    value: object,
    *,
    path: Path,
    cell: RegisteredCell,
    verified_output_sha256: dict[str, str],
) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: graph_source must be an object")
    if value.get("mode") != cell.graph_mode:
        raise RuntimeError(
            f"{path}: graph_source.mode must be {cell.graph_mode!r}"
        )

    if cell.graph_mode == "generate":
        if set(value) != {"mode"}:
            raise RuntimeError(
                f"{path}: generated graph_source may contain only mode"
            )
        return {"mode": "generate"}

    source_pair = value.get("source_pair")
    source_metadata_sha256 = value.get("source_metadata_sha256")
    source_output_sha256 = value.get("source_output_sha256")
    if not isinstance(source_pair, str) or not source_pair:
        raise RuntimeError(
            f"{path}: reused graph_source.source_pair must be non-empty"
        )
    source_path = PurePosixPath(source_pair)
    if (
        source_path.is_absolute()
        or ".." in source_path.parts
        or "\\" in source_pair
    ):
        raise RuntimeError(
            f"{path}: graph_source.source_pair must be a repo-relative "
            "POSIX path"
        )
    if not _is_sha256(source_metadata_sha256):
        raise RuntimeError(
            f"{path}: graph_source.source_metadata_sha256 must be a SHA256"
        )
    if not isinstance(source_output_sha256, dict):
        raise RuntimeError(
            f"{path}: graph_source.source_output_sha256 must be an object"
        )

    verified_source_hashes: dict[str, str] = {}
    for filename in REUSED_SOURCE_FILES:
        source_hash = source_output_sha256.get(filename)
        if not _is_sha256(source_hash):
            raise RuntimeError(
                f"{path}: graph_source.source_output_sha256[{filename!r}] "
                "must be a SHA256"
            )
        if source_hash != verified_output_sha256[filename]:
            raise RuntimeError(
                f"{path}: copied {filename} does not match its source hash"
            )
        verified_source_hashes[filename] = source_hash
    return {
        "mode": "reuse-original",
        "source_pair": source_pair,
        "source_metadata_sha256": source_metadata_sha256,
        "source_output_sha256": verified_source_hashes,
    }


def _validate_metadata(
    metadata: dict,
    *,
    path: Path,
    pair_dir: Path,
    cell: RegisteredCell,
    pair_index: int,
    contract: ExperimentContract,
    experiment_manifest_sha256: str,
) -> PairProvenance:
    if metadata.get("schema_version") != 1:
        raise RuntimeError(f"{path}: schema_version must be 1")
    if metadata.get("phase") != "actual":
        raise RuntimeError(
            f"{path}: phase must be 'actual'; debug inputs are invalid"
        )
    if metadata.get("completed") is not True:
        raise RuntimeError(f"{path}: completed must be true")
    if metadata.get("dataset") != cell.dataset:
        raise RuntimeError(
            f"{path}: dataset does not match directory {cell.dataset!r}"
        )
    if metadata.get("model") != cell.model:
        raise RuntimeError(
            f"{path}: model must equal the registered model {cell.model!r}"
        )
    if metadata.get("graph_mode") != cell.graph_mode:
        raise RuntimeError(
            f"{path}: graph_mode must equal the registered mode "
            f"{cell.graph_mode!r}"
        )

    recorded_pair_index = metadata.get("pair_index")
    if (
        isinstance(recorded_pair_index, bool)
        or not isinstance(recorded_pair_index, int)
        or recorded_pair_index != pair_index
    ):
        raise RuntimeError(
            f"{path}: pair_index must equal directory index {pair_index}"
        )
    pair_seed = metadata.get("pair_seed")
    if isinstance(pair_seed, bool) or not isinstance(pair_seed, int):
        raise RuntimeError(f"{path}: pair_seed must be an integer")
    if (
        cell.graph_mode == "generate"
        and pair_seed != contract.seed_base + pair_index
    ):
        raise RuntimeError(
            f"{path}: generated pair_seed must equal seed_base + pair_index"
        )
    if metadata.get("n_checkpoints") != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{path}: n_checkpoints must be {EXPECTED_CHECKPOINTS}"
        )
    if metadata.get("n_checkpoint_rows") != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{path}: n_checkpoint_rows must be {EXPECTED_CHECKPOINTS}"
        )
    sample_fraction = metadata.get("sample_fraction")
    if (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or float(sample_fraction) != contract.sample_fraction
    ):
        raise RuntimeError(
            f"{path}: sample_fraction must be {contract.sample_fraction}"
        )
    if metadata.get("all_original_predictions_reproduced") is not True:
        raise RuntimeError(
            f"{path}: all_original_predictions_reproduced must be true"
        )
    expected_design = {
        "code_version": contract.code_version,
        "design_id": contract.design_id,
        "seed_base": contract.seed_base,
        "experiment_manifest_sha256": experiment_manifest_sha256,
    }
    design_mismatch = {
        key: (expected, metadata.get(key))
        for key, expected in expected_design.items()
        if metadata.get(key) != expected
    }
    if design_mismatch:
        raise RuntimeError(
            f"{path}: metadata differs from experiment manifest: "
            f"{design_mismatch}"
        )

    audit = metadata.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{path}: audit must be an object")
    failed_audits = [
        field for field in REQUIRED_BOOLEAN_AUDITS if audit.get(field) is not True
    ]
    if failed_audits:
        raise RuntimeError(
            f"{path}: graph audits must all be true; failed={failed_audits}"
        )
    graph_sha256: dict[str, str] = {}
    for field in GRAPH_HASH_FIELDS:
        value = audit.get(field)
        if not _is_sha256(value):
            raise RuntimeError(f"{path}: audit.{field} must be a SHA256")
        graph_sha256[field] = value
    if len(set(graph_sha256.values())) != len(graph_sha256):
        raise RuntimeError(
            f"{path}: original, treatment, and control hashes must be distinct"
        )

    recorded_output_sha256 = metadata.get("output_sha256")
    if not isinstance(recorded_output_sha256, dict):
        raise RuntimeError(f"{path}: output_sha256 must be an object")
    verified_output_sha256: dict[str, str] = {}
    for filename in PAIR_OUTPUT_FILES:
        output_path = pair_dir / filename
        if not output_path.is_file():
            raise RuntimeError(f"{path}: required output is missing: {filename}")
        observed_hash = file_sha256(output_path)
        if recorded_output_sha256.get(filename) != observed_hash:
            raise RuntimeError(
                f"{path}: output_sha256 mismatch for {filename}"
            )
        verified_output_sha256[filename] = observed_hash

    graph_source = _validate_graph_source(
        metadata.get("graph_source"),
        path=path,
        cell=cell,
        verified_output_sha256=verified_output_sha256,
    )
    return PairProvenance(
        cell=cell,
        pair_index=pair_index,
        metadata_sha256=file_sha256(path),
        output_sha256=verified_output_sha256,
        graph_sha256=graph_sha256,
        graph_source=graph_source,
    )


def _read_checkpoint_csv(
    path: Path,
    *,
    cell: RegisteredCell,
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
        dataset = row["dataset"].strip()
        model = row["model"].strip()
        split = row["split"].strip()
        if dataset != cell.dataset or model != cell.model:
            raise RuntimeError(
                f"{location}: row cell {(dataset, model)} differs from "
                f"registered cell {(cell.dataset, cell.model)}"
            )
        try:
            observed_pair_index = int(row["pair_index"])
            seed = int(row["seed"])
            effect = float(row[EFFECT_COLUMN])
        except ValueError as exc:
            raise RuntimeError(
                f"{location}: pair_index, seed, and {EFFECT_COLUMN} must be "
                "numeric"
            ) from exc
        if observed_pair_index != pair_index:
            raise RuntimeError(
                f"{location}: pair_index differs from directory index "
                f"{pair_index}"
            )
        if not math.isfinite(effect):
            raise RuntimeError(
                f"{location}: {EFFECT_COLUMN} must be finite"
            )
        records.append(
            CheckpointRecord(
                row=dict(row),
                dataset=dataset,
                model=model,
                pair_index=pair_index,
                split=split,
                seed=seed,
                effect=effect,
            )
        )

    identities = [(record.split, record.seed) for record in records]
    if len(records) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{cell.dataset}/{cell.model}/pair-{pair_index:04d}: expected "
            f"{EXPECTED_CHECKPOINTS} checkpoint rows, found {len(records)}"
        )
    if len(set(identities)) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{cell.dataset}/{cell.model}/pair-{pair_index:04d}: split/seed "
            "rows must be unique"
        )
    expected_identities = set(EXPECTED_IDENTITIES[cell.dataset])
    if set(identities) != expected_identities:
        missing = sorted(expected_identities - set(identities))
        unexpected = sorted(set(identities) - expected_identities)
        raise RuntimeError(
            f"{cell.dataset}/{cell.model}/pair-{pair_index:04d}: checkpoint "
            f"identities differ from the registered 30-run panel; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return records, fieldnames


def validate_pair_provenance(pairs: Sequence[PairProvenance]) -> None:
    """Check graph identity within datasets and uniqueness of rewired graphs."""

    original_by_dataset: dict[str, set[str]] = defaultdict(set)
    rewired_hash_owners: dict[str, str] = {}
    for pair in pairs:
        original_by_dataset[pair.cell.dataset].add(
            pair.graph_sha256["original_edge_sha256"]
        )
        for condition in ("treatment", "control"):
            graph_hash = pair.graph_sha256[f"{condition}_edge_sha256"]
            owner = (
                f"{pair.cell.dataset}/pair-{pair.pair_index:04d}/{condition}"
            )
            previous = rewired_hash_owners.get(graph_hash)
            if previous is not None:
                raise RuntimeError(
                    f"Rewired graph hash is reused by {previous} and {owner}"
                )
            rewired_hash_owners[graph_hash] = owner
    inconsistent = {
        dataset: sorted(hashes)
        for dataset, hashes in original_by_dataset.items()
        if len(hashes) != 1
    }
    if inconsistent:
        raise RuntimeError(
            f"Original graph hash changes across dataset pairs: {inconsistent}"
        )


def load_records(
    input_root: Path,
) -> tuple[
    list[CheckpointRecord],
    list[str],
    list[PairProvenance],
    ExperimentContract,
    str,
]:
    """Load and validate every completed pair registered by the extension."""

    experiment_manifest_path = input_root / "experiment_manifest.json"
    if not experiment_manifest_path.is_file():
        raise RuntimeError(
            f"Missing extension experiment manifest: {experiment_manifest_path}"
        )
    contract = load_experiment_contract(experiment_manifest_path)
    experiment_manifest_sha256 = file_sha256(experiment_manifest_path)

    records: list[CheckpointRecord] = []
    provenance: list[PairProvenance] = []
    output_columns = list(REQUIRED_COLUMNS)
    seen_columns = set(output_columns)
    for cell, pair_index, pair_dir in discover_pair_directories(
        input_root, contract
    ):
        metadata_path = pair_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing pair metadata: {metadata_path}")
        metadata = _read_json(metadata_path)
        provenance.append(
            _validate_metadata(
                metadata,
                path=metadata_path,
                pair_dir=pair_dir,
                cell=cell,
                pair_index=pair_index,
                contract=contract,
                experiment_manifest_sha256=experiment_manifest_sha256,
            )
        )
        pair_records, fieldnames = _read_checkpoint_csv(
            pair_dir / "checkpoint_metrics.csv",
            cell=cell,
            pair_index=pair_index,
        )
        records.extend(pair_records)
        for column in fieldnames:
            if column not in seen_columns:
                output_columns.append(column)
                seen_columns.add(column)

    observed_cells = {(record.dataset, record.model) for record in records}
    expected_cells = {(cell.dataset, cell.model) for cell in contract.cells}
    if observed_cells != expected_cells:
        raise RuntimeError(
            "Extension checkpoint cells differ from the manifest; "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"unexpected={sorted(observed_cells - expected_cells)}"
        )
    validate_pair_provenance(provenance)
    return (
        records,
        output_columns,
        provenance,
        contract,
        experiment_manifest_sha256,
    )


def _cell_order(contract: ExperimentContract) -> dict[tuple[str, str], int]:
    return {
        (cell.dataset, cell.model): index
        for index, cell in enumerate(contract.cells)
    }


def build_pair_summary(
    records: Sequence[CheckpointRecord],
    contract: ExperimentContract,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str, int], list[CheckpointRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.model, record.pair_index)].append(record)

    order = _cell_order(contract)
    rows: list[dict[str, object]] = []
    for (dataset, model, pair_index), cell_records in sorted(
        grouped.items(),
        key=lambda item: (
            order[(item[0][0], item[0][1])],
            item[0][2],
        ),
    ):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "pair_index": pair_index,
                "n_checkpoints": len(cell_records),
                "mean_treatment_minus_control_macro_f1": fmean(
                    record.effect for record in cell_records
                ),
            }
        )
    return rows


def build_running_summary(
    pair_rows: Sequence[dict[str, object]],
    contract: ExperimentContract,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["dataset"]), str(row["model"]))].append(row)

    rows: list[dict[str, object]] = []
    for cell in contract.cells:
        cell_rows = sorted(
            grouped[(cell.dataset, cell.model)],
            key=lambda row: int(row["pair_index"]),
        )
        cumulative_total = 0.0
        for n_pairs, pair_row in enumerate(cell_rows, start=1):
            cumulative_total += float(
                pair_row["mean_treatment_minus_control_macro_f1"]
            )
            rows.append(
                {
                    "dataset": cell.dataset,
                    "model": cell.model,
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
    contract: ExperimentContract,
):
    """Build one panel per registered cell, including unequal pair counts."""

    n_cells = len(contract.cells)
    n_columns = 2 if n_cells > 1 else 1
    n_rows = math.ceil(n_cells / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(8.0, max(2.8, 2.7 * n_rows)),
        sharey=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    running_lookup = {
        (str(row["dataset"]), str(row["model"]), int(row["pair_index"])): float(
            row["running_mean_treatment_minus_control_macro_f1"]
        )
        for row in running_rows
    }

    for axis, cell in zip(flat_axes, contract.cells):
        cell_rows = sorted(
            (
                row
                for row in pair_rows
                if row["dataset"] == cell.dataset
                and row["model"] == cell.model
            ),
            key=lambda row: int(row["pair_index"]),
        )
        pair_indices = [int(row["pair_index"]) for row in cell_rows]
        pair_means = [
            float(row["mean_treatment_minus_control_macro_f1"])
            for row in cell_rows
        ]
        running_means = [
            running_lookup[(cell.dataset, cell.model, pair_index)]
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
            f"{DISPLAY_NAMES.get(cell.dataset, cell.dataset)} · "
            f"{DISPLAY_NAMES.get(cell.model, cell.model)}",
            fontsize=10,
        )
        axis.set_xlabel("Graph-pair index")
        axis.set_ylabel("Treatment − control macro-F1")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)

    for axis in flat_axes[n_cells:]:
        axis.set_visible(False)
    handles, labels = flat_axes[0].get_legend_handles_labels()
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
    contract: ExperimentContract,
    experiment_manifest_sha256: str,
    generated_outputs: dict[str, Path],
) -> dict:
    order = _cell_order(contract)
    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (
            order[(pair.cell.dataset, pair.cell.model)],
            pair.pair_index,
        ),
    )
    pair_indices_by_cell = {
        (cell.dataset, cell.model): sorted(
            pair.pair_index
            for pair in ordered_pairs
            if pair.cell == cell
        )
        for cell in contract.cells
    }
    return {
        "schema_version": 1,
        "analysis": "sequential-edge-perturbation-extension",
        "analysis_code": {
            "path": "scripts/analyze_edge_perturbation_extension.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "experiment_manifest": {
            "path": "experiment_manifest.json",
            "sha256": experiment_manifest_sha256,
        },
        "design": {
            "design_id": contract.design_id,
            "code_version": contract.code_version,
            "seed_base": contract.seed_base,
            "sample_fraction": contract.sample_fraction,
        },
        "scope": {
            "cells": [
                {
                    "dataset": cell.dataset,
                    "model": cell.model,
                    "graph_mode": cell.graph_mode,
                    "pair_indices": pair_indices_by_cell[
                        (cell.dataset, cell.model)
                    ],
                    "n_pairs": len(
                        pair_indices_by_cell[(cell.dataset, cell.model)]
                    ),
                    "checkpoints_per_pair": EXPECTED_CHECKPOINTS,
                }
                for cell in contract.cells
            ],
            "checkpoint_rows": len(records),
        },
        "included_pairs": [
            {
                "dataset": pair.cell.dataset,
                "model": pair.cell.model,
                "graph_mode": pair.cell.graph_mode,
                "pair_index": pair.pair_index,
            }
            for pair in ordered_pairs
        ],
        "input_hashes": [
            {
                "dataset": pair.cell.dataset,
                "model": pair.cell.model,
                "pair_index": pair.pair_index,
                "metadata_json_sha256": pair.metadata_sha256,
                "outputs_sha256": dict(pair.output_sha256),
                "graphs_sha256": dict(pair.graph_sha256),
                "graph_source": dict(pair.graph_source),
            }
            for pair in ordered_pairs
        ],
        "generated_outputs_sha256": {
            path.name: file_sha256(path)
            for path in generated_outputs.values()
        },
    }


def analyze(input_root: Path, output_root: Path) -> dict[str, Path]:
    """Validate one extension store and write its separate derived outputs."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    (
        records,
        checkpoint_columns,
        pair_provenance,
        contract,
        experiment_manifest_sha256,
    ) = load_records(input_root)

    order = _cell_order(contract)
    identity_order = {
        dataset: {
            identity: index
            for index, identity in enumerate(EXPECTED_IDENTITIES[dataset])
        }
        for dataset in EXPECTED_IDENTITIES
    }
    records = sorted(
        records,
        key=lambda record: (
            order[(record.dataset, record.model)],
            record.pair_index,
            identity_order[record.dataset][(record.split, record.seed)],
        ),
    )
    checkpoint_rows = [record.row for record in records]
    pair_rows = build_pair_summary(records, contract)
    running_rows = build_running_summary(pair_rows, contract)

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
    figure = build_convergence_figure(pair_rows, running_rows, contract)
    try:
        figure.savefig(
            outputs["figure"],
            dpi=200,
            bbox_inches="tight",
            metadata={
                "Creator": "scripts/analyze_edge_perturbation_extension.py"
            },
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
            contract=contract,
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
        help="separate extension root containing manifest and dataset pairs",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="separate directory for extension summaries and convergence plot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = analyze(args.input_root, args.output_root)
    print(
        f"Validated extension graph pairs and wrote {len(outputs)} outputs to "
        f"{args.output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
