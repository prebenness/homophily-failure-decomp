"""Validate and analyse the registered PubMed edge dose-response experiment.

The input layout is::

    <input-root>/experiment_manifest.json
    <input-root>/baseline/checkpoint_metrics.csv
    <input-root>/replicate-XXXX/pct-PPP/
        metadata.json
        checkpoint_metrics.csv

There is one baseline score for each of the 60 registered ERM checkpoints
(30 GCN and 30 GraphSAGE-mean checkpoints).  Every non-zero percentage cell
contains treatment and random-control scores for the same checkpoints.  The
analysis first averages checkpoints within each graph replicate.  It then
reports the mean and sample standard deviation across the five graph
replicates.  The common p=0 baseline is copied to each graph replicate, so its
between-graph standard deviation is exactly zero.
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
from statistics import fmean, stdev
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(REPO_ROOT / "artifacts" / "matplotlib-cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATASET = "pubmed"
MODELS = ("gcn", "graphsage-mean")
P_PERCENTAGES = tuple(range(0, 101, 5))
NONZERO_P_PERCENTAGES = P_PERCENTAGES[1:]
GRAPH_REPLICATES = tuple(range(1, 6))
EXPECTED_IDENTITIES = tuple(
    (f"geomgcn-{split_index}", seed)
    for split_index in range(10)
    for seed in range(1, 4)
)
EXPECTED_CHECKPOINTS = len(EXPECTED_IDENTITIES)
CONDITIONS = ("treatment", "control")
METRIC_POPULATION = "fixed-full-tail-test"
REPLICATE_PATTERN = re.compile(r"replicate-(\d{4})$")
PERCENTAGE_PATTERN = re.compile(r"(?:pct|p)-(\d{3})$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FIGURE_FILENAME = "pubmed_edge_dose_response.png"
BASELINE_OUTPUT_FILES = ("checkpoint_metrics.csv", "predictions.npz")
DOSE_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "checkpoint_metrics.csv",
    "predictions.npz",
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

BASELINE_METRIC_COLUMNS = (
    "tail_test_macro_f1",
    "baseline_macro_f1",
    "original_macro_f1",
)
TREATMENT_METRIC_COLUMNS = (
    "tail_test_treatment_macro_f1",
    "treatment_macro_f1",
)
CONTROL_METRIC_COLUMNS = (
    "tail_test_control_macro_f1",
    "control_macro_f1",
)


@dataclass(frozen=True)
class ExperimentContract:
    """The design values frozen before actual collection."""

    design_id: str
    code_version: str
    seed_base: int
    models: tuple[str, ...]
    p_percentages: tuple[int, ...]
    graph_replicates: tuple[int, ...]
    metric_population: str
    tail_count: int
    actual_generation_attempts: int


@dataclass(frozen=True)
class CheckpointScore:
    """One checkpoint score on the fixed full tail-test set."""

    model: str
    split: str
    seed: int
    p_percent: int
    replicate_index: int | None
    condition: str
    macro_f1: float
    n_tail_test: int
    tail_test_class_composition: str
    source_path: str


@dataclass(frozen=True)
class InputFile:
    """One validated input included in the analysis."""

    relative_path: str
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing checkpoint metrics: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc
    if not fieldnames:
        raise RuntimeError(f"{path}: checkpoint metrics have no header")
    if not rows:
        raise RuntimeError(f"{path}: checkpoint metrics are empty")
    return rows, fieldnames


def _require_string(value: object, *, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{path}: {field} must be a non-empty string")
    return value


def load_experiment_contract(path: Path) -> ExperimentContract:
    """Read and strictly validate the registered PubMed design."""

    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"{path}: schema_version must be 1")
    if manifest.get("dataset") != DATASET:
        raise RuntimeError(f"{path}: dataset must be {DATASET!r}")

    raw_models = manifest.get("models")
    if raw_models != list(MODELS):
        raise RuntimeError(f"{path}: models must be {list(MODELS)}")
    raw_percentages = manifest.get("p_percentages")
    if raw_percentages != list(P_PERCENTAGES):
        raise RuntimeError(
            f"{path}: p_percentages must be {list(P_PERCENTAGES)}"
        )
    if manifest.get("graph_replicates") != len(GRAPH_REPLICATES):
        raise RuntimeError(
            f"{path}: graph_replicates must be {len(GRAPH_REPLICATES)}"
        )
    raw_replicates = manifest.get("graph_replicate_indices")
    if raw_replicates != list(GRAPH_REPLICATES):
        raise RuntimeError(
            f"{path}: graph_replicate_indices must be "
            f"{list(GRAPH_REPLICATES)}"
        )
    if manifest.get("metric_population") != METRIC_POPULATION:
        raise RuntimeError(
            f"{path}: metric_population must be {METRIC_POPULATION!r}"
        )
    seed_base = manifest.get("replicate_seed_base")
    if not _is_int(seed_base):
        raise RuntimeError(
            f"{path}: replicate_seed_base must be an integer"
        )
    tail_count = manifest.get("tail_count")
    if not _is_int(tail_count) or tail_count <= 0:
        raise RuntimeError(f"{path}: tail_count must be a positive integer")
    generation_attempts = manifest.get("actual_generation_attempts")
    if not _is_int(generation_attempts) or generation_attempts != 1:
        raise RuntimeError(
            f"{path}: actual_generation_attempts must be 1"
        )

    return ExperimentContract(
        design_id=_require_string(
            manifest.get("design_id"), path=path, field="design_id"
        ),
        code_version=_require_string(
            manifest.get("code_version"), path=path, field="code_version"
        ),
        seed_base=seed_base,
        models=tuple(raw_models),
        p_percentages=tuple(raw_percentages),
        graph_replicates=tuple(raw_replicates),
        metric_population=METRIC_POPULATION,
        tail_count=tail_count,
        actual_generation_attempts=generation_attempts,
    )


def _select_metric_column(
    fieldnames: Sequence[str],
    candidates: Sequence[str],
    *,
    path: Path,
    label: str,
) -> str:
    matches = [candidate for candidate in candidates if candidate in fieldnames]
    if len(matches) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one {label} metric column from "
            f"{list(candidates)}; found {matches}"
        )
    return matches[0]


def _parse_identity(
    row: dict[str, str],
    *,
    path: Path,
    row_number: int,
) -> tuple[str, str, int]:
    location = f"{path}:{row_number}"
    missing = [field for field in ("dataset", "model", "split", "seed") if field not in row]
    if missing:
        raise RuntimeError(
            f"{path}: missing required columns {', '.join(missing)}"
        )
    dataset = row["dataset"].strip()
    model = row["model"].strip()
    split = row["split"].strip()
    if dataset != DATASET:
        raise RuntimeError(f"{location}: dataset must be {DATASET!r}")
    if model not in MODELS:
        raise RuntimeError(f"{location}: unsupported model {model!r}")
    if not split:
        raise RuntimeError(f"{location}: split must be non-empty")
    try:
        seed = int(row["seed"])
    except ValueError as exc:
        raise RuntimeError(f"{location}: seed must be an integer") from exc
    return model, split, seed


def _parse_metric(
    row: dict[str, str],
    column: str,
    *,
    path: Path,
    row_number: int,
) -> float:
    location = f"{path}:{row_number}"
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"{location}: {column} must be numeric"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(
            f"{location}: {column} must be finite and in [0, 1]"
        )
    return value


def _parse_tail_test_definition(
    row: dict[str, str],
    *,
    path: Path,
    row_number: int,
) -> tuple[int, str]:
    """Parse the row-level evidence for the fixed evaluation population."""

    location = f"{path}:{row_number}"
    missing = [
        field
        for field in ("n_tail_test", "tail_test_class_composition")
        if field not in row
    ]
    if missing:
        raise RuntimeError(
            f"{path}: missing required columns {', '.join(missing)}"
        )
    try:
        n_tail_test = int(row["n_tail_test"])
    except ValueError as exc:
        raise RuntimeError(
            f"{location}: n_tail_test must be an integer"
        ) from exc
    if n_tail_test <= 0:
        raise RuntimeError(
            f"{location}: n_tail_test must be positive"
        )
    try:
        raw_composition = json.loads(row["tail_test_class_composition"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{location}: tail_test_class_composition must be valid JSON"
        ) from exc
    if (
        not isinstance(raw_composition, dict)
        or not raw_composition
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for count in raw_composition.values()
        )
    ):
        raise RuntimeError(
            f"{location}: tail_test_class_composition must map classes to "
            "positive integer counts"
        )
    if sum(raw_composition.values()) != n_tail_test:
        raise RuntimeError(
            f"{location}: tail_test_class_composition counts do not sum to "
            "n_tail_test"
        )
    canonical = json.dumps(
        raw_composition, sort_keys=True, separators=(",", ":")
    )
    return n_tail_test, canonical


def _validate_model_identity_panel(
    scores: Iterable[CheckpointScore],
    *,
    cell: str,
) -> None:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for score in scores:
        grouped[score.model].append((score.split, score.seed))
    if set(grouped) != set(MODELS):
        raise RuntimeError(
            f"{cell}: expected models {list(MODELS)}; "
            f"found {sorted(grouped)}"
        )
    expected = set(EXPECTED_IDENTITIES)
    for model in MODELS:
        identities = grouped[model]
        if len(identities) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(
                f"{cell}/{model}: expected {EXPECTED_CHECKPOINTS} "
                f"checkpoint rows, found {len(identities)}"
            )
        if len(set(identities)) != len(identities):
            raise RuntimeError(
                f"{cell}/{model}: split/seed identities must be unique"
            )
        if set(identities) != expected:
            missing = sorted(expected - set(identities))
            unexpected = sorted(set(identities) - expected)
            raise RuntimeError(
                f"{cell}/{model}: checkpoint identities differ from the "
                f"registered panel; missing={missing}, "
                f"unexpected={unexpected}"
            )


def load_baseline(path: Path, input_root: Path) -> tuple[list[CheckpointScore], InputFile]:
    """Load the one common original-graph baseline."""

    rows, fieldnames = _read_rows(path)
    metric_column = _select_metric_column(
        fieldnames,
        BASELINE_METRIC_COLUMNS,
        path=path,
        label="baseline",
    )
    relative_path = path.relative_to(input_root).as_posix()
    scores: list[CheckpointScore] = []
    for row_number, row in enumerate(rows, start=2):
        model, split, seed = _parse_identity(
            row, path=path, row_number=row_number
        )
        n_tail_test, composition = _parse_tail_test_definition(
            row, path=path, row_number=row_number
        )
        scores.append(
            CheckpointScore(
                model=model,
                split=split,
                seed=seed,
                p_percent=0,
                replicate_index=None,
                condition="baseline",
                macro_f1=_parse_metric(
                    row,
                    metric_column,
                    path=path,
                    row_number=row_number,
                ),
                n_tail_test=n_tail_test,
                tail_test_class_composition=composition,
                source_path=relative_path,
            )
        )
    _validate_model_identity_panel(scores, cell="baseline")
    return scores, InputFile(relative_path, file_sha256(path))


def discover_dose_directories(
    input_root: Path,
) -> list[tuple[int, int, Path]]:
    """Find the exact five-by-twenty non-zero dose grid."""

    discovered: dict[tuple[int, int], Path] = {}
    for replicate_dir in input_root.iterdir():
        if not replicate_dir.is_dir():
            continue
        replicate_match = REPLICATE_PATTERN.fullmatch(replicate_dir.name)
        if replicate_match is None:
            continue
        replicate_index = int(replicate_match.group(1))
        for dose_dir in replicate_dir.iterdir():
            if not dose_dir.is_dir():
                continue
            percentage_match = PERCENTAGE_PATTERN.fullmatch(dose_dir.name)
            if percentage_match is None:
                continue
            p_percent = int(percentage_match.group(1))
            key = (replicate_index, p_percent)
            if key in discovered:
                raise RuntimeError(
                    f"Duplicate dose directory for replicate={replicate_index}, "
                    f"p={p_percent}"
                )
            discovered[key] = dose_dir

    expected = {
        (replicate_index, p_percent)
        for replicate_index in GRAPH_REPLICATES
        for p_percent in NONZERO_P_PERCENTAGES
    }
    observed = set(discovered)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise RuntimeError(
            "Dose grid differs from the registered five-by-twenty grid; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [
        (replicate_index, p_percent, discovered[(replicate_index, p_percent)])
        for replicate_index in GRAPH_REPLICATES
        for p_percent in NONZERO_P_PERCENTAGES
    ]


def _validate_output_hashes(
    metadata: dict,
    *,
    metadata_path: Path,
    dose_dir: Path,
    required_files: Sequence[str],
) -> list[Path]:
    recorded = metadata.get("output_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError(f"{metadata_path}: output_sha256 must be an object")
    if set(recorded) != set(required_files):
        raise RuntimeError(
            f"{metadata_path}: output_sha256 keys must be "
            f"{list(required_files)}"
        )
    validated: list[Path] = []
    for filename, expected_hash in sorted(recorded.items()):
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected_hash, str)
            or SHA256_PATTERN.fullmatch(expected_hash) is None
        ):
            raise RuntimeError(
                f"{metadata_path}: invalid output_sha256 entry "
                f"{filename!r}: {expected_hash!r}"
            )
        output_path = dose_dir / filename
        if not output_path.is_file():
            raise RuntimeError(
                f"{metadata_path}: recorded output is missing: {filename}"
            )
        if file_sha256(output_path) != expected_hash:
            raise RuntimeError(
                f"{metadata_path}: output_sha256 mismatch for {filename}"
            )
        validated.append(output_path)
    return validated


def _validate_checkpoint_identities(
    metadata: dict,
    *,
    path: Path,
) -> None:
    expected = [
        {"split": split, "seed": seed}
        for split, seed in EXPECTED_IDENTITIES
    ]
    if metadata.get("checkpoint_identities") != expected:
        raise RuntimeError(
            f"{path}: checkpoint_identities differ from the registered panel"
        )


def _validate_baseline_metadata(
    metadata: dict,
    *,
    metadata_path: Path,
    baseline_dir: Path,
    contract: ExperimentContract,
    manifest_sha256: str,
) -> list[Path]:
    expected = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": DATASET,
        "models": list(MODELS),
        "metric_population": METRIC_POPULATION,
        "n_checkpoints_per_model": EXPECTED_CHECKPOINTS,
        "n_checkpoint_rows": len(MODELS) * EXPECTED_CHECKPOINTS,
        "all_original_predictions_reproduced": True,
        "design_id": contract.design_id,
        "code_version": contract.code_version,
        "experiment_manifest_sha256": manifest_sha256,
    }
    mismatch = {
        field: (wanted, metadata.get(field))
        for field, wanted in expected.items()
        if metadata.get(field) != wanted
    }
    if mismatch:
        raise RuntimeError(
            f"{metadata_path}: metadata differs from the registered "
            f"baseline: {mismatch}"
        )
    _validate_checkpoint_identities(metadata, path=metadata_path)
    return _validate_output_hashes(
        metadata,
        metadata_path=metadata_path,
        dose_dir=baseline_dir,
        required_files=BASELINE_OUTPUT_FILES,
    )


def _validate_dose_metadata(
    metadata: dict,
    *,
    metadata_path: Path,
    dose_dir: Path,
    replicate_index: int,
    p_percent: int,
    contract: ExperimentContract,
    manifest_sha256: str,
) -> list[Path]:
    requested_targets = (contract.tail_count * p_percent) // 100
    expected = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": DATASET,
        "models": list(MODELS),
        "replicate_index": replicate_index,
        "replicate_seed": contract.seed_base + replicate_index,
        "p_percent": p_percent,
        "tail_count": contract.tail_count,
        "requested_targets": requested_targets,
        "selected_targets": requested_targets,
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "target_selection": (
            "prefix of the replicate's frozen random tail-node ranking"
        ),
        "graph_start": "registered original PubMed graph",
        "generation_attempt": 0,
        "generation_retry_failures": [],
        "metric_population": METRIC_POPULATION,
        "n_checkpoints_per_model": EXPECTED_CHECKPOINTS,
        "n_checkpoint_rows": len(MODELS) * EXPECTED_CHECKPOINTS,
        "design_id": contract.design_id,
        "code_version": contract.code_version,
        "seed_base": contract.seed_base,
        "experiment_manifest_sha256": manifest_sha256,
    }
    mismatch = {
        field: (wanted, metadata.get(field))
        for field, wanted in expected.items()
        if metadata.get(field) != wanted
    }
    if mismatch:
        raise RuntimeError(
            f"{metadata_path}: metadata differs from the registered cell: "
            f"{mismatch}"
        )
    _validate_checkpoint_identities(metadata, path=metadata_path)
    audit = metadata.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{metadata_path}: audit must be an object")
    failed_audits = [
        field
        for field in REQUIRED_BOOLEAN_AUDITS
        if audit.get(field) is not True
    ]
    if failed_audits:
        raise RuntimeError(
            f"{metadata_path}: graph audits failed: {failed_audits}"
        )
    graph_hashes = []
    for field in GRAPH_HASH_FIELDS:
        value = audit.get(field)
        if (
            not isinstance(value, str)
            or SHA256_PATTERN.fullmatch(value) is None
        ):
            raise RuntimeError(
                f"{metadata_path}: audit.{field} must be a SHA256"
            )
        graph_hashes.append(value)
    if len(set(graph_hashes)) != len(graph_hashes):
        raise RuntimeError(
            f"{metadata_path}: original, treatment, and control graph hashes "
            "must be distinct"
        )
    return _validate_output_hashes(
        metadata,
        metadata_path=metadata_path,
        dose_dir=dose_dir,
        required_files=DOSE_OUTPUT_FILES,
    )


def _validate_graph_arrays(
    path: Path,
    *,
    contract: ExperimentContract,
    p_percent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Verify that the stored targets are the registered ranking prefix."""

    try:
        with np.load(path, allow_pickle=False) as arrays:
            missing = [
                name
                for name in (
                    "treatment_edge_index",
                    "control_edge_index",
                    "targets",
                    "tail_ranking",
                )
                if name not in arrays.files
            ]
            if missing:
                raise RuntimeError(
                    f"{path}: missing arrays {', '.join(missing)}"
                )
            ranking = np.asarray(arrays["tail_ranking"], dtype=np.int64)
            targets = np.asarray(arrays["targets"], dtype=np.int64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    if ranking.ndim != 1 or ranking.size != contract.tail_count:
        raise RuntimeError(
            f"{path}: tail_ranking must contain exactly "
            f"{contract.tail_count} nodes"
        )
    if np.unique(ranking).size != ranking.size:
        raise RuntimeError(f"{path}: tail_ranking must contain unique nodes")
    expected_count = (contract.tail_count * p_percent) // 100
    if targets.ndim != 1 or targets.size != expected_count:
        raise RuntimeError(
            f"{path}: targets must contain exactly {expected_count} nodes"
        )
    if not np.array_equal(targets, ranking[:expected_count]):
        raise RuntimeError(
            f"{path}: targets are not the registered ranking prefix"
        )
    return ranking, targets


def _validate_fixed_tail_test_subsets(
    scores: Sequence[CheckpointScore],
) -> None:
    """Require an identical evaluation subset definition at every dose."""

    definitions: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for score in scores:
        definitions[score.split].add(
            (score.n_tail_test, score.tail_test_class_composition)
        )
    inconsistent = {
        split: sorted(values)
        for split, values in definitions.items()
        if len(values) != 1
    }
    if inconsistent:
        raise RuntimeError(
            "The fixed tail-test subset changes across models, graph "
            f"replicates, or percentages: {inconsistent}"
        )


def load_dose_cell(
    path: Path,
    *,
    input_root: Path,
    replicate_index: int,
    p_percent: int,
) -> list[CheckpointScore]:
    """Load treatment and control values from one graph-pair cell."""

    rows, fieldnames = _read_rows(path)
    treatment_column = _select_metric_column(
        fieldnames,
        TREATMENT_METRIC_COLUMNS,
        path=path,
        label="treatment",
    )
    control_column = _select_metric_column(
        fieldnames,
        CONTROL_METRIC_COLUMNS,
        path=path,
        label="control",
    )
    effect_column = "treatment_minus_control_macro_f1"
    if effect_column not in fieldnames:
        raise RuntimeError(
            f"{path}: missing required column {effect_column}"
        )
    relative_path = path.relative_to(input_root).as_posix()
    by_condition: dict[str, list[CheckpointScore]] = {
        condition: [] for condition in CONDITIONS
    }
    for row_number, row in enumerate(rows, start=2):
        model, split, seed = _parse_identity(
            row, path=path, row_number=row_number
        )
        try:
            row_replicate = int(row["replicate_index"])
            row_percentage = int(row["p_percent"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"{path}:{row_number}: replicate_index and p_percent must "
                "be integers"
            ) from exc
        if (
            row_replicate != replicate_index
            or row_percentage != p_percent
        ):
            raise RuntimeError(
                f"{path}:{row_number}: replicate_index or p_percent differs "
                "from the containing dose cell"
            )
        n_tail_test, composition = _parse_tail_test_definition(
            row, path=path, row_number=row_number
        )
        treatment_value = _parse_metric(
            row,
            treatment_column,
            path=path,
            row_number=row_number,
        )
        control_value = _parse_metric(
            row,
            control_column,
            path=path,
            row_number=row_number,
        )
        try:
            recorded_effect = float(row[effect_column])
        except ValueError as exc:
            raise RuntimeError(
                f"{path}:{row_number}: {effect_column} must be numeric"
            ) from exc
        if (
            not math.isfinite(recorded_effect)
            or not math.isclose(
                recorded_effect,
                treatment_value - control_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"{path}:{row_number}: {effect_column} is inconsistent "
                "with the treatment and control scores"
            )
        for condition, value in (
            ("treatment", treatment_value),
            ("control", control_value),
        ):
            by_condition[condition].append(
                CheckpointScore(
                    model=model,
                    split=split,
                    seed=seed,
                    p_percent=p_percent,
                    replicate_index=replicate_index,
                    condition=condition,
                    macro_f1=value,
                    n_tail_test=n_tail_test,
                    tail_test_class_composition=composition,
                    source_path=relative_path,
                )
            )
    for condition in CONDITIONS:
        _validate_model_identity_panel(
            by_condition[condition],
            cell=(
                f"replicate-{replicate_index:04d}/"
                f"pct-{p_percent:03d}/{condition}"
            ),
        )
    return by_condition["treatment"] + by_condition["control"]


def load_inputs(
    input_root: Path,
) -> tuple[
    list[CheckpointScore],
    list[InputFile],
    ExperimentContract,
    str,
]:
    """Validate and load the complete actual-experiment store."""

    input_root = input_root.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")
    manifest_path = input_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing experiment manifest: {manifest_path}")
    contract = load_experiment_contract(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    input_files = [
        InputFile("experiment_manifest.json", manifest_sha256)
    ]

    baseline_dir = input_root / "baseline"
    baseline_metadata_path = baseline_dir / "metadata.json"
    if not baseline_metadata_path.is_file():
        raise RuntimeError(
            f"Missing baseline metadata: {baseline_metadata_path}"
        )
    baseline_metadata = _read_json(baseline_metadata_path)
    baseline_outputs = _validate_baseline_metadata(
        baseline_metadata,
        metadata_path=baseline_metadata_path,
        baseline_dir=baseline_dir,
        contract=contract,
        manifest_sha256=manifest_sha256,
    )
    input_files.append(
        InputFile(
            baseline_metadata_path.relative_to(input_root).as_posix(),
            file_sha256(baseline_metadata_path),
        )
    )
    for output_path in baseline_outputs:
        input_files.append(
            InputFile(
                output_path.relative_to(input_root).as_posix(),
                file_sha256(output_path),
            )
        )
    baseline_path = baseline_dir / "checkpoint_metrics.csv"
    baseline_scores, baseline_input = load_baseline(
        baseline_path, input_root
    )
    scores = list(baseline_scores)
    assert baseline_input.relative_path in {
        item.relative_path for item in input_files
    }

    ranking_by_replicate: dict[int, np.ndarray] = {}
    p100_targets: np.ndarray | None = None
    original_graph_hash: str | None = None
    rewired_hash_owners: dict[str, str] = {}
    for replicate_index, p_percent, dose_dir in discover_dose_directories(
        input_root
    ):
        metadata_path = dose_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing dose metadata: {metadata_path}")
        metadata = _read_json(metadata_path)
        validated_outputs = _validate_dose_metadata(
            metadata,
            metadata_path=metadata_path,
            dose_dir=dose_dir,
            replicate_index=replicate_index,
            p_percent=p_percent,
            contract=contract,
            manifest_sha256=manifest_sha256,
        )
        graphs_path = dose_dir / "graphs.npz"
        ranking, targets = _validate_graph_arrays(
            graphs_path,
            contract=contract,
            p_percent=p_percent,
        )
        previous_ranking = ranking_by_replicate.get(replicate_index)
        if previous_ranking is None:
            ranking_by_replicate[replicate_index] = ranking
        elif not np.array_equal(previous_ranking, ranking):
            raise RuntimeError(
                f"replicate-{replicate_index:04d}: tail_ranking changes "
                "across percentages"
            )
        if p_percent == 100:
            sorted_targets = np.sort(targets)
            if p100_targets is None:
                p100_targets = sorted_targets
            elif not np.array_equal(p100_targets, sorted_targets):
                raise RuntimeError(
                    "The p=100 target population differs across graph "
                    "replicates"
                )

        audit = metadata["audit"]
        observed_original = str(audit["original_edge_sha256"])
        if original_graph_hash is None:
            original_graph_hash = observed_original
        elif original_graph_hash != observed_original:
            raise RuntimeError(
                "The registered original graph hash changes across dose cells"
            )
        for condition in CONDITIONS:
            graph_hash = str(audit[f"{condition}_edge_sha256"])
            owner = (
                f"replicate-{replicate_index:04d}/"
                f"pct-{p_percent:03d}/{condition}"
            )
            previous_owner = rewired_hash_owners.get(graph_hash)
            if previous_owner is not None:
                raise RuntimeError(
                    f"Rewired graph hash is reused by {previous_owner} and "
                    f"{owner}"
                )
            rewired_hash_owners[graph_hash] = owner
        scores.extend(
            load_dose_cell(
                dose_dir / "checkpoint_metrics.csv",
                input_root=input_root,
                replicate_index=replicate_index,
                p_percent=p_percent,
            )
        )
        input_files.append(
            InputFile(
                metadata_path.relative_to(input_root).as_posix(),
                file_sha256(metadata_path),
            )
        )
        for output_path in validated_outputs:
            input_files.append(
                InputFile(
                    output_path.relative_to(input_root).as_posix(),
                    file_sha256(output_path),
                )
            )
    ranking_hashes = {
        hashlib.sha256(ranking.tobytes()).hexdigest()
        for ranking in ranking_by_replicate.values()
    }
    if len(ranking_hashes) != len(GRAPH_REPLICATES):
        raise RuntimeError(
            "Graph replicates must use five distinct frozen tail rankings"
        )
    _validate_fixed_tail_test_subsets(scores)
    return scores, input_files, contract, manifest_sha256


def build_replicate_summary(
    scores: Sequence[CheckpointScore],
) -> list[dict[str, object]]:
    """Average the 30 checkpoint values within each graph replicate."""

    baseline: dict[str, list[CheckpointScore]] = defaultdict(list)
    nonzero: dict[
        tuple[str, int, int, str], list[CheckpointScore]
    ] = defaultdict(list)
    for score in scores:
        if score.p_percent == 0:
            baseline[score.model].append(score)
        else:
            assert score.replicate_index is not None
            nonzero[
                (
                    score.model,
                    score.p_percent,
                    score.replicate_index,
                    score.condition,
                )
            ].append(score)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        baseline_values = [score.macro_f1 for score in baseline[model]]
        if len(baseline_values) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(
                f"baseline/{model}: expected {EXPECTED_CHECKPOINTS} values"
            )
        baseline_mean = fmean(baseline_values)
        for replicate_index in GRAPH_REPLICATES:
            for condition in CONDITIONS:
                rows.append(
                    {
                        "dataset": DATASET,
                        "model": model,
                        "p_percent": 0,
                        "replicate_index": replicate_index,
                        "condition": condition,
                        "n_checkpoints": EXPECTED_CHECKPOINTS,
                        "mean_macro_f1": baseline_mean,
                    }
                )
        for p_percent in NONZERO_P_PERCENTAGES:
            for replicate_index in GRAPH_REPLICATES:
                for condition in CONDITIONS:
                    key = (model, p_percent, replicate_index, condition)
                    cell_scores = nonzero.get(key, [])
                    if len(cell_scores) != EXPECTED_CHECKPOINTS:
                        raise RuntimeError(
                            f"{model}/p={p_percent}/replicate="
                            f"{replicate_index}/{condition}: expected "
                            f"{EXPECTED_CHECKPOINTS} scores, found "
                            f"{len(cell_scores)}"
                        )
                    rows.append(
                        {
                            "dataset": DATASET,
                            "model": model,
                            "p_percent": p_percent,
                            "replicate_index": replicate_index,
                            "condition": condition,
                            "n_checkpoints": EXPECTED_CHECKPOINTS,
                            "mean_macro_f1": fmean(
                                score.macro_f1 for score in cell_scores
                            ),
                        }
                    )
    return rows


def build_dose_summary(
    replicate_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarise graph-replicate means at each percentage."""

    grouped: dict[
        tuple[str, int, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in replicate_rows:
        grouped[
            (
                str(row["model"]),
                int(row["p_percent"]),
                str(row["condition"]),
            )
        ].append(row)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for p_percent in P_PERCENTAGES:
            for condition in CONDITIONS:
                cell = grouped[(model, p_percent, condition)]
                replicate_indices = {
                    int(row["replicate_index"]) for row in cell
                }
                if (
                    len(cell) != len(GRAPH_REPLICATES)
                    or replicate_indices != set(GRAPH_REPLICATES)
                ):
                    raise RuntimeError(
                        f"{model}/p={p_percent}/{condition}: graph "
                        "replicate panel is incomplete"
                    )
                values = [float(row["mean_macro_f1"]) for row in cell]
                rows.append(
                    {
                        "dataset": DATASET,
                        "model": model,
                        "p_percent": p_percent,
                        "condition": condition,
                        "n_graph_replicates": len(values),
                        "n_checkpoints_per_graph": EXPECTED_CHECKPOINTS,
                        "mean_macro_f1": fmean(values),
                        "sd_macro_f1_across_graph_replicates": stdev(values),
                    }
                )
    return rows


def _filter_rows(
    rows: Sequence[dict[str, object]],
    *,
    model: str,
    condition: str,
) -> list[dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["model"] == model and row["condition"] == condition
        ),
        key=lambda row: int(row["p_percent"]),
    )


def build_figure(
    replicate_rows: Sequence[dict[str, object]],
    dose_rows: Sequence[dict[str, object]],
):
    """Build the registered two-panel PubMed dose-response figure."""

    colours = {"treatment": "#b4493f", "control": "#3d6f9c"}
    labels = {
        "treatment": "Homophily-directed rewiring",
        "control": "Random matched control",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.6), sharex=True, sharey=True
    )
    for axis, model in zip(axes, MODELS):
        for condition in CONDITIONS:
            for replicate_index in GRAPH_REPLICATES:
                replicate_curve = sorted(
                    (
                        row
                        for row in replicate_rows
                        if row["model"] == model
                        and row["condition"] == condition
                        and int(row["replicate_index"]) == replicate_index
                    ),
                    key=lambda row: int(row["p_percent"]),
                )
                axis.plot(
                    [int(row["p_percent"]) for row in replicate_curve],
                    [float(row["mean_macro_f1"]) for row in replicate_curve],
                    color=colours[condition],
                    linewidth=0.65,
                    alpha=0.18,
                    zorder=1,
                )
            summary_curve = _filter_rows(
                dose_rows, model=model, condition=condition
            )
            x = [int(row["p_percent"]) for row in summary_curve]
            mean = [float(row["mean_macro_f1"]) for row in summary_curve]
            sd = [
                float(row["sd_macro_f1_across_graph_replicates"])
                for row in summary_curve
            ]
            axis.fill_between(
                x,
                [value - spread for value, spread in zip(mean, sd)],
                [value + spread for value, spread in zip(mean, sd)],
                color=colours[condition],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            axis.plot(
                x,
                mean,
                color=colours[condition],
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                label=labels[condition],
                zorder=3,
            )
        title_model = "GCN" if model == "gcn" else "GraphSAGE-mean"
        axis.set_title(f"PubMed · {title_model}", fontsize=10)
        axis.set_xlabel("Tail nodes selected for rewiring (%)")
        axis.set_xlim(0, 100)
        axis.set_xticks((0, 20, 40, 60, 80, 100))
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Tail-test macro-F1")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    return figure


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


def build_analysis_manifest(
    *,
    scores: Sequence[CheckpointScore],
    input_files: Sequence[InputFile],
    contract: ExperimentContract,
    experiment_manifest_sha256: str,
    generated_outputs: dict[str, Path],
) -> dict:
    """Describe the validated inputs, estimands, and generated outputs."""

    return {
        "schema_version": 1,
        "analysis": "pubmed-edge-dose-response",
        "analysis_code": {
            "path": "scripts/analyze_pubmed_edge_dose_response.py",
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
            "dataset": DATASET,
            "models": list(contract.models),
            "p_percentages": list(contract.p_percentages),
            "graph_replicates": list(contract.graph_replicates),
            "metric_population": contract.metric_population,
            "tail_count": contract.tail_count,
            "actual_generation_attempts": (
                contract.actual_generation_attempts
            ),
            "checkpoint_identities_per_model": EXPECTED_CHECKPOINTS,
        },
        "estimand": {
            "checkpoint_aggregation": (
                "unweighted mean macro-F1 over the 30 registered ERM "
                "checkpoints within each graph replicate"
            ),
            "graph_aggregation": (
                "unweighted mean and sample standard deviation across the "
                "five graph-replicate means"
            ),
            "p_zero": (
                "one original-graph baseline per checkpoint copied to all "
                "five graph replicates"
            ),
        },
        "validation": {
            "complete_nonzero_graph_cells": (
                len(GRAPH_REPLICATES) * len(NONZERO_P_PERCENTAGES)
            ),
            "baseline_checkpoint_rows": len(
                [score for score in scores if score.p_percent == 0]
            ),
            "nonzero_condition_scores": len(
                [score for score in scores if score.p_percent > 0]
            ),
            "all_macro_f1_values_finite_and_in_unit_interval": True,
            "all_checkpoint_panels_complete": True,
            "tail_ranking_constant_within_each_graph_replicate": True,
            "tail_rankings_distinct_across_graph_replicates": True,
            "every_target_set_is_the_registered_ranking_prefix": True,
            "p100_target_population_identical_across_replicates": True,
            "original_graph_hash_constant_across_dose_cells": True,
            "rewired_graph_hashes_unique_across_dose_cells": True,
            "all_recorded_output_hashes_verified": True,
        },
        "input_files_sha256": {
            input_file.relative_path: input_file.sha256
            for input_file in sorted(
                input_files, key=lambda item: item.relative_path
            )
        },
        "generated_outputs_sha256": {
            path.name: file_sha256(path)
            for path in generated_outputs.values()
        },
    }


def analyze(input_root: Path, output_root: Path) -> dict[str, Path]:
    """Validate the actual store and write the figure and tidy summaries."""

    input_root = input_root.resolve()
    output_root = output_root.resolve()
    (
        scores,
        input_files,
        contract,
        experiment_manifest_sha256,
    ) = load_inputs(input_root)
    replicate_rows = build_replicate_summary(scores)
    dose_rows = build_dose_summary(replicate_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "replicate_summary": output_root / "replicate_summary.csv",
        "dose_summary": output_root / "dose_response_summary.csv",
        "figure": output_root / FIGURE_FILENAME,
        "analysis_manifest": output_root / "analysis_manifest.json",
    }
    _write_csv(
        outputs["replicate_summary"],
        replicate_rows,
        (
            "dataset",
            "model",
            "p_percent",
            "replicate_index",
            "condition",
            "n_checkpoints",
            "mean_macro_f1",
        ),
    )
    _write_csv(
        outputs["dose_summary"],
        dose_rows,
        (
            "dataset",
            "model",
            "p_percent",
            "condition",
            "n_graph_replicates",
            "n_checkpoints_per_graph",
            "mean_macro_f1",
            "sd_macro_f1_across_graph_replicates",
        ),
    )
    figure = build_figure(replicate_rows, dose_rows)
    try:
        figure.savefig(
            outputs["figure"],
            dpi=220,
            bbox_inches="tight",
            metadata={
                "Creator": "scripts/analyze_pubmed_edge_dose_response.py"
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
            scores=scores,
            input_files=input_files,
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
        help="actual PubMed dose-response root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="separate directory for the validated summaries and figure",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = analyze(args.input_root, args.output_root)
    print(
        "Validated the complete PubMed edge dose-response panel and wrote "
        f"{len(outputs)} outputs to {args.output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
