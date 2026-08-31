"""Validate and analyse one large edge dose-response dataset.

The input root may be either the ``actual`` directory or its selected dataset
directory.  The dataset layout is::

    <actual>/<dataset>/experiment_manifest.json
    <actual>/<dataset>/baseline/
        metadata.json
        checkpoint_metrics.csv
        predictions.npz
    <actual>/<dataset>/replicate-XXXX/pct-PPP/
        metadata.json
        graphs.npz
        swaps.json
        checkpoint_metrics.csv
        predictions.npz

For each graph replicate, the analysis takes the unweighted mean over the 30
registered checkpoints.  It then reports the mean and sample standard
deviation over the five graph-replicate means.  The original-graph baseline is
copied to all five graph replicates at p=0.
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


SUPPORTED_DATASETS = ("arxiv-year", "ogbn-arxiv")
DISPLAY_DATASETS = {
    "arxiv-year": "arXiv-year",
    "ogbn-arxiv": "ogbn-arxiv",
}
REGISTERED_DIRECTIONS = {
    "arxiv-year": {
        "tail_side": "high-h_v",
        "desired_same": False,
        "treatment_direction": "decrease-local-homophily",
        "generator": "singleton",
    },
    "ogbn-arxiv": {
        "tail_side": "low-h_v",
        "desired_same": True,
        "treatment_direction": "increase-local-homophily",
        "generator": "coordinated",
    },
}
DESIGN_ID = "large-tail-edge-dose-response-v1"
MODELS = ("gcn", "graphsage-mean")
P_PERCENTAGES = tuple(range(0, 101, 5))
NONZERO_P_PERCENTAGES = P_PERCENTAGES[1:]
GRAPH_REPLICATES = tuple(range(1, 6))
EXPECTED_CHECKPOINTS = 30
CONDITIONS = ("treatment", "control")
METRIC_POPULATION = "fixed-full-tail-test"
REPLICATE_PATTERN = re.compile(r"replicate-(\d{4})$")
PERCENTAGE_PATTERN = re.compile(r"pct-(\d{3})$")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
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
COORDINATED_REQUIRED_BOOLEAN_AUDITS = (
    "treatment_unselected_tail_homophily_unchanged",
    "control_full_tail_homophily_unchanged",
    "each_target_processed_once",
    "treatment_control_swap_counts_match",
)
GRAPH_HASH_FIELDS = (
    "original_edge_sha256",
    "treatment_edge_sha256",
    "control_edge_sha256",
)


@dataclass(frozen=True)
class ExperimentContract:
    design_id: str
    code_version: str
    dataset: str
    models: tuple[str, ...]
    p_percentages: tuple[int, ...]
    graph_replicates: tuple[int, ...]
    seed_base: int
    tail_side: str
    tail_count: int
    desired_same: bool
    treatment_direction: str
    generator: str
    graph_start: str
    checkpoint_identities: tuple[tuple[str, int], ...]
    metric_population: str
    actual_generation_attempts: int


@dataclass(frozen=True)
class CheckpointScore:
    dataset: str
    model: str
    split: str
    seed: int
    checkpoint_sha256: str
    p_percent: int
    replicate_index: int | None
    condition: str
    macro_f1: float
    n_tail_test: int
    tail_test_class_composition: str


@dataclass(frozen=True)
class InputFile:
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


def _require_string(value: object, *, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{path}: {field} must be a non-empty string")
    return value


def _parse_checkpoint_identities(
    value: object, *, path: Path
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{path}: checkpoint_identities must be a list")
    identities: list[tuple[str, int]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"{path}: checkpoint_identities[{index}] must be an object"
            )
        split = row.get("split")
        seed = row.get("seed")
        if not isinstance(split, str) or not split or not _is_int(seed):
            raise RuntimeError(
                f"{path}: invalid checkpoint identity at index {index}"
            )
        identities.append((split, seed))
    if len(identities) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_CHECKPOINTS} checkpoint identities; "
            f"found {len(identities)}"
        )
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"{path}: checkpoint identities must be unique")
    return tuple(identities)


def resolve_dataset_root(input_root: Path, dataset: str) -> Path:
    """Resolve either ``actual`` or ``actual/<dataset>`` input syntax."""

    root = input_root.expanduser().resolve()
    child = root / dataset
    if (child / "experiment_manifest.json").is_file():
        return child
    if root.name == dataset and (root / "experiment_manifest.json").is_file():
        return root
    raise RuntimeError(
        f"Cannot find {dataset}/experiment_manifest.json below {root}"
    )


def load_experiment_contract(
    path: Path, *, requested_dataset: str
) -> ExperimentContract:
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"{path}: schema_version must be 1")
    if manifest.get("design_id") != DESIGN_ID:
        raise RuntimeError(f"{path}: design_id must be {DESIGN_ID!r}")
    if manifest.get("dataset") != requested_dataset:
        raise RuntimeError(
            f"{path}: dataset must be {requested_dataset!r}"
        )
    if manifest.get("models") != list(MODELS):
        raise RuntimeError(f"{path}: models must be {list(MODELS)}")
    if manifest.get("p_percentages") != list(P_PERCENTAGES):
        raise RuntimeError(
            f"{path}: p_percentages must be {list(P_PERCENTAGES)}"
        )
    if manifest.get("graph_replicates") != len(GRAPH_REPLICATES):
        raise RuntimeError(
            f"{path}: graph_replicates must be {len(GRAPH_REPLICATES)}"
        )
    if manifest.get("graph_replicate_indices") != list(GRAPH_REPLICATES):
        raise RuntimeError(
            f"{path}: graph_replicate_indices must be "
            f"{list(GRAPH_REPLICATES)}"
        )
    seed_base = manifest.get("replicate_seed_base")
    if not _is_int(seed_base):
        raise RuntimeError(
            f"{path}: replicate_seed_base must be an integer"
        )
    tail_count = manifest.get("tail_count")
    if not _is_int(tail_count) or tail_count <= 0:
        raise RuntimeError(f"{path}: tail_count must be a positive integer")
    tail_side = manifest.get("tail_side")
    if tail_side not in {"low-h_v", "high-h_v"}:
        raise RuntimeError(f"{path}: invalid tail_side {tail_side!r}")
    desired_same = manifest.get("desired_same")
    if not isinstance(desired_same, bool):
        raise RuntimeError(f"{path}: desired_same must be Boolean")
    if desired_same != (tail_side == "low-h_v"):
        raise RuntimeError(
            f"{path}: desired_same is inconsistent with tail_side"
        )
    generator = manifest.get("generator")
    if generator not in {"singleton", "coordinated"}:
        raise RuntimeError(f"{path}: invalid generator {generator!r}")
    treatment_direction = _require_string(
        manifest.get("treatment_direction"),
        path=path,
        field="treatment_direction",
    )
    registered_direction = REGISTERED_DIRECTIONS[requested_dataset]
    observed_direction = {
        "tail_side": tail_side,
        "desired_same": desired_same,
        "treatment_direction": treatment_direction,
        "generator": generator,
    }
    if observed_direction != registered_direction:
        raise RuntimeError(
            f"{path}: treatment direction differs from the registered "
            f"{requested_dataset} contract"
        )
    if manifest.get("metric_population") != METRIC_POPULATION:
        raise RuntimeError(
            f"{path}: metric_population must be {METRIC_POPULATION!r}"
        )
    attempts = manifest.get("actual_generation_attempts")
    if attempts != 1:
        raise RuntimeError(
            f"{path}: actual_generation_attempts must be 1"
        )
    if manifest.get("n_checkpoints_per_model") != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            f"{path}: n_checkpoints_per_model must be {EXPECTED_CHECKPOINTS}"
        )
    graph_start = f"registered original {requested_dataset} graph"
    if manifest.get("graph_start_at_each_p") != graph_start:
        raise RuntimeError(
            f"{path}: graph_start_at_each_p must be {graph_start!r}"
        )
    if manifest.get("pair_output_files") != [
        *DOSE_OUTPUT_FILES,
        "metadata.json",
    ]:
        raise RuntimeError(f"{path}: pair_output_files differ from contract")
    if manifest.get("baseline_output_files") != [
        *BASELINE_OUTPUT_FILES,
        "metadata.json",
    ]:
        raise RuntimeError(
            f"{path}: baseline_output_files differ from contract"
        )

    return ExperimentContract(
        design_id=DESIGN_ID,
        code_version=_require_string(
            manifest.get("code_version"), path=path, field="code_version"
        ),
        dataset=requested_dataset,
        models=MODELS,
        p_percentages=P_PERCENTAGES,
        graph_replicates=GRAPH_REPLICATES,
        seed_base=seed_base,
        tail_side=tail_side,
        tail_count=tail_count,
        desired_same=desired_same,
        treatment_direction=treatment_direction,
        generator=generator,
        graph_start=graph_start,
        checkpoint_identities=_parse_checkpoint_identities(
            manifest.get("checkpoint_identities"), path=path
        ),
        metric_population=METRIC_POPULATION,
        actual_generation_attempts=attempts,
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
    if not fieldnames or not rows:
        raise RuntimeError(f"{path}: checkpoint metrics are empty")
    return rows, fieldnames


def _parse_identity(
    row: dict[str, str],
    *,
    path: Path,
    row_number: int,
    contract: ExperimentContract,
) -> tuple[str, str, int, str]:
    location = f"{path}:{row_number}"
    required = ("dataset", "model", "split", "seed", "checkpoint_sha256")
    missing = [field for field in required if field not in row]
    if missing:
        raise RuntimeError(f"{path}: missing required columns {missing}")
    dataset = row["dataset"].strip()
    model = row["model"].strip()
    split = row["split"].strip()
    if dataset != contract.dataset:
        raise RuntimeError(
            f"{location}: dataset must be {contract.dataset!r}"
        )
    if model not in contract.models:
        raise RuntimeError(f"{location}: unsupported model {model!r}")
    try:
        seed = int(row["seed"])
    except ValueError as exc:
        raise RuntimeError(f"{location}: seed must be an integer") from exc
    checkpoint_hash = row["checkpoint_sha256"].strip()
    if SHA256_PATTERN.fullmatch(checkpoint_hash) is None:
        raise RuntimeError(
            f"{location}: checkpoint_sha256 must be a SHA256"
        )
    return model, split, seed, checkpoint_hash


def _parse_metric(
    row: dict[str, str], column: str, *, path: Path, row_number: int
) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"{path}:{row_number}: {column} must be numeric"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(
            f"{path}:{row_number}: {column} must be finite and in [0, 1]"
        )
    return value


def _parse_tail_test_definition(
    row: dict[str, str], *, path: Path, row_number: int
) -> tuple[int, str]:
    location = f"{path}:{row_number}"
    try:
        n_tail_test = int(row["n_tail_test"])
        composition = json.loads(row["tail_test_class_composition"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{location}: invalid fixed tail-test definition"
        ) from exc
    if n_tail_test <= 0 or not isinstance(composition, dict) or not composition:
        raise RuntimeError(f"{location}: invalid fixed tail-test definition")
    if any(
        not _is_int(count) or count <= 0 for count in composition.values()
    ) or sum(composition.values()) != n_tail_test:
        raise RuntimeError(f"{location}: invalid tail-test class composition")
    return n_tail_test, json.dumps(
        composition, sort_keys=True, separators=(",", ":")
    )


def _validate_model_identity_panel(
    scores: Iterable[CheckpointScore],
    *,
    contract: ExperimentContract,
    cell: str,
) -> None:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for score in scores:
        grouped[score.model].append((score.split, score.seed))
    if set(grouped) != set(contract.models):
        raise RuntimeError(
            f"{cell}: expected models {list(contract.models)}; "
            f"found {sorted(grouped)}"
        )
    expected = set(contract.checkpoint_identities)
    for model in contract.models:
        identities = grouped[model]
        if len(identities) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(
                f"{cell}/{model}: expected {EXPECTED_CHECKPOINTS} rows; "
                f"found {len(identities)}"
            )
        if len(set(identities)) != len(identities):
            raise RuntimeError(
                f"{cell}/{model}: split/seed identities must be unique"
            )
        if set(identities) != expected:
            raise RuntimeError(
                f"{cell}/{model}: checkpoint identities differ from the "
                "registered panel"
            )


def _metadata_checkpoint_identities(
    metadata: dict, *, path: Path, contract: ExperimentContract
) -> None:
    observed = _parse_checkpoint_identities(
        metadata.get("checkpoint_identities"), path=path
    )
    if observed != contract.checkpoint_identities:
        raise RuntimeError(
            f"{path}: checkpoint_identities differ from experiment manifest"
        )


def _validate_output_hashes(
    metadata: dict,
    *,
    metadata_path: Path,
    output_dir: Path,
    dataset_root: Path,
    required_files: Sequence[str],
) -> list[InputFile]:
    """Verify each output and return its already-computed digest."""

    recorded = metadata.get("output_sha256")
    if not isinstance(recorded, dict) or set(recorded) != set(required_files):
        raise RuntimeError(
            f"{metadata_path}: output_sha256 keys must be "
            f"{list(required_files)}"
        )
    validated: list[InputFile] = []
    for filename in required_files:
        expected = recorded.get(filename)
        if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
            raise RuntimeError(
                f"{metadata_path}: invalid output hash for {filename}"
            )
        output_path = output_dir / filename
        if not output_path.is_file():
            raise RuntimeError(f"{metadata_path}: missing output {filename}")
        observed = file_sha256(output_path)
        if observed != expected:
            raise RuntimeError(
                f"{metadata_path}: output_sha256 mismatch for {filename}"
            )
        validated.append(
            InputFile(output_path.relative_to(dataset_root).as_posix(), observed)
        )
    return validated


def _validate_baseline_metadata(
    metadata: dict,
    *,
    metadata_path: Path,
    baseline_dir: Path,
    dataset_root: Path,
    contract: ExperimentContract,
    manifest_sha256: str,
) -> list[InputFile]:
    expected = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": contract.dataset,
        "models": list(contract.models),
        "metric_population": contract.metric_population,
        "n_checkpoints_per_model": EXPECTED_CHECKPOINTS,
        "n_checkpoint_rows": len(contract.models) * EXPECTED_CHECKPOINTS,
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
            f"{metadata_path}: baseline metadata differs from contract: "
            f"{mismatch}"
        )
    legacy_reproduction = (
        metadata.get("all_original_predictions_reproduced") is True
    )
    tail_reproduction = (
        metadata.get("all_original_tail_test_predictions_reproduced") is True
    )
    if not legacy_reproduction and not tail_reproduction:
        raise RuntimeError(
            f"{metadata_path}: fixed tail-test predictions were not "
            "reproduced"
        )
    if "all_original_tail_test_predictions_reproduced" in metadata:
        count = metadata.get(
            "n_checkpoints_with_full_graph_prediction_differences"
        )
        total = metadata.get("total_full_graph_prediction_differences")
        maximum = metadata.get(
            "max_full_graph_prediction_differences_per_checkpoint"
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (count, total, maximum)
        ):
            raise RuntimeError(
                f"{metadata_path}: invalid full-graph reproduction counts"
            )
        if metadata.get("all_original_predictions_reproduced") is not (
            total == 0
        ):
            raise RuntimeError(
                f"{metadata_path}: full-graph reproduction flag disagrees "
                "with its count"
            )
        records = metadata.get(
            "full_graph_prediction_differences_by_checkpoint"
        )
        if (
            not isinstance(records, list)
            or len(records) != count
            or any(not isinstance(record, dict) for record in records)
        ):
            raise RuntimeError(
                f"{metadata_path}: invalid full-graph reproduction records"
            )
        if (
            sum(record.get("count", -1) for record in records) != total
            or any(record.get("logits_allclose") is not True for record in records)
        ):
            raise RuntimeError(
                f"{metadata_path}: inconsistent full-graph reproduction records"
            )
    _metadata_checkpoint_identities(
        metadata, path=metadata_path, contract=contract
    )
    return _validate_output_hashes(
        metadata,
        metadata_path=metadata_path,
        output_dir=baseline_dir,
        dataset_root=dataset_root,
        required_files=BASELINE_OUTPUT_FILES,
    )


def _finite_audit_number(audit: dict, field: str, path: Path) -> float:
    value = audit.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{path}: audit.{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{path}: audit.{field} must be finite")
    return number


def _validate_audit_direction(
    audit: dict,
    *,
    metadata_path: Path,
    desired_same: bool,
    generator: str,
    requested_targets: int,
) -> None:
    required_audits = list(REQUIRED_BOOLEAN_AUDITS)
    if generator == "coordinated":
        required_audits.extend(COORDINATED_REQUIRED_BOOLEAN_AUDITS)
    failed = [field for field in required_audits if audit.get(field) is not True]
    if failed:
        raise RuntimeError(
            f"{metadata_path}: graph audits failed: {failed}"
        )
    if audit.get("num_targets") != requested_targets:
        raise RuntimeError(
            f"{metadata_path}: audit.num_targets differs from target count"
        )
    graph_hashes: list[str] = []
    for field in GRAPH_HASH_FIELDS:
        value = audit.get(field)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise RuntimeError(
                f"{metadata_path}: audit.{field} must be a SHA256"
            )
        graph_hashes.append(value)
    if len(set(graph_hashes)) != len(graph_hashes):
        raise RuntimeError(
            f"{metadata_path}: original, treatment and control hashes must differ"
        )

    original = _finite_audit_number(
        audit, "original_mean_target_h", metadata_path
    )
    treatment = _finite_audit_number(
        audit, "treatment_mean_target_h", metadata_path
    )
    control = _finite_audit_number(
        audit, "control_mean_target_h", metadata_path
    )
    treatment_change = _finite_audit_number(
        audit, "mean_treatment_change", metadata_path
    )
    control_change = _finite_audit_number(
        audit, "mean_control_change", metadata_path
    )
    if not math.isclose(control, original, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{metadata_path}: control mean target homophily changed"
        )
    if not math.isclose(control_change, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{metadata_path}: mean_control_change must be zero"
        )
    correct_direction = (
        treatment > original and treatment_change > 0.0
        if desired_same
        else treatment < original and treatment_change < 0.0
    )
    if not correct_direction:
        wanted = "increase" if desired_same else "decrease"
        raise RuntimeError(
            f"{metadata_path}: treatment must {wanted} target homophily"
        )
    if not math.isclose(
        treatment - original,
        treatment_change,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"{metadata_path}: mean_treatment_change is inconsistent"
        )


def _validate_dose_metadata(
    metadata: dict,
    *,
    metadata_path: Path,
    dose_dir: Path,
    dataset_root: Path,
    replicate_index: int,
    p_percent: int,
    contract: ExperimentContract,
    manifest_sha256: str,
) -> tuple[list[InputFile], dict]:
    requested_targets = (contract.tail_count * p_percent) // 100
    expected = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": contract.dataset,
        "models": list(contract.models),
        "tail_side": contract.tail_side,
        "desired_same": contract.desired_same,
        "treatment_direction": contract.treatment_direction,
        "generator": contract.generator,
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
        "graph_start": contract.graph_start,
        "generation_attempt": 0,
        "generation_retry_failures": [],
        "metric_population": contract.metric_population,
        "n_checkpoints_per_model": EXPECTED_CHECKPOINTS,
        "n_checkpoint_rows": len(contract.models) * EXPECTED_CHECKPOINTS,
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
            f"{metadata_path}: dose metadata differs from contract: {mismatch}"
        )
    _metadata_checkpoint_identities(
        metadata, path=metadata_path, contract=contract
    )
    audit = metadata.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError(f"{metadata_path}: audit must be an object")
    _validate_audit_direction(
        audit,
        metadata_path=metadata_path,
        desired_same=contract.desired_same,
        generator=contract.generator,
        requested_targets=requested_targets,
    )
    files = _validate_output_hashes(
        metadata,
        metadata_path=metadata_path,
        output_dir=dose_dir,
        dataset_root=dataset_root,
        required_files=DOSE_OUTPUT_FILES,
    )
    return files, audit


def _validate_graph_arrays(
    path: Path,
    *,
    contract: ExperimentContract,
    p_percent: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = (
                "treatment_edge_index",
                "control_edge_index",
                "targets",
                "target_order",
                "tail_ranking",
            )
            missing = [name for name in required if name not in arrays.files]
            if missing:
                raise RuntimeError(f"{path}: missing arrays {missing}")
            ranking = np.asarray(arrays["tail_ranking"], dtype=np.int64)
            targets = np.asarray(arrays["targets"], dtype=np.int64)
            target_order = np.asarray(arrays["target_order"], dtype=np.int64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    if (
        ranking.ndim != 1
        or ranking.size != contract.tail_count
        or np.unique(ranking).size != ranking.size
    ):
        raise RuntimeError(
            f"{path}: tail_ranking must contain {contract.tail_count} "
            "unique nodes"
        )
    expected_count = (contract.tail_count * p_percent) // 100
    if targets.ndim != 1 or targets.size != expected_count:
        raise RuntimeError(
            f"{path}: targets must contain exactly {expected_count} nodes"
        )
    if not np.array_equal(targets, ranking[:expected_count]):
        raise RuntimeError(f"{path}: targets are not the ranking prefix")
    if (
        target_order.ndim != 1
        or target_order.size != expected_count
        or not np.array_equal(np.sort(target_order), np.sort(targets))
    ):
        raise RuntimeError(
            f"{path}: target_order must be a permutation of targets"
        )
    return ranking, targets


def discover_dose_directories(
    dataset_root: Path,
) -> list[tuple[int, int, Path]]:
    discovered: dict[tuple[int, int], Path] = {}
    for replicate_dir in dataset_root.iterdir():
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
    if set(discovered) != expected:
        missing = sorted(expected - set(discovered))
        unexpected = sorted(set(discovered) - expected)
        raise RuntimeError(
            "Dose grid differs from the registered five-by-twenty grid; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [
        (replicate_index, p_percent, discovered[(replicate_index, p_percent)])
        for replicate_index in GRAPH_REPLICATES
        for p_percent in NONZERO_P_PERCENTAGES
    ]


def load_baseline(
    path: Path, *, contract: ExperimentContract
) -> list[CheckpointScore]:
    rows, fieldnames = _read_rows(path)
    if "tail_test_macro_f1" not in fieldnames:
        raise RuntimeError(f"{path}: missing tail_test_macro_f1")
    reproduction_fields = {
        "n_full_graph_prediction_differences_from_parent",
        "n_tail_test_prediction_differences_from_parent",
        "full_graph_prediction_difference_nodes_from_parent",
        "full_graph_logit_reproduction_checked",
        "max_abs_logit_difference_from_parent",
        "mean_abs_logit_difference_from_parent",
    }
    present_reproduction_fields = reproduction_fields.intersection(fieldnames)
    if present_reproduction_fields and (
        present_reproduction_fields != reproduction_fields
    ):
        raise RuntimeError(f"{path}: incomplete reproduction columns")
    scores: list[CheckpointScore] = []
    for row_number, row in enumerate(rows, start=2):
        model, split, seed, checkpoint_hash = _parse_identity(
            row,
            path=path,
            row_number=row_number,
            contract=contract,
        )
        n_tail_test, composition = _parse_tail_test_definition(
            row, path=path, row_number=row_number
        )
        if present_reproduction_fields:
            try:
                full_graph_differences = int(
                    row[
                        "n_full_graph_prediction_differences_from_parent"
                    ]
                )
                tail_test_differences = int(
                    row[
                        "n_tail_test_prediction_differences_from_parent"
                    ]
                )
                difference_nodes = json.loads(
                    row[
                        "full_graph_prediction_difference_nodes_from_parent"
                    ]
                )
                logit_reproduction_checked = row[
                    "full_graph_logit_reproduction_checked"
                ]
                max_abs_logit_difference = float(
                    row["max_abs_logit_difference_from_parent"]
                )
                mean_abs_logit_difference = float(
                    row["mean_abs_logit_difference_from_parent"]
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"{path}:{row_number}: invalid reproduction record"
                ) from exc
            if (
                full_graph_differences < 0
                or tail_test_differences < 0
                or tail_test_differences > full_graph_differences
                or not isinstance(difference_nodes, list)
                or len(difference_nodes) != full_graph_differences
                or any(
                    not isinstance(node, int) or isinstance(node, bool) or node < 0
                    for node in difference_nodes
                )
                or len(set(difference_nodes)) != len(difference_nodes)
                or logit_reproduction_checked not in {"True", "False"}
                or not math.isfinite(max_abs_logit_difference)
                or not math.isfinite(mean_abs_logit_difference)
                or max_abs_logit_difference < 0.0
                or mean_abs_logit_difference < 0.0
            ):
                raise RuntimeError(
                    f"{path}:{row_number}: inconsistent reproduction record"
                )
            if tail_test_differences:
                raise RuntimeError(
                    f"{path}:{row_number}: fixed tail-test predictions differ "
                    "from the parent run"
                )
            if full_graph_differences and logit_reproduction_checked != "True":
                raise RuntimeError(
                    f"{path}:{row_number}: full-graph prediction differences "
                    "lack a saved-logit reproduction check"
                )
        scores.append(
            CheckpointScore(
                dataset=contract.dataset,
                model=model,
                split=split,
                seed=seed,
                checkpoint_sha256=checkpoint_hash,
                p_percent=0,
                replicate_index=None,
                condition="baseline",
                macro_f1=_parse_metric(
                    row,
                    "tail_test_macro_f1",
                    path=path,
                    row_number=row_number,
                ),
                n_tail_test=n_tail_test,
                tail_test_class_composition=composition,
            )
        )
    _validate_model_identity_panel(
        scores, contract=contract, cell="baseline"
    )
    return scores


def load_dose_cell(
    path: Path,
    *,
    contract: ExperimentContract,
    replicate_index: int,
    p_percent: int,
) -> list[CheckpointScore]:
    rows, fieldnames = _read_rows(path)
    required_columns = (
        "treatment_macro_f1",
        "control_macro_f1",
        "treatment_minus_control_macro_f1",
    )
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise RuntimeError(f"{path}: missing required columns {missing}")
    by_condition: dict[str, list[CheckpointScore]] = {
        condition: [] for condition in CONDITIONS
    }
    for row_number, row in enumerate(rows, start=2):
        model, split, seed, checkpoint_hash = _parse_identity(
            row,
            path=path,
            row_number=row_number,
            contract=contract,
        )
        try:
            row_replicate = int(row["replicate_index"])
            row_percentage = int(row["p_percent"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"{path}:{row_number}: invalid replicate or percentage"
            ) from exc
        if row_replicate != replicate_index or row_percentage != p_percent:
            raise RuntimeError(
                f"{path}:{row_number}: row differs from containing dose cell"
            )
        n_tail_test, composition = _parse_tail_test_definition(
            row, path=path, row_number=row_number
        )
        treatment = _parse_metric(
            row,
            "treatment_macro_f1",
            path=path,
            row_number=row_number,
        )
        control = _parse_metric(
            row,
            "control_macro_f1",
            path=path,
            row_number=row_number,
        )
        try:
            recorded_effect = float(row["treatment_minus_control_macro_f1"])
        except ValueError as exc:
            raise RuntimeError(
                f"{path}:{row_number}: treatment-control effect must be numeric"
            ) from exc
        if not math.isfinite(recorded_effect) or not math.isclose(
            recorded_effect,
            treatment - control,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{path}:{row_number}: treatment-control effect is inconsistent"
            )
        for condition, value in (
            ("treatment", treatment),
            ("control", control),
        ):
            by_condition[condition].append(
                CheckpointScore(
                    dataset=contract.dataset,
                    model=model,
                    split=split,
                    seed=seed,
                    checkpoint_sha256=checkpoint_hash,
                    p_percent=p_percent,
                    replicate_index=replicate_index,
                    condition=condition,
                    macro_f1=value,
                    n_tail_test=n_tail_test,
                    tail_test_class_composition=composition,
                )
            )
    for condition in CONDITIONS:
        _validate_model_identity_panel(
            by_condition[condition],
            contract=contract,
            cell=(
                f"replicate-{replicate_index:04d}/"
                f"pct-{p_percent:03d}/{condition}"
            ),
        )
    return by_condition["treatment"] + by_condition["control"]


def _validate_fixed_tail_test_subsets(
    scores: Sequence[CheckpointScore],
) -> None:
    definitions: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for score in scores:
        definitions[score.split].add(
            (score.n_tail_test, score.tail_test_class_composition)
        )
    inconsistent = {
        split: values for split, values in definitions.items() if len(values) != 1
    }
    if inconsistent:
        raise RuntimeError(
            "The fixed tail-test subset changes across checkpoints or doses"
        )


def _validate_checkpoint_hashes(scores: Sequence[CheckpointScore]) -> None:
    hashes: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for score in scores:
        hashes[(score.model, score.split, score.seed)].add(
            score.checkpoint_sha256
        )
    inconsistent = [key for key, values in hashes.items() if len(values) != 1]
    if inconsistent:
        raise RuntimeError(
            "Checkpoint hashes change across the baseline or dose cells: "
            f"{inconsistent[:5]}"
        )


def load_inputs(
    input_root: Path, *, dataset: str
) -> tuple[
    list[CheckpointScore],
    list[InputFile],
    ExperimentContract,
    str,
    Path,
]:
    dataset_root = resolve_dataset_root(input_root, dataset)
    manifest_path = dataset_root / "experiment_manifest.json"
    contract = load_experiment_contract(
        manifest_path, requested_dataset=dataset
    )
    manifest_sha256 = file_sha256(manifest_path)
    input_files = [InputFile("experiment_manifest.json", manifest_sha256)]

    baseline_dir = dataset_root / "baseline"
    baseline_metadata_path = baseline_dir / "metadata.json"
    baseline_metadata = _read_json(baseline_metadata_path)
    input_files.append(
        InputFile(
            baseline_metadata_path.relative_to(dataset_root).as_posix(),
            file_sha256(baseline_metadata_path),
        )
    )
    input_files.extend(
        _validate_baseline_metadata(
            baseline_metadata,
            metadata_path=baseline_metadata_path,
            baseline_dir=baseline_dir,
            dataset_root=dataset_root,
            contract=contract,
            manifest_sha256=manifest_sha256,
        )
    )
    scores = load_baseline(
        baseline_dir / "checkpoint_metrics.csv", contract=contract
    )

    ranking_by_replicate: dict[int, np.ndarray] = {}
    p100_population: np.ndarray | None = None
    original_graph_hash: str | None = None
    rewired_hash_owners: dict[str, str] = {}
    for replicate_index, p_percent, dose_dir in discover_dose_directories(
        dataset_root
    ):
        metadata_path = dose_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        input_files.append(
            InputFile(
                metadata_path.relative_to(dataset_root).as_posix(),
                file_sha256(metadata_path),
            )
        )
        verified_files, audit = _validate_dose_metadata(
            metadata,
            metadata_path=metadata_path,
            dose_dir=dose_dir,
            dataset_root=dataset_root,
            replicate_index=replicate_index,
            p_percent=p_percent,
            contract=contract,
            manifest_sha256=manifest_sha256,
        )
        input_files.extend(verified_files)
        ranking, targets = _validate_graph_arrays(
            dose_dir / "graphs.npz",
            contract=contract,
            p_percent=p_percent,
        )
        previous = ranking_by_replicate.get(replicate_index)
        if previous is None:
            ranking_by_replicate[replicate_index] = ranking
        elif not np.array_equal(previous, ranking):
            raise RuntimeError(
                f"replicate-{replicate_index:04d}: tail ranking changes across p"
            )
        if p_percent == 100:
            population = np.sort(targets)
            if p100_population is None:
                p100_population = population
            elif not np.array_equal(p100_population, population):
                raise RuntimeError(
                    "The p=100 target population differs across graph replicates"
                )

        observed_original = str(audit["original_edge_sha256"])
        if original_graph_hash is None:
            original_graph_hash = observed_original
        elif original_graph_hash != observed_original:
            raise RuntimeError(
                "The original graph hash changes across dose cells"
            )
        for condition in CONDITIONS:
            graph_hash = str(audit[f"{condition}_edge_sha256"])
            owner = (
                f"replicate-{replicate_index:04d}/"
                f"pct-{p_percent:03d}/{condition}"
            )
            if graph_hash in rewired_hash_owners:
                raise RuntimeError(
                    f"Rewired graph hash is reused by "
                    f"{rewired_hash_owners[graph_hash]} and {owner}"
                )
            rewired_hash_owners[graph_hash] = owner
        scores.extend(
            load_dose_cell(
                dose_dir / "checkpoint_metrics.csv",
                contract=contract,
                replicate_index=replicate_index,
                p_percent=p_percent,
            )
        )

    ranking_hashes = {
        hashlib.sha256(ranking.tobytes()).hexdigest()
        for ranking in ranking_by_replicate.values()
    }
    if len(ranking_hashes) != len(GRAPH_REPLICATES):
        raise RuntimeError("Graph replicates must use distinct tail rankings")
    if p100_population is None or p100_population.size != contract.tail_count:
        raise RuntimeError("The complete p=100 target population is missing")
    _validate_fixed_tail_test_subsets(scores)
    _validate_checkpoint_hashes(scores)
    return scores, input_files, contract, manifest_sha256, dataset_root


def build_replicate_summary(
    scores: Sequence[CheckpointScore], contract: ExperimentContract
) -> list[dict[str, object]]:
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

    output: list[dict[str, object]] = []
    for model in contract.models:
        baseline_values = [score.macro_f1 for score in baseline[model]]
        if len(baseline_values) != EXPECTED_CHECKPOINTS:
            raise RuntimeError(f"baseline/{model}: incomplete checkpoint panel")
        baseline_mean = fmean(baseline_values)
        for replicate_index in contract.graph_replicates:
            for condition in CONDITIONS:
                output.append(
                    {
                        "dataset": contract.dataset,
                        "model": model,
                        "p_percent": 0,
                        "replicate_index": replicate_index,
                        "condition": condition,
                        "n_checkpoints": EXPECTED_CHECKPOINTS,
                        "mean_macro_f1": baseline_mean,
                    }
                )
        for p_percent in NONZERO_P_PERCENTAGES:
            for replicate_index in contract.graph_replicates:
                for condition in CONDITIONS:
                    cell = nonzero[
                        (model, p_percent, replicate_index, condition)
                    ]
                    if len(cell) != EXPECTED_CHECKPOINTS:
                        raise RuntimeError(
                            f"{model}/p={p_percent}/replicate="
                            f"{replicate_index}/{condition}: incomplete panel"
                        )
                    output.append(
                        {
                            "dataset": contract.dataset,
                            "model": model,
                            "p_percent": p_percent,
                            "replicate_index": replicate_index,
                            "condition": condition,
                            "n_checkpoints": EXPECTED_CHECKPOINTS,
                            "mean_macro_f1": fmean(
                                score.macro_f1 for score in cell
                            ),
                        }
                    )
    return output


def build_dose_summary(
    replicate_rows: Sequence[dict[str, object]],
    contract: ExperimentContract,
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[str, int, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in replicate_rows:
        grouped[
            (str(row["model"]), int(row["p_percent"]), str(row["condition"]))
        ].append(row)
    output: list[dict[str, object]] = []
    for model in contract.models:
        for p_percent in contract.p_percentages:
            for condition in CONDITIONS:
                cell = grouped[(model, p_percent, condition)]
                indices = {int(row["replicate_index"]) for row in cell}
                if len(cell) != 5 or indices != set(GRAPH_REPLICATES):
                    raise RuntimeError(
                        f"{model}/p={p_percent}/{condition}: incomplete graph panel"
                    )
                values = [float(row["mean_macro_f1"]) for row in cell]
                output.append(
                    {
                        "dataset": contract.dataset,
                        "model": model,
                        "p_percent": p_percent,
                        "condition": condition,
                        "n_graph_replicates": len(values),
                        "n_checkpoints_per_graph": EXPECTED_CHECKPOINTS,
                        "mean_macro_f1": fmean(values),
                        "sd_macro_f1_across_graph_replicates": stdev(values),
                    }
                )
    return output


def build_figure(
    replicate_rows: Sequence[dict[str, object]],
    dose_rows: Sequence[dict[str, object]],
    contract: ExperimentContract,
):
    colours = {"treatment": "#b4493f", "control": "#3d6f9c"}
    labels = {
        "treatment": r"Tail $h_v$ moved towards graph mean",
        "control": "Degree-matched random control",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.6), sharex=True, sharey=True
    )
    for axis, model in zip(axes, contract.models):
        for condition in CONDITIONS:
            for replicate_index in contract.graph_replicates:
                curve = sorted(
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
                    [int(row["p_percent"]) for row in curve],
                    [float(row["mean_macro_f1"]) for row in curve],
                    color=colours[condition],
                    linewidth=0.65,
                    alpha=0.18,
                    zorder=1,
                )
            summary = sorted(
                (
                    row
                    for row in dose_rows
                    if row["model"] == model
                    and row["condition"] == condition
                ),
                key=lambda row: int(row["p_percent"]),
            )
            x = [int(row["p_percent"]) for row in summary]
            mean = [float(row["mean_macro_f1"]) for row in summary]
            sd = [
                float(row["sd_macro_f1_across_graph_replicates"])
                for row in summary
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
        display_model = "GCN" if model == "gcn" else "GraphSAGE-mean"
        axis.set_title(
            f"{DISPLAY_DATASETS[contract.dataset]} · {display_model}",
            fontsize=10,
        )
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
    return {
        "schema_version": 1,
        "analysis": "large-edge-dose-response",
        "analysis_code": {
            "path": "scripts/analyze_large_edge_dose_response.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "experiment_manifest": {
            "path": "experiment_manifest.json",
            "sha256": experiment_manifest_sha256,
        },
        "design": {
            "design_id": contract.design_id,
            "code_version": contract.code_version,
            "dataset": contract.dataset,
            "models": list(contract.models),
            "p_percentages": list(contract.p_percentages),
            "graph_replicates": list(contract.graph_replicates),
            "seed_base": contract.seed_base,
            "tail_side": contract.tail_side,
            "tail_count": contract.tail_count,
            "desired_same": contract.desired_same,
            "treatment_direction": contract.treatment_direction,
            "generator": contract.generator,
            "metric_population": contract.metric_population,
            "checkpoint_identities": [
                {"split": split, "seed": seed}
                for split, seed in contract.checkpoint_identities
            ],
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
            "complete_nonzero_graph_cells": 100,
            "baseline_checkpoint_rows": len(
                [score for score in scores if score.p_percent == 0]
            ),
            "nonzero_condition_scores": len(
                [score for score in scores if score.p_percent > 0]
            ),
            "all_checkpoint_panels_complete": True,
            "checkpoint_hashes_constant_across_graphs": True,
            "fixed_tail_test_definitions_constant": True,
            "tail_ranking_constant_within_graph_replicate": True,
            "tail_rankings_distinct_across_graph_replicates": True,
            "every_target_set_is_the_registered_ranking_prefix": True,
            "p100_target_population_identical_across_replicates": True,
            "treatment_direction_matches_registered_tail_side": True,
            "original_graph_hash_constant_within_dataset": True,
            "rewired_graph_hashes_unique_within_dataset": True,
            "all_recorded_output_hashes_verified": True,
        },
        "input_files_sha256": {
            item.relative_path: item.sha256
            for item in sorted(input_files, key=lambda value: value.relative_path)
        },
        "generated_outputs_sha256": {
            path.name: file_sha256(path) for path in generated_outputs.values()
        },
    }


def analyze(
    input_root: Path,
    output_root: Path,
    *,
    dataset: str,
) -> dict[str, Path]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    (
        scores,
        input_files,
        contract,
        manifest_sha256,
        _,
    ) = load_inputs(input_root, dataset=dataset)
    replicate_rows = build_replicate_summary(scores, contract)
    dose_rows = build_dose_summary(replicate_rows, contract)

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "replicate_summary": output_root / "replicate_summary.csv",
        "dose_summary": output_root / "dose_response_summary.csv",
        "figure": output_root / f"{dataset}_edge_dose_response.png",
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
    figure = build_figure(replicate_rows, dose_rows, contract)
    try:
        figure.savefig(
            outputs["figure"],
            dpi=220,
            bbox_inches="tight",
            metadata={
                "Creator": "scripts/analyze_large_edge_dose_response.py"
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
            experiment_manifest_sha256=manifest_sha256,
            generated_outputs=generated_outputs,
        ),
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, required=True
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="actual root or selected dataset directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="dataset-specific analysis output directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = analyze(
        args.input_root, args.output_root, dataset=args.dataset
    )
    print(
        f"Validated {args.dataset} and wrote {len(outputs)} outputs to "
        f"{args.output_root.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
