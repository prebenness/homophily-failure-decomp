#!/usr/bin/env python3
"""Validate and aggregate complete Appendix J dose-response experiments.

The post-submission expansion is stored as two consolidated JSONL files plus
the validator record written by ``run_edge_dose_response_expansion.py``.  This
script authenticates those files, validates the complete treatment-control
pairing, aggregates over the registered checkpoints and graph replicates, and
combines the results with the six submitted GCN/GraphSAGE cells.

For last, head, and full preparations, the input release already contains all
30 model--dataset cells, so no submitted ERM summaries are mixed in.

Only the compact CSV outputs and their provenance manifest are tracked.  The
large checkpoint-level JSONL inputs remain in the ignored experiment store.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "edge-dose-response-expansion"
    / "actual"
    / "final"
)
DEFAULT_RECOVERY_ROOT = (
    REPO_ROOT / "outputs" / "edge-dose-response-recovery"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "edge_perturbation_dose_response_full"
)
DESIGN_ID = "appendix-j-edge-dose-response-expansion-v1"
EXPANSION_CODE_VERSION = "f66a53999fc01ae52a1e4aeaaf618409abab8fa7"
PREPARATIONS = ("erm", "last", "head", "full")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONDITIONS = ("treatment", "control")
ALL_PERCENTAGES = tuple(range(0, 101, 5))
ROMAN_PERCENTAGES = tuple(range(0, 71, 5))
DATASETS = (
    "pubmed",
    "ogbn-arxiv",
    "roman-empire",
    "amazon-ratings",
    "arxiv-year",
)
MODELS = (
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
EXPANSION_CELLS = (
    *(('pubmed', model) for model in ('acm-gcn', 'gpr-gnn', 'h2gcn', 'linkx')),
    *(('ogbn-arxiv', model) for model in ('acm-gcn', 'gpr-gnn', 'h2gcn', 'linkx')),
    *(('arxiv-year', model) for model in ('acm-gcn', 'gpr-gnn', 'h2gcn', 'linkx')),
    *(('roman-empire', model) for model in MODELS),
    *(('amazon-ratings', model) for model in MODELS),
)
PRIOR_DATASETS = ("pubmed", "ogbn-arxiv", "arxiv-year")
PRIOR_MODELS = ("gcn", "graphsage-mean")
FULL_MATRIX_CELLS = tuple(
    (dataset, model) for dataset in DATASETS for model in MODELS
)

REPLICATE_FIELDS = (
    "dataset",
    "model",
    "p_percent",
    "replicate_index",
    "condition",
    "n_checkpoints",
    "mean_macro_f1",
)
DOSE_FIELDS = (
    "dataset",
    "model",
    "p_percent",
    "condition",
    "n_graph_replicates",
    "n_checkpoints_per_graph",
    "mean_macro_f1",
    "sd_macro_f1_across_graph_replicates",
)
CSV_FLOAT_FIELDS = frozenset(
    {
        "mean_macro_f1",
        "sd_macro_f1_across_graph_replicates",
    }
)
CSV_FLOAT_DECIMAL_PLACES = 14


@dataclass(frozen=True)
class ExpansionProtocol:
    cells: tuple[tuple[str, str], ...]
    checkpoints_per_cell: int
    replicates: tuple[int, ...]
    percentages_by_dataset: Mapping[str, tuple[int, ...]]
    design_id: str = DESIGN_ID
    preparation: str = "erm"

    def nonzero_percentages(self, dataset: str) -> tuple[int, ...]:
        return tuple(
            percentage
            for percentage in self.percentages_by_dataset[dataset]
            if percentage != 0
        )

    @property
    def expected_baselines(self) -> int:
        return len(self.cells) * self.checkpoints_per_cell

    @property
    def expected_paired_rows(self) -> int:
        return sum(
            self.checkpoints_per_cell
            * len(self.replicates)
            * len(self.nonzero_percentages(dataset))
            for dataset, _ in self.cells
        )


DEFAULT_PROTOCOL = ExpansionProtocol(
    cells=tuple(EXPANSION_CELLS),
    checkpoints_per_cell=30,
    replicates=(1, 2, 3, 4, 5),
    percentages_by_dataset={
        dataset: ROMAN_PERCENTAGES if dataset == "roman-empire" else ALL_PERCENTAGES
        for dataset in DATASETS
    },
)


def design_id_for_preparation(preparation: str) -> str:
    if preparation not in PREPARATIONS:
        raise ValueError(f"unknown preparation {preparation!r}")
    if preparation == "erm":
        return DESIGN_ID
    return f"appendix-j-edge-dose-response-{preparation}-full-matrix-v1"


def protocol_for_preparation(preparation: str) -> ExpansionProtocol:
    if preparation == "erm":
        return DEFAULT_PROTOCOL
    return ExpansionProtocol(
        cells=FULL_MATRIX_CELLS,
        checkpoints_per_cell=30,
        replicates=(1, 2, 3, 4, 5),
        percentages_by_dataset={
            dataset: ROMAN_PERCENTAGES
            if dataset == "roman-empire"
            else ALL_PERCENTAGES
            for dataset in DATASETS
        },
        design_id=design_id_for_preparation(preparation),
        preparation=preparation,
    )


def default_final_root(preparation: str) -> Path:
    if preparation == "erm":
        return DEFAULT_FINAL_ROOT
    return (
        DEFAULT_RECOVERY_ROOT
        / "actual"
        / preparation
        / "final"
    )


def default_output_root(preparation: str) -> Path:
    if preparation == "erm":
        return DEFAULT_OUTPUT_ROOT
    return (
        REPO_ROOT
        / "results"
        / f"edge_perturbation_dose_response_preparation_{preparation}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RuntimeError(f"{path}:{line_number}: blank JSONL row")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                rows.append(row)
    except OSError as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    return rows


def _unit_interval(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{location}: expected a numeric value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{location}: expected a finite value in [0, 1]")
    return result


def _integer(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{location}: expected an integer")
    return value


def _validate_final_record(
    final_root: Path,
    protocol: ExpansionProtocol,
) -> tuple[dict, dict[str, str]]:
    validation_path = final_root / "validation.json"
    baseline_path = final_root / "baseline_rows.jsonl"
    paired_path = final_root / "paired_rows.jsonl"
    for path in (validation_path, baseline_path, paired_path):
        if not path.is_file():
            raise FileNotFoundError(f"required expansion output not found: {path}")

    validation = _read_json_object(validation_path)
    if validation.get("preparation", "erm") != protocol.preparation:
        raise RuntimeError(
            f"{validation_path}: preparation differs from the requested protocol"
        )
    expected_fields = {
        "schema_version": 1,
        "design_id": protocol.design_id,
        "validated": True,
        "n_model_dataset_cells": len(protocol.cells),
        "n_checkpoint_shards": protocol.expected_baselines,
        "n_baseline_rows": protocol.expected_baselines,
        "n_paired_rows": protocol.expected_paired_rows,
        "n_unique_treatment_prediction_keys": protocol.expected_paired_rows,
        "n_unique_control_prediction_keys": protocol.expected_paired_rows,
    }
    if protocol.preparation != "erm":
        expected_fields["preparation"] = protocol.preparation
    for field, expected in expected_fields.items():
        if validation.get(field) != expected:
            raise RuntimeError(
                f"{validation_path}: {field} must be {expected!r}; "
                f"found {validation.get(field)!r}"
            )

    recorded_hashes = validation.get("output_sha256")
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError(f"{validation_path}: output_sha256 must be an object")
    consolidated_hashes = {
        "baseline_rows.jsonl": file_sha256(baseline_path),
        "paired_rows.jsonl": file_sha256(paired_path),
    }
    if recorded_hashes != consolidated_hashes:
        raise RuntimeError(
            f"{validation_path}: consolidated output hashes do not match"
        )
    actual_hashes = {
        **consolidated_hashes,
        "validation.json": file_sha256(validation_path),
    }

    catalog_hashes = validation.get("graph_catalog_sha256")
    expected_datasets = {dataset for dataset, _ in protocol.cells}
    if not isinstance(catalog_hashes, dict) or set(catalog_hashes) != expected_datasets:
        raise RuntimeError(
            f"{validation_path}: graph catalog hashes do not cover the datasets"
        )
    if any(not _is_sha256(value) for value in catalog_hashes.values()):
        raise RuntimeError(f"{validation_path}: invalid graph catalog SHA256")
    return validation, actual_hashes


def validate_expansion_inputs(
    final_root: Path,
    *,
    protocol: ExpansionProtocol = DEFAULT_PROTOCOL,
) -> tuple[list[dict], list[dict], dict]:
    """Authenticate and validate every checkpoint and paired prediction row."""

    final_root = final_root.resolve()
    validation, actual_hashes = _validate_final_record(final_root, protocol)
    baseline_path = final_root / "baseline_rows.jsonl"
    paired_path = final_root / "paired_rows.jsonl"
    baselines = _read_jsonl(baseline_path)
    pairs = _read_jsonl(paired_path)
    if len(baselines) != protocol.expected_baselines:
        raise RuntimeError(
            f"{baseline_path}: expected {protocol.expected_baselines} rows; "
            f"found {len(baselines)}"
        )
    if len(pairs) != protocol.expected_paired_rows:
        raise RuntimeError(
            f"{paired_path}: expected {protocol.expected_paired_rows} rows; "
            f"found {len(pairs)}"
        )

    expected_cells = set(protocol.cells)
    baselines_by_run: dict[str, dict] = {}
    baseline_prediction_keys: set[str] = set()
    runs_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, row in enumerate(baselines, start=1):
        location = f"{baseline_path}:{index}"
        dataset = row.get("dataset")
        model = row.get("model")
        cell = (dataset, model)
        if cell not in expected_cells:
            raise RuntimeError(f"{location}: unexpected model-dataset cell {cell}")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(f"{location}: run_id must be a non-empty string")
        if run_id in baselines_by_run:
            raise RuntimeError(f"{location}: duplicate run_id {run_id}")
        prediction_key = row.get("prediction_key")
        if not isinstance(prediction_key, str) or not prediction_key:
            raise RuntimeError(f"{location}: invalid prediction_key")
        if prediction_key in baseline_prediction_keys:
            raise RuntimeError(f"{location}: duplicate prediction_key")
        if not _is_sha256(row.get("checkpoint_sha256")):
            raise RuntimeError(f"{location}: invalid checkpoint_sha256")
        if not _is_sha256(row.get("prediction_sha256")):
            raise RuntimeError(f"{location}: invalid prediction_sha256")
        _unit_interval(row.get("tail_test_macro_f1"), location=location)
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise RuntimeError(f"{location}: split must be a non-empty string")
        seed = _integer(row.get("seed"), location=f"{location}:seed")
        if row.get("preparation", "erm") != protocol.preparation:
            raise RuntimeError(f"{location}: preparation differs from protocol")
        if protocol.preparation != "erm":
            expected_run_id = (
                f"{dataset}_{model}_{protocol.preparation}_{split}_seed-{seed}"
            )
            if run_id != expected_run_id:
                raise RuntimeError(
                    f"{location}: run_id does not match preparation identity"
                )
        baselines_by_run[run_id] = row
        baseline_prediction_keys.add(prediction_key)
        runs_by_cell[cell].add(run_id)

    if set(runs_by_cell) != expected_cells:
        raise RuntimeError("baseline rows do not cover every expansion cell")
    for cell in protocol.cells:
        if len(runs_by_cell[cell]) != protocol.checkpoints_per_cell:
            raise RuntimeError(
                f"{cell}: expected {protocol.checkpoints_per_cell} checkpoints; "
                f"found {len(runs_by_cell[cell])}"
            )

    treatment_keys: set[str] = set()
    control_keys: set[str] = set()
    observed_by_run: dict[str, set[tuple[int, int]]] = defaultdict(set)
    graph_signatures: dict[tuple[str, int, int], tuple[object, ...]] = {}
    for index, row in enumerate(pairs, start=1):
        location = f"{paired_path}:{index}"
        dataset = row.get("dataset")
        model = row.get("model")
        cell = (dataset, model)
        if cell not in expected_cells:
            raise RuntimeError(f"{location}: unexpected model-dataset cell {cell}")
        run_id = row.get("run_id")
        baseline = baselines_by_run.get(run_id)
        if baseline is None:
            raise RuntimeError(f"{location}: no matching baseline for {run_id!r}")
        if (baseline["dataset"], baseline["model"]) != cell:
            raise RuntimeError(f"{location}: run_id belongs to a different cell")
        if row.get("preparation", "erm") != protocol.preparation:
            raise RuntimeError(f"{location}: preparation differs from protocol")
        for field in ("checkpoint_sha256", "split", "seed"):
            if row.get(field) != baseline.get(field):
                raise RuntimeError(
                    f"{location}: {field} does not match the baseline row"
                )

        replicate = _integer(
            row.get("replicate_index"), location=f"{location}:replicate_index"
        )
        percentage = _integer(
            row.get("p_percent"), location=f"{location}:p_percent"
        )
        if replicate not in protocol.replicates:
            raise RuntimeError(f"{location}: unexpected graph replicate {replicate}")
        if percentage not in protocol.nonzero_percentages(str(dataset)):
            raise RuntimeError(f"{location}: unexpected percentage {percentage}")
        pair_key = (replicate, percentage)
        if pair_key in observed_by_run[str(run_id)]:
            raise RuntimeError(f"{location}: duplicate checkpoint/graph pair")
        observed_by_run[str(run_id)].add(pair_key)

        expected_pairing_key = (
            f"{dataset}__replicate_{replicate:04d}__pct_{percentage:03d}"
        )
        if row.get("pairing_key") != expected_pairing_key:
            raise RuntimeError(f"{location}: invalid pairing_key")

        treatment = _unit_interval(
            row.get("treatment_macro_f1"), location=f"{location}:treatment"
        )
        control = _unit_interval(
            row.get("control_macro_f1"), location=f"{location}:control"
        )
        effect = row.get("treatment_minus_control_macro_f1")
        if (
            isinstance(effect, bool)
            or not isinstance(effect, (int, float))
            or not math.isfinite(float(effect))
            or not math.isclose(
                float(effect), treatment - control, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise RuntimeError(f"{location}: treatment-control effect is inconsistent")

        for condition, keys in (
            ("treatment", treatment_keys),
            ("control", control_keys),
        ):
            prediction_key = row.get(f"{condition}_prediction_key")
            if not isinstance(prediction_key, str) or not prediction_key:
                raise RuntimeError(
                    f"{location}: invalid {condition}_prediction_key"
                )
            if prediction_key in keys:
                raise RuntimeError(
                    f"{location}: duplicate {condition}_prediction_key"
                )
            keys.add(prediction_key)
            if not _is_sha256(row.get(f"{condition}_prediction_sha256")):
                raise RuntimeError(
                    f"{location}: invalid {condition}_prediction_sha256"
                )
            if not _is_sha256(row.get(f"{condition}_edge_sha256")):
                raise RuntimeError(f"{location}: invalid {condition}_edge_sha256")
        if not _is_sha256(row.get("source_graphs_sha256")):
            raise RuntimeError(f"{location}: invalid source_graphs_sha256")

        graph_key = (str(dataset), replicate, percentage)
        signature = (
            row.get("replicate_seed"),
            row.get("source_graphs_sha256"),
            row.get("treatment_edge_sha256"),
            row.get("control_edge_sha256"),
        )
        previous = graph_signatures.setdefault(graph_key, signature)
        if signature != previous:
            raise RuntimeError(
                f"{location}: graph provenance differs within {expected_pairing_key}"
            )

    if treatment_keys & control_keys:
        raise RuntimeError("treatment and control prediction key spaces overlap")
    if len(treatment_keys) != protocol.expected_paired_rows:
        raise RuntimeError("treatment prediction keys are not globally unique")
    if len(control_keys) != protocol.expected_paired_rows:
        raise RuntimeError("control prediction keys are not globally unique")

    for run_id, baseline in baselines_by_run.items():
        expected_pairs = {
            (replicate, percentage)
            for replicate in protocol.replicates
            for percentage in protocol.nonzero_percentages(baseline["dataset"])
        }
        if observed_by_run[run_id] != expected_pairs:
            raise RuntimeError(f"{run_id}: incomplete graph-pair panel")

    validation_summary = {
        "raw_validation_record": validation,
        "raw_input_sha256": actual_hashes,
        "all_baseline_identities_unique": True,
        "all_checkpoint_graph_pairs_complete": True,
        "all_graph_provenance_constant_within_pair": True,
        "all_scores_finite_and_in_unit_interval": True,
        "treatment_control_pairing_valid": True,
    }
    return baselines, pairs, validation_summary


def aggregate_expansion(
    baselines: Sequence[dict],
    pairs: Sequence[dict],
    *,
    protocol: ExpansionProtocol = DEFAULT_PROTOCOL,
) -> tuple[list[dict], list[dict]]:
    baselines_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in baselines:
        baselines_by_cell[(row["dataset"], row["model"])].append(
            float(row["tail_test_macro_f1"])
        )

    scores: dict[tuple[str, str, int, int, str], list[float]] = defaultdict(list)
    for row in pairs:
        prefix = (
            row["dataset"],
            row["model"],
            int(row["p_percent"]),
            int(row["replicate_index"]),
        )
        scores[(*prefix, "treatment")].append(float(row["treatment_macro_f1"]))
        scores[(*prefix, "control")].append(float(row["control_macro_f1"]))

    replicate_rows: list[dict] = []
    for dataset, model in protocol.cells:
        baseline_values = baselines_by_cell[(dataset, model)]
        if len(baseline_values) != protocol.checkpoints_per_cell:
            raise RuntimeError(f"{dataset}/{model}: incomplete baseline panel")
        baseline_mean = fmean(baseline_values)
        for percentage in protocol.percentages_by_dataset[dataset]:
            for replicate in protocol.replicates:
                for condition in CONDITIONS:
                    if percentage == 0:
                        values = baseline_values
                    else:
                        values = scores[
                            (dataset, model, percentage, replicate, condition)
                        ]
                    if len(values) != protocol.checkpoints_per_cell:
                        raise RuntimeError(
                            f"{dataset}/{model}/{percentage}/{replicate}/{condition}: "
                            f"expected {protocol.checkpoints_per_cell} scores; "
                            f"found {len(values)}"
                        )
                    replicate_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "p_percent": percentage,
                            "replicate_index": replicate,
                            "condition": condition,
                            "n_checkpoints": protocol.checkpoints_per_cell,
                            "mean_macro_f1": baseline_mean
                            if percentage == 0
                            else fmean(values),
                        }
                    )

    dose_rows = dose_from_replicates(
        replicate_rows,
        cells=protocol.cells,
        percentages_by_dataset=protocol.percentages_by_dataset,
        replicates=protocol.replicates,
        checkpoints_per_cell=protocol.checkpoints_per_cell,
    )
    return replicate_rows, dose_rows


def dose_from_replicates(
    replicate_rows: Sequence[dict],
    *,
    cells: Sequence[tuple[str, str]],
    percentages_by_dataset: Mapping[str, Sequence[int]],
    replicates: Sequence[int],
    checkpoints_per_cell: int,
) -> list[dict]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in replicate_rows:
        grouped[
            (
                row["dataset"],
                row["model"],
                int(row["p_percent"]),
                row["condition"],
            )
        ].append(float(row["mean_macro_f1"]))

    dose_rows: list[dict] = []
    for dataset, model in cells:
        for percentage in percentages_by_dataset[dataset]:
            for condition in CONDITIONS:
                values = grouped[(dataset, model, percentage, condition)]
                if len(values) != len(replicates):
                    raise RuntimeError(
                        f"{dataset}/{model}/{percentage}/{condition}: "
                        "incomplete replicate summary"
                    )
                dose_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "p_percent": percentage,
                        "condition": condition,
                        "n_graph_replicates": len(replicates),
                        "n_checkpoints_per_graph": checkpoints_per_cell,
                        "mean_macro_f1": fmean(values),
                        "sd_macro_f1_across_graph_replicates": 0.0
                        if percentage == 0
                        else stdev(values),
                    }
                )
    return dose_rows


def prior_summary_paths(
    results_root: Path = REPO_ROOT / "results", *,
    pubmed_root: Path | None = None, large_root: Path | None = None,
) -> dict[str, dict[str, Path]]:
    if pubmed_root is not None and large_root is not None:
        return {
            dataset: {
                kind: root / filename
                for kind, filename in (
                    ("dose", "dose_response_summary.csv"),
                    ("replicate", "replicate_summary.csv"),
                    ("manifest", "analysis_manifest.json"),
                )
            }
            for dataset, root in (
                ("pubmed", pubmed_root),
                ("ogbn-arxiv", large_root / "ogbn-arxiv"),
                ("arxiv-year", large_root / "arxiv-year"),
            )
        }
    return {
        "pubmed": {
            "dose": results_root
            / "edge_perturbation_dose_response_pubmed"
            / "dose_response_summary.csv",
            "replicate": results_root
            / "edge_perturbation_dose_response_pubmed"
            / "replicate_summary.csv",
            "manifest": results_root
            / "edge_perturbation_dose_response_pubmed"
            / "analysis_manifest.json",
        },
        "ogbn-arxiv": {
            "dose": results_root
            / "edge_perturbation_dose_response_large"
            / "ogbn-arxiv"
            / "dose_response_summary.csv",
            "replicate": results_root
            / "edge_perturbation_dose_response_large"
            / "ogbn-arxiv"
            / "replicate_summary.csv",
            "manifest": results_root
            / "edge_perturbation_dose_response_large"
            / "ogbn-arxiv"
            / "analysis_manifest.json",
        },
        "arxiv-year": {
            "dose": results_root
            / "edge_perturbation_dose_response_large"
            / "arxiv-year"
            / "dose_response_summary.csv",
            "replicate": results_root
            / "edge_perturbation_dose_response_large"
            / "arxiv-year"
            / "replicate_summary.csv",
            "manifest": results_root
            / "edge_perturbation_dose_response_large"
            / "arxiv-year"
            / "analysis_manifest.json",
        },
    }


def _load_csv(path: Path, fields: Sequence[str]) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise RuntimeError(
                    f"{path}: expected columns {list(fields)}; "
                    f"found {reader.fieldnames}"
                )
            return [dict(row) for row in reader]
    except OSError as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def load_prior_summaries(
    paths: Mapping[str, Mapping[str, Path]],
) -> tuple[list[dict], list[dict], dict[str, str]]:
    replicate_rows: list[dict] = []
    dose_rows: list[dict] = []
    input_hashes: dict[str, str] = {}
    for dataset in PRIOR_DATASETS:
        dataset_paths = paths[dataset]
        for kind in ("dose", "replicate", "manifest"):
            path = dataset_paths[kind]
            if not path.is_file():
                raise FileNotFoundError(f"prior result not found: {path}")
            input_hashes[f"{dataset}/{path.name}"] = file_sha256(path)

        raw_replicates = _load_csv(dataset_paths["replicate"], REPLICATE_FIELDS)
        expected_replicate_keys = {
            (model, percentage, replicate, condition)
            for model in PRIOR_MODELS
            for percentage in ALL_PERCENTAGES
            for replicate in DEFAULT_PROTOCOL.replicates
            for condition in CONDITIONS
        }
        observed_replicate_keys: set[tuple[str, int, int, str]] = set()
        converted_replicates: list[dict] = []
        for raw in raw_replicates:
            if raw["dataset"] != dataset or raw["model"] not in PRIOR_MODELS:
                raise RuntimeError(
                    f"{dataset_paths['replicate']}: unexpected dataset/model"
                )
            row = {
                "dataset": raw["dataset"],
                "model": raw["model"],
                "p_percent": int(raw["p_percent"]),
                "replicate_index": int(raw["replicate_index"]),
                "condition": raw["condition"],
                "n_checkpoints": int(raw["n_checkpoints"]),
                "mean_macro_f1": float(raw["mean_macro_f1"]),
            }
            key = (
                row["model"],
                row["p_percent"],
                row["replicate_index"],
                row["condition"],
            )
            if key in observed_replicate_keys:
                raise RuntimeError(f"{dataset_paths['replicate']}: duplicate row")
            observed_replicate_keys.add(key)
            if row["n_checkpoints"] != 30:
                raise RuntimeError(
                    f"{dataset_paths['replicate']}: expected 30 checkpoints"
                )
            _unit_interval(
                row["mean_macro_f1"], location=str(dataset_paths["replicate"])
            )
            converted_replicates.append(row)
        if observed_replicate_keys != expected_replicate_keys:
            raise RuntimeError(
                f"{dataset_paths['replicate']}: incomplete prior replicate panel"
            )

        recomputed = dose_from_replicates(
            converted_replicates,
            cells=tuple((dataset, model) for model in PRIOR_MODELS),
            percentages_by_dataset={dataset: ALL_PERCENTAGES},
            replicates=DEFAULT_PROTOCOL.replicates,
            checkpoints_per_cell=30,
        )
        raw_dose = _load_csv(dataset_paths["dose"], DOSE_FIELDS)
        expected_dose = {
            (row["model"], row["p_percent"], row["condition"]): row
            for row in recomputed
        }
        if len(raw_dose) != len(expected_dose):
            raise RuntimeError(f"{dataset_paths['dose']}: incomplete prior dose panel")
        converted_dose: list[dict] = []
        observed_dose_keys: set[tuple[str, int, str]] = set()
        for raw in raw_dose:
            row = {
                "dataset": raw["dataset"],
                "model": raw["model"],
                "p_percent": int(raw["p_percent"]),
                "condition": raw["condition"],
                "n_graph_replicates": int(raw["n_graph_replicates"]),
                "n_checkpoints_per_graph": int(raw["n_checkpoints_per_graph"]),
                "mean_macro_f1": float(raw["mean_macro_f1"]),
                "sd_macro_f1_across_graph_replicates": float(
                    raw["sd_macro_f1_across_graph_replicates"]
                ),
            }
            key = (row["model"], row["p_percent"], row["condition"])
            if key in observed_dose_keys or key not in expected_dose:
                raise RuntimeError(f"{dataset_paths['dose']}: unexpected row")
            observed_dose_keys.add(key)
            expected = expected_dose[key]
            for field in (
                "mean_macro_f1",
                "sd_macro_f1_across_graph_replicates",
            ):
                if not math.isclose(
                    row[field], expected[field], rel_tol=0.0, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"{dataset_paths['dose']}: {field} does not match "
                        "the replicate summary"
                    )
            if (
                row["dataset"] != dataset
                or row["n_graph_replicates"] != 5
                or row["n_checkpoints_per_graph"] != 30
            ):
                raise RuntimeError(f"{dataset_paths['dose']}: protocol mismatch")
            converted_dose.append(row)
        if observed_dose_keys != set(expected_dose):
            raise RuntimeError(f"{dataset_paths['dose']}: incomplete prior dose panel")

        replicate_rows.extend(converted_replicates)
        dose_rows.extend(converted_dose)
    return replicate_rows, dose_rows, input_hashes


def _sort_replicates(rows: Iterable[dict]) -> list[dict]:
    dataset_order = {value: index for index, value in enumerate(DATASETS)}
    model_order = {value: index for index, value in enumerate(MODELS)}
    condition_order = {value: index for index, value in enumerate(CONDITIONS)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order[row["dataset"]],
            model_order[row["model"]],
            int(row["p_percent"]),
            int(row["replicate_index"]),
            condition_order[row["condition"]],
        ),
    )


def _sort_dose(rows: Iterable[dict]) -> list[dict]:
    dataset_order = {value: index for index, value in enumerate(DATASETS)}
    model_order = {value: index for index, value in enumerate(MODELS)}
    condition_order = {value: index for index, value in enumerate(CONDITIONS)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order[row["dataset"]],
            model_order[row["model"]],
            int(row["p_percent"]),
            condition_order[row["condition"]],
        ),
    )


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    format(float(row[field]), f".{CSV_FLOAT_DECIMAL_PLACES}f")
                    if field in CSV_FLOAT_FIELDS
                    else row[field]
                )
                for field in fields
            }
            for row in rows
        )


def analyze(
    final_root: Path,
    output_root: Path,
    *,
    prior_paths: Mapping[str, Mapping[str, Path]] | None = None,
    preparation: str = "erm",
) -> dict:
    protocol = protocol_for_preparation(preparation)
    baselines, pairs, expansion_validation = validate_expansion_inputs(
        final_root, protocol=protocol
    )
    expansion_replicates, expansion_dose = aggregate_expansion(
        baselines, pairs, protocol=protocol
    )
    if preparation == "erm":
        if prior_paths is None:
            raise ValueError("ERM analysis requires explicit two-model summary paths")
        prior_replicates, prior_dose, prior_hashes = load_prior_summaries(
            prior_paths
        )
    else:
        if prior_paths is not None:
            raise ValueError("prior ERM summaries apply only to preparation='erm'")
        prior_replicates, prior_dose, prior_hashes = [], [], {}

    replicate_rows = _sort_replicates([*prior_replicates, *expansion_replicates])
    dose_rows = _sort_dose([*prior_dose, *expansion_dose])
    expected_cells = set(FULL_MATRIX_CELLS)
    if {(row["dataset"], row["model"]) for row in dose_rows} != expected_cells:
        raise RuntimeError("combined summary does not cover the complete 6x5 matrix")
    if len(replicate_rows) != 5940 or len(dose_rows) != 1188:
        raise RuntimeError(
            "combined summary row counts are wrong: "
            f"{len(replicate_rows)} replicate, {len(dose_rows)} dose"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    replicate_path = output_root / "replicate_summary.csv"
    dose_path = output_root / "dose_response_summary.csv"
    manifest_path = output_root / "analysis_manifest.json"
    _write_csv(replicate_path, replicate_rows, REPLICATE_FIELDS)
    _write_csv(dose_path, dose_rows, DOSE_FIELDS)

    source_validation = expansion_validation["raw_validation_record"]
    preparation_label = "ERM" if preparation == "erm" else preparation
    sources = {
        "expansion": {
            "design_id": source_validation["design_id"],
            "experiment_code_version": source_validation.get("experiment_revision"),
            "validated_at": source_validation.get("validated_at"),
            "graph_catalog_sha256": source_validation["graph_catalog_sha256"],
            "input_sha256": expansion_validation["raw_input_sha256"],
        }
    }
    if preparation == "erm":
        sources["submitted_cells"] = {
            "datasets": list(PRIOR_DATASETS),
            "models": list(PRIOR_MODELS),
            "input_sha256": prior_hashes,
        }
    manifest = {
        "schema_version": 1,
        "analysis": "complete-edge-perturbation-dose-response",
        "analysis_code": {
            "path": "scripts/analyze_edge_dose_response_full.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "design": {
            "datasets": list(DATASETS),
            "models": list(MODELS),
            "conditions": list(CONDITIONS),
            "graph_replicates": list(protocol.replicates),
            "checkpoint_identities_per_model": 30,
            "p_percentages": list(ALL_PERCENTAGES),
            "roman_empire_p_percentages": list(ROMAN_PERCENTAGES),
            "metric": "tail_test_macro_f1",
            "metric_population": "fixed-full-tail-test",
        },
        "estimand": {
            "checkpoint_aggregation": (
                "unweighted mean macro-F1 over the 30 registered "
                f"{preparation_label} "
                "checkpoints within each graph replicate"
            ),
            "graph_aggregation": (
                "unweighted mean and sample standard deviation across the "
                "five graph-replicate means"
            ),
            "p_zero": (
                "one original-graph baseline per checkpoint copied to all "
                "five graph replicates and both conditions"
            ),
        },
        "sources": sources,
        "validation": {
            "raw_checkpoint_shards": source_validation["n_checkpoint_shards"],
            "raw_baseline_rows": source_validation["n_baseline_rows"],
            "raw_paired_rows": source_validation["n_paired_rows"],
            "unique_treatment_prediction_keys": source_validation[
                "n_unique_treatment_prediction_keys"
            ],
            "unique_control_prediction_keys": source_validation[
                "n_unique_control_prediction_keys"
            ],
            "all_baseline_identities_unique": True,
            "all_checkpoint_graph_pairs_complete": True,
            "all_graph_provenance_constant_within_pair": True,
            "all_scores_finite_and_in_unit_interval": True,
            "treatment_control_pairing_valid": True,
            "complete_model_dataset_cells": len(expected_cells),
            "replicate_summary_rows": len(replicate_rows),
            "dose_response_summary_rows": len(dose_rows),
            "roman_empire_max_percentage": 70,
        },
        "generated_outputs_sha256": {
            "dose_response_summary.csv": file_sha256(dose_path),
            "replicate_summary.csv": file_sha256(replicate_path),
        },
    }
    if preparation != "erm":
        manifest["design"]["preparation"] = preparation
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preparation", choices=PREPARATIONS, default="erm"
    )
    parser.add_argument("--final-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pubmed-summary-root", type=Path)
    parser.add_argument("--large-summary-root", type=Path)
    args = parser.parse_args(argv)
    if args.preparation == "erm":
        if args.pubmed_summary_root is None or args.large_summary_root is None:
            parser.error("ERM requires --pubmed-summary-root and --large-summary-root")
    elif args.pubmed_summary_root is not None or args.large_summary_root is not None:
        parser.error("two-model ERM summaries cannot be used for recovery preparations")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    final_root = args.final_root or default_final_root(args.preparation)
    output_root = args.output_root or default_output_root(args.preparation)
    manifest = analyze(
        final_root.expanduser().resolve(),
        output_root.expanduser().resolve(),
        preparation=args.preparation,
        prior_paths=(
            prior_summary_paths(
                pubmed_root=args.pubmed_summary_root.expanduser().resolve(),
                large_root=args.large_summary_root.expanduser().resolve(),
            ) if args.preparation == "erm" else None
        ),
    )
    validation = manifest["validation"]
    print(
        "validated and summarized "
        f"{validation['raw_checkpoint_shards']} expansion checkpoints, "
        f"{validation['raw_paired_rows']} paired rows, and "
        f"{validation['complete_model_dataset_cells']} full-matrix cells"
    )


if __name__ == "__main__":
    main()
