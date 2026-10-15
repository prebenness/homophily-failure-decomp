"""Authenticate reverse-direction inference, aggregate it and draw four figures.

Every preparation uses its own complete six-model/five-dataset evaluation.
No forward-direction or previous submitted result is loaded by this script.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from statistics import fmean, stdev
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
PREPARATIONS = ("erm", "last", "head", "full")
MODELS = ("gcn", "graphsage-mean", "h2gcn", "gpr-gnn", "acm-gcn", "linkx")
DATASETS = ("pubmed", "ogbn-arxiv", "roman-empire", "amazon-ratings", "arxiv-year")
CONDITIONS = ("treatment", "control")
DESIGN_ID = "reverse-tail-edge-perturbation-four-preparation-analysis-v1"
GRAPH_DESIGN_ID = "reverse-tail-edge-perturbation-graphs-v1"
EXPECTED_BASELINES = 3600
EXPECTED_PAIRED_ROWS = 176400
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
BACKEND_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
INFERENCE_IMPLEMENTATION_FILES = (
    "scripts/run_reverse_edge_inference.py",
    "scripts/run_edge_dose_response_expansion.py",
    "scripts/run_reverse_edge_graphs.py",
    "scripts/run_edge_perturbation.py",
)
REPLICATE_FIELDS = ("preparation", "dataset", "model", "target_count", "actual_pct", "replicate_index", "condition", "n_checkpoints", "mean_macro_f1")
SUMMARY_FIELDS = ("preparation", "dataset", "model", "target_count", "actual_pct", "condition", "n_graph_replicates", "n_checkpoints_per_graph", "mean_macro_f1", "sd_macro_f1_across_graph_replicates")


class ValidationError(RuntimeError):
    """A source or derived artefact is incomplete or inconsistent."""


@dataclass(frozen=True)
class GraphContract:
    dataset: str
    original_tail_count: int
    desired_same: bool
    doses: tuple[tuple[int, float], ...]
    checkpoint_identities: tuple[tuple[str, int], ...]
    graph_pairs: Mapping[tuple[int, int], dict]
    catalog_sha256: str
    replicates: tuple[int, ...] = (1, 2, 3, 4, 5)

    @property
    def expected_pairs_per_checkpoint(self) -> int:
        return len(self.doses) * len(self.replicates)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"{path.name}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValidationError(f"{path.name}: blank JSONL row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValidationError(f"{path.name}: expected JSON objects")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValidationError("source artefact path must be portable and relative")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValidationError("source artefact path escapes its registered root")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValidationError("source artefact resolves outside its registered root")
    return result


def _sha(value, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"invalid {label} SHA256")
    return value


def _implementation_map(value, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"{label} implementation hashes are missing")
    for name, digest in value.items():
        _safe_relative(REPO_ROOT, name)
        _sha(digest, f"{label} implementation")
    return value


def _implementation_signature(value: dict[str, str]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_implementation_provenance(
    validation: dict,
    shard_implementations: Sequence[dict],
    current_validator_implementation: dict,
) -> dict:
    validator = _implementation_map(
        validation.get("validator_implementation_sha256"), "validator"
    )
    current = _implementation_map(current_validator_implementation, "current validator")
    if validator != current:
        raise ValidationError("inference validator implementation hashes differ from current source")

    raw_producers = validation.get("producer_implementation_sha256")
    if not isinstance(raw_producers, list) or not raw_producers:
        raise ValidationError("producer implementation hash list is missing or empty")
    producers = [
        _implementation_map(value, "producer") for value in raw_producers
    ]
    producer_signatures = [_implementation_signature(value) for value in producers]
    if len(producer_signatures) != len(set(producer_signatures)):
        raise ValidationError("producer implementation hash list contains duplicates")

    observed = [
        _implementation_map(value, "checkpoint producer")
        for value in shard_implementations
    ]
    observed_signatures = sorted({_implementation_signature(value) for value in observed})
    if producer_signatures != sorted(producer_signatures):
        raise ValidationError("producer implementation hash list is not in canonical order")
    if not observed_signatures or producer_signatures != observed_signatures:
        raise ValidationError("producer implementation hashes differ from checkpoint shards")
    return {
        "validator_implementation_sha256": validator,
        "producer_implementation_sha256": producers,
        "implementation_provenance_verified": True,
    }


def current_inference_implementation_hashes() -> dict[str, str]:
    root = REPO_ROOT.resolve()
    paths = [(root / name).resolve() for name in INFERENCE_IMPLEMENTATION_FILES]
    paths.extend(path.resolve() for path in sorted((root / "src/hfd").rglob("*.py")))
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(set(paths))
    }


def _inference_validation_context(validation: dict) -> dict:
    if validation.get("schema_version") != 1:
        raise ValidationError("inference validation schema version differs")
    experiment_revision = validation.get("experiment_revision")
    if not isinstance(experiment_revision, str) or REVISION_RE.fullmatch(experiment_revision) is None:
        raise ValidationError("inference experiment revision is invalid")
    source_manifest_sha256 = validation.get("source_hash_manifest_sha256")
    if source_manifest_sha256 is not None:
        source_manifest_sha256 = _sha(source_manifest_sha256, "source-hash manifest")
    return {
        "inference_schema_version": validation["schema_version"],
        "experiment_revision": experiment_revision,
        "input_snapshot_sha256": _sha(validation.get("input_snapshot_sha256"), "input snapshot"),
        "source_hash_manifest_sha256": source_manifest_sha256,
    }


def _backend_id(value) -> str:
    if not isinstance(value, str) or BACKEND_ID_RE.fullmatch(value) is None:
        raise ValidationError("checkpoint shard contains a non-portable backend identifier")
    return value


def _shared_inference_context(validations: Mapping[str, dict]) -> dict:
    fields = (
        "inference_schema_version",
        "experiment_revision",
        "input_snapshot_sha256",
        "source_hash_manifest_sha256",
    )
    first = validations[PREPARATIONS[0]]
    context = {name: first[name] for name in fields}
    for preparation in PREPARATIONS[1:]:
        if any(validations[preparation][name] != value for name, value in context.items()):
            raise ValidationError("inference schema or source context differs across preparations")
    return context


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{label} must be finite numeric data")
    return float(value)


def _score(value, label: str) -> float:
    result = _number(value, label)
    if not 0 <= result <= 1:
        raise ValidationError(f"{label} must be in [0, 1]")
    return result


def _integer(value, label: str) -> int:
    if type(value) is not int:
        raise ValidationError(f"{label} must be an integer")
    return value


def load_graph_contracts(graph_root: Path) -> tuple[dict[str, GraphContract], dict]:
    contracts = {}
    provenance = {}
    for dataset in DATASETS:
        root = graph_root / dataset
        actual = root / "actual"
        catalog_path = actual / "catalog.json"
        catalog = read_json(catalog_path)
        validation = read_json(actual / "validation.json")
        if catalog.get("design_id") != GRAPH_DESIGN_ID or catalog.get("dataset") != dataset:
            raise ValidationError("graph catalog identity differs")
        if validation.get("validated") is not True or validation.get("catalog") != file_record(catalog_path):
            raise ValidationError("graph validation does not authenticate its catalog")
        if catalog.get("all_planned_doses_constructed") is not True or validation.get("all_planned_doses_constructed") is not True:
            raise ValidationError("full reverse inference requires the complete accepted graph panel")
        tail_count = _integer(catalog["original_tail_count"], "original-tail count")
        if tail_count <= 0 or catalog.get("denominator") != "original frozen tail-node count":
            raise ValidationError("graph denominator differs from the original tail")
        doses = tuple((int(row["target_count"]), float(row["actual_pct"])) for row in catalog["accepted_doses"])
        if not doses or [count for count, _ in doses] != sorted({count for count, _ in doses}):
            raise ValidationError("graph dose counts must be positive, unique and increasing")
        for count, percentage in doses:
            if count <= 0 or count > tail_count or not math.isclose(percentage, 100 * count / tail_count, rel_tol=0, abs_tol=1e-12):
                raise ValidationError("graph dose uses an incorrect denominator")
        registered_path = root / "prepared" / "registered-manifest.json"
        prepared_path = root / "prepared" / "manifest.json"
        prepared = read_json(prepared_path)
        if catalog.get("prepared_manifest") != file_record(prepared_path):
            raise ValidationError("prepared-input manifest differs from graph catalog")
        if file_sha256(registered_path) != prepared["registered_source_sha256"]:
            raise ValidationError("registered-input manifest differs from prepared inputs")
        registered = read_json(registered_path)
        checkpoint_identities = tuple((split["split_id"], seed) for split in registered["splits"] for seed in registered["training_seeds"])
        if len(checkpoint_identities) != 30 or len(set(checkpoint_identities)) != 30:
            raise ValidationError("registered checkpoint panel must contain exactly 30 split/seed identities")
        graph_pairs = {}
        for pair in catalog["pairs"]:
            dose = pair["dose"]
            count = int(dose["target_count"])
            replicate = _integer(pair["replicate_index"], "graph replicate")
            key = (count, replicate)
            if key in graph_pairs:
                raise ValidationError("duplicate graph pair in catalog")
            directory = _safe_relative(actual, pair["relative_directory"])
            if set(pair["files"]) != {"graphs.npz", "swaps.json", "metadata.json"}:
                raise ValidationError("graph catalog must authenticate all three pair artefacts")
            for name, record in pair["files"].items():
                if file_record(directory / name) != record:
                    raise ValidationError("graph file hash differs from its catalog")
            metadata = read_json(directory / "metadata.json")
            for name, expected in (("target_count", count), ("replicate_index", replicate), ("actual_pct", dose["actual_pct"]), ("desired_same", catalog["desired_same"]), ("original_tail_count", tail_count)):
                if metadata.get(name) != expected:
                    raise ValidationError("graph pair metadata differs from catalog")
            graph_pairs[key] = {
                "replicate_seed": metadata["replicate_seed"],
                "source_graphs_sha256": pair["files"]["graphs.npz"]["sha256"],
                "treatment_edge_sha256": metadata["audit"]["treatment_edge_sha256"],
                "control_edge_sha256": metadata["audit"]["control_edge_sha256"],
            }
        expected_pairs = {(count, replicate) for count, _ in doses for replicate in range(1, 6)}
        if set(graph_pairs) != expected_pairs or catalog.get("pair_count") != len(expected_pairs):
            raise ValidationError("graph panel does not contain exactly five pairs for every dose")
        contracts[dataset] = GraphContract(dataset, tail_count, catalog["desired_same"], doses, checkpoint_identities, graph_pairs, file_sha256(catalog_path))
        provenance[dataset] = {"catalog": file_record(catalog_path), "validation": file_record(actual / "validation.json"), "registered_manifest": file_record(registered_path), "prepared_manifest": file_record(prepared_path)}
    return contracts, provenance


def validate_rows(baselines, pairs, preparation, contracts, *, models=MODELS) -> dict:
    if preparation not in PREPARATIONS:
        raise ValidationError("unknown checkpoint preparation")
    cells = {(dataset, model) for dataset in contracts for model in models}
    expected_baselines = sum(len(contract.checkpoint_identities) * len(models) for contract in contracts.values())
    expected_pairs = sum(len(contract.checkpoint_identities) * len(models) * contract.expected_pairs_per_checkpoint for contract in contracts.values())
    if len(baselines) != expected_baselines or len(pairs) != expected_pairs:
        raise ValidationError("baseline or paired-row count differs from the full registered matrix")
    by_run = {}
    identities = defaultdict(set)
    for baseline in baselines:
        dataset, model = baseline.get("dataset"), baseline.get("model")
        if (dataset, model) not in cells or baseline.get("preparation") != preparation:
            raise ValidationError("baseline preparation, dataset or model differs")
        identity = (baseline.get("split"), _integer(baseline.get("seed"), "checkpoint seed"))
        run_id = f"{dataset}_{model}_{preparation}_{identity[0]}_seed-{identity[1]}"
        if baseline.get("run_id") != run_id or run_id in by_run:
            raise ValidationError("baseline run identity is duplicate or incorrect")
        if identity not in contracts[dataset].checkpoint_identities or identity in identities[(dataset, model)]:
            raise ValidationError("baseline split/seed panel differs")
        if baseline.get("prediction_key") != run_id + "__original":
            raise ValidationError("baseline prediction identity differs")
        _sha(baseline.get("checkpoint_sha256"), "checkpoint")
        _sha(baseline.get("prediction_sha256"), "baseline prediction")
        _score(baseline.get("tail_test_macro_f1"), "baseline macro-F1")
        _score(baseline.get("tail_test_accuracy"), "baseline accuracy")
        if _integer(baseline.get("n_tail_test"), "tail test-node count") < 1:
            raise ValidationError("baseline tail evaluation population is empty")
        by_run[run_id] = baseline
        identities[(dataset, model)].add(identity)
    for dataset, model in cells:
        if identities[(dataset, model)] != set(contracts[dataset].checkpoint_identities):
            raise ValidationError("baseline checkpoint panel is incomplete")
    seen = defaultdict(set)
    prediction_keys = set()
    for pair in pairs:
        run_id = pair.get("run_id")
        baseline = by_run.get(run_id)
        if baseline is None:
            raise ValidationError("paired row has no matching baseline")
        for name in ("preparation", "dataset", "model", "split", "seed", "checkpoint_sha256", "n_tail_test", "tail_test_class_composition"):
            if pair.get(name) != baseline.get(name):
                raise ValidationError(f"paired row differs from its baseline: {name}")
        contract = contracts[pair["dataset"]]
        count = _integer(pair.get("target_count"), "target count")
        replicate = _integer(pair.get("replicate_index"), "graph replicate")
        key = (count, replicate)
        graph = contract.graph_pairs.get(key)
        if graph is None or key in seen[run_id]:
            raise ValidationError("paired row has an unexpected or duplicate graph identity")
        seen[run_id].add(key)
        percentage = 100.0 * count / contract.original_tail_count
        for name in ("p_percent", "actual_pct"):
            if not math.isclose(_number(pair.get(name), name), percentage, rel_tol=0, abs_tol=1e-12):
                raise ValidationError("paired percentage does not use the original-tail denominator")
        if pair.get("n_selected_tail_nodes") != count or pair.get("original_tail_count") != contract.original_tail_count or pair.get("desired_same") != contract.desired_same:
            raise ValidationError("paired target count, denominator or treatment direction differs")
        pairing_key = f"reverse__{contract.dataset}__replicate_{replicate:04d}__targets_{count:08d}"
        if pair.get("pairing_key") != pairing_key:
            raise ValidationError("paired graph key differs")
        for name, expected in graph.items():
            if pair.get(name) != expected:
                raise ValidationError(f"paired graph provenance differs: {name}")
        for condition in CONDITIONS:
            prediction_key = run_id + "__" + pairing_key + "__" + condition
            if pair.get(condition + "_prediction_key") != prediction_key or prediction_key in prediction_keys:
                raise ValidationError("treatment/control prediction keys differ or repeat")
            prediction_keys.add(prediction_key)
            _sha(pair.get(condition + "_prediction_sha256"), "perturbed prediction")
            _sha(pair.get(condition + "_edge_sha256"), "perturbed graph")
        for metric in ("macro_f1", "accuracy"):
            treatment = _score(pair.get("treatment_" + metric), "treatment " + metric)
            control = _score(pair.get("control_" + metric), "control " + metric)
            difference = _number(pair.get("treatment_minus_control_" + metric), "paired difference")
            if not math.isclose(difference, treatment - control, rel_tol=0, abs_tol=1e-12):
                raise ValidationError("treatment-control difference is inconsistent")
    for run_id, baseline in by_run.items():
        if seen[run_id] != set(contracts[baseline["dataset"]].graph_pairs):
            raise ValidationError("checkpoint graph-pair panel is incomplete")
    return {"n_model_dataset_cells": len(cells), "n_checkpoint_shards": expected_baselines, "n_baseline_rows": len(baselines), "n_paired_rows": len(pairs), "pairing_validated": True, "original_tail_denominator_verified": True, "complete_registered_checkpoint_panel": True}


def load_preparation(
    input_root: Path,
    preparation: str,
    contracts,
    *,
    current_validator_implementation: dict,
) -> tuple[list, list, dict]:
    root = input_root / "actual" / preparation
    final = root / "final"
    validation_path = final / "validation.json"
    validation = read_json(validation_path)
    expected_design = f"reverse-tail-edge-perturbation-{preparation}-inference-v1"
    if validation.get("validated") is not True or validation.get("pairing_validated") is not True or validation.get("design_id") != expected_design or validation.get("preparation") != preparation:
        raise ValidationError("inference has not passed validation for this preparation")
    inference_context = _inference_validation_context(validation)
    experiment_revision = inference_context["experiment_revision"]
    input_snapshot_sha256 = inference_context["input_snapshot_sha256"]
    graph_hashes = {dataset: contract.catalog_sha256 for dataset, contract in contracts.items()}
    if validation.get("graph_catalog_sha256") != graph_hashes:
        raise ValidationError("inference graph catalogs differ from current authenticated inputs")
    output_hashes = {name: file_sha256(final / name) for name in ("baseline_rows.jsonl", "paired_rows.jsonl")}
    if validation.get("output_sha256") != output_hashes:
        raise ValidationError("inference consolidated-output hashes differ")
    shard_hashes = validation.get("checkpoint_shard_sha256")
    if not isinstance(shard_hashes, dict) or len(shard_hashes) != 900:
        raise ValidationError("inference needs 900 authenticated checkpoint files per preparation")
    actual_shards = {path.relative_to(root).as_posix() for path in root.glob("*/*/shards/*.json")}
    if set(shard_hashes) != actual_shards:
        raise ValidationError("checkpoint files differ from the validated shard set")
    shard_implementations = []
    shard_backend_ids = set()
    for relative, digest in shard_hashes.items():
        shard_path = _safe_relative(root, relative)
        if file_sha256(shard_path) != _sha(digest, "checkpoint shard"):
            raise ValidationError("checkpoint output hash differs")
        shard = read_json(shard_path)
        shard_implementations.append(shard.get("implementation_sha256"))
        backend_id = _backend_id(shard.get("backend_id"))
        if (shard.get("schema_version") != 1
                or shard.get("experiment_revision") != experiment_revision
                or shard.get("input_snapshot_sha256") != input_snapshot_sha256):
            raise ValidationError("checkpoint shard validation context differs")
        shard_backend_ids.add(backend_id)
    backend_ids = sorted(shard_backend_ids)
    if validation.get("backend_ids") != backend_ids:
        raise ValidationError("inference backend identifiers differ from checkpoint shards")
    implementation_provenance = _validate_implementation_provenance(
        validation, shard_implementations, current_validator_implementation
    )
    baselines = read_jsonl(final / "baseline_rows.jsonl")
    pairs = read_jsonl(final / "paired_rows.jsonl")
    independent = validate_rows(baselines, pairs, preparation, contracts)
    for field in ("n_model_dataset_cells", "n_checkpoint_shards", "n_baseline_rows", "n_paired_rows"):
        if validation.get(field) != independent[field]:
            raise ValidationError(f"inference validation count differs: {field}")
    for field in ("n_unique_treatment_prediction_keys", "n_unique_control_prediction_keys"):
        if validation.get(field) != independent["n_paired_rows"]:
            raise ValidationError(f"inference validation unique-key count differs: {field}")
    return baselines, pairs, {
        **independent, "preparation": preparation,
        **inference_context,
        "backend_ids": backend_ids,
        "input_sha256": {**output_hashes, "validation.json": file_sha256(validation_path)},
        "checkpoint_shard_manifest_sha256": hashlib.sha256(json.dumps(shard_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        **implementation_provenance, "graph_catalog_sha256": graph_hashes,
    }


def aggregate_rows(baselines, pairs, preparation, contracts, *, models=MODELS) -> tuple[list[dict], list[dict]]:
    """Average checkpoints first, then compute mean and sample SD over five graphs."""
    validate_rows(baselines, pairs, preparation, contracts, models=models)
    baseline_scores = defaultdict(list)
    scores = defaultdict(list)
    for row in baselines:
        baseline_scores[(row["dataset"], row["model"])].append(row["tail_test_macro_f1"])
    for row in pairs:
        for condition in CONDITIONS:
            scores[(row["dataset"], row["model"], row["target_count"], row["replicate_index"], condition)].append(row[condition + "_macro_f1"])
    replicates = []
    summaries = []
    for dataset, contract in contracts.items():
        for model in models:
            for count, percentage in ((0, 0.0), *contract.doses):
                for condition in CONDITIONS:
                    means = []
                    for replicate in contract.replicates:
                        values = baseline_scores[(dataset, model)] if count == 0 else scores[(dataset, model, count, replicate, condition)]
                        if len(values) != len(contract.checkpoint_identities):
                            raise ValidationError("aggregation encountered an incomplete checkpoint group")
                        mean = fmean(values)
                        means.append(mean)
                        replicates.append({"preparation": preparation, "dataset": dataset, "model": model, "target_count": count, "actual_pct": percentage, "replicate_index": replicate, "condition": condition, "n_checkpoints": len(values), "mean_macro_f1": mean})
                    summaries.append({"preparation": preparation, "dataset": dataset, "model": model, "target_count": count, "actual_pct": percentage, "condition": condition, "n_graph_replicates": len(contract.replicates), "n_checkpoints_per_graph": len(contract.checkpoint_identities), "mean_macro_f1": fmean(means), "sd_macro_f1_across_graph_replicates": 0.0 if count == 0 else stdev(means)})
    return replicates, summaries


def write_csv(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format(value, ".14g") if isinstance(value, float) else value for name, value in row.items()})
    temporary.replace(path)


def build_preparation_figure(summaries: Sequence[dict], contracts):
    from plot_edge_dose_response_composite import SummaryRow, build_figure

    rows = [SummaryRow(dataset=row["dataset"], model=row["model"], p_percent=row["actual_pct"], condition=row["condition"], n_graph_replicates=row["n_graph_replicates"], n_checkpoints_per_graph=row["n_checkpoints_per_graph"], mean_macro_f1=row["mean_macro_f1"], sd_macro_f1=row["sd_macro_f1_across_graph_replicates"]) for row in summaries]
    endpoints = {dataset: contract.doses[-1][1] for dataset, contract in contracts.items()}
    return build_figure(rows, x_limits_by_dataset=endpoints, xlabel="Original tail nodes rewired (%)")


def validate_pdf(path: Path) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValidationError("each reverse figure must be a single portrait PDF page")
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    if not math.isclose(width, 396, abs_tol=0.1) or not math.isclose(height, 468, abs_tol=0.1):
        raise ValidationError("reverse figure dimensions differ from the approved portrait layout")
    metadata = reader.metadata or {}
    if metadata.get("/Author", "") not in {"", "Anonymous"}:
        raise ValidationError("reverse figure contains unexpected author metadata")
    text = page.extract_text() or ""
    for required in ("PubMed", "ogbn-arxiv", "Roman-empire", "Amazon-ratings", "arxiv-year", "GCN", "GraphSAGE-mean", "H2GCN-2", "GPR-GNN", "ACM-GCN", "LINKX", "Treatment", "Control", "Original tail nodes rewired"):
        if required not in text:
            raise ValidationError(f"reverse figure is missing expected text: {required}")
    return {"pages": 1, "width_points": width, "height_points": height, "portrait": True, "required_labels_present": True, **file_record(path)}


def analyze(input_root: Path, graph_root: Path, output_root: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contracts, graph_provenance = load_graph_contracts(graph_root)
    current_validator_implementation = current_inference_implementation_hashes()
    expected_rows = sum(len(contract.checkpoint_identities) * len(MODELS) * len(PREPARATIONS) * contract.expected_pairs_per_checkpoint for contract in contracts.values())
    if expected_rows != EXPECTED_PAIRED_ROWS:
        raise ValidationError(f"accepted graph panel implies {expected_rows} paired rows, not {EXPECTED_PAIRED_ROWS}")
    outputs = {}
    validations = {}
    total_baselines = total_pairs = 0
    for preparation in PREPARATIONS:
        baselines, pairs, validation = load_preparation(
            input_root,
            preparation,
            contracts,
            current_validator_implementation=current_validator_implementation,
        )
        replicate_rows, summary_rows = aggregate_rows(baselines, pairs, preparation, contracts)
        total_baselines += len(baselines)
        total_pairs += len(pairs)
        root = output_root / preparation
        write_csv(root / "replicate_summary.csv", replicate_rows, REPLICATE_FIELDS)
        write_csv(root / "dose_response_summary.csv", summary_rows, SUMMARY_FIELDS)
        figure = build_preparation_figure(summary_rows, contracts)
        pdf_path = root / f"reverse_edge_perturbation_{preparation}.pdf"
        png_path = root / f"reverse_edge_perturbation_{preparation}.png"
        figure.savefig(pdf_path, metadata={"Title": f"Reverse-direction edge perturbation: {preparation}", "Author": "Anonymous", "Creator": "analyze_reverse_edge_dose_response.py", "CreationDate": None, "ModDate": None})
        figure.savefig(png_path, dpi=300)
        plt.close(figure)
        pdf_check = validate_pdf(pdf_path)
        files = {path.relative_to(output_root).as_posix(): file_record(path) for path in (root / "replicate_summary.csv", root / "dose_response_summary.csv", pdf_path, png_path)}
        preparation_manifest = {"schema_version": 1, "design_id": DESIGN_ID, "preparation": preparation, "inference_validation": validation, "summary_rows": len(summary_rows), "replicate_summary_rows": len(replicate_rows), "aggregation": "mean across 30 checkpoints within each graph replicate, then mean and sample standard deviation across five graph replicates", "baseline_standard_deviation": "zero; all five baseline references are the same original graph", "axis_denominator": "original frozen tail-node count", "pdf_validation": pdf_check, "files": files}
        write_json(root / "analysis_manifest.json", preparation_manifest)
        outputs.update(files)
        outputs[(root / "analysis_manifest.json").relative_to(output_root).as_posix()] = file_record(root / "analysis_manifest.json")
        validations[preparation] = {**validation, "pdf": pdf_check}
        print(f"FIGURE_COMPLETE preparation={preparation} baseline_rows={len(baselines)} paired_rows={len(pairs)}", flush=True)
        del pairs, baselines
    if total_baselines != EXPECTED_BASELINES or total_pairs != EXPECTED_PAIRED_ROWS:
        raise ValidationError("final four-preparation row counts differ")
    inference_context = _shared_inference_context(validations)
    source_files = ("scripts/analyze_reverse_edge_dose_response.py", "scripts/plot_edge_dose_response_composite.py")
    final = {"schema_version": 1, "design_id": DESIGN_ID, "validated": True, "preparations": list(PREPARATIONS), "n_model_dataset_preparation_combinations": 120, "n_checkpoint_shards": total_baselines, "n_baseline_rows": total_baselines, "n_paired_rows": total_pairs, "n_figure_pdfs": 4, "treatment_control_pairing_validated": True, "original_tail_denominator_verified": True, "all_four_figures_use_new_reverse_inference_only": True, "inference_validation_context": inference_context, "graph_provenance": graph_provenance, "per_preparation": validations, "source_sha256": {name: file_sha256(REPO_ROOT / name) for name in source_files}, "files": outputs}
    write_json(output_root / "final-validation.json", final)
    print(f"FINAL_VALIDATED checkpoint_shards={total_baselines} baseline_rows={total_baselines} paired_rows={total_pairs} figure_pdfs=4", flush=True)
    return final


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = analyze(args.input_root, args.graph_root, args.output_root)
    if args.paper_output_dir is not None:
        args.paper_output_dir.mkdir(parents=True, exist_ok=True)
        for preparation in PREPARATIONS:
            name = f"reverse_edge_perturbation_{preparation}.pdf"
            shutil.copyfile(args.output_root / preparation / name,
                            args.paper_output_dir / name)
    return result


if __name__ == "__main__":
    main()
