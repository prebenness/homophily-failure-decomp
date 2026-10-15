"""Evaluate the four checkpoint preparations on authenticated reverse graphs.

This runner deliberately does not change the original Appendix-J evaluator.
It reuses its tested model forwards, checkpoint loading and metrics, but owns
the reverse protocol's graph identities, fractional doses and result storage.
All six graph models are evaluated for every dataset and preparation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import run_edge_dose_response_expansion as base
import run_reverse_edge_graphs as graphs

SCHEMA_VERSION = 1
BASELINE_VALIDATION_MODE = "fixed-tail-prediction-and-metric-equivalence-v1"
COMPATIBLE_BASELINE_VALIDATION_MODES = {
    BASELINE_VALIDATION_MODE,
    "fixed-tail-prediction-and-metrics-logits-record-only-v1",
}
PREPARATIONS = base.PREPARATIONS
FULL_MODELS = base.FULL_MODELS
DENOMINATOR = "original frozen tail-node count"
DATASET_SPECS = {
    name: replace(spec, models=FULL_MODELS,
                  desired_same=graphs.SPECS[name]["desired_same"],
                  generator="singleton", reused_manifest_sha256=None)
    for name, spec in base.DATASET_SPECS.items()
}
read_json = graphs.read_json
file_sha256 = base.file_sha256
atomic_json = base.atomic_json


def design_id_for_preparation(preparation: str) -> str:
    if preparation not in PREPARATIONS:
        raise ValueError(f"unknown preparation {preparation!r}")
    return f"reverse-tail-edge-perturbation-{preparation}-inference-v1"


def expected_checkpoint_count(preparation: str) -> int:
    design_id_for_preparation(preparation)
    return sum(len(spec.checkpoint_identities) * len(FULL_MODELS)
               for spec in DATASET_SPECS.values())


def expected_paired_row_count(catalogs: dict, preparation: str) -> int:
    design_id_for_preparation(preparation)
    if set(catalogs) != set(DATASET_SPECS):
        raise RuntimeError("graph catalogs do not cover all five datasets")
    return sum(int(catalogs[name]["pair_count"]) * len(FULL_MODELS)
               * len(DATASET_SPECS[name].checkpoint_identities) for name in catalogs)


def pairing_key(dataset: str, replicate_index: int, target_count: int) -> str:
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("target count must be a positive integer")
    if type(replicate_index) is not int or replicate_index not in graphs.REPLICATES:
        raise ValueError("replicate index must be in 1..5")
    return (f"reverse__{dataset}__replicate_{replicate_index:04d}"
            f"__targets_{target_count:08d}")


def implementation_hashes() -> dict[str, str]:
    # Resolve both the repository root and source paths before computing their
    # relative names. This keeps provenance stable when the same checkout is
    # reached through a filesystem alias or symlink.
    root = REPO_ROOT.resolve()
    paths = [Path(__file__).resolve(), Path(base.__file__).resolve(),
             Path(graphs.__file__).resolve(),
             (root / "scripts/run_edge_perturbation.py").resolve()]
    paths.extend(path.resolve() for path in sorted((root / "src/hfd").rglob("*.py")))
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(set(paths))}


def _equal_fields(actual: dict, expected: dict, context: str) -> None:
    wrong = {name: (value, actual.get(name)) for name, value in expected.items()
             if actual.get(name) != value}
    if wrong:
        raise RuntimeError(f"{context}: fields differ {wrong}")


def _validate_implementation_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError("source-hash manifest contains no implementation hashes")
    for name, digest in value.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("source-hash manifest contains a non-portable source path")
        path = Path(name)
        if ("\\" in name or name.startswith("/")
                or (len(name) > 1 and name[1] == ":")
                or path.is_absolute() or ".." in path.parts):
            raise RuntimeError("source-hash manifest contains a non-portable source path")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise RuntimeError("source-hash manifest contains an invalid SHA-256")
    return value


def _implementation_is_accepted(value: object,
                                accepted: list[dict[str, str]]) -> bool:
    return isinstance(value, dict) and any(value == candidate for candidate in accepted)


def _canonical_implementation_maps(values: list[dict[str, str]]) -> list[dict[str, str]]:
    maps = [_validate_implementation_map(value) for value in values]
    signatures = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in maps]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("producer implementation list contains duplicates")
    return [value for _, value in sorted(zip(signatures, maps), key=lambda item: item[0])]


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RuntimeError("graph catalog contains an escaping path")
    return path


def _verify_file(path: Path, expected: dict) -> None:
    if not path.is_file() or graphs.file_record(path) != expected:
        raise RuntimeError(f"file hash or size differs: {path.name}")


def validate_catalog_structure(catalog: dict, dataset: str) -> None:
    spec = DATASET_SPECS[dataset]
    _equal_fields(catalog, {
        "schema_version": graphs.SCHEMA_VERSION, "design_id": graphs.DESIGN_ID,
        "dataset": dataset, "original_tail_count": spec.tail_count,
        "desired_same": spec.desired_same, "denominator": DENOMINATOR,
    }, f"{dataset} graph catalog")
    eligible = catalog.get("eligible_tail_count")
    if type(eligible) is not int or not 0 <= eligible <= spec.tail_count:
        raise RuntimeError("invalid eligible count")
    planned = graphs.dose_plan(spec.tail_count, eligible)
    accepted = catalog.get("accepted_doses")
    if not isinstance(accepted, list) or not accepted or accepted != planned[:len(accepted)]:
        raise RuntimeError("accepted doses are not a nonempty prefix of the original-tail plan")
    complete = len(accepted) == len(planned)
    if catalog.get("all_planned_doses_constructed") is not complete:
        raise RuntimeError("graph construction completion claim differs")
    if complete and catalog.get("construction_limit") is not None:
        raise RuntimeError("complete graph range has an unexpected cutoff")
    if not complete:
        limit = catalog.get("construction_limit")
        if not isinstance(limit, dict) or limit.get("dose") != planned[len(accepted)]:
            raise RuntimeError("missing first-unsupported-dose construction cutoff")
    expected = {(replicate, dose["target_count"]): dose
                for dose in accepted for replicate in graphs.REPLICATES}
    observed = set()
    for entry in catalog.get("pairs", []):
        dose = entry.get("dose", {})
        key = (entry.get("replicate_index"), dose.get("target_count"))
        if key in observed or key not in expected or dose != expected[key]:
            raise RuntimeError("duplicate, unexpected or altered graph dose/replicate")
        observed.add(key)
        if entry.get("replicate_seed") != spec.seed_base + key[0]:
            raise RuntimeError("graph replicate seed differs")
        wanted_dir = f"doses/targets-{key[1]:06d}/replicate-{key[0]:04d}"
        if entry.get("relative_directory") != wanted_dir:
            raise RuntimeError("graph directory does not encode its target count and replicate")
    if observed != set(expected) or catalog.get("pair_count") != len(expected):
        raise RuntimeError("graph catalog does not contain exactly five pairs per dose")
    _equal_fields(catalog.get("baseline", {}), {
        "target_count": 0, "actual_pct": 0.0, "original_tail_count": spec.tail_count,
        "replicate_indices": list(graphs.REPLICATES),
        "treatment_equals_control_equals_original": True,
    }, "graph baseline")


def load_graph_catalog(graph_root: Path, dataset: str, verify_files: bool = True) -> dict:
    root = Path(graph_root) / dataset
    actual = root / "actual"
    catalog = read_json(actual / "catalog.json")
    validate_catalog_structure(catalog, dataset)
    validation = read_json(actual / "validation.json")
    _equal_fields(validation, {
        "design_id": graphs.DESIGN_ID, "dataset": dataset, "validated": True,
        "pair_count": catalog["pair_count"], "replicates_per_accepted_dose": 5,
        "no_partial_replicate_averages": True,
        "original_tail_denominator_verified": True,
        "independent_swap_replay_verified": True,
        "accepted_nonzero_dose_count": len(catalog["accepted_doses"]),
        "all_planned_doses_constructed": catalog["all_planned_doses_constructed"],
        "maximum_accepted_pct": catalog["accepted_doses"][-1]["actual_pct"],
    }, "graph validation")
    _verify_file(actual / "catalog.json", validation["catalog"])
    _verify_file(root / "prepared/manifest.json", catalog["prepared_manifest"])
    _verify_file(actual / "experiment_manifest.json", catalog["experiment_manifest"])
    prepared = read_json(root / "prepared/manifest.json")
    experiment = read_json(actual / "experiment_manifest.json")
    _equal_fields(prepared, {
        "dataset": dataset, "design_id": graphs.DESIGN_ID,
        "original_tail_count": catalog["original_tail_count"],
        "eligible_tail_count": catalog["eligible_tail_count"],
        "desired_same": catalog["desired_same"], "source_sha256": catalog["source_sha256"],
    }, "graph preparation")
    if catalog["source_sha256"] != graphs.source_hashes():
        raise RuntimeError("graph generator sources differ from authenticated construction")
    _equal_fields(experiment, {
        "dataset": dataset, "design_id": graphs.DESIGN_ID, "phase": "actual",
        "desired_same": catalog["desired_same"], "source_sha256": catalog["source_sha256"],
        "prepared_manifest_sha256": catalog["prepared_manifest"]["sha256"],
        "dose_plan": prepared["dose_plan"],
    }, "graph experiment")
    if read_json(actual / "baseline.json") != catalog["baseline"]:
        raise RuntimeError("graph baseline reference differs")
    normalized = []
    for entry in catalog["pairs"]:
        path = _safe_path(actual, entry["relative_directory"])
        if set(entry.get("files", {})) != {"graphs.npz", "swaps.json", "metadata.json"}:
            raise RuntimeError("graph file inventory differs")
        for name, record in entry["files"].items():
            if verify_files or name == "metadata.json":
                _verify_file(path / name, record)
        metadata = read_json(path / "metadata.json")
        _equal_fields(metadata, {
            "design_id": graphs.DESIGN_ID, "dataset": dataset, "phase": "actual",
            "replicate_index": entry["replicate_index"],
            "replicate_seed": entry["replicate_seed"], **entry["dose"],
            "original_tail_count": catalog["original_tail_count"],
            "desired_same": catalog["desired_same"],
            "experiment_manifest_sha256": catalog["experiment_manifest"]["sha256"],
        }, "graph metadata")
        audit = metadata["audit"]
        _equal_fields(audit, {
            "all_node_degrees_preserved": True, "edge_count_preserved": True,
            "swaps_replay_exactly": True, "control_target_same_count_change": 0,
            "treatment_target_same_count_change": 1 if catalog["desired_same"] else -1,
            "target_count": entry["dose"]["target_count"],
        }, "graph audit")
        normalized.append({
            **entry, "target_count": entry["dose"]["target_count"],
            "p_percent": entry["dose"]["actual_pct"],
            "actual_pct": entry["dose"]["actual_pct"],
            "pairing_key": pairing_key(dataset, entry["replicate_index"], entry["dose"]["target_count"]),
            "graphs": entry["files"]["graphs.npz"],
            "treatment_edge_sha256": audit["treatment_edge_sha256"],
            "control_edge_sha256": audit["control_edge_sha256"],
            "relative_dir": "actual/" + entry["relative_directory"],
            "selected_targets": entry["dose"]["target_count"],
        })
    return {**catalog, "pairs": normalized,
            "catalog_sha256": file_sha256(actual / "catalog.json"),
            "validation_sha256": file_sha256(actual / "validation.json"),
            "prepared": prepared,
            "source_experiment_manifest_sha256": catalog["experiment_manifest"]["sha256"]}


def authenticate(args) -> dict:
    reports = {}
    for dataset in DATASET_SPECS:
        catalog = load_graph_catalog(args.graph_root, dataset)
        root = args.graph_root / dataset
        prepared = catalog["prepared"]
        for name, record in prepared["files"].items():
            _verify_file(root / "prepared" / name, record)
        if file_sha256(args.array_root / f"{dataset}.npz") != prepared["frozen_array_file_sha256"]:
            raise RuntimeError("registered frozen input archive differs")
        if file_sha256(args.manifest_root / f"{dataset}.json") != prepared["registered_source_sha256"]:
            raise RuntimeError("registered dataset manifest differs")
        bundle, frozen, registered = base.load_registered_inputs(
            DATASET_SPECS[dataset], data_root=args.data_root,
            array_root=args.array_root, manifest_root=args.manifest_root)
        try:
            with np.load(root / "prepared/inputs.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            if {key: graphs.array_sha256(value) for key, value in arrays.items()} != prepared["prepared_array_sha256"]:
                raise RuntimeError("prepared graph arrays differ")
            for name in ("labels", "tail_mask"):
                if not np.array_equal(arrays[name], frozen[name]):
                    raise RuntimeError("prepared graph population differs from registered evaluation population")
            if graphs.array_sha256(arrays["edge_index"]) != registered["graph_checksums_sha256"]["simple_edge_index"]:
                raise RuntimeError("prepared baseline graph differs from registered graph")
            if getattr(args, "replay_swaps", False):
                experiment = read_json(root / "actual/experiment_manifest.json")
                for entry in catalog["pairs"]:
                    graphs._verify_pair(root / entry["relative_dir"], arrays, prepared,
                                        experiment, entry["dose"], entry["replicate_index"])
            reports[dataset] = {"catalog_sha256": catalog["catalog_sha256"],
                                "pair_count": catalog["pair_count"],
                                "independent_swap_replay_verified": True,
                                "swap_replay_repeated": bool(getattr(args, "replay_swaps", False))}
            print(f"AUTHENTICATED {dataset} pairs={catalog['pair_count']}", flush=True)
        finally:
            frozen.close()
            del bundle
    result = {"design_id": graphs.DESIGN_ID, "authenticated": True,
              "datasets": reports, "implementation_sha256": args.implementation_sha256,
              "n_graph_pairs": sum(row["pair_count"] for row in reports.values())}
    atomic_json(args.output_root / "graph-authentication.json", result)
    return result


def checkpoint_file_hashes(checkpoint) -> dict:
    root = checkpoint.checkpoint_path.parent
    return {name: file_sha256(root / name)
            for name in ("checkpoint.pt", "config.json", "run_data.npz")}


def verify_checkpoint_snapshot(args, catalog, checkpoint, inputs) -> None:
    snapshot = getattr(args, "input_snapshot_value", None)
    if snapshot is None:
        if args.phase == "actual":
            raise RuntimeError("actual inference requires an authenticated input snapshot")
        return
    record = args.input_snapshot_runs.get(checkpoint.run_id)
    if record is None:
        raise RuntimeError("checkpoint is absent from the authenticated input snapshot")
    _equal_fields(record, {"dataset": checkpoint.dataset, "model": checkpoint.model,
                           "preparation": checkpoint.preparation, "split": checkpoint.split,
                           "seed": checkpoint.seed}, "authenticated checkpoint")
    if inputs != {name: record["files"][name]["sha256"] for name in inputs}:
        raise RuntimeError("checkpoint/configuration/saved predictions differ from authenticated inputs")
    bound = snapshot["datasets"][catalog["dataset"]]
    if (bound["reverse_graph_catalog"]["sha256"] != catalog["catalog_sha256"]
            or bound["reverse_graph_validation"]["sha256"] != catalog["validation_sha256"]):
        raise RuntimeError("reverse graphs differ from authenticated checkpoint input snapshot")


def _confusion_counts(labels: np.ndarray, predictions: np.ndarray,
                      indices: np.ndarray) -> list[dict[str, int]]:
    pairs, counts = np.unique(
        np.stack((labels[indices], predictions[indices]), axis=1),
        axis=0,
        return_counts=True,
    )
    return [
        {"true": int(pair[0]), "predicted": int(pair[1]), "count": int(count)}
        for pair, count in zip(pairs, counts)
    ]


def baseline_row_from_logits(checkpoint, *, bundle, arrays, current_logits,
                             logit_atol=1e-5, logit_rtol=1e-5) -> dict:
    """Validate the fixed tail predictions and record, but do not gate on, logits."""
    del logit_atol, logit_rtol  # Retained in the call signature for old callers.
    current_logits = np.asarray(current_logits)
    if (current_logits.ndim != 2 or current_logits.shape[0] != bundle.num_nodes
            or not np.isfinite(current_logits).all()):
        raise RuntimeError(f"{checkpoint.run_id}: invalid baseline logits")
    with np.load(checkpoint.run_data_path, allow_pickle=False) as saved:
        if "logits" not in saved.files:
            raise RuntimeError(f"{checkpoint.run_id}: saved baseline logits missing")
        saved_logits = saved["logits"].astype(current_logits.dtype, copy=True)
    if saved_logits.shape != current_logits.shape or not np.isfinite(saved_logits).all():
        raise RuntimeError(f"{checkpoint.run_id}: invalid baseline logits")
    saved_predictions = np.asarray(checkpoint.saved_predictions, dtype=np.int64)
    if not np.array_equal(saved_logits.argmax(axis=-1), saved_predictions):
        raise RuntimeError(f"{checkpoint.run_id}: saved logits/predictions differ")
    current_predictions = current_logits.argmax(axis=-1).astype(np.int64, copy=False)
    indices = base.tail_test_indices(arrays, checkpoint.split)
    difference_nodes = np.flatnonzero(current_predictions != saved_predictions).astype(np.int64)
    tail_differences = int(np.count_nonzero(
        current_predictions[indices] != saved_predictions[indices]
    ))
    if tail_differences:
        raise RuntimeError(
            f"{checkpoint.run_id}: fixed tail-test predictions differ "
            f"({tail_differences} nodes)"
        )
    saved_macro_f1, saved_accuracy = base.subset_metrics(
        bundle.y, saved_predictions, indices
    )
    current_macro_f1, current_accuracy = base.subset_metrics(
        bundle.y, current_predictions, indices
    )
    saved_confusion = _confusion_counts(bundle.y, saved_predictions, indices)
    current_confusion = _confusion_counts(bundle.y, current_predictions, indices)
    if ((current_macro_f1, current_accuracy) != (saved_macro_f1, saved_accuracy)
            or current_confusion != saved_confusion):
        raise RuntimeError(f"{checkpoint.run_id}: fixed tail-test metrics differ")
    differences = np.abs(current_logits - saved_logits)
    saved_tail = np.ascontiguousarray(saved_predictions[indices], dtype=np.int64)
    current_tail = np.ascontiguousarray(current_predictions[indices], dtype=np.int64)
    return {
        "dataset": checkpoint.dataset,
        "model": checkpoint.model,
        "run_id": checkpoint.run_id,
        "split": checkpoint.split,
        "seed": checkpoint.seed,
        "preparation": checkpoint.preparation,
        "n_tail_test": int(indices.size),
        "tail_test_class_composition": base.class_composition(bundle.y, indices),
        "tail_test_macro_f1": current_macro_f1,
        "tail_test_accuracy": current_accuracy,
        "saved_tail_test_macro_f1": saved_macro_f1,
        "saved_tail_test_accuracy": saved_accuracy,
        "saved_tail_prediction_sha256": base.array_sha256(saved_tail),
        "current_tail_prediction_sha256": base.array_sha256(current_tail),
        "saved_tail_confusion_counts": saved_confusion,
        "current_tail_confusion_counts": current_confusion,
        "n_full_graph_prediction_differences_from_parent": int(difference_nodes.size),
        "n_tail_test_prediction_differences_from_parent": tail_differences,
        "full_graph_prediction_difference_nodes_from_parent": difference_nodes.tolist(),
        "baseline_validation_mode": BASELINE_VALIDATION_MODE,
        "raw_logit_differences_recorded": True,
        "raw_logit_comparison_is_acceptance_criterion": False,
        "full_graph_logit_reproduction_checked": False,
        "max_abs_logit_difference_from_parent": float(differences.max()),
        "mean_abs_logit_difference_from_parent": float(differences.mean()),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_training_code_version": checkpoint.training_code_version,
        "prediction_key": base.prediction_key(checkpoint.run_id, "original"),
        "prediction_sha256": base.array_sha256(current_predictions),
    }


def paired_metric_row(checkpoint, *, bundle, arrays, graph_entry, catalog,
                      treatment_predictions, control_predictions) -> dict:
    # The reused metric routine's legacy integer dose field is replaced here;
    # target_count, never a rounded percentage, supplies the graph identity.
    row = base.paired_metric_row(
        checkpoint, bundle=bundle, arrays=arrays, graph_entry=graph_entry,
        treatment_predictions=treatment_predictions, control_predictions=control_predictions)
    return {**row, "preparation": checkpoint.preparation,
            "p_percent": graph_entry["actual_pct"], "actual_pct": graph_entry["actual_pct"],
            "target_count": graph_entry["target_count"],
            "original_tail_count": catalog["original_tail_count"],
            "desired_same": catalog["desired_same"]}


def _validate_baseline(row, run_id, checkpoint_hash, preparation, atol=None, rtol=None):
    _equal_fields(row, {"run_id": run_id, "checkpoint_sha256": checkpoint_hash,
                       "preparation": preparation,
                       "prediction_key": base.prediction_key(run_id, "original"),
                       "n_tail_test_prediction_differences_from_parent": 0}, "baseline")
    if row.get("baseline_validation_mode") in COMPATIBLE_BASELINE_VALIDATION_MODES:
        _equal_fields(row, {
            "raw_logit_differences_recorded": True,
            "raw_logit_comparison_is_acceptance_criterion": False,
            "full_graph_logit_reproduction_checked": False,
        }, "prediction-equivalent baseline")
        if row.get("saved_tail_prediction_sha256") != row.get("current_tail_prediction_sha256"):
            raise RuntimeError("fixed tail prediction hashes differ")
        if row.get("saved_tail_confusion_counts") != row.get("current_tail_confusion_counts"):
            raise RuntimeError("fixed tail confusion counts differ")
        if row.get("saved_tail_test_macro_f1") != row.get("tail_test_macro_f1"):
            raise RuntimeError("fixed tail macro-F1 differs")
        if row.get("saved_tail_test_accuracy") != row.get("tail_test_accuracy"):
            raise RuntimeError("fixed tail accuracy differs")
        count = row.get("n_full_graph_prediction_differences_from_parent")
        nodes = row.get("full_graph_prediction_difference_nodes_from_parent")
        if not isinstance(count, int) or count < 0 or not isinstance(nodes, list) or len(nodes) != count:
            raise RuntimeError("invalid full-graph prediction difference record")
        for field in ("max_abs_logit_difference_from_parent",
                      "mean_abs_logit_difference_from_parent"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("invalid recorded raw-logit difference")
    else:
        # Backward compatibility for already completed reverse-inference shards.
        _equal_fields(row, {"full_graph_logit_reproduction_checked": True}, "legacy baseline")
        if atol is not None and row.get("full_graph_logit_reproduction_atol") != atol:
            raise RuntimeError("baseline absolute tolerance differs")
        if rtol is not None and row.get("full_graph_logit_reproduction_rtol") != rtol:
            raise RuntimeError("baseline relative tolerance differs")
    for metric in ("tail_test_macro_f1", "tail_test_accuracy"):
        if not isinstance(row.get(metric), (int, float)) or not 0 <= row[metric] <= 1:
            raise RuntimeError("invalid baseline metric")
    if not isinstance(row.get("n_tail_test"), int) or row["n_tail_test"] <= 0:
        raise RuntimeError("invalid baseline evaluation population")


def validate_shard_payload(payload, *, catalog, checkpoint=None, preparation,
                           phase="actual", implementation_sha256=None,
                           logit_atol=None, logit_rtol=None,
                           input_snapshot_sha256=None, experiment_revision=None,
                           backend_id=None):
    run_id = payload.get("run_id")
    dataset = catalog["dataset"]
    model = payload.get("model")
    if model not in FULL_MODELS:
        raise RuntimeError("unknown reverse-evaluation model")
    spec = DATASET_SPECS[dataset]
    valid_runs = {base.parent_run_id(spec, model, split, seed, preparation): (split, seed)
                  for split, seed in spec.checkpoint_identities}
    if run_id not in valid_runs:
        raise RuntimeError("checkpoint identity is outside the fixed panel")
    _equal_fields(payload, {
        "schema_version": SCHEMA_VERSION, "design_id": design_id_for_preparation(preparation),
        "completed": True, "dataset": dataset, "preparation": preparation,
        "split": valid_runs[run_id][0], "seed": valid_runs[run_id][1],
        "phase": phase, "graph_catalog_sha256": catalog["catalog_sha256"],
        "source_graph_manifest_sha256": catalog["source_experiment_manifest_sha256"],
    }, "reverse checkpoint output")
    if implementation_sha256 is not None and payload.get("implementation_sha256") != implementation_sha256:
        raise RuntimeError("inference implementation hashes differ")
    if input_snapshot_sha256 is not None and payload.get("input_snapshot_sha256") != input_snapshot_sha256:
        raise RuntimeError("checkpoint output input snapshot identity differs")
    if experiment_revision is not None and payload.get("experiment_revision") != experiment_revision:
        raise RuntimeError("checkpoint output experiment revision differs")
    observed_backend = payload.get("backend_id")
    if (not isinstance(observed_backend, str) or not observed_backend
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in observed_backend)):
        raise RuntimeError("missing or invalid backend identity")
    if backend_id is not None and observed_backend != backend_id:
        raise RuntimeError("checkpoint output backend identity differs")
    checkpoint_hash = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise RuntimeError("invalid checkpoint hash")
    if checkpoint is not None:
        _equal_fields(payload, {"run_id": checkpoint.run_id, "checkpoint_sha256": checkpoint.checkpoint_sha256,
                               "input_file_sha256": checkpoint_file_hashes(checkpoint)}, "checkpoint source")
    _validate_baseline(payload.get("baseline", {}), run_id, checkpoint_hash,
                       preparation, logit_atol, logit_rtol)
    _equal_fields(payload["baseline"], {"dataset": dataset, "model": model,
                                       "split": valid_runs[run_id][0], "seed": valid_runs[run_id][1]},
                  "baseline checkpoint identity")
    entries = {entry["pairing_key"]: entry for entry in catalog["pairs"]}
    rows = payload.get("pairs")
    if not isinstance(rows, list) or payload.get("n_paired_rows") != len(rows):
        raise RuntimeError("paired row count differs")
    observed = set()
    for row in rows:
        key = row.get("pairing_key")
        if key in observed or key not in entries:
            raise RuntimeError("duplicate or unexpected graph-pair identity")
        observed.add(key)
        entry = entries[key]
        _equal_fields(row, {
            "run_id": run_id, "dataset": dataset, "model": model,
            "preparation": preparation, "checkpoint_sha256": checkpoint_hash,
            "split": valid_runs[run_id][0], "seed": valid_runs[run_id][1],
            "replicate_index": entry["replicate_index"], "replicate_seed": entry["replicate_seed"],
            "target_count": entry["target_count"], "n_selected_tail_nodes": entry["target_count"],
            "p_percent": entry["actual_pct"], "actual_pct": entry["actual_pct"],
            "original_tail_count": catalog["original_tail_count"],
            "desired_same": catalog["desired_same"],
            "source_graphs_sha256": entry["graphs"]["sha256"],
            "treatment_edge_sha256": entry["treatment_edge_sha256"],
            "control_edge_sha256": entry["control_edge_sha256"],
            "treatment_prediction_key": base.prediction_key(run_id, key + "__treatment"),
            "control_prediction_key": base.prediction_key(run_id, key + "__control"),
            "n_tail_test": payload["baseline"]["n_tail_test"],
            "tail_test_class_composition": payload["baseline"]["tail_test_class_composition"],
        }, "paired result")
        for metric in ("macro_f1", "accuracy"):
            left, right = row.get(f"treatment_{metric}"), row.get(f"control_{metric}")
            if not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in (left, right)):
                raise RuntimeError("invalid paired performance metric")
            if not math.isclose(row.get(f"treatment_minus_control_{metric}", float("nan")),
                                left - right, rel_tol=0, abs_tol=1e-14):
                raise RuntimeError("paired performance difference differs")
    if observed != set(entries):
        raise RuntimeError("incomplete graph pairing in checkpoint output")
    return payload


def result_root(args) -> Path:
    return args.output_root / args.phase / args.preparation


def _header(args, catalog, checkpoint, inputs) -> dict:
    return {"schema_version": SCHEMA_VERSION, "design_id": design_id_for_preparation(args.preparation),
            "dataset": checkpoint.dataset, "model": checkpoint.model, "run_id": checkpoint.run_id,
            "split": checkpoint.split, "seed": checkpoint.seed, "preparation": args.preparation,
            "phase": args.phase, "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "input_file_sha256": inputs, "graph_catalog_sha256": catalog["catalog_sha256"],
            "source_graph_manifest_sha256": catalog["source_experiment_manifest_sha256"],
            "experiment_revision": args.experiment_revision,
            "backend_id": args.backend_id,
            "input_snapshot_sha256": getattr(args, "input_snapshot_sha256", None),
            "implementation_sha256": args.implementation_sha256}


def _write_shard(args, catalog, checkpoint, inputs, baseline, pairs, schedule, seconds):
    payload = {**_header(args, catalog, checkpoint, inputs), "completed": True,
               "baseline": baseline, "pairs": pairs, "n_baseline_rows": 1,
               "n_paired_rows": len(pairs), "schedule": schedule,
               "evaluation_seconds": seconds, "completed_at": base.now_utc()}
    validate_shard_payload(payload, catalog=catalog, preparation=args.preparation,
                           phase=args.phase, implementation_sha256=args.implementation_sha256,
                           input_snapshot_sha256=getattr(args, "input_snapshot_sha256", None),
                           experiment_revision=args.experiment_revision,
                           backend_id=args.backend_id,
                           logit_atol=args.logit_reproduction_atol, logit_rtol=args.logit_reproduction_rtol)
    path = base.shard_path(result_root(args), checkpoint)
    atomic_json(path, payload)
    print(f"CHECKPOINT_COMPLETE {args.preparation} {checkpoint.dataset} {checkpoint.model} "
          f"{checkpoint.run_id} pairs={len(pairs)}", flush=True)


def _load_baseline(args, catalog, checkpoint, inputs, path):
    if not path.exists():
        return None
    payload = read_json(path)
    _equal_fields(payload, _header(args, catalog, checkpoint, inputs), "saved baseline source")
    _validate_baseline(payload["baseline"], checkpoint.run_id, checkpoint.checkpoint_sha256,
                       args.preparation, args.logit_reproduction_atol, args.logit_reproduction_rtol)
    return payload["baseline"]


def _save_baseline(args, catalog, checkpoint, inputs, row, path):
    atomic_json(path, {**_header(args, catalog, checkpoint, inputs), "baseline": row})


def load_graph_edges(graph_root, catalog, entry):
    path = graph_root / catalog["dataset"] / entry["relative_dir"] / "graphs.npz"
    with np.load(path, allow_pickle=False) as archive:
        treatment = archive["treatment_edge_index"].astype(np.int64, copy=True)
        control = archive["control_edge_index"].astype(np.int64, copy=True)
    for condition, edges in (("treatment", treatment), ("control", control)):
        if graphs.array_sha256(edges) != entry[f"{condition}_edge_sha256"]:
            raise RuntimeError("loaded graph array hash differs")
    return treatment, control


def _network(checkpoint, bundle, device):
    return base.load_network(checkpoint, input_dim=int(bundle.x.shape[1]),
                             num_classes=bundle.num_classes, num_nodes=bundle.num_nodes, device=device)


def _single_predictions(network, x, edges, device):
    tensor = torch.from_numpy(edges).to(device=device, dtype=torch.long)
    try:
        return base._forward_logits(network, x, tensor)
    finally:
        del tensor


def evaluate(args) -> dict:
    catalog = load_graph_catalog(args.graph_root, args.dataset)
    spec = DATASET_SPECS[args.dataset]
    bundle, arrays, registered = base.load_registered_inputs(
        spec, data_root=args.data_root, array_root=args.array_root, manifest_root=args.manifest_root)
    try:
        if registered["array_checksums_sha256"] != catalog["prepared"]["registered_frozen_array_sha256"]:
            raise RuntimeError("evaluation population differs from reverse graph registration")
        identities = spec.checkpoint_identities
        selected = catalog
        if args.phase == "smoke":
            identities = identities[:args.max_checkpoints]
            selected = {**catalog, "pairs": [] if args.baseline_only else catalog["pairs"][:args.max_pairs]}
        device = torch.device(args.device)
        x = bundle.x.to(device)
        original = np.asarray(base.to_simple_undirected(bundle.edge_index, bundle.num_nodes), dtype=np.int64)
        if graphs.array_sha256(original) != catalog["baseline"]["original_edge_sha256"]:
            raise RuntimeError("evaluation baseline graph differs from constructed baseline")
        checkpoints = []
        for split, seed in identities:
            checkpoint = base.load_checkpoint_spec(
                spec, model=args.model, split=split, seed=seed,
                checkpoint_root=args.checkpoint_root,
                fixed_input_checksums=registered["array_checksums_sha256"],
                num_nodes=bundle.num_nodes, preparation=args.preparation)
            inputs = checkpoint_file_hashes(checkpoint)
            verify_checkpoint_snapshot(args, catalog, checkpoint, inputs)
            path = base.shard_path(result_root(args), checkpoint)
            if path.exists() and not args.baseline_only:
                validate_shard_payload(read_json(path), catalog=selected, checkpoint=checkpoint,
                                       preparation=args.preparation, phase=args.phase,
                                       implementation_sha256=args.implementation_sha256,
                                       input_snapshot_sha256=getattr(args, "input_snapshot_sha256", None),
                                       experiment_revision=args.experiment_revision,
                                       backend_id=args.backend_id,
                                       logit_atol=args.logit_reproduction_atol,
                                       logit_rtol=args.logit_reproduction_rtol)
                print(f"CHECKPOINT_REUSED {checkpoint.run_id}", flush=True)
                continue
            checkpoints.append((checkpoint, inputs))
        if args.model == "h2gcn":
            _evaluate_h2(args, selected, bundle, arrays, x, original, device, checkpoints)
        else:
            _evaluate_checkpoint_major(args, selected, bundle, arrays, x, original, device, checkpoints)
        if args.baseline_only:
            gate = {"validated": True, "baseline_only": True,
                    "design_id": design_id_for_preparation(args.preparation),
                    "dataset": args.dataset, "model": args.model, "preparation": args.preparation,
                    "backend_id": args.backend_id,
                    "n_checkpoints": len(identities),
                    "baseline_validation_mode": BASELINE_VALIDATION_MODE,
                    "raw_logit_differences_recorded": True,
                    "raw_logit_comparison_is_acceptance_criterion": False,
                    "fixed_tail_predictions_unchanged": True,
                    "implementation_sha256": args.implementation_sha256,
                    "graph_catalog_sha256": catalog["catalog_sha256"]}
            atomic_json(result_root(args) / args.dataset / args.model / "baseline-gate.json", gate)
            print(f"BASELINE_GATE_COMPLETE {args.preparation} {args.dataset} {args.model} "
                  f"checkpoints={len(identities)}", flush=True)
            return gate
        print(f"MODEL_DATASET_COMPLETE {args.phase} {args.preparation} {args.dataset} {args.model} "
              f"checkpoints={len(identities)} pairs_per_checkpoint={len(selected['pairs'])}", flush=True)
        return {"completed": True}
    finally:
        arrays.close()


def _evaluate_checkpoint_major(args, catalog, bundle, arrays, x, original, device, checkpoints):
    for checkpoint, inputs in checkpoints:
        started = time.perf_counter()
        network = _network(checkpoint, bundle, device)
        try:
            path = result_root(args) / args.dataset / args.model / "baselines" / f"{checkpoint.run_id}.json"
            # Never trust a baseline produced by an earlier process or backend.
            # Reproduction is the gate for every evaluation invocation.
            baseline = baseline_row_from_logits(
                checkpoint, bundle=bundle, arrays=arrays,
                current_logits=_single_predictions(network, x, original, device),
                logit_atol=args.logit_reproduction_atol, logit_rtol=args.logit_reproduction_rtol)
            _save_baseline(args, catalog, checkpoint, inputs, baseline, path)
            if args.baseline_only:
                continue
            rows = []
            for entry in catalog["pairs"]:
                treatment, control = load_graph_edges(args.graph_root, catalog, entry)
                t_pred = _single_predictions(network, x, treatment, device).argmax(axis=-1).astype(np.int64)
                c_pred = _single_predictions(network, x, control, device).argmax(axis=-1).astype(np.int64)
                rows.append(paired_metric_row(checkpoint, bundle=bundle, arrays=arrays,
                                              graph_entry=entry, catalog=catalog,
                                              treatment_predictions=t_pred, control_predictions=c_pred))
            _write_shard(args, catalog, checkpoint, inputs, baseline, rows, "checkpoint-major",
                          time.perf_counter() - started)
        finally:
            del network
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _evaluate_h2(args, catalog, bundle, arrays, x, original, device, checkpoints):
    if not checkpoints:
        return
    networks = {checkpoint.run_id: _network(checkpoint, bundle, device)
                for checkpoint, _ in checkpoints}
    directory = result_root(args) / args.dataset / args.model
    baselines = {}
    try:
        # Build the original graph operators once, but always reproduce every
        # pending checkpoint in this process before accepting any graph result.
        logits = base.h2_predictions_for_graph(networks, x, original, device=device, pause_seconds=0)
        for checkpoint, inputs in checkpoints:
            row = baseline_row_from_logits(checkpoint, bundle=bundle, arrays=arrays,
                                            current_logits=logits[checkpoint.run_id],
                                            logit_atol=args.logit_reproduction_atol,
                                            logit_rtol=args.logit_reproduction_rtol)
            _save_baseline(args, catalog, checkpoint, inputs, row,
                           directory / "baselines" / f"{checkpoint.run_id}.json")
            baselines[checkpoint.run_id] = row
        del logits
        if args.baseline_only:
            return
        rows_by_run = {checkpoint.run_id: [] for checkpoint, _ in checkpoints}
        for index, entry in enumerate(catalog["pairs"], 1):
            progress_path = directory / "progress" / f"{entry['pairing_key']}.json"
            source = {"design_id": design_id_for_preparation(args.preparation),
                      "phase": args.phase, "graph_catalog_sha256": catalog["catalog_sha256"],
                      "pairing_key": entry["pairing_key"],
                      "backend_id": args.backend_id,
                      "input_snapshot_sha256": getattr(args, "input_snapshot_sha256", None),
                      "implementation_sha256": args.implementation_sha256,
                      "input_file_sha256": {cp.run_id: hashes for cp, hashes in checkpoints},
                      "baseline_validation_mode": BASELINE_VALIDATION_MODE}
            if progress_path.exists():
                progress = read_json(progress_path)
                # Completed shards may have removed checkpoints from the pending set.
                old_inputs = progress.get("input_file_sha256", {})
                for key, hashes in source["input_file_sha256"].items():
                    if old_inputs.get(key) != hashes:
                        raise RuntimeError("H2GCN saved progress checkpoint input differs")
                _equal_fields(progress, {key: value for key, value in source.items()
                                         if key != "input_file_sha256"}, "H2GCN saved progress")
                rows = {row["run_id"]: row for row in progress["rows"]}
                if len(rows) != len(progress["rows"]):
                    raise RuntimeError("duplicate H2GCN progress checkpoint")
            else:
                treatment, control = load_graph_edges(args.graph_root, catalog, entry)
                t_logits = base.h2_predictions_for_graph(networks, x, treatment, device=device, pause_seconds=0)
                c_logits = base.h2_predictions_for_graph(networks, x, control, device=device, pause_seconds=0)
                rows = {checkpoint.run_id: paired_metric_row(
                    checkpoint, bundle=bundle, arrays=arrays, graph_entry=entry, catalog=catalog,
                    treatment_predictions=t_logits[checkpoint.run_id].argmax(axis=-1).astype(np.int64),
                    control_predictions=c_logits[checkpoint.run_id].argmax(axis=-1).astype(np.int64))
                    for checkpoint, _ in checkpoints}
                del t_logits, c_logits, treatment, control
                atomic_json(progress_path, {**source, "rows": list(rows.values())})
            for checkpoint, _ in checkpoints:
                rows_by_run[checkpoint.run_id].append(rows[checkpoint.run_id])
            print(f"H2_GRAPH_PAIR_COMPLETE {args.preparation} {args.dataset} "
                  f"{index}/{len(catalog['pairs'])}", flush=True)
        for checkpoint, inputs in checkpoints:
            _write_shard(args, catalog, checkpoint, inputs, baselines[checkpoint.run_id],
                          rows_by_run[checkpoint.run_id], "graph-major-prepared-h2-operators", 0.0)
    finally:
        networks.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def validate(args) -> dict:
    args.phase = "actual"
    catalogs = {name: load_graph_catalog(args.graph_root, name) for name in DATASET_SPECS}
    if getattr(args, "input_snapshot_value", None) is None:
        raise RuntimeError("final validation requires an authenticated input snapshot")
    rows, baselines, shard_hashes = [], [], {}
    backend_ids = set()
    producer_implementations = []
    expected_paths = set()
    root = result_root(args)
    for dataset, spec in DATASET_SPECS.items():
        catalog = catalogs[dataset]
        for model in FULL_MODELS:
            for split, seed in spec.checkpoint_identities:
                run_id = base.parent_run_id(spec, model, split, seed, args.preparation)
                path = root / dataset / model / "shards" / f"{run_id}.json"
                expected_paths.add(path.resolve())
                payload = read_json(path)
                validate_shard_payload(payload, catalog=catalog, preparation=args.preparation,
                                       input_snapshot_sha256=args.input_snapshot_sha256,
                                       experiment_revision=args.experiment_revision)
                if not _implementation_is_accepted(
                    payload.get("implementation_sha256"),
                    args.accepted_implementation_sha256,
                ):
                    raise RuntimeError(f"{run_id}: inference implementation is not accepted")
                if payload["implementation_sha256"] not in producer_implementations:
                    producer_implementations.append(payload["implementation_sha256"])
                source = args.checkpoint_root / run_id
                observed_inputs = {name: file_sha256(source / name)
                                   for name in ("checkpoint.pt", "config.json", "run_data.npz")}
                if observed_inputs != payload.get("input_file_sha256") or observed_inputs["checkpoint.pt"] != payload["checkpoint_sha256"]:
                    raise RuntimeError(f"{run_id}: checkpoint source files differ")
                snapshot = args.input_snapshot_runs[run_id]
                if observed_inputs != {name: snapshot["files"][name]["sha256"] for name in observed_inputs}:
                    raise RuntimeError(f"{run_id}: checkpoint inputs differ from authenticated snapshot")
                if payload.get("input_snapshot_sha256") != args.input_snapshot_sha256:
                    raise RuntimeError(f"{run_id}: input snapshot identity differs")
                shard_hashes[path.relative_to(root).as_posix()] = file_sha256(path)
                backend_ids.add(payload["backend_id"])
                baselines.append(payload["baseline"])
                rows.extend(payload["pairs"])
    observed_paths = {path.resolve() for path in root.glob("*/*/shards/*.json")}
    if observed_paths != expected_paths:
        raise RuntimeError("actual checkpoint output set differs from the complete fixed panel")
    t_keys = {row["treatment_prediction_key"] for row in rows}
    c_keys = {row["control_prediction_key"] for row in rows}
    expected_rows = expected_paired_row_count(catalogs, args.preparation)
    if (len(baselines) != expected_checkpoint_count(args.preparation) or len(rows) != expected_rows
            or len(t_keys) != expected_rows or len(c_keys) != expected_rows or t_keys & c_keys):
        raise RuntimeError("final baseline count or treatment/control pairing differs")
    producer_implementations = _canonical_implementation_maps(producer_implementations)
    final = root / "final"
    base.atomic_jsonl(final / "baseline_rows.jsonl", baselines)
    base.atomic_jsonl(final / "paired_rows.jsonl", rows)
    result = {"schema_version": SCHEMA_VERSION, "design_id": design_id_for_preparation(args.preparation),
              "preparation": args.preparation, "validated": True,
              "n_model_dataset_cells": len(DATASET_SPECS) * len(FULL_MODELS),
              "n_checkpoint_shards": len(shard_hashes), "n_baseline_rows": len(baselines),
              "n_paired_rows": len(rows), "pairing_validated": True,
              "backend_ids": sorted(backend_ids),
              "n_unique_treatment_prediction_keys": len(t_keys),
              "n_unique_control_prediction_keys": len(c_keys),
              "graph_catalog_sha256": {name: cat["catalog_sha256"] for name, cat in catalogs.items()},
              "checkpoint_shard_sha256": shard_hashes,
              "output_sha256": {name: file_sha256(final / name)
                                 for name in ("baseline_rows.jsonl", "paired_rows.jsonl")},
              "validator_implementation_sha256": args.implementation_sha256,
              "producer_implementation_sha256": producer_implementations,
              "source_hash_manifest_sha256": getattr(args, "source_hash_manifest_sha256", None),
              "input_snapshot_sha256": args.input_snapshot_sha256,
              "experiment_revision": args.experiment_revision, "validated_at": base.now_utc()}
    atomic_json(final / "validation.json", result)
    print(f"PREPARATION_VALIDATED {args.preparation} checkpoints={len(shard_hashes)} paired_rows={len(rows)}", flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("authenticate", "evaluate", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--graph-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--experiment-revision", required=True)
        command.add_argument("--source-hash-manifest", type=Path)
        command.add_argument("--input-snapshot", type=Path)
        if name == "authenticate":
            command.add_argument("--replay-swaps", action="store_true",
                                 help="independently replay again; default authenticates the already-validated graph files")
        if name in {"authenticate", "evaluate"}:
            for key in ("data-root", "array-root", "manifest-root"):
                command.add_argument(f"--{key}", type=Path, required=True)
        if name in {"evaluate", "validate"}:
            command.add_argument("--checkpoint-root", type=Path, required=True)
            command.add_argument("--preparation", choices=PREPARATIONS, required=True)
        if name == "evaluate":
            command.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
            command.add_argument("--model", choices=FULL_MODELS, required=True)
            command.add_argument("--phase", choices=("smoke", "actual"), required=True)
            command.add_argument("--backend-id", required=True,
                                 help="portable backend family such as v100-cuda")
            command.add_argument("--max-checkpoints", type=int, default=1)
            command.add_argument("--max-pairs", type=int, default=1)
            command.add_argument("--baseline-only", action="store_true")
            command.add_argument("--device", default="cuda")
            command.add_argument("--torch-threads", type=int, default=4)
            command.add_argument("--cuda-memory-fraction", type=float, default=0.9)
            command.add_argument(
                "--logit-reproduction-atol", type=float, default=1e-5,
                help="deprecated compatibility option; raw-logit differences are recorded, not gated",
            )
            command.add_argument(
                "--logit-reproduction-rtol", type=float, default=1e-5,
                help="deprecated compatibility option; raw-logit differences are recorded, not gated",
            )
    args = parser.parse_args(argv)
    if (len(args.experiment_revision) != 40
            or any(character not in "0123456789abcdef" for character in args.experiment_revision)):
        parser.error("experiment revision must be a complete lowercase Git commit")
    if args.command == "evaluate":
        if args.phase == "actual" and (args.baseline_only or args.max_checkpoints != 1 or args.max_pairs != 1):
            parser.error("baseline-only and checkpoint/pair selectors are smoke-only")
        if args.phase == "actual" and args.input_snapshot is None:
            parser.error("actual inference requires --input-snapshot")
        if not 1 <= args.max_checkpoints <= 30 or args.max_pairs < 1 or args.torch_threads < 1:
            parser.error("invalid checkpoint, pair or CPU-thread limit")
        if (not args.backend_id
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in args.backend_id)):
            parser.error("backend id must be a lowercase letter/digit/hyphen slug")
        if not 0 < args.cuda_memory_fraction <= 1:
            parser.error("CUDA memory fraction must be in (0, 1]")
        if any(not math.isfinite(value) or value < 0
               for value in (args.logit_reproduction_atol, args.logit_reproduction_rtol)):
            parser.error("logit tolerances must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    for name, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.implementation_sha256 = implementation_hashes()
    if args.input_snapshot is not None:
        from authenticate_reverse_edge_inputs import validate_inventory
        args.input_snapshot_value = read_json(args.input_snapshot)
        validate_inventory(args.input_snapshot_value)
        args.input_snapshot_runs = {row["run_id"]: row for row in args.input_snapshot_value["runs"]}
        args.input_snapshot_sha256 = file_sha256(args.input_snapshot)
    if args.source_hash_manifest is not None:
        pinned = read_json(args.source_hash_manifest)
        args.source_hash_manifest_sha256 = file_sha256(args.source_hash_manifest)
        if args.command == "validate" and "accepted_implementation_sha256" in pinned:
            accepted = pinned["accepted_implementation_sha256"]
            if not isinstance(accepted, list) or not accepted:
                raise RuntimeError("accepted implementation list is empty")
            args.accepted_implementation_sha256 = [
                _validate_implementation_map(value) for value in accepted
            ]
        else:
            expected = _validate_implementation_map(
                pinned.get("implementation_sha256", pinned)
            )
            if args.command != "validate" and expected != args.implementation_sha256:
                raise RuntimeError("deployed inference source hashes differ from pinned manifest")
            args.accepted_implementation_sha256 = [expected]
    else:
        args.accepted_implementation_sha256 = [args.implementation_sha256]
    if args.command == "evaluate":
        torch.set_num_threads(args.torch_threads)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device is unavailable")
            device = torch.device(args.device)
            device_index = torch.cuda.current_device() if device.index is None else device.index
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device_index)
    result = {"authenticate": authenticate, "evaluate": evaluate, "validate": validate}[args.command](args)
    if implementation_hashes() != args.implementation_sha256:
        raise RuntimeError("inference sources changed during execution")
    return result


if __name__ == "__main__":
    main()
