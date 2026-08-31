"""Run the registered PubMed edge-perturbation dose-response experiment.

For each graph replicate, one random ranking of the frozen PubMed tail nodes
defines nested target sets at 5 percentage-point increments.  Every non-zero
level is generated afresh from the registered original graph.  The treatment
raises each selected target's local homophily by one neighbour; the matched
control performs the same number of degree-preserving swaps without changing
the selected targets' local homophily.

Existing GCN and GraphSAGE-mean ERM checkpoints are evaluated on the complete
fixed tail-test subset.  The original-graph baseline is evaluated once.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hfd.datasets import load_dataset  # noqa: E402
from hfd.edge_perturbation import generate_rewire_pair  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.metrics import accuracy, macro_f1  # noqa: E402
from hfd.models import build_model  # noqa: E402
from run_edge_perturbation import (  # noqa: E402
    atomic_json,
    file_sha256,
    git_state,
    lower_process_priority,
    predict,
    validate_fixed_inputs,
    validate_parent_run,
)


SCHEMA_VERSION = 1
DESIGN_ID = "pubmed-tail-edge-dose-response-v1"
DATASET = "pubmed"
MODELS = ("gcn", "graphsage-mean")
PERCENTAGES = tuple(range(0, 101, 5))
NONZERO_PERCENTAGES = PERCENTAGES[1:]
REPLICATE_INDICES = tuple(range(1, 6))
EXPECTED_TAIL_COUNT = 2696
DEFAULT_SEED_BASE = 2026080100
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "edge_perturbation_dose_response_pubmed"
)
BASELINE_OUTPUT_FILES = ("checkpoint_metrics.csv", "predictions.npz")
PAIR_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "checkpoint_metrics.csv",
    "predictions.npz",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def registered_checkpoint_identities(
    manifest: dict,
) -> list[tuple[str, int]]:
    splits = [str(row["split_id"]) for row in manifest["splits"]]
    seeds = [int(seed) for seed in manifest["training_seeds"]]
    identities = [(split, seed) for split in splits for seed in seeds]
    if len(identities) != int(manifest["n_runs"]):
        raise RuntimeError(
            "registered split/seed product differs from the declared run count"
        )
    if len(identities) != 30:
        raise RuntimeError(
            f"the PubMed dose response requires 30 checkpoints; got "
            f"{len(identities)}"
        )
    return identities


def replicate_seed(seed_base: int, replicate_index: int) -> int:
    return int(seed_base) + int(replicate_index)


def tail_ranking(tail_mask: np.ndarray, seed: int) -> np.ndarray:
    """Return the replicate's deterministic permutation of all tail nodes."""
    nodes = np.flatnonzero(np.asarray(tail_mask, dtype=bool)).astype(np.int64)
    return np.random.default_rng(int(seed)).permutation(nodes)


def target_count_for_percentage(tail_count: int, p_percent: int) -> int:
    if p_percent not in PERCENTAGES:
        raise ValueError(f"p_percent must be one of {PERCENTAGES}")
    return (int(tail_count) * int(p_percent)) // 100


def targets_for_percentage(
    ranking: np.ndarray, p_percent: int
) -> np.ndarray:
    count = target_count_for_percentage(int(ranking.size), p_percent)
    return np.asarray(ranking[:count], dtype=np.int64)


def tail_test_indices(
    arrays: np.lib.npyio.NpzFile,
    split: str,
) -> np.ndarray:
    test_idx = arrays[f"{split}.test"].astype(np.int64, copy=False)
    tail_mask = arrays["tail_mask"].astype(bool, copy=False)
    indices = test_idx[tail_mask[test_idx]]
    if indices.size == 0:
        raise RuntimeError(f"{split}: fixed tail-test subset is empty")
    return indices


def subset_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    true = labels[indices]
    predicted = predictions[indices]
    return (
        macro_f1(true, predicted, np.unique(true)),
        accuracy(true, predicted),
    )


def class_composition(labels: np.ndarray, indices: np.ndarray) -> str:
    values, counts = np.unique(labels[indices], return_counts=True)
    return json.dumps(
        {
            int(label): int(count)
            for label, count in zip(values, counts)
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def prediction_key(
    model: str,
    split: str,
    seed: int,
    graph: str,
) -> str:
    return (
        f"{model.replace('-', '_')}__{split.replace('-', '_')}"
        f"__seed_{seed}__{graph}"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty checkpoint table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen_manifest(args) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "code_version": args.code_version,
        "dataset": DATASET,
        "models": list(MODELS),
        "p_percentages": list(PERCENTAGES),
        "graph_replicates": len(REPLICATE_INDICES),
        "graph_replicate_indices": list(REPLICATE_INDICES),
        "replicate_seed_base": args.seed_base,
        "replicate_seed_rule": "replicate_seed_base + replicate_index",
        "tail_definition": "frozen registered PubMed low-h_v tail",
        "tail_count": EXPECTED_TAIL_COUNT,
        "target_ranking": (
            "one numpy default_rng(replicate_seed) permutation of the frozen "
            "tail-node indices per graph replicate"
        ),
        "target_sets": "nested prefixes of the replicate target ranking",
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "graph_start_at_each_p": "registered original PubMed graph",
        "treatment": (
            "one degree-preserving swap per selected target replacing a "
            "different-class neighbour with a same-class neighbour"
        ),
        "control": (
            "one degree-preserving swap per selected target replacing the "
            "target neighbour with another neighbour of the same class "
            "relation"
        ),
        "shared_graph_pairs": "the same pair is evaluated by both models",
        "checkpoint_panel": (
            "registered PubMed split IDs crossed with registered training "
            "seeds; 30 ERM checkpoints per model"
        ),
        "metric_population": "fixed-full-tail-test",
        "evaluation_subset": (
            "complete frozen tail-test subset for each checkpoint split"
        ),
        "metric": (
            "macro-F1 over classes present in that split's fixed tail-test "
            "labels, fixed before scoring, with zero_division=0"
        ),
        "baseline": "registered original graph, evaluated once per checkpoint",
        "actual_generation_attempts": 1,
        "pair_output_files": [*PAIR_OUTPUT_FILES, "metadata.json"],
        "baseline_output_files": [*BASELINE_OUTPUT_FILES, "metadata.json"],
    }


def ensure_frozen_manifest(path: Path, expected: dict) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"dose-response experiment settings differ from {path}"
            )
    else:
        atomic_json(path, expected)


def prepare_actual_experiment(args) -> None:
    actual_root = args.output_root / "actual"
    actual_root.mkdir(parents=True, exist_ok=True)
    manifest_path = actual_root / "experiment_manifest.json"
    ensure_frozen_manifest(manifest_path, frozen_manifest(args))
    args.experiment_manifest_sha256 = file_sha256(manifest_path)


def validate_registered_inputs(bundle, arrays, manifest: dict) -> None:
    validate_fixed_inputs(DATASET, bundle, arrays, manifest)
    if manifest["local_homophily"]["tail_side"] != "low-h_v":
        raise RuntimeError("the PubMed registered tail is not low-h_v")
    if int(np.count_nonzero(arrays["tail_mask"])) != EXPECTED_TAIL_COUNT:
        raise RuntimeError(
            "the frozen PubMed tail count differs from "
            f"{EXPECTED_TAIL_COUNT}"
        )
    live_splits = [split.split_id for split in bundle.splits]
    registered_splits = [
        str(row["split_id"]) for row in manifest["splits"]
    ]
    if live_splits != registered_splits:
        raise RuntimeError("PubMed live split IDs differ from registration")
    if list(bundle.training_seeds) != [
        int(seed) for seed in manifest["training_seeds"]
    ]:
        raise RuntimeError("PubMed live seeds differ from registration")


def _archive_or_reject_incomplete(
    incomplete_dir: Path,
    *,
    phase: str,
    replace_debug: bool,
    restart_incomplete: bool,
) -> None:
    if not incomplete_dir.exists():
        return
    if phase == "actual" and restart_incomplete:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = incomplete_dir.with_name(
            f"{incomplete_dir.name.removesuffix('.in-progress')}"
            f".incomplete-{timestamp}"
        )
        incomplete_dir.replace(archived)
        print(f"archived incomplete attempt at {archived}", flush=True)
        return
    if phase == "debug" and replace_debug:
        shutil.rmtree(incomplete_dir)
        return
    raise FileExistsError(incomplete_dir)


def validate_completed_baseline(path: Path, args) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"completed baseline lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": DATASET,
        "models": list(MODELS),
        "code_version": args.code_version,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: completed baseline mismatch {mismatch}")
    hashes = metadata.get("output_sha256", {})
    for filename in BASELINE_OUTPUT_FILES:
        output = path / filename
        if not output.is_file() or hashes.get(filename) != file_sha256(output):
            raise RuntimeError(f"{path}: missing or altered output {filename}")


def evaluate_baseline(
    *,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    models: Sequence[str],
    identities: Sequence[tuple[str, int]],
    checkpoint_root: Path,
    device: torch.device,
    pause_seconds: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    x = bundle.x.to(device)
    original_edges = np.asarray(
        to_simple_undirected(bundle.edge_index, bundle.num_nodes),
        dtype=np.int64,
    )
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    total = len(models) * len(identities)
    completed = 0

    for model_name in models:
        for split, seed in identities:
            run_id = f"{DATASET}_{model_name}_erm_{split}_seed-{seed}"
            config, checkpoint_path, saved_original = validate_parent_run(
                checkpoint_root / run_id,
                dataset=DATASET,
                model=model_name,
                split=split,
                seed=seed,
                fixed_input_checksums=manifest["array_checksums_sha256"],
            )
            network = build_model(
                model_name,
                int(bundle.x.shape[1]),
                bundle.num_classes,
                config["model_config"],
                num_nodes=bundle.num_nodes,
            )
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            network.load_state_dict(state)
            network.to(device)
            network.eval()
            original_pred = predict(
                network, x, original_edges, device, pause_seconds
            )
            if not np.array_equal(original_pred, saved_original):
                differences = int(
                    np.count_nonzero(original_pred != saved_original)
                )
                raise RuntimeError(
                    f"{run_id}: original predictions fail reproduction "
                    f"({differences} nodes differ)"
                )
            indices = tail_test_indices(arrays, split)
            score, acc = subset_metrics(bundle.y, original_pred, indices)
            key = prediction_key(model_name, split, seed, "original")
            predictions[key] = original_pred
            rows.append(
                {
                    "dataset": DATASET,
                    "model": model_name,
                    "split": split,
                    "seed": seed,
                    "n_tail_test": int(indices.size),
                    "tail_test_class_composition": class_composition(
                        bundle.y, indices
                    ),
                    "tail_test_macro_f1": score,
                    "tail_test_accuracy": acc,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "checkpoint_training_code_version": config[
                        "code_version"
                    ],
                    "prediction_key": key,
                }
            )
            completed += 1
            print(
                f"PubMed baseline: {completed}/{total} checkpoints; "
                f"{model_name} {split} seed {seed}",
                flush=True,
            )
            del network, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del x
    return rows, predictions


def run_baseline(
    args,
    *,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    identities: Sequence[tuple[str, int]],
) -> None:
    phase_root = args.output_root / args.phase
    final_dir = phase_root / "baseline"
    incomplete_dir = phase_root / "baseline.in-progress"
    if final_dir.exists():
        if args.phase == "actual":
            validate_completed_baseline(final_dir, args)
            print(f"validated existing {final_dir}; skipping", flush=True)
            return
        if not args.replace_debug:
            raise FileExistsError(final_dir)
        shutil.rmtree(final_dir)
    _archive_or_reject_incomplete(
        incomplete_dir,
        phase=args.phase,
        replace_debug=args.replace_debug,
        restart_incomplete=args.restart_incomplete,
    )
    phase_root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": args.phase,
        "completed": False,
        "dataset": DATASET,
        "models": list(args.models),
        "checkpoint_identities": [
            {"split": split, "seed": seed} for split, seed in identities
        ],
        "n_checkpoints_per_model": len(identities),
        "metric_population": "fixed-full-tail-test",
        "evaluation_subset": "complete fixed tail-test subset",
        "code_version": args.code_version,
        "git_dirty": args.git_dirty,
        "experiment_manifest_sha256": (
            args.experiment_manifest_sha256
            if args.phase == "actual"
            else None
        ),
        "started_at": now_utc(),
        "resource_limits": resource_record(args),
    }
    atomic_json(incomplete_dir / "metadata.json", metadata)
    try:
        started = time.perf_counter()
        rows, predictions = evaluate_baseline(
            bundle=bundle,
            arrays=arrays,
            manifest=manifest,
            models=args.models,
            identities=identities,
            checkpoint_root=args.checkpoint_root,
            device=torch.device(args.device),
            pause_seconds=args.pause_seconds,
        )
        write_csv(incomplete_dir / "checkpoint_metrics.csv", rows)
        np.savez_compressed(
            incomplete_dir / "predictions.npz", **predictions
        )
        output_hashes = {
            filename: file_sha256(incomplete_dir / filename)
            for filename in BASELINE_OUTPUT_FILES
        }
        metadata.update(
            {
                "completed": True,
                "completed_at": now_utc(),
                "evaluation_seconds": time.perf_counter() - started,
                "n_checkpoint_rows": len(rows),
                "all_original_predictions_reproduced": True,
                "output_sha256": output_hashes,
            }
        )
        atomic_json(incomplete_dir / "metadata.json", metadata)
        incomplete_dir.replace(final_dir)
        print(f"completed {final_dir}", flush=True)
    except Exception:
        print(f"incomplete output retained at {incomplete_dir}", flush=True)
        raise


def evaluate_pair(
    *,
    pair,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    models: Sequence[str],
    identities: Sequence[tuple[str, int]],
    checkpoint_root: Path,
    device: torch.device,
    pause_seconds: float,
    replicate_index: int,
    replicate_seed_value: int,
    p_percent: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    x = bundle.x.to(device)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    total = len(models) * len(identities)
    completed = 0

    for model_name in models:
        for split, seed in identities:
            run_id = f"{DATASET}_{model_name}_erm_{split}_seed-{seed}"
            config, checkpoint_path, _ = validate_parent_run(
                checkpoint_root / run_id,
                dataset=DATASET,
                model=model_name,
                split=split,
                seed=seed,
                fixed_input_checksums=manifest["array_checksums_sha256"],
            )
            network = build_model(
                model_name,
                int(bundle.x.shape[1]),
                bundle.num_classes,
                config["model_config"],
                num_nodes=bundle.num_nodes,
            )
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            network.load_state_dict(state)
            network.to(device)
            network.eval()
            treatment_pred = predict(
                network,
                x,
                pair.treatment_edge_index,
                device,
                pause_seconds,
            )
            control_pred = predict(
                network,
                x,
                pair.control_edge_index,
                device,
                pause_seconds,
            )
            indices = tail_test_indices(arrays, split)
            treatment_f1, treatment_acc = subset_metrics(
                bundle.y, treatment_pred, indices
            )
            control_f1, control_acc = subset_metrics(
                bundle.y, control_pred, indices
            )
            prefix = prediction_key(
                model_name, split, seed, "treatment"
            ).rsplit("__", 1)[0]
            predictions[f"{prefix}__treatment"] = treatment_pred
            predictions[f"{prefix}__control"] = control_pred
            rows.append(
                {
                    "dataset": DATASET,
                    "model": model_name,
                    "replicate_index": replicate_index,
                    "replicate_seed": replicate_seed_value,
                    "p_percent": p_percent,
                    "n_selected_tail_nodes": int(pair.targets.size),
                    "split": split,
                    "seed": seed,
                    "n_tail_test": int(indices.size),
                    "tail_test_class_composition": class_composition(
                        bundle.y, indices
                    ),
                    "treatment_macro_f1": treatment_f1,
                    "control_macro_f1": control_f1,
                    "treatment_minus_control_macro_f1": (
                        treatment_f1 - control_f1
                    ),
                    "treatment_accuracy": treatment_acc,
                    "control_accuracy": control_acc,
                    "treatment_minus_control_accuracy": (
                        treatment_acc - control_acc
                    ),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "checkpoint_training_code_version": config[
                        "code_version"
                    ],
                    "prediction_key_prefix": prefix,
                }
            )
            completed += 1
            print(
                f"PubMed replicate {replicate_index} p={p_percent}: "
                f"{completed}/{total} checkpoints; "
                f"{model_name} {split} seed {seed}",
                flush=True,
            )
            del network, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del x
    return rows, predictions


def validate_completed_pair(
    path: Path,
    *,
    replicate_index: int,
    p_percent: int,
    args,
) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"completed graph pair lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": DATASET,
        "models": list(MODELS),
        "replicate_index": replicate_index,
        "replicate_seed": replicate_seed(
            args.seed_base, replicate_index
        ),
        "p_percent": p_percent,
        "requested_targets": target_count_for_percentage(
            EXPECTED_TAIL_COUNT, p_percent
        ),
        "code_version": args.code_version,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: completed pair mismatch {mismatch}")
    hashes = metadata.get("output_sha256", {})
    for filename in PAIR_OUTPUT_FILES:
        output = path / filename
        if not output.is_file() or hashes.get(filename) != file_sha256(output):
            raise RuntimeError(f"{path}: missing or altered output {filename}")


def resource_record(args) -> dict:
    return {
        "device": str(args.device),
        "torch_cpu_threads": args.torch_threads,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "pause_seconds_after_each_forward": args.pause_seconds,
        "below_normal_process_priority": args.priority_lowered,
        "parallel_workers": 0,
    }


def run_graph_pair(
    args,
    *,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    identities: Sequence[tuple[str, int]],
    replicate_index: int,
    p_percent: int,
) -> None:
    replicate_root = (
        args.output_root
        / args.phase
        / f"replicate-{replicate_index:04d}"
    )
    final_dir = replicate_root / f"pct-{p_percent:03d}"
    incomplete_dir = replicate_root / f"pct-{p_percent:03d}.in-progress"
    if final_dir.exists():
        if args.phase == "actual":
            validate_completed_pair(
                final_dir,
                replicate_index=replicate_index,
                p_percent=p_percent,
                args=args,
            )
            print(f"validated existing {final_dir}; skipping", flush=True)
            return
        if not args.replace_debug:
            raise FileExistsError(final_dir)
        shutil.rmtree(final_dir)
    _archive_or_reject_incomplete(
        incomplete_dir,
        phase=args.phase,
        replace_debug=args.replace_debug,
        restart_incomplete=args.restart_incomplete,
    )
    replicate_root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()

    seed = replicate_seed(args.seed_base, replicate_index)
    ranking = tail_ranking(arrays["tail_mask"], seed)
    targets = targets_for_percentage(ranking, p_percent)
    try:
        generation_started = time.perf_counter()
        pair = generate_rewire_pair(
            bundle.edge_index,
            bundle.y,
            arrays["tail_mask"],
            desired_same=True,
            pair_seed=seed,
            target_nodes=targets,
            max_attempts=(
                1
                if args.phase == "actual"
                else args.max_generation_attempts
            ),
        )
        generation_seconds = time.perf_counter() - generation_started
        if not np.array_equal(
            np.sort(pair.targets), np.sort(targets)
        ):
            raise RuntimeError("generated target set differs from ranking prefix")
        np.savez_compressed(
            incomplete_dir / "graphs.npz",
            treatment_edge_index=pair.treatment_edge_index,
            control_edge_index=pair.control_edge_index,
            targets=targets,
            target_order=pair.target_order,
            tail_ranking=ranking,
        )
        atomic_json(incomplete_dir / "swaps.json", pair.swap_records())
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "design_id": DESIGN_ID,
            "phase": args.phase,
            "completed": False,
            "dataset": DATASET,
            "models": list(args.models),
            "replicate_index": replicate_index,
            "replicate_seed": seed,
            "p_percent": p_percent,
            "tail_count": int(ranking.size),
            "requested_targets": int(targets.size),
            "selected_targets": int(pair.targets.size),
            "target_count_rule": "floor(p_percent * tail_count / 100)",
            "target_selection": (
                "prefix of the replicate's frozen random tail-node ranking"
            ),
            "graph_start": "registered original PubMed graph",
            "generation_attempt": pair.attempt,
            "generation_retry_failures": pair.retry_failures,
            "generation_seconds": generation_seconds,
            "audit": pair.audit,
            "checkpoint_identities": [
                {"split": split, "seed": checkpoint_seed}
                for split, checkpoint_seed in identities
            ],
            "n_checkpoints_per_model": len(identities),
            "metric_population": "fixed-full-tail-test",
            "evaluation_subset": "complete fixed tail-test subset",
            "code_version": args.code_version,
            "git_dirty": args.git_dirty,
            "seed_base": args.seed_base,
            "experiment_manifest_sha256": (
                args.experiment_manifest_sha256
                if args.phase == "actual"
                else None
            ),
            "started_at": now_utc(),
            "resource_limits": resource_record(args),
        }
        atomic_json(incomplete_dir / "metadata.json", metadata)

        evaluation_started = time.perf_counter()
        rows, predictions = evaluate_pair(
            pair=pair,
            bundle=bundle,
            arrays=arrays,
            manifest=manifest,
            models=args.models,
            identities=identities,
            checkpoint_root=args.checkpoint_root,
            device=torch.device(args.device),
            pause_seconds=args.pause_seconds,
            replicate_index=replicate_index,
            replicate_seed_value=seed,
            p_percent=p_percent,
        )
        write_csv(incomplete_dir / "checkpoint_metrics.csv", rows)
        np.savez_compressed(
            incomplete_dir / "predictions.npz", **predictions
        )
        output_hashes = {
            filename: file_sha256(incomplete_dir / filename)
            for filename in PAIR_OUTPUT_FILES
        }
        metadata.update(
            {
                "completed": True,
                "completed_at": now_utc(),
                "evaluation_seconds": (
                    time.perf_counter() - evaluation_started
                ),
                "n_checkpoint_rows": len(rows),
                "output_sha256": output_hashes,
            }
        )
        atomic_json(incomplete_dir / "metadata.json", metadata)
        incomplete_dir.replace(final_dir)
        print(f"completed {final_dir}", flush=True)
    except Exception:
        print(f"incomplete output retained at {incomplete_dir}", flush=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("debug", "actual"), required=True)
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        default=list(REPLICATE_INDICES),
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=list(PERCENTAGES),
    )
    parser.add_argument(
        "--models",
        choices=MODELS,
        nargs="+",
        default=list(MODELS),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--debug-checkpoints", type=int, default=1)
    parser.add_argument("--max-generation-attempts", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--replace-debug", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    args.replicates = sorted(set(args.replicates))
    args.percentages = sorted(set(args.percentages))
    args.models = list(dict.fromkeys(args.models))
    invalid_replicates = set(args.replicates) - set(REPLICATE_INDICES)
    if invalid_replicates:
        raise SystemExit(
            f"replicates must be selected from {REPLICATE_INDICES}; got "
            f"{sorted(invalid_replicates)}"
        )
    invalid_percentages = set(args.percentages) - set(PERCENTAGES)
    if invalid_percentages:
        raise SystemExit(
            f"percentages must be selected from {PERCENTAGES}; got "
            f"{sorted(invalid_percentages)}"
        )
    if not args.replicates:
        raise SystemExit("at least one graph replicate is required")
    if not args.percentages:
        raise SystemExit("at least one percentage is required")
    if args.phase == "actual" and args.models != list(MODELS):
        raise SystemExit("actual collection requires both registered models")
    if args.debug_checkpoints < 1 or args.debug_checkpoints > 30:
        raise SystemExit("--debug-checkpoints must be between 1 and 30")
    if args.max_generation_attempts < 1:
        raise SystemExit("--max-generation-attempts must be positive")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        raise SystemExit("--cuda-memory-fraction must be in (0, 1]")
    if args.pause_seconds < 0.0:
        raise SystemExit("--pause-seconds must be non-negative")
    if not str(args.device).startswith("cuda"):
        raise SystemExit("the dose-response runner requires a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    for name in (
        "data_root",
        "array_root",
        "manifest_root",
        "checkpoint_root",
        "output_root",
    ):
        path = getattr(args, name).expanduser().resolve()
        if name != "output_root" and not path.exists():
            raise SystemExit(f"{name.replace('_', ' ')} does not exist: {path}")
        setattr(args, name, path)
    args.output_root.mkdir(parents=True, exist_ok=True)

    args.code_version, args.git_dirty = git_state()
    if args.phase == "actual" and args.git_dirty:
        raise SystemExit("actual collection requires a clean committed worktree")
    if args.phase == "actual":
        prepare_actual_experiment(args)
    else:
        args.experiment_manifest_sha256 = None

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    device_index = torch.device(args.device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(
        args.cuda_memory_fraction, device=device_index
    )
    args.priority_lowered = lower_process_priority()

    bundle = load_dataset(DATASET, args.data_root)
    arrays = np.load(
        args.array_root / f"{DATASET}.npz", allow_pickle=False
    )
    try:
        manifest = json.loads(
            (args.manifest_root / f"{DATASET}.json").read_text(
                encoding="utf-8"
            )
        )
        validate_registered_inputs(bundle, arrays, manifest)
        identities = registered_checkpoint_identities(manifest)
        if args.phase == "debug":
            identities = identities[: args.debug_checkpoints]

        if args.phase == "actual" or 0 in args.percentages:
            run_baseline(
                args,
                bundle=bundle,
                arrays=arrays,
                manifest=manifest,
                identities=identities,
            )
        for replicate_index in args.replicates:
            for p_percent in args.percentages:
                if p_percent == 0:
                    continue
                run_graph_pair(
                    args,
                    bundle=bundle,
                    arrays=arrays,
                    manifest=manifest,
                    identities=identities,
                    replicate_index=replicate_index,
                    p_percent=p_percent,
                )
    finally:
        arrays.close()


if __name__ == "__main__":
    main()
