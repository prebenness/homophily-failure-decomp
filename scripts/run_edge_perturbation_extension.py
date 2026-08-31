"""Run the registered non-rectangular edge-perturbation extension panel.

The extension has four explicit dataset/model cells:

* PubMed / GCN and Roman-empire / GCN reuse graph pairs from the original
  edge-perturbation experiment.
* arxiv-year / GraphSAGE-mean and ogbn-arxiv / GraphSAGE-mean generate new
  graph pairs.

Every dataset uses the exact split IDs and training seeds in its registered
dataset manifest. Outputs are separate from the original experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
    PAIR_OUTPUT_FILES,
    atomic_json,
    file_sha256,
    lower_process_priority,
    predict,
    validate_fixed_inputs,
    validate_parent_run,
)


DESIGN_ID = "tail-half-registered-cell-extension-v1"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "edge_perturbation_extension"
DEFAULT_SOURCE_PAIR_ROOT = (
    REPO_ROOT / "artifacts" / "edge_perturbation" / "actual"
)


@dataclass(frozen=True)
class ExtensionCell:
    dataset: str
    model: str
    graph_mode: str

    def as_record(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "graph_mode": self.graph_mode,
        }


REGISTERED_CELLS = (
    ExtensionCell("pubmed", "gcn", "reuse-original"),
    ExtensionCell("roman-empire", "gcn", "reuse-original"),
    ExtensionCell("arxiv-year", "graphsage-mean", "generate"),
    ExtensionCell("ogbn-arxiv", "graphsage-mean", "generate"),
)
CELL_BY_DATASET = {cell.dataset: cell for cell in REGISTERED_CELLS}


@dataclass
class PairInput:
    treatment_edge_index: np.ndarray
    control_edge_index: np.ndarray
    targets: np.ndarray
    target_order: np.ndarray
    pair_seed: int
    requested_targets: int
    attempt: int
    retry_failures: list[str]
    audit: dict
    graph_source: dict


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def repo_relative_posix(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"reused source pair must be below the repository root: {path}"
        ) from exc


def registered_checkpoint_identities(manifest: dict) -> list[tuple[str, int]]:
    """Return the manifest's exact split/seed panel in registered order."""
    split_ids = [str(row["split_id"]) for row in manifest["splits"]]
    seeds = [int(seed) for seed in manifest["training_seeds"]]
    if len(split_ids) != len(set(split_ids)):
        raise RuntimeError("dataset manifest contains duplicate split IDs")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("dataset manifest contains duplicate training seeds")
    identities = [(split, seed) for split in split_ids for seed in seeds]
    if len(identities) != int(manifest["n_runs"]):
        raise RuntimeError(
            "dataset manifest n_runs differs from its split/seed product"
        )
    if len(identities) != 30:
        raise RuntimeError(
            f"extension cells require 30 registered checkpoints; found "
            f"{len(identities)}"
        )
    return identities


def desired_same_from_manifest(manifest: dict) -> bool:
    side = manifest["local_homophily"]["tail_side"]
    if side == "low-h_v":
        return True
    if side == "high-h_v":
        return False
    raise RuntimeError(f"unknown registered tail side: {side!r}")


def frozen_manifest(args, repo_root: Path = REPO_ROOT) -> dict:
    source_manifest = args.source_pair_root / "experiment_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "code_version": args.code_version,
        "sample_fraction": 0.5,
        "seed_base": args.seed_base,
        "cells": [cell.as_record() for cell in REGISTERED_CELLS],
        "checkpoint_panel": (
            "exact dataset-manifest split IDs crossed with registered "
            "training seeds; 30 checkpoints per cell"
        ),
        "target_selection": "uniform sample without replacement",
        "targets_per_dataset": "floor(0.5 * frozen tail count)",
        "swaps_per_target_per_condition": 1,
        "generated_pair_max_attempts": 1,
        "source_pair_root": repo_relative_posix(
            args.source_pair_root, repo_root
        ),
        "source_experiment_manifest_sha256": file_sha256(source_manifest),
        "pair_output_files": [*PAIR_OUTPUT_FILES, "metadata.json"],
    }


def ensure_frozen_manifest(path: Path, expected: dict) -> None:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"extension experiment settings differ from {path}"
            )
    else:
        atomic_json(path, expected)


def prepare_actual_experiment(
    args, repo_root: Path = REPO_ROOT
) -> None:
    actual_root = args.output_root / "actual"
    actual_root.mkdir(parents=True, exist_ok=True)
    manifest_path = actual_root / "experiment_manifest.json"
    ensure_frozen_manifest(
        manifest_path, frozen_manifest(args, repo_root)
    )
    args.experiment_manifest_sha256 = file_sha256(manifest_path)

    requested = set(args.pair_indices)
    for dataset in args.datasets:
        cell = CELL_BY_DATASET[dataset]
        dataset_root = actual_root / cell.dataset
        existing = set()
        if dataset_root.is_dir():
            for path in dataset_root.glob("pair-[0-9][0-9][0-9][0-9]"):
                if path.is_dir():
                    existing.add(int(path.name.removeprefix("pair-")))
        combined = existing | requested
        if combined and combined != set(range(1, max(combined) + 1)):
            raise RuntimeError(
                f"{cell.dataset}: extension pair indices must be contiguous "
                f"from 1; existing={sorted(existing)}, "
                f"requested={sorted(requested)}"
            )
        if cell.graph_mode == "reuse-original":
            for pair_index in requested:
                source = (
                    args.source_pair_root
                    / cell.dataset
                    / f"pair-{pair_index:04d}"
                )
                if not source.is_dir():
                    raise FileNotFoundError(source)


def _validate_source_pair(
    source_dir: Path,
    *,
    dataset: str,
    pair_index: int,
) -> tuple[dict, dict[str, str]]:
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "phase": "actual",
        "completed": True,
        "dataset": dataset,
        "pair_index": pair_index,
        "sample_fraction": 0.5,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            f"{source_dir}: source-pair metadata mismatch {mismatch}"
        )
    recorded_hashes = metadata.get("output_sha256", {})
    selected_hashes: dict[str, str] = {}
    for filename in ("graphs.npz", "swaps.json"):
        path = source_dir / filename
        observed = file_sha256(path) if path.is_file() else None
        recorded = recorded_hashes.get(filename)
        if observed is None or observed != recorded:
            raise RuntimeError(
                f"{source_dir}: missing or altered source output {filename}"
            )
        selected_hashes[filename] = observed
    return metadata, selected_hashes


def copy_reused_pair(
    source_dir: Path,
    destination_dir: Path,
    *,
    dataset: str,
    pair_index: int,
    repo_root: Path = REPO_ROOT,
) -> PairInput:
    """Copy and authenticate one original-experiment graph pair."""
    metadata, source_hashes = _validate_source_pair(
        source_dir,
        dataset=dataset,
        pair_index=pair_index,
    )
    for filename in ("graphs.npz", "swaps.json"):
        shutil.copyfile(source_dir / filename, destination_dir / filename)
        if file_sha256(destination_dir / filename) != source_hashes[filename]:
            raise RuntimeError(f"copied {filename} differs from its source")

    with np.load(destination_dir / "graphs.npz", allow_pickle=False) as graphs:
        required = {
            "treatment_edge_index",
            "control_edge_index",
            "targets",
            "target_order",
        }
        if not required.issubset(graphs.files):
            raise RuntimeError(
                f"{source_dir / 'graphs.npz'} lacks required graph arrays"
            )
        treatment = graphs["treatment_edge_index"].astype(
            np.int64, copy=True
        )
        control = graphs["control_edge_index"].astype(np.int64, copy=True)
        targets = graphs["targets"].astype(np.int64, copy=True)
        target_order = graphs["target_order"].astype(np.int64, copy=True)

    graph_source = {
        "mode": "reuse-original",
        "source_pair": repo_relative_posix(source_dir, repo_root),
        "source_metadata_sha256": file_sha256(source_dir / "metadata.json"),
        "source_output_sha256": source_hashes,
    }
    return PairInput(
        treatment_edge_index=treatment,
        control_edge_index=control,
        targets=targets,
        target_order=target_order,
        pair_seed=int(metadata["pair_seed"]),
        requested_targets=int(metadata["requested_targets"]),
        attempt=int(metadata["generation_attempt"]),
        retry_failures=list(metadata.get("generation_retry_failures", [])),
        audit=dict(metadata["audit"]),
        graph_source=graph_source,
    )


def generate_pair(
    *,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    pair_seed: int,
    target_count: int | None,
    max_attempts: int,
    destination_dir: Path,
) -> PairInput:
    pair = generate_rewire_pair(
        bundle.edge_index,
        bundle.y,
        arrays["tail_mask"],
        desired_same=desired_same_from_manifest(manifest),
        pair_seed=pair_seed,
        sample_fraction=0.5,
        target_count=target_count,
        max_attempts=max_attempts,
    )
    np.savez_compressed(
        destination_dir / "graphs.npz",
        treatment_edge_index=pair.treatment_edge_index,
        control_edge_index=pair.control_edge_index,
        targets=pair.targets,
        target_order=pair.target_order,
    )
    atomic_json(destination_dir / "swaps.json", pair.swap_records())
    return PairInput(
        treatment_edge_index=pair.treatment_edge_index,
        control_edge_index=pair.control_edge_index,
        targets=pair.targets,
        target_order=pair.target_order,
        pair_seed=pair.pair_seed,
        requested_targets=pair.requested_targets,
        attempt=pair.attempt,
        retry_failures=pair.retry_failures,
        audit=pair.audit,
        graph_source={"mode": "generate"},
    )


def subset_metrics(
    labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    if indices.size == 0:
        raise RuntimeError("the evaluated subset is empty")
    true = labels[indices]
    predicted = predictions[indices]
    return (
        macro_f1(true, predicted, np.unique(true)),
        accuracy(true, predicted),
    )


def prediction_key(model: str, split: str, seed: int, graph: str) -> str:
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


def evaluate_cell(
    *,
    cell: ExtensionCell,
    pair: PairInput,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    checkpoint_root: Path,
    device: torch.device,
    pause_seconds: float,
    pair_index: int,
    identities: Sequence[tuple[str, int]],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    x = bundle.x.to(device)
    original_edges = np.asarray(
        to_simple_undirected(bundle.edge_index, bundle.num_nodes),
        dtype=np.int64,
    )
    target_mask = np.zeros(bundle.num_nodes, dtype=bool)
    target_mask[pair.targets] = True
    rows: list[dict] = []
    saved_predictions: dict[str, np.ndarray] = {}

    for position, (split, seed) in enumerate(identities, start=1):
        test_idx = arrays[f"{split}.test"].astype(np.int64, copy=False)
        sampled_test_idx = test_idx[target_mask[test_idx]]
        class_values, class_counts = np.unique(
            bundle.y[sampled_test_idx], return_counts=True
        )
        class_composition = json.dumps(
            {
                int(label): int(count)
                for label, count in zip(class_values, class_counts)
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        run_id = (
            f"{cell.dataset}_{cell.model}_erm_{split}_seed-{seed}"
        )
        run_dir = checkpoint_root / run_id
        config, checkpoint_path, saved_original = validate_parent_run(
            run_dir,
            dataset=cell.dataset,
            model=cell.model,
            split=split,
            seed=seed,
            fixed_input_checksums=manifest["array_checksums_sha256"],
        )
        model = build_model(
            cell.model,
            int(bundle.x.shape[1]),
            bundle.num_classes,
            config["model_config"],
            num_nodes=bundle.num_nodes,
        )
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        original_pred = predict(
            model, x, original_edges, device, pause_seconds
        )
        if not np.array_equal(original_pred, saved_original):
            differences = int(
                np.count_nonzero(original_pred != saved_original)
            )
            raise RuntimeError(
                f"{run_id}: original predictions fail reproduction "
                f"({differences} nodes differ)"
            )
        treatment_pred = predict(
            model,
            x,
            pair.treatment_edge_index,
            device,
            pause_seconds,
        )
        control_pred = predict(
            model,
            x,
            pair.control_edge_index,
            device,
            pause_seconds,
        )

        original_f1, original_acc = subset_metrics(
            bundle.y, original_pred, sampled_test_idx
        )
        treatment_f1, treatment_acc = subset_metrics(
            bundle.y, treatment_pred, sampled_test_idx
        )
        control_f1, control_acc = subset_metrics(
            bundle.y, control_pred, sampled_test_idx
        )
        whole_original_f1, whole_original_acc = subset_metrics(
            bundle.y, original_pred, test_idx
        )
        whole_treatment_f1, whole_treatment_acc = subset_metrics(
            bundle.y, treatment_pred, test_idx
        )
        whole_control_f1, whole_control_acc = subset_metrics(
            bundle.y, control_pred, test_idx
        )

        base_key = prediction_key(cell.model, split, seed, "original")
        saved_predictions[base_key] = original_pred
        saved_predictions[
            prediction_key(cell.model, split, seed, "treatment")
        ] = treatment_pred
        saved_predictions[
            prediction_key(cell.model, split, seed, "control")
        ] = control_pred
        rows.append(
            {
                "dataset": cell.dataset,
                "model": cell.model,
                "pair_index": pair_index,
                "pair_seed": pair.pair_seed,
                "split": split,
                "seed": seed,
                "n_sampled_test": int(sampled_test_idx.size),
                "sampled_test_class_composition": class_composition,
                "original_macro_f1": original_f1,
                "treatment_macro_f1": treatment_f1,
                "control_macro_f1": control_f1,
                "treatment_minus_control_macro_f1": (
                    treatment_f1 - control_f1
                ),
                "treatment_minus_original_macro_f1": (
                    treatment_f1 - original_f1
                ),
                "control_minus_original_macro_f1": (
                    control_f1 - original_f1
                ),
                "original_accuracy": original_acc,
                "treatment_accuracy": treatment_acc,
                "control_accuracy": control_acc,
                "treatment_minus_control_accuracy": (
                    treatment_acc - control_acc
                ),
                "whole_test_original_macro_f1": whole_original_f1,
                "whole_test_treatment_macro_f1": whole_treatment_f1,
                "whole_test_control_macro_f1": whole_control_f1,
                "whole_test_treatment_minus_control_macro_f1": (
                    whole_treatment_f1 - whole_control_f1
                ),
                "whole_test_original_accuracy": whole_original_acc,
                "whole_test_treatment_accuracy": whole_treatment_acc,
                "whole_test_control_accuracy": whole_control_acc,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_training_code_version": config["code_version"],
                "prediction_key_prefix": base_key.rsplit("__", 1)[0],
            }
        )
        print(
            f"{cell.dataset}/{cell.model} pair {pair_index}: "
            f"{position}/{len(identities)} checkpoints; {split} seed {seed}",
            flush=True,
        )
        del model, state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del x
    return rows, saved_predictions


def validate_completed_pair(
    path: Path,
    *,
    cell: ExtensionCell,
    pair_index: int,
    args,
) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"completed extension pair lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": cell.dataset,
        "model": cell.model,
        "graph_mode": cell.graph_mode,
        "pair_index": pair_index,
        "sample_fraction": 0.5,
        "code_version": args.code_version,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: completed metadata mismatch {mismatch}")
    hashes = metadata.get("output_sha256", {})
    for filename in PAIR_OUTPUT_FILES:
        output = path / filename
        if not output.is_file() or hashes.get(filename) != file_sha256(output):
            raise RuntimeError(f"{path}: missing or altered output {filename}")


def run_cell_pair(args, cell: ExtensionCell, pair_index: int) -> None:
    phase_root = args.output_root / args.phase
    dataset_root = phase_root / cell.dataset
    final_dir = dataset_root / f"pair-{pair_index:04d}"
    incomplete_dir = dataset_root / f"pair-{pair_index:04d}.in-progress"

    if final_dir.exists():
        if args.phase == "actual":
            validate_completed_pair(
                final_dir,
                cell=cell,
                pair_index=pair_index,
                args=args,
            )
            print(f"validated existing {final_dir}; skipping", flush=True)
            return
        if not args.replace_debug:
            raise FileExistsError(final_dir)
        shutil.rmtree(final_dir)
    if incomplete_dir.exists():
        if args.phase == "actual" and args.restart_incomplete:
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            archived = dataset_root / (
                f"pair-{pair_index:04d}.incomplete-{timestamp}"
            )
            incomplete_dir.replace(archived)
        elif args.phase == "actual" or not args.replace_debug:
            raise FileExistsError(incomplete_dir)
        else:
            shutil.rmtree(incomplete_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()

    arrays = None
    try:
        bundle = load_dataset(cell.dataset, args.data_root)
        arrays = np.load(
            args.array_root / f"{cell.dataset}.npz", allow_pickle=False
        )
        manifest = json.loads(
            (
                args.manifest_root / f"{cell.dataset}.json"
            ).read_text(encoding="utf-8")
        )
        validate_fixed_inputs(cell.dataset, bundle, arrays, manifest)
        identities = registered_checkpoint_identities(manifest)
        if args.phase == "debug":
            identities = identities[: args.debug_checkpoints]
        live_splits = [split.split_id for split in bundle.splits]
        registered_splits = [
            str(row["split_id"]) for row in manifest["splits"]
        ]
        if live_splits != registered_splits:
            raise RuntimeError(
                f"{cell.dataset}: live split IDs differ from registration"
            )
        if list(bundle.training_seeds) != [
            int(seed) for seed in manifest["training_seeds"]
        ]:
            raise RuntimeError(
                f"{cell.dataset}: live seeds differ from registration"
            )

        generation_started = time.perf_counter()
        if cell.graph_mode == "reuse-original":
            source_dir = (
                args.source_pair_root
                / cell.dataset
                / f"pair-{pair_index:04d}"
            )
            pair = copy_reused_pair(
                source_dir,
                incomplete_dir,
                dataset=cell.dataset,
                pair_index=pair_index,
            )
        else:
            target_count = (
                args.debug_target_count if args.phase == "debug" else None
            )
            pair = generate_pair(
                bundle=bundle,
                arrays=arrays,
                manifest=manifest,
                pair_seed=args.seed_base + pair_index,
                target_count=target_count,
                max_attempts=(
                    1
                    if args.phase == "actual"
                    else args.max_generation_attempts
                ),
                destination_dir=incomplete_dir,
            )
        generation_seconds = time.perf_counter() - generation_started

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "design_id": DESIGN_ID,
            "phase": args.phase,
            "completed": False,
            "dataset": cell.dataset,
            "model": cell.model,
            "graph_mode": cell.graph_mode,
            "graph_source": pair.graph_source,
            "pair_index": pair_index,
            "pair_seed": pair.pair_seed,
            "sample_fraction": 0.5,
            "requested_targets": pair.requested_targets,
            "selected_targets": int(pair.targets.size),
            "target_sampling": "uniform sample without replacement",
            "generation_attempt": pair.attempt,
            "generation_retry_failures": pair.retry_failures,
            "generation_seconds": generation_seconds,
            "audit": pair.audit,
            "checkpoint_identities": [
                {"split": split, "seed": seed}
                for split, seed in identities
            ],
            "n_checkpoints": len(identities),
            "code_version": args.code_version,
            "git_dirty": args.git_dirty,
            "seed_base": args.seed_base,
            "experiment_manifest_sha256": (
                args.experiment_manifest_sha256
                if args.phase == "actual"
                else None
            ),
            "started_at": now_utc(),
            "resource_limits": {
                "device": str(args.device),
                "torch_cpu_threads": args.torch_threads,
                "cuda_memory_fraction": args.cuda_memory_fraction,
                "pause_seconds_after_each_forward": args.pause_seconds,
                "below_normal_process_priority": args.priority_lowered,
                "parallel_workers": 0,
            },
        }
        atomic_json(incomplete_dir / "metadata.json", metadata)

        evaluation_started = time.perf_counter()
        rows, predictions = evaluate_cell(
            cell=cell,
            pair=pair,
            bundle=bundle,
            arrays=arrays,
            manifest=manifest,
            checkpoint_root=args.checkpoint_root,
            device=torch.device(args.device),
            pause_seconds=args.pause_seconds,
            pair_index=pair_index,
            identities=identities,
        )
        write_csv(incomplete_dir / "checkpoint_metrics.csv", rows)
        np.savez_compressed(
            incomplete_dir / "predictions.npz", **predictions
        )
        output_hashes = {
            filename: file_sha256(incomplete_dir / filename)
            for filename in PAIR_OUTPUT_FILES
        }
        if cell.graph_mode == "reuse-original":
            source_hashes = pair.graph_source["source_output_sha256"]
            for filename in ("graphs.npz", "swaps.json"):
                if output_hashes[filename] != source_hashes[filename]:
                    raise RuntimeError(
                        f"copied {filename} no longer matches its source"
                    )
        metadata.update(
            {
                "completed": True,
                "completed_at": now_utc(),
                "evaluation_seconds": (
                    time.perf_counter() - evaluation_started
                ),
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
    finally:
        if arrays is not None:
            arrays.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("debug", "actual"), required=True)
    parser.add_argument("--pair-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--datasets",
        choices=tuple(CELL_BY_DATASET),
        nargs="+",
        default=list(CELL_BY_DATASET),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--source-pair-root",
        type=Path,
        default=DEFAULT_SOURCE_PAIR_ROOT,
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=2026073100)
    parser.add_argument("--debug-target-count", type=int, default=10)
    parser.add_argument("--debug-checkpoints", type=int, default=1)
    parser.add_argument("--max-generation-attempts", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--replace-debug", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if any(index < 1 for index in args.pair_indices):
        raise SystemExit("pair indices must be positive")
    args.pair_indices = sorted(set(args.pair_indices))
    args.datasets = list(dict.fromkeys(args.datasets))
    if args.debug_target_count < 1:
        raise SystemExit("--debug-target-count must be positive")
    if args.debug_checkpoints < 1 or args.debug_checkpoints > 30:
        raise SystemExit("--debug-checkpoints must be between 1 and 30")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        raise SystemExit("--cuda-memory-fraction must be in (0, 1]")
    if args.pause_seconds < 0.0:
        raise SystemExit("--pause-seconds must be non-negative")
    if not str(args.device).startswith("cuda"):
        raise SystemExit("the extension runner requires a CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    for name in (
        "data_root",
        "array_root",
        "manifest_root",
        "checkpoint_root",
        "source_pair_root",
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

    for pair_index in args.pair_indices:
        for dataset in args.datasets:
            run_cell_pair(args, CELL_BY_DATASET[dataset], pair_index)


if __name__ == "__main__":
    main()
