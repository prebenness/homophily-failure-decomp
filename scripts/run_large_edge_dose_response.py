"""Run the registered large-graph tail edge dose-response experiments.

For each dataset and graph replicate, one random ranking of the frozen tail
nodes defines nested target sets at five percentage-point increments.  Every
non-zero level starts from the registered original graph.  The treatment moves
the selected tail nodes' local homophily towards the graph mean by one
neighbour, while the matched control preserves each selected target's local
homophily.  Both graph constructions preserve every node's degree.

Existing GCN and GraphSAGE-mean ERM checkpoints are evaluated on the complete
fixed tail-test subset.  Parent runs and their saved original predictions are
validated once when a dataset starts.  Dose cells subsequently reuse a
lightweight checkpoint specification and never reopen ``run_data.npz``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    atomic_json,
    file_sha256,
    git_state,
    lower_process_priority,
    predict,
    validate_fixed_inputs,
    validate_parent_run,
)


SCHEMA_VERSION = 1
DESIGN_ID = "large-tail-edge-dose-response-v1"
MODELS = ("gcn", "graphsage-mean")
LOGIT_REPRODUCTION_ATOL = 1e-5
LOGIT_REPRODUCTION_RTOL = 1e-5
PERCENTAGES = tuple(range(0, 101, 5))
NONZERO_PERCENTAGES = PERCENTAGES[1:]
REPLICATE_INDICES = tuple(range(1, 6))
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "edge_perturbation_dose_response_large"
)
BASELINE_OUTPUT_FILES = ("checkpoint_metrics.csv", "predictions.npz")
PAIR_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "checkpoint_metrics.csv",
    "predictions.npz",
)


@dataclass(frozen=True)
class DatasetSpec:
    """The frozen dataset-specific part of the experiment design."""

    dataset: str
    display_name: str
    tail_count: int
    tail_side: str
    desired_same: bool
    split_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    generator: str
    default_seed_base: int

    @property
    def checkpoint_identities(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, seed)
            for split in self.split_ids
            for seed in self.training_seeds
        )

    @property
    def treatment_direction(self) -> str:
        return "increase-local-homophily" if self.desired_same else (
            "decrease-local-homophily"
        )


DATASET_SPECS = {
    "arxiv-year": DatasetSpec(
        dataset="arxiv-year",
        display_name="arxiv-year",
        tail_count=21869,
        tail_side="high-h_v",
        desired_same=False,
        split_ids=tuple(f"lim-{index}" for index in range(5)),
        training_seeds=tuple(range(1, 7)),
        generator="singleton",
        default_seed_base=2026080200,
    ),
    "ogbn-arxiv": DatasetSpec(
        dataset="ogbn-arxiv",
        display_name="ogbn-arxiv",
        tail_count=30364,
        tail_side="low-h_v",
        desired_same=True,
        split_ids=("ogb-time",),
        training_seeds=tuple(range(1, 31)),
        generator="coordinated",
        default_seed_base=2026080300,
    ),
}


@dataclass(frozen=True)
class CheckpointSpec:
    """Validated parent-run information retained between dose cells."""

    model: str
    split: str
    seed: int
    model_config: dict
    checkpoint_path: Path
    run_data_path: Path
    checkpoint_sha256: str
    training_code_version: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def effective_seed_base(args, spec: DatasetSpec) -> int:
    if args.seed_base is None:
        return spec.default_seed_base
    return int(args.seed_base)


def replicate_seed(seed_base: int, replicate_index: int) -> int:
    return int(seed_base) + int(replicate_index)


def tail_ranking(tail_mask: np.ndarray, seed: int) -> np.ndarray:
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


def dataset_phase_root(args, spec: DatasetSpec) -> Path:
    return args.output_root / args.phase / spec.dataset


def pair_output_path(
    root: Path,
    dataset: str,
    replicate_index: int,
    p_percent: int,
) -> Path:
    return (
        root
        / dataset
        / f"replicate-{replicate_index:04d}"
        / f"pct-{p_percent:03d}"
    )


def registered_checkpoint_identities(
    manifest: dict, spec: DatasetSpec
) -> list[tuple[str, int]]:
    split_ids = [str(row["split_id"]) for row in manifest["splits"]]
    seeds = [int(seed) for seed in manifest["training_seeds"]]
    identities = [(split, seed) for split in split_ids for seed in seeds]
    if len(identities) != int(manifest["n_runs"]):
        raise RuntimeError(
            "registered split/seed product differs from the declared run count"
        )
    if identities != list(spec.checkpoint_identities):
        raise RuntimeError(
            f"{spec.dataset}: registered checkpoint identities differ from "
            "the frozen dose-response panel"
        )
    if len(identities) != 30:
        raise RuntimeError(
            f"{spec.dataset}: dose response requires 30 checkpoints; got "
            f"{len(identities)}"
        )
    return identities


def tail_test_indices(
    arrays: np.lib.npyio.NpzFile, split: str
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


def frozen_manifest(
    args,
    spec: DatasetSpec,
    identities: Sequence[tuple[str, int]],
    seed_base: int,
) -> dict:
    direction_word = "same-class" if spec.desired_same else "different-class"
    source_word = "different-class" if spec.desired_same else "same-class"
    if spec.generator == "coordinated":
        treatment_scope = (
            "one designated incident-edge replacement per selected target; "
            "coordinated same-class target pairs are used where required, "
            "and selected tail nodes may additionally serve as "
            "relation-preserving auxiliary endpoints in other swaps"
        )
        control_scope = (
            "one matched designated incident-edge replacement per selected "
            "target; coordinated same-class target pairs are used where "
            "required, and selected tail nodes may additionally serve as "
            "relation-preserving auxiliary endpoints in other swaps"
        )
    else:
        treatment_scope = (
            "one incident-edge replacement per selected target via an "
            "individual degree-preserving double-edge swap"
        )
        control_scope = (
            "one matched incident-edge replacement per selected target via "
            "an individual degree-preserving double-edge swap"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "code_version": args.code_version,
        "dataset": spec.dataset,
        "models": list(MODELS),
        "p_percentages": list(PERCENTAGES),
        "graph_replicates": len(REPLICATE_INDICES),
        "graph_replicate_indices": list(REPLICATE_INDICES),
        "replicate_seed_base": seed_base,
        "replicate_seed_rule": "replicate_seed_base + replicate_index",
        "tail_definition": (
            f"frozen registered {spec.dataset} {spec.tail_side} tail"
        ),
        "tail_side": spec.tail_side,
        "tail_count": spec.tail_count,
        "desired_same": spec.desired_same,
        "treatment_direction": spec.treatment_direction,
        "generator": spec.generator,
        "target_ranking": (
            "one numpy default_rng(replicate_seed) permutation of the frozen "
            "tail-node indices per graph replicate"
        ),
        "target_sets": "nested prefixes of the replicate target ranking",
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "graph_start_at_each_p": (
            f"registered original {spec.dataset} graph"
        ),
        "treatment": (
            f"{treatment_scope}, replacing a {source_word} neighbour with a "
            f"{direction_word} neighbour while preserving every node degree"
        ),
        "control": (
            f"{control_scope}, replacing the target neighbour with another "
            "neighbour of the same class relation while preserving every "
            "node degree"
        ),
        "shared_graph_pairs": "the same pair is evaluated by both models",
        "checkpoint_panel": (
            "registered dataset split IDs crossed with registered training "
            "seeds; 30 ERM checkpoints per model"
        ),
        "checkpoint_identities": [
            {"split": split, "seed": seed} for split, seed in identities
        ],
        "n_checkpoints_per_model": len(identities),
        "metric_population": "fixed-full-tail-test",
        "evaluation_subset": (
            "complete frozen tail-test subset for each checkpoint split"
        ),
        "metric": (
            "macro-F1 over classes present in that split's fixed tail-test "
            "labels, fixed before scoring, with zero_division=0"
        ),
        "baseline": "registered original graph, evaluated once per checkpoint",
        "baseline_prediction_reproduction": (
            "exact agreement with the parent predictions on each checkpoint's "
            "fixed tail-test subset; full-graph differences outside that "
            "subset are accepted only when current and saved full-graph logits "
            f"agree with atol={LOGIT_REPRODUCTION_ATOL:g} and "
            f"rtol={LOGIT_REPRODUCTION_RTOL:g}, and are recorded"
        ),
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


def prepare_actual_experiment(
    args,
    spec: DatasetSpec,
    identities: Sequence[tuple[str, int]],
    seed_base: int,
) -> str:
    actual_root = dataset_phase_root(args, spec)
    actual_root.mkdir(parents=True, exist_ok=True)
    manifest_path = actual_root / "experiment_manifest.json"
    ensure_frozen_manifest(
        manifest_path,
        frozen_manifest(args, spec, identities, seed_base),
    )
    return file_sha256(manifest_path)


def validate_registered_inputs(
    spec: DatasetSpec,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
) -> None:
    validate_fixed_inputs(spec.dataset, bundle, arrays, manifest)
    if manifest["local_homophily"]["tail_side"] != spec.tail_side:
        raise RuntimeError(
            f"{spec.dataset}: registered tail side differs from "
            f"{spec.tail_side}"
        )
    if int(np.count_nonzero(arrays["tail_mask"])) != spec.tail_count:
        raise RuntimeError(
            f"{spec.dataset}: frozen tail count differs from "
            f"{spec.tail_count}"
        )
    live_splits = [split.split_id for split in bundle.splits]
    if live_splits != list(spec.split_ids):
        raise RuntimeError(
            f"{spec.dataset}: live split IDs differ from registration"
        )
    if list(bundle.training_seeds) != list(spec.training_seeds):
        raise RuntimeError(
            f"{spec.dataset}: live training seeds differ from registration"
        )


def prepare_checkpoint_specs(
    *,
    spec: DatasetSpec,
    models: Sequence[str],
    identities: Sequence[tuple[str, int]],
    checkpoint_root: Path,
    fixed_input_checksums: dict,
    num_nodes: int,
) -> tuple[list[CheckpointSpec], dict[tuple[str, str, int], np.ndarray]]:
    """Validate each parent run exactly once and retain lightweight specs."""

    checkpoint_specs: list[CheckpointSpec] = []
    saved_originals: dict[tuple[str, str, int], np.ndarray] = {}
    total = len(models) * len(identities)
    completed = 0
    for model in models:
        for split, seed in identities:
            run_id = f"{spec.dataset}_{model}_erm_{split}_seed-{seed}"
            config, checkpoint_path, saved_original = validate_parent_run(
                checkpoint_root / run_id,
                dataset=spec.dataset,
                model=model,
                split=split,
                seed=seed,
                fixed_input_checksums=fixed_input_checksums,
            )
            if saved_original.shape != (num_nodes,):
                raise RuntimeError(
                    f"{run_id}: saved predictions have shape "
                    f"{saved_original.shape}; expected {(num_nodes,)}"
                )
            checkpoint_specs.append(
                CheckpointSpec(
                    model=model,
                    split=split,
                    seed=seed,
                    model_config=dict(config["model_config"]),
                    checkpoint_path=checkpoint_path,
                    run_data_path=checkpoint_path.parent / "run_data.npz",
                    checkpoint_sha256=file_sha256(checkpoint_path),
                    training_code_version=str(config["code_version"]),
                )
            )
            saved_originals[(model, split, seed)] = saved_original
            completed += 1
            print(
                f"{spec.dataset}: validated parent {completed}/{total}; "
                f"{model} {split} seed {seed}",
                flush=True,
            )
    return checkpoint_specs, saved_originals


def load_network(
    checkpoint: CheckpointSpec,
    *,
    input_dim: int,
    num_classes: int,
    num_nodes: int,
    device: torch.device,
):
    network = build_model(
        checkpoint.model,
        input_dim,
        num_classes,
        checkpoint.model_config,
        num_nodes=num_nodes,
    )
    state = torch.load(
        checkpoint.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    network.load_state_dict(state)
    network.to(device)
    network.eval()
    return network, state


def predict_logits(
    model,
    x: torch.Tensor,
    edge_index: np.ndarray,
    device: torch.device,
    pause_seconds: float,
) -> np.ndarray:
    """Run one graph forward and retain its logits for reproduction checks."""

    edge_tensor = torch.from_numpy(edge_index).to(device=device, dtype=torch.long)
    clear_cache = getattr(model.encoder, "clear_graph_cache", None)
    if clear_cache is None:
        raise RuntimeError(
            f"{type(model.encoder).__name__} does not expose clear_graph_cache()"
        )
    clear_cache()
    with torch.inference_mode():
        logits = model(x, edge_tensor).cpu().numpy()
    clear_cache()
    del edge_tensor
    if pause_seconds:
        time.sleep(pause_seconds)
    return np.asarray(logits)


def evaluate_baseline(
    *,
    spec: DatasetSpec,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    checkpoint_specs: Sequence[CheckpointSpec],
    saved_originals: dict[tuple[str, str, int], np.ndarray],
    x: torch.Tensor,
    original_edges: np.ndarray,
    device: torch.device,
    pause_seconds: float,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    total = len(checkpoint_specs)
    for completed, checkpoint in enumerate(checkpoint_specs, start=1):
        network, state = load_network(
            checkpoint,
            input_dim=int(bundle.x.shape[1]),
            num_classes=bundle.num_classes,
            num_nodes=bundle.num_nodes,
            device=device,
        )
        current_logits = predict_logits(
            network, x, original_edges, device, pause_seconds
        )
        original_pred = current_logits.argmax(axis=-1).astype(
            np.int64, copy=False
        )
        saved_original = saved_originals[
            (checkpoint.model, checkpoint.split, checkpoint.seed)
        ]
        indices = tail_test_indices(arrays, checkpoint.split)
        difference_mask = original_pred != saved_original
        difference_nodes = np.flatnonzero(difference_mask).astype(np.int64)
        full_graph_differences = int(np.count_nonzero(difference_mask))
        tail_test_differences = int(
            np.count_nonzero(difference_mask[indices])
        )
        if tail_test_differences:
            raise RuntimeError(
                f"{spec.dataset}/{checkpoint.model}/{checkpoint.split}/"
                f"seed-{checkpoint.seed}: fixed tail-test predictions fail "
                f"reproduction ({tail_test_differences} tail-test nodes; "
                f"{full_graph_differences} full-graph nodes differ)"
            )
        logit_reproduction_checked = False
        max_abs_logit_difference = 0.0
        mean_abs_logit_difference = 0.0
        if full_graph_differences:
            with np.load(checkpoint.run_data_path, allow_pickle=False) as saved:
                if "logits" not in saved.files:
                    raise RuntimeError(
                        f"{checkpoint.run_data_path}: saved logits are missing"
                    )
                saved_logits = saved["logits"].astype(
                    current_logits.dtype, copy=True
                )
            if saved_logits.shape != current_logits.shape:
                raise RuntimeError(
                    f"{checkpoint.run_data_path}: saved logits have shape "
                    f"{saved_logits.shape}; expected {current_logits.shape}"
                )
            if not np.isfinite(saved_logits).all() or not np.isfinite(
                current_logits
            ).all():
                raise RuntimeError("baseline logits contain non-finite values")
            if not np.array_equal(
                saved_logits.argmax(axis=-1), saved_original
            ):
                raise RuntimeError(
                    f"{checkpoint.run_data_path}: saved predictions differ "
                    "from saved-logit argmax"
                )
            absolute_logit_difference = np.abs(current_logits - saved_logits)
            max_abs_logit_difference = float(absolute_logit_difference.max())
            mean_abs_logit_difference = float(absolute_logit_difference.mean())
            if not np.allclose(
                current_logits,
                saved_logits,
                atol=LOGIT_REPRODUCTION_ATOL,
                rtol=LOGIT_REPRODUCTION_RTOL,
            ):
                raise RuntimeError(
                    f"{spec.dataset}/{checkpoint.model}/{checkpoint.split}/"
                    f"seed-{checkpoint.seed}: current logits do not "
                    "numerically reproduce saved logits "
                    f"(max absolute difference "
                    f"{max_abs_logit_difference:.9g})"
                )
            logit_reproduction_checked = True
            print(
                f"{spec.dataset}/{checkpoint.model}/{checkpoint.split}/"
                f"seed-{checkpoint.seed}: accepted {full_graph_differences} "
                "full-graph prediction differences outside the fixed "
                "tail-test subset",
                flush=True,
            )
        score, acc = subset_metrics(bundle.y, original_pred, indices)
        key = prediction_key(
            checkpoint.model,
            checkpoint.split,
            checkpoint.seed,
            "original",
        )
        predictions[key] = original_pred
        rows.append(
            {
                "dataset": spec.dataset,
                "model": checkpoint.model,
                "split": checkpoint.split,
                "seed": checkpoint.seed,
                "n_tail_test": int(indices.size),
                "tail_test_class_composition": class_composition(
                    bundle.y, indices
                ),
                "tail_test_macro_f1": score,
                "tail_test_accuracy": acc,
                "n_full_graph_prediction_differences_from_parent": (
                    full_graph_differences
                ),
                "n_tail_test_prediction_differences_from_parent": (
                    tail_test_differences
                ),
                "full_graph_prediction_difference_nodes_from_parent": (
                    json.dumps(difference_nodes.tolist(), separators=(",", ":"))
                ),
                "full_graph_logit_reproduction_checked": (
                    logit_reproduction_checked
                ),
                "max_abs_logit_difference_from_parent": (
                    max_abs_logit_difference
                ),
                "mean_abs_logit_difference_from_parent": (
                    mean_abs_logit_difference
                ),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "checkpoint_training_code_version": (
                    checkpoint.training_code_version
                ),
                "prediction_key": key,
            }
        )
        print(
            f"{spec.dataset} baseline: {completed}/{total}; "
            f"{checkpoint.model} {checkpoint.split} seed {checkpoint.seed}",
            flush=True,
        )
        del network, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, predictions


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


def validate_completed_baseline(
    path: Path,
    *,
    args,
    spec: DatasetSpec,
    experiment_manifest_sha256: str,
) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"completed baseline lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": spec.dataset,
        "models": list(MODELS),
        "code_version": args.code_version,
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "all_original_tail_test_predictions_reproduced": True,
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


def run_baseline(
    args,
    *,
    spec: DatasetSpec,
    identities: Sequence[tuple[str, int]],
    rows: list[dict],
    predictions: dict[str, np.ndarray],
    evaluation_seconds: float,
    parent_validation_seconds: float,
    experiment_manifest_sha256: str | None,
) -> None:
    root = dataset_phase_root(args, spec)
    final_dir = root / "baseline"
    incomplete_dir = root / "baseline.in-progress"
    if final_dir.exists():
        if args.phase == "actual":
            assert experiment_manifest_sha256 is not None
            validate_completed_baseline(
                final_dir,
                args=args,
                spec=spec,
                experiment_manifest_sha256=experiment_manifest_sha256,
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
    root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": args.phase,
        "completed": False,
        "dataset": spec.dataset,
        "models": list(args.models),
        "checkpoint_identities": [
            {"split": split, "seed": seed} for split, seed in identities
        ],
        "n_checkpoints_per_model": len(identities),
        "metric_population": "fixed-full-tail-test",
        "evaluation_subset": "complete fixed tail-test subset",
        "code_version": args.code_version,
        "git_dirty": args.git_dirty,
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "started_at": now_utc(),
        "resource_limits": resource_record(args),
    }
    atomic_json(incomplete_dir / "metadata.json", metadata)
    try:
        write_csv(incomplete_dir / "checkpoint_metrics.csv", rows)
        np.savez_compressed(
            incomplete_dir / "predictions.npz", **predictions
        )
        output_hashes = {
            filename: file_sha256(incomplete_dir / filename)
            for filename in BASELINE_OUTPUT_FILES
        }
        full_graph_differences = [
            int(row["n_full_graph_prediction_differences_from_parent"])
            for row in rows
        ]
        tail_test_differences = [
            int(row["n_tail_test_prediction_differences_from_parent"])
            for row in rows
        ]
        if any(tail_test_differences):
            raise RuntimeError(
                "baseline contains an unreproduced fixed tail-test prediction"
            )
        metadata.update(
            {
                "completed": True,
                "completed_at": now_utc(),
                "parent_validation_seconds": parent_validation_seconds,
                "evaluation_seconds": evaluation_seconds,
                "n_checkpoint_rows": len(rows),
                "all_parent_runs_validated_once": True,
                "all_original_predictions_reproduced": not any(
                    full_graph_differences
                ),
                "all_original_tail_test_predictions_reproduced": True,
                "n_checkpoints_with_full_graph_prediction_differences": sum(
                    count > 0 for count in full_graph_differences
                ),
                "total_full_graph_prediction_differences": sum(
                    full_graph_differences
                ),
                "max_full_graph_prediction_differences_per_checkpoint": max(
                    full_graph_differences, default=0
                ),
                "full_graph_prediction_differences_by_checkpoint": [
                    {
                        "model": row["model"],
                        "split": row["split"],
                        "seed": int(row["seed"]),
                        "count": int(
                            row[
                                "n_full_graph_prediction_differences_from_parent"
                            ]
                        ),
                        "nodes": json.loads(
                            row[
                                "full_graph_prediction_difference_nodes_from_parent"
                            ]
                        ),
                        "logits_allclose": bool(
                            row["full_graph_logit_reproduction_checked"]
                        ),
                        "max_abs_logit_difference": float(
                            row["max_abs_logit_difference_from_parent"]
                        ),
                        "mean_abs_logit_difference": float(
                            row["mean_abs_logit_difference_from_parent"]
                        ),
                        "logit_atol": LOGIT_REPRODUCTION_ATOL,
                        "logit_rtol": LOGIT_REPRODUCTION_RTOL,
                    }
                    for row in rows
                    if int(
                        row[
                            "n_full_graph_prediction_differences_from_parent"
                        ]
                    )
                ],
                "output_sha256": output_hashes,
            }
        )
        atomic_json(incomplete_dir / "metadata.json", metadata)
        incomplete_dir.replace(final_dir)
        print(f"completed {final_dir}", flush=True)
    except Exception:
        print(f"incomplete output retained at {incomplete_dir}", flush=True)
        raise


def generate_pair(
    *,
    spec: DatasetSpec,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    targets: np.ndarray,
    pair_seed: int,
    max_attempts: int,
):
    if spec.generator == "singleton":
        return generate_rewire_pair(
            bundle.edge_index,
            bundle.y,
            arrays["tail_mask"],
            desired_same=spec.desired_same,
            pair_seed=pair_seed,
            target_nodes=targets,
            max_attempts=max_attempts,
        )
    if spec.generator == "coordinated":
        from hfd.coordinated_edge_perturbation import (
            generate_coordinated_rewire_pair,
        )

        return generate_coordinated_rewire_pair(
            bundle.edge_index,
            bundle.y,
            arrays["tail_mask"],
            target_nodes=targets,
            pair_seed=pair_seed,
            max_attempts=max_attempts,
        )
    raise RuntimeError(f"unknown graph generator: {spec.generator}")


def evaluate_pair(
    *,
    spec: DatasetSpec,
    pair,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    checkpoint_specs: Sequence[CheckpointSpec],
    x: torch.Tensor,
    device: torch.device,
    pause_seconds: float,
    replicate_index: int,
    replicate_seed_value: int,
    p_percent: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Evaluate a cell using specs; parent run-data files are not reopened."""

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    total = len(checkpoint_specs)
    for completed, checkpoint in enumerate(checkpoint_specs, start=1):
        network, state = load_network(
            checkpoint,
            input_dim=int(bundle.x.shape[1]),
            num_classes=bundle.num_classes,
            num_nodes=bundle.num_nodes,
            device=device,
        )
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
        indices = tail_test_indices(arrays, checkpoint.split)
        treatment_f1, treatment_acc = subset_metrics(
            bundle.y, treatment_pred, indices
        )
        control_f1, control_acc = subset_metrics(
            bundle.y, control_pred, indices
        )
        prefix = prediction_key(
            checkpoint.model,
            checkpoint.split,
            checkpoint.seed,
            "treatment",
        ).rsplit("__", 1)[0]
        predictions[f"{prefix}__treatment"] = treatment_pred
        predictions[f"{prefix}__control"] = control_pred
        rows.append(
            {
                "dataset": spec.dataset,
                "model": checkpoint.model,
                "replicate_index": replicate_index,
                "replicate_seed": replicate_seed_value,
                "p_percent": p_percent,
                "n_selected_tail_nodes": int(pair.targets.size),
                "split": checkpoint.split,
                "seed": checkpoint.seed,
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
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "checkpoint_training_code_version": (
                    checkpoint.training_code_version
                ),
                "prediction_key_prefix": prefix,
            }
        )
        print(
            f"{spec.dataset} replicate {replicate_index} p={p_percent}: "
            f"{completed}/{total}; {checkpoint.model} {checkpoint.split} "
            f"seed {checkpoint.seed}",
            flush=True,
        )
        del network, state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, predictions


def validate_completed_pair(
    path: Path,
    *,
    args,
    spec: DatasetSpec,
    seed_base: int,
    replicate_index: int,
    p_percent: int,
    experiment_manifest_sha256: str,
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
        "dataset": spec.dataset,
        "models": list(MODELS),
        "generator": spec.generator,
        "replicate_index": replicate_index,
        "replicate_seed": replicate_seed(seed_base, replicate_index),
        "p_percent": p_percent,
        "requested_targets": target_count_for_percentage(
            spec.tail_count, p_percent
        ),
        "code_version": args.code_version,
        "experiment_manifest_sha256": experiment_manifest_sha256,
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
    spec: DatasetSpec,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    identities: Sequence[tuple[str, int]],
    checkpoint_specs: Sequence[CheckpointSpec],
    x: torch.Tensor,
    device: torch.device,
    seed_base: int,
    experiment_manifest_sha256: str | None,
    replicate_index: int,
    p_percent: int,
) -> None:
    root = dataset_phase_root(args, spec)
    replicate_root = root / f"replicate-{replicate_index:04d}"
    final_dir = replicate_root / f"pct-{p_percent:03d}"
    incomplete_dir = replicate_root / f"pct-{p_percent:03d}.in-progress"
    if final_dir.exists():
        if args.phase == "actual":
            assert experiment_manifest_sha256 is not None
            validate_completed_pair(
                final_dir,
                args=args,
                spec=spec,
                seed_base=seed_base,
                replicate_index=replicate_index,
                p_percent=p_percent,
                experiment_manifest_sha256=experiment_manifest_sha256,
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

    seed = replicate_seed(seed_base, replicate_index)
    ranking = tail_ranking(arrays["tail_mask"], seed)
    targets = targets_for_percentage(ranking, p_percent)
    try:
        generation_started = time.perf_counter()
        pair = generate_pair(
            spec=spec,
            bundle=bundle,
            arrays=arrays,
            targets=targets,
            pair_seed=seed,
            max_attempts=(
                1
                if args.phase == "actual"
                else args.max_generation_attempts
            ),
        )
        generation_seconds = time.perf_counter() - generation_started
        if not np.array_equal(np.sort(pair.targets), np.sort(targets)):
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
            "dataset": spec.dataset,
            "models": list(args.models),
            "tail_side": spec.tail_side,
            "desired_same": spec.desired_same,
            "treatment_direction": spec.treatment_direction,
            "generator": spec.generator,
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
            "graph_start": f"registered original {spec.dataset} graph",
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
            "seed_base": seed_base,
            "experiment_manifest_sha256": experiment_manifest_sha256,
            "started_at": now_utc(),
            "resource_limits": resource_record(args),
        }
        atomic_json(incomplete_dir / "metadata.json", metadata)

        evaluation_started = time.perf_counter()
        rows, predictions = evaluate_pair(
            spec=spec,
            pair=pair,
            bundle=bundle,
            arrays=arrays,
            checkpoint_specs=checkpoint_specs,
            x=x,
            device=device,
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
                "evaluation_seconds": time.perf_counter()
                - evaluation_started,
                "n_checkpoint_rows": len(rows),
                "parent_run_data_reopened": False,
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
        "--datasets",
        choices=tuple(DATASET_SPECS),
        nargs="+",
        default=list(DATASET_SPECS),
    )
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
        "--models", choices=MODELS, nargs="+", default=list(MODELS)
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed-base",
        type=int,
        help="override the frozen dataset-specific seed base",
    )
    parser.add_argument("--debug-checkpoints", type=int, default=1)
    parser.add_argument("--max-generation-attempts", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--replace-debug", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    args.datasets = list(dict.fromkeys(args.datasets))
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
    if not args.datasets:
        raise SystemExit("at least one dataset is required")
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


def run_dataset(args, spec: DatasetSpec, device: torch.device) -> None:
    bundle = load_dataset(spec.dataset, args.data_root)
    arrays = np.load(
        args.array_root / f"{spec.dataset}.npz", allow_pickle=False
    )
    try:
        manifest = json.loads(
            (args.manifest_root / f"{spec.dataset}.json").read_text(
                encoding="utf-8"
            )
        )
        validate_registered_inputs(spec, bundle, arrays, manifest)
        all_identities = registered_checkpoint_identities(manifest, spec)
        identities = list(all_identities)
        if args.phase == "debug":
            identities = identities[: args.debug_checkpoints]
        seed_base = effective_seed_base(args, spec)
        experiment_manifest_sha256 = None
        if args.phase == "actual":
            experiment_manifest_sha256 = prepare_actual_experiment(
                args, spec, all_identities, seed_base
            )

        completed_baseline = False
        baseline_dir = dataset_phase_root(args, spec) / "baseline"
        if args.phase == "actual" and baseline_dir.exists():
            assert experiment_manifest_sha256 is not None
            validate_completed_baseline(
                baseline_dir,
                args=args,
                spec=spec,
                experiment_manifest_sha256=experiment_manifest_sha256,
            )
            completed_baseline = True
            print(
                f"validated existing {baseline_dir}; original-graph "
                "forwards will not be repeated",
                flush=True,
            )

        x = bundle.x.to(device)
        validation_started = time.perf_counter()
        checkpoint_specs, saved_originals = prepare_checkpoint_specs(
            spec=spec,
            models=args.models,
            identities=identities,
            checkpoint_root=args.checkpoint_root,
            fixed_input_checksums=manifest["array_checksums_sha256"],
            num_nodes=bundle.num_nodes,
        )
        parent_validation_seconds = time.perf_counter() - validation_started
        if completed_baseline:
            del saved_originals
        else:
            original_edges = np.asarray(
                to_simple_undirected(bundle.edge_index, bundle.num_nodes),
                dtype=np.int64,
            )
            baseline_started = time.perf_counter()
            baseline_rows, baseline_predictions = evaluate_baseline(
                spec=spec,
                bundle=bundle,
                arrays=arrays,
                checkpoint_specs=checkpoint_specs,
                saved_originals=saved_originals,
                x=x,
                original_edges=original_edges,
                device=device,
                pause_seconds=args.pause_seconds,
            )
            baseline_seconds = time.perf_counter() - baseline_started
            del original_edges
            del saved_originals

            if args.phase == "actual" or 0 in args.percentages:
                run_baseline(
                    args,
                    spec=spec,
                    identities=identities,
                    rows=baseline_rows,
                    predictions=baseline_predictions,
                    evaluation_seconds=baseline_seconds,
                    parent_validation_seconds=parent_validation_seconds,
                    experiment_manifest_sha256=experiment_manifest_sha256,
                )
            del baseline_predictions
        for replicate_index in args.replicates:
            for p_percent in args.percentages:
                if p_percent == 0:
                    continue
                run_graph_pair(
                    args,
                    spec=spec,
                    bundle=bundle,
                    arrays=arrays,
                    identities=identities,
                    checkpoint_specs=checkpoint_specs,
                    x=x,
                    device=device,
                    seed_base=seed_base,
                    experiment_manifest_sha256=experiment_manifest_sha256,
                    replicate_index=replicate_index,
                    p_percent=p_percent,
                )
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        arrays.close()


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

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(
        args.cuda_memory_fraction, device=device_index
    )
    args.priority_lowered = lower_process_priority()

    for dataset in args.datasets:
        run_dataset(args, DATASET_SPECS[dataset], device)


if __name__ == "__main__":
    main()
