"""Generate and evaluate paired degree-preserving edge perturbations.

The fixed panel is PubMed and Roman-empire crossed with GraphSAGE-mean and
ACM-GCN.  Existing ERM checkpoints are evaluated without retraining.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hfd.datasets import load_dataset  # noqa: E402
from hfd.edge_perturbation import generate_rewire_pair  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.metrics import accuracy, macro_f1  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.stats import _sha256  # noqa: E402


DATASETS = ("pubmed", "roman-empire")
MODELS = ("graphsage-mean", "acm-gcn")
DIRECTION = {"pubmed": True, "roman-empire": False}
SPLIT_PREFIX = {"pubmed": "geomgcn", "roman-empire": "platonov"}
SEEDS = (1, 2, 3)
DESIGN_ID = "tail-half-one-degree-preserving-swap-v1"
PAIR_OUTPUT_FILES = (
    "graphs.npz",
    "swaps.json",
    "predictions.npz",
    "checkpoint_metrics.csv",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def lower_process_priority() -> bool:
    """Use below-normal priority on Windows; failure is harmless and recorded."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process()
            if os.name == "nt":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(5)
            return True
        except (OSError, psutil.Error):
            pass
    if os.name != "nt":
        try:
            os.nice(5)
            return True
        except OSError:
            return False
    below_normal_priority_class = 0x00004000
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    return bool(
        ctypes.windll.kernel32.SetPriorityClass(
            handle, below_normal_priority_class
        )
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metrics table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_fixed_inputs(
    dataset: str,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
) -> None:
    recorded = manifest["array_checksums_sha256"]
    if set(arrays.files) != set(recorded):
        raise RuntimeError(f"{dataset}: frozen array keys differ from the manifest")
    for key in sorted(recorded):
        if _sha256(arrays[key]) != recorded[key]:
            raise RuntimeError(f"{dataset}: checksum mismatch for frozen array {key}")
    if not np.array_equal(bundle.y, arrays["labels"]):
        raise RuntimeError(f"{dataset}: live labels differ from frozen labels")
    graph_checksums = manifest.get("graph_checksums_sha256", {})
    if "features" in graph_checksums:
        features = bundle.x.detach().cpu().numpy()
        if _sha256(features) != graph_checksums["features"]:
            raise RuntimeError(f"{dataset}: live features differ from registration")
    if "simple_edge_index" in graph_checksums:
        simple_edges = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
        if _sha256(simple_edges) != graph_checksums["simple_edge_index"]:
            raise RuntimeError(
                f"{dataset}: live simple graph differs from registration"
            )


def prepare_actual_experiment(args) -> None:
    actual_root = args.output_root.resolve() / "actual"
    actual_root.mkdir(parents=True, exist_ok=True)
    manifest_path = actual_root / "experiment_manifest.json"
    expected = {
        "design_id": DESIGN_ID,
        "code_version": args.code_version,
        "sample_fraction": args.sample_fraction,
        "seed_base": args.seed_base,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "split_numbers": list(range(10)),
        "seeds": list(SEEDS),
        "target_selection": "uniform sample without replacement",
        "targets_per_dataset": "floor(0.5 * frozen tail count)",
        "swaps_per_target_per_condition": 1,
        "max_generation_attempts": 1,
    }
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                f"actual experiment settings differ from {manifest_path}"
            )
    else:
        atomic_json(manifest_path, expected)
    args.experiment_manifest_sha256 = file_sha256(manifest_path)

    requested = set(args.pair_indices)
    for dataset in DATASETS:
        dataset_root = actual_root / dataset
        existing = set()
        if dataset_root.is_dir():
            for path in dataset_root.glob("pair-[0-9][0-9][0-9][0-9]"):
                if path.is_dir():
                    existing.add(int(path.name.removeprefix("pair-")))
        combined = existing | requested
        if combined and combined != set(range(1, max(combined) + 1)):
            raise RuntimeError(
                f"{dataset}: actual pair indices must be contiguous from 1; "
                f"existing={sorted(existing)}, requested={sorted(requested)}"
            )


def validate_completed_actual_pair(
    path: Path,
    *,
    dataset: str,
    pair_index: int,
    args,
) -> None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"completed actual pair lacks metadata: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "phase": "actual",
        "completed": True,
        "dataset": dataset,
        "pair_index": pair_index,
        "pair_seed": args.seed_base + pair_index,
        "sample_fraction": 0.5,
        "design_id": DESIGN_ID,
        "code_version": args.code_version,
        "experiment_manifest_sha256": args.experiment_manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: completed actual metadata mismatch {mismatch}")
    hashes = metadata.get("output_sha256", {})
    for filename in PAIR_OUTPUT_FILES:
        output = path / filename
        if not output.is_file() or hashes.get(filename) != file_sha256(output):
            raise RuntimeError(f"{path}: missing or altered output {filename}")


def validate_parent_run(
    run_dir: Path,
    *,
    dataset: str,
    model: str,
    split: str,
    seed: int,
    fixed_input_checksums: dict,
    preparation: str = "erm",
) -> tuple[dict, Path, np.ndarray]:
    if preparation not in {"erm", "last", "head", "full"}:
        raise ValueError(f"unknown preparation {preparation!r}")
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    run_data_path = run_dir / "run_data.npz"
    for path in (config_path, checkpoint_path, run_data_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": run_dir.name,
        "dataset": dataset,
        "model": model,
        "preparation": preparation,
        "split": split,
        "seed": seed,
        "parent_run_id": (
            f"{dataset}_{model}_erm_{split}_seed-{seed}"
            if preparation in {"last", "head"}
            else None
        ),
    }
    mismatch = {
        key: (value, config.get(key))
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{run_dir.name}: parent config mismatch {mismatch}")
    if config.get("fixed_input_checksums") != fixed_input_checksums:
        raise RuntimeError(f"{run_dir.name}: fixed-input checksums differ")
    with np.load(run_data_path, allow_pickle=False) as saved:
        if "predictions" not in saved.files:
            raise RuntimeError(f"{run_data_path}: predictions are missing")
        predictions = saved["predictions"].astype(np.int64, copy=True)
    return config, checkpoint_path, predictions


def predict(
    model,
    x: torch.Tensor,
    edge_index: np.ndarray,
    device: torch.device,
    pause_seconds: float,
) -> np.ndarray:
    edge_tensor = torch.from_numpy(edge_index).to(device=device, dtype=torch.long)
    clear_cache = getattr(model.encoder, "clear_graph_cache", None)
    if clear_cache is None:
        raise RuntimeError(
            f"{type(model.encoder).__name__} does not expose clear_graph_cache()"
        )
    clear_cache()
    with torch.inference_mode():
        predictions = model(x, edge_tensor).argmax(dim=-1).cpu().numpy()
    clear_cache()
    del edge_tensor
    if pause_seconds:
        time.sleep(pause_seconds)
    return predictions.astype(np.int64, copy=False)


def subset_metrics(
    labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    if indices.size == 0:
        raise RuntimeError("the evaluated subset is empty")
    true = labels[indices]
    predicted = predictions[indices]
    label_set = np.unique(true)
    return macro_f1(true, predicted, label_set), accuracy(true, predicted)


def prediction_key(model: str, split: str, seed: int, graph: str) -> str:
    safe_model = model.replace("-", "_")
    safe_split = split.replace("-", "_")
    return f"{safe_model}__{safe_split}__seed_{seed}__{graph}"


def evaluate_pair(
    *,
    dataset: str,
    pair,
    bundle,
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
    models: list[str],
    checkpoint_root: Path,
    device: torch.device,
    pause_seconds: float,
    pair_index: int,
    split_numbers: list[int],
    seeds: list[int],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    x = bundle.x.to(device)
    original_edges = np.asarray(
        to_simple_undirected(bundle.edge_index, bundle.num_nodes), dtype=np.int64
    )
    target_mask = np.zeros(bundle.num_nodes, dtype=bool)
    target_mask[pair.targets] = True
    rows: list[dict] = []
    saved_predictions: dict[str, np.ndarray] = {}
    split_prefix = SPLIT_PREFIX[dataset]
    checkpoints_per_model = len(split_numbers) * len(seeds)

    for model_index, model_name in enumerate(models, start=1):
        for split_position, split_number in enumerate(split_numbers):
            split = f"{split_prefix}-{split_number}"
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
            for seed_position, seed in enumerate(seeds, start=1):
                run_id = (
                    f"{dataset}_{model_name}_erm_{split}_seed-{seed}"
                )
                run_dir = checkpoint_root / run_id
                config, checkpoint_path, saved_original = validate_parent_run(
                    run_dir,
                    dataset=dataset,
                    model=model_name,
                    split=split,
                    seed=seed,
                    fixed_input_checksums=manifest["array_checksums_sha256"],
                )
                model = build_model(
                    model_name,
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
                    differences = int(np.count_nonzero(original_pred != saved_original))
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
                base_key = prediction_key(model_name, split, seed, "original")
                saved_predictions[base_key] = original_pred
                saved_predictions[
                    prediction_key(model_name, split, seed, "treatment")
                ] = treatment_pred
                saved_predictions[
                    prediction_key(model_name, split, seed, "control")
                ] = control_pred
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "pair_index": pair_index,
                        "pair_seed": pair.pair_seed,
                        "split": split,
                        "seed": seed,
                        "n_sampled_test": int(sampled_test_idx.size),
                        "sampled_test_class_composition": class_composition,
                        "original_macro_f1": original_f1,
                        "treatment_macro_f1": treatment_f1,
                        "control_macro_f1": control_f1,
                        "treatment_minus_control_macro_f1": treatment_f1
                        - control_f1,
                        "treatment_minus_original_macro_f1": treatment_f1
                        - original_f1,
                        "control_minus_original_macro_f1": control_f1
                        - original_f1,
                        "original_accuracy": original_acc,
                        "treatment_accuracy": treatment_acc,
                        "control_accuracy": control_acc,
                        "treatment_minus_control_accuracy": treatment_acc
                        - control_acc,
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
                completed = (
                    checkpoints_per_model * (model_index - 1)
                    + len(seeds) * split_position
                    + seed_position
                )
                print(
                    f"{dataset} pair {pair_index}: "
                    f"{completed}/{checkpoints_per_model * len(models)} "
                    f"checkpoints; {model_name} {split} seed {seed}",
                    flush=True,
                )
                del model, state
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    del x
    return rows, saved_predictions


def run_dataset_pair(args, dataset: str, pair_index: int) -> None:
    phase_root = args.output_root.resolve() / args.phase
    dataset_root = phase_root / dataset
    final_dir = dataset_root / f"pair-{pair_index:04d}"
    incomplete_dir = dataset_root / f"pair-{pair_index:04d}.in-progress"
    if final_dir.exists():
        if args.phase == "actual":
            validate_completed_actual_pair(
                final_dir,
                dataset=dataset,
                pair_index=pair_index,
                args=args,
            )
            print(f"validated existing {final_dir}; skipping", flush=True)
            return
        if not args.replace_debug:
            raise FileExistsError(f"completed pair already exists: {final_dir}")
        shutil.rmtree(final_dir)
    if incomplete_dir.exists():
        if args.phase == "actual" and args.restart_incomplete:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived = dataset_root / (
                f"pair-{pair_index:04d}.incomplete-{timestamp}"
            )
            incomplete_dir.replace(archived)
            print(f"archived incomplete attempt at {archived}", flush=True)
        elif args.phase == "actual" or not args.replace_debug:
            raise FileExistsError(f"incomplete pair already exists: {incomplete_dir}")
        else:
            shutil.rmtree(incomplete_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()

    try:
        bundle = load_dataset(dataset, args.data_root)
        arrays = np.load(args.array_root / f"{dataset}.npz", allow_pickle=False)
        manifest = json.loads(
            (args.manifest_root / f"{dataset}.json").read_text(encoding="utf-8")
        )
        validate_fixed_inputs(dataset, bundle, arrays, manifest)
        pair_seed = args.seed_base + pair_index
        target_count = args.debug_target_count if args.phase == "debug" else None
        started = time.perf_counter()
        pair = generate_rewire_pair(
            bundle.edge_index,
            bundle.y,
            arrays["tail_mask"],
            desired_same=DIRECTION[dataset],
            pair_seed=pair_seed,
            sample_fraction=args.sample_fraction,
            target_count=target_count,
            max_attempts=(
                1 if args.phase == "actual" else args.max_generation_attempts
            ),
        )
        generation_seconds = time.perf_counter() - started

        np.savez_compressed(
            incomplete_dir / "graphs.npz",
            treatment_edge_index=pair.treatment_edge_index,
            control_edge_index=pair.control_edge_index,
            targets=pair.targets,
            target_order=pair.target_order,
        )
        atomic_json(
            incomplete_dir / "swaps.json",
            {
                "treatment": pair.swap_records()["treatment"],
                "control": pair.swap_records()["control"],
            },
        )
        code_version, git_dirty = git_state()
        metadata = {
            "design_id": DESIGN_ID,
            "phase": args.phase,
            "completed": False,
            "dataset": dataset,
            "pair_index": pair_index,
            "pair_seed": pair_seed,
            "sample_fraction": args.sample_fraction,
            "requested_targets": pair.requested_targets,
            "selected_targets": int(pair.targets.size),
            "target_sampling": "uniform sample without replacement",
            "sampled_target_auxiliary_rule": (
                "a sampled target may be the removed neighbour of another "
                "target; its same/different neighbour count is preserved"
            ),
            "treatment": (
                "replace one different-class neighbour with a same-class "
                "neighbour"
                if DIRECTION[dataset]
                else "replace one same-class neighbour with a different-class "
                "neighbour"
            ),
            "control": (
                "replace the same target-edge relation with another neighbour "
                "having that same relation"
            ),
            "generation_attempt": pair.attempt,
            "generation_retry_failures": pair.retry_failures,
            "generation_seconds": generation_seconds,
            "audit": pair.audit,
            "models": list(args.models),
            "checkpoints_per_model": len(args.split_numbers) * len(args.seeds),
            "code_version": code_version,
            "git_dirty": git_dirty,
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
        rows, predictions = evaluate_pair(
            dataset=dataset,
            pair=pair,
            bundle=bundle,
            arrays=arrays,
            manifest=manifest,
            models=list(args.models),
            checkpoint_root=args.checkpoint_root,
            device=torch.device(args.device),
            pause_seconds=args.pause_seconds,
            pair_index=pair_index,
            split_numbers=args.split_numbers,
            seeds=args.seeds,
        )
        write_csv(incomplete_dir / "checkpoint_metrics.csv", rows)
        np.savez_compressed(incomplete_dir / "predictions.npz", **predictions)
        output_hashes = {
            filename: file_sha256(incomplete_dir / filename)
            for filename in PAIR_OUTPUT_FILES
        }
        metadata.update(
            {
                "completed": True,
                "completed_at": now_utc(),
                "evaluation_seconds": time.perf_counter() - evaluation_started,
                "n_checkpoint_rows": len(rows),
                "all_original_predictions_reproduced": True,
                "output_sha256": output_hashes,
            }
        )
        atomic_json(incomplete_dir / "metadata.json", metadata)
        incomplete_dir.replace(final_dir)
        print(f"completed {final_dir}", flush=True)
        arrays.close()
    except Exception:
        print(f"incomplete output retained at {incomplete_dir}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("debug", "actual"), required=True)
    parser.add_argument("--pair-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--datasets", choices=DATASETS, nargs="+", default=list(DATASETS)
    )
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument(
        "--split-numbers", type=int, nargs="+", default=list(range(10))
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "edge_perturbation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=2026073000)
    parser.add_argument("--sample-fraction", type=float, default=0.5)
    parser.add_argument("--debug-target-count", type=int, default=10)
    parser.add_argument("--max-generation-attempts", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    parser.add_argument("--pause-seconds", type=float, default=0.1)
    parser.add_argument("--replace-debug", action="store_true")
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="archive and restart a retained incomplete actual attempt",
    )
    args = parser.parse_args()

    if any(index < 1 for index in args.pair_indices):
        parser.error("pair indices must be positive")
    if args.phase == "actual" and args.models != list(MODELS):
        parser.error("actual pairs require both registered models")
    if args.phase == "actual" and args.datasets != list(DATASETS):
        parser.error("actual collection requires both registered datasets")
    if args.phase == "actual" and sorted(set(args.split_numbers)) != list(range(10)):
        parser.error("actual collection requires split numbers 0 through 9")
    if args.phase == "actual" and sorted(set(args.seeds)) != list(SEEDS):
        parser.error("actual collection requires seeds 1, 2, and 3")
    if args.phase == "actual" and args.debug_target_count != 10:
        parser.error("--debug-target-count applies only to debug runs")
    if args.phase == "actual" and args.sample_fraction != 0.5:
        parser.error("actual collection fixes --sample-fraction at 0.5")
    if any(split < 0 or split > 9 for split in args.split_numbers):
        parser.error("split numbers must be between 0 and 9")
    if any(seed not in SEEDS for seed in args.seeds):
        parser.error("seeds must be drawn from 1, 2, and 3")
    args.split_numbers = sorted(set(args.split_numbers))
    args.seeds = sorted(set(args.seeds))
    if not str(args.device).startswith("cuda"):
        parser.error("this experiment runner requires an explicit CUDA device")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    for name, path in (
        ("data root", args.data_root),
        ("array root", args.array_root),
        ("manifest root", args.manifest_root),
        ("checkpoint root", args.checkpoint_root),
    ):
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            parser.error(f"{name} does not exist: {resolved}")
        setattr(args, name.replace(" ", "_"), resolved)

    args.code_version, dirty = git_state()
    if args.phase == "actual" and dirty:
        parser.error(
            "actual collection requires a clean committed worktree"
        )
    if args.phase == "actual":
        prepare_actual_experiment(args)
    else:
        args.experiment_manifest_sha256 = None
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if str(args.device).startswith("cuda"):
        device_index = torch.device(args.device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=device_index
        )
    args.priority_lowered = lower_process_priority()

    for pair_index in sorted(set(args.pair_indices)):
        for dataset in args.datasets:
            run_dataset_pair(args, dataset, pair_index)


if __name__ == "__main__":
    main()
