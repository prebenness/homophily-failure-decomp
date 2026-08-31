"""Run the fixed PubMed DFR frozen-representation pilot.

The DFR mode refits a regularised linear classifier on the canonical
validation split.  The reserved training-split sensitivity is rejected until
the core fitter can select its regularisation using external canonical
validation data.  The implemented mode uses the classifier features
immediately before the ERM model's final affine layer and leaves the canonical
ERM runs unchanged.

The complete panel is:

- models: ACM-GCN, GCN, and GraphSAGE-mean;
- primary DFR mode: validation split;
- pairs: ten Geom-GCN splits by three ERM training seeds.

The panel is 90 independent atomic outputs.  The runner uses one process,
limits CPU numerical libraries to one thread, holds only one model on the
GPU, and can resume after interruption by validating every completed run
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# DFR fitting is deliberately single-threaded so the pilot can run alongside
# interactive work on the same machine.  These must be set before importing
# NumPy, PyTorch, or scikit-learn.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from hfd.datasets import load_dataset  # noqa: E402
from hfd.dfr import DFRConfig, fit_dfr  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.metrics import accuracy, gap_audit_metrics, macro_f1  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.stats import _sha256  # noqa: E402


DATASET = "pubmed"
# Put the shortest model first so an informative complete cell is available
# before the longer Platonov-skeleton cells finish.
MODELS = ["acm-gcn", "gcn", "graphsage-mean"]
MODES = ["val", "train"]
SPLITS = [f"geomgcn-{index}" for index in range(10)]
SEEDS = [1, 2, 3]
C_GRID = (1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01)
PROTOCOL = "dfr_l1_joint_balanced_v1"
FEATURE_CUT = "classifier_features"
REQUIRED_OUTPUTS = {
    "config.json",
    "metrics.json",
    "coefficients.npz",
    "run_data.npz",
}
FLOAT_TOLERANCE = 1e-10


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def filesystem_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(
            to_jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    write_json(temporary, value)
    temporary.replace(path)


def run_id(model: str, mode: str, split: str, seed: int) -> str:
    return f"pubmed_{model}_dfr-{mode}_{split}_seed-{seed}"


def parent_run_id(model: str, split: str, seed: int) -> str:
    return f"pubmed_{model}_erm_{split}_seed-{seed}"


def paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def verify_fixed_inputs(
    arrays: np.lib.npyio.NpzFile,
    manifest: dict,
) -> None:
    recorded = manifest["array_checksums_sha256"]
    for key in ("labels", "merged_bin", "tail_mask", "eligible"):
        actual = _sha256(arrays[key])
        if recorded[key] != actual:
            raise RuntimeError(
                f"PubMed checksum mismatch for frozen array {key!r}"
            )


def verify_dataset_view(bundle, split, manifest: dict) -> None:
    recorded = manifest["array_checksums_sha256"]
    observed = {
        "labels": _sha256(bundle.y),
        f"{split.split_id}.train": _sha256(split.train_idx),
        f"{split.split_id}.val": _sha256(split.val_idx),
        f"{split.split_id}.test": _sha256(split.test_idx),
    }
    for key, checksum in observed.items():
        if recorded[key] != checksum:
            raise RuntimeError(
                f"PubMed loaded array {key!r} differs from the manifest"
            )


def read_parent_provenance(
    parent_root: Path,
    *,
    model: str,
    split: str,
    seed: int,
    model_config: dict,
    fixed_input_checksums: dict,
) -> dict:
    identifier = parent_run_id(model, split, seed)
    run_dir = parent_root / identifier
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Canonical ERM parent is incomplete: {run_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": identifier,
        "dataset": DATASET,
        "model": model,
        "preparation": "erm",
        "split": split,
        "seed": seed,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Canonical ERM parent configuration mismatch: {mismatches}"
        )
    if config.get("model_config") != model_config:
        raise RuntimeError(
            f"{identifier}: model configuration differs from this checkout"
        )
    if config.get("fixed_input_checksums") != fixed_input_checksums:
        raise RuntimeError(
            f"{identifier}: fixed-input checksums differ from this checkout"
        )
    if checkpoint_path.stat().st_size == 0:
        raise RuntimeError(f"{identifier}: checkpoint.pt is empty")
    return {
        "run_id": identifier,
        "run_dir": run_dir,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "training_code_version": config.get("code_version"),
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_classifier_features(
    *,
    model_name: str,
    model_config: dict,
    parent: dict,
    bundle,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    model = build_model(
        model_name,
        int(x.shape[1]),
        bundle.num_classes,
        model_config,
        num_nodes=bundle.num_nodes,
    )
    model.load_state_dict(load_state_dict(parent["checkpoint_path"]))
    model.to(device)
    model.eval()
    with torch.inference_mode():
        z = model.encoder(x, edge_index)
        features = (
            model.head.classifier_features(z)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    seconds = time.perf_counter() - started
    del z
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if features.shape != (bundle.num_nodes, 512):
        raise RuntimeError(
            f"Unexpected classifier-feature shape {features.shape}; "
            f"expected {(bundle.num_nodes, 512)}"
        )
    if not np.isfinite(features).all():
        raise RuntimeError("Classifier features contain a non-finite value")
    return features, seconds


def non_tail_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    test_idx: np.ndarray,
    merged_bin: np.ndarray,
    tail_mask: np.ndarray,
) -> dict:
    idx = test_idx[(merged_bin[test_idx] >= 0) & ~tail_mask[test_idx]]
    present_labels = np.unique(labels[idx])
    return {
        "n_test": int(idx.size),
        "true_labels": [int(label) for label in present_labels],
        "macro_f1": macro_f1(
            labels[idx],
            predictions[idx],
            present_labels,
        ),
        "accuracy": accuracy(labels[idx], predictions[idx]),
    }


def close(first: float, second: float) -> bool:
    return abs(float(first) - float(second)) <= FLOAT_TOLERANCE


def validate_complete_run(
    run_dir: Path,
    *,
    identifier: str,
    model: str,
    mode: str,
    split: str,
    seed: int,
    code_version: str,
    parent: dict,
) -> tuple[bool, str | None]:
    missing = sorted(
        filename
        for filename in REQUIRED_OUTPUTS
        if not (run_dir / filename).is_file()
    )
    if missing:
        return False, f"missing files: {missing}"
    try:
        config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        with np.load(
            run_dir / "coefficients.npz", allow_pickle=False
        ) as coefficients:
            coefficient_keys = set(coefficients.files)
            classes = coefficients["classes"]
            coef = coefficients["coef"]
            intercept = coefficients["intercept"]
            scaler_mean = coefficients["scaler_mean"]
            scaler_scale = coefficients["scaler_scale"]
        with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
            saved_keys = set(saved.files)
            probabilities = saved["probabilities"]
            predictions = saved["predictions"]
            labels = saved["labels"]
            test_idx = saved["test_idx"]
            merged_bin = saved["merged_bin"]
            tail_mask = saved["tail_mask"]
            probability_classes = saved["probability_classes"]
    except Exception as error:
        return False, f"unreadable output: {error}"

    expected_config = {
        "run_id": identifier,
        "dataset": DATASET,
        "model": model,
        "preparation": "dfr",
        "dfr_mode": mode,
        "split": split,
        "seed": seed,
        "parent_run_id": parent["run_id"],
        "code_version": code_version,
        "parent_config_sha256": parent["config_sha256"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        return False, f"config mismatch: {mismatches}"
    objective = metrics.get("training_objective", {})
    if objective.get("name") != "dfr" or objective.get("mode") != mode:
        return False, "metrics do not identify the expected DFR mode"

    required_coefficient_arrays = {
        "classes",
        "coef",
        "intercept",
        "scaler_mean",
        "scaler_scale",
    }
    if not required_coefficient_arrays.issubset(coefficient_keys):
        return (
            False,
            "coefficients.npz lacks "
            f"{sorted(required_coefficient_arrays - coefficient_keys)}",
        )
    required_run_arrays = {
        "probabilities",
        "probability_classes",
        "predictions",
        "labels",
        "train_idx",
        "val_idx",
        "test_idx",
        "merged_bin",
        "tail_mask",
        "eligible",
        "refit_idx",
    }
    if not required_run_arrays.issubset(saved_keys):
        return (
            False,
            f"run_data.npz lacks {sorted(required_run_arrays - saved_keys)}",
        )
    if coef.ndim != 2 or coef.shape[1] != scaler_mean.size:
        return False, "coefficient and scaler dimensions differ"
    if scaler_scale.shape != scaler_mean.shape:
        return False, "scaler mean and scale dimensions differ"
    if intercept.ndim != 1 or classes.ndim != 1:
        return False, "classes or intercept has the wrong rank"
    if not np.array_equal(classes, probability_classes):
        return False, "coefficient and probability class orders differ"
    if probabilities.shape != (labels.size, classes.size):
        return False, "probability matrix has the wrong shape"
    if predictions.shape != labels.shape:
        return False, "prediction array has the wrong shape"
    if not np.isfinite(probabilities).all():
        return False, "probabilities contain a non-finite value"
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=2e-6,
    ):
        return False, "probability rows do not sum to one"

    intervals = [
        row["bin"]
        for row in metrics.get("per_bin", [])
    ]
    if not intervals:
        return False, "metrics contain no per-bin records"
    try:
        recomputed = gap_audit_metrics(
            y_true=labels,
            y_pred=predictions,
            test_idx=test_idx,
            merged_bin=merged_bin,
            tail_mask=tail_mask,
            bin_intervals=intervals,
            standard_metric="accuracy",
        )
    except Exception as error:
        return False, f"metrics cannot be recomputed: {error}"
    for key in (
        "whole_test_standard",
        "whole_test_macro_f1",
        "tail_macro_f1",
        "tail_gap",
    ):
        if not close(metrics[key], recomputed[key]):
            return False, f"stored {key} differs from saved predictions"
    return True, None


def archive_incomplete_run(run_dir: Path, output_root: Path) -> Path:
    if run_dir.parent.resolve() != output_root.resolve():
        raise RuntimeError(f"Refusing to move unexpected path: {run_dir}")
    archived = output_root / (
        f".{run_dir.name}.incomplete-{filesystem_timestamp()}"
    )
    run_dir.replace(archived)
    return archived


def fit_atomic_run(
    *,
    output_root: Path,
    identifier: str,
    model: str,
    mode: str,
    split,
    seed: int,
    code_version: str,
    parent: dict,
    features: np.ndarray,
    feature_seconds: float,
    labels: np.ndarray,
    merged_bin: np.ndarray,
    tail_mask: np.ndarray,
    manifest: dict,
) -> None:
    if mode == "val":
        refit_idx = split.val_idx
        refit_source = "canonical validation split"
        external_tuning_idx = np.empty(0, dtype=np.int64)
        c_selection_source = (
            "stratified tuning half of the canonical validation split"
        )
    elif mode == "train":
        refit_idx = split.train_idx
        refit_source = "canonical training split"
        external_tuning_idx = split.val_idx[
            merged_bin[split.val_idx] >= 0
        ]
        c_selection_source = "canonical validation split"
    else:
        raise ValueError(f"Unknown DFR mode: {mode}")
    eligible = merged_bin[refit_idx] >= 0
    refit_idx = refit_idx[eligible]
    expected_classes = np.arange(
        int(manifest["graph"]["num_classes"]),
        dtype=np.int64,
    )
    expected_bins = np.arange(
        len(manifest["local_homophily"]["merged_bins"]),
        dtype=np.int64,
    )
    dfr_config = DFRConfig(
        c_grid=C_GRID,
        tuning_fraction=0.5,
        n_final_fits=10,
        random_seed=seed,
        max_iter=5000,
    )

    fit_started = time.perf_counter()
    tuning_arguments = {}
    if mode == "train":
        tuning_arguments = {
            "tuning_features": features[external_tuning_idx],
            "tuning_y": labels[external_tuning_idx],
            "tuning_bins": merged_bin[external_tuning_idx],
        }
    with threadpool_limits(limits=1):
        fitted = fit_dfr(
            features[refit_idx],
            labels[refit_idx],
            merged_bin[refit_idx],
            config=dfr_config,
            expected_classes=expected_classes,
            expected_bins=expected_bins,
            **tuning_arguments,
        )
        probabilities = fitted.predict_proba(features)
        predictions = fitted.predict(features)
    fit_seconds = time.perf_counter() - fit_started
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.int64)

    intervals = [
        definition["interval"]
        for definition in manifest["local_homophily"]["merged_bins"]
    ]
    metrics = gap_audit_metrics(
        y_true=labels,
        y_pred=predictions,
        test_idx=split.test_idx,
        merged_bin=merged_bin,
        tail_mask=tail_mask,
        bin_intervals=intervals,
        standard_metric="accuracy",
    )
    non_tail = non_tail_metrics(
        labels,
        predictions,
        split.test_idx,
        merged_bin,
        tail_mask,
    )
    tail_test_idx = split.test_idx[tail_mask[split.test_idx]]
    metrics["whole_test_accuracy"] = metrics["whole_test_standard"]
    metrics["tail_accuracy"] = accuracy(
        labels[tail_test_idx],
        predictions[tail_test_idx],
    )
    metrics["non_tail_macro_f1"] = non_tail["macro_f1"]
    metrics["non_tail_accuracy"] = non_tail["accuracy"]
    metrics["non_tail"] = non_tail
    metrics["training_objective"] = {
        "name": "dfr",
        "protocol": PROTOCOL,
        "mode": mode,
        "refit_source": refit_source,
        "c_selection_source": c_selection_source,
        "groups": "class-by-fixed-local-homophily-bin",
    }
    metrics["timing"] = {
        "feature_extraction_seconds": feature_seconds,
        "dfr_fit_and_full_prediction_seconds": fit_seconds,
    }
    metrics["dfr_diagnostics"] = to_jsonable(fitted.diagnostics)

    config = {
        "schema_version": 1,
        "run_id": identifier,
        "dataset": DATASET,
        "model": model,
        "preparation": "dfr",
        "mode": mode,
        "dfr_mode": mode,
        "split": split.split_id,
        "seed": seed,
        "protocol": PROTOCOL,
        "protocol_id": PROTOCOL,
        "protocol_details": {
            "representation": (
                "frozen ERM classifier features immediately before the "
                "final affine layer"
            ),
            "refit_source": refit_source,
            "standardization": (
                "tuning standardiser fit on the tuning-fit partition; one "
                "final standardiser fit on the complete refit pool and "
                "shared by all final balanced fits"
            ),
            "classifier": "L1-regularised multinomial logistic regression",
            "balancing": "uniform subsampling over class-by-bin cells",
            "internal_tuning_fraction": 0.5 if mode == "val" else None,
            "c_selection_source": c_selection_source,
            "final_balanced_fits": 10,
            "coefficient_aggregation": "arithmetic mean",
            "test_use": "evaluation only",
        },
        "feature_cut": FEATURE_CUT,
        "C_grid": list(C_GRID),
        "n_final_fits": dfr_config.n_final_fits,
        "dfr_config": to_jsonable(asdict(dfr_config)),
        "parent_run_id": parent["run_id"],
        "parent_training_code_version": parent["training_code_version"],
        "parent_config_sha256": parent["config_sha256"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "classifier_features_sha256": _sha256(features),
        "code_version": code_version,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "fixed_input_checksums": manifest["array_checksums_sha256"],
    }

    temporary = output_root / (
        f".{identifier}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        write_json(temporary / "config.json", config)
        write_json(temporary / "metrics.json", metrics)
        np.savez_compressed(
            temporary / "coefficients.npz",
            classes=np.asarray(fitted.classes_, dtype=np.int64),
            coef=np.asarray(fitted.coef_, dtype=np.float64),
            intercept=np.asarray(fitted.intercept_, dtype=np.float64),
            scaler_mean=np.asarray(fitted.scaler_mean_, dtype=np.float64),
            scaler_scale=np.asarray(fitted.scaler_scale_, dtype=np.float64),
        )
        np.savez_compressed(
            temporary / "run_data.npz",
            probabilities=probabilities,
            probability_classes=np.asarray(fitted.classes_, dtype=np.int64),
            predictions=predictions,
            labels=labels,
            train_idx=split.train_idx,
            val_idx=split.val_idx,
            test_idx=split.test_idx,
            merged_bin=merged_bin,
            tail_mask=tail_mask,
            eligible=merged_bin >= 0,
            refit_idx=refit_idx,
            external_tuning_idx=external_tuning_idx,
        )
        complete, reason = validate_complete_run(
            temporary,
            identifier=identifier,
            model=model,
            mode=mode,
            split=split.split_id,
            seed=seed,
            code_version=code_version,
            parent=parent,
        )
        if not complete:
            raise RuntimeError(
                f"Temporary atomic output failed validation: {reason}"
            )
        destination = output_root / identifier
        if destination.exists():
            archive_incomplete_run(destination, output_root)
        temporary.replace(destination)
    except BaseException:
        # Leave the temporary directory intact for diagnosis after an
        # exception or interruption.  It is never treated as a complete run.
        raise


def selected_panel(args) -> list[tuple[str, str, str, int]]:
    return [
        (model, mode, split, seed)
        for model in args.models
        for split in args.splits
        for seed in args.seeds
        for mode in args.modes
    ]


def initialise_status(
    *,
    status_path: Path,
    args,
    code_version: str,
    paths: dict,
    panel: list[tuple[str, str, str, int]],
    completed_runs: list[str],
) -> dict:
    scope = {
        "dataset": DATASET,
        "models": list(args.models),
        "modes": list(args.modes),
        "splits": list(args.splits),
        "seeds": list(args.seeds),
        "expected_runs": len(panel),
    }
    started = now_utc()
    if status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                previous.get("code_version") == code_version
                and previous.get("scope") == scope
                and previous.get("paths") == paths
            ):
                started = previous.get("started_utc", started)
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "schema_version": 1,
        "experiment": "dfr-pubmed-frozen-representation-pilot",
        "code_version": code_version,
        "started_utc": started,
        "updated_utc": now_utc(),
        "completed_utc": None,
        "scope": scope,
        "paths": paths,
        "resource_limits": {
            "processes": 1,
            "cpu_numerical_threads": 1,
            "cuda_memory_fraction": args.cuda_memory_fraction,
        },
        "completed_runs": completed_runs,
        "in_progress_run": None,
        "failed_run": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.35,
        help="maximum fraction available to this process (default: 0.35)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=["val"],
        help=(
            "val is implemented; train is reserved and rejected until the "
            "core fitter supports external validation tuning"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=SEEDS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        parser.error("--cuda-memory-fraction must be in (0, 1]")
    if "train" in args.modes:
        required_tuning_parameters = {
            "tuning_features",
            "tuning_y",
            "tuning_bins",
        }
        fit_parameters = set(inspect.signature(fit_dfr).parameters)
        if not required_tuning_parameters.issubset(fit_parameters):
            parser.error(
                "The installed hfd.dfr API cannot tune the train-mode C on "
                "canonical validation data. Run the primary val mode only."
            )

    data_root = args.data_root.resolve()
    array_root = args.array_root.resolve()
    parent_root = args.parent_run_root.resolve()
    output_root = args.output_root.resolve()
    for name, path in (
        ("data root", data_root),
        ("array root", array_root),
        ("canonical parent run root", parent_root),
    ):
        if not path.is_dir():
            parser.error(f"{name} does not exist: {path}")
    if paths_overlap(parent_root, output_root):
        parser.error(
            "The DFR output root and canonical parent root must be separate "
            "non-overlapping directories"
        )

    panel = selected_panel(args)
    code_version = git_output("rev-parse", "HEAD")
    dirty = git_output("status", "--porcelain", "--untracked-files=all")
    if args.dry_run:
        for model, mode, split, seed in panel:
            print(run_id(model, mode, split, seed))
        print(f"{len(panel)} runs")
        print(f"code version: {code_version}")
        print(f"worktree clean: {not bool(dirty)}")
        return
    if dirty:
        parser.error(
            "The worktree has tracked or untracked changes. Commit the exact "
            "implementation before running."
        )

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
        cuda_device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=cuda_device_index,
        )

    dataset_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            REPO_ROOT
            / dataset_config["manifest_dir"]
            / f"{DATASET}.json"
        ).read_text(encoding="utf-8")
    )
    with np.load(array_root / f"{DATASET}.npz", allow_pickle=False) as arrays:
        verify_fixed_inputs(arrays, manifest)
        merged_bin = arrays["merged_bin"].copy()
        tail_mask = arrays["tail_mask"].copy()
    bundle = load_dataset(DATASET, data_root)
    split_by_id = {split.split_id: split for split in bundle.splits}
    for split_name in args.splits:
        verify_dataset_view(bundle, split_by_id[split_name], manifest)

    simple_edges = to_simple_undirected(
        bundle.edge_index,
        bundle.num_nodes,
    )
    x = bundle.x.to(device)
    edge_index = torch.from_numpy(simple_edges).to(device)

    model_configs = {
        model: yaml.safe_load(
            (
                REPO_ROOT / "configs" / "models" / f"{model}.yaml"
            ).read_text(encoding="utf-8")
        )
        for model in args.models
    }
    parent_cache: dict[tuple[str, str, int], dict] = {}
    for model in args.models:
        for split_name in args.splits:
            for seed in args.seeds:
                key = (model, split_name, seed)
                parent_cache[key] = read_parent_provenance(
                    parent_root,
                    model=model,
                    split=split_name,
                    seed=seed,
                    model_config=model_configs[model],
                    fixed_input_checksums=manifest[
                        "array_checksums_sha256"
                    ],
                )

    output_root.mkdir(parents=True, exist_ok=True)
    completed_runs = []
    for model, mode, split_name, seed in panel:
        identifier = run_id(model, mode, split_name, seed)
        complete, _ = validate_complete_run(
            output_root / identifier,
            identifier=identifier,
            model=model,
            mode=mode,
            split=split_name,
            seed=seed,
            code_version=code_version,
            parent=parent_cache[(model, split_name, seed)],
        )
        if complete:
            completed_runs.append(identifier)

    status_path = output_root / "panel_status.json"
    path_record = {
        "data_root": str(data_root),
        "array_root": str(array_root),
        "canonical_parent_run_root": str(parent_root),
        "dfr_output_root": str(output_root),
    }
    status = initialise_status(
        status_path=status_path,
        args=args,
        code_version=code_version,
        paths=path_record,
        panel=panel,
        completed_runs=completed_runs,
    )
    atomic_json(status_path, status)

    total = len(panel)
    position_by_id = {
        run_id(model, mode, split_name, seed): index
        for index, (model, mode, split_name, seed) in enumerate(
            panel,
            start=1,
        )
    }
    for model in args.models:
        for split_name in args.splits:
            for seed in args.seeds:
                parent = parent_cache[(model, split_name, seed)]
                pending_modes = []
                for mode in args.modes:
                    identifier = run_id(model, mode, split_name, seed)
                    complete, reason = validate_complete_run(
                        output_root / identifier,
                        identifier=identifier,
                        model=model,
                        mode=mode,
                        split=split_name,
                        seed=seed,
                        code_version=code_version,
                        parent=parent,
                    )
                    if complete:
                        print(
                            f"[{position_by_id[identifier]}/{total}] "
                            f"complete: {identifier}",
                            flush=True,
                        )
                    else:
                        pending_modes.append((mode, identifier, reason))
                if not pending_modes:
                    continue

                split = split_by_id[split_name]
                features, feature_seconds = extract_classifier_features(
                    model_name=model,
                    model_config=model_configs[model],
                    parent=parent,
                    bundle=bundle,
                    x=x,
                    edge_index=edge_index,
                    device=device,
                )
                for mode, identifier, reason in pending_modes:
                    destination = output_root / identifier
                    if destination.exists():
                        print(
                            f"[{position_by_id[identifier]}/{total}] "
                            f"replacing incomplete output ({reason}): "
                            f"{identifier}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{position_by_id[identifier]}/{total}] "
                            f"running: {identifier}",
                            flush=True,
                        )
                    status["in_progress_run"] = identifier
                    status["updated_utc"] = now_utc()
                    atomic_json(status_path, status)
                    try:
                        fit_atomic_run(
                            output_root=output_root,
                            identifier=identifier,
                            model=model,
                            mode=mode,
                            split=split,
                            seed=seed,
                            code_version=code_version,
                            parent=parent,
                            features=features,
                            feature_seconds=feature_seconds,
                            labels=bundle.y,
                            merged_bin=merged_bin,
                            tail_mask=tail_mask,
                            manifest=manifest,
                        )
                    except BaseException as error:
                        status["failed_run"] = {
                            "run_id": identifier,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "failed_utc": now_utc(),
                        }
                        status["in_progress_run"] = None
                        status["updated_utc"] = now_utc()
                        atomic_json(status_path, status)
                        raise
                    if identifier not in status["completed_runs"]:
                        status["completed_runs"].append(identifier)
                    status["in_progress_run"] = None
                    status["failed_run"] = None
                    status["updated_utc"] = now_utc()
                    atomic_json(status_path, status)
                del features

    status["completed_utc"] = now_utc()
    status["updated_utc"] = status["completed_utc"]
    status["in_progress_run"] = None
    atomic_json(status_path, status)
    print(
        f"Completed and validated {len(panel)} PubMed DFR runs.",
        flush=True,
    )


if __name__ == "__main__":
    main()
