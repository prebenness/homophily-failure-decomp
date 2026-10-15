"""Validate and summarise the post-registration PubMed DFR sensitivity.

The reported sensitivity panel contains:

* models: GraphSAGE-mean, GCN, and ACM-GCN;
* DFR refit pool: the canonical validation split (``val``);
* ten Geom-GCN splits and three training seeds.

The original training split (``train``) is accepted as an explicitly selected
sensitivity if its regularisation parameter was tuned on canonical validation
data. It is outside the fixed primary panel.

Every valid DFR run is paired with the canonical ERM, A_last, A_head, and
A_full runs having the same model, split, and seed. Scores are recomputed from
saved predictions. The analysis writes:

* ``panel_audit.csv``: status of every selected DFR atomic run;
* ``matched_run_metrics.csv``: DFR and canonical scores for every matched run;
* ``absolute_score_summary.csv``: absolute-score means and bootstrap intervals;
* ``paired_contrasts.csv``: paired DFR-minus-reference contrasts;
* ``recovered_shares.csv``: tail-gain recovery relative to A_full where the
  A_full denominator is meaningful; and
* ``analysis_manifest.json``: exploratory status, input provenance, metric
  definitions, panel completeness, and output checksums.

Schema assumptions
------------------
DFR runs are direct children of ``--dfr-raw-root`` and use the identifier
``pubmed_{model}_dfr-{mode}_{split}_seed-{seed}``. A completed DFR directory
contains ``config.json``, ``metrics.json``, ``coefficients.npz``, and
``run_data.npz``. Its configuration identifies ``run_id``, ``dataset``,
``model``, ``dfr_mode``, ``split``, ``seed``, ``parent_run_id``,
``code_version``, ``protocol_id``, ``C_grid``, ``n_final_fits``, and
``feature_cut``. Its saved run data contain full-length ``labels``,
``predictions``, ``merged_bin``, and ``tail_mask`` arrays plus the canonical
split indices, the eligible mask, ``refit_idx``, and
``external_tuning_idx``. The coefficients archive contains equivalent names
for the fitted standardiser's mean and scale, averaged coefficients and
intercept, and classifier classes. The configuration's ``protocol_id`` must
be ``dfr_l1_joint_balanced_v1``.

Canonical runs are direct children of ``--canonical-raw-root`` and use the
existing ``pubmed_{model}_{preparation}_{split}_seed-{seed}`` identifier.
Their active ``metrics.json`` must agree with true-label-support macro-F1
recomputed from ``run_data.npz``, which must contain the same evaluation
arrays.

The default invocation requires the complete 90-run validation-refit panel.
Interrupted, deliberately restricted, or training-refit sensitivity panels
require ``--allow-partial``. In partial mode, incomplete directories are
reported but excluded. A directory named as completed in
``panel_status.json`` is never silently excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from hfd.metrics import accuracy, macro_f1  # noqa: E402


DATASET = "pubmed"
MODELS = ("graphsage-mean", "gcn", "acm-gcn")
MODES = ("val", "train")
PRIMARY_MODES = ("val",)
SPLITS = tuple(f"geomgcn-{index}" for index in range(10))
SEEDS = (1, 2, 3)
CANONICAL_PREPARATIONS = ("erm", "last", "head", "full")
EXPECTED_PROTOCOL = "dfr_l1_joint_balanced_v1"
CANONICAL_ANALYSIS_VERSION = "true-label-support-v1"
REQUIRED_DFR_FILES = (
    "config.json",
    "metrics.json",
    "coefficients.npz",
    "run_data.npz",
)
REQUIRED_CANONICAL_FILES = ("config.json", "metrics.json", "run_data.npz")
METRICS = (
    "tail_macro_f1",
    "whole_test_macro_f1",
    "non_tail_macro_f1",
    "tail_accuracy",
    "whole_test_accuracy",
    "non_tail_accuracy",
)
PRIMARY_METRIC = "tail_macro_f1"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_728
FULL_GAIN_MINIMUM = 0.03
TOLERANCE = 1e-10
SCRIPT_SHA256 = hashlib.sha256(
    Path(__file__)
    .read_text(encoding="utf-8")
    .replace("\r\n", "\n")
    .encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class RunScore:
    run_id: str
    model: str
    split: str
    seed: int
    method: str
    metrics: dict[str, float]
    n_test: int
    n_tail: int
    n_non_tail: int
    code_version: str
    path: Path


@dataclass(frozen=True)
class MatchedRun:
    model: str
    mode: str
    split: str
    seed: int
    dfr: RunScore
    canonical: dict[str, RunScore]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Cannot serialise an empty CSV")
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, csv_text(rows))


def expected_dfr_id(model: str, mode: str, split: str, seed: int) -> str:
    return f"{DATASET}_{model}_dfr-{mode}_{split}_seed-{seed}"


def expected_canonical_id(
    model: str, preparation: str, split: str, seed: int
) -> str:
    return f"{DATASET}_{model}_{preparation}_{split}_seed-{seed}"


def status_completed_ids(status: dict[str, Any] | None) -> set[str]:
    if status is None:
        return set()
    completed = status.get("completed_runs", [])
    if not isinstance(completed, list):
        raise RuntimeError("panel_status.json: completed_runs must be a list")
    identifiers: set[str] = set()
    for item in completed:
        if isinstance(item, str):
            identifiers.add(item)
        elif isinstance(item, dict):
            identifier = item.get("run_id") or item.get("id")
            if isinstance(identifier, str):
                identifiers.add(identifier)
            else:
                raise RuntimeError(
                    "panel_status.json: completed-run objects need run_id or id"
                )
        else:
            raise RuntimeError(
                "panel_status.json: completed_runs contains an unsupported value"
            )
    return identifiers


def semantic_status_sha256(status: dict[str, Any] | None) -> str | None:
    """Hash stable panel-status fields without paths or execution times."""
    if status is None:
        return None
    payload = {
        "schema_version": status.get("schema_version"),
        "experiment": status.get("experiment"),
        "code_version": status.get("code_version"),
        "scope": status.get("scope"),
        "resource_limits": status.get("resource_limits"),
        "completed_runs": sorted(status_completed_ids(status)),
        "failed_run": status.get("failed_run"),
        "in_progress_run": status.get("in_progress_run"),
    }
    serialised = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def require_config_values(
    config: dict[str, Any],
    expected: dict[str, Any],
    *,
    path: Path,
) -> None:
    mismatches = {
        key: {"expected": expected_value, "observed": config.get(key)}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"{path}: configuration mismatch: {mismatches}")


def load_run_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {"labels", "predictions", "test_idx", "merged_bin", "tail_mask"}
    try:
        with np.load(path, allow_pickle=False) as saved:
            missing = sorted(required - set(saved.files))
            if missing:
                raise RuntimeError(f"{path}: missing arrays {missing}")
            return {key: np.asarray(saved[key]).copy() for key in required}
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read array archive {path}: {error}") from error


def validate_fixed_arrays(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    split: str,
    run_id: str,
) -> None:
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    merged_bin = arrays["merged_bin"]
    tail_mask = arrays["tail_mask"]
    test_idx = arrays["test_idx"]

    if labels.ndim != 1:
        raise RuntimeError(f"{run_id}: labels must be one-dimensional")
    n_nodes = labels.size
    for name in ("predictions", "merged_bin", "tail_mask"):
        if arrays[name].ndim != 1 or arrays[name].size != n_nodes:
            raise RuntimeError(
                f"{run_id}: {name} must be a full-length one-dimensional array"
            )
    if test_idx.ndim != 1 or test_idx.size == 0:
        raise RuntimeError(f"{run_id}: test_idx must be a non-empty vector")
    if not np.issubdtype(test_idx.dtype, np.integer):
        raise RuntimeError(f"{run_id}: test_idx must contain integers")
    if test_idx.min() < 0 or test_idx.max() >= n_nodes:
        raise RuntimeError(f"{run_id}: test_idx contains an invalid node index")
    if np.unique(test_idx).size != test_idx.size:
        raise RuntimeError(f"{run_id}: test_idx contains duplicate nodes")
    if not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"{run_id}: labels must contain integers")
    if not np.issubdtype(predictions.dtype, np.integer):
        raise RuntimeError(f"{run_id}: predictions must contain integers")
    manifest_classes = int(manifest["graph"]["num_classes"])
    if labels.min() < 0 or labels.max() >= manifest_classes:
        raise RuntimeError(f"{run_id}: labels fall outside the registered classes")
    if predictions.min() < 0 or predictions.max() >= manifest_classes:
        raise RuntimeError(
            f"{run_id}: predictions fall outside the registered classes"
        )

    expected_checksums = manifest["array_checksums_sha256"]
    fixed_arrays = {
        "labels": labels,
        "merged_bin": merged_bin,
        "tail_mask": tail_mask,
        f"{split}.test": test_idx,
    }
    for key, array in fixed_arrays.items():
        observed = array_sha256(array)
        if observed != expected_checksums[key]:
            raise RuntimeError(
                f"{run_id}: saved array checksum mismatch for {key}"
            )


def subset_score(
    labels: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    if indices.size == 0:
        raise RuntimeError("Cannot score an empty evaluation subset")
    true_labels = np.unique(labels[indices])
    return (
        macro_f1(labels[indices], predictions[indices], true_labels),
        accuracy(labels[indices], predictions[indices]),
    )


def recompute_scores(
    arrays: dict[str, np.ndarray],
    *,
    run_id: str,
) -> tuple[dict[str, float], tuple[int, int, int]]:
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    test_idx = arrays["test_idx"]
    merged_bin = arrays["merged_bin"]
    tail_mask = arrays["tail_mask"].astype(bool, copy=False)

    tail_idx = test_idx[tail_mask[test_idx]]
    non_tail_idx = test_idx[
        (merged_bin[test_idx] >= 0) & ~tail_mask[test_idx]
    ]
    if tail_idx.size == 0:
        raise RuntimeError(f"{run_id}: fixed test tail is empty")
    if non_tail_idx.size == 0:
        raise RuntimeError(f"{run_id}: fixed eligible non-tail test set is empty")

    whole_f1, whole_acc = subset_score(labels, predictions, test_idx)
    tail_f1, tail_acc = subset_score(labels, predictions, tail_idx)
    non_tail_f1, non_tail_acc = subset_score(labels, predictions, non_tail_idx)
    metrics = {
        "tail_macro_f1": tail_f1,
        "whole_test_macro_f1": whole_f1,
        "non_tail_macro_f1": non_tail_f1,
        "tail_accuracy": tail_acc,
        "whole_test_accuracy": whole_acc,
        "non_tail_accuracy": non_tail_acc,
    }
    for name, value in metrics.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{run_id}: invalid recomputed {name}={value}")
    return metrics, (int(test_idx.size), int(tail_idx.size), int(non_tail_idx.size))


def stored_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("metrics")
    return nested if isinstance(nested, dict) else payload


def audit_stored_scores(
    stored_payload: dict[str, Any],
    recomputed: dict[str, float],
    *,
    run_id: str,
) -> None:
    stored = stored_metric_payload(stored_payload)
    aliases = {
        "whole_test_macro_f1": ("whole_test_macro_f1",),
        "tail_macro_f1": ("tail_macro_f1",),
        "whole_test_accuracy": ("whole_test_accuracy", "whole_test_standard"),
        "tail_accuracy": ("tail_accuracy",),
        "non_tail_macro_f1": ("non_tail_macro_f1",),
        "non_tail_accuracy": ("non_tail_accuracy",),
    }
    required = {"whole_test_macro_f1", "tail_macro_f1", "whole_test_accuracy"}
    for metric, names in aliases.items():
        present_name = next((name for name in names if name in stored), None)
        if present_name is None:
            if metric in required:
                raise RuntimeError(
                    f"{run_id}: stored metrics do not contain {metric}"
                )
            continue
        try:
            observed = float(stored[present_name])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"{run_id}: stored {present_name} is not numeric"
            ) from error
        if abs(observed - recomputed[metric]) > TOLERANCE:
            raise RuntimeError(
                f"{run_id}: stored {present_name}={observed} differs from "
                f"recomputed {metric}={recomputed[metric]}"
            )
    standard_metric = stored.get("standard_metric", "accuracy")
    if standard_metric != "accuracy":
        raise RuntimeError(
            f"{run_id}: expected PubMed standard metric accuracy; "
            f"observed {standard_metric!r}"
        )


def validate_recorded_checksums(
    config: dict[str, Any],
    manifest: dict[str, Any],
    split: str,
    run_id: str,
) -> None:
    recorded = config.get("fixed_input_checksums")
    if recorded is None:
        return
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{run_id}: fixed_input_checksums must be an object")
    expected = manifest["array_checksums_sha256"]
    for key in ("labels", "merged_bin", "tail_mask", f"{split}.test"):
        if recorded.get(key) != expected[key]:
            raise RuntimeError(
                f"{run_id}: recorded fixed-input checksum mismatch for {key}"
            )


def validate_coefficients(
    path: Path,
    *,
    manifest: dict[str, Any],
    run_id: str,
) -> np.ndarray:
    aliases = {
        "mean": ("scaler_mean", "mean", "standardizer_mean"),
        "scale": ("scaler_scale", "scale", "standardizer_scale"),
        "coef": ("coef", "averaged_coef", "coefficients"),
        "intercept": ("intercept", "averaged_intercept"),
        "classes": ("classes", "class_labels"),
    }
    try:
        with np.load(path, allow_pickle=False) as saved:
            resolved: dict[str, np.ndarray] = {}
            for logical_name, candidates in aliases.items():
                name = next(
                    (candidate for candidate in candidates if candidate in saved.files),
                    None,
                )
                if name is None:
                    raise RuntimeError(
                        f"{path}: no array found for {logical_name}; "
                        f"accepted names are {candidates}"
                    )
                resolved[logical_name] = np.asarray(saved[name])
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read coefficients archive {path}: {error}") from error

    mean = np.ravel(resolved["mean"])
    scale = np.ravel(resolved["scale"])
    coefficients = np.asarray(resolved["coef"])
    intercept = np.ravel(resolved["intercept"])
    classes = np.ravel(resolved["classes"])
    expected_classes = np.arange(int(manifest["graph"]["num_classes"]))
    if mean.size == 0 or mean.shape != scale.shape:
        raise RuntimeError(f"{run_id}: invalid standardiser mean/scale shapes")
    if not np.all(np.isfinite(mean)):
        raise RuntimeError(f"{run_id}: standardiser mean contains non-finite values")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError(f"{run_id}: standardiser scale must be finite and positive")
    if coefficients.ndim != 2 or coefficients.shape[1] != mean.size:
        raise RuntimeError(f"{run_id}: coefficient and feature dimensions differ")
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError(f"{run_id}: coefficients contain non-finite values")
    if not np.all(np.isfinite(intercept)):
        raise RuntimeError(f"{run_id}: intercept contains non-finite values")
    if not np.array_equal(classes.astype(np.int64, copy=False), expected_classes):
        raise RuntimeError(f"{run_id}: fitted classifier classes differ from PubMed")
    if coefficients.shape[0] != classes.size or intercept.size != classes.size:
        raise RuntimeError(f"{run_id}: coefficient, intercept, and class shapes differ")
    return classes.astype(np.int64, copy=False)


def validate_dfr_run_data(
    path: Path,
    *,
    base_arrays: dict[str, np.ndarray],
    coefficient_classes: np.ndarray,
    manifest: dict[str, Any],
    mode: str,
    split: str,
    run_id: str,
) -> None:
    required = {
        "probabilities",
        "probability_classes",
        "train_idx",
        "val_idx",
        "eligible",
        "refit_idx",
        "external_tuning_idx",
    }
    try:
        with np.load(path, allow_pickle=False) as saved:
            missing = sorted(required - set(saved.files))
            if missing:
                raise RuntimeError(f"{path}: missing DFR arrays {missing}")
            arrays = {
                key: np.asarray(saved[key]).copy()
                for key in required
            }
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read DFR run data {path}: {error}") from error

    probabilities = arrays["probabilities"]
    probability_classes = np.ravel(arrays["probability_classes"])
    train_idx = np.ravel(arrays["train_idx"])
    val_idx = np.ravel(arrays["val_idx"])
    eligible = np.ravel(arrays["eligible"])
    refit_idx = np.ravel(arrays["refit_idx"])
    external_tuning_idx = np.ravel(arrays["external_tuning_idx"])
    labels = base_arrays["labels"]
    predictions = base_arrays["predictions"]
    test_idx = base_arrays["test_idx"]
    merged_bin = base_arrays["merged_bin"]

    if not np.array_equal(probability_classes, coefficient_classes):
        raise RuntimeError(
            f"{run_id}: coefficient and probability class orders differ"
        )
    if probabilities.shape != (labels.size, coefficient_classes.size):
        raise RuntimeError(f"{run_id}: probability matrix has the wrong shape")
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{run_id}: probabilities contain non-finite values")
    if np.any(probabilities < -2e-7) or np.any(probabilities > 1.0 + 2e-7):
        raise RuntimeError(f"{run_id}: probabilities fall outside [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise RuntimeError(f"{run_id}: probability rows do not sum to one")
    probability_predictions = probability_classes[
        np.argmax(probabilities, axis=1)
    ]
    if not np.array_equal(probability_predictions, predictions):
        raise RuntimeError(
            f"{run_id}: saved predictions differ from probability argmax"
        )

    expected_checksums = manifest["array_checksums_sha256"]
    for key, indices in (
        (f"{split}.train", train_idx),
        (f"{split}.val", val_idx),
    ):
        if not np.issubdtype(indices.dtype, np.integer):
            raise RuntimeError(f"{run_id}: {key} indices are not integers")
        if array_sha256(indices) != expected_checksums[key]:
            raise RuntimeError(f"{run_id}: saved array checksum mismatch for {key}")
    if eligible.shape != merged_bin.shape or not np.array_equal(
        eligible.astype(bool, copy=False), merged_bin >= 0
    ):
        raise RuntimeError(f"{run_id}: saved eligible mask differs from fixed bins")
    if array_sha256(eligible) != expected_checksums["eligible"]:
        raise RuntimeError(f"{run_id}: saved array checksum mismatch for eligible")
    source_idx = val_idx if mode == "val" else train_idx
    expected_refit_idx = source_idx[merged_bin[source_idx] >= 0]
    if not np.array_equal(refit_idx, expected_refit_idx):
        raise RuntimeError(
            f"{run_id}: refit_idx differs from the eligible canonical {mode} split"
        )
    if np.intersect1d(refit_idx, test_idx).size:
        raise RuntimeError(f"{run_id}: refitting data overlap the test split")
    expected_external_tuning_idx = (
        np.empty(0, dtype=np.int64)
        if mode == "val"
        else val_idx[merged_bin[val_idx] >= 0]
    )
    if not np.array_equal(external_tuning_idx, expected_external_tuning_idx):
        raise RuntimeError(
            f"{run_id}: external_tuning_idx differs from the protocol"
        )
    if np.intersect1d(external_tuning_idx, test_idx).size:
        raise RuntimeError(f"{run_id}: tuning data overlap the test split")
    if np.intersect1d(external_tuning_idx, refit_idx).size:
        raise RuntimeError(f"{run_id}: external tuning and refit data overlap")


def load_canonical_run(
    canonical_root: Path,
    manifest: dict[str, Any],
    *,
    model: str,
    preparation: str,
    split: str,
    seed: int,
    input_digest: Any,
) -> RunScore:
    run_id = expected_canonical_id(model, preparation, split, seed)
    run_dir = canonical_root / run_id
    missing = [
        filename
        for filename in REQUIRED_CANONICAL_FILES
        if not (run_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(f"{run_id}: missing canonical files {missing}")

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    data_path = run_dir / "run_data.npz"
    config = read_json(config_path)
    stored_metrics = read_json(metrics_path)
    require_config_values(
        config,
        {
            "run_id": run_id,
            "dataset": DATASET,
            "model": model,
            "preparation": preparation,
            "split": split,
            "seed": seed,
        },
        path=config_path,
    )
    # Fresh training and checkpoint evaluation already use true-label support.
    # Verify the scores below rather than require an archival migration marker.
    validate_recorded_checksums(config, manifest, split, run_id)
    arrays = load_run_arrays(data_path)
    validate_fixed_arrays(arrays, manifest, split, run_id)
    metrics, supports = recompute_scores(arrays, run_id=run_id)
    audit_stored_scores(stored_metrics, metrics, run_id=run_id)

    input_digest.update(run_id.encode("utf-8"))
    input_digest.update(file_sha256(config_path).encode("ascii"))
    input_digest.update(file_sha256(metrics_path).encode("ascii"))
    input_digest.update(array_sha256(arrays["predictions"]).encode("ascii"))
    return RunScore(
        run_id=run_id,
        model=model,
        split=split,
        seed=seed,
        method=preparation,
        metrics=metrics,
        n_test=supports[0],
        n_tail=supports[1],
        n_non_tail=supports[2],
        code_version=str(config["code_version"]),
        path=run_dir,
    )


def load_dfr_run(
    dfr_root: Path,
    manifest: dict[str, Any],
    *,
    model: str,
    mode: str,
    split: str,
    seed: int,
    input_digest: Any,
) -> RunScore:
    run_id = expected_dfr_id(model, mode, split, seed)
    run_dir = dfr_root / run_id
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    coefficients_path = run_dir / "coefficients.npz"
    data_path = run_dir / "run_data.npz"
    config = read_json(config_path)
    stored_metrics = read_json(metrics_path)
    parent_run_id = expected_canonical_id(model, "erm", split, seed)
    require_config_values(
        config,
        {
            "run_id": run_id,
            "dataset": DATASET,
            "model": model,
            "dfr_mode": mode,
            "split": split,
            "seed": seed,
            "parent_run_id": parent_run_id,
            "protocol_id": EXPECTED_PROTOCOL,
        },
        path=config_path,
    )
    for required_key in ("code_version", "C_grid", "n_final_fits", "feature_cut"):
        if required_key not in config:
            raise RuntimeError(f"{run_id}: configuration lacks {required_key}")
    objective = stored_metrics.get("training_objective")
    if not isinstance(objective, dict):
        raise RuntimeError(f"{run_id}: metrics lack training_objective")
    if (
        objective.get("name") != "dfr"
        or objective.get("mode") != mode
        or objective.get("protocol") != EXPECTED_PROTOCOL
    ):
        raise RuntimeError(
            f"{run_id}: metrics do not identify the expected DFR objective"
        )
    if int(config["n_final_fits"]) <= 0:
        raise RuntimeError(f"{run_id}: n_final_fits must be positive")
    c_grid = config["C_grid"]
    if not isinstance(c_grid, list) or not c_grid:
        raise RuntimeError(f"{run_id}: C_grid must be a non-empty list")
    try:
        c_values = np.asarray(c_grid, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{run_id}: C_grid must be numeric") from error
    if not np.all(np.isfinite(c_values)) or np.any(c_values <= 0.0):
        raise RuntimeError(f"{run_id}: C_grid values must be finite and positive")

    validate_recorded_checksums(config, manifest, split, run_id)
    coefficient_classes = validate_coefficients(
        coefficients_path, manifest=manifest, run_id=run_id
    )
    arrays = load_run_arrays(data_path)
    validate_fixed_arrays(arrays, manifest, split, run_id)
    validate_dfr_run_data(
        data_path,
        base_arrays=arrays,
        coefficient_classes=coefficient_classes,
        manifest=manifest,
        mode=mode,
        split=split,
        run_id=run_id,
    )
    metrics, supports = recompute_scores(arrays, run_id=run_id)
    audit_stored_scores(stored_metrics, metrics, run_id=run_id)

    input_digest.update(run_id.encode("utf-8"))
    for path in (config_path, metrics_path, coefficients_path):
        input_digest.update(file_sha256(path).encode("ascii"))
    input_digest.update(array_sha256(arrays["predictions"]).encode("ascii"))
    return RunScore(
        run_id=run_id,
        model=model,
        split=split,
        seed=seed,
        method=f"dfr-{mode}",
        metrics=metrics,
        n_test=supports[0],
        n_tail=supports[1],
        n_non_tail=supports[2],
        code_version=str(config["code_version"]),
        path=run_dir,
    )


def deterministic_seed(base_seed: int, *parts: str) -> int:
    key = ":".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def sample_std(values: np.ndarray) -> float | None:
    return float(values.std(ddof=1)) if values.size >= 2 else None


def percentile_bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None, float | None]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty vector")
    if values.size < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    smaller_tail = min(
        int(np.count_nonzero(means <= 0.0)),
        int(np.count_nonzero(means >= 0.0)),
    )
    p_value = min(1.0, 2.0 * smaller_tail / resamples)
    return float(lower), float(upper), float(p_value)


def recovered_share_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None, int]:
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("Recovered-share vectors must have the same vector shape")
    if numerator.size < 2:
        return None, None, 0
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, numerator.size, size=(resamples, numerator.size)
    )
    numerator_means = numerator[indices].mean(axis=1)
    denominator_means = denominator[indices].mean(axis=1)
    valid = np.abs(denominator_means) > np.finfo(np.float64).eps
    ratios = numerator_means[valid] / denominator_means[valid]
    if ratios.size < max(100, int(0.95 * resamples)):
        return None, None, int(ratios.size)
    lower, upper = np.quantile(ratios, [0.025, 0.975], method="linear")
    return float(lower), float(upper), int(ratios.size)


def selected_panel(
    models: Iterable[str],
    modes: Iterable[str],
    splits: Iterable[str],
    seeds: Iterable[int],
) -> list[tuple[str, str, str, int]]:
    return [
        (model, mode, split, seed)
        for model in models
        for mode in modes
        for split in splits
        for seed in seeds
    ]


def discover_panel(
    dfr_root: Path,
    panel: list[tuple[str, str, str, int]],
    *,
    allow_partial: bool,
    declared_completed: set[str],
) -> tuple[list[tuple[str, str, str, int]], list[dict[str, Any]]]:
    valid: list[tuple[str, str, str, int]] = []
    audit_rows: list[dict[str, Any]] = []
    for model, mode, split, seed in panel:
        run_id = expected_dfr_id(model, mode, split, seed)
        run_dir = dfr_root / run_id
        missing = [
            filename
            for filename in REQUIRED_DFR_FILES
            if not (run_dir / filename).is_file()
        ]
        if not run_dir.is_dir():
            status = "missing"
            reason = "run directory absent"
        elif missing:
            status = "incomplete"
            reason = f"missing files: {missing}"
        else:
            status = "candidate_complete"
            reason = ""
            valid.append((model, mode, split, seed))
        claimed = run_id in declared_completed
        if claimed and status != "candidate_complete":
            raise RuntimeError(
                f"{run_id}: panel_status.json declares completion, but {reason}"
            )
        audit_rows.append(
            {
                "run_id": run_id,
                "model": model,
                "mode": mode,
                "split": split,
                "seed": seed,
                "status": status,
                "reason": reason,
                "declared_completed": claimed,
            }
        )
    if not allow_partial and len(valid) != len(panel):
        counts: dict[str, int] = {}
        for row in audit_rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        raise RuntimeError(
            "DFR panel is incomplete. Re-run with --allow-partial to analyse "
            f"validated completed runs. Status counts: {counts}"
        )
    if not valid:
        raise RuntimeError("No complete DFR atomic run is available to analyse")
    return valid, audit_rows


def matched_run_rows(
    matches: list[MatchedRun],
    *,
    analysis_id: str,
    analysis_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in sorted(
        matches, key=lambda item: (item.model, item.mode, item.split, item.seed)
    ):
        row: dict[str, Any] = {
            "analysis_id": analysis_id,
            "analysis_status": analysis_status,
            "dataset": DATASET,
            "model": match.model,
            "dfr_mode": match.mode,
            "split": match.split,
            "seed": match.seed,
            "dfr_run_id": match.dfr.run_id,
            "n_test": match.dfr.n_test,
            "n_tail": match.dfr.n_tail,
            "n_non_tail": match.dfr.n_non_tail,
        }
        for metric in METRICS:
            row[f"dfr_{metric}"] = match.dfr.metrics[metric]
        for preparation in CANONICAL_PREPARATIONS:
            reference = match.canonical[preparation]
            row[f"{preparation}_run_id"] = reference.run_id
            for metric in METRICS:
                row[f"{preparation}_{metric}"] = reference.metrics[metric]
                row[f"dfr_minus_{preparation}_{metric}"] = (
                    match.dfr.metrics[metric] - reference.metrics[metric]
                )
        rows.append(row)
    return rows


def group_matches(
    matches: list[MatchedRun],
) -> dict[tuple[str, str], list[MatchedRun]]:
    grouped: dict[tuple[str, str], list[MatchedRun]] = {}
    for match in matches:
        grouped.setdefault((match.model, match.mode), []).append(match)
    for key in grouped:
        grouped[key].sort(key=lambda item: (item.split, item.seed))
    return grouped


def absolute_summary_rows(
    grouped: dict[tuple[str, str], list[MatchedRun]],
    *,
    analysis_id: str,
    analysis_status: str,
    bootstrap_resamples: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, mode), cell in sorted(grouped.items()):
        method_scores: dict[str, list[RunScore]] = {
            f"dfr-{mode}": [match.dfr for match in cell]
        }
        for preparation in CANONICAL_PREPARATIONS:
            method_scores[preparation] = [
                match.canonical[preparation] for match in cell
            ]
        for method, scores in method_scores.items():
            for metric in METRICS:
                values = np.asarray(
                    [score.metrics[metric] for score in scores], dtype=np.float64
                )
                lower, upper, _ = percentile_bootstrap_mean(
                    values,
                    seed=deterministic_seed(
                        base_seed, model, mode, method, metric, "absolute"
                    ),
                    resamples=bootstrap_resamples,
                )
                rows.append(
                    {
                        "analysis_id": analysis_id,
                        "analysis_status": analysis_status,
                        "dataset": DATASET,
                        "model": model,
                        "dfr_mode": mode,
                        "method": method,
                        "metric": metric,
                        "n_matched_runs": values.size,
                        "mean": float(values.mean()),
                        "sample_std": sample_std(values),
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                    }
                )
    return rows


def paired_contrast_rows(
    grouped: dict[tuple[str, str], list[MatchedRun]],
    *,
    analysis_id: str,
    analysis_status: str,
    bootstrap_resamples: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, mode), cell in sorted(grouped.items()):
        comparisons: list[tuple[str, str, str, str, np.ndarray]] = []
        for preparation in CANONICAL_PREPARATIONS:
            for metric in METRICS:
                differences = np.asarray(
                    [
                        match.dfr.metrics[metric]
                        - match.canonical[preparation].metrics[metric]
                        for match in cell
                    ],
                    dtype=np.float64,
                )
                comparisons.append(
                    (
                        f"dfr-{mode}_minus_{preparation}",
                        f"dfr-{mode}",
                        preparation,
                        metric,
                        differences,
                    )
                )
        for preparation in ("last", "head", "full"):
            for metric in METRICS:
                differences = np.asarray(
                    [
                        match.canonical[preparation].metrics[metric]
                        - match.canonical["erm"].metrics[metric]
                        for match in cell
                    ],
                    dtype=np.float64,
                )
                comparisons.append(
                    (
                        f"{preparation}_minus_erm",
                        preparation,
                        "erm",
                        metric,
                        differences,
                    )
                )

        for comparison, method, reference, metric, differences in comparisons:
            lower, upper, p_value = percentile_bootstrap_mean(
                differences,
                seed=deterministic_seed(
                    base_seed,
                    model,
                    mode,
                    comparison,
                    metric,
                    "paired-contrast",
                ),
                resamples=bootstrap_resamples,
            )
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "analysis_status": analysis_status,
                    "dataset": DATASET,
                    "model": model,
                    "dfr_mode": mode,
                    "comparison": comparison,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "n_pairs": differences.size,
                    "mean_paired_difference": float(differences.mean()),
                    "sample_std_paired_difference": sample_std(differences),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_two_sided_p_value": p_value,
                    "ci_excludes_zero": (
                        None
                        if lower is None or upper is None
                        else lower > 0.0 or upper < 0.0
                    ),
                }
            )
    return rows


def recovered_share_rows(
    grouped: dict[tuple[str, str], list[MatchedRun]],
    *,
    analysis_id: str,
    analysis_status: str,
    bootstrap_resamples: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, mode), cell in sorted(grouped.items()):
        dfr_gain = np.asarray(
            [
                match.dfr.metrics[PRIMARY_METRIC]
                - match.canonical["erm"].metrics[PRIMARY_METRIC]
                for match in cell
            ],
            dtype=np.float64,
        )
        full_gain = np.asarray(
            [
                match.canonical["full"].metrics[PRIMARY_METRIC]
                - match.canonical["erm"].metrics[PRIMARY_METRIC]
                for match in cell
            ],
            dtype=np.float64,
        )
        erm_tail_gap = np.asarray(
            [
                match.canonical["erm"].metrics["whole_test_macro_f1"]
                - match.canonical["erm"].metrics["tail_macro_f1"]
                for match in cell
            ],
            dtype=np.float64,
        )
        full_lower, full_upper, _ = percentile_bootstrap_mean(
            full_gain,
            seed=deterministic_seed(
                base_seed, model, mode, "full-minus-erm", PRIMARY_METRIC
            ),
            resamples=bootstrap_resamples,
        )
        gap_lower, gap_upper, _ = percentile_bootstrap_mean(
            erm_tail_gap,
            seed=deterministic_seed(
                base_seed, model, mode, "erm-tail-gap", PRIMARY_METRIC
            ),
            resamples=bootstrap_resamples,
        )
        denominator = float(full_gain.mean())
        numerator = float(dfr_gain.mean())
        qualifies = (
            gap_lower is not None
            and float(erm_tail_gap.mean()) > 0.0
            and gap_lower > 0.0
            and full_lower is not None
            and denominator >= FULL_GAIN_MINIMUM
            and full_lower > 0.0
        )
        share = numerator / denominator if qualifies else None
        share_lower: float | None = None
        share_upper: float | None = None
        valid_ratio_resamples = 0
        if qualifies:
            share_lower, share_upper, valid_ratio_resamples = (
                recovered_share_interval(
                    dfr_gain,
                    full_gain,
                    seed=deterministic_seed(
                        base_seed, model, mode, "recovered-share"
                    ),
                    resamples=bootstrap_resamples,
                )
            )
        rows.append(
            {
                "analysis_id": analysis_id,
                "analysis_status": analysis_status,
                "dataset": DATASET,
                "model": model,
                "dfr_mode": mode,
                "metric": PRIMARY_METRIC,
                "n_pairs": len(cell),
                "mean_erm_tail_gap": float(erm_tail_gap.mean()),
                "erm_tail_gap_bootstrap_ci_lower": gap_lower,
                "erm_tail_gap_bootstrap_ci_upper": gap_upper,
                "mean_dfr_gain_vs_erm": numerator,
                "mean_full_gain_vs_erm": denominator,
                "full_gain_bootstrap_ci_lower": full_lower,
                "full_gain_bootstrap_ci_upper": full_upper,
                "minimum_mean_full_gain": FULL_GAIN_MINIMUM,
                "qualifies_for_interpretation": qualifies,
                "recovered_share": share,
                "recovered_share_bootstrap_ci_lower": share_lower,
                "recovered_share_bootstrap_ci_upper": share_upper,
                "valid_ratio_bootstrap_resamples": valid_ratio_resamples,
            }
        )
    return rows


def full_scope_selected(
    models: list[str],
    modes: list[str],
    splits: list[str],
    seeds: list[int],
) -> bool:
    return (
        set(models) == set(MODELS)
        and set(modes) == set(PRIMARY_MODES)
        and set(splits) == set(SPLITS)
        and set(seeds) == set(SEEDS)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfr-raw-root", type=Path, required=True)
    parser.add_argument("--canonical-raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "main"
            / "rq5"
            / "dfr_pubmed_sensitivity"
        ),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(PRIMARY_MODES)
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Explicitly permit missing or interrupted DFR atomic runs. "
            "Validated completed runs are still paired and summarised."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if not full_scope_selected(args.models, args.modes, args.splits, args.seeds):
        if not args.allow_partial:
            parser.error(
                "A restricted sensitivity scope is explicitly partial; "
                "add --allow-partial"
            )

    dfr_root = args.dfr_raw_root.resolve()
    canonical_root = args.canonical_raw_root.resolve()
    output_root = args.output_root.resolve()
    if not dfr_root.is_dir():
        parser.error(f"DFR raw root does not exist: {dfr_root}")
    if not canonical_root.is_dir():
        parser.error(f"Canonical raw root does not exist: {canonical_root}")
    if dfr_root == canonical_root:
        parser.error("DFR and canonical raw roots must be separate")

    manifest_path = REPO_ROOT / "manifests" / "datasets" / "pubmed.json"
    manifest = read_json(manifest_path)
    if manifest.get("dataset") != DATASET or manifest.get("metric") != "accuracy":
        raise RuntimeError("Tracked PubMed manifest has an unexpected identity")

    status_path = dfr_root / "panel_status.json"
    panel_status = read_json(status_path) if status_path.is_file() else None
    declared_completed = status_completed_ids(panel_status)
    panel = selected_panel(args.models, args.modes, args.splits, args.seeds)
    valid_panel, panel_audit = discover_panel(
        dfr_root,
        panel,
        allow_partial=args.allow_partial,
        declared_completed=declared_completed,
    )

    input_digest = hashlib.sha256()
    input_digest.update(SCRIPT_SHA256.encode("ascii"))
    input_digest.update(file_sha256(manifest_path).encode("ascii"))
    status_semantic_sha256 = semantic_status_sha256(panel_status)
    if status_semantic_sha256 is not None:
        input_digest.update(status_semantic_sha256.encode("ascii"))

    canonical_cache: dict[tuple[str, str, str, int], RunScore] = {}
    matches: list[MatchedRun] = []
    validated_ids: set[str] = set()
    for model, mode, split, seed in valid_panel:
        run_id = expected_dfr_id(model, mode, split, seed)
        try:
            dfr = load_dfr_run(
                dfr_root,
                manifest,
                model=model,
                mode=mode,
                split=split,
                seed=seed,
                input_digest=input_digest,
            )
        except Exception as error:
            claimed = run_id in declared_completed
            if claimed or not args.allow_partial:
                raise
            for row in panel_audit:
                if row["run_id"] == run_id:
                    row["status"] = "invalid"
                    row["reason"] = str(error)
                    break
            continue

        # Canonical archive failures are never converted into partial DFR
        # status. The canonical panel is the fixed reference and must remain
        # valid for every otherwise valid DFR result.
        canonical: dict[str, RunScore] = {}
        for preparation in CANONICAL_PREPARATIONS:
            cache_key = (model, preparation, split, seed)
            if cache_key not in canonical_cache:
                canonical_cache[cache_key] = load_canonical_run(
                    canonical_root,
                    manifest,
                    model=model,
                    preparation=preparation,
                    split=split,
                    seed=seed,
                    input_digest=input_digest,
                )
            canonical[preparation] = canonical_cache[cache_key]
        supports = {
            (score.n_test, score.n_tail, score.n_non_tail)
            for score in (dfr, *canonical.values())
        }
        if len(supports) != 1:
            raise RuntimeError(
                f"{run_id}: DFR and canonical evaluation supports differ"
            )
        matches.append(
            MatchedRun(
                model=model,
                mode=mode,
                split=split,
                seed=seed,
                dfr=dfr,
                canonical=canonical,
            )
        )
        validated_ids.add(run_id)

    if not matches:
        raise RuntimeError("No DFR run passed full validation")
    for row in panel_audit:
        if row["run_id"] in validated_ids:
            row["status"] = "validated"
            if (
                declared_completed
                and row["run_id"] not in declared_completed
                and not row["reason"]
            ):
                row["reason"] = "complete output absent from status completed_runs"

    complete = len(matches) == len(panel)
    primary_scope = full_scope_selected(
        args.models, args.modes, args.splits, args.seeds
    )
    if complete and primary_scope:
        analysis_status = "exploratory_complete_primary_panel"
    elif complete:
        analysis_status = "exploratory_complete_restricted_or_sensitivity_panel"
    else:
        analysis_status = "exploratory_partial_panel"
    input_digest.update(str(args.bootstrap_resamples).encode("ascii"))
    input_digest.update(str(args.bootstrap_seed).encode("ascii"))
    analysis_id = input_digest.hexdigest()

    for row in panel_audit:
        row["analysis_id"] = analysis_id
        row["analysis_status"] = analysis_status
    panel_audit = [
        {
            "analysis_id": row.pop("analysis_id"),
            "analysis_status": row.pop("analysis_status"),
            **row,
        }
        for row in panel_audit
    ]

    grouped = group_matches(matches)
    matched_rows = matched_run_rows(
        matches,
        analysis_id=analysis_id,
        analysis_status=analysis_status,
    )
    absolute_rows = absolute_summary_rows(
        grouped,
        analysis_id=analysis_id,
        analysis_status=analysis_status,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )
    contrast_rows = paired_contrast_rows(
        grouped,
        analysis_id=analysis_id,
        analysis_status=analysis_status,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )
    share_rows = recovered_share_rows(
        grouped,
        analysis_id=analysis_id,
        analysis_status=analysis_status,
        bootstrap_resamples=args.bootstrap_resamples,
        base_seed=args.bootstrap_seed,
    )

    output_files = {
        "panel_audit.csv": panel_audit,
        "matched_run_metrics.csv": matched_rows,
        "absolute_score_summary.csv": absolute_rows,
        "paired_contrasts.csv": contrast_rows,
        "recovered_shares.csv": share_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, rows in output_files.items():
        write_csv(output_root / filename, rows)

    output_checksums = {
        filename: file_sha256(output_root / filename)
        for filename in output_files
    }
    manifest_payload = {
        "schema_version": 1,
        "analysis_id": analysis_id,
        "analysis": "post-registration-dfr-pubmed-sensitivity",
        "status": analysis_status,
        "warning": (
            "This is a post-registration sensitivity. Only the complete "
            "validation-refit panel represents the reported analysis. "
            "Partial, restricted, and training-refit summaries are "
            "provisional."
        ),
        "analysis_code": {
            "path": "scripts/analyze_dfr_pubmed.py",
            "sha256": SCRIPT_SHA256,
            "numpy_version": np.__version__,
        },
        "scope": {
            "dataset": DATASET,
            "selected_models": args.models,
            "selected_modes": args.modes,
            "selected_splits": args.splits,
            "selected_seeds": args.seeds,
            "selected_expected_runs": len(panel),
            "validated_dfr_runs": len(matches),
            "complete": complete,
            "primary_scope": primary_scope,
            "allow_partial": args.allow_partial,
            "full_reported_panel_runs": (
                len(MODELS)
                * len(PRIMARY_MODES)
                * len(SPLITS)
                * len(SEEDS)
            ),
            "optional_modes_outside_primary_panel": sorted(
                set(MODES) - set(PRIMARY_MODES)
            ),
        },
        "schema_assumptions": {
            "dfr_run_id": (
                "pubmed_{model}_dfr-{val|train}_{split}_seed-{seed}"
            ),
            "dfr_required_files": list(REQUIRED_DFR_FILES),
            "dfr_config_required_fields": [
                "run_id",
                "dataset",
                "model",
                "dfr_mode",
                "split",
                "seed",
                "parent_run_id",
                "code_version",
                "protocol_id",
                "C_grid",
                "n_final_fits",
                "feature_cut",
            ],
            "dfr_protocol": EXPECTED_PROTOCOL,
            "fixed_primary_mode": list(PRIMARY_MODES),
            "dfr_run_data_required_arrays": [
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
                "external_tuning_idx",
            ],
            "canonical_preparations": list(CANONICAL_PREPARATIONS),
            "canonical_metric_version": CANONICAL_ANALYSIS_VERSION,
        },
        "metrics": {
            "reported": list(METRICS),
            "macro_f1": (
                "classes present in each evaluated subset's true labels, "
                "fixed before predictions are evaluated; zero_division=0"
            ),
            "whole_test": "all fixed test nodes",
            "tail": "fixed minority-pattern tail test nodes",
            "non_tail": (
                "fixed eligible test nodes outside the minority-pattern tail"
            ),
            "standard_metric": "accuracy",
            "primary_recovery_metric": PRIMARY_METRIC,
        },
        "recovered_share": {
            "definition": (
                "mean paired DFR-minus-ERM tail macro-F1 divided by mean "
                "paired A_full-minus-ERM tail macro-F1"
            ),
            "interpretation_rule": (
                "The ERM tail-gap mean is positive with 95% paired-bootstrap "
                "lower bound > 0; A_full mean gain >= 0.03 and its 95% "
                "paired-bootstrap interval has lower bound > 0"
            ),
            "ratio_interval": (
                "paired bootstrap of the ratio of resampled mean gains"
            ),
        },
        "bootstrap": {
            "unit": "complete matched split-seed run",
            "method": "two-sided percentile bootstrap",
            "resamples": args.bootstrap_resamples,
            "base_seed": args.bootstrap_seed,
            "quantile_method": "linear",
            "multiple_comparison_correction": None,
            "intervals_with_fewer_than_two_runs": None,
        },
        "external_inputs": {
            "dfr_raw_store": {
                "argument": "--dfr-raw-root",
                "role": "90-run PubMed DFR raw result store",
            },
            "canonical_raw_store": {
                "argument": "--canonical-raw-root",
                "role": "canonical ERM and recovery raw result store",
            },
            "panel_status": {
                "relative_path": "panel_status.json",
                "present": status_path.is_file(),
                "semantic_sha256": status_semantic_sha256,
            },
        },
        "training_code_versions": {
            "dfr": sorted({match.dfr.code_version for match in matches}),
            "canonical": sorted(
                {
                    score.code_version
                    for match in matches
                    for score in match.canonical.values()
                }
            ),
        },
        "outputs_sha256": output_checksums,
    }
    atomic_write_text(
        output_root / "analysis_manifest.json",
        json.dumps(manifest_payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    print(
        f"Validated {len(matches)}/{len(panel)} selected DFR runs and wrote "
        f"{analysis_status} outputs to {output_root}"
    )


if __name__ == "__main__":
    main()
