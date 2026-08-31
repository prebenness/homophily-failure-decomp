"""Verify a canonical raw-result store against the registered run panel.

The default metadata level checks every registered run without reading the
large compressed arrays. ``--level arrays`` additionally verifies the
node-level arrays against the tracked dataset manifests.

Example:
    python scripts/verify_raw_store.py --raw-root D:\\data\\raw-runs
    python scripts/verify_raw_store.py --raw-root D:\\data\\raw-runs \
        --level arrays --report results/main/raw_store_verification.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hfd.metrics import gap_audit_metrics  # noqa: E402


DEFAULT_CONTRACT = REPO_ROOT / "manifests" / "raw_store_contract.json"
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(RuntimeError):
    """Raised when a tracked contract or panel manifest is invalid."""


@dataclass(frozen=True)
class PanelRun:
    run_order: int
    run_id: str
    dataset: str
    model: str
    preparation: str
    split: str
    seed: int
    parent_run_id: str | None


def _normalise_parent(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def expected_run_id(run: PanelRun) -> str:
    return (
        f"{run.dataset}_{run.model}_{run.preparation}_"
        f"{run.split}_seed-{run.seed}"
    )


def expected_parent_run_id(run: PanelRun) -> str | None:
    if run.preparation not in {"last", "head"}:
        return None
    return f"{run.dataset}_{run.model}_erm_{run.split}_seed-{run.seed}"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"Cannot read JSON {path}: {error}") from error


def load_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ContractError(f"Unsupported raw-store contract: {path}")
    return contract


def load_panel(path: Path) -> list[PanelRun]:
    required = {
        "run_order",
        "run_id",
        "dataset",
        "model",
        "preparation",
        "split",
        "seed",
        "parent_run_id",
    }
    rows: list[PanelRun] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"Cannot read run panel {path}: {error}") from error

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"{path}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number}: expected a JSON object")
        missing = sorted(required - value.keys())
        if missing:
            raise ContractError(
                f"{path}:{line_number}: missing fields {', '.join(missing)}"
            )
        try:
            run = PanelRun(
                run_order=int(value["run_order"]),
                run_id=str(value["run_id"]),
                dataset=str(value["dataset"]),
                model=str(value["model"]),
                preparation=str(value["preparation"]),
                split=str(value["split"]),
                seed=int(value["seed"]),
                parent_run_id=_normalise_parent(value["parent_run_id"]),
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"{path}:{line_number}: invalid panel field: {error}"
            ) from error
        rows.append(run)

    if not rows:
        raise ContractError(f"Run panel is empty: {path}")
    if [run.run_order for run in rows] != list(range(1, len(rows) + 1)):
        raise ContractError("run_order must be consecutive and match JSONL line order")

    run_ids = [run.run_id for run in rows]
    duplicate_ids = sorted(
        run_id for run_id, count in Counter(run_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ContractError(f"Duplicate run_id values: {duplicate_ids[:10]}")

    seen: set[str] = set()
    for run in rows:
        if run.preparation not in {"erm", "last", "head", "full"}:
            raise ContractError(
                f"{run.run_id}: unknown preparation {run.preparation!r}"
            )
        generated_id = expected_run_id(run)
        if run.run_id != generated_id:
            raise ContractError(
                f"{run.run_id}: expected run_id {generated_id!r}"
            )
        generated_parent = expected_parent_run_id(run)
        if run.parent_run_id != generated_parent:
            raise ContractError(
                f"{run.run_id}: expected parent_run_id {generated_parent!r}, "
                f"found {run.parent_run_id!r}"
            )
        if generated_parent is not None and generated_parent not in seen:
            raise ContractError(
                f"{run.run_id}: parent {generated_parent!r} must occur earlier "
                "in dependency-safe panel order"
            )
        seen.add(run.run_id)
    return rows


def load_dataset_manifests(path: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(path.glob("*.json")):
        value = load_json(manifest_path)
        if not isinstance(value, dict) or "dataset" not in value:
            raise ContractError(f"Invalid dataset manifest: {manifest_path}")
        dataset = str(value["dataset"])
        if dataset in manifests:
            raise ContractError(f"Duplicate dataset manifest for {dataset!r}")
        manifests[dataset] = value
    if not manifests:
        raise ContractError(f"No dataset manifests found under {path}")
    return manifests


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_configs(path: Path, models: set[str]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for model in sorted(models):
        config_path = path / f"{model}.yaml"
        if not config_path.is_file():
            raise ContractError(f"Missing model configuration: {config_path}")
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("model") != model:
            raise ContractError(f"Invalid model configuration: {config_path}")
        configs[model] = value
    return configs


def expected_recovery_config(train: dict[str, Any], preparation: str) -> dict[str, Any] | None:
    if preparation == "erm":
        return None
    multiplier = 1 if preparation == "full" else 10
    learning_rate_divisor = 1 if preparation == "full" else 10
    return {
        "optimizer": train["optimizer"],
        "lr": train["lr"] / learning_rate_divisor,
        "weight_decay": train["weight_decay"],
        "robust_step_size": 0.01,
        "max_epochs": train["max_epochs"] * multiplier,
    }


def _json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{path.parent.name}: cannot read {path.name}: {error}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.parent.name}: {path.name} must contain a JSON object")
        return None
    return value


def validate_metadata(
    run: PanelRun,
    run_dir: Path,
    contract: dict[str, Any],
    dataset_manifest: dict[str, Any],
    model_config: dict[str, Any],
    *,
    experiment_manifest_sha256: str,
    require_reference_environment: bool = False,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    required_files = tuple(contract["required_run_files"])
    for name in required_files:
        path = run_dir / name
        if not path.is_file():
            errors.append(f"{run.run_id}: missing required file {name}")
        elif path.stat().st_size == 0:
            errors.append(f"{run.run_id}: required file is empty: {name}")
    if errors:
        return errors, None

    config = _json_object(run_dir / "config.json", errors)
    metrics = _json_object(run_dir / "metrics.json", errors)
    try:
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{run.run_id}: cannot read history.json: {error}")
        history = None
    if history is not None and (not isinstance(history, list) or not history):
        errors.append(f"{run.run_id}: history.json must contain a non-empty list")
    if config is None or metrics is None:
        return errors, None

    missing_config = sorted(
        set(contract["required_config_fields"]) - config.keys()
    )
    if missing_config:
        errors.append(
            f"{run.run_id}: config.json missing fields {', '.join(missing_config)}"
        )
    if require_reference_environment:
        missing_fresh = sorted(
            set(contract.get("fresh_run_additional_config_fields", ()))
            - config.keys()
        )
        if missing_fresh:
            errors.append(
                f"{run.run_id}: fresh config.json missing fields "
                f"{', '.join(missing_fresh)}"
            )
        if config.get("git_dirty") is not False:
            errors.append(
                f"{run.run_id}: fresh run was not recorded from a clean "
                "Git working tree"
            )
        if (
            config.get("experiment_manifest_sha256")
            != experiment_manifest_sha256
        ):
            errors.append(
                f"{run.run_id}: experiment manifest checksum differs from "
                "the registered panel"
            )
        if config.get("graph_checksums") != dataset_manifest.get(
            "graph_checksums_sha256"
        ):
            errors.append(
                f"{run.run_id}: graph checksums differ from the dataset "
                "registration"
            )
    expected_values: dict[str, object] = {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model": run.model,
        "preparation": run.preparation,
        "split": run.split,
        "seed": run.seed,
        "parent_run_id": run.parent_run_id,
    }
    for field, expected in expected_values.items():
        actual = _normalise_parent(config.get(field)) if field == "parent_run_id" else config.get(field)
        if actual != expected:
            errors.append(
                f"{run.run_id}: config.{field}={actual!r}; expected {expected!r}"
            )

    if config.get("model_config") != model_config:
        errors.append(
            f"{run.run_id}: config.model_config differs from "
            f"configs/models/{run.model}.yaml"
        )
    expected_train = model_config.get("train")
    if config.get("train_config") != expected_train:
        errors.append(
            f"{run.run_id}: config.train_config differs from the registered "
            "model training configuration"
        )
    if isinstance(expected_train, dict):
        expected_recovery = expected_recovery_config(
            expected_train,
            run.preparation,
        )
        if config.get("recovery_config") != expected_recovery:
            errors.append(
                f"{run.run_id}: config.recovery_config differs from the "
                "registered preparation rule"
            )

    code_version = config.get("code_version")
    if not isinstance(code_version, str) or HEX_COMMIT.fullmatch(code_version) is None:
        errors.append(
            f"{run.run_id}: config.code_version must be a full lowercase Git commit"
        )

    versions = config.get("versions")
    if not isinstance(versions, dict):
        errors.append(f"{run.run_id}: config.versions must be an object")
    else:
        for package, expected in contract["reference_training_versions"].items():
            actual = versions.get(package)
            if not isinstance(actual, str) or not actual:
                errors.append(
                    f"{run.run_id}: config.versions.{package} must be recorded"
                )
            elif require_reference_environment and actual != expected:
                errors.append(
                    f"{run.run_id}: config.versions.{package}={actual!r}; "
                    f"expected {expected!r}"
                )

    registered_checksums = dataset_manifest.get("array_checksums_sha256")
    recorded_checksums = config.get("fixed_input_checksums")
    if not isinstance(registered_checksums, dict):
        errors.append(
            f"{run.run_id}: dataset manifest has no array_checksums_sha256 object"
        )
    elif not isinstance(recorded_checksums, dict):
        errors.append(f"{run.run_id}: config.fixed_input_checksums must be an object")
    else:
        for key, expected in registered_checksums.items():
            actual = recorded_checksums.get(key)
            if actual != expected:
                errors.append(
                    f"{run.run_id}: fixed-input checksum {key!r} differs "
                    "from the registered dataset manifest"
                )

    split_ids = {
        str(value.get("split_id"))
        for value in dataset_manifest.get("splits", [])
        if isinstance(value, dict)
    }
    if run.split not in split_ids:
        errors.append(
            f"{run.run_id}: split {run.split!r} is absent from the dataset manifest"
        )
    if run.seed not in dataset_manifest.get("training_seeds", []):
        errors.append(
            f"{run.run_id}: seed {run.seed} is absent from the dataset manifest"
        )

    missing_metrics = sorted(
        set(contract["required_metric_fields"]) - metrics.keys()
    )
    if missing_metrics:
        errors.append(
            f"{run.run_id}: metrics.json missing fields {', '.join(missing_metrics)}"
        )
    checkpoint = metrics.get("checkpoint")
    if not isinstance(checkpoint, dict):
        errors.append(f"{run.run_id}: metrics.checkpoint must be an object")
    elif _normalise_parent(checkpoint.get("parent_run_id")) != run.parent_run_id:
        errors.append(
            f"{run.run_id}: metrics.checkpoint.parent_run_id does not match panel"
        )
    elif history is not None:
        epochs_run = checkpoint.get("epochs_run")
        if epochs_run != len(history):
            errors.append(
                f"{run.run_id}: metrics.checkpoint.epochs_run={epochs_run!r}; "
                f"history contains {len(history)} rows"
            )
        history_epochs = {
            row.get("epoch")
            for row in history
            if isinstance(row, dict)
        }
        if checkpoint.get("best_epoch") not in history_epochs:
            errors.append(
                f"{run.run_id}: metrics.checkpoint.best_epoch is absent "
                "from history.json"
            )
        train_seconds = checkpoint.get("train_seconds")
        if (
            not isinstance(train_seconds, (int, float))
            or isinstance(train_seconds, bool)
            or not math.isfinite(float(train_seconds))
            or float(train_seconds) <= 0.0
        ):
            errors.append(
                f"{run.run_id}: metrics.checkpoint.train_seconds must be "
                "finite and positive"
            )

    checkpoint_path = run_dir / "checkpoint.pt"
    if not zipfile.is_zipfile(checkpoint_path):
        errors.append(
            f"{run.run_id}: checkpoint.pt is not a readable PyTorch ZIP archive"
        )

    parent_hash = config.get("parent_checkpoint_sha256")
    if run.parent_run_id is None:
        if parent_hash not in (None, ""):
            errors.append(
                f"{run.run_id}: parent checkpoint hash recorded for a "
                "parentless preparation"
            )
    elif parent_hash is not None:
        parent_path = run_dir.parent / run.parent_run_id / "checkpoint.pt"
        if not isinstance(parent_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            parent_hash,
        ):
            errors.append(
                f"{run.run_id}: parent_checkpoint_sha256 is not a SHA256 value"
            )
        elif parent_path.is_file() and file_sha256(parent_path) != parent_hash:
            errors.append(
                f"{run.run_id}: parent checkpoint hash differs from "
                f"{run.parent_run_id}"
            )
    provenance = {
        "versions": {
            package: versions.get(package) if isinstance(versions, dict) else None
            for package in contract["reference_training_versions"]
        },
        "code_version": config.get("code_version"),
        "device": config.get("device"),
        "parent_checkpoint_hash_recorded": bool(parent_hash),
    }
    return errors, provenance


def validate_arrays(
    run: PanelRun,
    run_dir: Path,
    contract: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    path = run_dir / "run_data.npz"
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return [f"{run.run_id}: cannot read run_data.npz: {error}"]

    with archive:
        required = set(contract["required_run_data_arrays"])
        missing = sorted(required - set(archive.files))
        if missing:
            return [
                f"{run.run_id}: run_data.npz missing arrays {', '.join(missing)}"
            ]
        arrays = {name: archive[name] for name in required}

    graph = dataset_manifest.get("graph", {})
    n_nodes = int(graph.get("num_nodes", -1))
    n_classes = int(graph.get("num_classes", -1))
    vector_names = ("predictions", "labels", "merged_bin", "tail_mask")
    for name in vector_names:
        if arrays[name].shape != (n_nodes,):
            errors.append(
                f"{run.run_id}: {name} shape {arrays[name].shape}; "
                f"expected {(n_nodes,)}"
            )
    if arrays["logits"].shape != (n_nodes, n_classes):
        errors.append(
            f"{run.run_id}: logits shape {arrays['logits'].shape}; "
            f"expected {(n_nodes, n_classes)}"
        )
    for name in ("train_idx", "val_idx", "test_idx"):
        value = arrays[name]
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            errors.append(f"{run.run_id}: {name} must be a one-dimensional integer array")
        elif value.size and (int(value.min()) < 0 or int(value.max()) >= n_nodes):
            errors.append(f"{run.run_id}: {name} contains an out-of-range node index")
    if not np.issubdtype(arrays["labels"].dtype, np.integer):
        errors.append(f"{run.run_id}: labels must have an integer dtype")
    if not np.issubdtype(arrays["predictions"].dtype, np.integer):
        errors.append(f"{run.run_id}: predictions must have an integer dtype")
    if arrays["tail_mask"].dtype != np.bool_:
        errors.append(f"{run.run_id}: tail_mask must have boolean dtype")
    if not np.isfinite(arrays["logits"]).all():
        errors.append(f"{run.run_id}: logits contain NaN or infinity")
    if arrays["logits"].shape == (n_nodes, n_classes):
        predicted = arrays["logits"].argmax(axis=-1)
        if not np.array_equal(predicted, arrays["predictions"]):
            errors.append(f"{run.run_id}: predictions differ from argmax(logits)")

    checksums = dataset_manifest.get("array_checksums_sha256", {})
    for array_name, template in contract["array_checksum_keys"].items():
        manifest_key = template.format(split=run.split)
        expected = checksums.get(manifest_key)
        actual = array_sha256(arrays[array_name])
        if expected is None:
            errors.append(
                f"{run.run_id}: dataset manifest lacks checksum {manifest_key!r}"
            )
        elif actual != expected:
            errors.append(
                f"{run.run_id}: {array_name} checksum differs from "
                f"registered {manifest_key!r}"
            )

    if errors:
        return errors

    stored_metrics = load_json(run_dir / "metrics.json")
    if not isinstance(stored_metrics, dict):
        return [f"{run.run_id}: metrics.json must contain an object"]
    intervals = [
        str(value["interval"])
        for value in dataset_manifest["local_homophily"]["merged_bins"]
    ]
    recalculated = gap_audit_metrics(
        y_true=arrays["labels"],
        y_pred=arrays["predictions"],
        test_idx=arrays["test_idx"],
        merged_bin=arrays["merged_bin"],
        tail_mask=arrays["tail_mask"],
        bin_intervals=intervals,
        standard_metric=str(dataset_manifest["metric"]),
    )

    for field in (
        "whole_test_standard",
        "whole_test_macro_f1",
        "tail_macro_f1",
        "tail_gap",
        "worst_bin_macro_f1",
    ):
        actual = stored_metrics.get(field)
        expected = recalculated[field]
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not np.isclose(
                float(actual),
                float(expected),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        ):
            errors.append(
                f"{run.run_id}: metrics.{field} differs from run_data.npz"
            )
    for field in (
        "standard_metric",
        "whole_test_true_labels",
        "tail_true_labels",
        "tail_n_test",
        "worst_bin",
    ):
        if stored_metrics.get(field) != recalculated[field]:
            errors.append(
                f"{run.run_id}: metrics.{field} differs from run_data.npz"
            )

    stored_bins = stored_metrics.get("per_bin")
    expected_bins = recalculated["per_bin"]
    if not isinstance(stored_bins, list) or len(stored_bins) != len(expected_bins):
        errors.append(
            f"{run.run_id}: metrics.per_bin has the wrong number of rows"
        )
    else:
        for index, (actual, expected) in enumerate(
            zip(stored_bins, expected_bins)
        ):
            if not isinstance(actual, dict):
                errors.append(
                    f"{run.run_id}: metrics.per_bin[{index}] is not an object"
                )
                continue
            for field in ("bin", "n_test", "true_labels"):
                if actual.get(field) != expected[field]:
                    errors.append(
                        f"{run.run_id}: metrics.per_bin[{index}].{field} "
                        "differs from run_data.npz"
                    )
            for field in ("macro_f1", "accuracy"):
                value = actual.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not np.isclose(
                        float(value),
                        float(expected[field]),
                        rtol=0.0,
                        atol=1e-12,
                        equal_nan=True,
                    )
                ):
                    errors.append(
                        f"{run.run_id}: metrics.per_bin[{index}].{field} "
                        "differs from run_data.npz"
                    )
    return errors


def validate_store(
    *,
    raw_root: Path,
    panel_path: Path,
    dataset_manifest_dir: Path,
    model_config_dir: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
    level: str = "metadata",
    progress_every: int = 250,
    require_reference_environment: bool = False,
) -> dict[str, Any]:
    if level not in {"metadata", "arrays"}:
        raise ContractError(f"Unknown verification level: {level!r}")
    raw_root = raw_root.resolve()
    contract = load_contract(contract_path)
    panel = load_panel(panel_path)
    experiment_manifest_sha256 = file_sha256(panel_path)
    manifests = load_dataset_manifests(dataset_manifest_dir)
    if model_config_dir is None:
        model_config_dir = REPO_ROOT / contract["model_config_dir"]
    model_configs = load_model_configs(
        model_config_dir,
        {run.model for run in panel},
    )
    errors: list[str] = []

    missing_manifest_datasets = sorted(
        {run.dataset for run in panel} - manifests.keys()
    )
    if missing_manifest_datasets:
        raise ContractError(
            "Run panel names datasets without manifests: "
            + ", ".join(missing_manifest_datasets)
        )

    excluded_dirs = set(contract["excluded_root_directories"])
    allowed_files = set(contract["allowed_root_files"])
    if not raw_root.is_dir():
        raise ContractError(f"Raw root is not a directory: {raw_root}")
    active_dirs = {
        path.name: path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name not in excluded_dirs
    }
    unexpected_root_files = sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_file() and path.name not in allowed_files
    )
    if unexpected_root_files:
        errors.append(
            "Unexpected files at raw-store root: "
            + ", ".join(unexpected_root_files)
        )

    expected_ids = {run.run_id for run in panel}
    found_ids = set(active_dirs)
    missing_ids = sorted(expected_ids - found_ids)
    extra_ids = sorted(found_ids - expected_ids)
    if missing_ids:
        errors.append(
            f"Missing {len(missing_ids)} registered run directories; "
            f"first: {', '.join(missing_ids[:10])}"
        )
    if extra_ids:
        errors.append(
            f"Found {len(extra_ids)} unregistered active directories; "
            f"first: {', '.join(extra_ids[:10])}"
        )

    checked = 0
    environment_counts: Counter[str] = Counter()
    code_version_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    parent_hash_recorded = 0
    for run in panel:
        run_dir = active_dirs.get(run.run_id)
        if run_dir is None:
            continue
        metadata_errors, provenance = validate_metadata(
            run,
            run_dir,
            contract,
            manifests[run.dataset],
            model_configs[run.model],
            require_reference_environment=require_reference_environment,
            experiment_manifest_sha256=experiment_manifest_sha256,
        )
        errors.extend(metadata_errors)
        if provenance is not None:
            environment_key = json.dumps(
                provenance["versions"], sort_keys=True, separators=(",", ":")
            )
            environment_counts[environment_key] += 1
            code_version_counts[str(provenance["code_version"])] += 1
            device_counts[str(provenance["device"])] += 1
            parent_hash_recorded += int(
                provenance["parent_checkpoint_hash_recorded"]
            )
        if level == "arrays":
            errors.extend(validate_arrays(run, run_dir, contract, manifests[run.dataset]))
        checked += 1
        if progress_every and (checked % progress_every == 0 or checked == len(panel)):
            print(f"checked {checked}/{len(panel)} runs", file=sys.stderr)

    counts = Counter((run.dataset, run.model, run.preparation) for run in panel)
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "verification_level": level,
        "raw_root": str(raw_root),
        "panel_manifest": str(panel_path.resolve()),
        "dataset_manifest_dir": str(dataset_manifest_dir.resolve()),
        "expected_runs": len(panel),
        "found_active_run_directories": len(active_dirs),
        "checked_runs": checked,
        "panel_cells": len(counts),
        "require_reference_environment": require_reference_environment,
        "recorded_training_environments": [
            {"versions": json.loads(key), "runs": count}
            for key, count in sorted(
                environment_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "recorded_code_versions": dict(sorted(code_version_counts.items())),
        "recorded_devices": dict(sorted(device_counts.items())),
        "runs_with_parent_checkpoint_hash": parent_hash_recorded,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="JSONL run panel; defaults to the path recorded in the contract.",
    )
    parser.add_argument(
        "--dataset-manifest-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--level",
        choices=("metadata", "arrays"),
        default="metadata",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--require-reference-environment",
        action="store_true",
        help=(
            "Fail runs whose recorded Python, PyTorch or NumPy version differs "
            "from the reference lock. Historical deposited runs are reported "
            "but accepted by default."
        ),
    )
    parser.add_argument(
        "--max-reported-errors",
        type=int,
        default=50,
        help="Maximum errors printed to stderr; the JSON report retains all errors.",
    )
    args = parser.parse_args()

    contract = load_contract(args.contract)
    panel_path = args.panel or REPO_ROOT / contract["panel_manifest"]
    manifest_dir = (
        args.dataset_manifest_dir
        or REPO_ROOT / contract["dataset_manifest_dir"]
    )
    try:
        report = validate_store(
            raw_root=args.raw_root,
            panel_path=panel_path,
            dataset_manifest_dir=manifest_dir,
            model_config_dir=args.model_config_dir,
            contract_path=args.contract,
            level=args.level,
            progress_every=args.progress_every,
            require_reference_environment=args.require_reference_environment,
        )
    except ContractError as error:
        parser.exit(2, f"contract error: {error}\n")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {key: value for key, value in report.items() if key != "errors"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if report["errors"]:
        for error in report["errors"][: args.max_reported_errors]:
            print(f"ERROR: {error}", file=sys.stderr)
        remaining = len(report["errors"]) - args.max_reported_errors
        if remaining > 0:
            print(f"ERROR: {remaining} further errors omitted", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
