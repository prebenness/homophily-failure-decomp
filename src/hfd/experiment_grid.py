"""Experiment-grid construction, validation, and resumable execution planning."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml


GRID_FIELDS = (
    "run_order",
    "run_id",
    "dataset",
    "model",
    "preparation",
    "split",
    "seed",
    "parent_run_id",
)
RUN_DATA_ARRAYS = {
    "logits",
    "predictions",
    "labels",
    "train_idx",
    "val_idx",
    "test_idx",
    "merged_bin",
    "tail_mask",
}


@dataclass(frozen=True)
class ExperimentRun:
    """One atomic training run in the complete experiment panel."""

    run_order: int
    run_id: str
    dataset: str
    model: str
    preparation: str
    split: str
    seed: int
    parent_run_id: str = ""

    def as_record(self) -> dict[str, str | int]:
        return {
            "run_order": self.run_order,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "model": self.model,
            "preparation": self.preparation,
            "split": self.split,
            "seed": self.seed,
            "parent_run_id": self.parent_run_id,
        }


class RunState(str, Enum):
    """Filesystem state of one expected run."""

    MISSING = "missing"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RunInspection:
    run: ExperimentRun
    state: RunState
    detail: str = ""


def canonical_run_id(
    dataset: str,
    model: str,
    preparation: str,
    split: str,
    seed: int,
) -> str:
    return f"{dataset}_{model}_{preparation}_{split}_seed-{seed}"


def _read_yaml_mapping(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return document


def build_experiment_grid(
    spec_path: Path,
    dataset_manifest_dir: Path,
    model_config_dir: Path,
) -> tuple[list[ExperimentRun], tuple[str, ...]]:
    """Build and validate the deterministic run panel from tracked inputs."""

    spec = _read_yaml_mapping(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError(f"{spec_path}: unsupported schema_version")

    datasets = spec.get("datasets")
    models = spec.get("models")
    preparation_records = spec.get("preparations")
    output_files = spec.get("run_output_files")
    if not all(isinstance(values, list) and values for values in (
        datasets,
        models,
        preparation_records,
        output_files,
    )):
        raise ValueError(
            f"{spec_path}: datasets, models, preparations, and run_output_files "
            "must be non-empty lists"
        )
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"{spec_path}: duplicate dataset")
    if len(models) != len(set(models)):
        raise ValueError(f"{spec_path}: duplicate model")
    if len(output_files) != len(set(output_files)):
        raise ValueError(f"{spec_path}: duplicate run output filename")

    preparations: list[str] = []
    dependencies: dict[str, str] = {}
    for record in preparation_records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError(f"{spec_path}: each preparation requires a name")
        name = record["name"]
        if name in preparations:
            raise ValueError(f"{spec_path}: duplicate preparation {name!r}")
        preparations.append(name)
        dependency = record.get("depends_on")
        if dependency is not None:
            if not isinstance(dependency, str):
                raise ValueError(
                    f"{spec_path}: dependency for {name!r} must be a string"
                )
            dependencies[name] = dependency

    for preparation, dependency in dependencies.items():
        if dependency not in preparations:
            raise ValueError(
                f"{spec_path}: {preparation!r} depends on unknown {dependency!r}"
            )
        if preparations.index(dependency) >= preparations.index(preparation):
            raise ValueError(
                f"{spec_path}: dependency {dependency!r} must precede "
                f"{preparation!r}"
            )

    expected_cell_runs = int(spec["expected_runs_per_dataset_model_preparation"])
    rows: list[ExperimentRun] = []
    run_order = 1
    for dataset in datasets:
        manifest_path = dataset_manifest_dir / f"{dataset}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != dataset:
            raise ValueError(
                f"{manifest_path}: dataset field does not match {dataset!r}"
            )
        splits = [record["split_id"] for record in manifest["splits"]]
        seeds = [int(seed) for seed in manifest["training_seeds"]]
        if len(splits) != len(set(splits)):
            raise ValueError(f"{manifest_path}: duplicate split_id")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{manifest_path}: duplicate training seed")
        cell_runs = len(splits) * len(seeds)
        if cell_runs != expected_cell_runs or manifest.get("n_runs") != cell_runs:
            raise ValueError(
                f"{manifest_path}: expected {expected_cell_runs} split/seed "
                f"combinations, found {cell_runs}"
            )

        for model in models:
            model_path = model_config_dir / f"{model}.yaml"
            model_config = _read_yaml_mapping(model_path)
            if model_config.get("model") != model:
                raise ValueError(
                    f"{model_path}: model field does not match {model!r}"
                )
            for split in splits:
                for seed in seeds:
                    run_ids = {
                        preparation: canonical_run_id(
                            dataset, model, preparation, split, seed
                        )
                        for preparation in preparations
                    }
                    for preparation in preparations:
                        dependency = dependencies.get(preparation)
                        rows.append(
                            ExperimentRun(
                                run_order=run_order,
                                run_id=run_ids[preparation],
                                dataset=dataset,
                                model=model,
                                preparation=preparation,
                                split=split,
                                seed=seed,
                                parent_run_id=run_ids[dependency] if dependency else "",
                            )
                        )
                        run_order += 1

    expected_total = int(spec["expected_total_runs"])
    if len(rows) != expected_total:
        raise ValueError(
            f"{spec_path}: expected {expected_total} total runs, built {len(rows)}"
        )
    _validate_grid(rows)
    return rows, tuple(str(filename) for filename in output_files)


def _validate_grid(rows: Sequence[ExperimentRun]) -> None:
    if not rows:
        raise ValueError("experiment grid is empty")
    ids = {row.run_id for row in rows}
    if len(ids) != len(rows):
        raise ValueError("experiment grid contains duplicate run_id values")
    by_id = {row.run_id: row for row in rows}
    for expected_order, row in enumerate(rows, start=1):
        if row.run_order != expected_order:
            raise ValueError("run_order must be contiguous and start at 1")
        expected_id = canonical_run_id(
            row.dataset, row.model, row.preparation, row.split, row.seed
        )
        if row.run_id != expected_id:
            raise ValueError(
                f"run_id {row.run_id!r} does not match its experiment fields"
            )
        if row.parent_run_id:
            try:
                parent = by_id[row.parent_run_id]
            except KeyError as exc:
                raise ValueError(
                    f"{row.run_id}: parent {row.parent_run_id!r} is absent"
                ) from exc
            if parent.run_order >= row.run_order:
                raise ValueError(f"{row.run_id}: parent must precede child")
            if (
                parent.dataset,
                parent.model,
                parent.split,
                parent.seed,
            ) != (
                row.dataset,
                row.model,
                row.split,
                row.seed,
            ):
                raise ValueError(
                    f"{row.run_id}: parent does not share dataset/model/split/seed"
                )


def grid_jsonl_text(rows: Sequence[ExperimentRun]) -> str:
    _validate_grid(rows)
    return "".join(
        json.dumps(row.as_record(), separators=(",", ":")) + "\n"
        for row in rows
    )


def read_experiment_grid(path: Path) -> list[ExperimentRun]:
    rows: list[ExperimentRun] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            missing = [field for field in GRID_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields {', '.join(missing)}"
                )
            rows.append(
                ExperimentRun(
                    run_order=int(record["run_order"]),
                    run_id=record["run_id"],
                    dataset=record["dataset"],
                    model=record["model"],
                    preparation=record["preparation"],
                    split=record["split"],
                    seed=int(record["seed"]),
                    parent_run_id=record["parent_run_id"],
                )
            )
    _validate_grid(rows)
    return rows


def select_runs(
    rows: Sequence[ExperimentRun],
    *,
    datasets: Iterable[str] = (),
    models: Iterable[str] = (),
    preparations: Iterable[str] = (),
    splits: Iterable[str] = (),
    seeds: Iterable[int] = (),
    include_dependencies: bool = True,
) -> list[ExperimentRun]:
    """Select rows while retaining required parent runs."""

    filters = {
        "dataset": set(datasets),
        "model": set(models),
        "preparation": set(preparations),
        "split": set(splits),
        "seed": set(seeds),
    }

    def matches(row: ExperimentRun) -> bool:
        return all(
            not accepted or getattr(row, field) in accepted
            for field, accepted in filters.items()
        )

    selected_ids = {row.run_id for row in rows if matches(row)}
    if include_dependencies:
        by_id = {row.run_id: row for row in rows}
        pending = list(selected_ids)
        while pending:
            row = by_id[pending.pop()]
            if row.parent_run_id and row.parent_run_id not in selected_ids:
                selected_ids.add(row.parent_run_id)
                pending.append(row.parent_run_id)
    return [row for row in rows if row.run_id in selected_ids]


def inspect_run(
    row: ExperimentRun,
    runs_root: Path,
    required_output_files: Sequence[str],
) -> RunInspection:
    run_dir = runs_root / row.run_id
    if not run_dir.exists():
        return RunInspection(row, RunState.MISSING)
    if not run_dir.is_dir():
        return RunInspection(row, RunState.INVALID, "run path is not a directory")

    absent = [
        filename
        for filename in required_output_files
        if not (run_dir / filename).is_file()
    ]
    if absent:
        return RunInspection(
            row,
            RunState.INCOMPLETE,
            f"missing files: {', '.join(absent)}",
        )
    empty = [
        filename
        for filename in required_output_files
        if (run_dir / filename).stat().st_size == 0
    ]
    if empty:
        return RunInspection(
            row,
            RunState.INCOMPLETE,
            f"empty files: {', '.join(empty)}",
        )

    config_path = run_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunInspection(row, RunState.INVALID, f"invalid config.json: {exc}")
    expected: Mapping[str, object] = {
        "run_id": row.run_id,
        "dataset": row.dataset,
        "model": row.model,
        "preparation": row.preparation,
        "split": row.split,
        "seed": row.seed,
        "parent_run_id": row.parent_run_id or None,
    }
    mismatches = [
        key for key, value in expected.items() if config.get(key) != value
    ]
    if mismatches:
        return RunInspection(
            row,
            RunState.INVALID,
            f"config mismatch: {', '.join(mismatches)}",
        )

    for filename, expected_type in (
        ("metrics.json", dict),
        ("history.json", list),
    ):
        path = run_dir / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return RunInspection(
                row,
                RunState.INVALID,
                f"invalid {filename}: {exc}",
            )
        if not isinstance(value, expected_type) or (
            filename == "history.json" and not value
        ):
            return RunInspection(
                row,
                RunState.INVALID,
                f"{filename} has the wrong structure",
            )

    checkpoint_path = run_dir / "checkpoint.pt"
    if not zipfile.is_zipfile(checkpoint_path):
        return RunInspection(
            row,
            RunState.INVALID,
            "checkpoint.pt is not a PyTorch ZIP checkpoint",
        )

    try:
        with np.load(run_dir / "run_data.npz", allow_pickle=False) as archive:
            missing_arrays = sorted(RUN_DATA_ARRAYS - set(archive.files))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return RunInspection(
            row,
            RunState.INVALID,
            f"invalid run_data.npz: {exc}",
        )
    if missing_arrays:
        return RunInspection(
            row,
            RunState.INVALID,
            f"run_data.npz missing arrays: {', '.join(missing_arrays)}",
        )
    return RunInspection(row, RunState.COMPLETE)


def inspect_runs(
    rows: Sequence[ExperimentRun],
    runs_root: Path,
    required_output_files: Sequence[str],
) -> list[RunInspection]:
    return [
        inspect_run(row, runs_root, required_output_files)
        for row in rows
    ]


def state_counts(inspections: Sequence[RunInspection]) -> Counter[RunState]:
    counts: Counter[RunState] = Counter(item.state for item in inspections)
    for state in RunState:
        counts[state] += 0
    return counts


def execution_plan(
    rows: Sequence[ExperimentRun],
    inspections: Sequence[RunInspection],
    *,
    rerun_invalid: bool = False,
) -> list[ExperimentRun]:
    """Return dependency-safe rows that require execution.

    A recovery child is rerun whenever its ERM parent is rerun. This prevents
    a completed child from silently retaining weights derived from a
    superseded parent checkpoint.
    """

    if [row.run_id for row in rows] != [
        inspection.run.run_id for inspection in inspections
    ]:
        raise ValueError("rows and inspections are not aligned")
    states = {inspection.run.run_id: inspection.state for inspection in inspections}
    planned_ids: set[str] = set()
    for row, inspection in zip(rows, inspections):
        if inspection.state is RunState.INVALID and not rerun_invalid:
            raise RuntimeError(
                f"{row.run_id}: existing output is invalid ({inspection.detail}); "
                "inspect it or pass rerun_invalid=True"
            )
        if inspection.state is not RunState.COMPLETE:
            planned_ids.add(row.run_id)

    for row in rows:
        if row.parent_run_id in planned_ids:
            planned_ids.add(row.run_id)

    planned: list[ExperimentRun] = []
    scheduled_ids: set[str] = set()
    for row in rows:
        if row.run_id not in planned_ids:
            continue
        if row.parent_run_id:
            parent_ready = (
                states.get(row.parent_run_id) is RunState.COMPLETE
                or row.parent_run_id in scheduled_ids
            )
            if not parent_ready:
                raise RuntimeError(
                    f"{row.run_id}: required parent {row.parent_run_id} is neither "
                    "complete nor scheduled"
                )
        planned.append(row)
        scheduled_ids.add(row.run_id)
    return planned


def train_command(
    python_executable: str,
    training_script: Path,
    row: ExperimentRun,
    *,
    output_root: Path,
    device: str | None = None,
    dataset_config: Path | None = None,
    experiment_manifest: Path | None = None,
    replace_existing: bool = False,
    require_clean: bool = False,
) -> list[str]:
    command = [
        python_executable,
        str(training_script),
        "--dataset",
        row.dataset,
        "--model",
        row.model,
        "--preparation",
        row.preparation,
        "--split",
        row.split,
        "--seed",
        str(row.seed),
        "--output-root",
        str(output_root),
    ]
    if dataset_config is not None:
        command.extend(["--config", str(dataset_config)])
    if experiment_manifest is not None:
        command.extend(["--experiment-manifest", str(experiment_manifest)])
    if device is not None:
        command.extend(["--device", device])
    if replace_existing:
        command.append("--replace-existing")
    if require_clean:
        command.append("--require-clean")
    return command
