"""Summarise recorded GPU-hours for an original, additional, or extended panel.

The raw archive stores ``checkpoint.train_seconds`` in each run's
``metrics.json``. Every retained run used one GPU through PyTorch's ``cuda``
device interface (which PyTorch also uses for ROCm), so total GPU-hours are the
sum of these durations divided by 3,600.

Example:
    python scripts/analyze_training_compute.py \
        --raw-dir /path/to/raw-runs \
        --output-dir results/main/compute
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from hfd.analysis_datasets import (
    ADDITIONAL_DATASET_ORDER,
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    RegisteredRun,
    dataset_order,
    load_registered_runs,
    validate_registered_run_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_ORDER = ("erm", "last", "head", "full")
EXPECTED_RUNS_PER_CELL = 30
MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "linkx",
    "acm-gcn",
)
# Legacy constants retained for callers that use the original five-dataset
# analysis directly.
EXPECTED_RUNS_PER_PREPARATION = 1_050
EXPECTED_RUNS = EXPECTED_RUNS_PER_PREPARATION * len(PREPARATION_ORDER)
DEFAULT_OUTPUT_DIRS = {
    AnalysisDatasetMode.ORIGINAL: REPO_ROOT / "results" / "main" / "compute",
    AnalysisDatasetMode.ADDITIONAL: (
        REPO_ROOT / "results" / "additional" / "compute"
    ),
    AnalysisDatasetMode.EXTENDED: REPO_ROOT / "results" / "extended" / "compute",
}
ORIGINAL_HARDWARE_ROWS = (
    {"gpu_model": "NVIDIA A100", "memory_gb": 80},
    {
        "gpu_model": "NVIDIA GeForce RTX 3080 Laptop GPU",
        "memory_gb": 16,
    },
)


@dataclass(frozen=True)
class RunTime:
    dataset: str
    model: str
    preparation: str
    seconds: float
    split: str = ""
    seed: int = 0
    gpu_name: str | None = None


def selected_datasets(
    mode: AnalysisDatasetMode | str,
    excluded_datasets: tuple[str, ...] | list[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the full mode order and the non-excluded analysis datasets."""

    mode_order = dataset_order(mode)
    invalid = sorted(set(excluded_datasets) - set(mode_order))
    if invalid:
        raise ValueError(
            f"Cannot exclude datasets outside {AnalysisDatasetMode(mode).value!r} "
            f"mode: {invalid}"
        )
    selected = tuple(
        dataset for dataset in mode_order if dataset not in excluded_datasets
    )
    if not selected:
        raise ValueError("At least one dataset must remain after exclusions")
    return mode_order, selected


def raw_root_specs(
    mode: AnalysisDatasetMode | str,
    raw_root: Path | None,
    additional_raw_root: Path | None,
) -> tuple[tuple[str, Path, frozenset[str]], ...]:
    """Resolve the run stores required by one dataset mode."""

    mode = AnalysisDatasetMode(mode)
    specs: list[tuple[str, Path, frozenset[str]]] = []
    if mode in {AnalysisDatasetMode.ORIGINAL, AnalysisDatasetMode.EXTENDED}:
        if raw_root is None:
            raise ValueError(f"--raw-root is required for {mode.value!r} mode")
        specs.append(("original", raw_root, frozenset(ORIGINAL_DATASET_ORDER)))
    if mode in {AnalysisDatasetMode.ADDITIONAL, AnalysisDatasetMode.EXTENDED}:
        if additional_raw_root is None:
            raise ValueError(
                f"--additional-raw-root is required for {mode.value!r} mode"
            )
        specs.append(
            (
                "additional",
                additional_raw_root,
                frozenset(ADDITIONAL_DATASET_ORDER),
            )
        )
    return tuple(specs)


def collect_runs(
    root_specs: tuple[tuple[str, Path, frozenset[str]], ...],
    included_datasets: tuple[str, ...],
    registrations: tuple[RegisteredRun, ...] | None = None,
) -> list[RunTime]:
    """Read run times while enforcing raw-store ownership and unique IDs."""

    runs: list[RunTime] = []
    included = set(included_datasets)
    seen_run_ids: set[str] = set()
    selected_configs: list[dict] = []
    for store_name, root, allowed_datasets in root_specs:
        if not root.is_dir():
            raise FileNotFoundError(f"Raw run store not found: {root}")
        run_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name != "historical-records"
        )
        unexpected_directories = [
            path.name for path in run_dirs if not (path / "config.json").is_file()
        ]
        if unexpected_directories:
            raise ValueError(
                f"Unexpected non-run directories in {store_name} raw root: "
                f"{unexpected_directories}"
            )
        for run_dir in run_dirs:
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            dataset = str(config.get("dataset"))
            if dataset not in allowed_datasets:
                raise ValueError(
                    f"{run_dir.name}: dataset {dataset!r} is in the wrong raw "
                    f"store ({store_name})"
                )
            run_id = str(config.get("run_id"))
            if run_id != run_dir.name:
                raise ValueError(f"{run_dir.name}: directory and run_id differ")
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate run_id across raw stores: {run_id}")
            seen_run_ids.add(run_id)
            if dataset not in included:
                continue
            selected_configs.append(config)

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                raise ValueError(f"{run_dir.name}: missing metrics.json")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            preparation = str(config["preparation"])
            if preparation not in PREPARATION_ORDER:
                raise ValueError(
                    f"{run_dir.name}: unexpected preparation {preparation!r}"
                )
            if config.get("device") != "cuda":
                raise ValueError(
                    f"{run_dir.name}: expected one GPU through the PyTorch "
                    "cuda device interface"
                )

            model = str(config["model"])
            if model not in MODEL_ORDER:
                raise ValueError(f"{run_dir.name}: unexpected model {model!r}")
            seconds = float(metrics["checkpoint"]["train_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(
                    f"{run_dir.name}: invalid checkpoint.train_seconds={seconds!r}"
                )
            accelerator = config.get("accelerator")
            gpu_name = (
                accelerator.get("gpu_name")
                if isinstance(accelerator, dict)
                and isinstance(accelerator.get("gpu_name"), str)
                else None
            )
            runs.append(
                RunTime(
                    dataset=dataset,
                    model=model,
                    preparation=preparation,
                    split=str(config["split"]),
                    seed=int(config["seed"]),
                    seconds=seconds,
                    gpu_name=gpu_name,
                )
            )
    if registrations is not None:
        validate_registered_run_configs(
            selected_configs,
            registrations,
            context="training-compute input",
        )
    return runs


def validate_run_panel(
    runs: list[RunTime],
    manifests: dict[str, dict],
    included_datasets: tuple[str, ...],
    *,
    model_order: tuple[str, ...] = MODEL_ORDER,
) -> None:
    """Require every registered split/seed pair in every analysis cell."""

    by_cell: dict[tuple[str, str, str], list[RunTime]] = defaultdict(list)
    for run in runs:
        by_cell[(run.dataset, run.model, run.preparation)].append(run)

    expected_total = 0
    for dataset in included_datasets:
        manifest = manifests[dataset]
        expected_pairs = {
            (str(split["split_id"]), int(seed))
            for split in manifest["splits"]
            for seed in manifest["training_seeds"]
        }
        if len(expected_pairs) != EXPECTED_RUNS_PER_CELL:
            raise ValueError(
                f"{dataset}: registered split/seed panel has "
                f"{len(expected_pairs)} pairs; expected {EXPECTED_RUNS_PER_CELL}"
            )
        for model in model_order:
            for preparation in PREPARATION_ORDER:
                cell = by_cell[(dataset, model, preparation)]
                expected_total += len(expected_pairs)
                if len(cell) != len(expected_pairs):
                    raise ValueError(
                        f"{dataset}/{model}/{preparation}: expected "
                        f"{len(expected_pairs)} runs, found {len(cell)}"
                    )
                observed_pairs = {(run.split, run.seed) for run in cell}
                if observed_pairs != expected_pairs:
                    missing = sorted(expected_pairs - observed_pairs)
                    unexpected = sorted(observed_pairs - expected_pairs)
                    raise ValueError(
                        f"{dataset}/{model}/{preparation}: run panel mismatch; "
                        f"missing={missing}, unexpected={unexpected}"
                    )
    if len(runs) != expected_total:
        raise ValueError(f"Expected {expected_total} runs, found {len(runs)}")


def load_runs(
    raw_root: Path | None,
    *,
    additional_raw_root: Path | None = None,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
    exclude_datasets: tuple[str, ...] | list[str] = (),
) -> list[RunTime]:
    """Load and validate one original, additional, or extended run panel."""

    _, included_datasets = selected_datasets(dataset_mode, exclude_datasets)
    specs = raw_root_specs(dataset_mode, raw_root, additional_raw_root)
    registrations = load_registered_runs(
        REPO_ROOT, dataset_mode, included_datasets
    )
    manifests = {
        dataset: json.loads(
            DATASET_REGISTRY[dataset]
            .manifest_path(REPO_ROOT)
            .read_text(encoding="utf-8")
        )
        for dataset in included_datasets
    }
    runs = collect_runs(specs, included_datasets, registrations)
    validate_run_panel(runs, manifests, included_datasets)
    return runs


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse(
    runs: list[RunTime],
    output_dir: Path,
    *,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
    included_datasets: tuple[str, ...] = ORIGINAL_DATASET_ORDER,
    registered_run_panel_exact_match: bool | None = None,
) -> None:
    mode = AnalysisDatasetMode(dataset_mode)
    by_preparation: dict[str, float] = defaultdict(float)
    counts = Counter(run.preparation for run in runs)

    for run in runs:
        by_preparation[run.preparation] += run.seconds

    total_seconds = sum(by_preparation.values())

    gpu_rows: list[dict[str, object]] = []
    for preparation in PREPARATION_ORDER:
        gpu_rows.append(
            {
                "preparation": preparation,
                "run_count": counts[preparation],
                "gpu_hours": f"{by_preparation[preparation] / 3_600:.6f}",
            }
        )
    gpu_rows.append(
        {
            "preparation": "total",
            "run_count": len(runs),
            "gpu_hours": f"{total_seconds / 3_600:.6f}",
        }
    )
    write_csv(
        output_dir / "gpu_hours_by_preparation.csv",
        [
            "preparation",
            "run_count",
            "gpu_hours",
        ],
        gpu_rows,
    )

    if mode is AnalysisDatasetMode.ORIGINAL:
        # Preserve the released five-dataset compute inventory exactly.
        hardware_rows = [dict(row) for row in ORIGINAL_HARDWARE_ROWS]
        hardware_assignment_note = (
            "The final raw records do not retain a reliable per-run GPU-model "
            "assignment, so GPU-hours are not partitioned by GPU model."
        )
    else:
        recorded_additional_rows = [
            {"gpu_model": gpu_name, "memory_gb": None}
            for gpu_name in sorted(
                {
                    run.gpu_name
                    for run in runs
                    if run.dataset in ADDITIONAL_DATASET_ORDER
                    and run.gpu_name is not None
                }
            )
        ]
        if mode is AnalysisDatasetMode.EXTENDED and any(
            dataset in ORIGINAL_DATASET_ORDER for dataset in included_datasets
        ):
            hardware_rows = [
                *(dict(row) for row in ORIGINAL_HARDWARE_ROWS),
                *recorded_additional_rows,
            ]
            hardware_assignment_note = (
                "The original-panel inventory is retained because its final raw "
                "records do not contain reliable per-run GPU assignments. GPU "
                "names for Actor and BlogCatalog are read from "
                "config.accelerator.gpu_name; their per-run accelerator memory "
                "was not recorded, so memory_gb is blank."
            )
        else:
            hardware_rows = recorded_additional_rows
            hardware_assignment_note = (
                "GPU model names for Actor and BlogCatalog are read from "
                "config.accelerator.gpu_name. Runs without a recorded name are "
                "omitted from this inventory. Per-run accelerator memory was not "
                "recorded, so memory_gb is blank."
            )
    write_csv(
        output_dir / "hardware_inventory.csv",
        ["gpu_model", "memory_gb"],
        hardware_rows,
    )

    manifest = {
        "analysis": (
            "main-panel-gpu-hours"
            if mode is AnalysisDatasetMode.ORIGINAL
            else f"{mode.value}-panel-gpu-hours"
        ),
        "dataset_mode": mode.value,
        "datasets": list(included_datasets),
        "run_count": len(runs),
        "preparations": list(PREPARATION_ORDER),
        "runs_per_preparation": counts[PREPARATION_ORDER[0]],
        "duration_field": "metrics.json: checkpoint.train_seconds",
        "duration_definition": "Time inside the optimisation loop.",
        "gpu_hours_definition": (
            "Sum of recorded training-loop seconds divided by 3600; every run "
            + (
                "used one CUDA GPU."
                if mode is AnalysisDatasetMode.ORIGINAL
                else (
                    "used one GPU through PyTorch's cuda device interface "
                    "(also used by ROCm)."
                )
            )
        ),
        "hardware": hardware_rows,
        "hardware_assignment_note": hardware_assignment_note,
    }
    if registered_run_panel_exact_match is not None:
        manifest["registered_run_panel_exact_match"] = (
            registered_run_panel_exact_match
        )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(mode.value for mode in AnalysisDatasetMode),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--raw-root",
        "--raw-dir",
        dest="raw_root",
        type=Path,
        help=(
            "Raw run store for the original five datasets. --raw-dir remains "
            "as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="Separate raw run store for Actor and BlogCatalog.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory; defaults to the mode-specific compute directory "
            "below results/main, results/additional, or results/extended."
        ),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        choices=tuple(DATASET_REGISTRY),
    )
    args = parser.parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    try:
        _, included_datasets = selected_datasets(
            mode, tuple(args.exclude_dataset)
        )
        raw_root_specs(mode, args.raw_root, args.additional_raw_root)
    except ValueError as error:
        parser.error(str(error))
    args.dataset_mode = mode
    args.included_datasets = included_datasets
    args.output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[mode]
    return args


def main() -> None:
    args = parse_args()
    runs = load_runs(
        args.raw_root,
        additional_raw_root=args.additional_raw_root,
        dataset_mode=args.dataset_mode,
        exclude_datasets=tuple(args.exclude_dataset),
    )
    analyse(
        runs,
        args.output_dir,
        dataset_mode=args.dataset_mode,
        included_datasets=args.included_datasets,
        registered_run_panel_exact_match=True,
    )
    print(
        f"Summarised {len(runs)} runs in {args.output_dir}; "
        "no host or facility identifiers were written."
    )


if __name__ == "__main__":
    main()
