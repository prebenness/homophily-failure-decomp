"""Shared dataset metadata for the main and additional-dataset analyses.

The default analysis mode remains the five-dataset panel used in the paper.
Callers must request ``additional`` or ``extended`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Literal, Mapping


TailSide = Literal["low-h_v", "high-h_v"]
DatasetFamily = Literal["homophilic", "heterophilic"]


class AnalysisDatasetMode(str, Enum):
    """Dataset panel selected by an analysis."""

    ORIGINAL = "original"
    ADDITIONAL = "additional"
    EXTENDED = "extended"


ORIGINAL_DATASET_ORDER = (
    "pubmed",
    "ogbn-arxiv",
    "roman-empire",
    "amazon-ratings",
    "arxiv-year",
)
ADDITIONAL_DATASET_ORDER = ("actor", "blogcatalog")
EXTENDED_DATASET_ORDER = ORIGINAL_DATASET_ORDER + ADDITIONAL_DATASET_ORDER

ORIGINAL_MANIFEST_DIR = Path("manifests/datasets")
ADDITIONAL_MANIFEST_DIR = Path("manifests/additional-datasets")
ORIGINAL_RUN_PANEL = Path("manifests/main-runs.jsonl")
ADDITIONAL_RUN_PANEL = Path("manifests/additional-dataset-runs.jsonl")


@dataclass(frozen=True)
class AnalysisDataset:
    """Analysis metadata for one registered dataset."""

    name: str
    display_label: str
    tail_side: TailSide
    family: DatasetFamily
    manifest_dir: Path

    def manifest_path(self, repo_root: Path) -> Path:
        """Return this dataset's frozen-manifest path below ``repo_root``."""

        return repo_root / self.manifest_dir / f"{self.name}.json"


@dataclass(frozen=True)
class RegisteredRun:
    """Identity of one run in a frozen JSONL experiment panel."""

    run_id: str
    dataset: str
    model: str
    preparation: str
    split: str
    seed: int
    parent_run_id: str | None


DATASET_REGISTRY: Mapping[str, AnalysisDataset] = MappingProxyType(
    {
        "pubmed": AnalysisDataset(
            name="pubmed",
            display_label="PubMed",
            tail_side="low-h_v",
            family="homophilic",
            manifest_dir=ORIGINAL_MANIFEST_DIR,
        ),
        "ogbn-arxiv": AnalysisDataset(
            name="ogbn-arxiv",
            display_label="ogbn-arxiv",
            tail_side="low-h_v",
            family="homophilic",
            manifest_dir=ORIGINAL_MANIFEST_DIR,
        ),
        "roman-empire": AnalysisDataset(
            name="roman-empire",
            display_label="Roman-empire",
            tail_side="high-h_v",
            family="heterophilic",
            manifest_dir=ORIGINAL_MANIFEST_DIR,
        ),
        "amazon-ratings": AnalysisDataset(
            name="amazon-ratings",
            display_label="Amazon-ratings",
            tail_side="high-h_v",
            family="heterophilic",
            manifest_dir=ORIGINAL_MANIFEST_DIR,
        ),
        "arxiv-year": AnalysisDataset(
            name="arxiv-year",
            display_label="arxiv-year",
            tail_side="high-h_v",
            family="heterophilic",
            manifest_dir=ORIGINAL_MANIFEST_DIR,
        ),
        "actor": AnalysisDataset(
            name="actor",
            display_label="Actor",
            tail_side="high-h_v",
            family="heterophilic",
            manifest_dir=ADDITIONAL_MANIFEST_DIR,
        ),
        "blogcatalog": AnalysisDataset(
            name="blogcatalog",
            display_label="BlogCatalog",
            tail_side="high-h_v",
            family="heterophilic",
            manifest_dir=ADDITIONAL_MANIFEST_DIR,
        ),
    }
)

_DATASET_ORDERS = MappingProxyType(
    {
        AnalysisDatasetMode.ORIGINAL: ORIGINAL_DATASET_ORDER,
        AnalysisDatasetMode.ADDITIONAL: ADDITIONAL_DATASET_ORDER,
        AnalysisDatasetMode.EXTENDED: EXTENDED_DATASET_ORDER,
    }
)


def dataset_order(
    mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> tuple[str, ...]:
    """Return the registered order for one analysis mode."""

    return _DATASET_ORDERS[AnalysisDatasetMode(mode)]


def run_panel_paths(
    repo_root: Path,
    mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> tuple[Path, ...]:
    """Return the frozen run panels required by an analysis mode."""

    selected_mode = AnalysisDatasetMode(mode)
    paths: list[Path] = []
    if selected_mode in {
        AnalysisDatasetMode.ORIGINAL,
        AnalysisDatasetMode.EXTENDED,
    }:
        paths.append(repo_root / ORIGINAL_RUN_PANEL)
    if selected_mode in {
        AnalysisDatasetMode.ADDITIONAL,
        AnalysisDatasetMode.EXTENDED,
    }:
        paths.append(repo_root / ADDITIONAL_RUN_PANEL)
    return tuple(paths)


def _normalise_parent_run_id(value: object) -> str | None:
    return None if value in {None, ""} else str(value)


def load_registered_runs(
    repo_root: Path,
    mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
    included_datasets: Iterable[str] | None = None,
) -> tuple[RegisteredRun, ...]:
    """Load and validate run identities from the frozen JSONL panels."""

    selected_mode = AnalysisDatasetMode(mode)
    allowed = set(dataset_order(selected_mode))
    included = allowed if included_datasets is None else set(included_datasets)
    invalid = sorted(included - allowed)
    if invalid:
        raise ValueError(
            f"Datasets outside {selected_mode.value!r} mode: {invalid}"
        )

    required = {
        "run_id",
        "dataset",
        "model",
        "preparation",
        "split",
        "seed",
        "parent_run_id",
    }
    registrations: list[RegisteredRun] = []
    seen_ids: set[str] = set()
    for panel_path in run_panel_paths(repo_root, selected_mode):
        if not panel_path.is_file():
            raise FileNotFoundError(f"Frozen run panel not found: {panel_path}")
        for line_number, line in enumerate(
            panel_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{panel_path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{panel_path}:{line_number}: expected a JSON object"
                )
            missing = sorted(required - value.keys())
            if missing:
                raise ValueError(
                    f"{panel_path}:{line_number}: missing fields {missing}"
                )
            dataset = str(value["dataset"])
            if dataset not in allowed:
                raise ValueError(
                    f"{panel_path}:{line_number}: dataset {dataset!r} is "
                    f"outside {selected_mode.value!r} mode"
                )
            if dataset not in included:
                continue
            registration = RegisteredRun(
                run_id=str(value["run_id"]),
                dataset=dataset,
                model=str(value["model"]),
                preparation=str(value["preparation"]),
                split=str(value["split"]),
                seed=int(value["seed"]),
                parent_run_id=_normalise_parent_run_id(
                    value["parent_run_id"]
                ),
            )
            expected_id = (
                f"{registration.dataset}_{registration.model}_"
                f"{registration.preparation}_{registration.split}_"
                f"seed-{registration.seed}"
            )
            if registration.run_id != expected_id:
                raise ValueError(
                    f"{panel_path}:{line_number}: run_id "
                    f"{registration.run_id!r} should be {expected_id!r}"
                )
            if registration.run_id in seen_ids:
                raise ValueError(
                    f"Duplicate registered run_id: {registration.run_id}"
                )
            seen_ids.add(registration.run_id)
            registrations.append(registration)

    registered_datasets = {run.dataset for run in registrations}
    if registered_datasets != included:
        raise ValueError(
            "Frozen run-panel dataset mismatch; "
            f"missing={sorted(included - registered_datasets)}, "
            f"unexpected={sorted(registered_datasets - included)}"
        )
    return tuple(registrations)


def validate_registered_run_configs(
    configs: Iterable[Mapping[str, object]],
    registrations: Iterable[RegisteredRun],
    *,
    context: str = "raw run store",
) -> None:
    """Require exactly the registered IDs and identity fields in ``configs``."""

    expected = {run.run_id: run for run in registrations}
    observed: dict[str, Mapping[str, object]] = {}
    duplicate_ids: list[str] = []
    for config in configs:
        run_id = str(config.get("run_id"))
        if run_id in observed:
            duplicate_ids.append(run_id)
        observed[run_id] = config
    if duplicate_ids:
        raise ValueError(
            f"{context}: duplicate run IDs: {sorted(set(duplicate_ids))[:10]}"
        )

    expected_ids = set(expected)
    observed_ids = set(observed)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"{context}: registered run ID mismatch; "
            f"missing={missing[:10]} ({len(missing)} total), "
            f"unexpected={unexpected[:10]} ({len(unexpected)} total)"
        )

    for run_id, registration in expected.items():
        config = observed[run_id]
        actual = {
            "dataset": str(config.get("dataset")),
            "model": str(config.get("model")),
            "preparation": str(config.get("preparation")),
            "split": str(config.get("split")),
            "seed": int(config.get("seed", -1)),
            "parent_run_id": _normalise_parent_run_id(
                config.get("parent_run_id")
            ),
        }
        registered = {
            "dataset": registration.dataset,
            "model": registration.model,
            "preparation": registration.preparation,
            "split": registration.split,
            "seed": registration.seed,
            "parent_run_id": registration.parent_run_id,
        }
        if actual != registered:
            raise ValueError(
                f"{context}: {run_id} identity differs from the frozen panel; "
                f"observed={actual}, registered={registered}"
            )
