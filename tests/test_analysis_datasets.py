from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from hfd.analysis_datasets import (
    ADDITIONAL_DATASET_ORDER,
    ORIGINAL_DATASET_ORDER,
    load_registered_runs,
    validate_registered_run_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_additional_panel_loads_exact_frozen_run_ids() -> None:
    registrations = load_registered_runs(REPO_ROOT, "additional")

    assert len(registrations) == 1_680
    assert {run.dataset for run in registrations} == set(
        ADDITIONAL_DATASET_ORDER
    )
    assert registrations[0].run_id == "actor_mlp_erm_geomgcn-0_seed-1"
    assert registrations[-1].run_id == (
        "blogcatalog_acm-gcn_full_gaug-public_seed-30"
    )


def test_registered_panel_can_be_filtered_without_changing_ids() -> None:
    actor = load_registered_runs(REPO_ROOT, "additional", ("actor",))
    original = load_registered_runs(REPO_ROOT, "original")

    assert len(actor) == 840
    assert {run.dataset for run in actor} == {"actor"}
    assert len(original) == 4_200
    assert {run.dataset for run in original} == set(ORIGINAL_DATASET_ORDER)


def test_exact_run_validation_rejects_missing_and_changed_identity() -> None:
    registrations = load_registered_runs(
        REPO_ROOT, "additional", ("actor",)
    )[:2]
    configs = [asdict(run) for run in registrations]

    validate_registered_run_configs(configs, registrations)
    with pytest.raises(ValueError, match="registered run ID mismatch"):
        validate_registered_run_configs(configs[:1], registrations)

    changed = [dict(config) for config in configs]
    changed[0]["seed"] = 99
    with pytest.raises(ValueError, match="identity differs"):
        validate_registered_run_configs(changed, registrations)
