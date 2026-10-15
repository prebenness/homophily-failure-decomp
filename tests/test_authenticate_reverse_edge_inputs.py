"""Tests for the four-preparation input inventory and checkpoint binding."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "authenticate_reverse_edge_inputs.py"
sys.path.insert(0, str(SCRIPT.parent))
API = runpy.run_path(str(SCRIPT))


def _inventory() -> dict:
    runs = [
        {
            **item,
            "files": {
                name: {"bytes": 1, "sha256": "a" * 64}
                for name in API["RUN_FILES"]
            },
            "training_code_version": "training-code",
        }
        for item in API["expected_runs"]()
    ]
    return {
        "schema_version": 1,
        "design_id": API["DESIGN_ID"],
        "authenticated": True,
        "preparations": list(API["base"].PREPARATIONS),
        "models": list(API["base"].FULL_MODELS),
        "run_record_files": list(API["RUN_FILES"]),
        "run_count": 3600,
        "datasets": {dataset: {} for dataset in API["base"].DATASET_SPECS},
        "runs": runs,
    }


def test_expected_inventory_has_nine_hundred_runs_per_preparation() -> None:
    payload = _inventory()
    API["validate_inventory"](payload)
    assert len(payload["runs"]) == 3600
    assert len({row["run_id"] for row in payload["runs"]}) == 3600
    assert Counter(row["preparation"] for row in payload["runs"]) == {
        "erm": 900, "last": 900, "head": 900, "full": 900,
    }
    combinations = Counter(
        (row["preparation"], row["dataset"], row["model"])
        for row in payload["runs"]
    )
    assert len(combinations) == 120
    assert set(combinations.values()) == {30}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-preparation", "wrong-seed"])
def test_inventory_rejects_incomplete_duplicate_or_misidentified_runs(mutation: str) -> None:
    payload = _inventory()
    if mutation == "missing":
        payload["runs"].pop()
    elif mutation == "duplicate":
        payload["runs"][-1] = deepcopy(payload["runs"][0])
    elif mutation == "wrong-preparation":
        payload["runs"][0]["preparation"] = "head"
    else:
        payload["runs"][0]["seed"] = 999
    with pytest.raises(RuntimeError):
        API["validate_inventory"](payload)


@pytest.mark.parametrize("mutation", ["missing-file", "bad-sha", "empty-file", "extra-file"])
def test_inventory_requires_result_files_with_valid_hashes_and_sizes(mutation: str) -> None:
    payload = _inventory()
    files = payload["runs"][0]["files"]
    if mutation == "missing-file":
        del files["checkpoint.pt"]
    elif mutation == "extra-file":
        files["unregistered.npz"] = {"bytes": 1, "sha256": "a" * 64}
    elif mutation == "bad-sha":
        files["checkpoint.pt"]["sha256"] = "z" * 64
    else:
        files["checkpoint.pt"]["bytes"] = 0
    with pytest.raises(RuntimeError):
        API["validate_inventory"](payload)


def test_checkpoint_inference_inventory_accepts_missing_training_history() -> None:
    payload = _inventory()
    assert set(payload["runs"][0]["files"]) == {
        "checkpoint.pt", "config.json", "run_data.npz", "metrics.json",
    }
    API["validate_inventory"](payload)


def test_optional_training_history_is_still_authenticated() -> None:
    payload = _inventory()
    history = {"bytes": 2, "sha256": "b" * 64}
    payload["runs"][0]["files"]["history.json"] = history
    API["validate_inventory"](payload)
    history["sha256"] = "invalid"
    with pytest.raises(RuntimeError, match="Invalid input hash"):
        API["validate_inventory"](payload)


def _one_output(tmp_path: Path, source: dict, *, mutation: str) -> None:
    payload = {
        **{key: source[key] for key in ("dataset", "preparation", "model", "run_id")},
        "checkpoint_sha256": source["files"]["checkpoint.pt"]["sha256"],
        "completed": True,
        "phase": "actual",
        "pairs": [{"checkpoint_sha256": "a" * 64}],
    }
    if mutation == "checkpoint":
        payload["checkpoint_sha256"] = "b" * 64
    elif mutation == "pair-checkpoint":
        payload["pairs"][0]["checkpoint_sha256"] = "b" * 64
    elif mutation == "preparation":
        payload["preparation"] = "head"
    elif mutation == "smoke":
        payload["phase"] = "smoke"
    else:
        payload["completed"] = False
    path = (
        tmp_path / "actual" / source["preparation"] / source["dataset"]
        / source["model"] / "shards" / f"{source['run_id']}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("mutation", ["checkpoint", "pair-checkpoint", "preparation", "smoke", "incomplete"])
def test_output_inventory_rejects_wrong_checkpoint_or_preparation_and_nonactual_outputs(
    tmp_path: Path, mutation: str,
) -> None:
    inventory = _inventory()
    _one_output(tmp_path, inventory["runs"][0], mutation=mutation)
    with pytest.raises(RuntimeError):
        API["verify_outputs"](inventory, tmp_path)
