from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_edge_dose_response_full.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_edge_dose_response_full", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(value: int) -> str:
    return f"{value:064x}"


def _write_fixture(module, root: Path, *, preparation: str = "erm"):
    protocol = module.ExpansionProtocol(
        cells=(("pubmed", "acm-gcn"),),
        checkpoints_per_cell=2,
        replicates=(1, 2),
        percentages_by_dataset={"pubmed": (0, 5)},
        design_id=module.design_id_for_preparation(preparation),
        preparation=preparation,
    )
    baselines = []
    pairs = []
    for checkpoint in (1, 2):
        split = f"split-{checkpoint}"
        run_id = (
            f"pubmed_acm-gcn_{preparation}_{split}_seed-{checkpoint}"
            if preparation != "erm"
            else f"pubmed_acm-gcn_run-{checkpoint}"
        )
        baseline = {
            "dataset": "pubmed",
            "model": "acm-gcn",
            "run_id": run_id,
            "split": split,
            "seed": checkpoint,
            "checkpoint_sha256": _sha(checkpoint),
            "prediction_key": f"{run_id}__original",
            "prediction_sha256": _sha(100 + checkpoint),
            "tail_test_macro_f1": 0.4 + checkpoint * 0.1,
        }
        if preparation != "erm":
            baseline["preparation"] = preparation
        baselines.append(baseline)
        for replicate in protocol.replicates:
            treatment = 0.5 + checkpoint * 0.01 + replicate * 0.001
            control = 0.4 + checkpoint * 0.01 + replicate * 0.001
            pair_key = f"pubmed__replicate_{replicate:04d}__pct_005"
            row = {
                "dataset": "pubmed",
                "model": "acm-gcn",
                "run_id": run_id,
                "split": baseline["split"],
                "seed": baseline["seed"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "replicate_index": replicate,
                "replicate_seed": 2026080100 + replicate,
                "p_percent": 5,
                "pairing_key": pair_key,
                "source_graphs_sha256": _sha(200),
                "treatment_edge_sha256": _sha(210 + replicate),
                "control_edge_sha256": _sha(220 + replicate),
                "treatment_macro_f1": treatment,
                "control_macro_f1": control,
                "treatment_minus_control_macro_f1": treatment - control,
                "treatment_prediction_key": f"{run_id}__{pair_key}__treatment",
                "control_prediction_key": f"{run_id}__{pair_key}__control",
                "treatment_prediction_sha256": _sha(
                    300 + 10 * checkpoint + replicate
                ),
                "control_prediction_sha256": _sha(
                    400 + 10 * checkpoint + replicate
                ),
            }
            if preparation != "erm":
                row["preparation"] = preparation
            pairs.append(row)

    root.mkdir(parents=True)
    baseline_path = root / "baseline_rows.jsonl"
    paired_path = root / "paired_rows.jsonl"
    baseline_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in baselines),
        encoding="utf-8",
    )
    paired_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pairs),
        encoding="utf-8",
    )
    validation = {
        "schema_version": 1,
        "design_id": protocol.design_id,
        "validated": True,
        "n_model_dataset_cells": 1,
        "n_checkpoint_shards": 2,
        "n_baseline_rows": 2,
        "n_paired_rows": 4,
        "n_unique_treatment_prediction_keys": 4,
        "n_unique_control_prediction_keys": 4,
        "graph_catalog_sha256": {"pubmed": _sha(500)},
        "output_sha256": {
            "baseline_rows.jsonl": module.file_sha256(baseline_path),
            "paired_rows.jsonl": module.file_sha256(paired_path),
        },
    }
    if preparation != "erm":
        validation["preparation"] = preparation
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return protocol, baselines, pairs


def test_small_expansion_is_authenticated_and_aggregated(tmp_path: Path) -> None:
    module = load_module()
    protocol, _, _ = _write_fixture(module, tmp_path / "final")

    baselines, pairs, validation = module.validate_expansion_inputs(
        tmp_path / "final", protocol=protocol
    )
    replicate_rows, dose_rows = module.aggregate_expansion(
        baselines, pairs, protocol=protocol
    )

    assert len(baselines) == 2
    assert len(pairs) == 4
    assert len(replicate_rows) == 8
    assert len(dose_rows) == 4
    assert validation["treatment_control_pairing_valid"] is True
    assert set(validation["raw_input_sha256"]) == {
        "baseline_rows.jsonl",
        "paired_rows.jsonl",
        "validation.json",
    }
    p_zero = [row for row in dose_rows if row["p_percent"] == 0]
    assert {row["mean_macro_f1"] for row in p_zero} == {0.55}
    assert {row["sd_macro_f1_across_graph_replicates"] for row in p_zero} == {
        0.0
    }


@pytest.mark.parametrize("preparation", ("last", "head", "full"))
def test_recovery_protocol_is_full_matrix_and_has_separate_defaults(
    preparation: str,
) -> None:
    module = load_module()
    protocol = module.protocol_for_preparation(preparation)
    assert len(protocol.cells) == 30
    assert protocol.expected_baselines == 900
    assert protocol.expected_paired_rows == 84_600
    assert protocol.preparation == preparation
    assert protocol.design_id == (
        f"appendix-j-edge-dose-response-{preparation}-full-matrix-v1"
    )
    assert module.default_final_root(preparation) == (
        module.DEFAULT_RECOVERY_ROOT / "actual" / preparation / "final"
    )
    assert module.default_output_root(preparation) != module.DEFAULT_OUTPUT_ROOT


def test_recovery_input_rejects_cross_preparation_rows(tmp_path: Path) -> None:
    module = load_module()
    protocol, _, _ = _write_fixture(
        module, tmp_path / "final", preparation="head"
    )
    module.validate_expansion_inputs(tmp_path / "final", protocol=protocol)

    baseline_path = tmp_path / "final" / "baseline_rows.jsonl"
    rows = [json.loads(line) for line in baseline_path.read_text().splitlines()]
    rows[0]["preparation"] = "last"
    baseline_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    validation_path = tmp_path / "final" / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["output_sha256"]["baseline_rows.jsonl"] = module.file_sha256(
        baseline_path
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="preparation differs"):
        module.validate_expansion_inputs(tmp_path / "final", protocol=protocol)


def test_erm_input_rejects_an_explicit_other_preparation(tmp_path: Path) -> None:
    module = load_module()
    protocol, _, _ = _write_fixture(module, tmp_path / "final")
    validation_path = tmp_path / "final" / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["preparation"] = "head"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="preparation differs"):
        module.validate_expansion_inputs(tmp_path / "final", protocol=protocol)


def test_hash_mismatch_is_rejected_before_aggregation(tmp_path: Path) -> None:
    module = load_module()
    protocol, _, _ = _write_fixture(module, tmp_path / "final")
    paired_path = tmp_path / "final" / "paired_rows.jsonl"
    paired_path.write_text(
        paired_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hashes do not match"):
        module.validate_expansion_inputs(
            tmp_path / "final", protocol=protocol
        )


def test_csv_float_serialization_normalizes_one_ulp_differences(
    tmp_path: Path,
) -> None:
    module = load_module()
    output = tmp_path / "summary.csv"

    module._write_csv(
        output,
        [
            {"mean_macro_f1": 0.00445750974897643},
            {"mean_macro_f1": 0.0044575097489764305},
            {"mean_macro_f1": 0.55},
        ],
        ("mean_macro_f1",),
    )

    assert output.read_text(encoding="utf-8") == (
        "mean_macro_f1\n"
        "0.00445750974898\n"
        "0.00445750974898\n"
        "0.55000000000000\n"
    )


def test_tracked_full_summary_manifest_matches_compact_outputs() -> None:
    module = load_module()
    output_root = module.DEFAULT_OUTPUT_ROOT
    manifest = json.loads(
        (output_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["validation"]["raw_checkpoint_shards"] == 720
    assert manifest["validation"]["raw_paired_rows"] == 66600
    assert manifest["validation"]["complete_model_dataset_cells"] == 30
    assert manifest["validation"]["dose_response_summary_rows"] == 1188
    assert manifest["validation"]["roman_empire_max_percentage"] == 70
    expansion = manifest["sources"]["expansion"]
    assert expansion["experiment_code_version"] == module.EXPANSION_CODE_VERSION
    assert set(expansion["input_sha256"]) == {
        "baseline_rows.jsonl",
        "paired_rows.jsonl",
        "validation.json",
    }
    assert all(module._is_sha256(value) for value in expansion["input_sha256"].values())
    for filename, expected in manifest["generated_outputs_sha256"].items():
        assert module.file_sha256(output_root / filename) == expected
