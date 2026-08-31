from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_pubmed_edge_dose_response.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

MODELS = NAMESPACE["MODELS"]
P_PERCENTAGES = NAMESPACE["P_PERCENTAGES"]
GRAPH_REPLICATES = NAMESPACE["GRAPH_REPLICATES"]
IDENTITIES = NAMESPACE["EXPECTED_IDENTITIES"]
CODE_VERSION = "a" * 40
DESIGN_ID = "pubmed-tail-dose-response-v1"
SEED_BASE = 2026080100


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(input_root: Path) -> str:
    input_root.mkdir(parents=True, exist_ok=True)
    path = input_root / "experiment_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design_id": DESIGN_ID,
                "code_version": CODE_VERSION,
                "seed_base": SEED_BASE,
                "dataset": "pubmed",
                "models": list(MODELS),
                "p_percentages": list(P_PERCENTAGES),
                "graph_replicates": len(GRAPH_REPLICATES),
                "graph_replicate_indices": list(GRAPH_REPLICATES),
                "replicate_seed_base": SEED_BASE,
                "metric_population": "fixed-full-tail-test",
                "tail_count": 2696,
                "actual_generation_attempts": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return file_sha256(path)


def write_baseline(input_root: Path) -> None:
    rows = []
    for model_index, model in enumerate(MODELS):
        for split, seed in IDENTITIES:
            rows.append(
                {
                    "dataset": "pubmed",
                    "model": model,
                    "split": split,
                    "seed": seed,
                    "n_tail_test": 100,
                    "tail_test_class_composition": '{"0":50,"1":50}',
                    "tail_test_macro_f1": 0.4 + 0.1 * model_index,
                    "tail_test_accuracy": 0.41 + 0.1 * model_index,
                }
            )
    baseline_dir = input_root / "baseline"
    metrics_path = baseline_dir / "checkpoint_metrics.csv"
    write_csv(metrics_path, rows)
    predictions_path = baseline_dir / "predictions.npz"
    np.savez_compressed(predictions_path, predictions=np.asarray([1]))
    metadata = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": "pubmed",
        "models": list(MODELS),
        "metric_population": "fixed-full-tail-test",
        "n_checkpoints_per_model": 30,
        "n_checkpoint_rows": 60,
        "all_original_predictions_reproduced": True,
        "checkpoint_identities": [
            {"split": split, "seed": seed}
            for split, seed in IDENTITIES
        ],
        "design_id": DESIGN_ID,
        "code_version": CODE_VERSION,
        "experiment_manifest_sha256": file_sha256(
            input_root / "experiment_manifest.json"
        ),
        "output_sha256": {
            "checkpoint_metrics.csv": file_sha256(metrics_path),
            "predictions.npz": file_sha256(predictions_path),
        },
    }
    (baseline_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def write_dose_cell(
    input_root: Path,
    *,
    replicate_index: int,
    p_percent: int,
    manifest_sha256: str,
    duplicate_identity: bool = False,
    invalid_value: bool = False,
) -> None:
    cell = (
        input_root
        / f"replicate-{replicate_index:04d}"
        / f"pct-{p_percent:03d}"
    )
    cell.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_index, model in enumerate(MODELS):
        identities = list(IDENTITIES)
        if duplicate_identity:
            identities[-1] = identities[0]
        for split, seed in identities:
            baseline = 0.4 + 0.1 * model_index
            treatment = (
                baseline + p_percent / 1000 + replicate_index / 10000
            )
            control = baseline + replicate_index / 10000
            if invalid_value and model == MODELS[0]:
                treatment = 1.1
            rows.append(
                {
                    "dataset": "pubmed",
                    "model": model,
                    "replicate_index": replicate_index,
                    "replicate_seed": SEED_BASE + replicate_index,
                    "p_percent": p_percent,
                    "n_selected_tail_nodes": p_percent,
                    "split": split,
                    "seed": seed,
                    "n_tail_test": 100,
                    "tail_test_class_composition": '{"0":50,"1":50}',
                    "treatment_macro_f1": treatment,
                    "control_macro_f1": control,
                    "treatment_minus_control_macro_f1": (
                        treatment - control
                    ),
                    "treatment_accuracy": treatment,
                    "control_accuracy": control,
                    "treatment_minus_control_accuracy": treatment - control,
                }
            )
    write_csv(cell / "checkpoint_metrics.csv", rows)
    ranking = np.roll(np.arange(2696, dtype=np.int64), replicate_index)
    target_count = (2696 * p_percent) // 100
    np.savez_compressed(
        cell / "graphs.npz",
        treatment_edge_index=np.asarray(
            [[0, 1], [1, 0]], dtype=np.int64
        ),
        control_edge_index=np.asarray(
            [[0, 2], [2, 0]], dtype=np.int64
        ),
        targets=ranking[:target_count],
        tail_ranking=ranking,
    )
    (cell / "swaps.json").write_text("{}", encoding="utf-8")
    (cell / "predictions.npz").write_bytes(
        f"{replicate_index}:{p_percent}:predictions".encode()
    )
    output_hashes = {
        filename: file_sha256(cell / filename)
        for filename in (
            "checkpoint_metrics.csv",
            "graphs.npz",
            "swaps.json",
            "predictions.npz",
        )
    }
    metadata = {
        "schema_version": 1,
        "phase": "actual",
        "completed": True,
        "dataset": "pubmed",
        "models": list(MODELS),
        "replicate_index": replicate_index,
        "replicate_seed": SEED_BASE + replicate_index,
        "p_percent": p_percent,
        "tail_count": 2696,
        "requested_targets": target_count,
        "selected_targets": target_count,
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "target_selection": (
            "prefix of the replicate's frozen random tail-node ranking"
        ),
        "graph_start": "registered original PubMed graph",
        "generation_attempt": 0,
        "generation_retry_failures": [],
        "metric_population": "fixed-full-tail-test",
        "n_checkpoints_per_model": 30,
        "n_checkpoint_rows": 60,
        "checkpoint_identities": [
            {"split": split, "seed": seed}
            for split, seed in IDENTITIES
        ],
        "design_id": DESIGN_ID,
        "code_version": CODE_VERSION,
        "seed_base": SEED_BASE,
        "experiment_manifest_sha256": manifest_sha256,
        "audit": {
            "edge_count_preserved": True,
            "treatment_degrees_preserved": True,
            "control_degrees_preserved": True,
            "treatment_homophily_step_exact": True,
            "control_target_homophily_unchanged": True,
            "original_edge_sha256": hashlib.sha256(
                b"original"
            ).hexdigest(),
            "treatment_edge_sha256": hashlib.sha256(
                f"{replicate_index}:{p_percent}:treatment".encode()
            ).hexdigest(),
            "control_edge_sha256": hashlib.sha256(
                f"{replicate_index}:{p_percent}:control".encode()
            ).hexdigest(),
        },
        "output_sha256": output_hashes,
    }
    (cell / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def write_panel(
    input_root: Path,
    *,
    omit: tuple[int, int] | None = None,
    duplicate_at: tuple[int, int] | None = None,
    invalid_at: tuple[int, int] | None = None,
) -> None:
    manifest_sha256 = write_manifest(input_root)
    write_baseline(input_root)
    for replicate_index in GRAPH_REPLICATES:
        for p_percent in P_PERCENTAGES[1:]:
            key = (replicate_index, p_percent)
            if key == omit:
                continue
            write_dose_cell(
                input_root,
                replicate_index=replicate_index,
                p_percent=p_percent,
                manifest_sha256=manifest_sha256,
                duplicate_identity=(key == duplicate_at),
                invalid_value=(key == invalid_at),
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_analyze_writes_tidy_summaries_figure_and_manifest(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    output_root = tmp_path / "analysis"
    write_panel(input_root)

    outputs = NAMESPACE["analyze"](input_root, output_root)

    assert set(outputs) == {
        "replicate_summary",
        "dose_summary",
        "figure",
        "analysis_manifest",
    }
    assert all(path.is_file() for path in outputs.values())
    replicate_rows = read_csv(outputs["replicate_summary"])
    dose_rows = read_csv(outputs["dose_summary"])
    assert len(replicate_rows) == 2 * 21 * 5 * 2
    assert len(dose_rows) == 2 * 21 * 2
    assert plt.imread(outputs["figure"]).size > 0

    p_zero = [
        row
        for row in dose_rows
        if row["model"] == "gcn"
        and row["condition"] == "treatment"
        and int(row["p_percent"]) == 0
    ]
    assert len(p_zero) == 1
    assert float(p_zero[0]["mean_macro_f1"]) == pytest.approx(0.4)
    assert float(
        p_zero[0]["sd_macro_f1_across_graph_replicates"]
    ) == pytest.approx(0.0)

    p_fifty = [
        row
        for row in dose_rows
        if row["model"] == "gcn"
        and row["condition"] == "treatment"
        and int(row["p_percent"]) == 50
    ]
    assert len(p_fifty) == 1
    assert float(p_fifty[0]["mean_macro_f1"]) == pytest.approx(0.4503)
    assert float(
        p_fifty[0]["sd_macro_f1_across_graph_replicates"]
    ) == pytest.approx(0.000158113883)

    manifest = json.loads(
        outputs["analysis_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["design"]["metric_population"] == "fixed-full-tail-test"
    assert manifest["validation"]["complete_nonzero_graph_cells"] == 100
    assert manifest["validation"]["baseline_checkpoint_rows"] == 60
    assert manifest["validation"]["nonzero_condition_scores"] == 12000
    assert set(manifest["generated_outputs_sha256"]) == {
        "replicate_summary.csv",
        "dose_response_summary.csv",
        "pubmed_edge_dose_response.png",
    }


def test_figure_has_two_model_panels_and_both_mean_curves(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_panel(input_root)
    scores, _, _, _ = NAMESPACE["load_inputs"](input_root)
    replicate_rows = NAMESPACE["build_replicate_summary"](scores)
    dose_rows = NAMESPACE["build_dose_summary"](replicate_rows)

    figure = NAMESPACE["build_figure"](replicate_rows, dose_rows)
    try:
        assert [axis.get_title() for axis in figure.axes] == [
            "PubMed · GCN",
            "PubMed · GraphSAGE-mean",
        ]
        assert all(len(axis.lines) == 12 for axis in figure.axes)
        assert all(
            axis.get_xlabel()
            == "Tail nodes selected for rewiring (%)"
            for axis in figure.axes
        )
        assert figure.axes[0].get_ylabel() == "Tail-test macro-F1"
    finally:
        plt.close(figure)


def test_missing_nonzero_cell_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_panel(input_root, omit=(5, 100))

    with pytest.raises(RuntimeError, match="Dose grid differs"):
        NAMESPACE["load_inputs"](input_root)


def test_duplicate_checkpoint_identity_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_panel(input_root, duplicate_at=(1, 5))

    with pytest.raises(RuntimeError, match="must be unique"):
        NAMESPACE["load_inputs"](input_root)


def test_macro_f1_outside_unit_interval_is_rejected(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_panel(input_root, invalid_at=(1, 5))

    with pytest.raises(RuntimeError, match=r"finite and in \[0, 1\]"):
        NAMESPACE["load_inputs"](input_root)


def test_cli_requires_explicit_input_and_output_roots() -> None:
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"]([])
