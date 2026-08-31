from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_edge_perturbation.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

DATASETS = ("pubmed", "roman-empire")
MODELS = ("graphsage-mean", "acm-gcn")
SPLITS = {
    "pubmed": tuple(f"geomgcn-{index}" for index in range(10)),
    "roman-empire": tuple(f"platonov-{index}" for index in range(10)),
}
CODE_VERSION = "a" * 40
DESIGN_ID = "tail-half-one-swap-v1"
SEED_BASE = 2026073000
BOOLEAN_AUDITS = (
    "edge_count_preserved",
    "treatment_degrees_preserved",
    "control_degrees_preserved",
    "treatment_homophily_step_exact",
    "control_target_homophily_unchanged",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ensure_experiment_manifest(input_root: Path) -> str:
    path = input_root / "experiment_manifest.json"
    if not path.exists():
        input_root.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "design_id": DESIGN_ID,
                    "code_version": CODE_VERSION,
                    "seed_base": SEED_BASE,
                    "sample_fraction": 0.5,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return file_sha256(path)


def write_pair(
    input_root: Path,
    dataset: str,
    pair_index: int,
    *,
    effects: dict[str, float] | None = None,
    phase: str = "actual",
    completed: bool = True,
    rows_per_model: int = 30,
    duplicate_last_identity: bool = False,
    unexpected_last_identity: bool = False,
    metadata_overrides: dict | None = None,
    audit_overrides: dict | None = None,
) -> Path:
    experiment_manifest_sha256 = ensure_experiment_manifest(input_root)
    pair_dir = input_root / dataset / f"pair-{pair_index:04d}"
    pair_dir.mkdir(parents=True)
    effects = effects or {
        "graphsage-mean": 0.01 * pair_index,
        "acm-gcn": -0.01 * pair_index,
    }
    rows = []
    for model in MODELS:
        for row_index in range(rows_per_model):
            split_index, seed_offset = divmod(row_index, 3)
            split = SPLITS[dataset][split_index]
            seed = seed_offset + 1
            if duplicate_last_identity and row_index == rows_per_model - 1:
                split = SPLITS[dataset][0]
                seed = 1
            if unexpected_last_identity and row_index == rows_per_model - 1:
                split = (
                    "geomgcn-10"
                    if dataset == "pubmed"
                    else "platonov-10"
                )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "pair_index": pair_index,
                    "split": split,
                    "seed": seed,
                    "treatment_minus_control_macro_f1": effects[model],
                    "n_target_test": 100 + split_index,
                }
            )
    with (pair_dir / "checkpoint_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (pair_dir / "graphs.npz").write_bytes(
        f"{dataset}:{pair_index}:graphs".encode()
    )
    (pair_dir / "swaps.json").write_text(
        json.dumps({"dataset": dataset, "pair_index": pair_index}),
        encoding="utf-8",
    )
    (pair_dir / "predictions.npz").write_bytes(
        f"{dataset}:{pair_index}:predictions".encode()
    )
    output_sha256 = {
        filename: file_sha256(pair_dir / filename)
        for filename in NAMESPACE["PAIR_OUTPUT_FILES"]
    }
    audit = {name: True for name in BOOLEAN_AUDITS}
    audit.update(
        {
            "original_edge_sha256": text_sha256(f"{dataset}:original"),
            "treatment_edge_sha256": text_sha256(
                f"{dataset}:{pair_index}:treatment"
            ),
            "control_edge_sha256": text_sha256(
                f"{dataset}:{pair_index}:control"
            ),
        }
    )
    if audit_overrides:
        audit.update(audit_overrides)
    metadata = {
        "phase": phase,
        "completed": completed,
        "dataset": dataset,
        "pair_index": pair_index,
        "models": list(MODELS),
        "checkpoints_per_model": 30,
        "n_checkpoint_rows": len(rows),
        "sample_fraction": 0.5,
        "all_original_predictions_reproduced": True,
        "code_version": CODE_VERSION,
        "design_id": DESIGN_ID,
        "seed_base": SEED_BASE,
        "experiment_manifest_sha256": experiment_manifest_sha256,
        "audit": audit,
        "output_sha256": output_sha256,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (pair_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    return pair_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_complete_panel(input_root: Path) -> None:
    for dataset in DATASETS:
        write_pair(
            input_root,
            dataset,
            1,
            effects={"graphsage-mean": 0.01, "acm-gcn": -0.02},
        )
        write_pair(
            input_root,
            dataset,
            2,
            effects={"graphsage-mean": 0.03, "acm-gcn": 0.02},
        )


def test_analyze_writes_checkpoint_pair_running_and_figure_outputs(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    output_root = tmp_path / "analysis"
    write_complete_panel(input_root)

    outputs = NAMESPACE["analyze"](input_root, output_root)

    assert set(outputs) == {
        "checkpoint_metrics",
        "pair_summary",
        "running_summary",
        "figure",
        "analysis_manifest",
    }
    assert all(path.is_file() for path in outputs.values())
    checkpoint_rows = read_csv(outputs["checkpoint_metrics"])
    pair_rows = read_csv(outputs["pair_summary"])
    running_rows = read_csv(outputs["running_summary"])
    assert len(checkpoint_rows) == 2 * 2 * 2 * 30
    assert len(pair_rows) == 2 * 2 * 2
    assert len(running_rows) == 2 * 2 * 2
    assert "n_target_test" in checkpoint_rows[0]

    pubmed_sage_pairs = [
        row
        for row in pair_rows
        if row["dataset"] == "pubmed"
        and row["model"] == "graphsage-mean"
    ]
    assert [int(row["n_checkpoints"]) for row in pubmed_sage_pairs] == [30, 30]
    assert [
        float(row["mean_treatment_minus_control_macro_f1"])
        for row in pubmed_sage_pairs
    ] == pytest.approx([0.01, 0.03])

    pubmed_sage_running = [
        row
        for row in running_rows
        if row["dataset"] == "pubmed"
        and row["model"] == "graphsage-mean"
    ]
    assert [int(row["n_pairs"]) for row in pubmed_sage_running] == [1, 2]
    assert [
        float(row["running_mean_treatment_minus_control_macro_f1"])
        for row in pubmed_sage_running
    ] == pytest.approx([0.01, 0.02])
    assert plt.imread(outputs["figure"]).size > 0

    analysis_manifest = json.loads(
        outputs["analysis_manifest"].read_text(encoding="utf-8")
    )
    assert analysis_manifest["scope"]["pair_indices"] == [1, 2]
    assert analysis_manifest["scope"]["checkpoint_rows"] == 240
    assert len(analysis_manifest["included_pairs"]) == 4
    assert len(analysis_manifest["input_hashes"]) == 4
    assert analysis_manifest["experiment_manifest"]["sha256"] == file_sha256(
        input_root / "experiment_manifest.json"
    )
    assert set(analysis_manifest["generated_outputs_sha256"]) == {
        "checkpoint_metrics.csv",
        "pair_summary.csv",
        "running_summary.csv",
        "running_convergence.png",
    }


def test_convergence_figure_has_four_panels(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_complete_panel(input_root)
    records, _, _, _ = NAMESPACE["load_records"](input_root)
    pair_rows = NAMESPACE["build_pair_summary"](records)
    running_rows = NAMESPACE["build_running_summary"](pair_rows)

    figure = NAMESPACE["build_convergence_figure"](pair_rows, running_rows)
    try:
        assert len(figure.axes) == 4
        assert [axis.get_title() for axis in figure.axes] == [
            "PubMed · GraphSAGE-mean",
            "PubMed · ACM-GCN",
            "Roman-empire · GraphSAGE-mean",
            "Roman-empire · ACM-GCN",
        ]
        assert all(len(axis.collections) == 1 for axis in figure.axes)
    finally:
        plt.close(figure)


def test_debug_pair_is_rejected_before_aggregation(tmp_path: Path) -> None:
    input_root = tmp_path / "debug"
    write_pair(input_root, "pubmed", 1, phase="debug")

    with pytest.raises(RuntimeError, match="phase must be 'actual'"):
        NAMESPACE["load_records"](input_root)


def test_uncompleted_actual_pair_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_pair(input_root, "pubmed", 1, completed=False)

    with pytest.raises(RuntimeError, match="completed must be true"):
        NAMESPACE["load_records"](input_root)


def test_checkpoint_cell_requires_exactly_thirty_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_pair(
        input_root,
        "pubmed",
        1,
        rows_per_model=29,
        metadata_overrides={"n_checkpoint_rows": 60},
    )

    with pytest.raises(RuntimeError, match="expected 30 checkpoint rows, found 29"):
        NAMESPACE["load_records"](input_root)


def test_checkpoint_split_seed_identities_must_be_unique(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_pair(
        input_root,
        "pubmed",
        1,
        duplicate_last_identity=True,
    )

    with pytest.raises(RuntimeError, match="split/seed rows must be unique"):
        NAMESPACE["load_records"](input_root)


def test_checkpoint_identities_must_match_the_registered_panel(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_pair(
        input_root,
        "pubmed",
        1,
        unexpected_last_identity=True,
    )

    with pytest.raises(RuntimeError, match="registered 10x3 panel"):
        NAMESPACE["load_records"](input_root)


def test_pair_indices_must_be_contiguous_from_one(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    for dataset in DATASETS:
        write_pair(input_root, dataset, 2)

    with pytest.raises(RuntimeError, match="contiguous beginning at 1"):
        NAMESPACE["load_records"](input_root)


def test_all_four_cells_must_have_identical_pair_indices(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_pair(input_root, "pubmed", 1)
    write_pair(input_root, "pubmed", 2)
    write_pair(input_root, "roman-empire", 1)

    with pytest.raises(RuntimeError, match="identical across all four"):
        NAMESPACE["load_records"](input_root)


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"models": list(reversed(MODELS))}, "models must be"),
        ({"checkpoints_per_model": 29}, "checkpoints_per_model must be 30"),
        ({"n_checkpoint_rows": 59}, "n_checkpoint_rows must be 60"),
        ({"sample_fraction": 0.25}, "sample_fraction must be 0.5"),
        (
            {"all_original_predictions_reproduced": False},
            "all_original_predictions_reproduced must be true",
        ),
    ],
)
def test_required_metadata_contract_is_enforced(
    tmp_path: Path,
    metadata_overrides: dict,
    message: str,
) -> None:
    input_root = tmp_path / "actual"
    write_pair(
        input_root,
        "pubmed",
        1,
        metadata_overrides=metadata_overrides,
    )

    with pytest.raises(RuntimeError, match=message):
        NAMESPACE["load_records"](input_root)


def test_all_boolean_graph_audits_must_pass(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    write_pair(
        input_root,
        "pubmed",
        1,
        audit_overrides={"edge_count_preserved": False},
    )

    with pytest.raises(RuntimeError, match="graph audits must all be true"):
        NAMESPACE["load_records"](input_root)


def test_pair_output_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    pair_dir = write_pair(input_root, "pubmed", 1)
    (pair_dir / "swaps.json").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output_sha256 mismatch for swaps.json"):
        NAMESPACE["load_records"](input_root)


def test_experiment_manifest_hash_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_pair(input_root, "pubmed", 1)
    (input_root / "experiment_manifest.json").write_text(
        json.dumps({"changed": True}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="experiment_manifest_sha256"):
        NAMESPACE["load_records"](input_root)


def test_design_metadata_must_be_consistent_across_pairs(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    for dataset in DATASETS:
        write_pair(input_root, dataset, 1)
        write_pair(
            input_root,
            dataset,
            2,
            metadata_overrides=(
                {"design_id": "changed-design"}
                if dataset == "roman-empire"
                else None
            ),
        )

    with pytest.raises(RuntimeError, match="must be identical"):
        NAMESPACE["load_records"](input_root)


def test_rewired_graph_hashes_must_be_unique(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    duplicate = text_sha256("pubmed:1:treatment")
    write_pair(input_root, "pubmed", 1)
    write_pair(
        input_root,
        "pubmed",
        2,
        audit_overrides={"treatment_edge_sha256": duplicate},
    )
    write_pair(input_root, "roman-empire", 1)
    write_pair(input_root, "roman-empire", 2)

    with pytest.raises(RuntimeError, match="Rewired graph hash is reused"):
        NAMESPACE["load_records"](input_root)


def test_cli_requires_explicit_input_and_output_roots() -> None:
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"]([])
