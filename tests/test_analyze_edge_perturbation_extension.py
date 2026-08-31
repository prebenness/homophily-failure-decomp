from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_edge_perturbation_extension.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

CELLS = (
    {
        "dataset": "pubmed",
        "model": "gcn",
        "graph_mode": "reuse-original",
    },
    {
        "dataset": "roman-empire",
        "model": "gcn",
        "graph_mode": "reuse-original",
    },
    {
        "dataset": "arxiv-year",
        "model": "graphsage-mean",
        "graph_mode": "generate",
    },
    {
        "dataset": "ogbn-arxiv",
        "model": "graphsage-mean",
        "graph_mode": "generate",
    },
)
CODE_VERSION = "a" * 40
DESIGN_ID = "tail-half-one-degree-preserving-swap-v1-extension"
SEED_BASE = 2026073100
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


def write_manifest(input_root: Path, cells=CELLS) -> str:
    input_root.mkdir(parents=True, exist_ok=True)
    path = input_root / "experiment_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design_id": DESIGN_ID,
                "code_version": CODE_VERSION,
                "seed_base": SEED_BASE,
                "sample_fraction": 0.5,
                "cells": list(cells),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return file_sha256(path)


def write_pair(
    input_root: Path,
    cell: dict[str, str],
    pair_index: int,
    *,
    effect: float | None = None,
    phase: str = "actual",
    completed: bool = True,
    graph_source_override: dict | None = None,
) -> Path:
    manifest_sha256 = file_sha256(input_root / "experiment_manifest.json")
    dataset = cell["dataset"]
    model = cell["model"]
    pair_dir = input_root / dataset / f"pair-{pair_index:04d}"
    pair_dir.mkdir(parents=True)

    identities = NAMESPACE["EXPECTED_IDENTITIES"][dataset]
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "pair_index": pair_index,
            "split": split,
            "seed": seed,
            "treatment_minus_control_macro_f1": (
                effect if effect is not None else pair_index / 100
            ),
            "n_sampled_test": 100 + row_index,
        }
        for row_index, (split, seed) in enumerate(identities)
    ]
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
    if cell["graph_mode"] == "reuse-original":
        graph_source = {
            "mode": "reuse-original",
            "source_pair": (
                "artifacts/edge_perturbation/actual/"
                f"{dataset}/pair-{pair_index:04d}"
            ),
            "source_metadata_sha256": text_sha256(
                f"{dataset}:{pair_index}:source-metadata"
            ),
            "source_output_sha256": {
                filename: output_sha256[filename]
                for filename in NAMESPACE["REUSED_SOURCE_FILES"]
            },
        }
    else:
        graph_source = {"mode": "generate"}
    if graph_source_override is not None:
        graph_source = graph_source_override

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
    metadata = {
        "schema_version": 1,
        "phase": phase,
        "completed": completed,
        "dataset": dataset,
        "model": model,
        "graph_mode": cell["graph_mode"],
        "pair_index": pair_index,
        "pair_seed": (
            SEED_BASE + pair_index
            if cell["graph_mode"] == "generate"
            else 2026073000 + pair_index
        ),
        "n_checkpoints": 30,
        "n_checkpoint_rows": 30,
        "sample_fraction": 0.5,
        "all_original_predictions_reproduced": True,
        "code_version": CODE_VERSION,
        "design_id": DESIGN_ID,
        "seed_base": SEED_BASE,
        "experiment_manifest_sha256": manifest_sha256,
        "graph_source": graph_source,
        "audit": audit,
        "output_sha256": output_sha256,
    }
    (pair_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    return pair_dir


def write_panel(input_root: Path, pair_counts: dict[str, int]) -> None:
    write_manifest(input_root)
    for cell in CELLS:
        for pair_index in range(1, pair_counts[cell["dataset"]] + 1):
            write_pair(
                input_root,
                cell,
                pair_index,
                effect=pair_index / 100,
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_analyze_writes_all_outputs_for_unequal_pair_counts(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    output_root = tmp_path / "analysis"
    pair_counts = {
        "pubmed": 2,
        "roman-empire": 1,
        "ogbn-arxiv": 3,
        "arxiv-year": 2,
    }
    write_panel(input_root, pair_counts)

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
    assert len(checkpoint_rows) == sum(pair_counts.values()) * 30
    assert len(pair_rows) == sum(pair_counts.values())
    assert len(running_rows) == sum(pair_counts.values())
    assert "n_sampled_test" in checkpoint_rows[0]
    assert plt.imread(outputs["figure"]).size > 0

    ogbn_running = [
        row for row in running_rows if row["dataset"] == "ogbn-arxiv"
    ]
    assert [int(row["n_pairs"]) for row in ogbn_running] == [1, 2, 3]
    assert [
        float(row["running_mean_treatment_minus_control_macro_f1"])
        for row in ogbn_running
    ] == pytest.approx([0.01, 0.015, 0.02])

    manifest = json.loads(
        outputs["analysis_manifest"].read_text(encoding="utf-8")
    )
    cells = {
        (row["dataset"], row["model"]): row
        for row in manifest["scope"]["cells"]
    }
    assert cells[("pubmed", "gcn")]["pair_indices"] == [1, 2]
    assert cells[("roman-empire", "gcn")]["pair_indices"] == [1]
    assert cells[("ogbn-arxiv", "graphsage-mean")]["pair_indices"] == [1, 2, 3]
    assert cells[("arxiv-year", "graphsage-mean")]["pair_indices"] == [1, 2]
    assert manifest["scope"]["checkpoint_rows"] == sum(pair_counts.values()) * 30
    assert set(manifest["generated_outputs_sha256"]) == {
        "checkpoint_metrics.csv",
        "pair_summary.csv",
        "running_summary.csv",
        "running_convergence.png",
    }


def test_convergence_figure_has_one_panel_per_registered_cell(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_panel(
        input_root,
        {
            "pubmed": 1,
            "roman-empire": 1,
            "ogbn-arxiv": 1,
            "arxiv-year": 1,
        },
    )
    records, _, _, contract, _ = NAMESPACE["load_records"](input_root)
    pair_rows = NAMESPACE["build_pair_summary"](records, contract)
    running_rows = NAMESPACE["build_running_summary"](pair_rows, contract)

    figure = NAMESPACE["build_convergence_figure"](
        pair_rows, running_rows, contract
    )
    try:
        assert [axis.get_title() for axis in figure.axes] == [
            "PubMed · GCN",
            "Roman-empire · GCN",
            "arxiv-year · GraphSAGE-mean",
            "ogbn-arxiv · GraphSAGE-mean",
        ]
        assert all(len(axis.collections) == 1 for axis in figure.axes)
    finally:
        plt.close(figure)


def test_each_registered_cell_must_have_a_completed_pair(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_manifest(input_root)
    for cell in CELLS[:-1]:
        write_pair(input_root, cell, 1)

    with pytest.raises(RuntimeError, match="have no completed pairs"):
        NAMESPACE["load_records"](input_root)


def test_unregistered_dataset_pair_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    registered_cells = CELLS[:1]
    write_manifest(input_root, registered_cells)
    write_pair(input_root, registered_cells[0], 1)
    unregistered = {
        "dataset": "roman-empire",
        "model": "gcn",
        "graph_mode": "reuse-original",
    }
    write_pair(input_root, unregistered, 1)

    with pytest.raises(RuntimeError, match="Unexpected dataset directory"):
        NAMESPACE["load_records"](input_root)


def test_pair_indices_are_contiguous_separately_within_dataset(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_manifest(input_root, CELLS[:1])
    write_pair(input_root, CELLS[0], 2)

    with pytest.raises(RuntimeError, match="contiguous beginning at 1"):
        NAMESPACE["load_records"](input_root)


def test_reused_copies_must_match_recorded_source_hashes(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "actual"
    write_manifest(input_root, CELLS[:1])
    bad_source = {
        "mode": "reuse-original",
        "source_pair": "artifacts/edge_perturbation/actual/pubmed/pair-0001",
        "source_metadata_sha256": "b" * 64,
        "source_output_sha256": {
            "graphs.npz": "c" * 64,
            "swaps.json": "d" * 64,
        },
    }
    write_pair(
        input_root,
        CELLS[0],
        1,
        graph_source_override=bad_source,
    )

    with pytest.raises(RuntimeError, match="does not match its source hash"):
        NAMESPACE["load_records"](input_root)


def test_generated_graph_source_has_no_reuse_fields(tmp_path: Path) -> None:
    input_root = tmp_path / "actual"
    generated_cell = CELLS[2]
    write_manifest(input_root, (generated_cell,))
    write_pair(
        input_root,
        generated_cell,
        1,
        graph_source_override={
            "mode": "generate",
            "source_pair": "unexpected",
        },
    )

    with pytest.raises(RuntimeError, match="may contain only mode"):
        NAMESPACE["load_records"](input_root)


def test_cli_requires_explicit_separate_roots() -> None:
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"]([])
