"""Focused tests for the registered-cell edge-perturbation extension."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import runpy
import shutil
from uuid import uuid4

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_edge_perturbation_extension.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


@pytest.fixture
def workspace_tmp_path() -> Path:
    path = (
        REPO_ROOT
        / "artifacts"
        / "_test_edge_perturbation_extension"
        / uuid4().hex
    )
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_registered_cells_are_explicit_not_rectangular() -> None:
    records = [
        cell.as_record() for cell in NAMESPACE["REGISTERED_CELLS"]
    ]
    assert records == [
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
    ]


@pytest.mark.parametrize(
    "split_ids,seeds,expected_first,expected_last",
    [
        (
            [f"lim-{index}" for index in range(5)],
            list(range(1, 7)),
            ("lim-0", 1),
            ("lim-4", 6),
        ),
        (
            ["ogb-time"],
            list(range(1, 31)),
            ("ogb-time", 1),
            ("ogb-time", 30),
        ),
    ],
)
def test_registered_checkpoint_identities_follow_manifest(
    split_ids: list[str],
    seeds: list[int],
    expected_first: tuple[str, int],
    expected_last: tuple[str, int],
) -> None:
    manifest = {
        "splits": [{"split_id": split} for split in split_ids],
        "training_seeds": seeds,
        "n_runs": 30,
    }
    identities = NAMESPACE["registered_checkpoint_identities"](manifest)
    assert len(identities) == 30
    assert identities[0] == expected_first
    assert identities[-1] == expected_last


def test_direction_is_derived_from_registered_tail_side() -> None:
    function = NAMESPACE["desired_same_from_manifest"]
    assert function({"local_homophily": {"tail_side": "low-h_v"}}) is True
    assert function({"local_homophily": {"tail_side": "high-h_v"}}) is False
    with pytest.raises(RuntimeError, match="unknown registered tail side"):
        function({"local_homophily": {"tail_side": "middle"}})


def _write_source_pair(
    repo_root: Path, dataset: str = "pubmed", pair_index: int = 1
) -> Path:
    source = (
        repo_root
        / "artifacts"
        / "edge_perturbation"
        / "actual"
        / dataset
        / f"pair-{pair_index:04d}"
    )
    source.mkdir(parents=True)
    graphs_path = source / "graphs.npz"
    np.savez_compressed(
        graphs_path,
        treatment_edge_index=np.asarray([[0, 1], [1, 0]]),
        control_edge_index=np.asarray([[0, 2], [2, 0]]),
        targets=np.asarray([0]),
        target_order=np.asarray([0]),
    )
    swaps_path = source / "swaps.json"
    swaps_path.write_text(
        json.dumps({"treatment": [], "control": []}),
        encoding="utf-8",
    )
    file_sha256 = NAMESPACE["file_sha256"]
    metadata = {
        "phase": "actual",
        "completed": True,
        "dataset": dataset,
        "pair_index": pair_index,
        "pair_seed": 101,
        "sample_fraction": 0.5,
        "requested_targets": 1,
        "generation_attempt": 0,
        "generation_retry_failures": [],
        "audit": {"edge_count_preserved": True},
        "output_sha256": {
            "graphs.npz": file_sha256(graphs_path),
            "swaps.json": file_sha256(swaps_path),
        },
    }
    (source / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return source


def test_reused_pair_is_copied_and_has_exact_source_provenance(
    workspace_tmp_path: Path,
) -> None:
    tmp_path = workspace_tmp_path
    source = _write_source_pair(tmp_path)
    destination = tmp_path / "extension-pair"
    destination.mkdir()

    pair = NAMESPACE["copy_reused_pair"](
        source,
        destination,
        dataset="pubmed",
        pair_index=1,
        repo_root=tmp_path,
    )

    file_sha256 = NAMESPACE["file_sha256"]
    source_hashes = {
        filename: file_sha256(source / filename)
        for filename in ("graphs.npz", "swaps.json")
    }
    assert pair.graph_source == {
        "mode": "reuse-original",
        "source_pair": (
            "artifacts/edge_perturbation/actual/pubmed/pair-0001"
        ),
        "source_metadata_sha256": file_sha256(
            source / "metadata.json"
        ),
        "source_output_sha256": source_hashes,
    }
    assert file_sha256(destination / "graphs.npz") == source_hashes[
        "graphs.npz"
    ]
    assert file_sha256(destination / "swaps.json") == source_hashes[
        "swaps.json"
    ]
    assert pair.targets.tolist() == [0]
    assert pair.pair_seed == 101


def test_frozen_manifest_rejects_changed_extension_settings(
    workspace_tmp_path: Path,
) -> None:
    tmp_path = workspace_tmp_path
    path = tmp_path / "experiment_manifest.json"
    expected = {
        "schema_version": 1,
        "cells": [{"dataset": "pubmed", "model": "gcn"}],
    }
    NAMESPACE["ensure_frozen_manifest"](path, expected)
    NAMESPACE["ensure_frozen_manifest"](path, expected)

    with pytest.raises(
        RuntimeError, match="extension experiment settings differ"
    ):
        NAMESPACE["ensure_frozen_manifest"](
            path,
            {
                "schema_version": 1,
                "cells": [
                    {"dataset": "pubmed", "model": "graphsage-mean"}
                ],
            },
        )


def test_frozen_manifest_has_no_fixed_pair_count(
    workspace_tmp_path: Path,
) -> None:
    tmp_path = workspace_tmp_path
    source_root = (
        tmp_path / "artifacts" / "edge_perturbation" / "actual"
    )
    source_root.mkdir(parents=True)
    source_manifest = source_root / "experiment_manifest.json"
    source_manifest.write_text("{}", encoding="utf-8")
    args = Namespace(
        code_version="abc123",
        seed_base=2026073100,
        source_pair_root=source_root,
    )

    manifest = NAMESPACE["frozen_manifest"](args, tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["cells"] == [
        cell.as_record() for cell in NAMESPACE["REGISTERED_CELLS"]
    ]
    assert "pair_count" not in manifest


def test_actual_preflight_allows_selected_cells_only(
    workspace_tmp_path: Path,
) -> None:
    tmp_path = workspace_tmp_path
    source_root = (
        tmp_path / "artifacts" / "edge_perturbation" / "actual"
    )
    source_root.mkdir(parents=True)
    (source_root / "experiment_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    output_root = tmp_path / "artifacts" / "edge_perturbation_extension"
    unrelated_gap = (
        output_root / "actual" / "pubmed" / "pair-0005"
    )
    unrelated_gap.mkdir(parents=True)
    args = Namespace(
        code_version="abc123",
        seed_base=2026073100,
        source_pair_root=source_root,
        output_root=output_root,
        pair_indices=[1],
        datasets=["arxiv-year"],
    )

    NAMESPACE["prepare_actual_experiment"](args, tmp_path)

    manifest = json.loads(
        (output_root / "actual" / "experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["cells"] == [
        cell.as_record() for cell in NAMESPACE["REGISTERED_CELLS"]
    ]
