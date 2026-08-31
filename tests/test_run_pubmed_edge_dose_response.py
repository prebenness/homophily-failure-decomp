"""Focused tests for the PubMed edge dose-response runner."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import runpy

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_pubmed_edge_dose_response.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_target_counts_use_registered_floor_rule() -> None:
    count = NAMESPACE["target_count_for_percentage"]
    assert count(2696, 0) == 0
    assert count(2696, 5) == 134
    assert count(2696, 50) == 1348
    assert count(2696, 100) == 2696
    with pytest.raises(ValueError, match="p_percent"):
        count(2696, 7)


def test_replicate_ranking_gives_nested_target_prefixes() -> None:
    tail_mask = np.zeros(20, dtype=bool)
    tail_mask[[1, 3, 5, 7, 9, 11, 13, 15]] = True
    ranking = NAMESPACE["tail_ranking"](tail_mask, 101)
    repeated = NAMESPACE["tail_ranking"](tail_mask, 101)
    targets_25 = NAMESPACE["targets_for_percentage"](ranking, 25)
    targets_50 = NAMESPACE["targets_for_percentage"](ranking, 50)
    targets_100 = NAMESPACE["targets_for_percentage"](ranking, 100)

    assert np.array_equal(ranking, repeated)
    assert set(ranking) == set(np.flatnonzero(tail_mask))
    assert np.array_equal(targets_25, targets_50[: targets_25.size])
    assert np.array_equal(targets_50, targets_100[: targets_50.size])
    assert targets_100.size == 8
    assert set(targets_100) == set(np.flatnonzero(tail_mask))


def test_tail_test_indices_use_the_complete_fixed_tail_test_set() -> None:
    arrays = {
        "geomgcn-0.test": np.asarray([0, 1, 2, 3, 4]),
        "tail_mask": np.asarray([True, False, True, True, False]),
    }
    indices = NAMESPACE["tail_test_indices"](arrays, "geomgcn-0")
    assert indices.tolist() == [0, 2, 3]


def test_frozen_manifest_records_registered_dose_response_design() -> None:
    args = Namespace(code_version="abc123", seed_base=2026080100)
    manifest = NAMESPACE["frozen_manifest"](args)

    assert manifest["models"] == ["gcn", "graphsage-mean"]
    assert manifest["p_percentages"] == list(range(0, 101, 5))
    assert manifest["graph_replicates"] == 5
    assert manifest["graph_replicate_indices"] == [1, 2, 3, 4, 5]
    assert manifest["tail_count"] == 2696
    assert manifest["target_sets"] == (
        "nested prefixes of the replicate target ranking"
    )
    assert manifest["graph_start_at_each_p"] == (
        "registered original PubMed graph"
    )
    assert manifest["metric_population"] == "fixed-full-tail-test"
    assert manifest["actual_generation_attempts"] == 1


def test_pair_output_path_encodes_replicate_and_percentage() -> None:
    root = Path("output") / "actual"
    replicate_index = 4
    p_percent = 35
    expected = (
        root
        / f"replicate-{replicate_index:04d}"
        / f"pct-{p_percent:03d}"
    )
    assert expected.as_posix().endswith(
        "actual/replicate-0004/pct-035"
    )
