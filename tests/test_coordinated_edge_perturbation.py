"""Tests for coordinated low-homophily edge perturbations."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hfd.coordinated_edge_perturbation import (
    generate_coordinated_rewire_pair,
)
from hfd.edge_perturbation import PairGenerationError
from hfd.homophily import local_homophily


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(edges, dtype=np.int64).T


def _undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in edge_index.T
        if a != b
    }


def _degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    edges = np.asarray(list(_undirected_edges(edge_index)), dtype=np.int64)
    if not edges.size:
        return np.zeros(num_nodes, dtype=np.int64)
    return np.bincount(edges.ravel(), minlength=num_nodes)


def _assert_simple_bidirectional(edge_index: np.ndarray) -> None:
    directed = [tuple(map(int, edge)) for edge in edge_index.T]
    directed_set = set(directed)
    assert all(a != b for a, b in directed)
    assert len(directed) == len(directed_set)
    assert all((b, a) in directed_set for a, b in directed)


def _paired_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Targets 0 and 1 share label 0.  Their cross neighbours have different
    # labels, so treatment can use (0,2),(1,3) -> (0,1),(2,3).  Node 2 is an
    # unselected tail node and remains cross-connected in both graphs.
    labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    edge_index = _edge_index([(0, 2), (1, 3)])
    tail_mask = np.asarray([True, True, True, False])
    return edge_index, labels, tail_mask


def _singleton_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Target 0 has one cross neighbour, tail node 1.  Non-tail edge (2,3)
    # supplies the treatment donor and (4,5) supplies the neutral control.
    labels = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int64)
    edge_index = _edge_index([(0, 1), (2, 3), (4, 5)])
    tail_mask = np.asarray([True, True, False, False, False, False])
    return edge_index, labels, tail_mask


def test_same_label_pair_has_exact_treatment_and_neutral_control():
    edge_index, labels, tail_mask = _paired_fixture()
    pair = generate_coordinated_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        target_nodes=np.asarray([1, 0]),
        pair_seed=101,
        max_attempts=2,
    )
    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )
    control_h = local_homophily(
        pair.control_edge_index, labels, labels.size
    )

    assert pair.targets.tolist() == [0, 1]
    assert sorted(pair.target_order.tolist()) == [0, 1]
    assert treatment_h[0] - original_h[0] == 1.0
    assert treatment_h[1] - original_h[1] == 1.0
    assert treatment_h[2] == original_h[2]
    assert np.array_equal(control_h[tail_mask], original_h[tail_mask])
    assert pair.audit["paired_swap_count"] == 1
    assert pair.audit["singleton_swap_count"] == 0
    assert pair.audit["treatment_swap_count"] == 1
    assert pair.audit["control_swap_count"] == 1

    records = pair.swap_records()
    assert records["treatment"][0]["mode"] == "paired"
    assert sorted(records["treatment"][0]["targets"]) == [0, 1]
    json.dumps(records)


def test_singleton_uses_non_tail_donor_and_preserves_other_tail_h():
    edge_index, labels, tail_mask = _singleton_fixture()
    pair = generate_coordinated_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        target_nodes=np.asarray([0]),
        pair_seed=103,
        max_attempts=2,
    )
    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )
    control_h = local_homophily(
        pair.control_edge_index, labels, labels.size
    )

    assert treatment_h[0] - original_h[0] == 1.0
    assert treatment_h[1] == original_h[1]
    assert np.array_equal(control_h[tail_mask], original_h[tail_mask])
    assert pair.audit["paired_swap_count"] == 0
    assert pair.audit["singleton_swap_count"] == 1
    for condition in ("treatment", "control"):
        record = pair.swap_records()[condition][0]
        assert record["mode"] == "singleton"
        donor_nodes = set(record["removed_edges"][1])
        assert not any(tail_mask[node] for node in donor_nodes)


def test_incompatible_even_pair_falls_back_to_two_singletons():
    # Both targets see only class 1, so joining their source neighbours would
    # create a same-label edge and alter those unselected tail nodes.  Two
    # non-tail singleton donors give the exact construction instead.
    labels = np.asarray(
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 1, 2], dtype=np.int64
    )
    edge_index = _edge_index(
        [(0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    )
    tail_mask = np.zeros(labels.size, dtype=bool)
    tail_mask[[0, 1, 2, 3]] = True

    pair = generate_coordinated_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        target_nodes=np.asarray([0, 1]),
        pair_seed=105,
        max_attempts=2,
    )
    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )

    assert pair.audit["paired_swap_count"] == 0
    assert pair.audit["singleton_swap_count"] == 2
    assert np.array_equal(treatment_h[[0, 1]], np.asarray([1.0, 1.0]))
    assert np.array_equal(treatment_h[[2, 3]], original_h[[2, 3]])


def test_graph_invariants_and_required_audit_fields():
    edge_index, labels, tail_mask = _singleton_fixture()
    pair = generate_coordinated_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        target_nodes=np.asarray([0]),
        pair_seed=107,
    )
    original_edges = _undirected_edges(edge_index)
    original_degree = _degree(edge_index, labels.size)
    for rewired in (pair.treatment_edge_index, pair.control_edge_index):
        assert len(_undirected_edges(rewired)) == len(original_edges)
        assert np.array_equal(_degree(rewired, labels.size), original_degree)
        _assert_simple_bidirectional(rewired)

    required = {
        "edge_count_preserved",
        "treatment_degrees_preserved",
        "control_degrees_preserved",
        "treatment_homophily_step_exact",
        "control_target_homophily_unchanged",
        "num_nodes",
        "num_undirected_edges",
        "num_targets",
        "original_mean_target_h",
        "treatment_mean_target_h",
        "control_mean_target_h",
        "mean_treatment_change",
        "mean_control_change",
        "treatment_symmetric_difference",
        "control_symmetric_difference",
        "original_edge_sha256",
        "treatment_edge_sha256",
        "control_edge_sha256",
    }
    assert required <= pair.audit.keys()
    assert all(
        value is True
        for value in pair.audit.values()
        if isinstance(value, bool)
    )


def test_generation_is_reproducible():
    edge_index, labels, tail_mask = _paired_fixture()
    kwargs = {
        "target_nodes": np.asarray([0, 1]),
        "pair_seed": 109,
        "max_attempts": 3,
    }
    first = generate_coordinated_rewire_pair(
        edge_index, labels, tail_mask, **kwargs
    )
    second = generate_coordinated_rewire_pair(
        edge_index, labels, tail_mask, **kwargs
    )

    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.target_order, second.target_order)
    assert np.array_equal(
        first.treatment_edge_index, second.treatment_edge_index
    )
    assert np.array_equal(first.control_edge_index, second.control_edge_index)
    assert first.swap_records() == second.swap_records()
    assert first.attempt == second.attempt
    assert first.retry_failures == second.retry_failures
    assert first.audit == second.audit


def test_validation_rejects_invalid_targets_and_inputs():
    edge_index, labels, tail_mask = _paired_fixture()
    common = {"pair_seed": 113, "max_attempts": 1}
    with pytest.raises(ValueError, match="at least one"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([], dtype=np.int64),
            **common,
        )
    with pytest.raises(ValueError, match="duplicates"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0, 0]),
            **common,
        )
    with pytest.raises(ValueError, match="must be a tail node"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([3]),
            **common,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([labels.size]),
            **common,
        )
    with pytest.raises(ValueError, match="integer node indices"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0.0]),
            **common,
        )
    with pytest.raises(ValueError, match="max_attempts must be positive"):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0]),
            pair_seed=113,
            max_attempts=0,
        )


def test_impossible_construction_fails_without_mutating_input():
    edge_index = _edge_index([(0, 1)])
    original = edge_index.copy()
    labels = np.asarray([0, 1], dtype=np.int64)
    tail_mask = np.asarray([True, True])

    with pytest.raises(
        PairGenerationError,
        match="could not generate a complete coordinated pair after 2 attempts",
    ):
        generate_coordinated_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0]),
            pair_seed=127,
            max_attempts=2,
        )
    assert np.array_equal(edge_index, original)
