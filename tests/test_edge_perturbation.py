"""Tests for paired, degree-preserving tail-edge perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from hfd.edge_perturbation import PairGenerationError, generate_rewire_pair
from hfd.homophily import local_homophily


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    """Return one orientation of a simple undirected edge list."""
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
    assert all(a != b for a, b in directed)
    assert len(directed) == len(set(directed))
    assert all((b, a) in set(directed) for a, b in directed)


def _increase_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Target 0 has one different-class neighbour, 1.  Edges (1,2) and
    # (1,4) make the reverse orientations of the donor edges invalid, so the
    # only treatment donor is oriented 2->3 and the only control donor 4->5.
    labels = np.asarray([0, 1, 0, 0, 1, 1], dtype=np.int64)
    edge_index = _edge_index([(0, 1), (2, 3), (4, 5), (1, 2), (1, 4)])
    tail_mask = np.asarray([True, False, False, False, False, False])
    return edge_index, labels, tail_mask


def _decrease_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Target 0 has one same-class neighbour, 1.  The uniquely oriented
    # treatment donor is 2->3 and the uniquely oriented control donor 4->5.
    labels = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)
    edge_index = _edge_index([(0, 1), (2, 3), (4, 5), (1, 2), (1, 4)])
    tail_mask = np.asarray([True, False, False, False, False, False])
    return edge_index, labels, tail_mask


def _two_target_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Two disjoint copies of the increase fixture.  There are ample donors
    # even if the two targets are processed in either order.
    labels = np.asarray(
        [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int64
    )
    edges = [
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (1, 4),
        (6, 7),
        (8, 9),
        (10, 11),
        (7, 8),
        (7, 10),
    ]
    tail_mask = np.zeros(labels.size, dtype=bool)
    tail_mask[[0, 6]] = True
    return _edge_index(edges), labels, tail_mask


def test_pubmed_direction_raises_target_homophily():
    edge_index, labels, tail_mask = _increase_fixture()
    pair = generate_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        desired_same=True,
        pair_seed=11,
        target_count=1,
        max_attempts=1,
        donor_random_tries=0,
    )

    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )
    control_h = local_homophily(pair.control_edge_index, labels, labels.size)

    assert pair.targets.tolist() == [0]
    assert pair.treatment_swaps[0].target == 0
    assert pair.treatment_swaps[0].removed_neighbour == 1
    assert (
        pair.treatment_swaps[0].donor_near,
        pair.treatment_swaps[0].donor_far,
    ) == (2, 3)
    assert (
        pair.control_swaps[0].donor_near,
        pair.control_swaps[0].donor_far,
    ) == (4, 5)
    assert original_h[0] == 0.0
    assert treatment_h[0] == 1.0
    assert control_h[0] == original_h[0]


def test_roman_direction_lowers_target_homophily():
    edge_index, labels, tail_mask = _decrease_fixture()
    pair = generate_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        desired_same=False,
        pair_seed=17,
        target_count=1,
        max_attempts=1,
        donor_random_tries=0,
    )

    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )
    control_h = local_homophily(pair.control_edge_index, labels, labels.size)

    assert pair.targets.tolist() == [0]
    assert pair.treatment_swaps[0].removed_neighbour == 1
    assert (
        pair.treatment_swaps[0].donor_near,
        pair.treatment_swaps[0].donor_far,
    ) == (2, 3)
    assert (
        pair.control_swaps[0].donor_near,
        pair.control_swaps[0].donor_far,
    ) == (4, 5)
    assert original_h[0] == 1.0
    assert treatment_h[0] == 0.0
    assert control_h[0] == original_h[0]


def test_pair_preserves_degree_edge_count_and_simple_graph_invariants():
    edge_index, labels, tail_mask = _two_target_fixture()
    pair = generate_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        desired_same=True,
        pair_seed=23,
        target_count=2,
        max_attempts=10,
    )

    original_edges = _undirected_edges(edge_index)
    original_degree = _degree(edge_index, labels.size)
    for rewired in (pair.treatment_edge_index, pair.control_edge_index):
        assert len(_undirected_edges(rewired)) == len(original_edges)
        assert np.array_equal(_degree(rewired, labels.size), original_degree)
        assert len(original_edges.symmetric_difference(_undirected_edges(rewired))) == 8
        _assert_simple_bidirectional(rewired)

    assert all(value is True for value in pair.audit.values() if isinstance(value, bool))


def test_targets_are_protected_and_source_edges_are_matched():
    edge_index, labels, tail_mask = _two_target_fixture()
    pair = generate_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        desired_same=True,
        pair_seed=29,
        target_count=2,
        max_attempts=10,
    )

    target_set = set(map(int, pair.targets))
    treatment = {record.target: record for record in pair.treatment_swaps}
    control = {record.target: record for record in pair.control_swaps}
    assert set(treatment) == target_set == set(control)
    assert len(pair.treatment_swaps) == len(pair.control_swaps) == 2

    for target in target_set:
        treatment_record = treatment[target]
        control_record = control[target]
        assert treatment_record.removed_neighbour == control_record.removed_neighbour
        for record in (treatment_record, control_record):
            auxiliaries = {
                record.removed_neighbour,
                record.donor_near,
                record.donor_far,
            }
            assert target_set.isdisjoint(auxiliaries)


def test_pair_generation_is_reproducible():
    edge_index, labels, tail_mask = _two_target_fixture()
    kwargs = {
        "desired_same": True,
        "pair_seed": 31,
        "target_count": 2,
        "max_attempts": 10,
    }
    first = generate_rewire_pair(edge_index, labels, tail_mask, **kwargs)
    second = generate_rewire_pair(edge_index, labels, tail_mask, **kwargs)

    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.target_order, second.target_order)
    assert np.array_equal(
        first.treatment_edge_index, second.treatment_edge_index
    )
    assert np.array_equal(first.control_edge_index, second.control_edge_index)
    assert first.treatment_swaps == second.treatment_swaps
    assert first.control_swaps == second.control_swaps
    assert first.attempt == second.attempt
    assert first.retry_failures == second.retry_failures
    assert first.audit == second.audit


def test_explicit_target_nodes_are_used_exactly_and_reproducibly():
    edge_index, labels, tail_mask = _two_target_fixture()
    kwargs = {
        "desired_same": True,
        "pair_seed": 32,
        "target_nodes": np.asarray([6], dtype=np.int64),
        "max_attempts": 10,
    }
    first = generate_rewire_pair(edge_index, labels, tail_mask, **kwargs)
    second = generate_rewire_pair(edge_index, labels, tail_mask, **kwargs)

    assert first.targets.tolist() == [6]
    assert first.requested_targets == 1
    assert np.array_equal(first.target_order, second.target_order)
    assert np.array_equal(
        first.treatment_edge_index, second.treatment_edge_index
    )
    assert np.array_equal(first.control_edge_index, second.control_edge_index)


def test_explicit_target_nodes_are_validated():
    edge_index, labels, tail_mask = _two_target_fixture()
    common = {
        "desired_same": True,
        "pair_seed": 34,
        "max_attempts": 1,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_count=1,
            target_nodes=np.asarray([0]),
            **common,
        )
    with pytest.raises(ValueError, match="duplicates"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0, 0]),
            **common,
        )
    with pytest.raises(ValueError, match="must be a tail node"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([1]),
            **common,
        )
    with pytest.raises(ValueError, match="out-of-range"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([labels.size]),
            **common,
        )
    with pytest.raises(ValueError, match="integer node indices"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([0.0]),
            **common,
        )
    with pytest.raises(ValueError, match="between 1 and"):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            target_nodes=np.asarray([], dtype=np.int64),
            **common,
        )


def test_uniform_sample_can_include_adjacent_degree_one_targets():
    labels = np.asarray([0, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int64)
    edge_index = _edge_index([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])
    tail_mask = np.zeros(labels.size, dtype=bool)
    tail_mask[[0, 1]] = True

    pair = generate_rewire_pair(
        edge_index,
        labels,
        tail_mask,
        desired_same=True,
        pair_seed=33,
        target_count=2,
        max_attempts=10,
    )
    original_h = local_homophily(edge_index, labels, labels.size)
    treatment_h = local_homophily(
        pair.treatment_edge_index, labels, labels.size
    )
    control_h = local_homophily(pair.control_edge_index, labels, labels.size)

    assert pair.targets.tolist() == [0, 1]
    assert any(
        record.removed_neighbour in {0, 1}
        for record in pair.treatment_swaps + pair.control_swaps
    )
    assert np.array_equal(original_h[pair.targets], np.asarray([0.0, 0.0]))
    assert np.array_equal(treatment_h[pair.targets], np.asarray([1.0, 1.0]))
    assert np.array_equal(control_h[pair.targets], original_h[pair.targets])


def test_impossible_pair_fails_cleanly_without_mutating_input():
    edge_index = _edge_index([(0, 1)])
    original = edge_index.copy()
    labels = np.asarray([0, 1], dtype=np.int64)
    tail_mask = np.asarray([True, False])

    with pytest.raises(
        PairGenerationError,
        match="could not generate a complete pair after 2 attempts",
    ):
        generate_rewire_pair(
            edge_index,
            labels,
            tail_mask,
            desired_same=True,
            pair_seed=37,
            target_count=1,
            max_attempts=2,
            donor_random_tries=0,
        )

    assert np.array_equal(edge_index, original)
