"""Unit tests for h_v computation, base bins, and the merge rule."""

import numpy as np
import pytest

from hfd.homophily import (
    assign_base_bins,
    compute_homophily_groups,
    local_homophily,
    merge_bins,
    to_simple_undirected,
)


def test_simple_undirected_removes_self_loops_and_duplicates():
    # Directed edges with a self-loop, a duplicate, and a reciprocal pair.
    ei = np.array([[0, 0, 1, 2, 3], [1, 1, 0, 2, 0]])
    simple = to_simple_undirected(ei, 4)
    pairs = {tuple(sorted(e)) for e in simple.T.tolist()}
    assert pairs == {(0, 1), (0, 3)}
    assert simple.shape[1] == 4  # both directions of two undirected edges


def test_local_homophily_path_graph():
    # Path 0-1-2-3 with labels [0, 0, 1, 1].
    ei = np.array([[0, 1, 2], [1, 2, 3]])
    y = np.array([0, 0, 1, 1])
    h = local_homophily(ei, y, 4)
    assert h[0] == 1.0  # neighbour 1: same
    assert h[1] == 0.5  # neighbours 0 (same), 2 (diff)
    assert h[2] == 0.5
    assert h[3] == 1.0


def test_local_homophily_isolated_and_unlabelled():
    # Node 2 is isolated; node 3 unlabelled; node 4's only neighbour is node 3.
    ei = np.array([[0, 3], [1, 4]])
    y = np.array([0, 0, 1, -1, 1])
    h = local_homophily(ei, y, 5)
    assert h[0] == 1.0 and h[1] == 1.0
    assert np.isnan(h[2])  # isolated -> undefined
    assert np.isnan(h[3])  # unlabelled -> undefined
    assert np.isnan(h[4])  # only neighbour unlabelled -> undefined


def test_base_bins_boundaries():
    h = np.array([0.0, 0.24, 0.25, 0.49, 0.5, 0.74, 0.75, 1.0, np.nan])
    assert assign_base_bins(h).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1]


def test_merge_noop_when_all_bins_large():
    s = merge_bins([100, 100, 100, 100], mean_h=0.7)
    assert len(s.bins) == 4
    assert s.merge_history == []
    assert s.direction == "homophilic"
    assert s.tail_bin_index == 0  # low-h_v tail


def test_merge_small_endpoint_bin():
    # Bin 3 has <1% of 1000; must merge into its only neighbour, bin 2.
    s = merge_bins([500, 300, 191, 9], mean_h=0.3)
    assert len(s.bins) == 3
    assert s.bins[2].base_ids == [2, 3]
    assert s.bins[2].count == 200
    # Heterophilic -> high-h_v tail = the merged [0.5, 1] bin.
    assert s.direction == "heterophilic"
    assert s.tail_bin is s.bins[2]


def test_merge_interior_bin_goes_to_larger_neighbour():
    # Bin 1 (n=5) below 1% of 1000; neighbours are bin 0 (n=600), bin 2 (n=200).
    s = merge_bins([600, 5, 200, 195], mean_h=0.6)
    assert [b.base_ids for b in s.bins] == [[0, 1], [2], [3]]
    assert s.bins[0].count == 605


def test_merge_tie_breaks_toward_mean():
    # Bin 1 (n=5) below threshold; neighbours both n=400. Mean h_v = 0.8 is
    # closer to bin 2's midpoint (0.625) than bin 0's (0.125): merge right.
    s = merge_bins([400, 5, 400, 195], mean_h=0.8)
    assert [b.base_ids for b in s.bins] == [[0], [1, 2], [3]]


def test_merge_selects_smallest_then_lowest():
    # Bins 0 and 3 both n=4 (< 1% of 1000): lowest-h_v bin (0) merges first,
    # into bin 1. Then bin 3 merges into bin 2.
    s = merge_bins([4, 500, 492, 4], mean_h=0.55)
    assert len(s.bins) == 2
    assert s.bins[0].base_ids == [0, 1]
    assert s.bins[1].base_ids == [2, 3]


def test_merge_cascades_until_threshold():
    # 10,000 nodes; bins 2 and 3 tiny. After merging both, all bins >= 1%.
    s = merge_bins([9000, 800, 60, 140], mean_h=0.9)
    assert all(b.count >= 100 for b in s.bins)
    assert sum(b.count for b in s.bins) == 10000


def test_mean_half_raises():
    with pytest.raises(ValueError):
        merge_bins([100, 100, 100, 100], mean_h=0.5)


def test_compute_homophily_groups_end_to_end():
    # Star: centre 0 (label 0) with 9 leaves, 6 label-0 and 3 label-1.
    # Plus one isolated labelled node 10.
    src = np.zeros(9, dtype=np.int64)
    dst = np.arange(1, 10, dtype=np.int64)
    ei = np.stack([src, dst])
    y = np.array([0] * 7 + [1] * 3 + [0])
    g = compute_homophily_groups(ei, y, 11)
    assert g.h[0] == pytest.approx(6 / 9)
    assert g.n_undefined_h == 1  # node 10
    assert g.eligible.sum() == 10
    # Leaves 1-6 (label 0, neighbour label 0): h=1. Leaves 7-9: h=0.
    assert g.structure.mean_h == pytest.approx((6 / 9 + 6 * 1.0 + 3 * 0.0) / 10)
    assert g.structure.direction == "homophilic"
    assert g.tail_mask.sum() == 3  # the three h=0 leaves
    assert g.merged_bin[10] == -1
