"""Local homophily (h_v), bin construction, and minority-pattern tail.

Implements the procedure fixed in the proposal's Experimental Contract
(appendix "Fixed inputs", "Local-homophily groups"):

1. h_v is the fraction of one-hop neighbours with the same class label as v,
   computed once for every labelled node on the full preprocessed graph.
2. The graph used is the simple undirected graph: directed edges symmetrised,
   self-loops removed, duplicate edges collapsed.
3. Labelled nodes with no neighbours after preprocessing have undefined h_v
   and are excluded from bin analyses; their count is reported.
4. Dataset homophily is the mean h_v over eligible labelled nodes and fixes
   the minority-pattern tail direction (>0.5 homophilic -> low-h_v tail,
   <0.5 heterophilic -> high-h_v tail).
5-9. Four base bins [0,.25), [.25,.5), [.5,.75), [.75,1]; bins holding less
   than 1% of eligible labelled nodes are merged iteratively by the
   deterministic rule; the tail is the extreme post-merge bin in the
   minority-pattern direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
MERGE_THRESHOLD_FRACTION = 0.01


def to_simple_undirected(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Symmetrise, remove self-loops, and collapse duplicate edges.

    Args:
        edge_index: int array of shape (2, E), directed edges.
        num_nodes: number of nodes N.

    Returns:
        int64 array of shape (2, E'), containing both directions of every
        undirected edge, no self-loops, no duplicates.
    """
    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # Symmetrise, then dedupe unordered pairs via a scalar key on (min, max).
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    key = lo * np.int64(num_nodes) + hi
    _, unique_idx = np.unique(key, return_index=True)
    lo, hi = lo[unique_idx], hi[unique_idx]
    return np.stack([np.concatenate([lo, hi]), np.concatenate([hi, lo])])


def local_homophily(
    edge_index: np.ndarray, labels: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Compute h_v for every labelled node on the simple undirected graph.

    Args:
        edge_index: directed or undirected edge index, shape (2, E).
        labels: int array of shape (N,); unlabelled nodes are < 0.
        num_nodes: number of nodes N.

    Returns:
        float64 array of shape (N,): h_v for labelled nodes with at least one
        labelled neighbour, NaN where h_v is undefined (unlabelled node or no
        labelled neighbour). Neighbours without labels are excluded from both
        numerator and denominator; on the study's datasets every node is
        labelled, so this equals the plain one-hop fraction.
    """
    simple = to_simple_undirected(edge_index, num_nodes)
    src, dst = simple[0], simple[1]
    labelled = labels >= 0

    # Count, per node, its labelled neighbours and same-label neighbours.
    edge_keep = labelled[dst]
    src_k, dst_k = src[edge_keep], dst[edge_keep]
    deg = np.bincount(src_k, minlength=num_nodes).astype(np.float64)
    same = np.bincount(
        src_k, weights=(labels[src_k] == labels[dst_k]), minlength=num_nodes
    )

    h = np.full(num_nodes, np.nan)
    defined = labelled & (deg > 0)
    h[defined] = same[defined] / deg[defined]
    return h


def assign_base_bins(h: np.ndarray) -> np.ndarray:
    """Map defined h_v values to base bins 0..3; NaN maps to -1.

    Intervals: [0,.25) -> 0, [.25,.5) -> 1, [.5,.75) -> 2, [.75,1] -> 3.
    """
    bins = np.full(h.shape, -1, dtype=np.int64)
    defined = ~np.isnan(h)
    bins[defined] = np.digitize(h[defined], BASE_BIN_EDGES[1:-1], right=False)
    return bins


@dataclass
class MergedBin:
    """A contiguous group of base bins after merging."""

    base_ids: list[int]
    count: int
    lo: float
    hi: float

    @property
    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2.0


@dataclass
class BinStructure:
    """Result of the deterministic bin-merge procedure."""

    bins: list[MergedBin]
    merge_history: list[str] = field(default_factory=list)
    mean_h: float = float("nan")
    direction: str = ""  # "homophilic" or "heterophilic"
    tail_bin_index: int = -1  # index into `bins`

    @property
    def tail_bin(self) -> MergedBin:
        return self.bins[self.tail_bin_index]

    def base_to_merged(self) -> dict[int, int]:
        """Map base bin id -> index of its merged bin."""
        return {b: i for i, mb in enumerate(self.bins) for b in mb.base_ids}


def merge_bins(base_counts: list[int], mean_h: float) -> BinStructure:
    """Apply the deterministic merge rule from the experimental contract.

    Args:
        base_counts: eligible labelled-node counts for the four base bins.
        mean_h: dataset mean h_v over eligible labelled nodes; fixes the
            tail direction and breaks equal-count merge ties.

    Returns:
        BinStructure with post-merge bins, merge log, and tail bin.
    """
    if mean_h == 0.5:
        raise ValueError(
            "Dataset mean h_v is exactly 0.5; the contract does not define "
            "a tail direction. Resolve manually."
        )
    total = int(sum(base_counts))
    threshold = MERGE_THRESHOLD_FRACTION * total
    edges = BASE_BIN_EDGES
    bins = [
        MergedBin(base_ids=[i], count=int(c), lo=edges[i], hi=edges[i + 1])
        for i, c in enumerate(base_counts)
    ]
    history: list[str] = []

    while len(bins) > 1:
        below = [b for b in bins if b.count < threshold]
        if not below:
            break
        # Select: smallest eligible count; ties -> lowest-h_v bin. Position in
        # `bins` is ordered by interval, so min() with (count, lo) is exact.
        selected = min(below, key=lambda b: (b.count, b.lo))
        i = bins.index(selected)
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(bins)]
        if len(neighbours) == 1:
            target = neighbours[0]
        else:
            left, right = bins[neighbours[0]], bins[neighbours[1]]
            if left.count != right.count:
                target = neighbours[0] if left.count > right.count else neighbours[1]
            else:
                # Equal counts: merge toward the adjacent bin whose interval is
                # closer to the dataset mean h_v (interval midpoint distance);
                # still tied -> merge toward the lower-h_v side.
                d_left = abs(left.midpoint - mean_h)
                d_right = abs(right.midpoint - mean_h)
                target = neighbours[0] if d_left <= d_right else neighbours[1]
        t = bins[target]

        def _fmt(b: MergedBin) -> str:
            closed = "]" if b.hi == 1.0 else ")"
            return f"[{b.lo:g},{b.hi:g}{closed}"

        history.append(
            f"merged {_fmt(selected)} (n={selected.count}) "
            f"into {_fmt(t)} (n={t.count})"
        )
        t.base_ids = sorted(t.base_ids + selected.base_ids)
        t.count += selected.count
        t.lo = min(t.lo, selected.lo)
        t.hi = max(t.hi, selected.hi)
        bins.pop(i)

    direction = "homophilic" if mean_h > 0.5 else "heterophilic"
    tail_index = 0 if direction == "homophilic" else len(bins) - 1
    return BinStructure(
        bins=bins,
        merge_history=history,
        mean_h=mean_h,
        direction=direction,
        tail_bin_index=tail_index,
    )


@dataclass
class HomophilyGroups:
    """Full fixed-input group assignment for one dataset."""

    h: np.ndarray  # (N,) float64, NaN where undefined
    eligible: np.ndarray  # (N,) bool: labelled and h_v defined
    base_bin: np.ndarray  # (N,) int64, -1 where ineligible
    merged_bin: np.ndarray  # (N,) int64 index into structure.bins, -1 ineligible
    tail_mask: np.ndarray  # (N,) bool: eligible and in the tail bin
    structure: BinStructure
    n_labelled: int
    n_undefined_h: int  # labelled nodes with undefined h_v


def compute_homophily_groups(
    edge_index: np.ndarray, labels: np.ndarray, num_nodes: int
) -> HomophilyGroups:
    """Compute the fixed local-homophily group assignment for a dataset."""
    h = local_homophily(edge_index, labels, num_nodes)
    labelled = labels >= 0
    eligible = labelled & ~np.isnan(h)
    base_bin = assign_base_bins(h)

    mean_h = float(h[eligible].mean())
    base_counts = [int((base_bin[eligible] == b).sum()) for b in range(4)]
    structure = merge_bins(base_counts, mean_h)

    lookup = structure.base_to_merged()
    merged = np.full(num_nodes, -1, dtype=np.int64)
    for base_id, merged_id in lookup.items():
        merged[eligible & (base_bin == base_id)] = merged_id
    tail_mask = merged == structure.tail_bin_index

    return HomophilyGroups(
        h=h,
        eligible=eligible,
        base_bin=np.where(eligible, base_bin, -1),
        merged_bin=merged,
        tail_mask=tail_mask,
        structure=structure,
        n_labelled=int(labelled.sum()),
        n_undefined_h=int((labelled & np.isnan(h)).sum()),
    )
