"""Coordinated degree-preserving perturbations for low-homophily targets.

Selected targets are paired within label classes whenever possible.  One
double-edge swap then raises the same-neighbour count of both targets.  A
non-tail donor swap handles targets left unmatched within a class.  The
matched control changes one cross-label edge for every selected target while
leaving the local homophily of every tail node unchanged.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .edge_perturbation import (
    PairGenerationError,
    _canonical_edges,
    _degree_vector,
    _directed_edge_index,
    edge_index_sha256,
)
from .homophily import local_homophily


_RANDOM_PROBES = 512


@dataclass(frozen=True)
class CoordinatedSwapRecord:
    """One double-edge swap and the selected targets it services."""

    mode: str
    targets: tuple[int, ...]
    removed_edges: tuple[tuple[int, int], tuple[int, int]]
    added_edges: tuple[tuple[int, int], tuple[int, int]]

    def serialisable(self) -> dict:
        return {
            "mode": self.mode,
            "targets": list(self.targets),
            "removed_edges": [list(edge) for edge in self.removed_edges],
            "added_edges": [list(edge) for edge in self.added_edges],
        }


@dataclass
class CoordinatedRewirePair:
    """A completed coordinated treatment/control graph pair."""

    treatment_edge_index: np.ndarray
    control_edge_index: np.ndarray
    targets: np.ndarray
    target_order: np.ndarray
    treatment_swaps: list[CoordinatedSwapRecord]
    control_swaps: list[CoordinatedSwapRecord]
    pair_seed: int
    attempt: int
    requested_targets: int
    retry_failures: list[str]
    audit: dict

    def swap_records(self) -> dict[str, list[dict]]:
        return {
            "treatment": [row.serialisable() for row in self.treatment_swaps],
            "control": [row.serialisable() for row in self.control_swaps],
        }


class _CoordinatedGraphState:
    """Mutable simple graph with exact original-graph delta bookkeeping."""

    def __init__(self, edges: set[tuple[int, int]], num_nodes: int):
        self.edges = set(edges)
        self.removed: set[tuple[int, int]] = set()
        self.added: set[tuple[int, int]] = set()
        self.adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
        for a, b in edges:
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

    @staticmethod
    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def can_swap(self, v: int, u: int, w: int, x: int) -> bool:
        """Check ``(v,u),(w,x) -> (v,w),(u,x)``.

        Either removed edge may itself have been added by an earlier swap.
        That permission is needed when selected targets form cross-class
        chains.  Previously removed original edges are not restored.
        """
        if len({v, u, w, x}) != 4:
            return False
        old_edges = (self.key(v, u), self.key(w, x))
        new_edges = (self.key(v, w), self.key(u, x))
        if old_edges[0] == old_edges[1] or new_edges[0] == new_edges[1]:
            return False
        if any(edge not in self.edges for edge in old_edges):
            return False
        if any(edge in self.edges for edge in new_edges):
            return False
        if any(edge in self.removed for edge in new_edges):
            return False
        return True

    def apply(
        self,
        v: int,
        u: int,
        w: int,
        x: int,
        *,
        mode: str,
        targets: tuple[int, ...],
    ) -> CoordinatedSwapRecord:
        if not self.can_swap(v, u, w, x):
            raise ValueError("invalid coordinated double-edge swap")
        old_edges = (self.key(v, u), self.key(w, x))
        new_edges = (self.key(v, w), self.key(u, x))

        for edge in old_edges:
            self.edges.remove(edge)
            if edge in self.added:
                self.added.remove(edge)
            else:
                self.removed.add(edge)
        for edge in new_edges:
            self.edges.add(edge)
            self.added.add(edge)

        for a, b in old_edges:
            self.adjacency[a].remove(b)
            self.adjacency[b].remove(a)
        for a, b in new_edges:
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

        return CoordinatedSwapRecord(
            mode=mode,
            targets=targets,
            removed_edges=old_edges,
            added_edges=new_edges,
        )


def _cross_neighbours(
    state: _CoordinatedGraphState,
    target: int,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    neighbours = np.asarray(
        sorted(
            node
            for node in state.adjacency[target]
            if labels[node] != labels[target]
        ),
        dtype=np.int64,
    )
    rng.shuffle(neighbours)
    return neighbours


def _cross_label_set(
    state: _CoordinatedGraphState,
    target: int,
    labels: np.ndarray,
) -> set[int]:
    return {
        int(labels[node])
        for node in state.adjacency[target]
        if labels[node] != labels[target]
    }


def _find_paired_treatment_swap(
    state: _CoordinatedGraphState,
    labels: np.ndarray,
    v: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    """Find a swap raising the homophily of same-label ``v`` and ``w``."""
    if labels[v] != labels[w] or state.key(v, w) in state.edges:
        return None
    v_neighbours = _cross_neighbours(state, v, labels, rng)
    w_neighbours = _cross_neighbours(state, w, labels, rng)
    if not v_neighbours.size or not w_neighbours.size:
        return None

    probe_count = min(
        _RANDOM_PROBES, int(v_neighbours.size * w_neighbours.size)
    )
    for _ in range(probe_count):
        u = int(v_neighbours[int(rng.integers(v_neighbours.size))])
        x = int(w_neighbours[int(rng.integers(w_neighbours.size))])
        if labels[u] != labels[x] and state.can_swap(v, u, w, x):
            return v, u, w, x

    for u_value in v_neighbours:
        u = int(u_value)
        for x_value in w_neighbours:
            x = int(x_value)
            if labels[u] != labels[x] and state.can_swap(v, u, w, x):
                return v, u, w, x
    return None


def _find_paired_control_swap(
    state: _CoordinatedGraphState,
    labels: np.ndarray,
    v: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    """Find a neutral cross-neighbour exchange for same-label targets."""
    if labels[v] != labels[w]:
        return None
    v_neighbours = _cross_neighbours(state, v, labels, rng)
    w_neighbours = _cross_neighbours(state, w, labels, rng)
    if not v_neighbours.size or not w_neighbours.size:
        return None

    probe_count = min(
        _RANDOM_PROBES, int(v_neighbours.size * w_neighbours.size)
    )
    for _ in range(probe_count):
        u = int(v_neighbours[int(rng.integers(v_neighbours.size))])
        x = int(w_neighbours[int(rng.integers(w_neighbours.size))])
        # Passing x,w as the donor orientation gives
        # (v,u),(w,x) -> (v,x),(u,w).
        if state.can_swap(v, u, x, w):
            return v, u, x, w

    for u_value in v_neighbours:
        u = int(u_value)
        for x_value in w_neighbours:
            x = int(x_value)
            if state.can_swap(v, u, x, w):
                return v, u, x, w
    return None


def _find_singleton_swap(
    state: _CoordinatedGraphState,
    donor_pool: np.ndarray,
    labels: np.ndarray,
    tail_mask: np.ndarray,
    *,
    target: int,
    want_same: bool,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    """Find a non-tail-donor swap for one unmatched target."""
    source_neighbours = _cross_neighbours(state, target, labels, rng)
    if not source_neighbours.size or not donor_pool.size:
        return None

    def acceptable(
        removed_neighbour: int, row_index: int
    ) -> tuple[int, int, int, int] | None:
        w, x = map(int, donor_pool[row_index])
        if (labels[w] == labels[target]) != want_same:
            return None
        # Every tail node other than the primary treatment target must retain
        # its same/cross relation.  The removed target edge is cross-label.
        if tail_mask[removed_neighbour] and (
            labels[x] == labels[removed_neighbour]
        ):
            return None
        if state.can_swap(target, removed_neighbour, w, x):
            return target, removed_neighbour, w, x
        return None

    pool_size = int(donor_pool.shape[0])
    for neighbour_value in source_neighbours:
        neighbour = int(neighbour_value)
        probe_count = min(_RANDOM_PROBES, pool_size)
        probes = rng.integers(0, pool_size, size=probe_count)
        for row_index in probes:
            found = acceptable(neighbour, int(row_index))
            if found is not None:
                return found
        start = int(rng.integers(0, pool_size))
        for offset in range(pool_size):
            found = acceptable(neighbour, (start + offset) % pool_size)
            if found is not None:
                return found
    return None


def _plan_class_targets(
    state: _CoordinatedGraphState,
    class_targets: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pair compatible target profiles and return unavoidable singletons."""
    mono: dict[int, list[int]] = {}
    flexible: list[int] = []
    for target_value in class_targets:
        target = int(target_value)
        neighbour_labels = _cross_label_set(state, target, labels)
        if not neighbour_labels:
            raise PairGenerationError(
                f"target {target} has no cross-label neighbour"
            )
        if len(neighbour_labels) == 1:
            mono.setdefault(next(iter(neighbour_labels)), []).append(target)
        else:
            flexible.append(target)

    for group in mono.values():
        rng.shuffle(group)
    rng.shuffle(flexible)

    heap = [(-len(group), label) for label, group in mono.items()]
    heapq.heapify(heap)
    pairs: list[tuple[int, int]] = []
    singletons: list[int] = []

    # Targets whose cross neighbours all have the same label cannot be paired
    # with one another.  Pair the largest distinct groups first.
    while len(heap) >= 2:
        count_a, label_a = heapq.heappop(heap)
        count_b, label_b = heapq.heappop(heap)
        pairs.append((mono[label_a].pop(), mono[label_b].pop()))
        count_a += 1
        count_b += 1
        if count_a < 0:
            heapq.heappush(heap, (count_a, label_a))
        if count_b < 0:
            heapq.heappush(heap, (count_b, label_b))

    if heap:
        negative_count, label = heapq.heappop(heap)
        remaining = -negative_count
        paired_with_flexible = min(remaining, len(flexible))
        for _ in range(paired_with_flexible):
            pairs.append((mono[label].pop(), flexible.pop()))
        singletons.extend(mono[label])

    while len(flexible) >= 2:
        pairs.append((flexible.pop(), flexible.pop()))
    singletons.extend(flexible)

    rng.shuffle(pairs)
    rng.shuffle(singletons)
    return pairs, singletons


def _same_label_donor_pools(
    donor_pool: np.ndarray,
    labels: np.ndarray,
    target_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Index the small same-label donor slices needed by singleton treatment."""
    near_labels = labels[donor_pool[:, 0]]
    return {
        int(label): donor_pool[near_labels == label]
        for label in np.unique(target_labels)
    }


def _paired_swaps(
    treatment: _CoordinatedGraphState,
    control: _CoordinatedGraphState,
    labels: np.ndarray,
    pair: tuple[int, int],
    treatment_rng: np.random.Generator,
    control_rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    v, w = pair
    treatment_swap = _find_paired_treatment_swap(
        treatment, labels, v, w, treatment_rng
    )
    if treatment_swap is None:
        return None
    control_swap = _find_paired_control_swap(
        control, labels, v, w, control_rng
    )
    if control_swap is None:
        return None
    return treatment_swap, control_swap


def _repair_pair_plan(
    pairs: list[tuple[int, int]],
    index: int,
    treatment: _CoordinatedGraphState,
    control: _CoordinatedGraphState,
    labels: np.ndarray,
    treatment_rng: np.random.Generator,
    control_rng: np.random.Generator,
    order_rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Find current swaps, exchanging later partners when necessary."""
    direct = _paired_swaps(
        treatment,
        control,
        labels,
        pairs[index],
        treatment_rng,
        control_rng,
    )
    if direct is not None:
        return direct

    v, w = pairs[index]
    later = np.arange(index + 1, len(pairs), dtype=np.int64)
    order_rng.shuffle(later)
    for later_index_value in later:
        later_index = int(later_index_value)
        a, b = pairs[later_index]
        for current, future in (
            ((v, a), (w, b)),
            ((v, b), (w, a)),
        ):
            current_swaps = _paired_swaps(
                treatment,
                control,
                labels,
                current,
                treatment_rng,
                control_rng,
            )
            if current_swaps is None:
                continue
            future_swaps = _paired_swaps(
                treatment,
                control,
                labels,
                future,
                treatment_rng,
                control_rng,
            )
            if future_swaps is None:
                continue
            pairs[index] = current
            pairs[later_index] = future
            return current_swaps
    return None


def _attempt_pair(
    canonical: np.ndarray,
    original_edges: set[tuple[int, int]],
    labels: np.ndarray,
    tail_mask: np.ndarray,
    targets: np.ndarray,
    *,
    pair_seed: int,
    attempt: int,
) -> tuple[
    _CoordinatedGraphState,
    _CoordinatedGraphState,
    np.ndarray,
    list[CoordinatedSwapRecord],
    list[CoordinatedSwapRecord],
]:
    seed_sequence = np.random.SeedSequence([int(pair_seed), int(attempt)])
    order_rng, treatment_rng, control_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(3)
    ]
    treatment = _CoordinatedGraphState(original_edges, labels.size)
    control = _CoordinatedGraphState(original_edges, labels.size)

    oriented = np.concatenate([canonical, canonical[:, ::-1]], axis=0)
    donor_pool = oriented[
        ~tail_mask[oriented[:, 0]] & ~tail_mask[oriented[:, 1]]
    ]
    donor_by_label = _same_label_donor_pools(
        donor_pool, labels, labels[targets]
    )

    treatment_records: list[CoordinatedSwapRecord] = []
    control_records: list[CoordinatedSwapRecord] = []
    target_order: list[int] = []

    for label_value in np.unique(labels[targets]):
        label = int(label_value)
        class_targets = targets[labels[targets] == label]
        class_targets = class_targets.copy()
        order_rng.shuffle(class_targets)
        pairs, singletons = _plan_class_targets(
            treatment, class_targets, labels, order_rng
        )

        for target in singletons:
            treatment_swap = _find_singleton_swap(
                treatment,
                donor_by_label[label],
                labels,
                tail_mask,
                target=target,
                want_same=True,
                rng=treatment_rng,
            )
            control_swap = _find_singleton_swap(
                control,
                donor_pool,
                labels,
                tail_mask,
                target=target,
                want_same=False,
                rng=control_rng,
            )
            if treatment_swap is None or control_swap is None:
                raise PairGenerationError(
                    f"target {target} has no jointly valid singleton swaps"
                )
            treatment_records.append(
                treatment.apply(
                    *treatment_swap, mode="singleton", targets=(target,)
                )
            )
            control_records.append(
                control.apply(
                    *control_swap, mode="singleton", targets=(target,)
                )
            )
            target_order.append(target)

        pair_index = 0
        while pair_index < len(pairs):
            v, w = pairs[pair_index]
            swaps = _repair_pair_plan(
                pairs,
                pair_index,
                treatment,
                control,
                labels,
                treatment_rng,
                control_rng,
                order_rng,
            )
            if swaps is None:
                # A simple-graph conflict may make a profile-compatible pair
                # unusable.  Two singleton swaps preserve the class parity.
                for target in (v, w):
                    treatment_swap = _find_singleton_swap(
                        treatment,
                        donor_by_label[label],
                        labels,
                        tail_mask,
                        target=target,
                        want_same=True,
                        rng=treatment_rng,
                    )
                    control_swap = _find_singleton_swap(
                        control,
                        donor_pool,
                        labels,
                        tail_mask,
                        target=target,
                        want_same=False,
                        rng=control_rng,
                    )
                    if treatment_swap is None or control_swap is None:
                        raise PairGenerationError(
                            f"class {label} pair ({v}, {w}) cannot be paired "
                            "or replaced by singleton swaps"
                        )
                    treatment_records.append(
                        treatment.apply(
                            *treatment_swap,
                            mode="singleton",
                            targets=(target,),
                        )
                    )
                    control_records.append(
                        control.apply(
                            *control_swap,
                            mode="singleton",
                            targets=(target,),
                        )
                    )
                    target_order.append(target)
            else:
                treatment_swap, control_swap = swaps
                v, w = pairs[pair_index]
                treatment_records.append(
                    treatment.apply(
                        *treatment_swap, mode="paired", targets=(v, w)
                    )
                )
                control_records.append(
                    control.apply(
                        *control_swap, mode="paired", targets=(v, w)
                    )
                )
                target_order.extend((v, w))
            pair_index += 1

    target_order_array = np.asarray(target_order, dtype=np.int64)
    if target_order_array.size != targets.size or not np.array_equal(
        np.sort(target_order_array), targets
    ):
        raise PairGenerationError(
            "coordinated construction did not process every target exactly once"
        )
    return (
        treatment,
        control,
        target_order_array,
        treatment_records,
        control_records,
    )


def _audit_pair(
    original_edges: set[tuple[int, int]],
    treatment: _CoordinatedGraphState,
    control: _CoordinatedGraphState,
    labels: np.ndarray,
    tail_mask: np.ndarray,
    targets: np.ndarray,
    treatment_records: list[CoordinatedSwapRecord],
    control_records: list[CoordinatedSwapRecord],
) -> tuple[np.ndarray, np.ndarray, dict]:
    num_nodes = int(labels.size)
    original_edge_index = _directed_edge_index(original_edges)
    treatment_edge_index = _directed_edge_index(treatment.edges)
    control_edge_index = _directed_edge_index(control.edges)

    original_degree = _degree_vector(original_edges, num_nodes)
    treatment_degree = _degree_vector(treatment.edges, num_nodes)
    control_degree = _degree_vector(control.edges, num_nodes)
    original_h = local_homophily(original_edge_index, labels, num_nodes)
    treatment_h = local_homophily(treatment_edge_index, labels, num_nodes)
    control_h = local_homophily(control_edge_index, labels, num_nodes)
    expected_step = 1.0 / original_degree[targets]
    observed_step = treatment_h[targets] - original_h[targets]

    unselected_tail = tail_mask.copy()
    unselected_tail[targets] = False
    treatment_occurrences = sorted(
        target
        for record in treatment_records
        for target in record.targets
    )
    control_occurrences = sorted(
        target for record in control_records for target in record.targets
    )
    target_list = targets.tolist()
    paired_count = sum(row.mode == "paired" for row in treatment_records)
    singleton_count = sum(
        row.mode == "singleton" for row in treatment_records
    )

    checks = {
        "edge_count_preserved": (
            len(treatment.edges) == len(original_edges) == len(control.edges)
        ),
        "treatment_degrees_preserved": np.array_equal(
            treatment_degree, original_degree
        ),
        "control_degrees_preserved": np.array_equal(
            control_degree, original_degree
        ),
        "treatment_homophily_step_exact": np.allclose(
            observed_step, expected_step, atol=1e-12, rtol=0.0
        ),
        "control_target_homophily_unchanged": np.allclose(
            control_h[targets], original_h[targets], atol=1e-12, rtol=0.0
        ),
        "treatment_unselected_tail_homophily_unchanged": np.allclose(
            treatment_h[unselected_tail],
            original_h[unselected_tail],
            atol=1e-12,
            rtol=0.0,
        ),
        "control_full_tail_homophily_unchanged": np.allclose(
            control_h[tail_mask],
            original_h[tail_mask],
            atol=1e-12,
            rtol=0.0,
        ),
        "each_target_processed_once": (
            treatment_occurrences == target_list
            and control_occurrences == target_list
        ),
        "treatment_control_swap_counts_match": (
            len(treatment_records) == len(control_records)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise PairGenerationError(f"generated pair failed audit: {failed}")

    audit = {
        **checks,
        "num_nodes": num_nodes,
        "num_undirected_edges": len(original_edges),
        "num_targets": int(targets.size),
        "paired_swap_count": int(paired_count),
        "singleton_swap_count": int(singleton_count),
        "treatment_swap_count": len(treatment_records),
        "control_swap_count": len(control_records),
        "original_mean_target_h": float(original_h[targets].mean()),
        "treatment_mean_target_h": float(treatment_h[targets].mean()),
        "control_mean_target_h": float(control_h[targets].mean()),
        "mean_treatment_change": float(observed_step.mean()),
        "mean_control_change": float(
            (control_h[targets] - original_h[targets]).mean()
        ),
        "treatment_symmetric_difference": len(
            original_edges.symmetric_difference(treatment.edges)
        ),
        "control_symmetric_difference": len(
            original_edges.symmetric_difference(control.edges)
        ),
        "original_edge_sha256": edge_index_sha256(original_edge_index),
        "treatment_edge_sha256": edge_index_sha256(treatment_edge_index),
        "control_edge_sha256": edge_index_sha256(control_edge_index),
    }
    return treatment_edge_index, control_edge_index, audit


def generate_coordinated_rewire_pair(
    edge_index: np.ndarray,
    labels: np.ndarray,
    tail_mask: np.ndarray,
    *,
    target_nodes: np.ndarray,
    pair_seed: int,
    max_attempts: int = 100,
) -> CoordinatedRewirePair:
    """Generate an exact coordinated treatment/control graph pair.

    Every selected low-homophily target loses one cross-label neighbour and
    gains one same-label neighbour in treatment.  The corresponding control
    replaces a cross-label neighbour with another cross-label neighbour.
    All node degrees, all unselected-tail homophily values in treatment, and
    all tail homophily values in control are preserved exactly.
    """
    raw_edge_index = np.asarray(edge_index)
    raw_labels = np.asarray(labels)
    raw_tail = np.asarray(tail_mask)
    raw_targets = np.asarray(target_nodes)

    if raw_edge_index.ndim != 2 or raw_edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if not np.issubdtype(raw_edge_index.dtype, np.integer):
        raise ValueError("edge_index must contain integer node indices")
    if raw_labels.ndim != 1 or raw_tail.shape != raw_labels.shape:
        raise ValueError("labels and tail_mask must be one-dimensional and aligned")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must contain integers")
    if np.any(raw_labels < 0):
        raise ValueError("coordinated rewiring requires every node to be labelled")
    if raw_edge_index.size and (
        np.any(raw_edge_index < 0)
        or np.any(raw_edge_index >= raw_labels.size)
    ):
        raise ValueError("edge_index contains an out-of-range node index")
    if raw_targets.ndim != 1:
        raise ValueError("target_nodes must be one-dimensional")
    if not np.issubdtype(raw_targets.dtype, np.integer):
        raise ValueError("target_nodes must contain integer node indices")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    labels_array = raw_labels.astype(np.int64, copy=False)
    tail_array = raw_tail.astype(bool, copy=False)
    targets = raw_targets.astype(np.int64, copy=True)
    if targets.size < 1:
        raise ValueError("target_nodes must contain at least one target")
    if np.unique(targets).size != targets.size:
        raise ValueError("target_nodes must not contain duplicates")
    if np.any(targets < 0) or np.any(targets >= labels_array.size):
        raise ValueError("target_nodes contains an out-of-range node index")
    if not np.all(tail_array[targets]):
        raise ValueError("every target_nodes entry must be a tail node")
    targets.sort()

    canonical, original_edges = _canonical_edges(
        raw_edge_index.astype(np.int64, copy=False), labels_array.size
    )
    failures: list[str] = []
    for attempt in range(int(max_attempts)):
        try:
            (
                treatment,
                control,
                target_order,
                treatment_records,
                control_records,
            ) = _attempt_pair(
                canonical,
                original_edges,
                labels_array,
                tail_array,
                targets,
                pair_seed=int(pair_seed),
                attempt=attempt,
            )
            treatment_edge_index, control_edge_index, audit = _audit_pair(
                original_edges,
                treatment,
                control,
                labels_array,
                tail_array,
                targets,
                treatment_records,
                control_records,
            )
            return CoordinatedRewirePair(
                treatment_edge_index=treatment_edge_index,
                control_edge_index=control_edge_index,
                targets=targets,
                target_order=target_order,
                treatment_swaps=treatment_records,
                control_swaps=control_records,
                pair_seed=int(pair_seed),
                attempt=attempt,
                requested_targets=int(targets.size),
                retry_failures=failures,
                audit=audit,
            )
        except PairGenerationError as error:
            failures.append(str(error))

    detail = failures[-1] if failures else "unknown failure"
    raise PairGenerationError(
        f"could not generate a complete coordinated pair after "
        f"{max_attempts} attempts; last failure: {detail}"
    )
