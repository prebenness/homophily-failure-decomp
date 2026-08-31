"""Paired degree-preserving edge perturbations for tail nodes.

Each selected target receives one double-edge swap in two independently
evolving graph copies.  The treatment changes whether one of the target's
neighbours has the same label.  The matched control replaces that neighbour
with another node having the same label relation to the target.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from .homophily import local_homophily, to_simple_undirected


class PairGenerationError(RuntimeError):
    """Raised when a complete paired perturbation cannot be generated."""


@dataclass(frozen=True)
class SwapRecord:
    """One undirected double-edge swap."""

    target: int
    removed_neighbour: int
    donor_near: int
    donor_far: int


@dataclass
class RewirePair:
    """A completed treatment/control graph pair and its audit data."""

    treatment_edge_index: np.ndarray
    control_edge_index: np.ndarray
    targets: np.ndarray
    target_order: np.ndarray
    treatment_swaps: list[SwapRecord]
    control_swaps: list[SwapRecord]
    pair_seed: int
    attempt: int
    requested_targets: int
    retry_failures: list[str]
    audit: dict

    def swap_records(self) -> dict[str, list[dict[str, int]]]:
        return {
            "treatment": [asdict(row) for row in self.treatment_swaps],
            "control": [asdict(row) for row in self.control_swaps],
        }


class _GraphState:
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

    def can_swap(
        self,
        v: int,
        u: int,
        w: int,
        x: int,
        *,
        allow_added_target_edge: bool = False,
    ) -> bool:
        if len({v, u, w, x}) != 4:
            return False
        old_target = self.key(v, u)
        old_donor = self.key(w, x)
        new_target = self.key(v, w)
        new_other = self.key(u, x)
        if old_target not in self.edges or old_donor not in self.edges:
            return False
        if old_target in self.added and not allow_added_target_edge:
            return False
        if old_donor in self.added:
            return False
        if new_target in self.edges or new_other in self.edges:
            return False
        if new_target in self.removed or new_other in self.removed:
            return False
        return new_target != new_other

    def apply(
        self,
        v: int,
        u: int,
        w: int,
        x: int,
        *,
        allow_added_target_edge: bool = False,
    ) -> None:
        if not self.can_swap(
            v,
            u,
            w,
            x,
            allow_added_target_edge=allow_added_target_edge,
        ):
            raise ValueError("invalid double-edge swap")
        old_target = self.key(v, u)
        old_donor = self.key(w, x)
        new_target = self.key(v, w)
        new_other = self.key(u, x)
        self.edges.remove(old_target)
        self.edges.remove(old_donor)
        self.edges.add(new_target)
        self.edges.add(new_other)
        if old_target in self.added:
            self.added.remove(old_target)
        else:
            self.removed.add(old_target)
        self.removed.add(old_donor)
        self.added.update((new_target, new_other))
        for a, b in (old_target, old_donor):
            self.adjacency[a].remove(b)
            self.adjacency[b].remove(a)
        for a, b in (new_target, new_other):
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)


def _canonical_edges(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    simple = to_simple_undirected(edge_index, num_nodes)
    lo = np.minimum(simple[0], simple[1])
    hi = np.maximum(simple[0], simple[1])
    keys = np.unique(lo.astype(np.int64) * np.int64(num_nodes) + hi)
    canonical = np.stack([keys // num_nodes, keys % num_nodes], axis=1)
    edge_set = {(int(a), int(b)) for a, b in canonical}
    return canonical, edge_set


def _directed_edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    canonical = np.asarray(sorted(edges), dtype=np.int64)
    if canonical.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack(
        [
            np.concatenate([canonical[:, 0], canonical[:, 1]]),
            np.concatenate([canonical[:, 1], canonical[:, 0]]),
        ]
    )


def edge_index_sha256(edge_index: np.ndarray) -> str:
    """Checksum an edge index after canonical simple-undirected ordering."""
    if edge_index.size:
        num_nodes = int(edge_index.max()) + 1
    else:
        num_nodes = 0
    canonical, _ = _canonical_edges(edge_index, num_nodes)
    directed = _directed_edge_index({tuple(map(int, row)) for row in canonical})
    return hashlib.sha256(np.ascontiguousarray(directed).tobytes()).hexdigest()


def _degree_vector(edges: set[tuple[int, int]], num_nodes: int) -> np.ndarray:
    if not edges:
        return np.zeros(num_nodes, dtype=np.int64)
    canonical = np.asarray(list(edges), dtype=np.int64)
    return np.bincount(canonical.ravel(), minlength=num_nodes).astype(np.int64)


def _find_donor(
    state: _GraphState,
    donor_pool: np.ndarray,
    labels: np.ndarray,
    *,
    target: int,
    removed_neighbour: int,
    want_same: bool,
    rng: np.random.Generator,
    random_tries: int,
    protected: np.ndarray,
) -> tuple[int, int] | None:
    """Find a valid oriented donor edge; random probes precede a full scan."""
    n_pool = int(donor_pool.shape[0])
    if n_pool == 0:
        return None

    def acceptable(row_index: int) -> tuple[int, int] | None:
        w, x = map(int, donor_pool[row_index])
        if (labels[w] == labels[target]) != want_same:
            return None
        if protected[removed_neighbour] and (
            (labels[x] == labels[removed_neighbour])
            != (labels[target] == labels[removed_neighbour])
        ):
            return None
        if state.can_swap(
            target,
            removed_neighbour,
            w,
            x,
            allow_added_target_edge=True,
        ):
            return w, x
        return None

    probes = rng.integers(0, n_pool, size=min(random_tries, n_pool))
    for row_index in probes:
        found = acceptable(int(row_index))
        if found is not None:
            return found

    start = int(rng.integers(0, n_pool))
    for offset in range(n_pool):
        found = acceptable((start + offset) % n_pool)
        if found is not None:
            return found
    return None


def _find_target_swap(
    state: _GraphState,
    donor_pool: np.ndarray,
    labels: np.ndarray,
    *,
    target: int,
    source_same: bool,
    want_same: bool,
    rng: np.random.Generator,
    random_tries: int,
    protected: np.ndarray,
) -> tuple[int, int, int] | None:
    source_neighbours = np.asarray(
        [
            neighbour
            for neighbour in state.adjacency[target]
            if (labels[neighbour] == labels[target]) == source_same
        ],
        dtype=np.int64,
    )
    rng.shuffle(source_neighbours)
    for neighbour_value in source_neighbours:
        neighbour = int(neighbour_value)
        donor = _find_donor(
            state,
            donor_pool,
            labels,
            target=target,
            removed_neighbour=neighbour,
            want_same=want_same,
            rng=rng,
            random_tries=random_tries,
            protected=protected,
        )
        if donor is not None:
            return neighbour, donor[0], donor[1]
    return None


def _attempt_pair(
    canonical: np.ndarray,
    original_edges: set[tuple[int, int]],
    labels: np.ndarray,
    tail_nodes: np.ndarray,
    *,
    target_count: int,
    target_nodes: np.ndarray | None,
    desired_same: bool,
    pair_seed: int,
    attempt: int,
    donor_random_tries: int,
) -> tuple[
    _GraphState,
    _GraphState,
    np.ndarray,
    np.ndarray,
    list[SwapRecord],
    list[SwapRecord],
]:
    seed_sequence = np.random.SeedSequence([int(pair_seed), int(attempt)])
    target_rng, order_rng, treatment_rng, control_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    ]

    source_same = not desired_same
    if target_nodes is None:
        targets = np.sort(
            target_rng.choice(
                tail_nodes, size=target_count, replace=False
            ).astype(np.int64)
        )
    else:
        targets = np.sort(target_nodes.astype(np.int64, copy=True))
    target_order = targets.copy()
    order_rng.shuffle(target_order)
    protected = np.zeros(labels.shape[0], dtype=bool)
    protected[targets] = True
    oriented = np.concatenate([canonical, canonical[:, ::-1]], axis=0)
    donor_pool = oriented[
        ~protected[oriented[:, 0]] & ~protected[oriented[:, 1]]
    ]
    treatment = _GraphState(original_edges, labels.shape[0])
    control = _GraphState(original_edges, labels.shape[0])
    treatment_swaps: list[SwapRecord] = []
    control_swaps: list[SwapRecord] = []

    for target_value in target_order:
        target = int(target_value)
        treatment_swap = _find_target_swap(
            treatment,
            donor_pool,
            labels,
            target=target,
            source_same=source_same,
            want_same=desired_same,
            rng=treatment_rng,
            random_tries=donor_random_tries,
            protected=protected,
        )
        control_swap = _find_target_swap(
            control,
            donor_pool,
            labels,
            target=target,
            source_same=source_same,
            want_same=source_same,
            rng=control_rng,
            random_tries=donor_random_tries,
            protected=protected,
        )
        if treatment_swap is None or control_swap is None:
            raise PairGenerationError(
                f"target {target} has no jointly valid treatment/control swap"
            )
        treatment_u, treatment_w, treatment_x = treatment_swap
        control_u, control_w, control_x = control_swap
        treatment.apply(
            target,
            treatment_u,
            treatment_w,
            treatment_x,
            allow_added_target_edge=True,
        )
        control.apply(
            target,
            control_u,
            control_w,
            control_x,
            allow_added_target_edge=True,
        )
        treatment_swaps.append(
            SwapRecord(target, treatment_u, treatment_w, treatment_x)
        )
        control_swaps.append(SwapRecord(target, control_u, control_w, control_x))

    return (
        treatment,
        control,
        targets,
        target_order,
        treatment_swaps,
        control_swaps,
    )


def _audit_pair(
    original_edges: set[tuple[int, int]],
    treatment: _GraphState,
    control: _GraphState,
    labels: np.ndarray,
    targets: np.ndarray,
    *,
    desired_same: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    num_nodes = int(labels.shape[0])
    original_edge_index = _directed_edge_index(original_edges)
    treatment_edge_index = _directed_edge_index(treatment.edges)
    control_edge_index = _directed_edge_index(control.edges)
    original_degree = _degree_vector(original_edges, num_nodes)
    treatment_degree = _degree_vector(treatment.edges, num_nodes)
    control_degree = _degree_vector(control.edges, num_nodes)

    original_h = local_homophily(original_edge_index, labels, num_nodes)
    treatment_h = local_homophily(treatment_edge_index, labels, num_nodes)
    control_h = local_homophily(control_edge_index, labels, num_nodes)
    expected_step = (1.0 if desired_same else -1.0) / original_degree[targets]
    observed_step = treatment_h[targets] - original_h[targets]

    checks = {
        "edge_count_preserved": (
            len(treatment.edges) == len(original_edges) == len(control.edges)
        ),
        "treatment_degrees_preserved": np.array_equal(
            treatment_degree, original_degree
        ),
        "control_degrees_preserved": np.array_equal(control_degree, original_degree),
        "treatment_homophily_step_exact": np.allclose(
            observed_step, expected_step, atol=1e-12, rtol=0.0
        ),
        "control_target_homophily_unchanged": np.allclose(
            control_h[targets], original_h[targets], atol=1e-12, rtol=0.0
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


def generate_rewire_pair(
    edge_index: np.ndarray,
    labels: np.ndarray,
    tail_mask: np.ndarray,
    *,
    desired_same: bool,
    pair_seed: int,
    sample_fraction: float = 0.5,
    target_count: int | None = None,
    target_nodes: np.ndarray | None = None,
    max_attempts: int = 100,
    donor_random_tries: int = 256,
) -> RewirePair:
    """Generate one complete degree-preserving treatment/control graph pair.

    ``desired_same=True`` raises target-node homophily by one neighbour;
    ``False`` lowers it.  ``target_nodes`` supplies an exact target set and is
    mutually exclusive with ``target_count``.  Failed attempts are discarded
    transactionally and a fresh deterministic attempt is tried without
    changing an explicit target set.
    """
    labels = np.asarray(labels, dtype=np.int64)
    tail_mask = np.asarray(tail_mask, dtype=bool)
    if labels.ndim != 1 or tail_mask.shape != labels.shape:
        raise ValueError("labels and tail_mask must be one-dimensional and aligned")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if target_count is not None and target_nodes is not None:
        raise ValueError("target_count and target_nodes are mutually exclusive")

    tail_nodes = np.flatnonzero(tail_mask).astype(np.int64)
    explicit_targets = None
    if target_nodes is not None:
        raw_targets = np.asarray(target_nodes)
        if raw_targets.ndim != 1:
            raise ValueError("target_nodes must be one-dimensional")
        if not np.issubdtype(raw_targets.dtype, np.integer):
            raise ValueError("target_nodes must contain integer node indices")
        explicit_targets = raw_targets.astype(np.int64, copy=True)
        if np.unique(explicit_targets).size != explicit_targets.size:
            raise ValueError("target_nodes must not contain duplicates")
        if np.any(explicit_targets < 0) or np.any(
            explicit_targets >= labels.size
        ):
            raise ValueError("target_nodes contains an out-of-range node index")
        if explicit_targets.size and not np.all(tail_mask[explicit_targets]):
            raise ValueError("every target_nodes entry must be a tail node")
        requested = int(explicit_targets.size)
    else:
        requested = (
            int(np.floor(sample_fraction * tail_nodes.size))
            if target_count is None
            else int(target_count)
        )
    if requested < 1 or requested > tail_nodes.size:
        raise ValueError(
            f"target_count must be between 1 and {tail_nodes.size}; got {requested}"
        )

    canonical, original_edges = _canonical_edges(edge_index, labels.size)
    failures: list[str] = []
    for attempt in range(max_attempts):
        try:
            (
                treatment,
                control,
                targets,
                target_order,
                treatment_swaps,
                control_swaps,
            ) = _attempt_pair(
                canonical,
                original_edges,
                labels,
                tail_nodes,
                target_count=requested,
                target_nodes=explicit_targets,
                desired_same=desired_same,
                pair_seed=pair_seed,
                attempt=attempt,
                donor_random_tries=donor_random_tries,
            )
            treatment_edge_index, control_edge_index, audit = _audit_pair(
                original_edges,
                treatment,
                control,
                labels,
                targets,
                desired_same=desired_same,
            )
            return RewirePair(
                treatment_edge_index=treatment_edge_index,
                control_edge_index=control_edge_index,
                targets=targets,
                target_order=target_order,
                treatment_swaps=treatment_swaps,
                control_swaps=control_swaps,
                pair_seed=int(pair_seed),
                attempt=attempt,
                requested_targets=requested,
                retry_failures=failures,
                audit=audit,
            )
        except PairGenerationError as error:
            failures.append(str(error))

    detail = failures[-1] if failures else "unknown failure"
    raise PairGenerationError(
        f"could not generate a complete pair after {max_attempts} attempts; "
        f"last failure: {detail}"
    )
