"""Construct and authenticate reverse-direction edge perturbation graphs.

The dose denominator is the original frozen tail, not its nonsaturated subset.
Only selected nodes have a guaranteed homophily change; donor-side changes are
measured and recorded. This runner does not load model checkpoints or infer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from hfd.edge_perturbation import PairGenerationError, generate_rewire_pair  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402

DESIGN_ID = "reverse-tail-edge-perturbation-graphs-v1"
SCHEMA_VERSION = 1
REPLICATES = (1, 2, 3, 4, 5)
SPECS = {
    "pubmed": dict(desired_same=False, seed_base=2026080100, tail_side="low-h_v", tail_count=2696),
    "ogbn-arxiv": dict(desired_same=False, seed_base=2026080300, tail_side="low-h_v", tail_count=30364),
    "arxiv-year": dict(desired_same=True, seed_base=2026080200, tail_side="high-h_v", tail_count=21869),
    "roman-empire": dict(desired_same=True, seed_base=2026080400, tail_side="high-h_v", tail_count=1012),
    "amazon-ratings": dict(desired_same=True, seed_base=2026080500, tail_side="high-h_v", tail_count=2943),
}
SOURCE_FILES = (
    "scripts/run_reverse_edge_graphs.py",
    "src/hfd/__init__.py",
    "src/hfd/edge_perturbation.py",
    "src/hfd/homophily.py",
    "src/hfd/datasets.py",
    "src/hfd/stats.py",
)
PREPARED_ARRAYS = {"edge_index", "labels", "tail_mask", "eligible_mask", "degrees", "same_counts"}
PAIR_ARRAYS = {"treatment_edge_index", "control_edge_index", "targets", "target_order", "tail_ranking"}


class AuditError(RuntimeError):
    """An invariant or existing authenticated artefact is inconsistent."""


class PairTimeBudgetExceeded(TimeoutError):
    """The construction's wall-clock budget expired; feasibility is unknown."""


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def software_versions() -> dict[str, str]:
    return {"python": ".".join(map(str, sys.version_info[:3])), "numpy": np.__version__}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def json_file_sha256(value: object) -> str:
    """Digest the exact UTF-8 representation written by atomic_json."""
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def immutable_json(path: Path, value: object) -> None:
    if path.exists():
        if read_json(path) != value:
            raise AuditError(f"existing {path.name} differs from the requested design")
    else:
        atomic_json(path, value)


def _files_exact(path: Path, expected: set[str]) -> None:
    observed = {item.name for item in path.iterdir()}
    if observed != expected:
        raise AuditError(f"{path.name}: unexpected or missing artefact names")


def file_record(path: Path) -> dict:
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def degree_and_same_counts(edges: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = len(labels)
    degrees = np.bincount(edges[0], minlength=num_nodes).astype(np.int64)
    same_edges = labels[edges[0]] == labels[edges[1]]
    same = np.bincount(edges[0, same_edges], minlength=num_nodes).astype(np.int64)
    return degrees, same


def eligible_tail_mask(tail_mask, degrees, same_counts, desired_same: bool) -> np.ndarray:
    tail = np.asarray(tail_mask, dtype=bool)
    degrees = np.asarray(degrees)
    same = np.asarray(same_counts)
    if tail.shape != degrees.shape or tail.shape != same.shape:
        raise AuditError("eligibility arrays are not aligned")
    if np.any(same < 0) or np.any(same > degrees):
        raise AuditError("invalid same-neighbour count")
    movable = same < degrees if desired_same else same > 0
    return tail & (degrees > 0) & movable


def dose_plan(tail_count: int, eligible_count: int) -> list[dict]:
    if tail_count <= 0 or not 0 <= eligible_count <= tail_count:
        raise ValueError("invalid original-tail or eligible count")
    by_count: dict[int, dict] = {}
    for percentage in range(5, 101, 5):
        count = tail_count * percentage // 100
        if count > 0 and count <= eligible_count and count not in by_count:
            by_count[count] = {
                "target_count": count,
                "nominal_pct": percentage,
                "actual_pct": 100.0 * count / tail_count,
                "kind": "five-percentage-point-grid",
            }
    if eligible_count and eligible_count not in by_count:
        by_count[eligible_count] = {
            "target_count": eligible_count,
            "nominal_pct": None,
            "actual_pct": 100.0 * eligible_count / tail_count,
            "kind": "exact-eligible-endpoint",
        }
    return [by_count[count] for count in sorted(by_count)]


def eligible_ranking(tail_mask: np.ndarray, eligible_mask: np.ndarray, seed: int) -> np.ndarray:
    tail = np.asarray(tail_mask, dtype=bool)
    eligible = np.asarray(eligible_mask, dtype=bool)
    if tail.shape != eligible.shape or np.any(eligible & ~tail):
        raise AuditError("eligible nodes are not a subset of the original tail")
    ranking = np.random.default_rng(int(seed)).permutation(np.flatnonzero(tail))
    return ranking[eligible[ranking]].astype(np.int64)


def validate_replicate_set(replicates: Sequence[int]) -> None:
    if sorted(replicates) != list(REPLICATES):
        raise AuditError("an accepted dose requires exactly five distinct replicates")


def _simple_graph(edges: np.ndarray, num_nodes: int) -> None:
    if edges.dtype != np.int64 or edges.ndim != 2 or edges.shape[0] != 2:
        raise AuditError("graph must be a two-row int64 edge array")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise AuditError("graph contains an invalid node index")
    if not np.array_equal(edges, to_simple_undirected(edges, num_nodes)):
        raise AuditError("graph is not canonically ordered simple undirected data")


def replay_swaps(original_edges: np.ndarray, records: list[dict], num_nodes: int) -> np.ndarray:
    """Replay ordinary double-edge swaps independently of the generator state."""
    _simple_graph(original_edges, num_nodes)
    edges = {(int(a), int(b)) for a, b in original_edges.T if a < b}
    for row in records:
        if set(row) != {"target", "removed_neighbour", "donor_near", "donor_far"}:
            raise AuditError("unexpected swap record fields")
        values = [row[k] for k in ("target", "removed_neighbour", "donor_near", "donor_far")]
        if any(type(value) is not int for value in values):
            raise AuditError("swap endpoints must be integers")
        v, u, w, x = values
        if len(set(values)) != 4 or min(values) < 0 or max(values) >= num_nodes:
            raise AuditError("invalid swap endpoints")
        removed = {tuple(sorted((v, u))), tuple(sorted((w, x)))}
        added = {tuple(sorted((v, w))), tuple(sorted((u, x)))}
        if not removed <= edges or added & edges:
            raise AuditError("swap replay removes an absent edge or adds an existing edge")
        edges.difference_update(removed)
        edges.update(added)
    canonical = np.asarray(sorted(edges), dtype=np.int64)
    if canonical.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack((np.concatenate((canonical[:, 0], canonical[:, 1])), np.concatenate((canonical[:, 1], canonical[:, 0]))))


def _change_summary(original_same, changed_same, degrees, mask) -> dict:
    selected = np.array(mask, dtype=bool, copy=True)
    selected &= degrees > 0
    changes = changed_same[selected] - original_same[selected]
    steps = changes / degrees[selected]
    return {
        "nodes": int(np.count_nonzero(selected)),
        "nodes_with_changed_homophily": int(np.count_nonzero(changes)),
        "nodes_with_increased_homophily": int(np.count_nonzero(changes > 0)),
        "nodes_with_decreased_homophily": int(np.count_nonzero(changes < 0)),
        "mean_homophily_change": float(steps.mean()) if steps.size else 0.0,
        "maximum_absolute_homophily_change": float(np.max(np.abs(steps))) if steps.size else 0.0,
    }


def audit_pair(original, labels, tail, eligible, targets, treatment, control, swaps, desired_same) -> dict:
    labels = np.asarray(labels)
    tail = np.asarray(tail, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    targets = np.asarray(targets)
    for edges in (original, treatment, control):
        _simple_graph(edges, len(labels))
    if targets.dtype != np.int64 or targets.ndim != 1 or targets.size < 1:
        raise AuditError("targets must be a nonempty int64 vector")
    if np.unique(targets).size != targets.size or targets.min() < 0 or targets.max() >= len(labels):
        raise AuditError("invalid or repeated targets")
    if np.any(~tail[targets]) or np.any(~eligible[targets]):
        raise AuditError("a selected target is not an eligible original-tail node")
    if set(swaps) != {"treatment", "control"}:
        raise AuditError("a matched pair needs treatment and control swap records")
    degree, original_same = degree_and_same_counts(original, labels)
    treatment_degree, treatment_same = degree_and_same_counts(treatment, labels)
    control_degree, control_same = degree_and_same_counts(control, labels)
    if treatment.shape != original.shape or control.shape != original.shape:
        raise AuditError("edge count changed")
    if not np.array_equal(degree, treatment_degree) or not np.array_equal(degree, control_degree):
        raise AuditError("node degrees changed")
    sign = 1 if desired_same else -1
    if not np.all(treatment_same[targets] - original_same[targets] == sign):
        raise AuditError("treatment did not change every target by exactly one same-label neighbour")
    if not np.array_equal(control_same[targets], original_same[targets]):
        raise AuditError("control changed target homophily")
    expected_targets = sorted(map(int, targets))
    protected = set(expected_targets)
    for condition, graph in (("treatment", treatment), ("control", control)):
        records = swaps[condition]
        replayed = replay_swaps(original, records, len(labels))
        if sorted(row["target"] for row in records) != expected_targets:
            raise AuditError("each target must have exactly one swap per condition")
        source_same = not desired_same
        wanted_same = desired_same if condition == "treatment" else source_same
        for row in records:
            v, u, w, x = (row[key] for key in ("target", "removed_neighbour", "donor_near", "donor_far"))
            if w in protected or x in protected:
                raise AuditError("a selected target was used as a donor endpoint")
            if (labels[u] == labels[v]) != source_same:
                raise AuditError("removed target edge has the wrong label relation")
            if (labels[w] == labels[v]) != wanted_same:
                raise AuditError("new target edge has the wrong label relation")
            if u in protected and (labels[x] == labels[u]) != (labels[v] == labels[u]):
                raise AuditError("a swap changes a protected neighbour's homophily")
        if not np.array_equal(replayed, graph):
            raise AuditError("recorded swaps do not reproduce the saved graph")
    unselected = tail.copy()
    unselected[targets] = False
    saturated = tail & ~eligible
    original_keys = set(map(tuple, original[:, original[0] < original[1]].T.tolist()))
    changed_counts = {}
    collateral = {}
    for condition, graph, same in (("treatment", treatment, treatment_same), ("control", control, control_same)):
        keys = set(map(tuple, graph[:, graph[0] < graph[1]].T.tolist()))
        changed_counts[condition] = len(original_keys.symmetric_difference(keys))
        collateral[condition] = {
            "unselected_tail": _change_summary(original_same, same, degree, unselected),
            "saturated_tail": _change_summary(original_same, same, degree, saturated),
            "non_tail": _change_summary(original_same, same, degree, ~tail),
        }
    return {
        "simple_undirected": True,
        "all_node_degrees_preserved": True,
        "edge_count_preserved": True,
        "treatment_target_same_count_change": sign,
        "control_target_same_count_change": 0,
        "every_target_processed_once_per_condition": True,
        "swaps_replay_exactly": True,
        "target_count": int(targets.size),
        "treatment_swap_count": len(swaps["treatment"]),
        "control_swap_count": len(swaps["control"]),
        "undirected_edge_symmetric_difference": changed_counts,
        "collateral_homophily_changes": collateral,
        "original_edge_sha256": array_sha256(original),
        "treatment_edge_sha256": array_sha256(treatment),
        "control_edge_sha256": array_sha256(control),
    }


@contextmanager
def pair_time_budget(seconds: float):
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("bounded graph construction requires POSIX interval timers")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise RuntimeError("cannot replace an existing process interval timer")

    def expire(signum, frame):
        raise PairTimeBudgetExceeded(f"pair construction exceeded its {seconds:g}-second budget")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def make_stop_record(dataset: str, dose: dict, replicate: int, error: Exception) -> dict:
    if isinstance(error, AuditError) or "generated pair failed audit" in str(error):
        raise AuditError(str(error)) from error
    if isinstance(error, PairTimeBudgetExceeded):
        reason = "resource_timeout"
    elif isinstance(error, PairGenerationError):
        reason = "construction_budget_exhausted"
    else:
        raise error
    return {
        "dataset": dataset,
        "dose": dose,
        "replicate": replicate,
        "reason": reason,
        "detail": str(error),
        "theoretical_maximum_proven": False,
        "policy": "omit this entire dose and stop higher doses; retain completed five-replicate doses",
    }


def _dataset_root(output_root: Path, dataset: str) -> Path:
    if dataset not in SPECS:
        raise ValueError("unknown dataset")
    return Path(output_root) / dataset


def prepare_dataset(args) -> dict:
    from hfd.datasets import load_dataset

    spec = SPECS[args.dataset]
    destination = _dataset_root(args.output_root, args.dataset) / "prepared"
    manifest_path = Path(args.manifest_root) / f"{args.dataset}.json"
    registered = read_json(manifest_path)
    if registered["dataset"] != args.dataset:
        raise AuditError("registered dataset identity differs")
    if registered["local_homophily"]["tail_side"] != spec["tail_side"]:
        raise AuditError("registered tail direction differs")
    if destination.exists():
        _, manifest = load_prepared(args.output_root, args.dataset)
        if manifest["registered_source_sha256"] != file_sha256(manifest_path):
            raise AuditError("registered manifest changed after preparation")
        frozen_path = Path(args.array_root) / f"{args.dataset}.npz"
        if manifest["frozen_array_file_sha256"] != file_sha256(frozen_path):
            raise AuditError("frozen input archive changed after preparation")
        print(f"PREPARED_REUSED {args.dataset}", flush=True)
        return manifest
    frozen_path = Path(args.array_root) / f"{args.dataset}.npz"
    with np.load(frozen_path, allow_pickle=False) as frozen:
        registered_hashes = registered["array_checksums_sha256"]
        if set(frozen.files) != set(registered_hashes):
            raise AuditError("frozen array names differ from registration")
        for name, digest in registered_hashes.items():
            if array_sha256(frozen[name]) != digest:
                raise AuditError(f"registered frozen array hash differs: {name}")
        bundle = load_dataset(args.dataset, Path(args.data_root))
        labels = np.asarray(bundle.y, dtype=np.int64)
        if np.any(labels < 0) or not np.array_equal(labels, frozen["labels"]):
            raise AuditError("live labels differ or include unsupported unlabelled nodes")
        graph_hashes = registered["graph_checksums_sha256"]
        for required in ("features", "simple_edge_index"):
            if required not in graph_hashes:
                raise AuditError(f"registration lacks required {required} hash")
        if array_sha256(bundle.x.detach().cpu().numpy()) != graph_hashes["features"]:
            raise AuditError("live feature hash differs")
        edges = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
        if array_sha256(edges) != graph_hashes["simple_edge_index"]:
            raise AuditError("live simple graph hash differs")
        if list(bundle.training_seeds) != registered["training_seeds"]:
            raise AuditError("live training seed panel differs")
        for split in bundle.splits:
            for partition, values in (("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)):
                if not np.array_equal(values, frozen[f"{split.split_id}.{partition}"]):
                    raise AuditError("live split indices differ from frozen inputs")
        tail = np.asarray(frozen["tail_mask"], dtype=bool)
        if int(tail.sum()) != spec["tail_count"]:
            raise AuditError("original frozen tail count differs")
        degree, same = degree_and_same_counts(edges, labels)
        h = np.full(labels.shape, np.nan)
        np.divide(same, degree, out=h, where=degree > 0)
        if not np.allclose(h, frozen["h"], rtol=0, atol=1e-12, equal_nan=True):
            raise AuditError("live homophily differs from frozen inputs")
        eligible = eligible_tail_mask(tail, degree, same, spec["desired_same"])
        arrays = dict(edge_index=edges, labels=labels, tail_mask=tail, eligible_mask=eligible, degrees=degree, same_counts=same)
        del bundle
    pending = destination.with_name("prepared.in-progress")
    if pending.exists():
        raise AuditError("incomplete preparation requires inspection")
    pending.mkdir(parents=True)
    np.savez_compressed(pending / "inputs.npz", **arrays)
    registered_bytes = manifest_path.read_bytes()
    (pending / "registered-manifest.json").write_bytes(registered_bytes)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "dataset": args.dataset,
        "desired_same": spec["desired_same"],
        "tail_side": spec["tail_side"],
        "original_tail_count": int(tail.sum()),
        "eligible_tail_count": int(eligible.sum()),
        "eligibility_only_upper_pct": 100.0 * int(eligible.sum()) / int(tail.sum()),
        "denominator": "original frozen tail-node count",
        "eligible_rule": "positive degree and same-neighbour count below degree" if spec["desired_same"] else "positive degree and positive same-neighbour count",
        "seed_base": spec["seed_base"],
        "replicates": list(REPLICATES),
        "dose_plan": dose_plan(int(tail.sum()), int(eligible.sum())),
        "source_sha256": source_hashes(),
        "software_versions": software_versions(),
        "registered_source_sha256": file_sha256(manifest_path),
        "frozen_array_file_sha256": file_sha256(frozen_path),
        "registered_frozen_array_sha256": registered_hashes,
        "registered_graph_sha256": graph_hashes,
        "prepared_array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
        "files": {name: file_record(pending / name) for name in ("inputs.npz", "registered-manifest.json")},
    }
    atomic_json(pending / "manifest.json", manifest)
    pending.replace(destination)
    print(f"PREPARED {args.dataset} eligible={int(eligible.sum())}/{int(tail.sum())} endpoint={manifest['eligibility_only_upper_pct']:.8f}%", flush=True)
    return manifest


def load_prepared(output_root: Path, dataset: str) -> tuple[dict, dict]:
    path = _dataset_root(output_root, dataset) / "prepared"
    _files_exact(path, {"inputs.npz", "registered-manifest.json", "manifest.json"})
    manifest = read_json(path / "manifest.json")
    if manifest.get("design_id") != DESIGN_ID or manifest.get("dataset") != dataset or manifest.get("source_sha256") != source_hashes():
        raise AuditError("prepared design, dataset, or implementation hashes differ")
    if manifest.get("software_versions") != software_versions():
        raise AuditError("prepared Python or NumPy version differs")
    if manifest["desired_same"] != SPECS[dataset]["desired_same"] or manifest["seed_base"] != SPECS[dataset]["seed_base"]:
        raise AuditError("prepared direction or replicate seeds differ")
    for name, expected in manifest["files"].items():
        if name not in {"inputs.npz", "registered-manifest.json"} or file_record(path / name) != expected:
            raise AuditError("prepared file hash differs")
    registered = read_json(path / "registered-manifest.json")
    if file_sha256(path / "registered-manifest.json") != manifest["registered_source_sha256"]:
        raise AuditError("registered manifest hash differs")
    with np.load(path / "inputs.npz", allow_pickle=False) as stored:
        if set(stored.files) != PREPARED_ARRAYS:
            raise AuditError("prepared array names differ")
        arrays = {name: stored[name] for name in stored.files}
    if {name: array_sha256(value) for name, value in arrays.items()} != manifest["prepared_array_sha256"]:
        raise AuditError("prepared array hashes differ")
    _simple_graph(arrays["edge_index"], len(arrays["labels"]))
    degrees, same = degree_and_same_counts(arrays["edge_index"], arrays["labels"])
    if not np.array_equal(degrees, arrays["degrees"]) or not np.array_equal(same, arrays["same_counts"]):
        raise AuditError("prepared neighbour counts differ")
    eligible = eligible_tail_mask(arrays["tail_mask"], degrees, same, manifest["desired_same"])
    if not np.array_equal(eligible, arrays["eligible_mask"]):
        raise AuditError("prepared eligibility differs")
    for name in ("labels", "tail_mask"):
        if array_sha256(arrays[name]) != registered["array_checksums_sha256"][name]:
            raise AuditError("prepared labels or tail differ from registration")
    if array_sha256(arrays["edge_index"]) != registered["graph_checksums_sha256"]["simple_edge_index"]:
        raise AuditError("prepared graph differs from registration")
    if manifest["original_tail_count"] != int(arrays["tail_mask"].sum()) or manifest["eligible_tail_count"] != int(eligible.sum()):
        raise AuditError("prepared original-tail denominator or eligible count differs")
    if manifest["dose_plan"] != dose_plan(manifest["original_tail_count"], manifest["eligible_tail_count"]):
        raise AuditError("prepared doses differ")
    return arrays, manifest


def _experiment_manifest(prepared: dict, args, phase: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "phase": phase,
        "dataset": args.dataset,
        "prepared_manifest_sha256": file_sha256(_dataset_root(args.output_root, args.dataset) / "prepared" / "manifest.json"),
        "source_sha256": prepared["source_sha256"],
        "software_versions": prepared["software_versions"],
        "desired_same": prepared["desired_same"],
        "original_tail_count": prepared["original_tail_count"],
        "eligible_tail_count": prepared["eligible_tail_count"],
        "denominator": "original frozen tail-node count",
        "dose_plan": prepared["dose_plan"],
        "replicate_indices": list(REPLICATES),
        "seed_base": prepared["seed_base"],
        "seed_rule": "seed_base + replicate_index",
        "ranking_rule": "default_rng(replicate_seed).permutation(original_tail_indices), then filter eligible nodes",
        "target_rule": "nested prefixes of the filtered original-tail permutation",
        "graph_start_rule": "registered original graph independently for every dose and replicate",
        "generator": "ordinary signed degree-preserving double-edge swaps",
        "collateral_rule": "only selected targets are protected; unselected and saturated nodes may change as donors",
        "max_attempts": int(args.max_attempts),
        "pair_timeout_seconds": float(args.pair_timeout_seconds),
        "budget_interpretation": "computational limits, not mathematical feasibility limits",
        "acceptance_rule": "exactly five complete authenticated pairs per accepted dose",
        "stop_rule": "first unsupported pair omits entire dose and stops higher doses",
    }


def _baseline(prepared: dict, arrays: dict) -> dict:
    return {
        "design_id": DESIGN_ID,
        "target_count": 0,
        "actual_pct": 0.0,
        "nominal_pct": 0,
        "original_tail_count": prepared["original_tail_count"],
        "replicate_indices": list(REPLICATES),
        "treatment_equals_control_equals_original": True,
        "original_graph_reference": "../prepared/inputs.npz:edge_index",
        "original_edge_sha256": array_sha256(arrays["edge_index"]),
    }


def _pair_metadata(arrays, prepared, experiment, dose, replicate, pair, seconds, audit, hashes) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": DESIGN_ID,
        "dataset": prepared["dataset"],
        "phase": experiment["phase"],
        "replicate_index": replicate,
        "replicate_seed": prepared["seed_base"] + replicate,
        **dose,
        "original_tail_count": prepared["original_tail_count"],
        "eligible_tail_count": prepared["eligible_tail_count"],
        "desired_same": prepared["desired_same"],
        "generation_attempt": pair.attempt,
        "generation_retry_failures": pair.retry_failures,
        "generation_seconds": seconds,
        "experiment_manifest_sha256": json_file_sha256(experiment),
        "array_sha256": hashes,
        "audit": audit,
    }


def _generate_pair(path, arrays, prepared, experiment, dose, replicate) -> dict:
    if path.exists():
        return _verify_pair(path, arrays, prepared, experiment, dose, replicate)
    seed = prepared["seed_base"] + replicate
    ranking = eligible_ranking(arrays["tail_mask"], arrays["eligible_mask"], seed)
    targets = ranking[:dose["target_count"]]
    started = time.perf_counter()
    with pair_time_budget(experiment["pair_timeout_seconds"]):
        pair = generate_rewire_pair(
            arrays["edge_index"], arrays["labels"], arrays["tail_mask"],
            desired_same=prepared["desired_same"], pair_seed=seed, target_nodes=targets,
            max_attempts=experiment["max_attempts"],
        )
    seconds = time.perf_counter() - started
    if any("generated pair failed audit" in item for item in pair.retry_failures):
        raise AuditError("generator audit failed during a retry")
    if not np.array_equal(np.sort(pair.targets), np.sort(targets)):
        raise AuditError("generator silently changed requested targets")
    swaps = pair.swap_records()
    audit = audit_pair(arrays["edge_index"], arrays["labels"], arrays["tail_mask"], arrays["eligible_mask"], targets, pair.treatment_edge_index, pair.control_edge_index, swaps, prepared["desired_same"])
    pending = path.with_name(path.name + ".in-progress")
    if pending.exists():
        raise AuditError("incomplete pair serialization requires inspection")
    pending.mkdir(parents=True)
    output_arrays = dict(treatment_edge_index=pair.treatment_edge_index, control_edge_index=pair.control_edge_index, targets=targets, target_order=pair.target_order, tail_ranking=ranking)
    np.savez_compressed(pending / "graphs.npz", **output_arrays)
    atomic_json(pending / "swaps.json", swaps)
    metadata = _pair_metadata(arrays, prepared, experiment, dose, replicate, pair, seconds, audit, {name: array_sha256(value) for name, value in output_arrays.items()})
    metadata["files"] = {name: file_record(pending / name) for name in ("graphs.npz", "swaps.json")}
    atomic_json(pending / "metadata.json", metadata)
    pending.replace(path)
    print(f"PAIR_COMPLETE {prepared['dataset']} targets={dose['target_count']} actual_pct={dose['actual_pct']:.8f} replicate={replicate} seconds={seconds:.2f}", flush=True)
    return metadata


def _verify_pair(path, arrays, prepared, experiment, dose, replicate) -> dict:
    _files_exact(path, {"graphs.npz", "swaps.json", "metadata.json"})
    metadata = read_json(path / "metadata.json")
    expected = {
        "schema_version": SCHEMA_VERSION, "design_id": DESIGN_ID,
        "dataset": prepared["dataset"], "phase": experiment["phase"],
        "replicate_index": replicate, "replicate_seed": prepared["seed_base"] + replicate,
        **dose, "original_tail_count": prepared["original_tail_count"],
        "eligible_tail_count": prepared["eligible_tail_count"], "desired_same": prepared["desired_same"],
        "experiment_manifest_sha256": json_file_sha256(experiment),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise AuditError("saved pair identity or denominator differs")
    attempt = metadata.get("generation_attempt")
    failures = metadata.get("generation_retry_failures")
    if type(attempt) is not int or not 0 <= attempt < experiment["max_attempts"] or not isinstance(failures, list) or len(failures) != attempt:
        raise AuditError("saved retry history differs")
    if any("generated pair failed audit" in item for item in failures):
        raise AuditError("saved retry history contains an audit failure")
    if set(metadata.get("files", {})) != {"graphs.npz", "swaps.json"}:
        raise AuditError("saved pair file manifest differs")
    for name, expected_record in metadata["files"].items():
        if file_record(path / name) != expected_record:
            raise AuditError("saved graph or swap file hash differs")
    with np.load(path / "graphs.npz", allow_pickle=False) as stored:
        if set(stored.files) != PAIR_ARRAYS:
            raise AuditError("saved graph array names differ")
        values = {name: stored[name] for name in stored.files}
    if {name: array_sha256(value) for name, value in values.items()} != metadata["array_sha256"]:
        raise AuditError("saved graph array hash differs")
    ranking = eligible_ranking(arrays["tail_mask"], arrays["eligible_mask"], expected["replicate_seed"])
    targets = ranking[:dose["target_count"]]
    if not np.array_equal(values["tail_ranking"], ranking) or not np.array_equal(values["targets"], targets):
        raise AuditError("saved seed ranking or selected prefix differs")
    if not np.array_equal(np.sort(values["target_order"]), np.sort(targets)):
        raise AuditError("target processing order differs")
    swaps = read_json(path / "swaps.json")
    if values["target_order"].tolist() != [row["target"] for row in swaps["treatment"]] or values["target_order"].tolist() != [row["target"] for row in swaps["control"]]:
        raise AuditError("swap records disagree with target processing order")
    audit = audit_pair(arrays["edge_index"], arrays["labels"], arrays["tail_mask"], arrays["eligible_mask"], targets, values["treatment_edge_index"], values["control_edge_index"], swaps, prepared["desired_same"])
    if metadata.get("audit") != audit:
        raise AuditError("saved graph audit differs from independent recomputation")
    return metadata


def _dose_name(dose: dict) -> str:
    return f"targets-{dose['target_count']:06d}"


def _acceptance(path, dose, metadata) -> dict:
    validate_replicate_set([row["replicate_index"] for row in metadata])
    return {"design_id": DESIGN_ID, "dose": dose, "replicate_indices": list(REPLICATES), "pairs": {f"replicate-{replicate:04d}": file_record(path / f"replicate-{replicate:04d}" / "metadata.json") for replicate in REPLICATES}}


def generate_dataset(args) -> dict:
    arrays, prepared = load_prepared(args.output_root, args.dataset)
    actual = _dataset_root(args.output_root, args.dataset) / "actual"
    experiment = _experiment_manifest(prepared, args, "actual")
    immutable_json(actual / "experiment_manifest.json", experiment)
    immutable_json(actual / "baseline.json", _baseline(prepared, arrays))
    if (actual / "stopped.json").exists():
        print(f"GENERATION_PREVIOUSLY_STOPPED {args.dataset}; existing cutoff retained", flush=True)
        return verify_dataset(args)
    for dose in prepared["dose_plan"]:
        final = actual / "doses" / _dose_name(dose)
        pending = final.with_name(final.name + ".in-progress")
        if final.exists():
            metadata = [_verify_pair(final / f"replicate-{replicate:04d}", arrays, prepared, experiment, dose, replicate) for replicate in REPLICATES]
            _files_exact(final, {"accepted.json", *(f"replicate-{replicate:04d}" for replicate in REPLICATES)})
            if read_json(final / "accepted.json") != _acceptance(final, dose, metadata):
                raise AuditError("accepted dose marker differs")
            continue
        pending.mkdir(parents=True, exist_ok=True)
        if (pending / "failure.json").exists():
            raise AuditError("dose failure exists without the terminal stop record")
        metadata = []
        for replicate in REPLICATES:
            try:
                metadata.append(_generate_pair(pending / f"replicate-{replicate:04d}", arrays, prepared, experiment, dose, replicate))
            except (PairGenerationError, PairTimeBudgetExceeded) as error:
                stopped = make_stop_record(args.dataset, dose, replicate, error)
                stopped["max_attempts"] = experiment["max_attempts"]
                stopped["pair_timeout_seconds"] = experiment["pair_timeout_seconds"]
                atomic_json(pending / "failure.json", stopped)
                atomic_json(actual / "stopped.json", stopped)
                print(f"GENERATION_CAPPED {args.dataset} unsupported_targets={dose['target_count']} replicate={replicate} reason={stopped['reason']}", flush=True)
                return verify_dataset(args)
        immutable_json(pending / "accepted.json", _acceptance(pending, dose, metadata))
        pending.replace(final)
        print(f"DOSE_ACCEPTED {args.dataset} targets={dose['target_count']} actual_pct={dose['actual_pct']:.8f} replicates=5", flush=True)
    print(f"GENERATION_COMPLETE {args.dataset}", flush=True)
    return verify_dataset(args)


def smoke_dataset(args) -> dict:
    arrays, prepared = load_prepared(args.output_root, args.dataset)
    root = _dataset_root(args.output_root, args.dataset) / "smoke"
    experiment = _experiment_manifest(prepared, args, "smoke")
    immutable_json(root / "experiment_manifest.json", experiment)
    plan = prepared["dose_plan"]
    selected = [plan[0], plan[-1]] if len(plan) > 1 else plan[:1]
    results = []
    for dose in selected:
        final = root / _dose_name(dose) / "replicate-0001"
        failure_path = root / _dose_name(dose) / "failure.json"
        if failure_path.exists():
            previous = read_json(failure_path)
            if previous.get("dose") != dose or previous.get("dataset") != args.dataset or final.exists():
                raise AuditError("saved smoke failure differs")
            results.append({"dose": dose, "constructed": False, "failure": previous})
            continue
        try:
            metadata = _generate_pair(final, arrays, prepared, experiment, dose, 1)
            results.append({"dose": dose, "constructed": True, "metadata_sha256": file_sha256(final / "metadata.json"), "audit": metadata["audit"]})
        except (PairGenerationError, PairTimeBudgetExceeded) as error:
            failure = make_stop_record(args.dataset, dose, 1, error)
            failure["policy"] = "smoke result only; endpoint construction failure does not change the actual dose plan"
            atomic_json(failure_path, failure)
            results.append({"dose": dose, "constructed": False, "failure": failure})
            print(f"SMOKE_CONSTRUCTION_LIMIT {args.dataset} targets={dose['target_count']} reason={failure['reason']}", flush=True)
    report = {"design_id": DESIGN_ID, "dataset": args.dataset, "correctness_checks_passed": True, "all_attempted_pairs_constructed": all(row["constructed"] for row in results), "results": results}
    immutable_json(root / "validation.json", report)
    print(f"SMOKE_COMPLETE {args.dataset} constructed={sum(row['constructed'] for row in results)}/{len(results)} correctness_checks_passed=true", flush=True)
    return report


def verify_dataset(args) -> dict:
    arrays, prepared = load_prepared(args.output_root, args.dataset)
    actual = _dataset_root(args.output_root, args.dataset) / "actual"
    experiment = read_json(actual / "experiment_manifest.json")
    if experiment.get("phase") != "actual" or experiment.get("dataset") != args.dataset:
        raise AuditError("actual experiment manifest identity differs")
    validation_args = argparse.Namespace(**vars(args))
    validation_args.max_attempts = experiment["max_attempts"]
    validation_args.pair_timeout_seconds = experiment["pair_timeout_seconds"]
    if experiment != _experiment_manifest(prepared, validation_args, "actual"):
        raise AuditError("actual experiment settings differ")
    if read_json(actual / "baseline.json") != _baseline(prepared, arrays):
        raise AuditError("baseline differs from the original graph")
    stopped = read_json(actual / "stopped.json") if (actual / "stopped.json").exists() else None
    accepted = []
    entries = []
    encountered_gap = False
    expected_dirs = set()
    for dose in prepared["dose_plan"]:
        path = actual / "doses" / _dose_name(dose)
        if not path.exists():
            encountered_gap = True
            continue
        if encountered_gap:
            raise AuditError("accepted doses are not a contiguous prefix of the planned doses")
        expected_dirs.add(path.name)
        _files_exact(path, {"accepted.json", *(f"replicate-{replicate:04d}" for replicate in REPLICATES)})
        metadata = []
        for replicate in REPLICATES:
            pair_path = path / f"replicate-{replicate:04d}"
            row = _verify_pair(pair_path, arrays, prepared, experiment, dose, replicate)
            metadata.append(row)
            entries.append({"dose": dose, "replicate_index": replicate, "replicate_seed": prepared["seed_base"] + replicate, "relative_directory": pair_path.relative_to(actual).as_posix(), "files": {name: file_record(pair_path / name) for name in ("graphs.npz", "swaps.json", "metadata.json")}})
        if read_json(path / "accepted.json") != _acceptance(path, dose, metadata):
            raise AuditError("accepted five-replicate dose marker differs")
        accepted.append(dose)
    complete = len(accepted) == len(prepared["dose_plan"])
    if stopped is not None:
        if complete or stopped.get("dataset") != args.dataset or stopped.get("theoretical_maximum_proven") is not False:
            raise AuditError("invalid terminal construction-limit record")
        next_dose = prepared["dose_plan"][len(accepted)]
        if stopped.get("dose") != next_dose or stopped.get("replicate") not in REPLICATES or stopped.get("reason") not in {"resource_timeout", "construction_budget_exhausted"}:
            raise AuditError("terminal limit is not the first unsupported planned dose")
        pending = actual / "doses" / (_dose_name(next_dose) + ".in-progress")
        if read_json(pending / "failure.json") != stopped:
            raise AuditError("failed dose and terminal record differ")
        expected_dirs.add(pending.name)
    elif not complete:
        raise AuditError("generation is incomplete and has no recorded construction limit")
    dose_root = actual / "doses"
    if dose_root.exists() and {path.name for path in dose_root.iterdir()} != expected_dirs:
        raise AuditError("unexpected or incomplete dose directories remain")
    catalog = {
        "schema_version": SCHEMA_VERSION, "design_id": DESIGN_ID, "dataset": args.dataset,
        "original_tail_count": prepared["original_tail_count"], "eligible_tail_count": prepared["eligible_tail_count"],
        "denominator": "original frozen tail-node count", "desired_same": prepared["desired_same"],
        "prepared_manifest": file_record(_dataset_root(args.output_root, args.dataset) / "prepared" / "manifest.json"),
        "experiment_manifest": file_record(actual / "experiment_manifest.json"),
        "source_sha256": prepared["source_sha256"], "baseline": _baseline(prepared, arrays),
        "accepted_doses": accepted, "pairs": entries, "pair_count": len(entries),
        "all_planned_doses_constructed": complete, "construction_limit": stopped,
        "maximum_accepted_pct": accepted[-1]["actual_pct"] if accepted else 0.0,
        "theoretical_maximum_proven": False,
    }
    immutable_json(actual / "catalog.json", catalog)
    report = {
        "schema_version": SCHEMA_VERSION, "design_id": DESIGN_ID, "dataset": args.dataset,
        "validated": True, "pair_count": len(entries), "accepted_nonzero_dose_count": len(accepted),
        "replicates_per_accepted_dose": 5, "baseline_replicates": list(REPLICATES),
        "no_partial_replicate_averages": True, "original_tail_denominator_verified": True,
        "independent_swap_replay_verified": True, "all_planned_doses_constructed": complete,
        "maximum_accepted_pct": catalog["maximum_accepted_pct"], "catalog": file_record(actual / "catalog.json"),
        "construction_limit_reason": stopped["reason"] if stopped else None,
    }
    immutable_json(actual / "validation.json", report)
    print(f"VALIDATED {args.dataset} pairs={len(entries)} doses={len(accepted)} max_pct={catalog['maximum_accepted_pct']:.8f} planned_complete={str(complete).lower()}", flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "smoke", "generate", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=tuple(SPECS), required=True)
        sub.add_argument("--output-root", type=Path, required=True)
        if command == "prepare":
            for name in ("data-root", "array-root", "manifest-root"):
                sub.add_argument(f"--{name}", type=Path, required=True)
        if command in ("smoke", "generate"):
            sub.add_argument("--max-attempts", type=int, default=1000)
            sub.add_argument("--pair-timeout-seconds", type=float, default=120.0 if command == "smoke" else 180.0)
    args = parser.parse_args(argv)
    if getattr(args, "max_attempts", 1) < 1 or getattr(args, "pair_timeout_seconds", 1) <= 0:
        parser.error("construction budgets must be positive")
    return args


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    return {"prepare": prepare_dataset, "smoke": smoke_dataset, "generate": generate_dataset, "verify": verify_dataset}[args.command](args)


if __name__ == "__main__":
    main()
