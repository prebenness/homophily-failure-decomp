"""Generate, authenticate, evaluate, and validate Appendix-J dose responses.

The accepted ERM panel contains the 24 model--dataset cells that were not run
before submission.  Recovery preparations (last, head, and full) use the same
graphs and the complete 6x5 non-MLP matrix. Existing PubMed, arxiv-year, and
ogbn-arxiv graph pairs are authenticated and reused. Roman-empire and
Amazon-ratings graph pairs are generated separately with frozen seeds.
Evaluation outputs are separate from graph inputs and are written as one
atomic, resumable shard per selected training checkpoint.

The execution order is deliberate:

* GCN, GraphSAGE-mean, ACM-GCN, GPR-GNN, and LINKX are checkpoint-major.  A
  checkpoint is loaded exactly once and is then evaluated on the original
  graph and every retained treatment/control pair.  Roman-empire stops at
  70%; the other datasets retain the full 5--100% dose range.
* H2GCN is graph-major.  The one-hop and two-hop operators are built once for
  each treatment or control graph and reused across all pending checkpoints.

The ERM validator retains its accepted 720-baseline, 66,600-pair contract. Each
recovery preparation requires 900 baselines and 84,600 paired rows. A paired
row contains treatment and control results for one checkpoint, replicate, and
non-zero perturbation percentage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hfd.datasets import load_dataset  # noqa: E402
from hfd.edge_perturbation import generate_rewire_pair  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.metrics import accuracy, macro_f1  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.stats import _sha256 as array_sha256  # noqa: E402
from run_edge_perturbation import (  # noqa: E402
    lower_process_priority,
    validate_fixed_inputs,
    validate_parent_run,
)


SCHEMA_VERSION = 1
GRAPH_CATALOG_SCHEMA_VERSION = 2
DESIGN_ID = "appendix-j-edge-dose-response-expansion-v1"
GRAPH_DESIGN_ID = "appendix-j-edge-dose-response-generated-graphs-v1"
PREPARATIONS = ("erm", "last", "head", "full")
FULL_MODELS = (
    "gcn",
    "graphsage-mean",
    "acm-gcn",
    "gpr-gnn",
    "h2gcn",
    "linkx",
)
ACCEPTED_RELEASE_EXPERIMENT_REVISION = (
    "f66a53999fc01ae52a1e4aeaaf618409abab8fa7"
)
PERCENTAGES = tuple(range(0, 101, 5))
NONZERO_PERCENTAGES = PERCENTAGES[1:]
ROMAN_PERCENTAGES = tuple(range(0, 71, 5))
REPLICATE_INDICES = tuple(range(1, 6))
LOGIT_REPRODUCTION_ATOL = 1e-5
LOGIT_REPRODUCTION_RTOL = 1e-5
GRAPH_FILES = ("graphs.npz", "swaps.json", "metadata.json")
GRAPH_ARRAY_KEYS = {
    "treatment_edge_index",
    "control_edge_index",
    "targets",
    "target_order",
    "tail_ranking",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    models: tuple[str, ...]
    tail_count: int
    tail_side: str
    desired_same: bool
    generator: str
    split_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    seed_base: int
    percentages: tuple[int, ...] = PERCENTAGES
    generation_max_attempts: int = 1
    reused_manifest_sha256: str | None = None

    @property
    def checkpoint_identities(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, seed)
            for split in self.split_ids
            for seed in self.training_seeds
        )

    @property
    def graph_is_reused(self) -> bool:
        return self.reused_manifest_sha256 is not None

    @property
    def nonzero_percentages(self) -> tuple[int, ...]:
        return self.percentages[1:]

    @property
    def pairs_per_checkpoint(self) -> int:
        return len(self.nonzero_percentages) * len(REPLICATE_INDICES)


DATASET_SPECS = {
    "pubmed": DatasetSpec(
        dataset="pubmed",
        models=("acm-gcn", "gpr-gnn", "h2gcn", "linkx"),
        tail_count=2696,
        tail_side="low-h_v",
        desired_same=True,
        generator="singleton",
        split_ids=tuple(f"geomgcn-{index}" for index in range(10)),
        training_seeds=(1, 2, 3),
        seed_base=2026080100,
        reused_manifest_sha256=(
            "ac11f23c139ee876295fbc4e324ba16c31382fe8a3ce49b63303148f1e95a118"
        ),
    ),
    "ogbn-arxiv": DatasetSpec(
        dataset="ogbn-arxiv",
        models=("acm-gcn", "gpr-gnn", "h2gcn", "linkx"),
        tail_count=30364,
        tail_side="low-h_v",
        desired_same=True,
        generator="coordinated",
        split_ids=("ogb-time",),
        training_seeds=tuple(range(1, 31)),
        seed_base=2026080300,
        reused_manifest_sha256=(
            "fb4d4b1f586df98f8fa884736c428ff6008aca494bc080ae824d30043ceaa834"
        ),
    ),
    "arxiv-year": DatasetSpec(
        dataset="arxiv-year",
        models=("acm-gcn", "gpr-gnn", "h2gcn", "linkx"),
        tail_count=21869,
        tail_side="high-h_v",
        desired_same=False,
        generator="singleton",
        split_ids=tuple(f"lim-{index}" for index in range(5)),
        training_seeds=tuple(range(1, 7)),
        seed_base=2026080200,
        reused_manifest_sha256=(
            "65503d3519824431bd4d7c70fe4138f61cb1ab533a4b9ba53c8367c90d5aad4b"
        ),
    ),
    "roman-empire": DatasetSpec(
        dataset="roman-empire",
        models=(
            "gcn",
            "graphsage-mean",
            "acm-gcn",
            "gpr-gnn",
            "h2gcn",
            "linkx",
        ),
        tail_count=1012,
        tail_side="high-h_v",
        desired_same=False,
        generator="singleton",
        split_ids=tuple(f"platonov-{index}" for index in range(10)),
        training_seeds=(1, 2, 3),
        seed_base=2026080400,
        percentages=ROMAN_PERCENTAGES,
        generation_max_attempts=1000,
    ),
    "amazon-ratings": DatasetSpec(
        dataset="amazon-ratings",
        models=(
            "gcn",
            "graphsage-mean",
            "acm-gcn",
            "gpr-gnn",
            "h2gcn",
            "linkx",
        ),
        tail_count=2943,
        tail_side="high-h_v",
        desired_same=False,
        generator="singleton",
        split_ids=tuple(f"platonov-{index}" for index in range(10)),
        training_seeds=(1, 2, 3),
        seed_base=2026080500,
    ),
}

MODEL_DATASET_CELLS = tuple(
    (dataset, model)
    for dataset, spec in DATASET_SPECS.items()
    for model in spec.models
)
EXPECTED_CHECKPOINTS = sum(
    len(spec.models) * len(spec.checkpoint_identities)
    for spec in DATASET_SPECS.values()
)
EXPECTED_PAIRED_ROWS = sum(
    len(spec.models) * len(spec.checkpoint_identities) * spec.pairs_per_checkpoint
    for spec in DATASET_SPECS.values()
)


def design_id_for_preparation(preparation: str) -> str:
    """Return the immutable evaluation identity for one preparation panel."""

    if preparation not in PREPARATIONS:
        raise ValueError(f"unknown preparation {preparation!r}")
    if preparation == "erm":
        return DESIGN_ID
    return f"appendix-j-edge-dose-response-{preparation}-full-matrix-v1"


def models_for_preparation(
    spec: DatasetSpec, preparation: str
) -> tuple[str, ...]:
    """Keep the accepted ERM expansion panel; use the full graph panel otherwise."""

    if preparation not in PREPARATIONS:
        raise ValueError(f"unknown preparation {preparation!r}")
    return spec.models if preparation == "erm" else FULL_MODELS


def model_dataset_cells_for_preparation(
    preparation: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (dataset, model)
        for dataset, spec in DATASET_SPECS.items()
        for model in models_for_preparation(spec, preparation)
    )


def expected_checkpoint_count(preparation: str) -> int:
    return sum(
        len(models_for_preparation(spec, preparation))
        * len(spec.checkpoint_identities)
        for spec in DATASET_SPECS.values()
    )


def expected_paired_row_count(preparation: str) -> int:
    return sum(
        len(models_for_preparation(spec, preparation))
        * len(spec.checkpoint_identities)
        * spec.pairs_per_checkpoint
        for spec in DATASET_SPECS.values()
    )


def preparation_from_args(args) -> str:
    """Treat old programmatic callers as ERM for backwards compatibility."""

    return str(getattr(args, "preparation", "erm"))


def evaluation_design_id(args) -> str:
    return design_id_for_preparation(preparation_from_args(args))


@dataclass(frozen=True)
class CheckpointSpec:
    dataset: str
    model: str
    split: str
    seed: int
    run_id: str
    model_config: dict
    checkpoint_path: Path
    run_data_path: Path
    checkpoint_sha256: str
    training_code_version: str
    saved_predictions: np.ndarray
    preparation: str = "erm"


@dataclass(frozen=True)
class H2Progress:
    rows_by_run: dict[str, dict]
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def reset_cuda_peak_memory(device: torch.device) -> None:
    """Start a new CUDA peak-memory measurement interval when applicable."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory(device: torch.device, *, scope: str) -> dict:
    """Return stable byte counters for the current measurement interval."""

    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        allocated = 0
        reserved = 0
    return {
        "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
        "cuda_peak_measurement_scope": scope,
    }


def validate_cuda_peak_memory(payload: dict, *, context: str) -> None:
    allocated = payload.get("cuda_peak_allocated_bytes")
    reserved = payload.get("cuda_peak_reserved_bytes")
    scope = payload.get("cuda_peak_measurement_scope")
    if type(allocated) is not int or allocated < 0:
        raise RuntimeError(f"{context}: invalid CUDA peak allocated bytes")
    if type(reserved) is not int or reserved < allocated:
        raise RuntimeError(f"{context}: invalid CUDA peak reserved bytes")
    if not isinstance(scope, str) or not scope:
        raise RuntimeError(f"{context}: missing CUDA peak measurement scope")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def jsonl_sha256(rows: Iterable[dict]) -> str:
    """Hash rows using the exact canonical JSONL encoding written above."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def resolve_git_revision(revision: str) -> str:
    """Resolve a recorded experiment revision without requiring it to be HEAD."""

    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"experiment revision is not a local Git commit: {revision}"
        ) from exc
    if len(resolved) != 40 or any(character not in "0123456789abcdef" for character in resolved):
        raise SystemExit(f"Git returned an invalid experiment revision: {resolved!r}")
    return resolved


def implementation_revision(args) -> str:
    """Return the executing commit, with compatibility for older call sites."""

    return str(getattr(args, "implementation_revision", args.code_version))


def target_count_for_percentage(tail_count: int, p_percent: int) -> int:
    if p_percent not in PERCENTAGES:
        raise ValueError(f"p_percent must be one of {PERCENTAGES}")
    return (int(tail_count) * int(p_percent)) // 100


def replicate_seed(spec: DatasetSpec, replicate_index: int) -> int:
    if replicate_index not in REPLICATE_INDICES:
        raise ValueError(f"invalid replicate index {replicate_index}")
    return spec.seed_base + replicate_index


def tail_ranking(tail_mask: np.ndarray, seed: int) -> np.ndarray:
    nodes = np.flatnonzero(np.asarray(tail_mask, dtype=bool)).astype(np.int64)
    return np.random.default_rng(int(seed)).permutation(nodes)


def graph_pair_path(
    graph_root: Path,
    dataset: str,
    replicate_index: int,
    p_percent: int,
) -> Path:
    return (
        graph_root
        / dataset
        / f"replicate-{replicate_index:04d}"
        / f"pct-{p_percent:03d}"
    )


def pairing_key(dataset: str, replicate_index: int, p_percent: int) -> str:
    return f"{dataset}__replicate_{replicate_index:04d}__pct_{p_percent:03d}"


def parent_run_id(
    spec: DatasetSpec,
    model: str,
    split: str,
    seed: int,
    preparation: str = "erm",
) -> str:
    if preparation not in PREPARATIONS:
        raise ValueError(f"unknown preparation {preparation!r}")
    return f"{spec.dataset}_{model}_{preparation}_{split}_seed-{seed}"


def prediction_key(run_id: str, graph_key: str) -> str:
    return f"{run_id}__{graph_key}"


def load_registered_inputs(
    spec: DatasetSpec,
    *,
    data_root: Path,
    array_root: Path,
    manifest_root: Path,
):
    bundle = load_dataset(spec.dataset, data_root)
    arrays = np.load(array_root / f"{spec.dataset}.npz", allow_pickle=False)
    manifest = json.loads(
        (manifest_root / f"{spec.dataset}.json").read_text(encoding="utf-8")
    )
    validate_fixed_inputs(spec.dataset, bundle, arrays, manifest)
    if manifest["local_homophily"]["tail_side"] != spec.tail_side:
        raise RuntimeError(f"{spec.dataset}: registered tail side changed")
    if int(np.count_nonzero(arrays["tail_mask"])) != spec.tail_count:
        raise RuntimeError(f"{spec.dataset}: registered tail count changed")
    live_splits = tuple(split.split_id for split in bundle.splits)
    if live_splits != spec.split_ids:
        raise RuntimeError(f"{spec.dataset}: registered split panel changed")
    if tuple(bundle.training_seeds) != spec.training_seeds:
        raise RuntimeError(f"{spec.dataset}: registered training seeds changed")
    if len(spec.checkpoint_identities) != 30:
        raise RuntimeError(f"{spec.dataset}: expected exactly 30 checkpoint identities")
    return bundle, arrays, manifest


def generated_graph_manifest(args, spec: DatasetSpec) -> dict:
    if spec.graph_is_reused:
        raise ValueError(f"{spec.dataset} must reuse its authenticated graph pairs")
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": GRAPH_DESIGN_ID,
        "dataset": spec.dataset,
        "code_version": args.code_version,
        "git_dirty": args.git_dirty,
        "p_percentages": list(spec.percentages),
        "graph_replicates": len(REPLICATE_INDICES),
        "graph_replicate_indices": list(REPLICATE_INDICES),
        "replicate_seed_base": spec.seed_base,
        "replicate_seed_rule": "replicate_seed_base + replicate_index",
        "tail_side": spec.tail_side,
        "tail_count": spec.tail_count,
        "desired_same": spec.desired_same,
        "generator": spec.generator,
        "target_ranking": (
            "one numpy default_rng(replicate_seed) permutation of the frozen "
            "tail-node indices per replicate"
        ),
        "target_sets": "nested prefixes of each replicate target ranking",
        "target_count_rule": "floor(p_percent * tail_count / 100)",
        "graph_start_at_each_p": f"registered original {spec.dataset} graph",
        "actual_generation_attempts": spec.generation_max_attempts,
        "models_sharing_graphs": list(spec.models),
    }


def validate_generation_attempt(pair, spec: DatasetSpec) -> None:
    """Check that the recorded deterministic attempt is within the fixed limit."""

    if pair.attempt < 0 or pair.attempt >= spec.generation_max_attempts:
        raise RuntimeError(
            f"generation attempt {pair.attempt} is outside the fixed limit "
            f"of {spec.generation_max_attempts} attempts"
        )
    if len(pair.retry_failures) != pair.attempt:
        raise RuntimeError("generation retry history does not match the attempt index")


def ensure_exact_manifest(path: Path, expected: dict) -> str:
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(f"experiment settings differ from {path}")
    else:
        atomic_json(path, expected)
    return file_sha256(path)


def _copy_exact_file(source: Path, destination: Path) -> None:
    """Copy one immutable input, or verify an already staged identical copy."""

    if not source.is_file():
        raise RuntimeError(f"missing graph input: {source}")
    source_sha256 = file_sha256(source)
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != source_sha256:
            raise RuntimeError(f"staged graph input differs from source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != source_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copied graph input failed SHA256 verification: {source}")
    temporary.replace(destination)


def stage_reused_graph_dataset(args, spec: DatasetSpec) -> None:
    """Stage only graph inputs from a legacy dose-response result directory."""

    if not spec.graph_is_reused:
        raise ValueError(f"{spec.dataset} graphs are generated by the expansion runner")
    source_root = args.source_dataset_root
    manifest_path = source_root / "experiment_manifest.json"
    _, manifest_sha256 = _validate_graph_manifest(spec, manifest_path)
    _validate_source_baseline(
        spec,
        source_root / "baseline" / "metadata.json",
        manifest_sha256=manifest_sha256,
    )
    expected_pair_dirs = {
        Path(f"replicate-{replicate:04d}") / f"pct-{percentage:03d}"
        for replicate in REPLICATE_INDICES
        for percentage in spec.nonzero_percentages
    }
    observed_pair_dirs = {
        path.relative_to(source_root)
        for path in source_root.glob("replicate-*/*")
        if path.is_dir() and not path.name.endswith(".in-progress")
    }
    if observed_pair_dirs != expected_pair_dirs:
        missing = sorted(map(str, expected_pair_dirs - observed_pair_dirs))
        extra = sorted(map(str, observed_pair_dirs - expected_pair_dirs))
        raise RuntimeError(
            f"{spec.dataset}: source graph directories differ; "
            f"missing={missing}, extra={extra}"
        )

    destination_root = args.graph_root / spec.dataset
    relative_files = [
        Path("experiment_manifest.json"),
        Path("baseline") / "metadata.json",
    ]
    for pair_dir in sorted(expected_pair_dirs):
        relative_files.extend(pair_dir / name for name in GRAPH_FILES)
    for relative_path in relative_files:
        _copy_exact_file(
            source_root / relative_path,
            destination_root / relative_path,
        )
    print(
        f"staged {len(relative_files)} graph-only files for {spec.dataset} "
        f"in {destination_root}",
        flush=True,
    )


def graph_baseline_metadata(
    args,
    spec: DatasetSpec,
    *,
    manifest_sha256: str,
    original_edges: np.ndarray,
    fixed_input_checksums: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": GRAPH_DESIGN_ID,
        "kind": "registered-original-graph-provenance",
        "phase": args.phase,
        "completed": True,
        "dataset": spec.dataset,
        "code_version": args.code_version,
        "git_dirty": args.git_dirty,
        "experiment_manifest_sha256": manifest_sha256,
        "original_edge_sha256": array_sha256(original_edges),
        "fixed_input_checksums": fixed_input_checksums,
        "created_at": now_utc(),
    }


def _validate_existing_generated_pair(
    path: Path,
    *,
    spec: DatasetSpec,
    manifest_sha256: str,
    replicate_index: int,
    p_percent: int,
) -> None:
    if set(item.name for item in path.iterdir() if item.is_file()) != set(GRAPH_FILES):
        raise RuntimeError(f"{path}: generated graph directory has unexpected files")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": GRAPH_DESIGN_ID,
        "phase": "actual",
        "completed": True,
        "dataset": spec.dataset,
        "replicate_index": replicate_index,
        "replicate_seed": replicate_seed(spec, replicate_index),
        "p_percent": p_percent,
        "requested_targets": target_count_for_percentage(spec.tail_count, p_percent),
        "selected_targets": target_count_for_percentage(spec.tail_count, p_percent),
        "experiment_manifest_sha256": manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: completed graph metadata mismatch {mismatch}")
    hashes = metadata.get("output_sha256", {})
    for filename in ("graphs.npz", "swaps.json"):
        if hashes.get(filename) != file_sha256(path / filename):
            raise RuntimeError(f"{path}: altered {filename}")


def generate_graph_dataset(args, spec: DatasetSpec) -> None:
    """Generate one immutable Roman/Amazon graph panel, stopping on failure."""

    if spec.graph_is_reused:
        raise SystemExit(f"{spec.dataset}: existing graph pairs must be reused")
    if args.phase == "actual":
        if tuple(args.replicates) != REPLICATE_INDICES:
            raise SystemExit("actual graph generation requires all five replicates")
        if tuple(args.percentages) != spec.nonzero_percentages:
            raise SystemExit(
                f"actual {spec.dataset} graph generation requires doses "
                f"{spec.nonzero_percentages}"
            )
    elif set(args.percentages) - set(spec.nonzero_percentages):
        raise SystemExit(
            f"{spec.dataset} percentages must come from {spec.nonzero_percentages}"
        )

    bundle, arrays, registered_manifest = load_registered_inputs(
        spec,
        data_root=args.data_root,
        array_root=args.array_root,
        manifest_root=args.manifest_root,
    )
    try:
        dataset_root = args.graph_root / spec.dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        manifest_sha256 = ensure_exact_manifest(
            dataset_root / "experiment_manifest.json",
            generated_graph_manifest(args, spec),
        )
        original_edges = np.asarray(
            to_simple_undirected(bundle.edge_index, bundle.num_nodes),
            dtype=np.int64,
        )
        baseline_path = dataset_root / "baseline" / "metadata.json"
        expected_baseline = graph_baseline_metadata(
            args,
            spec,
            manifest_sha256=manifest_sha256,
            original_edges=original_edges,
            fixed_input_checksums=registered_manifest["array_checksums_sha256"],
        )
        if baseline_path.exists():
            observed = json.loads(baseline_path.read_text(encoding="utf-8"))
            stable_keys = set(expected_baseline) - {"created_at"}
            if any(observed.get(key) != expected_baseline[key] for key in stable_keys):
                raise RuntimeError(f"baseline provenance differs at {baseline_path}")
        else:
            atomic_json(baseline_path, expected_baseline)

        for replicate_index in args.replicates:
            seed = replicate_seed(spec, replicate_index)
            ranking = tail_ranking(arrays["tail_mask"], seed)
            for p_percent in args.percentages:
                requested = target_count_for_percentage(spec.tail_count, p_percent)
                targets = np.asarray(ranking[:requested], dtype=np.int64)
                final_dir = graph_pair_path(
                    args.graph_root, spec.dataset, replicate_index, p_percent
                )
                incomplete_dir = final_dir.with_name(f"{final_dir.name}.in-progress")
                if final_dir.exists():
                    _validate_existing_generated_pair(
                        final_dir,
                        spec=spec,
                        manifest_sha256=manifest_sha256,
                        replicate_index=replicate_index,
                        p_percent=p_percent,
                    )
                    print(f"validated existing {final_dir}; skipping", flush=True)
                    continue
                if incomplete_dir.exists():
                    raise RuntimeError(
                        f"incomplete graph generation requires inspection: {incomplete_dir}"
                    )
                incomplete_dir.mkdir(parents=True)
                started = time.perf_counter()
                try:
                    pair = generate_rewire_pair(
                        bundle.edge_index,
                        bundle.y,
                        arrays["tail_mask"],
                        desired_same=spec.desired_same,
                        pair_seed=seed,
                        target_nodes=targets,
                        max_attempts=spec.generation_max_attempts,
                    )
                    validate_generation_attempt(pair, spec)
                    if not np.array_equal(np.sort(pair.targets), np.sort(targets)):
                        raise RuntimeError(
                            f"{spec.dataset} replicate {replicate_index} p={p_percent}: "
                            "generator did not reach the requested target set"
                        )
                    np.savez_compressed(
                        incomplete_dir / "graphs.npz",
                        treatment_edge_index=pair.treatment_edge_index,
                        control_edge_index=pair.control_edge_index,
                        targets=targets,
                        target_order=pair.target_order,
                        tail_ranking=ranking,
                    )
                    atomic_json(incomplete_dir / "swaps.json", pair.swap_records())
                    metadata = {
                        "schema_version": SCHEMA_VERSION,
                        "design_id": GRAPH_DESIGN_ID,
                        "phase": args.phase,
                        "completed": True,
                        "dataset": spec.dataset,
                        "replicate_index": replicate_index,
                        "replicate_seed": seed,
                        "p_percent": p_percent,
                        "tail_count": spec.tail_count,
                        "tail_side": spec.tail_side,
                        "desired_same": spec.desired_same,
                        "generator": spec.generator,
                        "requested_targets": requested,
                        "selected_targets": int(pair.targets.size),
                        "generation_attempt": pair.attempt,
                        "generation_retry_failures": pair.retry_failures,
                        "generation_seconds": time.perf_counter() - started,
                        "audit": pair.audit,
                        "code_version": args.code_version,
                        "git_dirty": args.git_dirty,
                        "experiment_manifest_sha256": manifest_sha256,
                        "completed_at": now_utc(),
                    }
                    metadata["output_sha256"] = {
                        filename: file_sha256(incomplete_dir / filename)
                        for filename in ("graphs.npz", "swaps.json")
                    }
                    atomic_json(incomplete_dir / "metadata.json", metadata)
                    incomplete_dir.replace(final_dir)
                    print(f"completed {final_dir}", flush=True)
                except Exception as error:
                    atomic_json(
                        incomplete_dir / "failure.json",
                        {
                            "dataset": spec.dataset,
                            "replicate_index": replicate_index,
                            "p_percent": p_percent,
                            "requested_targets": requested,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "failed_at": now_utc(),
                        },
                    )
                    raise
    finally:
        arrays.close()


COMMON_REQUIRED_AUDIT_BOOLEANS = {
    "control_degrees_preserved",
    "control_target_homophily_unchanged",
    "edge_count_preserved",
    "treatment_degrees_preserved",
    "treatment_homophily_step_exact",
}
OGBN_REQUIRED_AUDIT_FIELDS = {
    "control_full_tail_homophily_unchanged",
    "control_swap_count",
    "each_target_processed_once",
    "paired_swap_count",
    "singleton_swap_count",
    "treatment_control_swap_counts_match",
    "treatment_swap_count",
    "treatment_unselected_tail_homophily_unchanged",
}


def _validate_graph_manifest(spec: DatasetSpec, path: Path) -> tuple[dict, str]:
    digest = file_sha256(path)
    if spec.reused_manifest_sha256 is not None and digest != spec.reused_manifest_sha256:
        raise RuntimeError(
            f"{spec.dataset}: source graph manifest SHA256 is {digest}, expected "
            f"{spec.reused_manifest_sha256}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": spec.dataset,
        "p_percentages": list(spec.percentages),
        "graph_replicates": len(REPLICATE_INDICES),
        "graph_replicate_indices": list(REPLICATE_INDICES),
        "replicate_seed_base": spec.seed_base,
        "tail_count": spec.tail_count,
    }
    # The PubMed v1 manifest predates these two explicit fields.  Its frozen
    # whole-file hash above authenticates the omitted design values.
    if "tail_side" in manifest:
        expected["tail_side"] = spec.tail_side
    if "desired_same" in manifest:
        expected["desired_same"] = spec.desired_same
    mismatch = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: graph experiment manifest mismatch {mismatch}")
    if manifest.get("actual_generation_attempts") != spec.generation_max_attempts:
        raise RuntimeError(f"{path}: graph generation attempt limit differs")
    return manifest, digest


def _validate_source_baseline(
    spec: DatasetSpec,
    path: Path,
    *,
    manifest_sha256: str,
) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing source baseline provenance: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "phase": "actual",
        "completed": True,
        "dataset": spec.dataset,
        "experiment_manifest_sha256": manifest_sha256,
    }
    mismatch = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: source baseline provenance mismatch {mismatch}")
    if spec.dataset in {"pubmed", "arxiv-year"}:
        if metadata.get("all_original_predictions_reproduced") is not True:
            raise RuntimeError(f"{path}: source baseline was not reproduced")
    if spec.dataset == "ogbn-arxiv":
        if metadata.get("all_original_tail_test_predictions_reproduced") is not True:
            raise RuntimeError(f"{path}: source tail-test baseline was not reproduced")
    return metadata


def _validate_audit(spec: DatasetSpec, audit: dict, selected_targets: int) -> None:
    missing = COMMON_REQUIRED_AUDIT_BOOLEANS - set(audit)
    if missing:
        raise RuntimeError(f"{spec.dataset}: graph audit lacks {sorted(missing)}")
    if any(audit[key] is not True for key in COMMON_REQUIRED_AUDIT_BOOLEANS):
        raise RuntimeError(f"{spec.dataset}: a required graph audit check failed")
    false_booleans = sorted(
        key for key, value in audit.items() if isinstance(value, bool) and not value
    )
    if false_booleans:
        raise RuntimeError(
            f"{spec.dataset}: graph audit contains false checks {false_booleans}"
        )
    if int(audit.get("num_targets", -1)) != selected_targets:
        raise RuntimeError(f"{spec.dataset}: graph audit target count differs")
    if spec.dataset == "ogbn-arxiv":
        missing = OGBN_REQUIRED_AUDIT_FIELDS - set(audit)
        if missing:
            raise RuntimeError(f"ogbn-arxiv graph audit lacks {sorted(missing)}")
        treatment = int(audit["treatment_swap_count"])
        control = int(audit["control_swap_count"])
        parts = int(audit["paired_swap_count"]) + int(audit["singleton_swap_count"])
        if treatment != control or treatment != parts:
            raise RuntimeError("ogbn-arxiv coordinated swap counts differ")


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def authenticate_graph_dataset(args, spec: DatasetSpec) -> Path:
    """Authenticate graph-only inputs and write a portable content catalog."""

    dataset_root = args.graph_root / spec.dataset
    incomplete = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*")
        if path.name.endswith(".in-progress") or ".incomplete" in path.name
    )
    if incomplete:
        raise RuntimeError(
            f"{spec.dataset}: incomplete graph directories are present: {incomplete}"
        )
    expected_pair_dirs = {
        Path(f"replicate-{replicate_index:04d}") / f"pct-{p_percent:03d}"
        for replicate_index in REPLICATE_INDICES
        for p_percent in spec.nonzero_percentages
    }
    observed_pair_dirs = {
        path.relative_to(dataset_root)
        for path in dataset_root.glob("replicate-*/*")
        if path.is_dir()
    }
    if observed_pair_dirs != expected_pair_dirs:
        missing = sorted(map(str, expected_pair_dirs - observed_pair_dirs))
        extra = sorted(map(str, observed_pair_dirs - expected_pair_dirs))
        raise RuntimeError(
            f"{spec.dataset}: graph directories differ; missing={missing}, "
            f"extra={extra}"
        )
    manifest, manifest_sha256 = _validate_graph_manifest(
        spec, dataset_root / "experiment_manifest.json"
    )
    baseline_path = dataset_root / "baseline" / "metadata.json"
    _validate_source_baseline(
        spec, baseline_path, manifest_sha256=manifest_sha256
    )

    bundle, arrays, registered_manifest = load_registered_inputs(
        spec,
        data_root=args.data_root,
        array_root=args.array_root,
        manifest_root=args.manifest_root,
    )
    try:
        original_edges = np.asarray(
            to_simple_undirected(bundle.edge_index, bundle.num_nodes),
            dtype=np.int64,
        )
        original_degree = np.bincount(
            original_edges[0], minlength=bundle.num_nodes
        )
        original_edge_sha256 = array_sha256(original_edges)
        tail_nodes = np.flatnonzero(arrays["tail_mask"]).astype(np.int64)
        entries: list[dict] = []
        for replicate_index in REPLICATE_INDICES:
            expected_ranking = tail_ranking(
                arrays["tail_mask"], replicate_seed(spec, replicate_index)
            )
            for p_percent in spec.nonzero_percentages:
                pair_path = graph_pair_path(
                    args.graph_root, spec.dataset, replicate_index, p_percent
                )
                actual_files = set(
                    item.name for item in pair_path.iterdir() if item.is_file()
                )
                if actual_files != set(GRAPH_FILES):
                    raise RuntimeError(
                        f"{pair_path}: expected graph-only files {GRAPH_FILES}; "
                        f"got {sorted(actual_files)}"
                    )
                metadata_path = pair_path / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                requested = target_count_for_percentage(spec.tail_count, p_percent)
                expected_metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "design_id": manifest["design_id"],
                    "phase": "actual",
                    "completed": True,
                    "dataset": spec.dataset,
                    "replicate_index": replicate_index,
                    "replicate_seed": replicate_seed(spec, replicate_index),
                    "p_percent": p_percent,
                    "requested_targets": requested,
                    "selected_targets": requested,
                    "experiment_manifest_sha256": manifest_sha256,
                }
                mismatch = {
                    key: (value, metadata.get(key))
                    for key, value in expected_metadata.items()
                    if metadata.get(key) != value
                }
                if mismatch:
                    raise RuntimeError(f"{pair_path}: graph metadata mismatch {mismatch}")
                generation_attempt = int(metadata.get("generation_attempt", -1))
                retry_failures = metadata.get("generation_retry_failures")
                if not 0 <= generation_attempt < spec.generation_max_attempts:
                    raise RuntimeError(
                        f"{pair_path}: generation attempt is outside the fixed limit"
                    )
                if not isinstance(retry_failures, list) or len(retry_failures) != generation_attempt:
                    raise RuntimeError(
                        f"{pair_path}: generation retry history differs from attempt index"
                    )
                graphs_path = pair_path / "graphs.npz"
                swaps_path = pair_path / "swaps.json"
                graphs_sha256 = file_sha256(graphs_path)
                swaps_sha256 = file_sha256(swaps_path)
                output_hashes = metadata.get("output_sha256", {})
                if output_hashes.get("graphs.npz") != graphs_sha256:
                    raise RuntimeError(f"{graphs_path}: SHA256 differs from metadata")
                if output_hashes.get("swaps.json") != swaps_sha256:
                    raise RuntimeError(f"{swaps_path}: SHA256 differs from metadata")
                _validate_audit(spec, metadata.get("audit", {}), requested)

                with np.load(graphs_path, allow_pickle=False) as graph_arrays:
                    if set(graph_arrays.files) != GRAPH_ARRAY_KEYS:
                        raise RuntimeError(
                            f"{graphs_path}: graph array keys differ from contract"
                        )
                    treatment = graph_arrays["treatment_edge_index"].astype(
                        np.int64, copy=False
                    )
                    control = graph_arrays["control_edge_index"].astype(
                        np.int64, copy=False
                    )
                    targets = graph_arrays["targets"].astype(np.int64, copy=False)
                    target_order = graph_arrays["target_order"].astype(
                        np.int64, copy=False
                    )
                    ranking = graph_arrays["tail_ranking"].astype(
                        np.int64, copy=False
                    )
                    if treatment.shape != original_edges.shape:
                        raise RuntimeError(f"{graphs_path}: treatment edge shape changed")
                    if control.shape != original_edges.shape:
                        raise RuntimeError(f"{graphs_path}: control edge shape changed")
                    if not np.array_equal(
                        np.bincount(treatment[0], minlength=bundle.num_nodes),
                        original_degree,
                    ):
                        raise RuntimeError(f"{graphs_path}: treatment degrees changed")
                    if not np.array_equal(
                        np.bincount(control[0], minlength=bundle.num_nodes),
                        original_degree,
                    ):
                        raise RuntimeError(f"{graphs_path}: control degrees changed")
                    if ranking.shape != tail_nodes.shape or not np.array_equal(
                        ranking, expected_ranking
                    ):
                        raise RuntimeError(f"{graphs_path}: frozen tail ranking differs")
                    if not np.array_equal(targets, ranking[:requested]):
                        raise RuntimeError(f"{graphs_path}: targets are not ranking prefix")
                    if target_order.size != requested or not np.array_equal(
                        np.sort(target_order), np.sort(targets)
                    ):
                        raise RuntimeError(f"{graphs_path}: target order differs")
                    audit = metadata["audit"]
                    if audit.get("original_edge_sha256") != original_edge_sha256:
                        raise RuntimeError(f"{graphs_path}: original graph hash differs")
                    if audit.get("treatment_edge_sha256") != array_sha256(treatment):
                        raise RuntimeError(f"{graphs_path}: treatment graph hash differs")
                    if audit.get("control_edge_sha256") != array_sha256(control):
                        raise RuntimeError(f"{graphs_path}: control graph hash differs")

                relative_dir = pair_path.relative_to(dataset_root).as_posix()
                entries.append(
                    {
                        "pairing_key": pairing_key(
                            spec.dataset, replicate_index, p_percent
                        ),
                        "replicate_index": replicate_index,
                        "replicate_seed": replicate_seed(spec, replicate_index),
                        "p_percent": p_percent,
                        "selected_targets": requested,
                        "relative_dir": relative_dir,
                        "graphs": _file_record(graphs_path),
                        "swaps": _file_record(swaps_path),
                        "metadata": _file_record(metadata_path),
                        "treatment_edge_sha256": metadata["audit"][
                            "treatment_edge_sha256"
                        ],
                        "control_edge_sha256": metadata["audit"][
                            "control_edge_sha256"
                        ],
                    }
                )
        if len(entries) != spec.pairs_per_checkpoint:
            raise RuntimeError(
                f"{spec.dataset}: expected {spec.pairs_per_checkpoint} graph pairs"
            )
        catalog = {
            "schema_version": GRAPH_CATALOG_SCHEMA_VERSION,
            "design_id": DESIGN_ID,
            "kind": "authenticated-graph-catalog",
            "dataset": spec.dataset,
            "source_graph_design_id": manifest["design_id"],
            "source_experiment_manifest_sha256": manifest_sha256,
            "source_experiment_manifest": _file_record(
                dataset_root / "experiment_manifest.json"
            ),
            "source_baseline_metadata": _file_record(baseline_path),
            "registered_fixed_input_checksums": registered_manifest[
                "array_checksums_sha256"
            ],
            "original_edge_sha256": original_edge_sha256,
            "pair_count": len(entries),
            "pairs": entries,
        }
        output = args.catalog_root / f"{spec.dataset}.json"
        atomic_json(output, catalog)
        print(f"authenticated {len(entries)} graph pairs in {output}", flush=True)
        return output
    finally:
        arrays.close()


def _validate_bound_file(path: Path, record: dict) -> None:
    stat = path.stat()
    if stat.st_size != int(record["bytes"]):
        raise RuntimeError(f"{path}: size differs from graph catalog")
    if file_sha256(path) != record.get("sha256"):
        raise RuntimeError(f"{path}: SHA256 differs from graph catalog")


def load_graph_catalog(
    spec: DatasetSpec,
    *,
    graph_root: Path,
    catalog_root: Path,
) -> tuple[dict, str]:
    catalog_path = catalog_root / f"{spec.dataset}.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") not in {SCHEMA_VERSION, GRAPH_CATALOG_SCHEMA_VERSION}:
        raise RuntimeError(f"{catalog_path}: unsupported schema")
    if catalog.get("design_id") != DESIGN_ID:
        raise RuntimeError(f"{catalog_path}: wrong expansion design")
    if catalog.get("dataset") != spec.dataset:
        raise RuntimeError(f"{catalog_path}: wrong dataset")
    if int(catalog.get("pair_count", -1)) != spec.pairs_per_checkpoint:
        raise RuntimeError(
            f"{catalog_path}: expected {spec.pairs_per_checkpoint} graph pairs"
        )
    manifest_sha = catalog.get("source_experiment_manifest_sha256")
    if spec.reused_manifest_sha256 and manifest_sha != spec.reused_manifest_sha256:
        raise RuntimeError(f"{catalog_path}: reused manifest hash differs")
    dataset_root = graph_root / spec.dataset
    _validate_bound_file(
        dataset_root / "experiment_manifest.json",
        catalog["source_experiment_manifest"],
    )
    _validate_bound_file(
        dataset_root / "baseline" / "metadata.json",
        catalog["source_baseline_metadata"],
    )
    expected_cells = {
        (replicate, percentage)
        for replicate in REPLICATE_INDICES
        for percentage in spec.nonzero_percentages
    }
    observed_cells: set[tuple[int, int]] = set()
    for entry in catalog["pairs"]:
        cell = (int(entry["replicate_index"]), int(entry["p_percent"]))
        if cell in observed_cells:
            raise RuntimeError(f"{catalog_path}: duplicate graph cell {cell}")
        observed_cells.add(cell)
        pair_path = dataset_root / entry["relative_dir"]
        _validate_bound_file(pair_path / "graphs.npz", entry["graphs"])
        _validate_bound_file(pair_path / "swaps.json", entry["swaps"])
        _validate_bound_file(pair_path / "metadata.json", entry["metadata"])
    if observed_cells != expected_cells:
        raise RuntimeError(f"{catalog_path}: graph cells differ from fixed design")
    return catalog, file_sha256(catalog_path)


def load_checkpoint_spec(
    spec: DatasetSpec,
    *,
    model: str,
    split: str,
    seed: int,
    checkpoint_root: Path,
    fixed_input_checksums: dict,
    num_nodes: int,
    preparation: str = "erm",
) -> CheckpointSpec:
    run_id = parent_run_id(spec, model, split, seed, preparation)
    run_dir = checkpoint_root / run_id
    config, checkpoint_path, saved_predictions = validate_parent_run(
        run_dir,
        dataset=spec.dataset,
        model=model,
        split=split,
        seed=seed,
        fixed_input_checksums=fixed_input_checksums,
        preparation=preparation,
    )
    if saved_predictions.shape != (num_nodes,):
        raise RuntimeError(
            f"{run_id}: saved predictions have shape {saved_predictions.shape}; "
            f"expected {(num_nodes,)}"
        )
    return CheckpointSpec(
        dataset=spec.dataset,
        model=model,
        split=split,
        seed=seed,
        run_id=run_id,
        model_config=dict(config["model_config"]),
        checkpoint_path=checkpoint_path,
        run_data_path=run_dir / "run_data.npz",
        checkpoint_sha256=file_sha256(checkpoint_path),
        training_code_version=str(config["code_version"]),
        saved_predictions=saved_predictions,
        preparation=preparation,
    )


def load_network(
    checkpoint: CheckpointSpec,
    *,
    input_dim: int,
    num_classes: int,
    num_nodes: int,
    device: torch.device,
):
    network = build_model(
        checkpoint.model,
        input_dim,
        num_classes,
        checkpoint.model_config,
        num_nodes=num_nodes,
    )
    state = torch.load(
        checkpoint.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    network.load_state_dict(state)
    network.to(device)
    network.eval()
    return network


def _forward_logits(
    network,
    x: torch.Tensor,
    edge_tensor: torch.Tensor,
    *,
    prepared_h2_operators=None,
    pause_seconds: float = 0.0,
) -> np.ndarray:
    clear_cache = getattr(network.encoder, "clear_graph_cache", None)
    if clear_cache is not None:
        clear_cache()
    if prepared_h2_operators is not None:
        set_cache = getattr(network.encoder, "set_graph_cache", None)
        if set_cache is None:
            raise RuntimeError(
                f"{type(network.encoder).__name__} lacks set_graph_cache()"
            )
        set_cache(prepared_h2_operators, num_edges=int(edge_tensor.shape[1]))
    try:
        with torch.inference_mode():
            logits = network(x, edge_tensor).detach().cpu().numpy()
    finally:
        if clear_cache is not None:
            clear_cache()
    if pause_seconds:
        time.sleep(pause_seconds)
    logits = np.asarray(logits)
    if not np.isfinite(logits).all():
        raise RuntimeError("model forward produced non-finite logits")
    return logits


def h2_predictions_for_graph(
    networks: dict[str, object],
    x: torch.Tensor,
    edge_index: np.ndarray,
    *,
    device: torch.device,
    pause_seconds: float,
) -> dict[str, np.ndarray]:
    """Prepare H2GCN operators once and reuse them across checkpoints."""

    from hfd.models.h2gcn import prepare_h2gcn_operators

    edge_tensor = torch.from_numpy(np.asarray(edge_index)).to(
        device=device, dtype=torch.long
    )
    operators = prepare_h2gcn_operators(
        edge_tensor,
        int(x.shape[0]),
        device=device,
        dtype=x.dtype,
    )
    predictions: dict[str, np.ndarray] = {}
    try:
        for run_id, network in networks.items():
            logits = _forward_logits(
                network,
                x,
                edge_tensor,
                prepared_h2_operators=operators,
                pause_seconds=pause_seconds,
            )
            predictions[run_id] = logits
    finally:
        del operators, edge_tensor
    return predictions


def tail_test_indices(arrays, split: str) -> np.ndarray:
    test_idx = arrays[f"{split}.test"].astype(np.int64, copy=False)
    tail_mask = arrays["tail_mask"].astype(bool, copy=False)
    indices = test_idx[tail_mask[test_idx]]
    if indices.size == 0:
        raise RuntimeError(f"{split}: fixed tail-test subset is empty")
    return indices


def subset_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    true = labels[indices]
    predicted = predictions[indices]
    return (
        macro_f1(true, predicted, np.unique(true)),
        accuracy(true, predicted),
    )


def class_composition(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(values, counts)
    }


def baseline_row_from_logits(
    checkpoint: CheckpointSpec,
    *,
    bundle,
    arrays,
    current_logits: np.ndarray,
    logit_atol: float = LOGIT_REPRODUCTION_ATOL,
    logit_rtol: float = LOGIT_REPRODUCTION_RTOL,
) -> dict:
    if current_logits.shape[0] != bundle.num_nodes:
        raise RuntimeError(f"{checkpoint.run_id}: baseline logits have wrong shape")
    current_predictions = current_logits.argmax(axis=-1).astype(np.int64, copy=False)
    saved_predictions = checkpoint.saved_predictions
    indices = tail_test_indices(arrays, checkpoint.split)
    difference_mask = current_predictions != saved_predictions
    difference_nodes = np.flatnonzero(difference_mask).astype(np.int64)
    full_differences = int(difference_nodes.size)
    tail_differences = int(np.count_nonzero(difference_mask[indices]))
    if tail_differences:
        raise RuntimeError(
            f"{checkpoint.run_id}: fixed tail-test predictions fail reproduction "
            f"({tail_differences} nodes)"
        )
    logit_reproduction_checked = False
    max_abs_logit_difference = 0.0
    mean_abs_logit_difference = 0.0
    if full_differences:
        with np.load(checkpoint.run_data_path, allow_pickle=False) as saved:
            if "logits" not in saved.files:
                raise RuntimeError(f"{checkpoint.run_data_path}: saved logits missing")
            saved_logits = saved["logits"].astype(current_logits.dtype, copy=True)
        if saved_logits.shape != current_logits.shape:
            raise RuntimeError(f"{checkpoint.run_id}: saved logits have wrong shape")
        if not np.isfinite(saved_logits).all() or not np.isfinite(current_logits).all():
            raise RuntimeError(f"{checkpoint.run_id}: non-finite baseline logits")
        if not np.array_equal(saved_logits.argmax(axis=-1), saved_predictions):
            raise RuntimeError(f"{checkpoint.run_id}: saved logits/predictions differ")
        difference = np.abs(current_logits - saved_logits)
        max_abs_logit_difference = float(difference.max())
        mean_abs_logit_difference = float(difference.mean())
        if not np.allclose(
            current_logits,
            saved_logits,
            atol=logit_atol,
            rtol=logit_rtol,
        ):
            raise RuntimeError(
                f"{checkpoint.run_id}: baseline logits do not reproduce parent "
                f"(max absolute difference {max_abs_logit_difference:.9g}; "
                f"atol={logit_atol:.9g}, rtol={logit_rtol:.9g})"
            )
        logit_reproduction_checked = True
    score, acc = subset_metrics(bundle.y, current_predictions, indices)
    row = {
        "dataset": checkpoint.dataset,
        "model": checkpoint.model,
        "run_id": checkpoint.run_id,
        "split": checkpoint.split,
        "seed": checkpoint.seed,
        "n_tail_test": int(indices.size),
        "tail_test_class_composition": class_composition(bundle.y, indices),
        "tail_test_macro_f1": score,
        "tail_test_accuracy": acc,
        "n_full_graph_prediction_differences_from_parent": full_differences,
        "n_tail_test_prediction_differences_from_parent": tail_differences,
        "full_graph_prediction_difference_nodes_from_parent": difference_nodes.tolist(),
        "full_graph_logit_reproduction_checked": logit_reproduction_checked,
        "full_graph_logit_reproduction_atol": float(logit_atol),
        "full_graph_logit_reproduction_rtol": float(logit_rtol),
        "max_abs_logit_difference_from_parent": max_abs_logit_difference,
        "mean_abs_logit_difference_from_parent": mean_abs_logit_difference,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_training_code_version": checkpoint.training_code_version,
        "prediction_key": prediction_key(checkpoint.run_id, "original"),
        "prediction_sha256": array_sha256(current_predictions),
    }
    if checkpoint.preparation != "erm":
        row["preparation"] = checkpoint.preparation
    return row


def paired_metric_row(
    checkpoint: CheckpointSpec,
    *,
    bundle,
    arrays,
    graph_entry: dict,
    treatment_predictions: np.ndarray,
    control_predictions: np.ndarray,
) -> dict:
    indices = tail_test_indices(arrays, checkpoint.split)
    treatment_f1, treatment_acc = subset_metrics(
        bundle.y, treatment_predictions, indices
    )
    control_f1, control_acc = subset_metrics(bundle.y, control_predictions, indices)
    pair_key = graph_entry["pairing_key"]
    treatment_key = prediction_key(checkpoint.run_id, f"{pair_key}__treatment")
    control_key = prediction_key(checkpoint.run_id, f"{pair_key}__control")
    row = {
        "dataset": checkpoint.dataset,
        "model": checkpoint.model,
        "run_id": checkpoint.run_id,
        "split": checkpoint.split,
        "seed": checkpoint.seed,
        "pairing_key": pair_key,
        "replicate_index": int(graph_entry["replicate_index"]),
        "replicate_seed": int(graph_entry["replicate_seed"]),
        "p_percent": int(graph_entry["p_percent"]),
        "n_selected_tail_nodes": int(graph_entry["selected_targets"]),
        "n_tail_test": int(indices.size),
        "tail_test_class_composition": class_composition(bundle.y, indices),
        "treatment_macro_f1": treatment_f1,
        "control_macro_f1": control_f1,
        "treatment_minus_control_macro_f1": treatment_f1 - control_f1,
        "treatment_accuracy": treatment_acc,
        "control_accuracy": control_acc,
        "treatment_minus_control_accuracy": treatment_acc - control_acc,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_training_code_version": checkpoint.training_code_version,
        "source_graphs_sha256": graph_entry["graphs"]["sha256"],
        "treatment_edge_sha256": graph_entry["treatment_edge_sha256"],
        "control_edge_sha256": graph_entry["control_edge_sha256"],
        "treatment_prediction_key": treatment_key,
        "control_prediction_key": control_key,
        "treatment_prediction_sha256": array_sha256(treatment_predictions),
        "control_prediction_sha256": array_sha256(control_predictions),
    }
    if checkpoint.preparation != "erm":
        row["preparation"] = checkpoint.preparation
    return row


def load_graph_edges(graph_root: Path, spec: DatasetSpec, entry: dict):
    path = graph_root / spec.dataset / entry["relative_dir"] / "graphs.npz"
    with np.load(path, allow_pickle=False) as arrays:
        treatment = arrays["treatment_edge_index"].astype(np.int64, copy=True)
        control = arrays["control_edge_index"].astype(np.int64, copy=True)
    return treatment, control


def shard_path(output_root: Path, checkpoint: CheckpointSpec) -> Path:
    return (
        output_root
        / checkpoint.dataset
        / checkpoint.model
        / "shards"
        / f"{checkpoint.run_id}.json"
    )


def expected_pair_cells(spec: DatasetSpec) -> set[tuple[int, int]]:
    return {
        (replicate, percentage)
        for replicate in REPLICATE_INDICES
        for percentage in spec.nonzero_percentages
    }


def validate_shard_payload(
    payload: dict,
    *,
    spec: DatasetSpec,
    model: str,
    run_id: str,
    graph_catalog_sha256: str,
    expected_phase: str | None = None,
    expected_code_version: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_preparation: str = "erm",
) -> None:
    design_id = design_id_for_preparation(expected_preparation)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": design_id,
        "completed": True,
        "dataset": spec.dataset,
        "model": model,
        "run_id": run_id,
        "graph_catalog_sha256": graph_catalog_sha256,
    }
    if expected_preparation != "erm":
        expected["preparation"] = expected_preparation
    mismatch = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{run_id}: shard metadata mismatch {mismatch}")
    if payload.get("preparation", "erm") != expected_preparation:
        raise RuntimeError(f"{run_id}: shard preparation differs")
    phase = payload.get("phase", "actual")
    if phase not in {"smoke", "actual"}:
        raise RuntimeError(f"{run_id}: invalid shard phase {phase!r}")
    if expected_phase is not None and phase != expected_phase:
        raise RuntimeError(
            f"{run_id}: shard phase is {phase!r}, expected {expected_phase!r}"
        )
    if (
        expected_code_version is not None
        and payload.get("code_version") != expected_code_version
    ):
        raise RuntimeError(f"{run_id}: shard code version is stale")
    if (
        expected_checkpoint_sha256 is not None
        and payload.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise RuntimeError(f"{run_id}: shard checkpoint SHA256 is stale")
    payload_checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(payload_checkpoint_sha256, str) or len(
        payload_checkpoint_sha256
    ) != 64:
        raise RuntimeError(f"{run_id}: shard checkpoint SHA256 is invalid")
    validate_cuda_peak_memory(payload, context=f"{run_id} shard")
    baseline = payload.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("run_id") != run_id:
        raise RuntimeError(f"{run_id}: shard lacks one baseline row")
    if baseline.get("preparation", "erm") != expected_preparation:
        raise RuntimeError(f"{run_id}: baseline preparation differs")
    if baseline.get("prediction_key") != prediction_key(run_id, "original"):
        raise RuntimeError(f"{run_id}: baseline prediction key differs")
    if baseline.get("checkpoint_sha256") != payload_checkpoint_sha256:
        raise RuntimeError(f"{run_id}: baseline row checkpoint SHA256 differs")
    if int(baseline.get("n_tail_test_prediction_differences_from_parent", -1)) != 0:
        raise RuntimeError(f"{run_id}: baseline reproduction failed")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{run_id}: shard has no paired rows")
    if phase == "actual" and len(rows) != spec.pairs_per_checkpoint:
        raise RuntimeError(
            f"{run_id}: actual shard must contain {spec.pairs_per_checkpoint} paired rows"
        )
    cells: set[tuple[int, int]] = set()
    treatment_keys: set[str] = set()
    control_keys: set[str] = set()
    for row in rows:
        if row.get("run_id") != run_id:
            raise RuntimeError(f"{run_id}: paired row belongs to another checkpoint")
        if row.get("preparation", "erm") != expected_preparation:
            raise RuntimeError(f"{run_id}: paired row preparation differs")
        if row.get("checkpoint_sha256") != payload_checkpoint_sha256:
            raise RuntimeError(f"{run_id}: paired row checkpoint SHA256 differs")
        cell = (int(row["replicate_index"]), int(row["p_percent"]))
        if cell in cells:
            raise RuntimeError(f"{run_id}: duplicate paired cell {cell}")
        cells.add(cell)
        expected_pairing_key = pairing_key(spec.dataset, *cell)
        if row.get("pairing_key") != expected_pairing_key:
            raise RuntimeError(f"{run_id}: paired row key differs")
        treatment_key = str(row.get("treatment_prediction_key", ""))
        control_key = str(row.get("control_prediction_key", ""))
        expected_treatment_key = prediction_key(
            run_id, f"{expected_pairing_key}__treatment"
        )
        expected_control_key = prediction_key(
            run_id, f"{expected_pairing_key}__control"
        )
        if treatment_key != expected_treatment_key:
            raise RuntimeError(f"{run_id}: invalid treatment prediction key")
        if control_key != expected_control_key:
            raise RuntimeError(f"{run_id}: invalid control prediction key")
        if treatment_key == control_key:
            raise RuntimeError(f"{run_id}: treatment/control prediction keys collide")
        treatment_keys.add(treatment_key)
        control_keys.add(control_key)
    if phase == "actual" and cells != expected_pair_cells(spec):
        raise RuntimeError(f"{run_id}: paired cells differ from fixed design")
    if len(treatment_keys) != len(rows):
        raise RuntimeError(f"{run_id}: treatment prediction keys are not unique")
    if len(control_keys) != len(rows):
        raise RuntimeError(f"{run_id}: control prediction keys are not unique")


def write_checkpoint_shard(
    args,
    checkpoint: CheckpointSpec,
    *,
    graph_catalog_sha256: str,
    source_graph_manifest_sha256: str,
    baseline: dict,
    pairs: list[dict],
    schedule: str,
    evaluation_seconds: float,
    cuda_memory: dict,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design_id": evaluation_design_id(args),
        "completed": True,
        "phase": args.phase,
        "dataset": checkpoint.dataset,
        "model": checkpoint.model,
        "run_id": checkpoint.run_id,
        "split": checkpoint.split,
        "seed": checkpoint.seed,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_training_code_version": checkpoint.training_code_version,
        "code_version": args.code_version,
        "implementation_revision": implementation_revision(args),
        "git_dirty": args.git_dirty,
        "schedule": schedule,
        "checkpoint_loaded_once_for_all_graphs": schedule == "checkpoint-major",
        "graph_catalog_sha256": graph_catalog_sha256,
        "source_graph_manifest_sha256": source_graph_manifest_sha256,
        "baseline": baseline,
        "pairs": pairs,
        "n_baseline_rows": 1,
        "n_paired_rows": len(pairs),
        "evaluation_seconds": evaluation_seconds,
        "completed_at": now_utc(),
    }
    preparation = preparation_from_args(args)
    if preparation != "erm":
        payload["preparation"] = preparation
    payload.update(cuda_memory)
    validate_shard_payload(
        payload,
        spec=DATASET_SPECS[checkpoint.dataset],
        model=checkpoint.model,
        run_id=checkpoint.run_id,
        graph_catalog_sha256=graph_catalog_sha256,
        expected_phase=args.phase,
        expected_code_version=args.code_version,
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        expected_preparation=preparation,
    )
    path = shard_path(args.output_root, checkpoint)
    atomic_json(path, payload)
    return path


def _validate_existing_shard(
    path: Path,
    *,
    spec: DatasetSpec,
    model: str,
    run_id: str,
    graph_catalog_sha256: str,
    phase: str,
    code_version: str,
    checkpoint_sha256: str,
    preparation: str = "erm",
    logit_atol: float = LOGIT_REPRODUCTION_ATOL,
    logit_rtol: float = LOGIT_REPRODUCTION_RTOL,
) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_shard_payload(
        payload,
        spec=spec,
        model=model,
        run_id=run_id,
        graph_catalog_sha256=graph_catalog_sha256,
        expected_phase=phase,
        expected_code_version=code_version,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_preparation=preparation,
    )
    baseline = payload["baseline"]
    observed_atol = float(
        baseline.get(
            "full_graph_logit_reproduction_atol",
            LOGIT_REPRODUCTION_ATOL,
        )
    )
    observed_rtol = float(
        baseline.get(
            "full_graph_logit_reproduction_rtol",
            LOGIT_REPRODUCTION_RTOL,
        )
    )
    if (observed_atol, observed_rtol) != (logit_atol, logit_rtol):
        raise RuntimeError(
            f"{run_id}: existing shard logit-reproduction tolerance differs "
            f"(observed atol={observed_atol:.9g}, rtol={observed_rtol:.9g}; "
            f"requested atol={logit_atol:.9g}, rtol={logit_rtol:.9g})"
        )
    print(f"validated existing shard {path}; skipping", flush=True)
    return True


def evaluate_checkpoint_major(
    args,
    *,
    spec: DatasetSpec,
    model: str,
    bundle,
    arrays,
    registered_manifest: dict,
    catalog: dict,
    catalog_sha256: str,
    identities: Sequence[tuple[str, int]],
    x: torch.Tensor,
    original_edges: np.ndarray,
    device: torch.device,
) -> None:
    """Load each non-H2 checkpoint once and evaluate every graph before release."""

    for split, seed in identities:
        checkpoint = load_checkpoint_spec(
            spec,
            model=model,
            split=split,
            seed=seed,
            checkpoint_root=args.checkpoint_root,
            fixed_input_checksums=registered_manifest["array_checksums_sha256"],
            num_nodes=bundle.num_nodes,
            preparation=preparation_from_args(args),
        )
        candidate = shard_path(args.output_root, checkpoint)
        if _validate_existing_shard(
            candidate,
            spec=spec,
            model=model,
            run_id=checkpoint.run_id,
            graph_catalog_sha256=catalog_sha256,
            phase=args.phase,
            code_version=args.code_version,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            preparation=preparation_from_args(args),
            logit_atol=args.logit_reproduction_atol,
            logit_rtol=args.logit_reproduction_rtol,
        ):
            continue
        reset_cuda_peak_memory(device)
        started = time.perf_counter()
        network = load_network(
            checkpoint,
            input_dim=int(bundle.x.shape[1]),
            num_classes=bundle.num_classes,
            num_nodes=bundle.num_nodes,
            device=device,
        )
        try:
            original_tensor = torch.from_numpy(original_edges).to(
                device=device, dtype=torch.long
            )
            try:
                baseline_logits = _forward_logits(
                    network,
                    x,
                    original_tensor,
                    pause_seconds=args.pause_seconds,
                )
            finally:
                del original_tensor
            baseline = baseline_row_from_logits(
                checkpoint,
                bundle=bundle,
                arrays=arrays,
                current_logits=baseline_logits,
                logit_atol=args.logit_reproduction_atol,
                logit_rtol=args.logit_reproduction_rtol,
            )
            del baseline_logits
            pair_rows: list[dict] = []
            for graph_entry in catalog["pairs"]:
                treatment_edges, control_edges = load_graph_edges(
                    args.graph_root, spec, graph_entry
                )
                treatment_tensor = torch.from_numpy(treatment_edges).to(
                    device=device, dtype=torch.long
                )
                try:
                    treatment_logits = _forward_logits(
                        network,
                        x,
                        treatment_tensor,
                        pause_seconds=args.pause_seconds,
                    )
                finally:
                    del treatment_tensor, treatment_edges
                treatment_predictions = treatment_logits.argmax(axis=-1).astype(
                    np.int64, copy=False
                )
                del treatment_logits
                control_tensor = torch.from_numpy(control_edges).to(
                    device=device, dtype=torch.long
                )
                try:
                    control_logits = _forward_logits(
                        network,
                        x,
                        control_tensor,
                        pause_seconds=args.pause_seconds,
                    )
                finally:
                    del control_tensor, control_edges
                control_predictions = control_logits.argmax(axis=-1).astype(
                    np.int64, copy=False
                )
                del control_logits
                pair_rows.append(
                    paired_metric_row(
                        checkpoint,
                        bundle=bundle,
                        arrays=arrays,
                        graph_entry=graph_entry,
                        treatment_predictions=treatment_predictions,
                        control_predictions=control_predictions,
                    )
                )
                del treatment_predictions, control_predictions
            path = write_checkpoint_shard(
                args,
                checkpoint,
                graph_catalog_sha256=catalog_sha256,
                source_graph_manifest_sha256=catalog[
                    "source_experiment_manifest_sha256"
                ],
                baseline=baseline,
                pairs=pair_rows,
                schedule="checkpoint-major",
                evaluation_seconds=time.perf_counter() - started,
                cuda_memory=cuda_peak_memory(
                    device, scope="checkpoint-evaluation"
                ),
            )
            print(f"completed {path}", flush=True)
        finally:
            del network
            if device.type == "cuda":
                torch.cuda.empty_cache()


def h2_progress_path(
    output_root: Path,
    spec: DatasetSpec,
    graph_entry: dict | None,
) -> Path:
    root = output_root / spec.dataset / "h2gcn" / "progress"
    if graph_entry is None:
        return root / "baseline.json"
    return root / f"{graph_entry['pairing_key']}.json"


def _validate_h2_progress(
    payload: dict,
    *,
    args,
    spec: DatasetSpec,
    catalog_sha256: str,
    checkpoints: dict[str, CheckpointSpec],
    graph_entry: dict | None,
) -> H2Progress:
    kind = "baseline" if graph_entry is None else "paired-graph"
    expected = {
        "schema_version": SCHEMA_VERSION,
        "design_id": evaluation_design_id(args),
        "completed": True,
        "phase": args.phase,
        "kind": kind,
        "dataset": spec.dataset,
        "model": "h2gcn",
        "code_version": args.code_version,
        "graph_catalog_sha256": catalog_sha256,
    }
    preparation = preparation_from_args(args)
    if preparation != "erm":
        expected["preparation"] = preparation
    if graph_entry is not None:
        expected.update(
            {
                "pairing_key": graph_entry["pairing_key"],
                "replicate_index": int(graph_entry["replicate_index"]),
                "p_percent": int(graph_entry["p_percent"]),
                "source_graphs_sha256": graph_entry["graphs"]["sha256"],
            }
        )
    mismatch = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"H2GCN progress metadata mismatch {mismatch}")
    if payload.get("preparation", "erm") != preparation:
        raise RuntimeError("H2GCN progress preparation differs")
    observed_tolerance = (
        float(
            payload.get(
                "logit_reproduction_atol",
                LOGIT_REPRODUCTION_ATOL,
            )
        ),
        float(
            payload.get(
                "logit_reproduction_rtol",
                LOGIT_REPRODUCTION_RTOL,
            )
        ),
    )
    requested_tolerance = (
        getattr(args, "logit_reproduction_atol", LOGIT_REPRODUCTION_ATOL),
        getattr(args, "logit_reproduction_rtol", LOGIT_REPRODUCTION_RTOL),
    )
    if observed_tolerance != requested_tolerance:
        raise RuntimeError(
            "H2GCN progress logit-reproduction tolerance differs "
            f"(observed atol={observed_tolerance[0]:.9g}, "
            f"rtol={observed_tolerance[1]:.9g}; "
            f"requested atol={requested_tolerance[0]:.9g}, "
            f"rtol={requested_tolerance[1]:.9g})"
        )
    validate_cuda_peak_memory(payload, context=f"{spec.dataset} H2GCN progress")
    bindings = payload.get("checkpoint_sha256", {})
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("H2GCN progress rows are missing")
    rows_by_run: dict[str, dict] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in rows_by_run:
            raise RuntimeError("H2GCN progress has a duplicate or empty run ID")
        rows_by_run[run_id] = row
    for run_id, checkpoint in checkpoints.items():
        if checkpoint.preparation != preparation:
            raise RuntimeError(f"{run_id}: checkpoint preparation differs")
        if bindings.get(run_id) != checkpoint.checkpoint_sha256:
            raise RuntimeError(f"{run_id}: H2GCN progress checkpoint SHA256 is stale")
        row = rows_by_run.get(run_id)
        if row is None:
            raise RuntimeError(f"{run_id}: H2GCN progress row is missing")
        if row.get("checkpoint_sha256") != checkpoint.checkpoint_sha256:
            raise RuntimeError(f"{run_id}: H2GCN progress row checkpoint differs")
        if row.get("preparation", "erm") != preparation:
            raise RuntimeError(f"{run_id}: H2GCN progress row preparation differs")
        if graph_entry is None:
            if int(row.get("n_tail_test_prediction_differences_from_parent", -1)) != 0:
                raise RuntimeError(f"{run_id}: H2GCN baseline progress is unreproduced")
        else:
            observed_cell = (int(row["replicate_index"]), int(row["p_percent"]))
            expected_cell = (
                int(graph_entry["replicate_index"]),
                int(graph_entry["p_percent"]),
            )
            if observed_cell != expected_cell:
                raise RuntimeError(f"{run_id}: H2GCN graph progress cell differs")
    return H2Progress(
        rows_by_run=rows_by_run,
        cuda_peak_allocated_bytes=int(payload["cuda_peak_allocated_bytes"]),
        cuda_peak_reserved_bytes=int(payload["cuda_peak_reserved_bytes"]),
    )


def _load_h2_progress(
    path: Path,
    *,
    args,
    spec: DatasetSpec,
    catalog_sha256: str,
    checkpoints: dict[str, CheckpointSpec],
    graph_entry: dict | None,
) -> H2Progress | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    progress = _validate_h2_progress(
        payload,
        args=args,
        spec=spec,
        catalog_sha256=catalog_sha256,
        checkpoints=checkpoints,
        graph_entry=graph_entry,
    )
    print(f"validated existing H2GCN progress {path}; skipping", flush=True)
    return progress


def _write_h2_progress(
    path: Path,
    *,
    args,
    spec: DatasetSpec,
    catalog_sha256: str,
    checkpoints: dict[str, CheckpointSpec],
    graph_entry: dict | None,
    rows: list[dict],
    evaluation_seconds: float,
    cuda_memory: dict,
) -> H2Progress:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "design_id": evaluation_design_id(args),
        "completed": True,
        "phase": args.phase,
        "kind": "baseline" if graph_entry is None else "paired-graph",
        "dataset": spec.dataset,
        "model": "h2gcn",
        "code_version": args.code_version,
        "implementation_revision": implementation_revision(args),
        "git_dirty": args.git_dirty,
        "logit_reproduction_atol": getattr(
            args,
            "logit_reproduction_atol",
            LOGIT_REPRODUCTION_ATOL,
        ),
        "logit_reproduction_rtol": getattr(
            args,
            "logit_reproduction_rtol",
            LOGIT_REPRODUCTION_RTOL,
        ),
        "graph_catalog_sha256": catalog_sha256,
        "checkpoint_sha256": {
            run_id: checkpoint.checkpoint_sha256
            for run_id, checkpoint in checkpoints.items()
        },
        "rows": rows,
        "evaluation_seconds": evaluation_seconds,
        "completed_at": now_utc(),
    }
    preparation = preparation_from_args(args)
    if preparation != "erm":
        payload["preparation"] = preparation
    payload.update(cuda_memory)
    if graph_entry is not None:
        payload.update(
            {
                "pairing_key": graph_entry["pairing_key"],
                "replicate_index": int(graph_entry["replicate_index"]),
                "p_percent": int(graph_entry["p_percent"]),
                "source_graphs_sha256": graph_entry["graphs"]["sha256"],
            }
        )
    progress = _validate_h2_progress(
        payload,
        args=args,
        spec=spec,
        catalog_sha256=catalog_sha256,
        checkpoints=checkpoints,
        graph_entry=graph_entry,
    )
    atomic_json(path, payload)
    return progress


def evaluate_h2_graph_major(
    args,
    *,
    spec: DatasetSpec,
    bundle,
    arrays,
    registered_manifest: dict,
    catalog: dict,
    catalog_sha256: str,
    identities: Sequence[tuple[str, int]],
    x: torch.Tensor,
    original_edges: np.ndarray,
    device: torch.device,
) -> None:
    """Build each H2 operator once and atomically preserve graph progress."""

    pending: dict[str, CheckpointSpec] = {}
    for split, seed in identities:
        checkpoint = load_checkpoint_spec(
            spec,
            model="h2gcn",
            split=split,
            seed=seed,
            checkpoint_root=args.checkpoint_root,
            fixed_input_checksums=registered_manifest["array_checksums_sha256"],
            num_nodes=bundle.num_nodes,
            preparation=preparation_from_args(args),
        )
        if _validate_existing_shard(
            shard_path(args.output_root, checkpoint),
            spec=spec,
            model="h2gcn",
            run_id=checkpoint.run_id,
            graph_catalog_sha256=catalog_sha256,
            phase=args.phase,
            code_version=args.code_version,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            preparation=preparation_from_args(args),
            logit_atol=args.logit_reproduction_atol,
            logit_rtol=args.logit_reproduction_rtol,
        ):
            continue
        pending[checkpoint.run_id] = checkpoint
    if not pending:
        return

    baseline_path = h2_progress_path(args.output_root, spec, None)
    baseline_progress = _load_h2_progress(
        baseline_path,
        args=args,
        spec=spec,
        catalog_sha256=catalog_sha256,
        checkpoints=pending,
        graph_entry=None,
    )
    graph_progress: dict[str, H2Progress] = {}
    missing_entries: list[dict] = []
    for graph_entry in catalog["pairs"]:
        path = h2_progress_path(args.output_root, spec, graph_entry)
        existing = _load_h2_progress(
            path,
            args=args,
            spec=spec,
            catalog_sha256=catalog_sha256,
            checkpoints=pending,
            graph_entry=graph_entry,
        )
        if existing is None:
            missing_entries.append(graph_entry)
        else:
            graph_progress[graph_entry["pairing_key"]] = existing

    networks: dict[str, object] = {}
    if baseline_progress is None or missing_entries:
        for run_id, checkpoint in pending.items():
            networks[run_id] = load_network(
                checkpoint,
                input_dim=int(bundle.x.shape[1]),
                num_classes=bundle.num_classes,
                num_nodes=bundle.num_nodes,
                device=device,
            )
    try:
        if baseline_progress is None:
            reset_cuda_peak_memory(device)
            started = time.perf_counter()
            original_logits = h2_predictions_for_graph(
                networks,
                x,
                original_edges,
                device=device,
                pause_seconds=args.pause_seconds,
            )
            rows = [
                baseline_row_from_logits(
                    checkpoint,
                    bundle=bundle,
                    arrays=arrays,
                    current_logits=original_logits[run_id],
                    logit_atol=args.logit_reproduction_atol,
                    logit_rtol=args.logit_reproduction_rtol,
                )
                for run_id, checkpoint in pending.items()
            ]
            baseline_progress = _write_h2_progress(
                baseline_path,
                args=args,
                spec=spec,
                catalog_sha256=catalog_sha256,
                checkpoints=pending,
                graph_entry=None,
                rows=rows,
                evaluation_seconds=time.perf_counter() - started,
                cuda_memory=cuda_peak_memory(
                    device, scope="h2-baseline-progress-unit"
                ),
            )
            del original_logits
            print(f"completed H2GCN baseline progress {baseline_path}", flush=True)

        for graph_entry in missing_entries:
            reset_cuda_peak_memory(device)
            started = time.perf_counter()
            treatment_edges, control_edges = load_graph_edges(
                args.graph_root, spec, graph_entry
            )
            treatment_logits = h2_predictions_for_graph(
                networks,
                x,
                treatment_edges,
                device=device,
                pause_seconds=args.pause_seconds,
            )
            del treatment_edges
            control_logits = h2_predictions_for_graph(
                networks,
                x,
                control_edges,
                device=device,
                pause_seconds=args.pause_seconds,
            )
            del control_edges
            rows = []
            for run_id, checkpoint in pending.items():
                rows.append(
                    paired_metric_row(
                        checkpoint,
                        bundle=bundle,
                        arrays=arrays,
                        graph_entry=graph_entry,
                        treatment_predictions=treatment_logits[run_id]
                        .argmax(axis=-1)
                        .astype(np.int64, copy=False),
                        control_predictions=control_logits[run_id]
                        .argmax(axis=-1)
                        .astype(np.int64, copy=False),
                    )
                )
            del treatment_logits, control_logits
            path = h2_progress_path(args.output_root, spec, graph_entry)
            graph_progress[graph_entry["pairing_key"]] = _write_h2_progress(
                path,
                args=args,
                spec=spec,
                catalog_sha256=catalog_sha256,
                checkpoints=pending,
                graph_entry=graph_entry,
                rows=rows,
                evaluation_seconds=time.perf_counter() - started,
                cuda_memory=cuda_peak_memory(
                    device, scope="h2-paired-graph-progress-unit"
                ),
            )
            print(f"completed H2GCN graph progress {path}", flush=True)
    finally:
        networks.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert baseline_progress is not None
    progress_units = [baseline_progress, *graph_progress.values()]
    cuda_memory = {
        "cuda_peak_allocated_bytes": max(
            progress.cuda_peak_allocated_bytes for progress in progress_units
        ),
        "cuda_peak_reserved_bytes": max(
            progress.cuda_peak_reserved_bytes for progress in progress_units
        ),
        "cuda_peak_measurement_scope": (
            "h2-model-dataset-progress-unit-maximum"
        ),
    }
    for run_id, checkpoint in pending.items():
        rows = [
            graph_progress[entry["pairing_key"]].rows_by_run[run_id]
            for entry in catalog["pairs"]
        ]
        path = write_checkpoint_shard(
            args,
            checkpoint,
            graph_catalog_sha256=catalog_sha256,
            source_graph_manifest_sha256=catalog[
                "source_experiment_manifest_sha256"
            ],
            baseline=baseline_progress.rows_by_run[run_id],
            pairs=rows,
            schedule="graph-major-prepared-h2-operators",
            evaluation_seconds=0.0,
            cuda_memory=cuda_memory,
        )
        print(f"assembled {path} from atomic H2GCN progress", flush=True)


def evaluate_model_dataset(args, spec: DatasetSpec, model: str) -> None:
    preparation = preparation_from_args(args)
    if model not in models_for_preparation(spec, preparation):
        raise SystemExit(
            f"{spec.dataset}/{model} is not in the {preparation} evaluation panel"
        )
    catalog, catalog_sha256 = load_graph_catalog(
        spec,
        graph_root=args.graph_root,
        catalog_root=args.catalog_root,
    )
    bundle, arrays, registered_manifest = load_registered_inputs(
        spec,
        data_root=args.data_root,
        array_root=args.array_root,
        manifest_root=args.manifest_root,
    )
    try:
        identities = list(spec.checkpoint_identities)
        if args.phase == "smoke":
            identities = identities[: args.max_checkpoints]
            allowed_cells = {
                (replicate, percentage)
                for replicate in args.replicates
                for percentage in args.percentages
            }
            catalog = {
                **catalog,
                "pairs": [
                    entry
                    for entry in catalog["pairs"]
                    if (
                        int(entry["replicate_index"]),
                        int(entry["p_percent"]),
                    )
                    in allowed_cells
                ],
            }
        original_edges = np.asarray(
            to_simple_undirected(bundle.edge_index, bundle.num_nodes),
            dtype=np.int64,
        )
        device = torch.device(args.device)
        x = bundle.x.to(device)
        if model == "h2gcn":
            evaluate_h2_graph_major(
                args,
                spec=spec,
                bundle=bundle,
                arrays=arrays,
                registered_manifest=registered_manifest,
                catalog=catalog,
                catalog_sha256=catalog_sha256,
                identities=identities,
                x=x,
                original_edges=original_edges,
                device=device,
            )
        else:
            evaluate_checkpoint_major(
                args,
                spec=spec,
                model=model,
                bundle=bundle,
                arrays=arrays,
                registered_manifest=registered_manifest,
                catalog=catalog,
                catalog_sha256=catalog_sha256,
                identities=identities,
                x=x,
                original_edges=original_edges,
                device=device,
            )
        del x
    finally:
        arrays.close()


def cell_shard_set_sha256(
    output_root: Path, dataset: str, model: str
) -> tuple[int, str]:
    """Hash a cell as sorted relative shard paths plus each file's SHA256."""

    shard_root = output_root / dataset / model / "shards"
    entries = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(shard_root.glob("*.json"))
    ]
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return len(entries), hashlib.sha256(canonical).hexdigest()


def load_authenticated_validation(args) -> tuple[dict, str]:
    """Load the immutable accepted validation record after checking its hash."""

    provenance_path = args.release_provenance_manifest
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    release = provenance.get("release_validation", {})
    expected_sha256 = release.get("validation_json_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RuntimeError(
            f"{provenance_path}: missing release validation JSON SHA256"
        )
    validation_path = args.output_root / "final" / "validation.json"
    if not validation_path.is_file():
        raise RuntimeError(f"missing accepted validation record: {validation_path}")
    observed_sha256 = file_sha256(validation_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{validation_path}: SHA256 differs from accepted release; "
            f"expected {expected_sha256}, found {observed_sha256}"
        )
    return json.loads(validation_path.read_text(encoding="utf-8")), observed_sha256


def verify_release_provenance(args, validation: dict) -> None:
    """Authenticate an acquired release against its tracked provenance record."""

    path = args.release_provenance_manifest
    provenance = json.loads(path.read_text(encoding="utf-8"))
    expected_header = {
        "schema_version": 1,
        "design_id": evaluation_design_id(args),
        "record_kind": "accepted-release-provenance",
        "experiment_revision": args.code_version,
    }
    mismatch = {
        key: (value, provenance.get(key))
        for key, value in expected_header.items()
        if provenance.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"{path}: release provenance header mismatch {mismatch}")

    release = provenance.get("release_validation", {})
    expected_release = {
        "n_checkpoint_shards": validation["n_checkpoint_shards"],
        "n_baseline_rows": validation["n_baseline_rows"],
        "n_paired_rows": validation["n_paired_rows"],
        "baseline_rows_sha256": validation["output_sha256"]["baseline_rows.jsonl"],
        "paired_rows_sha256": validation["output_sha256"]["paired_rows.jsonl"],
    }
    release_mismatch = {
        key: (value, release.get(key))
        for key, value in expected_release.items()
        if release.get(key) != value
    }
    if release_mismatch:
        raise RuntimeError(
            f"{path}: consolidated release outputs differ {release_mismatch}"
        )
    if provenance.get("graph_catalog_sha256") != validation["graph_catalog_sha256"]:
        raise RuntimeError(f"{path}: graph catalog hashes differ from release")

    environments = provenance.get("runtime_environments")
    if not isinstance(environments, dict) or not environments:
        raise RuntimeError(f"{path}: runtime_environments must be a non-empty object")
    for environment_id, record in environments.items():
        requirements_path = REPO_ROOT / str(record.get("requirements", ""))
        if not requirements_path.is_file():
            raise RuntimeError(
                f"{path}: missing lock for environment {environment_id}: "
                f"{requirements_path}"
            )
        if file_sha256(requirements_path) != record.get("requirements_sha256"):
            raise RuntimeError(f"{path}: lock hash differs for {environment_id}")
    jobs = provenance.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise RuntimeError(f"{path}: jobs must be a non-empty object")

    accepted = provenance.get("accepted_cells")
    if not isinstance(accepted, list):
        raise RuntimeError(f"{path}: accepted_cells must be a list")
    preparation = preparation_from_args(args)
    expected_cells = set(model_dataset_cells_for_preparation(preparation))
    observed_cells: set[tuple[str, str]] = set()
    for record in accepted:
        cell = (str(record.get("dataset")), str(record.get("model")))
        if cell in observed_cells:
            raise RuntimeError(f"{path}: duplicate accepted cell {cell}")
        observed_cells.add(cell)
        contributors = record.get("contributing_jobs")
        if not isinstance(contributors, list) or not contributors:
            raise RuntimeError(f"{path}: accepted cell {cell} has no contributing jobs")
        unknown_jobs = set(map(str, contributors)) - set(jobs)
        if unknown_jobs:
            raise RuntimeError(
                f"{path}: accepted cell {cell} cites unknown jobs {sorted(unknown_jobs)}"
            )
        count, digest = cell_shard_set_sha256(args.output_root, *cell)
        if count != int(record.get("checkpoint_shards", -1)):
            raise RuntimeError(f"{path}: accepted shard count differs for {cell}")
        if digest != record.get("shard_set_sha256"):
            raise RuntimeError(f"{path}: accepted shard hash differs for {cell}")
    if observed_cells != expected_cells:
        raise RuntimeError(f"{path}: accepted cell panel differs from fixed design")


def collect_and_validate_outputs(args, *, write_outputs: bool = True) -> dict:
    """Validate the matrix, optionally writing deterministic consolidated rows."""

    baselines: list[dict] = []
    paired_rows: list[dict] = []
    treatment_keys: set[str] = set()
    control_keys: set[str] = set()
    expected_shards: set[Path] = set()
    dataset_catalog_hashes: dict[str, str] = {}
    preparation = preparation_from_args(args)
    design_id = design_id_for_preparation(preparation)
    expected_cells = model_dataset_cells_for_preparation(preparation)
    expected_checkpoints = expected_checkpoint_count(preparation)
    expected_pairs = expected_paired_row_count(preparation)
    for dataset, spec in DATASET_SPECS.items():
        _, catalog_sha256 = load_graph_catalog(
            spec,
            graph_root=args.graph_root,
            catalog_root=args.catalog_root,
        )
        dataset_catalog_hashes[dataset] = catalog_sha256
        for model in models_for_preparation(spec, preparation):
            for split, seed in spec.checkpoint_identities:
                run_id = parent_run_id(spec, model, split, seed, preparation)
                path = (
                    args.output_root
                    / dataset
                    / model
                    / "shards"
                    / f"{run_id}.json"
                )
                expected_shards.add(path.resolve())
                if not path.is_file():
                    raise RuntimeError(f"missing checkpoint shard: {path}")
                checkpoint_path = args.checkpoint_root / run_id / "checkpoint.pt"
                if not checkpoint_path.is_file():
                    raise RuntimeError(f"missing current checkpoint: {checkpoint_path}")
                checkpoint_sha256 = file_sha256(checkpoint_path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                validate_shard_payload(
                    payload,
                    spec=spec,
                    model=model,
                    run_id=run_id,
                    graph_catalog_sha256=catalog_sha256,
                    expected_phase="actual",
                    expected_code_version=args.code_version,
                    expected_checkpoint_sha256=checkpoint_sha256,
                    expected_preparation=preparation,
                )
                baselines.append(payload["baseline"])
                paired_rows.extend(payload["pairs"])
                for row in payload["pairs"]:
                    treatment_key = row["treatment_prediction_key"]
                    control_key = row["control_prediction_key"]
                    if treatment_key in treatment_keys:
                        raise RuntimeError(f"duplicate treatment key {treatment_key}")
                    if control_key in control_keys:
                        raise RuntimeError(f"duplicate control key {control_key}")
                    treatment_keys.add(treatment_key)
                    control_keys.add(control_key)
    observed_shards = {
        path.resolve()
        for dataset, spec in DATASET_SPECS.items()
        for model in models_for_preparation(spec, preparation)
        for path in (args.output_root / dataset / model / "shards").glob("*.json")
    }
    extras = observed_shards - expected_shards
    if extras:
        raise RuntimeError(f"unexpected expansion shards: {sorted(map(str, extras))}")
    if len(baselines) != expected_checkpoints:
        raise RuntimeError(
            f"expected {expected_checkpoints} baseline rows; got {len(baselines)}"
        )
    if len(paired_rows) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} paired rows; got {len(paired_rows)}"
        )
    if len(treatment_keys) != expected_pairs:
        raise RuntimeError("treatment prediction keys are not globally unique")
    if len(control_keys) != expected_pairs:
        raise RuntimeError("control prediction keys are not globally unique")
    if treatment_keys & control_keys:
        raise RuntimeError("treatment and control prediction key spaces overlap")

    final_root = args.output_root / "final"
    baseline_path = final_root / "baseline_rows.jsonl"
    paired_path = final_root / "paired_rows.jsonl"
    expected_output_sha256 = {
        "baseline_rows.jsonl": jsonl_sha256(baselines),
        "paired_rows.jsonl": jsonl_sha256(paired_rows),
    }
    if write_outputs:
        atomic_jsonl(baseline_path, baselines)
        atomic_jsonl(paired_path, paired_rows)
    else:
        for output_path in (baseline_path, paired_path):
            if not output_path.is_file():
                raise RuntimeError(f"missing accepted consolidated output: {output_path}")
            observed = file_sha256(output_path)
            expected = expected_output_sha256[output_path.name]
            if observed != expected:
                raise RuntimeError(
                    f"{output_path}: contents differ from the validated shards; "
                    f"expected {expected}, found {observed}"
                )
    validation = {
        "schema_version": SCHEMA_VERSION,
        "design_id": design_id,
        "validated": True,
        "validated_at": now_utc(),
        "n_model_dataset_cells": len(expected_cells),
        "n_checkpoint_shards": len(expected_shards),
        "n_baseline_rows": len(baselines),
        "n_paired_rows": len(paired_rows),
        "n_unique_treatment_prediction_keys": len(treatment_keys),
        "n_unique_control_prediction_keys": len(control_keys),
        "graph_catalog_sha256": dataset_catalog_hashes,
        "experiment_revision": args.code_version,
        "validation_implementation_revision": implementation_revision(args),
        "output_sha256": expected_output_sha256,
    }
    if preparation != "erm":
        validation["preparation"] = preparation
    if write_outputs:
        atomic_json(final_root / "validation.json", validation)
    return validation


def authenticate_existing_release(args) -> dict:
    """Authenticate an accepted release without changing any release file."""

    accepted_validation, accepted_sha256 = load_authenticated_validation(args)
    recomputed = collect_and_validate_outputs(args, write_outputs=False)
    provenance = json.loads(
        args.release_provenance_manifest.read_text(encoding="utf-8")
    )
    pinned_revision = provenance.get("experiment_revision")
    preparation = preparation_from_args(args)
    allow_legacy_revision_omission = (
        preparation == "erm"
        and "experiment_revision" not in accepted_validation
        and pinned_revision == ACCEPTED_RELEASE_EXPERIMENT_REVISION
        and args.code_version == ACCEPTED_RELEASE_EXPERIMENT_REVISION
        and recomputed.get("experiment_revision") == pinned_revision
    )
    stable_fields = (
        "schema_version",
        "design_id",
        "validated",
        "n_model_dataset_cells",
        "n_checkpoint_shards",
        "n_baseline_rows",
        "n_paired_rows",
        "n_unique_treatment_prediction_keys",
        "n_unique_control_prediction_keys",
        "graph_catalog_sha256",
        "experiment_revision",
        "output_sha256",
        "preparation",
    )
    mismatch = {
        key: (accepted_validation.get(key), recomputed.get(key))
        for key in stable_fields
        if accepted_validation.get(key) != recomputed.get(key)
        and not (
            key == "experiment_revision" and allow_legacy_revision_omission
        )
    }
    if mismatch:
        raise RuntimeError(
            "accepted validation record differs from read-only revalidation: "
            f"{mismatch}"
        )
    verify_release_provenance(args, recomputed)
    validation_path = args.output_root / "final" / "validation.json"
    if file_sha256(validation_path) != accepted_sha256:
        raise RuntimeError("accepted validation record changed during authentication")
    return accepted_validation


def _add_input_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)


def _add_experiment_revision(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-revision",
        help=(
            "frozen Git commit recorded in generated outputs; defaults to the "
            "current commit"
        ),
    )


def _add_preparation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preparation",
        choices=PREPARATIONS,
        default="erm",
        help=(
            "checkpoint preparation to evaluate; non-ERM preparations use the "
            "complete 6x5 graph-model matrix"
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-graphs")
    generate.add_argument("--phase", choices=("smoke", "actual"), required=True)
    generate.add_argument(
        "--datasets",
        nargs="+",
        choices=("roman-empire", "amazon-ratings"),
        required=True,
    )
    generate.add_argument(
        "--replicates", nargs="+", type=int, default=list(REPLICATE_INDICES)
    )
    generate.add_argument(
        "--percentages", nargs="+", type=int, required=True
    )
    generate.add_argument("--graph-root", type=Path, required=True)
    _add_input_roots(generate)
    _add_experiment_revision(generate)

    stage = subparsers.add_parser("stage-reused-graphs")
    stage.add_argument(
        "--dataset",
        choices=tuple(
            dataset for dataset, spec in DATASET_SPECS.items() if spec.graph_is_reused
        ),
        required=True,
    )
    stage.add_argument("--source-dataset-root", type=Path, required=True)
    stage.add_argument("--graph-root", type=Path, required=True)
    _add_experiment_revision(stage)

    authenticate = subparsers.add_parser("authenticate-graphs")
    authenticate.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    authenticate.add_argument("--graph-root", type=Path, required=True)
    authenticate.add_argument("--catalog-root", type=Path, required=True)
    _add_input_roots(authenticate)
    _add_experiment_revision(authenticate)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--phase", choices=("smoke", "actual"), required=True)
    evaluate.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    evaluate.add_argument(
        "--model",
        choices=tuple(sorted({model for _, model in MODEL_DATASET_CELLS})),
        required=True,
    )
    evaluate.add_argument("--graph-root", type=Path, required=True)
    evaluate.add_argument("--catalog-root", type=Path, required=True)
    evaluate.add_argument("--checkpoint-root", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    _add_preparation(evaluate)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--torch-threads", type=int, default=2)
    evaluate.add_argument("--cuda-memory-fraction", type=float, default=0.9)
    evaluate.add_argument("--pause-seconds", type=float, default=0.0)
    evaluate.add_argument(
        "--logit-reproduction-atol",
        type=float,
        default=LOGIT_REPRODUCTION_ATOL,
    )
    evaluate.add_argument(
        "--logit-reproduction-rtol",
        type=float,
        default=LOGIT_REPRODUCTION_RTOL,
    )
    evaluate.add_argument("--max-checkpoints", type=int, default=1)
    evaluate.add_argument("--replicates", nargs="+", type=int, default=[1])
    evaluate.add_argument("--percentages", nargs="+", type=int, default=[5])
    _add_input_roots(evaluate)
    _add_experiment_revision(evaluate)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--graph-root", type=Path, required=True)
    validate.add_argument("--catalog-root", type=Path, required=True)
    validate.add_argument("--checkpoint-root", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)
    _add_preparation(validate)
    validate.add_argument("--release-provenance-manifest", type=Path)
    _add_experiment_revision(validate)
    return parser.parse_args(argv)


def _resolve_paths(args) -> None:
    read_only_release = (
        args.command == "validate"
        and getattr(args, "release_provenance_manifest", None) is not None
    )
    for name in (
        "data_root",
        "array_root",
        "manifest_root",
        "graph_root",
        "catalog_root",
        "checkpoint_root",
        "output_root",
        "source_dataset_root",
        "release_provenance_manifest",
    ):
        if not hasattr(args, name):
            continue
        raw_path = getattr(args, name)
        if raw_path is None:
            continue
        path = raw_path.expanduser().resolve()
        output_like = (
            name in {"graph_root", "catalog_root", "output_root"}
            and not read_only_release
        )
        if not path.exists() and not output_like:
            raise SystemExit(f"{name.replace('_', ' ')} does not exist: {path}")
        if output_like:
            path.mkdir(parents=True, exist_ok=True)
        setattr(args, name, path)


def validate_args(args) -> None:
    if hasattr(args, "datasets"):
        args.datasets = list(dict.fromkeys(args.datasets))
    if args.command == "generate-graphs":
        args.replicates = sorted(set(args.replicates))
        args.percentages = sorted(set(args.percentages))
        if args.phase == "actual" and len(args.datasets) != 1:
            raise SystemExit("actual graph generation requires exactly one dataset")
        if set(args.replicates) - set(REPLICATE_INDICES):
            raise SystemExit(f"replicates must come from {REPLICATE_INDICES}")
        if set(args.percentages) - set(NONZERO_PERCENTAGES):
            raise SystemExit(f"percentages must come from {NONZERO_PERCENTAGES}")
    if args.command == "evaluate":
        spec = DATASET_SPECS[args.dataset]
        allowed_models = models_for_preparation(spec, args.preparation)
        if args.model not in allowed_models:
            raise SystemExit(
                f"{args.dataset}/{args.model} is not in the "
                f"{args.preparation} evaluation panel"
            )
        if args.torch_threads < 1:
            raise SystemExit("--torch-threads must be positive")
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise SystemExit("--cuda-memory-fraction must be in (0, 1]")
        if args.pause_seconds < 0:
            raise SystemExit("--pause-seconds must be non-negative")
        if not np.isfinite(args.logit_reproduction_atol) or args.logit_reproduction_atol < 0:
            raise SystemExit("--logit-reproduction-atol must be finite and non-negative")
        if not np.isfinite(args.logit_reproduction_rtol) or args.logit_reproduction_rtol < 0:
            raise SystemExit("--logit-reproduction-rtol must be finite and non-negative")
        if args.phase == "actual":
            if args.max_checkpoints != 1 or args.replicates != [1] or args.percentages != [5]:
                # Smoke-only selectors keep harmless parser defaults in actual mode.
                raise SystemExit(
                    "--max-checkpoints/--replicates/--percentages are smoke-only"
                )
        else:
            if not 1 <= args.max_checkpoints <= 30:
                raise SystemExit("--max-checkpoints must be in [1, 30]")
            if set(args.replicates) - set(REPLICATE_INDICES):
                raise SystemExit(f"replicates must come from {REPLICATE_INDICES}")
            if set(args.percentages) - set(spec.nonzero_percentages):
                raise SystemExit(
                    f"{spec.dataset} percentages must come from "
                    f"{spec.nonzero_percentages}"
                )
        if not str(args.device).startswith("cuda"):
            raise SystemExit("the expansion evaluator requires a CUDA device")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    _resolve_paths(args)
    args.implementation_revision, args.git_dirty = git_state()
    args.code_version = resolve_git_revision(
        args.experiment_revision or args.implementation_revision
    )
    args.experiment_revision = args.code_version
    if getattr(args, "phase", None) == "actual" and args.git_dirty:
        raise SystemExit("actual collection requires a clean committed worktree")

    if args.command == "generate-graphs":
        for dataset in args.datasets:
            generate_graph_dataset(args, DATASET_SPECS[dataset])
        return
    if args.command == "stage-reused-graphs":
        stage_reused_graph_dataset(args, DATASET_SPECS[args.dataset])
        return
    if args.command == "authenticate-graphs":
        for dataset in args.datasets:
            authenticate_graph_dataset(args, DATASET_SPECS[dataset])
        return
    if args.command == "evaluate":
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        device = torch.device(args.device)
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=device_index
        )
        args.priority_lowered = lower_process_priority()
        evaluate_model_dataset(args, DATASET_SPECS[args.dataset], args.model)
        return
    if args.command == "validate":
        if args.release_provenance_manifest is None:
            result = collect_and_validate_outputs(args)
            action = "validated"
        else:
            result = authenticate_existing_release(args)
            action = "authenticated"
        print(
            f"{action} {result['n_baseline_rows']} baselines and "
            f"{result['n_paired_rows']} paired rows",
            flush=True,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
