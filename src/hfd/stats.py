"""Dataset statistics and fixed-input manifest construction.

For each dataset this produces:
- a JSON manifest (git-tracked) with graph stats, the h_v/bin summary, the
  split panel description, and SHA256 checksums of every node-level array;
- an .npz of the node-level arrays themselves (h_v, bin ids, tail mask,
  split indices), stored under artifacts/ because binaries are not tracked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .datasets import DatasetBundle
from .homophily import HomophilyGroups, compute_homophily_groups, to_simple_undirected


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_arrays(bundle: DatasetBundle, groups: HomophilyGroups) -> dict[str, np.ndarray]:
    """Node-level fixed inputs to freeze: h_v, groups, and split indices."""
    arrays = {
        "h": groups.h,
        "eligible": groups.eligible,
        "base_bin": groups.base_bin,
        "merged_bin": groups.merged_bin,
        "tail_mask": groups.tail_mask,
        "labels": bundle.y,
    }
    for split in bundle.splits:
        arrays[f"{split.split_id}.train"] = split.train_idx
        arrays[f"{split.split_id}.val"] = split.val_idx
        arrays[f"{split.split_id}.test"] = split.test_idx
    return arrays


def compute_dataset_stats(bundle: DatasetBundle) -> tuple[dict, HomophilyGroups]:
    """Compute the full fixed-input stats manifest for one dataset."""
    n = bundle.num_nodes
    groups = compute_homophily_groups(bundle.edge_index, bundle.y, n)
    simple = to_simple_undirected(bundle.edge_index, n)
    eligible = groups.eligible
    n_eligible = int(eligible.sum())
    structure = groups.structure

    base_counts = [int((groups.base_bin[eligible] == b).sum()) for b in range(4)]
    base_edges = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1]"]

    merged_bins = []
    for i, mb in enumerate(structure.bins):
        closed = "]" if mb.hi == 1.0 else ")"
        in_bin = eligible & (groups.merged_bin == i)
        classes, counts = np.unique(bundle.y[in_bin], return_counts=True)
        merged_bins.append(
            {
                "interval": f"[{mb.lo:g},{mb.hi:g}{closed}",
                "base_bins": mb.base_ids,
                "count": mb.count,
                "fraction_of_eligible": mb.count / n_eligible,
                "is_tail": i == structure.tail_bin_index,
                "class_composition": {
                    int(c): int(n) for c, n in zip(classes, counts)
                },
            }
        )

    split_rows = []
    for split in bundle.splits:
        test_eligible = int(eligible[split.test_idx].sum())
        test_tail = int(groups.tail_mask[split.test_idx].sum())
        train_tail = int(groups.tail_mask[split.train_idx].sum())
        val_tail = int(groups.tail_mask[split.val_idx].sum())
        split_rows.append(
            {
                "split_id": split.split_id,
                "n_train": int(split.train_idx.size),
                "n_val": int(split.val_idx.size),
                "n_test": int(split.test_idx.size),
                "n_train_tail": train_tail,
                "n_val_tail": val_tail,
                "n_test_eligible": test_eligible,
                "n_test_tail": test_tail,
            }
        )

    stats = {
        "dataset": bundle.name,
        "metric": bundle.metric,
        "split_panel": bundle.split_panel,
        "training_seeds": bundle.training_seeds,
        "n_runs": len(bundle.splits) * len(bundle.training_seeds),
        "graph": {
            "num_nodes": n,
            "num_edges_as_released": int(bundle.edge_index.shape[1]),
            "num_undirected_simple_edges": int(simple.shape[1] // 2),
            "feature_dim": int(bundle.x.shape[1]),
            "num_classes": bundle.num_classes,
        },
        "local_homophily": {
            "n_labelled": groups.n_labelled,
            "n_undefined_h": groups.n_undefined_h,
            "n_eligible": n_eligible,
            "mean_h": structure.mean_h,
            "direction": structure.direction,
            "tail_side": "low-h_v" if structure.direction == "homophilic" else "high-h_v",
            "base_bins": [
                {
                    "interval": base_edges[b],
                    "count": base_counts[b],
                    "fraction_of_eligible": base_counts[b] / n_eligible,
                }
                for b in range(4)
            ],
            "merge_history": structure.merge_history,
            "merged_bins": merged_bins,
            "tail": {
                "interval": merged_bins[structure.tail_bin_index]["interval"],
                "count": structure.tail_bin.count,
                "fraction_of_eligible": structure.tail_bin.count / n_eligible,
            },
        },
        "splits": split_rows,
        "notes": bundle.notes,
        "graph_checksums_sha256": {
            "features": _sha256(bundle.x.detach().cpu().numpy()),
            "simple_edge_index": _sha256(simple),
        },
    }
    return stats, groups


def attach_checksums(stats: dict, arrays: dict[str, np.ndarray]) -> None:
    stats["array_checksums_sha256"] = {k: _sha256(v) for k, v in sorted(arrays.items())}


def save_arrays(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
