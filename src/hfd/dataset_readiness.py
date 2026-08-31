"""Pre-training checks for a dataset proposed for the tail-gap protocol."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .datasets import DatasetBundle
from .homophily import compute_homophily_groups, to_simple_undirected


def _counts(values: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def inspect_dataset_readiness(bundle: DatasetBundle) -> dict:
    """Check loading, splits, local-homophily bins, and tail support.

    This performs no model fitting. ``errors`` contains violations that make a
    run unsafe or incompatible with the registered protocol. Small per-class
    tail counts are reported separately because they limit interpretation but
    do not make training invalid.
    """
    y = np.asarray(bundle.y)
    if y.ndim != 1 or y.shape[0] != bundle.num_nodes:
        raise ValueError(
            "labels must be a one-dimensional array with one value per node"
        )
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must use an integer dtype")
    if bundle.x.ndim != 2 or bundle.x.shape[0] != bundle.num_nodes:
        raise ValueError("features must have shape (num_nodes, num_features)")
    if bundle.edge_index.ndim != 2 or bundle.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges)")

    errors: list[str] = []
    warnings: list[str] = []
    if np.any(y < 0):
        errors.append("all nodes must be labelled for this extension")
    if bundle.num_classes < 3:
        errors.append("the extension requires a multiclass dataset")

    simple = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
    groups = compute_homophily_groups(bundle.edge_index, y, bundle.num_nodes)
    tail_idx = np.flatnonzero(groups.tail_mask)
    tail_class_counts = _counts(y[tail_idx])
    if groups.structure.direction != "heterophilic":
        errors.append(
            "mean local homophily does not define the required high-h_v tail"
        )
    if tail_idx.size == 0:
        errors.append("the post-merge tail is empty")
    if len(tail_class_counts) < 2:
        errors.append("the post-merge tail contains fewer than two classes")
    absent_tail_classes = sorted(set(range(bundle.num_classes)) - {
        int(label) for label in tail_class_counts
    })
    if absent_tail_classes:
        warnings.append(
            "the global tail has no nodes from class(es) "
            + ", ".join(map(str, absent_tail_classes))
        )
    rare_tail_classes = sorted(
        int(label) for label, count in tail_class_counts.items() if count < 10
    )
    if rare_tail_classes:
        warnings.append(
            "the global tail has fewer than 10 nodes from class(es) "
            + ", ".join(map(str, rare_tail_classes))
        )
    if len(bundle.splits) * len(bundle.training_seeds) != 30:
        errors.append("the split/seed panel must define exactly 30 runs")

    all_nodes = np.arange(bundle.num_nodes, dtype=np.int64)
    split_rows = []
    for split in bundle.splits:
        parts = {
            "train": np.asarray(split.train_idx, dtype=np.int64),
            "val": np.asarray(split.val_idx, dtype=np.int64),
            "test": np.asarray(split.test_idx, dtype=np.int64),
        }
        for part_name, indices in parts.items():
            if indices.ndim != 1:
                raise ValueError(
                    f"{split.split_id}: {part_name} indices must be one-dimensional"
                )
            if np.any(indices < 0) or np.any(indices >= bundle.num_nodes):
                raise ValueError(
                    f"{split.split_id}: {part_name} contains an out-of-range index"
                )
        combined = np.concatenate(list(parts.values()))
        if (
            combined.size != bundle.num_nodes
            or np.unique(combined).size != bundle.num_nodes
            or not np.array_equal(np.sort(combined), all_nodes)
        ):
            errors.append(
                f"{split.split_id}: train/validation/test do not partition all nodes"
            )

        part_rows = {}
        for part_name, indices in parts.items():
            if indices.size == 0:
                errors.append(f"{split.split_id}: {part_name} split is empty")
            class_counts = _counts(y[indices])
            if len(class_counts) != bundle.num_classes:
                errors.append(
                    f"{split.split_id}: {part_name} does not contain every class"
                )
            part_tail = indices[groups.tail_mask[indices]]
            tail_counts = _counts(y[part_tail])
            eligible_bins = groups.merged_bin[indices]
            bin_counts = Counter(eligible_bins[eligible_bins >= 0].tolist())
            if part_name == "train" and len(bin_counts) != len(
                groups.structure.bins
            ):
                errors.append(
                    f"{split.split_id}: training split does not contain every "
                    "post-merge homophily bin required by GroupDRO"
                )
            if part_tail.size == 0:
                errors.append(f"{split.split_id}: {part_name} tail is empty")
            if len(tail_counts) < 2:
                errors.append(
                    f"{split.split_id}: {part_name} tail contains fewer than two classes"
                )
            if part_name in {"val", "test"} and tail_counts and min(
                tail_counts.values()
            ) < 10:
                warnings.append(
                    f"{split.split_id}: {part_name} tail includes a class with "
                    "fewer than 10 nodes"
                )
            part_rows[part_name] = {
                "n": int(indices.size),
                "class_counts": class_counts,
                "n_tail": int(part_tail.size),
                "tail_class_counts": tail_counts,
                "n_tail_classes": len(tail_counts),
                "n_tail_classes_ge_10": sum(
                    count >= 10 for count in tail_counts.values()
                ),
                "bin_counts": {
                    str(index): int(bin_counts[index])
                    for index in range(len(groups.structure.bins))
                },
            }
        split_rows.append({"split_id": split.split_id, **part_rows})

    base_counts = Counter(groups.base_bin[groups.eligible].tolist())
    merged_bins = [
        {
            "interval": (
                f"[{bin_.lo:g},{bin_.hi:g}{']' if bin_.hi == 1.0 else ')'}"
            ),
            "count": int(bin_.count),
            "is_tail": index == groups.structure.tail_bin_index,
        }
        for index, bin_ in enumerate(groups.structure.bins)
    ]
    return {
        "dataset": bundle.name,
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "graph": {
            "num_nodes": bundle.num_nodes,
            "num_edges_as_released": int(bundle.edge_index.shape[1]),
            "num_undirected_simple_edges": int(simple.shape[1] // 2),
            "feature_dim": int(bundle.x.shape[1]),
            "feature_dtype": str(bundle.x.dtype),
            "feature_layout": str(bundle.x.layout),
            "feature_storage_bytes": int(
                bundle.x.numel() * bundle.x.element_size()
            ),
            "num_classes": bundle.num_classes,
            "class_counts": _counts(y),
        },
        "local_homophily": {
            "mean_h": groups.structure.mean_h,
            "direction": groups.structure.direction,
            "n_undefined_h": groups.n_undefined_h,
            "base_bin_counts": {
                str(index): int(base_counts[index]) for index in range(4)
            },
            "merged_bins": merged_bins,
            "tail_count": int(tail_idx.size),
            "tail_fraction": float(tail_idx.size / groups.eligible.sum()),
            "tail_class_counts": tail_class_counts,
        },
        "split_panel": bundle.split_panel,
        "training_seeds": bundle.training_seeds,
        "n_runs_per_model_preparation": (
            len(bundle.splits) * len(bundle.training_seeds)
        ),
        "splits": split_rows,
    }
