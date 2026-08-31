"""Audit true-class support across equal-width local-homophily bins.

The prospective class-adjusted h_v curves require one common set of true
classes within each registered test split. A class belongs to that set only
when at least one test node of the class occurs in every plotted h_v bin.

This script reads only the frozen dataset manifests. It verifies their
checksums, applies the same [left, right) bin convention as the RQ1 plotting
code (with h_v=1 in the final bin), and writes split-level and dataset-level
support tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASETS = (
    "pubmed",
    "ogbn-arxiv",
    "roman-empire",
    "amazon-ratings",
    "arxiv-year",
)
BIN_COUNTS = (4, 6, 8, 10, 20)


def array_sha256(array: np.ndarray) -> str:
    """Return the checksum convention used by the fixed-input manifests."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_equal_width_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign [left, right) bins; assign h_v=1 to the final bin."""
    if n_bins < 2:
        raise ValueError("At least two h_v bins are required")
    values = np.asarray(values, dtype=np.float64)
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    result = np.full(values.shape, -1, dtype=np.int64)
    valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    result[valid] = np.searchsorted(edges, values[valid], side="right") - 1
    result[valid & (values == 1.0)] = n_bins - 1
    return result


def calculate_split_support(
    h: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    """Calculate the classes represented in every h_v bin of one test split."""
    test_h = h[test_idx]
    test_labels = labels[test_idx]
    test_bins = assign_equal_width_bins(test_h, n_bins)
    if np.any(test_bins < 0):
        raise RuntimeError("A registered test node has undefined or invalid h_v")

    class_sets: list[set[int]] = []
    class_counts_by_bin: list[int] = []
    test_node_counts_by_bin: list[int] = []
    for bin_index in range(n_bins):
        mask = test_bins == bin_index
        classes = set(int(value) for value in np.unique(test_labels[mask]))
        class_sets.append(classes)
        class_counts_by_bin.append(len(classes))
        test_node_counts_by_bin.append(int(mask.sum()))

    common_classes = (
        set.intersection(*class_sets) if class_sets else set()
    )
    return {
        "classes_present_every_bin": len(common_classes),
        "class_ids_present_every_bin": sorted(common_classes),
        "class_counts_by_bin": class_counts_by_bin,
        "test_node_counts_by_bin": test_node_counts_by_bin,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_base_arrays(
    arrays: np.lib.npyio.NpzFile,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    expected_checksums = manifest["array_checksums_sha256"]
    for key in ("h", "labels"):
        if key not in arrays.files:
            raise KeyError(f"Frozen array file lacks {key!r}")
        if array_sha256(arrays[key]) != expected_checksums[key]:
            raise RuntimeError(f"Frozen-array checksum mismatch for {key}")

    h = np.asarray(arrays["h"], dtype=np.float64)
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    expected_nodes = int(manifest["graph"]["num_nodes"])
    if h.shape != (expected_nodes,) or labels.shape != (expected_nodes,):
        raise RuntimeError("Frozen arrays do not match the manifest node count")
    return h, labels


def validate_test_indices(
    arrays: np.lib.npyio.NpzFile,
    manifest: dict[str, Any],
    split: dict[str, Any],
    num_nodes: int,
) -> tuple[str, np.ndarray]:
    split_id = str(split["split_id"])
    key = f"{split_id}.test"
    if key not in arrays.files:
        raise KeyError(f"Frozen array file lacks {key!r}")
    test_idx = np.asarray(arrays[key], dtype=np.int64)
    expected_checksum = manifest["array_checksums_sha256"][key]
    if array_sha256(test_idx) != expected_checksum:
        raise RuntimeError(f"Frozen-array checksum mismatch for {key}")
    if test_idx.shape != (int(split["n_test"]),):
        raise RuntimeError(f"{key} does not match the registered test size")
    if np.unique(test_idx).size != test_idx.size:
        raise RuntimeError(f"{key} contains duplicate node indices")
    if np.any(test_idx < 0) or np.any(test_idx >= num_nodes):
        raise RuntimeError(f"{key} contains an out-of-range node index")
    return split_id, test_idx


def analyse_dataset(
    repo_root: Path,
    dataset: str,
    bin_counts: Iterable[int],
    dataset_order: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = repo_root / "manifests" / "datasets" / f"{dataset}.json"
    arrays_path = repo_root / "artifacts" / "manifests" / f"{dataset}.npz"
    manifest = load_json(manifest_path)
    if manifest.get("dataset") != dataset:
        raise RuntimeError(f"Dataset identity mismatch in {manifest_path}")

    rows: list[dict[str, Any]] = []
    split_ids: list[str] = []
    with np.load(arrays_path) as arrays:
        h, labels = validate_base_arrays(arrays, manifest)
        num_nodes = h.size
        num_classes = int(manifest["graph"]["num_classes"])
        for split_order, split in enumerate(manifest["splits"]):
            split_id, test_idx = validate_test_indices(
                arrays, manifest, split, num_nodes
            )
            split_ids.append(split_id)
            test_labels = labels[test_idx]
            if np.any(test_labels < 0) or np.any(test_labels >= num_classes):
                raise RuntimeError(f"{split_id} has an invalid test label")

            for n_bins in bin_counts:
                support = calculate_split_support(h, labels, test_idx, n_bins)
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_order": dataset_order,
                        "split": split_id,
                        "split_order": split_order,
                        "num_total_classes": num_classes,
                        "n_bins": n_bins,
                        **support,
                        "total_test_nodes": int(test_idx.size),
                    }
                )

    input_record = {
        "dataset": dataset,
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "arrays_path": arrays_path.relative_to(repo_root).as_posix(),
        "arrays_sha256": file_sha256(arrays_path),
        "split_ids": split_ids,
    }
    return rows, input_record


def build_support_rows(
    repo_root: Path,
    datasets: Iterable[str] = DATASETS,
    bin_counts: Iterable[int] = BIN_COUNTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = tuple(datasets)
    bin_counts = tuple(bin_counts)
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for dataset_order, dataset in enumerate(datasets):
        dataset_rows, input_record = analyse_dataset(
            repo_root, dataset, bin_counts, dataset_order
        )
        rows.extend(dataset_rows)
        inputs.append(input_record)
    return rows, inputs


def build_summary_rows(
    split_rows: list[dict[str, Any]],
    datasets: Iterable[str] = DATASETS,
    bin_counts: Iterable[int] = BIN_COUNTS,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for dataset_order, dataset in enumerate(datasets):
        dataset_rows = [row for row in split_rows if row["dataset"] == dataset]
        for n_bins in bin_counts:
            rows = sorted(
                (row for row in dataset_rows if row["n_bins"] == n_bins),
                key=lambda row: row["split_order"],
            )
            if not rows:
                raise RuntimeError(f"No rows for {dataset} with {n_bins} bins")
            counts = [int(row["classes_present_every_bin"]) for row in rows]
            summary.append(
                {
                    "dataset": dataset,
                    "dataset_order": dataset_order,
                    "num_total_classes": rows[0]["num_total_classes"],
                    "num_splits": len(rows),
                    "n_bins": n_bins,
                    "min_classes_present_every_bin": min(counts),
                    "max_classes_present_every_bin": max(counts),
                    "counts_by_split": counts,
                }
            )
    return summary


def serialise_sequence(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def write_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "dataset",
        "dataset_order",
        "split",
        "split_order",
        "num_total_classes",
        "n_bins",
        "classes_present_every_bin",
        "class_ids_present_every_bin",
        "class_counts_by_bin",
        "test_node_counts_by_bin",
        "total_test_nodes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in (
                "class_ids_present_every_bin",
                "class_counts_by_bin",
                "test_node_counts_by_bin",
            ):
                output[field] = serialise_sequence(output[field])
            writer.writerow(output)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "dataset",
        "dataset_order",
        "num_total_classes",
        "num_splits",
        "n_bins",
        "min_classes_present_every_bin",
        "max_classes_present_every_bin",
        "counts_by_split",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["counts_by_split"] = serialise_sequence(
                output["counts_by_split"]
            )
            writer.writerow(output)


def build_manifest(
    repo_root: Path,
    script_path: Path,
    inputs: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_paths: Iterable[Path],
) -> dict[str, Any]:
    summary_records: list[dict[str, Any]] = []
    for row in summary_rows:
        summary_records.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"dataset_order"}
            }
        )
    return {
        "schema_version": 1,
        "analysis": "test-class-support-across-equal-width-hv-bins",
        "datasets": list(DATASETS),
        "bin_counts": list(BIN_COUNTS),
        "definitions": {
            "population": (
                "registered test nodes in each frozen dataset split"
            ),
            "bins": (
                "n equal-width intervals over [0,1], left-closed and "
                "right-open except that h_v=1 belongs to the final interval"
            ),
            "class_presence": (
                "a true class is present in a bin when the bin contains at "
                "least one test node of that class"
            ),
            "common_class_set": (
                "intersection of the true classes present in every h_v bin "
                "within the same test split"
            ),
            "split_range": (
                "minimum and maximum common-class count over the registered "
                "test splits of a dataset"
            ),
        },
        "validation": {
            "frozen_array_checksums_verified": True,
            "registered_test_index_checksums_verified": True,
            "registered_test_sizes_verified": True,
            "test_indices_unique_and_in_range": True,
            "test_labels_in_registered_class_range": True,
            "test_hv_finite_and_in_unit_interval": True,
            "split_rows": len(split_rows),
        },
        "inputs": inputs,
        "script": {
            "path": script_path.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(script_path),
        },
        "outputs": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": file_sha256(path),
            }
            for path in output_paths
        ],
        "summary": summary_records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parents[1]
    default_output_dir = (
        default_repo_root / "results" / "main" / "rq1" / "hv_bin_class_support"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help="Root of the homophily-failure-decomp repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for the generated CSV and analysis manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows, inputs = build_support_rows(repo_root)
    summary_rows = build_summary_rows(split_rows)
    split_path = output_dir / "split_class_support.csv"
    summary_path = output_dir / "dataset_class_support_summary.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    write_split_csv(split_path, split_rows)
    write_summary_csv(summary_path, summary_rows)

    script_path = Path(__file__).resolve()
    manifest = build_manifest(
        repo_root,
        script_path,
        inputs,
        split_rows,
        summary_rows,
        (split_path, summary_path),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(split_rows)} split-bin rows to {split_path}")
    print(f"Wrote {len(summary_rows)} dataset-bin rows to {summary_path}")
    print(f"Wrote analysis manifest to {manifest_path}")


if __name__ == "__main__":
    main()
