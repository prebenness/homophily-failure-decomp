import csv
from pathlib import Path
import runpy

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_hv_bin_class_support.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


EXPECTED_SPLIT_COUNTS = {
    "pubmed": {
        4: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        6: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        8: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        10: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        20: [1, 2, 0, 3, 1, 3, 2, 1, 1, 2],
    },
    "ogbn-arxiv": {
        4: [38],
        6: [36],
        8: [34],
        10: [33],
        20: [20],
    },
    "roman-empire": {
        4: [5, 4, 4, 5, 4, 3, 4, 4, 6, 4],
        6: [3, 3, 3, 2, 4, 2, 3, 2, 3, 3],
        8: [2, 1, 2, 2, 1, 2, 2, 1, 1, 2],
        10: [0, 0, 2, 0, 0, 1, 1, 0, 0, 0],
        20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "amazon-ratings": {
        4: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        6: [5, 5, 5, 5, 5, 5, 4, 5, 5, 5],
        8: [4, 3, 4, 4, 4, 3, 4, 4, 3, 4],
        10: [4, 3, 3, 4, 4, 3, 4, 3, 3, 4],
        20: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    },
    "arxiv-year": {
        4: [5, 5, 5, 5, 5],
        6: [5, 5, 5, 5, 5],
        8: [5, 5, 5, 5, 5],
        10: [5, 5, 5, 5, 5],
        20: [3, 3, 4, 2, 4],
    },
}


def assert_expected_split_counts(rows: list[dict[str, object]]) -> None:
    for dataset, expected_by_bin_count in EXPECTED_SPLIT_COUNTS.items():
        for n_bins, expected in expected_by_bin_count.items():
            observed_rows = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["n_bins"]) == n_bins
                ),
                key=lambda row: int(row["split_order"]),
            )
            observed = [
                int(row["classes_present_every_bin"])
                for row in observed_rows
            ]
            assert observed == expected


def test_equal_width_bins_are_left_closed_and_include_one() -> None:
    values = np.array([0.0, 0.249, 0.25, 0.5, 0.999, 1.0, np.nan])

    actual = NAMESPACE["assign_equal_width_bins"](values, 4)

    np.testing.assert_array_equal(actual, [0, 0, 1, 2, 3, 3, -1])


def test_common_class_requires_presence_in_every_bin() -> None:
    h = np.array([0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 1.0, 0.2])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 2, 2])
    test_idx = np.arange(h.size)

    actual = NAMESPACE["calculate_split_support"](h, labels, test_idx, 4)

    assert actual["class_ids_present_every_bin"] == [0]
    assert actual["classes_present_every_bin"] == 1
    assert actual["class_counts_by_bin"] == [3, 2, 2, 2]


def test_tracked_split_counts_match_canonical_analysis() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "main"
        / "rq1"
        / "hv_bin_class_support"
        / "split_class_support.csv"
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert_expected_split_counts(rows)


@pytest.mark.skipif(
    not all(
        (
            REPO_ROOT / "artifacts" / "manifests" / f"{dataset}.npz"
        ).is_file()
        for dataset in EXPECTED_SPLIT_COUNTS
    ),
    reason=(
        "requires verified frozen arrays; run "
        "scripts/compute_dataset_stats.py --verify"
    ),
)
def test_canonical_split_counts_match_frozen_manifests() -> None:
    rows, _ = NAMESPACE["build_support_rows"](REPO_ROOT)
    assert_expected_split_counts(rows)
