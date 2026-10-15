"""Small, deterministic tests for reverse-direction graph construction."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from hfd.edge_perturbation import (
    PairGenerationError,
    edge_index_sha256,
    generate_rewire_pair,
)
from hfd.homophily import to_simple_undirected


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_reverse_edge_graphs.py"
API = runpy.run_path(str(SCRIPT))


def _fixture(desired_same: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Target 0 has degree two and h=1/2. Node 3 is also in the original
    # tail, but is saturated in the requested direction and not selected.
    # It can still be a donor endpoint, so its collateral change is visible.
    labels = np.asarray(
        [0, 1, 0, 0, 1, 1, 0] if desired_same else [0, 0, 1, 0, 0, 0, 1],
        dtype=np.int64,
    )
    edges = np.asarray(
        [(0, 1), (2, 3), (4, 5), (1, 2), (1, 4), (0, 6)],
        dtype=np.int64,
    ).T
    tail = np.zeros(labels.size, dtype=bool)
    tail[[0, 3]] = True
    return to_simple_undirected(edges, labels.size), labels, tail


def _pair_and_audit(desired_same: bool):
    edges, labels, tail = _fixture(desired_same)
    degree, same = API["degree_and_same_counts"](edges, labels)
    eligible = API["eligible_tail_mask"](tail, degree, same, desired_same)
    pair = generate_rewire_pair(
        edges,
        labels,
        tail,
        desired_same=desired_same,
        target_nodes=np.asarray([0], dtype=np.int64),
        pair_seed=11,
        max_attempts=1,
        donor_random_tries=0,
    )
    args = (
        edges,
        labels,
        tail,
        eligible,
        pair.targets,
        pair.treatment_edge_index,
        pair.control_edge_index,
        pair.swap_records(),
        desired_same,
    )
    return pair, args, API["audit_pair"](*args)


def test_reverse_directions_and_original_seed_bases_are_fixed() -> None:
    expected = {
        "pubmed": (False, 2026080100),
        "ogbn-arxiv": (False, 2026080300),
        "arxiv-year": (True, 2026080200),
        "roman-empire": (True, 2026080400),
        "amazon-ratings": (True, 2026080500),
    }
    assert set(API["SPECS"]) == set(expected)
    assert API["REPLICATES"] == (1, 2, 3, 4, 5)
    for dataset, (desired_same, seed_base) in expected.items():
        spec = API["SPECS"][dataset]
        assert spec["desired_same"] is desired_same
        assert spec["seed_base"] == seed_base


def test_eligibility_excludes_saturated_isolated_and_nontail_nodes() -> None:
    tail = np.asarray([True, True, True, True, True, False])
    degree = np.asarray([2, 2, 0, 2, 1, 4])
    same = np.asarray([0, 1, 0, 2, 1, 3])
    assert API["eligible_tail_mask"](tail, degree, same, False).tolist() == [
        False, True, False, True, True, False,
    ]
    assert API["eligible_tail_mask"](tail, degree, same, True).tolist() == [
        True, True, False, False, False, False,
    ]


def test_doses_use_original_tail_denominator_and_fractional_endpoint() -> None:
    doses = API["dose_plan"](2696, 175)
    assert [dose["target_count"] for dose in doses] == [134, 175]
    assert doses[0]["nominal_pct"] == 5
    assert doses[0]["actual_pct"] == pytest.approx(100 * 134 / 2696)
    assert doses[-1]["actual_pct"] == pytest.approx(100 * 175 / 2696)
    assert doses[-1]["actual_pct"] < 6.5
    assert max(dose["actual_pct"] for dose in doses) < 100


def test_endpoint_below_five_percent_is_still_attempted() -> None:
    doses = API["dose_plan"](1000, 13)
    assert [dose["target_count"] for dose in doses] == [13]
    assert doses[0]["actual_pct"] == pytest.approx(1.3)


def test_doses_have_no_duplicate_counts_and_no_zero_target_generation() -> None:
    assert API["dose_plan"](1000, 0) == []
    assert [dose["target_count"] for dose in API["dose_plan"](1000, 100)] == [
        50, 100,
    ]
    doses = API["dose_plan"](9, 4)
    counts = [dose["target_count"] for dose in doses]
    assert counts == sorted(set(counts))
    assert counts[0] > 0
    assert counts[-1] == 4


def test_filtered_original_permutation_gives_fixed_nested_targets() -> None:
    tail = np.ones(40, dtype=bool)
    tail[[1, 9, 17]] = False
    eligible = tail & (np.arange(tail.size) % 3 != 0)
    seed = 2026080101
    original_ranking = np.random.default_rng(seed).permutation(np.flatnonzero(tail))
    expected = original_ranking[eligible[original_ranking]]
    ranking = API["eligible_ranking"](tail, eligible, seed)
    assert np.array_equal(ranking, expected)
    assert np.array_equal(ranking, API["eligible_ranking"](tail, eligible, seed))
    assert not np.array_equal(
        ranking, API["eligible_ranking"](tail, eligible, seed + 1)
    )
    previous: set[int] = set()
    for dose in API["dose_plan"](int(tail.sum()), int(eligible.sum())):
        selected = set(map(int, ranking[: dose["target_count"]]))
        assert previous <= selected
        assert len(selected) == dose["target_count"]
        assert all(eligible[node] for node in selected)
        previous = selected


@pytest.mark.parametrize("desired_same", [False, True])
def test_pair_has_exact_one_neighbour_treatment_and_unchanged_control(
    desired_same: bool,
) -> None:
    pair, args, audit = _pair_and_audit(desired_same)
    original, labels = args[:2]
    original_degree, original_same = API["degree_and_same_counts"](original, labels)
    treatment_degree, treatment_same = API["degree_and_same_counts"](
        pair.treatment_edge_index, labels
    )
    control_degree, control_same = API["degree_and_same_counts"](
        pair.control_edge_index, labels
    )
    assert np.array_equal(original_degree, treatment_degree)
    assert np.array_equal(original_degree, control_degree)
    assert original_degree[0] == 2
    assert treatment_same[0] - original_same[0] == (1 if desired_same else -1)
    assert control_same[0] == original_same[0]
    assert (treatment_same[0] - original_same[0]) / original_degree[0] == (
        0.5 if desired_same else -0.5
    )
    assert audit


@pytest.mark.parametrize("desired_same", [False, True])
def test_recorded_swaps_replay_exactly_and_generation_is_deterministic(
    desired_same: bool,
) -> None:
    first, args, first_audit = _pair_and_audit(desired_same)
    second, _, second_audit = _pair_and_audit(desired_same)
    assert np.array_equal(first.treatment_edge_index, second.treatment_edge_index)
    assert np.array_equal(first.control_edge_index, second.control_edge_index)
    assert first.swap_records() == second.swap_records()
    assert first_audit == second_audit
    original, labels = args[:2]
    for condition, expected in (
        ("treatment", first.treatment_edge_index),
        ("control", first.control_edge_index),
    ):
        replayed = API["replay_swaps"](
            original, first.swap_records()[condition], labels.size
        )
        assert edge_index_sha256(replayed) == edge_index_sha256(expected)


def test_audit_rejects_graph_tampering_even_if_degrees_are_preserved() -> None:
    _, args, _ = _pair_and_audit(True)
    tampered = list(args)
    # Replacing the control with the original graph still preserves degrees
    # and target h, but it does not match the recorded control swaps.
    tampered[6] = tampered[0].copy()
    with pytest.raises(API["AuditError"]):
        API["audit_pair"](*tampered)


def test_audit_rejects_swap_record_tampering() -> None:
    _, args, _ = _pair_and_audit(True)
    tampered = list(args)
    tampered[7] = deepcopy(tampered[7])
    tampered[7]["treatment"][0]["donor_near"] = 0
    with pytest.raises(API["AuditError"]):
        API["audit_pair"](*tampered)


def test_audit_rejects_target_identity_tampering() -> None:
    _, args, _ = _pair_and_audit(True)
    tampered = list(args)
    tampered[4] = np.asarray([3], dtype=np.int64)
    with pytest.raises(API["AuditError"]):
        API["audit_pair"](*tampered)


def test_only_exactly_five_distinct_replicates_can_be_accepted() -> None:
    API["validate_replicate_set"]([1, 2, 3, 4, 5])
    for replicates in ([], [1], [1, 2, 3, 4], [1, 2, 3, 4, 4], [1, 2, 3, 4, 6]):
        with pytest.raises(API["AuditError"]):
            API["validate_replicate_set"](replicates)


def test_stop_record_does_not_mislabel_search_limits_as_impossibility() -> None:
    dose = {"target_count": 50, "nominal_pct": 5.0, "actual_pct": 5.0, "kind": "regular"}
    for error, expected_reason in (
        (API["PairTimeBudgetExceeded"]("time limit reached"), "resource_timeout"),
        (
            PairGenerationError("could not generate a complete pair after 1000 attempts"),
            "construction_budget_exhausted",
        ),
    ):
        record = API["make_stop_record"]("pubmed", dose, 3, error)
        assert record["theoretical_maximum_proven"] is False
        assert record["reason"] == expected_reason
    with pytest.raises(API["AuditError"]):
        API["make_stop_record"]("pubmed", dose, 3, API["AuditError"]("bad checksum"))


@pytest.mark.parametrize("desired_same", [False, True])
def test_collateral_changes_to_unselected_saturated_tail_are_recorded(
    desired_same: bool,
) -> None:
    _, args, audit = _pair_and_audit(desired_same)
    tail, eligible = args[2:4]
    assert tail[3] and not eligible[3]
    collateral = audit["collateral_homophily_changes"]
    assert collateral["treatment"]["saturated_tail"]["nodes"] == 1
    assert collateral["treatment"]["saturated_tail"]["nodes_with_changed_homophily"] == 1
    assert collateral["treatment"]["unselected_tail"]["nodes_with_changed_homophily"] == 1
    assert collateral["control"]["saturated_tail"]["nodes_with_changed_homophily"] == 0


def _prepared_workflow(tmp_path: Path, monkeypatch) -> Namespace:
    """Make a tiny prepared dataset without loading any registered real data."""
    edges, labels, _ = _fixture(False)
    combined_labels = np.concatenate((labels, labels, np.zeros(18, dtype=np.int64)))
    combined_edges = to_simple_undirected(
        np.concatenate((edges, edges + labels.size), axis=1), combined_labels.size
    )
    tail = np.zeros(combined_labels.size, dtype=bool)
    tail[[0, labels.size]] = True
    tail[2 * labels.size :] = True
    degrees, same = API["degree_and_same_counts"](combined_edges, combined_labels)
    eligible = API["eligible_tail_mask"](tail, degrees, same, False)
    assert int(tail.sum()) == 20
    assert int(eligible.sum()) == 2
    arrays = dict(
        edge_index=combined_edges,
        labels=combined_labels,
        tail_mask=tail,
        eligible_mask=eligible,
        degrees=degrees,
        same_counts=same,
    )
    prepared = dict(
        dataset="pubmed",
        desired_same=False,
        seed_base=API["SPECS"]["pubmed"]["seed_base"],
        original_tail_count=20,
        eligible_tail_count=2,
        dose_plan=API["dose_plan"](20, 2),
        source_sha256={"synthetic_fixture": "test-only"},
        software_versions=API["software_versions"](),
    )
    prepared_path = tmp_path / "pubmed" / "prepared"
    prepared_path.mkdir(parents=True)
    (prepared_path / "manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
    globals_ = API["generate_dataset"].__globals__
    monkeypatch.setitem(globals_, "load_prepared", lambda *args: (arrays, prepared))
    monkeypatch.setitem(globals_, "source_hashes", lambda: prepared["source_sha256"])
    return Namespace(
        dataset="pubmed",
        output_root=tmp_path,
        max_attempts=20,
        pair_timeout_seconds=0.0,
    )


def test_end_to_end_generation_accepts_five_pairs_per_dose_and_resumes(
    tmp_path: Path, monkeypatch,
) -> None:
    args = _prepared_workflow(tmp_path, monkeypatch)
    report = API["generate_dataset"](args)
    assert report["validated"] is True
    assert report["pair_count"] == 10
    assert report["accepted_nonzero_dose_count"] == 2
    assert report["replicates_per_accepted_dose"] == 5
    assert report["all_planned_doses_constructed"] is True
    actual = tmp_path / "pubmed" / "actual"
    catalog = json.loads((actual / "catalog.json").read_text(encoding="utf-8"))
    assert [dose["target_count"] for dose in catalog["accepted_doses"]] == [1, 2]
    assert catalog["maximum_accepted_pct"] == 10.0
    assert API["verify_dataset"](args) == report
    hashes = {
        path.relative_to(actual).as_posix(): API["file_sha256"](path)
        for path in actual.rglob("*") if path.is_file()
    }
    assert API["generate_dataset"](args) == report
    assert hashes == {
        path.relative_to(actual).as_posix(): API["file_sha256"](path)
        for path in actual.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("tamper", ["remove", "corrupt"])
def test_end_to_end_verification_rejects_a_missing_or_corrupt_fifth_pair(
    tmp_path: Path, monkeypatch, tamper: str,
) -> None:
    args = _prepared_workflow(tmp_path, monkeypatch)
    API["generate_dataset"](args)
    path = (
        tmp_path / "pubmed" / "actual" / "doses" / "targets-000001"
        / "replicate-0005" / "graphs.npz"
    )
    if tamper == "remove":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(API["AuditError"]):
        API["verify_dataset"](args)


@pytest.mark.parametrize(
    "error_type,reason",
    [
        (PairGenerationError, "construction_budget_exhausted"),
        (API["PairTimeBudgetExceeded"], "resource_timeout"),
    ],
)
def test_failed_next_dose_keeps_earlier_five_pairs_and_excludes_partial_dose(
    tmp_path: Path, monkeypatch, error_type, reason: str,
) -> None:
    args = _prepared_workflow(tmp_path, monkeypatch)
    globals_ = API["generate_dataset"].__globals__
    real_generate = globals_["_generate_pair"]

    def fail_on_second_replicate(path, arrays, prepared, experiment, dose, replicate):
        if dose["target_count"] == 2 and replicate == 2:
            raise error_type("synthetic bounded construction failure")
        return real_generate(path, arrays, prepared, experiment, dose, replicate)

    monkeypatch.setitem(globals_, "_generate_pair", fail_on_second_replicate)
    report = API["generate_dataset"](args)
    assert report["validated"] is True
    assert report["pair_count"] == 5
    assert report["accepted_nonzero_dose_count"] == 1
    assert report["all_planned_doses_constructed"] is False
    assert report["maximum_accepted_pct"] == 5.0
    assert report["construction_limit_reason"] == reason
    actual = tmp_path / "pubmed" / "actual"
    assert not (actual / "doses" / "targets-000002").exists()
    assert (
        actual / "doses" / "targets-000002.in-progress" / "replicate-0001"
    ).is_dir()
    catalog = json.loads((actual / "catalog.json").read_text(encoding="utf-8"))
    assert [dose["target_count"] for dose in catalog["accepted_doses"]] == [1]
    assert len(catalog["pairs"]) == 5
    assert all(pair["dose"]["target_count"] == 1 for pair in catalog["pairs"])
    assert catalog["construction_limit"]["theoretical_maximum_proven"] is False
    assert API["generate_dataset"](args) == report
