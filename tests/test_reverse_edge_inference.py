"""Small tests for authenticated reverse-graph checkpoint inference."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_reverse_edge_inference.py"
API = runpy.run_path(str(SCRIPT))
GRAPH_API = runpy.run_path(str(SCRIPT.with_name("run_reverse_edge_graphs.py")))

TAIL_COUNTS = {
    "pubmed": 2696,
    "ogbn-arxiv": 30364,
    "arxiv-year": 21869,
    "roman-empire": 1012,
    "amazon-ratings": 2943,
}
ELIGIBLE_COUNTS = {
    "pubmed": 175,
    "ogbn-arxiv": 9565,
    "arxiv-year": 4670,
    "roman-empire": 943,
    "amazon-ratings": 2319,
}


def _dose_targets(dataset: str) -> list[int]:
    tail_count = TAIL_COUNTS[dataset]
    eligible_count = ELIGIBLE_COUNTS[dataset]
    counts = {
        tail_count * percentage // 100
        for percentage in range(5, 101, 5)
        if 0 < tail_count * percentage // 100 <= eligible_count
    }
    return sorted(counts | {eligible_count})


def _count_catalogs() -> dict:
    return {
        dataset: {
            "dataset": dataset,
            "pair_count": len(_dose_targets(dataset)) * 5,
            "pairs": [
                {"replicate_index": replicate, "target_count": targets}
                for targets in _dose_targets(dataset)
                for replicate in range(1, 6)
            ],
        }
        for dataset in TAIL_COUNTS
    }


def test_every_preparation_evaluates_all_thirty_model_dataset_combinations() -> None:
    assert API["PREPARATIONS"] == ("erm", "last", "head", "full")
    assert set(API["DATASET_SPECS"]) == set(TAIL_COUNTS)
    assert set(API["FULL_MODELS"]) == {
        "gcn", "graphsage-mean", "acm-gcn", "gpr-gnn", "h2gcn", "linkx",
    }
    combinations = []
    for dataset, spec in API["DATASET_SPECS"].items():
        assert set(spec.models) == set(API["FULL_MODELS"])
        assert len(spec.checkpoint_identities) == 30
        combinations.extend((dataset, model) for model in spec.models)
    assert len(combinations) == len(set(combinations)) == 30
    assert API["expected_checkpoint_count"]("erm") == 900
    assert sum(API["expected_checkpoint_count"](p) for p in API["PREPARATIONS"]) == 3600


def test_expected_rows_use_all_accepted_pairs_and_original_tail_endpoints() -> None:
    catalogs = _count_catalogs()
    assert {name: item["pair_count"] for name, item in catalogs.items()} == {
        "pubmed": 10,
        "ogbn-arxiv": 35,
        "arxiv-year": 25,
        "roman-empire": 95,
        "amazon-ratings": 80,
    }
    assert sum(item["pair_count"] for item in catalogs.values()) == 245
    for preparation in API["PREPARATIONS"]:
        assert API["expected_paired_row_count"](catalogs, preparation) == 44100
    assert sum(
        API["expected_paired_row_count"](catalogs, preparation)
        for preparation in API["PREPARATIONS"]
    ) == 176400


def test_implementation_hashes_are_stable_across_equivalent_checkout_paths(
    monkeypatch,
) -> None:
    function = API["implementation_hashes"]
    globals_ = function.__globals__
    root = globals_["REPO_ROOT"].resolve()
    expected = function()
    monkeypatch.setitem(globals_, "REPO_ROOT", root / "scripts" / "..")
    monkeypatch.setattr(
        globals_["base"], "__file__",
        str(root / "scripts" / ".." / "scripts" / "run_edge_dose_response_expansion.py"),
    )
    monkeypatch.setattr(
        globals_["graphs"], "__file__",
        str(root / "scripts" / ".." / "scripts" / "run_reverse_edge_graphs.py"),
    )
    assert function() == expected


def test_source_hash_manifests_accept_only_portable_paths_and_complete_hashes() -> None:
    first = {"scripts/runner.py": "1" * 64}
    second = {"scripts/runner.py": "2" * 64}
    assert API["_validate_implementation_map"](first) == first
    assert API["_implementation_is_accepted"](second, [first, second])
    assert not API["_implementation_is_accepted"](
        {"scripts/runner.py": "3" * 64}, [first, second]
    )
    for invalid in (
        {"../runner.py": "1" * 64},
        {"C:/private/runner.py": "1" * 64},
        {"private\\runner.py": "1" * 64},
        {"scripts/runner.py": "not-a-sha256"},
    ):
        with pytest.raises(RuntimeError):
            API["_validate_implementation_map"](invalid)


def test_producer_implementation_maps_are_unique_and_canonically_ordered() -> None:
    first = {"scripts/runner.py": "1" * 64}
    second = {"scripts/runner.py": "2" * 64}
    assert API["_canonical_implementation_maps"]([second, first]) == [first, second]
    with pytest.raises(RuntimeError):
        API["_canonical_implementation_maps"]([first, first])


def test_design_identity_is_reverse_and_distinct_for_every_preparation() -> None:
    designs = [API["design_id_for_preparation"](p) for p in API["PREPARATIONS"]]
    assert len(set(designs)) == 4
    assert all("reverse" in design for design in designs)
    assert "appendix-j-edge-dose-response-expansion-v1" not in designs


def test_pair_identity_uses_exact_target_count_not_a_rounded_percentage() -> None:
    first = API["pairing_key"]("pubmed", 1, 134)
    endpoint = API["pairing_key"]("pubmed", 1, 175)
    next_target = API["pairing_key"]("pubmed", 1, 176)
    other_replicate = API["pairing_key"]("pubmed", 2, 175)
    other_dataset = API["pairing_key"]("roman-empire", 1, 175)
    assert len({first, endpoint, next_target, other_replicate, other_dataset}) == 5
    assert endpoint == API["pairing_key"]("pubmed", 1, 175)


def _raw_catalog(dataset: str = "pubmed") -> dict:
    spec = GRAPH_API["SPECS"][dataset]
    tail_count = TAIL_COUNTS[dataset]
    eligible_count = ELIGIBLE_COUNTS[dataset]
    doses = GRAPH_API["dose_plan"](tail_count, eligible_count)
    return {
        "schema_version": 1,
        "design_id": GRAPH_API["DESIGN_ID"],
        "dataset": dataset,
        "original_tail_count": tail_count,
        "eligible_tail_count": eligible_count,
        "denominator": "original frozen tail-node count",
        "desired_same": spec["desired_same"],
        "accepted_doses": doses,
        "all_planned_doses_constructed": True,
        "construction_limit": None,
        "pairs": [
            {
                "dose": deepcopy(dose),
                "replicate_index": replicate,
                "replicate_seed": spec["seed_base"] + replicate,
                "relative_directory": (
                    f"doses/targets-{dose['target_count']:06d}/replicate-{replicate:04d}"
                ),
            }
            for dose in doses
            for replicate in range(1, 6)
        ],
        "pair_count": len(doses) * 5,
        "baseline": {
            "target_count": 0,
            "actual_pct": 0.0,
            "original_tail_count": tail_count,
            "replicate_indices": [1, 2, 3, 4, 5],
            "treatment_equals_control_equals_original": True,
        },
    }


@pytest.mark.parametrize("dataset", TAIL_COUNTS)
def test_authenticated_catalog_structure_retains_fractional_actual_percentages(
    dataset: str,
) -> None:
    catalog = _raw_catalog(dataset)
    API["validate_catalog_structure"](catalog, dataset)
    endpoint = catalog["accepted_doses"][-1]
    assert endpoint["target_count"] == ELIGIBLE_COUNTS[dataset]
    assert endpoint["actual_pct"] == pytest.approx(
        100 * ELIGIBLE_COUNTS[dataset] / TAIL_COUNTS[dataset]
    )
    assert endpoint["actual_pct"] != int(endpoint["actual_pct"])
    for dose in catalog["accepted_doses"]:
        rows = [pair for pair in catalog["pairs"] if pair["dose"] == dose]
        assert sorted(row["replicate_index"] for row in rows) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "field,value",
    [
        ("design_id", "appendix-j-edge-dose-response-expansion-v1"),
        ("dataset", "roman-empire"),
        ("desired_same", True),
        ("original_tail_count", 175),
        ("denominator", "eligible tail-node count"),
        ("pair_count", 9),
    ],
)
def test_catalog_rejects_wrong_direction_identity_or_denominator(field: str, value) -> None:
    catalog = _raw_catalog()
    catalog[field] = value
    with pytest.raises((RuntimeError, ValueError)):
        API["validate_catalog_structure"](catalog, "pubmed")


def test_catalog_rejects_four_replicates_even_if_pair_count_is_adjusted() -> None:
    catalog = _raw_catalog()
    catalog["pairs"] = catalog["pairs"][:-1]
    catalog["pair_count"] -= 1
    with pytest.raises((RuntimeError, ValueError)):
        API["validate_catalog_structure"](catalog, "pubmed")


def test_catalog_rejects_duplicate_replicates_wrong_seeds_or_changed_counts() -> None:
    for field, value in (("replicate_index", 1), ("replicate_seed", 123)):
        catalog = _raw_catalog()
        catalog["pairs"][1][field] = value
        with pytest.raises((RuntimeError, ValueError)):
            API["validate_catalog_structure"](catalog, "pubmed")
    catalog = _raw_catalog()
    catalog["pairs"][-1]["dose"]["target_count"] = 174
    with pytest.raises((RuntimeError, ValueError)):
        API["validate_catalog_structure"](catalog, "pubmed")


def test_catalog_rejects_rounded_or_eligible_normalized_x_coordinates() -> None:
    for wrong_percentage in (6.0, 100.0):
        catalog = _raw_catalog()
        catalog["accepted_doses"][-1]["actual_pct"] = wrong_percentage
        for pair in catalog["pairs"]:
            if pair["dose"]["target_count"] == 175:
                pair["dose"]["actual_pct"] = wrong_percentage
        with pytest.raises((RuntimeError, ValueError)):
            API["validate_catalog_structure"](catalog, "pubmed")


def test_catalog_rejects_pair_path_outside_its_graph_directory() -> None:
    catalog = _raw_catalog()
    catalog["pairs"][0]["relative_directory"] = "../forward-direction/pair"
    with pytest.raises((RuntimeError, ValueError)):
        API["validate_catalog_structure"](catalog, "pubmed")


def _baseline_fixture(tmp_path: Path, preparation: str = "erm"):
    saved = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    path = tmp_path / "run_data.npz"
    np.savez(path, logits=saved)
    checkpoint = SimpleNamespace(
        dataset="pubmed",
        model="gcn",
        run_id=f"pubmed_gcn_{preparation}_geomgcn-0_seed-1",
        split="geomgcn-0",
        seed=1,
        saved_predictions=saved.argmax(axis=-1),
        run_data_path=path,
        checkpoint_sha256="a" * 64,
        training_code_version="training-code",
        preparation=preparation,
    )
    bundle = SimpleNamespace(num_nodes=2, y=np.asarray([0, 1], dtype=np.int64))
    arrays = {
        "geomgcn-0.test": np.asarray([0, 1], dtype=np.int64),
        "tail_mask": np.asarray([True, True]),
    }
    return checkpoint, bundle, arrays, saved


@pytest.mark.parametrize("preparation", ("erm", "last", "head", "full"))
def test_baseline_requires_fixed_tail_prediction_and_metric_equivalence(
    tmp_path: Path, preparation: str,
) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path, preparation)
    row = API["baseline_row_from_logits"](
        checkpoint, bundle=bundle, arrays=arrays, current_logits=saved.copy()
    )
    assert row["preparation"] == preparation
    assert row["n_tail_test_prediction_differences_from_parent"] == 0
    assert row["baseline_validation_mode"] == API["BASELINE_VALIDATION_MODE"]
    assert row["saved_tail_prediction_sha256"] == row["current_tail_prediction_sha256"]
    assert row["saved_tail_confusion_counts"] == row["current_tail_confusion_counts"]
    assert row["saved_tail_test_macro_f1"] == row["tail_test_macro_f1"]
    assert row["saved_tail_test_accuracy"] == row["tail_test_accuracy"]
    assert row["raw_logit_differences_recorded"] is True
    assert row["raw_logit_comparison_is_acceptance_criterion"] is False


def test_baseline_records_changed_logits_when_fixed_tail_predictions_match(
    tmp_path: Path,
) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path)
    changed = saved + 0.5
    assert np.array_equal(changed.argmax(axis=-1), saved.argmax(axis=-1))
    row = API["baseline_row_from_logits"](
        checkpoint, bundle=bundle, arrays=arrays, current_logits=changed
    )
    assert row["n_tail_test_prediction_differences_from_parent"] == 0
    assert row["max_abs_logit_difference_from_parent"] == pytest.approx(0.5)
    assert row["mean_abs_logit_difference_from_parent"] == pytest.approx(0.5)


def test_validator_accepts_the_equivalent_record_only_policy_identity(
    tmp_path: Path,
) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path)
    row = API["baseline_row_from_logits"](
        checkpoint, bundle=bundle, arrays=arrays, current_logits=saved.copy()
    )
    row["baseline_validation_mode"] = (
        "fixed-tail-prediction-and-metrics-logits-record-only-v1"
    )
    API["_validate_baseline"](
        row,
        checkpoint.run_id,
        checkpoint.checkpoint_sha256,
        checkpoint.preparation,
    )


def test_baseline_records_but_does_not_reject_a_non_tail_prediction_change(
    tmp_path: Path,
) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path)
    saved = np.vstack((saved, np.asarray([[2.0, 0.0]], dtype=np.float32)))
    np.savez(checkpoint.run_data_path, logits=saved)
    checkpoint.saved_predictions = saved.argmax(axis=-1)
    bundle.num_nodes = 3
    bundle.y = np.asarray([0, 1, 0], dtype=np.int64)
    arrays["geomgcn-0.test"] = np.asarray([0, 1, 2], dtype=np.int64)
    arrays["tail_mask"] = np.asarray([True, True, False])
    changed = saved.copy()
    changed[2] = [0.0, 2.0]
    row = API["baseline_row_from_logits"](
        checkpoint, bundle=bundle, arrays=arrays, current_logits=changed
    )
    assert row["n_tail_test_prediction_differences_from_parent"] == 0
    assert row["n_full_graph_prediction_differences_from_parent"] == 1
    assert row["full_graph_prediction_difference_nodes_from_parent"] == [2]


def test_baseline_never_accepts_changed_original_tail_predictions(
    tmp_path: Path,
) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path)
    changed = saved[:, ::-1].copy()
    with pytest.raises(RuntimeError):
        API["baseline_row_from_logits"](
            checkpoint,
            bundle=bundle,
            arrays=arrays,
            current_logits=changed,
            logit_atol=100.0,
            logit_rtol=100.0,
        )


def test_baseline_rejects_nonfinite_logits(tmp_path: Path) -> None:
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path)
    changed = saved.copy()
    changed[0, 1] = np.nan
    with pytest.raises(RuntimeError):
        API["baseline_row_from_logits"](
            checkpoint, bundle=bundle, arrays=arrays, current_logits=changed
        )


def _normalized_catalog() -> dict:
    raw = _raw_catalog()
    entries = []
    for entry in raw["pairs"]:
        count = entry["dose"]["target_count"]
        percentage = entry["dose"]["actual_pct"]
        entries.append({
            **entry,
            "target_count": count,
            "selected_targets": count,
            "p_percent": percentage,
            "actual_pct": percentage,
            "pairing_key": API["pairing_key"]("pubmed", entry["replicate_index"], count),
            "graphs": {"sha256": "b" * 64, "bytes": 10},
            "treatment_edge_sha256": "a" * 64,
            "control_edge_sha256": "c" * 64,
        })
    return {
        **raw,
        "pairs": entries,
        "catalog_sha256": "c" * 64,
        "source_experiment_manifest_sha256": "e" * 64,
    }


def _shard_fixture(tmp_path: Path, preparation: str = "erm"):
    checkpoint, bundle, arrays, saved = _baseline_fixture(tmp_path, preparation)
    checkpoint.checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint.checkpoint_path.write_bytes(b"synthetic checkpoint")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    checkpoint.checkpoint_sha256 = API["file_sha256"](checkpoint.checkpoint_path)
    catalog = _normalized_catalog()
    args = SimpleNamespace(
        phase="actual", preparation=preparation,
        backend_id="v100-cuda",
        output_root=tmp_path / "outputs",
        experiment_revision="1" * 40,
        implementation_sha256={"runner.py": "1" * 64},
        logit_reproduction_atol=1e-5,
        logit_reproduction_rtol=1e-5,
        input_snapshot_sha256="f" * 64,
    )
    inputs = API["checkpoint_file_hashes"](checkpoint)
    baseline = API["baseline_row_from_logits"](
        checkpoint, bundle=bundle, arrays=arrays, current_logits=saved.copy()
    )
    rows = [
        API["paired_metric_row"](
            checkpoint, bundle=bundle, arrays=arrays,
            graph_entry=entry, catalog=catalog,
            treatment_predictions=np.asarray([0, 1], dtype=np.int64),
            control_predictions=np.asarray([0, 0], dtype=np.int64),
        )
        for entry in catalog["pairs"]
    ]
    payload = {
        **API["_header"](args, catalog, checkpoint, inputs),
        "completed": True,
        "baseline": baseline,
        "pairs": rows,
        "n_baseline_rows": 1,
        "n_paired_rows": len(rows),
    }
    return args, catalog, checkpoint, inputs, payload


def _validate_payload(args, catalog, checkpoint, payload) -> None:
    API["validate_shard_payload"](
        payload, catalog=catalog, checkpoint=checkpoint,
        preparation=args.preparation, phase=args.phase,
        implementation_sha256=args.implementation_sha256,
        logit_atol=args.logit_reproduction_atol,
        logit_rtol=args.logit_reproduction_rtol,
        backend_id=args.backend_id,
    )


@pytest.mark.parametrize("preparation", ("erm", "last", "head", "full"))
def test_actual_shard_preserves_all_five_pairs_and_fractional_endpoint(
    tmp_path: Path, preparation: str,
) -> None:
    args, catalog, checkpoint, _, payload = _shard_fixture(tmp_path, preparation)
    _validate_payload(args, catalog, checkpoint, payload)
    assert len(payload["pairs"]) == 10
    endpoints = [row for row in payload["pairs"] if row["target_count"] == 175]
    assert len(endpoints) == 5
    assert all(row["p_percent"] == 100 * 175 / 2696 for row in endpoints)
    assert all(row["original_tail_count"] == 2696 for row in endpoints)
    assert all(row["preparation"] == preparation for row in payload["pairs"])


@pytest.mark.parametrize(
    "location,field,value",
    [
        ("header", "design_id", "appendix-j-edge-dose-response-expansion-v1"),
        ("header", "preparation", "head"),
        ("header", "graph_catalog_sha256", "z" * 64),
        ("header", "source_graph_manifest_sha256", "z" * 64),
        ("header", "backend_id", "a40-cuda"),
        ("baseline", "preparation", "full"),
        ("baseline", "full_graph_logit_reproduction_checked", True),
        ("baseline", "n_tail_test_prediction_differences_from_parent", 1),
        ("pair", "preparation", "last"),
        ("pair", "target_count", 174),
        ("pair", "n_selected_tail_nodes", 174),
        ("pair", "p_percent", 6),
        ("pair", "actual_pct", 100.0),
        ("pair", "original_tail_count", 175),
        ("pair", "desired_same", True),
        ("pair", "replicate_seed", 12),
        ("pair", "source_graphs_sha256", "z" * 64),
        ("pair", "treatment_minus_control_macro_f1", 0.123),
    ],
)
def test_shards_reject_forward_results_wrong_preparations_and_tampered_identity(
    tmp_path: Path, location: str, field: str, value,
) -> None:
    args, catalog, checkpoint, _, payload = _shard_fixture(tmp_path)
    target = payload if location == "header" else (
        payload["baseline"] if location == "baseline" else payload["pairs"][-1]
    )
    target[field] = value
    with pytest.raises(RuntimeError):
        _validate_payload(args, catalog, checkpoint, payload)


def test_actual_shard_rejects_a_missing_fifth_pair_and_colliding_control_key(
    tmp_path: Path,
) -> None:
    args, catalog, checkpoint, _, payload = _shard_fixture(tmp_path)
    shortened = deepcopy(payload)
    shortened["pairs"].pop()
    shortened["n_paired_rows"] -= 1
    with pytest.raises(RuntimeError):
        _validate_payload(args, catalog, checkpoint, shortened)
    payload["pairs"][-1]["control_prediction_key"] = (
        payload["pairs"][-1]["treatment_prediction_key"]
    )
    with pytest.raises(RuntimeError):
        _validate_payload(args, catalog, checkpoint, payload)


def test_resume_rechecks_checkpoint_bytes_and_implementation_identity(tmp_path: Path) -> None:
    args, catalog, checkpoint, _, payload = _shard_fixture(tmp_path)
    _validate_payload(args, catalog, checkpoint, payload)
    changed_code = deepcopy(payload)
    changed_code["implementation_sha256"] = {"runner.py": "2" * 64}
    with pytest.raises(RuntimeError):
        _validate_payload(args, catalog, checkpoint, changed_code)
    checkpoint.checkpoint_path.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError):
        _validate_payload(args, catalog, checkpoint, payload)


def _evaluation_cli_args() -> list[str]:
    return [
        "evaluate", "--graph-root", "graphs", "--output-root", "outputs",
        "--experiment-revision", "1" * 40, "--data-root", "data",
        "--array-root", "arrays", "--manifest-root", "manifests",
        "--checkpoint-root", "checkpoints", "--preparation", "erm",
        "--dataset", "pubmed", "--model", "gcn", "--phase", "smoke",
    ]


def test_evaluation_requires_an_explicit_portable_backend_identity() -> None:
    with pytest.raises(SystemExit):
        API["parse_args"](_evaluation_cli_args())
    args = API["parse_args"](_evaluation_cli_args() + ["--backend-id", "v100-cuda"])
    assert args.backend_id == "v100-cuda"
    for invalid in ("", "../backend", "A40 CUDA", "/absolute/path"):
        with pytest.raises(SystemExit):
            API["parse_args"](_evaluation_cli_args() + ["--backend-id", invalid])


@pytest.mark.parametrize("evaluator_name", ["_evaluate_checkpoint_major", "_evaluate_h2"])
def test_every_invocation_recomputes_original_baselines_instead_of_loading_a_cache(
    tmp_path: Path, monkeypatch, evaluator_name: str,
) -> None:
    args, catalog, checkpoint, inputs, _ = _shard_fixture(tmp_path)
    args.phase = "smoke"
    args.baseline_only = True
    args.dataset = checkpoint.dataset
    args.model = "h2gcn" if evaluator_name == "_evaluate_h2" else checkpoint.model
    bundle = SimpleNamespace(num_nodes=2, y=np.asarray([0, 1], dtype=np.int64))
    arrays = {
        "geomgcn-0.test": np.asarray([0, 1], dtype=np.int64),
        "tail_mask": np.asarray([True, True]),
    }
    original = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    saved = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    calls = {"forwards": 0, "saved": 0}
    evaluator = API[evaluator_name]
    globals_ = evaluator.__globals__

    def forbidden_cache(*_args, **_kwargs):
        pytest.fail("an original-graph baseline was reused across invocations")

    def save_baseline(*_args, **_kwargs):
        calls["saved"] += 1

    def predict_single(*_args, **_kwargs):
        calls["forwards"] += 1
        return saved.copy()

    def predict_h2(networks, *_args, **_kwargs):
        calls["forwards"] += 1
        return {run_id: saved.copy() for run_id in networks}

    monkeypatch.setitem(globals_, "_network", lambda *_args: object())
    monkeypatch.setitem(globals_, "_load_baseline", forbidden_cache)
    monkeypatch.setitem(globals_, "_save_baseline", save_baseline)
    monkeypatch.setitem(globals_, "_single_predictions", predict_single)
    monkeypatch.setattr(globals_["base"], "h2_predictions_for_graph", predict_h2)
    for _ in range(2):
        evaluator(
            args, catalog, bundle, arrays, None, original,
            SimpleNamespace(type="cpu"), [(checkpoint, inputs)],
        )
    assert calls == {"forwards": 2, "saved": 2}


def test_h2_progress_from_another_backend_is_rejected(tmp_path: Path, monkeypatch) -> None:
    args, catalog, checkpoint, inputs, _ = _shard_fixture(tmp_path)
    args.dataset = "pubmed"
    args.model = "h2gcn"
    args.baseline_only = False
    checkpoint.model = "h2gcn"
    checkpoint.run_id = "pubmed_h2gcn_erm_geomgcn-0_seed-1"
    catalog = {**catalog, "pairs": catalog["pairs"][:1]}
    entry = catalog["pairs"][0]
    progress = {
        "design_id": API["design_id_for_preparation"]("erm"),
        "phase": args.phase,
        "graph_catalog_sha256": catalog["catalog_sha256"],
        "pairing_key": entry["pairing_key"],
        "backend_id": "a40-cuda",
        "input_snapshot_sha256": args.input_snapshot_sha256,
        "implementation_sha256": args.implementation_sha256,
        "input_file_sha256": {checkpoint.run_id: inputs},
        "baseline_validation_mode": API["BASELINE_VALIDATION_MODE"],
        "rows": [],
    }
    path = (
        API["result_root"](args) / "pubmed" / "h2gcn" / "progress"
        / f"{entry['pairing_key']}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(progress), encoding="utf-8")
    bundle = SimpleNamespace(num_nodes=2, y=np.asarray([0, 1], dtype=np.int64))
    arrays = {
        "geomgcn-0.test": np.asarray([0, 1], dtype=np.int64),
        "tail_mask": np.asarray([True, True]),
    }
    saved = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    globals_ = API["_evaluate_h2"].__globals__
    monkeypatch.setitem(globals_, "_network", lambda *_args: object())
    monkeypatch.setitem(globals_, "_save_baseline", lambda *_args: None)
    monkeypatch.setattr(
        globals_["base"], "h2_predictions_for_graph",
        lambda networks, *_args, **_kwargs: {run_id: saved.copy() for run_id in networks},
    )
    with pytest.raises(RuntimeError, match="H2GCN saved progress"):
        API["_evaluate_h2"](
            args, catalog, bundle, arrays, None,
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            SimpleNamespace(type="cpu"), [(checkpoint, inputs)],
        )


def _catalog_files(tmp_path: Path) -> Path:
    """Write tiny files to exercise the inference reader's hash boundaries.

    The graph generator's physical swap audit is tested separately; this
    fixture supplies an already-validated catalog with the real dose counts.
    """
    root = tmp_path / "graphs"
    actual = root / "pubmed" / "actual"
    prepared_root = root / "pubmed" / "prepared"
    raw = _raw_catalog()
    raw["source_sha256"] = GRAPH_API["source_hashes"]()
    prepared = {
        "design_id": GRAPH_API["DESIGN_ID"],
        "dataset": "pubmed",
        "original_tail_count": 2696,
        "eligible_tail_count": 175,
        "desired_same": False,
        "source_sha256": raw["source_sha256"],
        "dose_plan": raw["accepted_doses"],
    }
    GRAPH_API["atomic_json"](prepared_root / "manifest.json", prepared)
    raw["prepared_manifest"] = GRAPH_API["file_record"](prepared_root / "manifest.json")
    experiment = {
        "design_id": GRAPH_API["DESIGN_ID"],
        "dataset": "pubmed",
        "phase": "actual",
        "desired_same": False,
        "source_sha256": raw["source_sha256"],
        "prepared_manifest_sha256": raw["prepared_manifest"]["sha256"],
        "dose_plan": prepared["dose_plan"],
    }
    GRAPH_API["atomic_json"](actual / "experiment_manifest.json", experiment)
    raw["experiment_manifest"] = GRAPH_API["file_record"](actual / "experiment_manifest.json")
    GRAPH_API["atomic_json"](actual / "baseline.json", raw["baseline"])
    edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    edge_hash = GRAPH_API["array_sha256"](edges)
    for entry in raw["pairs"]:
        path = actual / entry["relative_directory"]
        path.mkdir(parents=True)
        np.savez(path / "graphs.npz", treatment_edge_index=edges, control_edge_index=edges)
        GRAPH_API["atomic_json"](path / "swaps.json", {"treatment": [], "control": []})
        metadata = {
            "design_id": GRAPH_API["DESIGN_ID"],
            "dataset": "pubmed",
            "phase": "actual",
            "replicate_index": entry["replicate_index"],
            "replicate_seed": entry["replicate_seed"],
            **entry["dose"],
            "original_tail_count": 2696,
            "desired_same": False,
            "experiment_manifest_sha256": raw["experiment_manifest"]["sha256"],
            "audit": {
                "all_node_degrees_preserved": True,
                "edge_count_preserved": True,
                "swaps_replay_exactly": True,
                "control_target_same_count_change": 0,
                "treatment_target_same_count_change": -1,
                "target_count": entry["dose"]["target_count"],
                "treatment_edge_sha256": edge_hash,
                "control_edge_sha256": edge_hash,
            },
        }
        GRAPH_API["atomic_json"](path / "metadata.json", metadata)
        entry["files"] = {
            name: GRAPH_API["file_record"](path / name)
            for name in ("graphs.npz", "swaps.json", "metadata.json")
        }
    GRAPH_API["atomic_json"](actual / "catalog.json", raw)
    validation = {
        "design_id": GRAPH_API["DESIGN_ID"],
        "dataset": "pubmed",
        "validated": True,
        "pair_count": raw["pair_count"],
        "replicates_per_accepted_dose": 5,
        "no_partial_replicate_averages": True,
        "original_tail_denominator_verified": True,
        "independent_swap_replay_verified": True,
        "accepted_nonzero_dose_count": len(raw["accepted_doses"]),
        "all_planned_doses_constructed": True,
        "maximum_accepted_pct": raw["accepted_doses"][-1]["actual_pct"],
        "catalog": GRAPH_API["file_record"](actual / "catalog.json"),
    }
    GRAPH_API["atomic_json"](actual / "validation.json", validation)
    return root


def test_graph_catalog_file_authentication_preserves_fractional_dose_and_pair_identity(
    tmp_path: Path,
) -> None:
    root = _catalog_files(tmp_path)
    catalog = API["load_graph_catalog"](root, "pubmed")
    assert catalog["pair_count"] == 10
    for entry in catalog["pairs"]:
        assert entry["target_count"] == entry["dose"]["target_count"]
        assert entry["actual_pct"] == 100 * entry["target_count"] / 2696
        assert entry["p_percent"] == entry["actual_pct"]
        assert entry["pairing_key"] == API["pairing_key"](
            "pubmed", entry["replicate_index"], entry["target_count"]
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "actual/doses/targets-000134/replicate-0005/graphs.npz",
        "actual/doses/targets-000134/replicate-0005/swaps.json",
        "actual/doses/targets-000134/replicate-0005/metadata.json",
        "prepared/manifest.json",
        "actual/experiment_manifest.json",
        "actual/catalog.json",
    ],
)
def test_graph_catalog_rejects_changed_bound_files(tmp_path: Path, relative_path: str) -> None:
    root = _catalog_files(tmp_path)
    API["load_graph_catalog"](root, "pubmed")
    path = root / "pubmed" / relative_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError):
        API["load_graph_catalog"](root, "pubmed")


def test_loaded_graph_arrays_are_rechecked_before_inference(tmp_path: Path) -> None:
    root = _catalog_files(tmp_path)
    catalog = API["load_graph_catalog"](root, "pubmed")
    entry = catalog["pairs"][0]
    treatment, control = API["load_graph_edges"](root, catalog, entry)
    assert np.array_equal(treatment, control)
    path = root / "pubmed" / entry["relative_dir"] / "graphs.npz"
    changed = np.asarray([[0, 2], [2, 0]], dtype=np.int64)
    np.savez(path, treatment_edge_index=changed, control_edge_index=control)
    with pytest.raises(RuntimeError):
        API["load_graph_edges"](root, catalog, entry)


@pytest.mark.parametrize("device,expected_index", [("cuda", 2), ("cuda:0", 0), ("cuda:3", 3)])
def test_main_resolves_cuda_memory_fraction_device_to_an_index(
    monkeypatch, device: str, expected_index: int,
) -> None:
    namespace = API["main"].__globals__
    args = SimpleNamespace(
        command="evaluate", input_snapshot=None, source_hash_manifest=None,
        device=device, torch_threads=4, cuda_memory_fraction=0.9,
    )
    calls = []
    monkeypatch.setitem(namespace, "parse_args", lambda argv: args)
    monkeypatch.setitem(namespace, "implementation_hashes", lambda: {"runner.py": "a" * 64})
    monkeypatch.setitem(namespace, "evaluate", lambda unused: {"completed": True})
    monkeypatch.setattr(namespace["torch"], "set_num_threads", lambda unused: None)
    monkeypatch.setattr(namespace["torch"].cuda, "is_available", lambda: True)
    monkeypatch.setattr(namespace["torch"].cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        namespace["torch"].cuda, "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )
    assert API["main"]([]) == {"completed": True}
    assert calls == [(0.9, expected_index)]
