"""Focused contract tests for the Appendix-J expansion runner."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_edge_dose_response_expansion.py"
NAMESPACE = runpy.run_path(str(SCRIPT))
SPECS = NAMESPACE["DATASET_SPECS"]
ACCEPTED_RELEASE_REVISION = NAMESPACE["ACCEPTED_RELEASE_EXPERIMENT_REVISION"]


def test_fixed_matrix_is_exact_and_excludes_mlp() -> None:
    expected = {
        "pubmed": {"acm-gcn", "gpr-gnn", "h2gcn", "linkx"},
        "ogbn-arxiv": {"acm-gcn", "gpr-gnn", "h2gcn", "linkx"},
        "arxiv-year": {"acm-gcn", "gpr-gnn", "h2gcn", "linkx"},
        "roman-empire": {
            "gcn",
            "graphsage-mean",
            "acm-gcn",
            "gpr-gnn",
            "h2gcn",
            "linkx",
        },
        "amazon-ratings": {
            "gcn",
            "graphsage-mean",
            "acm-gcn",
            "gpr-gnn",
            "h2gcn",
            "linkx",
        },
    }
    assert set(SPECS) == set(expected)
    for dataset, models in expected.items():
        assert set(SPECS[dataset].models) == models
        assert len(SPECS[dataset].checkpoint_identities) == 30
    assert len(NAMESPACE["MODEL_DATASET_CELLS"]) == 24
    assert all(model != "mlp" for _, model in NAMESPACE["MODEL_DATASET_CELLS"])
    assert NAMESPACE["EXPECTED_CHECKPOINTS"] == 720
    assert SPECS["pubmed"].pairs_per_checkpoint == 100
    assert SPECS["roman-empire"].pairs_per_checkpoint == 70
    assert NAMESPACE["EXPECTED_PAIRED_ROWS"] == 66_600


@pytest.mark.parametrize("preparation", ("last", "head", "full"))
def test_recovery_preparations_use_distinct_full_matrix_contracts(
    preparation: str,
) -> None:
    cells = NAMESPACE["model_dataset_cells_for_preparation"](preparation)
    assert len(cells) == 30
    assert set(model for _, model in cells) == set(NAMESPACE["FULL_MODELS"])
    assert all(model != "mlp" for _, model in cells)
    assert NAMESPACE["expected_checkpoint_count"](preparation) == 900
    assert NAMESPACE["expected_paired_row_count"](preparation) == 84_600
    assert NAMESPACE["design_id_for_preparation"](preparation) == (
        f"appendix-j-edge-dose-response-{preparation}-full-matrix-v1"
    )
    assert NAMESPACE["design_id_for_preparation"](preparation) != NAMESPACE[
        "DESIGN_ID"
    ]


def test_erm_contract_and_run_ids_remain_backwards_compatible() -> None:
    assert NAMESPACE["model_dataset_cells_for_preparation"]("erm") == NAMESPACE[
        "MODEL_DATASET_CELLS"
    ]
    assert NAMESPACE["expected_checkpoint_count"]("erm") == 720
    assert NAMESPACE["expected_paired_row_count"]("erm") == 66_600
    assert NAMESPACE["design_id_for_preparation"]("erm") == NAMESPACE["DESIGN_ID"]
    make_id = NAMESPACE["parent_run_id"]
    spec = SPECS["pubmed"]
    assert make_id(spec, "gcn", "geomgcn-0", 1) == (
        "pubmed_gcn_erm_geomgcn-0_seed-1"
    )
    assert make_id(spec, "gcn", "geomgcn-0", 1, "head") == (
        "pubmed_gcn_head_geomgcn-0_seed-1"
    )


def test_baseline_logit_tolerance_is_explicit_and_tail_predictions_stay_exact(
    tmp_path: Path,
) -> None:
    saved_logits = np.asarray([[1.0, 0.0], [1.5e-5, 0.0]], dtype=np.float32)
    current_logits = np.asarray([[1.0, 0.0], [0.0, 1.0e-6]], dtype=np.float32)
    run_data_path = tmp_path / "run_data.npz"
    np.savez(run_data_path, logits=saved_logits)
    checkpoint = SimpleNamespace(
        dataset="pubmed",
        model="gcn",
        run_id="pubmed_gcn_head_geomgcn-0_seed-1",
        split="geomgcn-0",
        seed=1,
        saved_predictions=saved_logits.argmax(axis=-1),
        run_data_path=run_data_path,
        checkpoint_sha256="a" * 64,
        training_code_version="training-code",
        preparation="head",
    )
    bundle = SimpleNamespace(
        num_nodes=2,
        y=np.asarray([0, 1], dtype=np.int64),
    )
    arrays = {
        "geomgcn-0.test": np.asarray([0], dtype=np.int64),
        "tail_mask": np.asarray([True, False]),
    }

    with pytest.raises(RuntimeError, match="baseline logits do not reproduce parent"):
        NAMESPACE["baseline_row_from_logits"](
            checkpoint,
            bundle=bundle,
            arrays=arrays,
            current_logits=current_logits,
        )

    row = NAMESPACE["baseline_row_from_logits"](
        checkpoint,
        bundle=bundle,
        arrays=arrays,
        current_logits=current_logits,
        logit_atol=2e-5,
        logit_rtol=1e-5,
    )
    assert row["n_tail_test_prediction_differences_from_parent"] == 0
    assert row["n_full_graph_prediction_differences_from_parent"] == 1
    assert row["full_graph_logit_reproduction_atol"] == 2e-5
    assert row["full_graph_logit_reproduction_rtol"] == 1e-5

    tail_changed_logits = current_logits.copy()
    tail_changed_logits[0] = np.asarray([0.0, 1.0], dtype=np.float32)
    with pytest.raises(
        RuntimeError,
        match="fixed tail-test predictions fail reproduction",
    ):
        NAMESPACE["baseline_row_from_logits"](
            checkpoint,
            bundle=bundle,
            arrays=arrays,
            current_logits=tail_changed_logits,
            logit_atol=10.0,
            logit_rtol=10.0,
        )


def test_checkpoint_loader_selects_requested_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict]] = []

    def fake_validate(run_dir: Path, **kwargs):
        calls.append((run_dir, kwargs))
        return (
            {"model_config": {}, "code_version": "training-code"},
            run_dir / "checkpoint.pt",
            np.asarray([0, 1], dtype=np.int64),
        )

    loader = NAMESPACE["load_checkpoint_spec"]
    monkeypatch.setitem(loader.__globals__, "validate_parent_run", fake_validate)
    monkeypatch.setitem(loader.__globals__, "file_sha256", lambda _path: "a" * 64)
    checkpoint = loader(
        SPECS["pubmed"],
        model="gcn",
        split="geomgcn-0",
        seed=1,
        checkpoint_root=Path("checkpoints"),
        fixed_input_checksums={"tail_mask": "registered"},
        num_nodes=2,
        preparation="head",
    )

    expected_run_id = "pubmed_gcn_head_geomgcn-0_seed-1"
    assert checkpoint.run_id == expected_run_id
    assert checkpoint.preparation == "head"
    assert calls == [
        (
            Path("checkpoints") / expected_run_id,
            {
                "dataset": "pubmed",
                "model": "gcn",
                "split": "geomgcn-0",
                "seed": 1,
                "fixed_input_checksums": {"tail_mask": "registered"},
                "preparation": "head",
            },
        )
    ]


@pytest.mark.parametrize(
    ("preparation", "has_parent"),
    (("erm", False), ("last", True), ("head", True), ("full", False)),
)
def test_parent_run_validation_authenticates_run_and_erm_parent_identity(
    tmp_path: Path,
    preparation: str,
    has_parent: bool,
) -> None:
    dataset = "pubmed"
    model = "gcn"
    split = "geomgcn-0"
    seed = 1
    run_id = f"{dataset}_{model}_{preparation}_{split}_seed-{seed}"
    erm_run_id = f"{dataset}_{model}_erm_{split}_seed-{seed}"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config = {
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "preparation": preparation,
        "split": split,
        "seed": seed,
        "fixed_input_checksums": {"tail_mask": "registered"},
    }
    if has_parent:
        config["parent_run_id"] = erm_run_id
    (run_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    np.savez(run_dir / "run_data.npz", predictions=np.asarray([0, 1]))

    validate = NAMESPACE["validate_parent_run"]
    observed, checkpoint_path, predictions = validate(
        run_dir,
        dataset=dataset,
        model=model,
        split=split,
        seed=seed,
        fixed_input_checksums={"tail_mask": "registered"},
        preparation=preparation,
    )
    assert observed == config
    assert checkpoint_path == run_dir / "checkpoint.pt"
    assert predictions.tolist() == [0, 1]

    config["run_id"] = f"wrong-{run_id}"
    (run_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="parent config mismatch"):
        validate(
            run_dir,
            dataset=dataset,
            model=model,
            split=split,
            seed=seed,
            fixed_input_checksums={"tail_mask": "registered"},
            preparation=preparation,
        )
    if has_parent:
        config["run_id"] = run_id
        config["parent_run_id"] = f"wrong-{erm_run_id}"
        (run_dir / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="parent config mismatch"):
            validate(
                run_dir,
                dataset=dataset,
                model=model,
                split=split,
                seed=seed,
                fixed_input_checksums={"tail_mask": "registered"},
                preparation=preparation,
            )


@pytest.mark.parametrize("preparation", ("erm", "full"))
def test_parent_run_validation_rejects_parent_for_fresh_training(
    tmp_path: Path,
    preparation: str,
) -> None:
    run_id = f"pubmed_gcn_{preparation}_geomgcn-0_seed-1"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": "pubmed",
                "model": "gcn",
                "preparation": preparation,
                "split": "geomgcn-0",
                "seed": 1,
                "parent_run_id": "pubmed_gcn_erm_geomgcn-0_seed-1",
                "fixed_input_checksums": {"tail_mask": "registered"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    np.savez(run_dir / "run_data.npz", predictions=np.asarray([0, 1]))
    with pytest.raises(RuntimeError, match="parent config mismatch"):
        NAMESPACE["validate_parent_run"](
            run_dir,
            dataset="pubmed",
            model="gcn",
            split="geomgcn-0",
            seed=1,
            fixed_input_checksums={"tail_mask": "registered"},
            preparation=preparation,
        )


def test_dataset_identity_and_graph_contracts_are_frozen() -> None:
    assert SPECS["pubmed"].checkpoint_identities[0] == ("geomgcn-0", 1)
    assert SPECS["pubmed"].checkpoint_identities[-1] == ("geomgcn-9", 3)
    assert SPECS["ogbn-arxiv"].checkpoint_identities == tuple(
        ("ogb-time", seed) for seed in range(1, 31)
    )
    assert SPECS["arxiv-year"].checkpoint_identities[-1] == ("lim-4", 6)
    assert SPECS["roman-empire"].tail_count == 1012
    assert SPECS["amazon-ratings"].tail_count == 2943
    assert SPECS["roman-empire"].seed_base == 2026080400
    assert SPECS["amazon-ratings"].seed_base == 2026080500
    assert SPECS["roman-empire"].desired_same is False
    assert SPECS["amazon-ratings"].desired_same is False
    assert SPECS["roman-empire"].percentages == tuple(range(0, 71, 5))
    assert SPECS["roman-empire"].generation_max_attempts == 1000
    assert SPECS["amazon-ratings"].percentages == tuple(range(0, 101, 5))
    assert SPECS["amazon-ratings"].generation_max_attempts == 1


def test_reused_graph_manifest_hashes_are_exact() -> None:
    assert SPECS["pubmed"].reused_manifest_sha256 == (
        "ac11f23c139ee876295fbc4e324ba16c31382fe8a3ce49b63303148f1e95a118"
    )
    assert SPECS["arxiv-year"].reused_manifest_sha256 == (
        "65503d3519824431bd4d7c70fe4138f61cb1ab533a4b9ba53c8367c90d5aad4b"
    )
    assert SPECS["ogbn-arxiv"].reused_manifest_sha256 == (
        "fb4d4b1f586df98f8fa884736c428ff6008aca494bc080ae824d30043ceaa834"
    )
    assert SPECS["roman-empire"].reused_manifest_sha256 is None
    assert SPECS["amazon-ratings"].reused_manifest_sha256 is None


@pytest.mark.parametrize(
    ("dataset", "expected"),
    (
        ("roman-empire", {5: 50, 50: 506, 70: 708}),
        ("amazon-ratings", {5: 147, 50: 1471, 100: 2943}),
    ),
)
def test_target_count_uses_registered_floor_rule(
    dataset: str, expected: dict[int, int]
) -> None:
    count = NAMESPACE["target_count_for_percentage"]
    for percentage, wanted in expected.items():
        assert count(SPECS[dataset].tail_count, percentage) == wanted


@pytest.mark.parametrize(
    ("dataset", "expected_percentages", "expected_attempts"),
    (
        ("roman-empire", list(range(0, 71, 5)), 1000),
        ("amazon-ratings", list(range(0, 101, 5)), 1),
    ),
)
def test_new_graph_manifest_freezes_doses_attempt_limit_and_seed(
    dataset: str, expected_percentages: list[int], expected_attempts: int
) -> None:
    args = Namespace(code_version="abc123", git_dirty=False)
    manifest = NAMESPACE["generated_graph_manifest"](args, SPECS[dataset])
    assert manifest["design_id"] == NAMESPACE["GRAPH_DESIGN_ID"]
    assert manifest["replicate_seed_base"] == SPECS[dataset].seed_base
    assert manifest["p_percentages"] == expected_percentages
    assert manifest["actual_generation_attempts"] == expected_attempts
    assert manifest["models_sharing_graphs"] == list(SPECS[dataset].models)


def test_generation_attempt_validation_uses_dataset_limit() -> None:
    validate = NAMESPACE["validate_generation_attempt"]
    validate(SimpleNamespace(attempt=0, retry_failures=[]), SPECS["amazon-ratings"])
    validate(
        SimpleNamespace(attempt=999, retry_failures=["failed"] * 999),
        SPECS["roman-empire"],
    )
    with pytest.raises(RuntimeError, match="outside the fixed limit"):
        validate(
            SimpleNamespace(attempt=1, retry_failures=["first failed"]),
            SPECS["amazon-ratings"],
        )
    with pytest.raises(RuntimeError, match="retry history"):
        validate(SimpleNamespace(attempt=2, retry_failures=[]), SPECS["roman-empire"])


def test_pair_layout_has_no_zero_percent_graph_directory() -> None:
    path = NAMESPACE["graph_pair_path"](
        Path("inputs") / "graphs", "pubmed", 3, 65
    )
    assert path.as_posix().endswith("pubmed/replicate-0003/pct-065")
    assert 0 not in NAMESPACE["NONZERO_PERCENTAGES"]
    assert len(NAMESPACE["expected_pair_cells"](SPECS["pubmed"])) == 100
    assert len(NAMESPACE["expected_pair_cells"](SPECS["roman-empire"])) == 70


def test_roman_smoke_generation_rejects_excluded_doses() -> None:
    args = Namespace(
        phase="smoke",
        replicates=[1],
        percentages=[75],
    )
    with pytest.raises(SystemExit, match="roman-empire percentages"):
        NAMESPACE["generate_graph_dataset"](args, SPECS["roman-empire"])


def test_actual_graph_cli_requires_one_dataset_and_explicit_doses() -> None:
    base = [
        "generate-graphs",
        "--phase",
        "actual",
        "--datasets",
        "roman-empire",
        "amazon-ratings",
        "--replicates",
        "1",
        "2",
        "3",
        "4",
        "5",
        "--graph-root",
        "graphs",
        "--data-root",
        "data",
        "--array-root",
        "arrays",
        "--manifest-root",
        "manifests",
    ]
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"](base)
    args = NAMESPACE["parse_args"](
        [*base, "--percentages", *map(str, range(5, 71, 5))]
    )
    with pytest.raises(SystemExit, match="exactly one dataset"):
        NAMESPACE["validate_args"](args)


def test_roman_manifest_accepts_only_cutoff_and_attempt_limit(tmp_path: Path) -> None:
    spec = SPECS["roman-empire"]
    args = Namespace(code_version="abc123", git_dirty=False)
    path = tmp_path / "experiment_manifest.json"
    manifest = NAMESPACE["generated_graph_manifest"](args, spec)
    NAMESPACE["atomic_json"](path, manifest)
    observed, _ = NAMESPACE["_validate_graph_manifest"](spec, path)
    assert observed == manifest

    NAMESPACE["atomic_json"](
        path,
        {
            **manifest,
            "p_percentages": list(range(0, 101, 5)),
        },
    )
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        NAMESPACE["_validate_graph_manifest"](spec, path)


def test_authentication_rejects_excluded_and_in_progress_directories(
    tmp_path: Path,
) -> None:
    graph_root = tmp_path / "graphs"
    dataset_root = graph_root / "roman-empire"
    excluded = dataset_root / "replicate-0001" / "pct-075"
    excluded.mkdir(parents=True)
    args = Namespace(graph_root=graph_root)
    with pytest.raises(RuntimeError, match="graph directories differ"):
        NAMESPACE["authenticate_graph_dataset"](args, SPECS["roman-empire"])

    excluded.rmdir()
    in_progress = dataset_root / "replicate-0001" / "pct-070.in-progress"
    in_progress.mkdir()
    with pytest.raises(RuntimeError, match="incomplete graph directories"):
        NAMESPACE["authenticate_graph_dataset"](args, SPECS["roman-empire"])


def test_h2_prepares_operators_once_and_reuses_same_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hfd.models.h2gcn as h2gcn

    prepared = (object(), object())
    prepare_calls: list[tuple] = []
    forward_calls: list[tuple[str, object]] = []

    def fake_prepare(edge_index, num_nodes, *, device, dtype):
        prepare_calls.append((edge_index.shape, num_nodes, device, dtype))
        return prepared

    def fake_forward(network, x, edge_tensor, *, prepared_h2_operators, **kwargs):
        forward_calls.append((network.name, prepared_h2_operators))
        return np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(h2gcn, "prepare_h2gcn_operators", fake_prepare)
    function = NAMESPACE["h2_predictions_for_graph"]
    monkeypatch.setitem(function.__globals__, "_forward_logits", fake_forward)
    networks = {
        "run-a": SimpleNamespace(name="a"),
        "run-b": SimpleNamespace(name="b"),
    }
    predictions = function(
        networks,
        torch.zeros((2, 3), dtype=torch.float32),
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        device=torch.device("cpu"),
        pause_seconds=0.0,
    )
    assert len(prepare_calls) == 1
    assert forward_calls == [("a", prepared), ("b", prepared)]
    assert set(predictions) == set(networks)


def test_h2_cache_is_installed_before_forward_and_then_cleared() -> None:
    events: list[str] = []

    class Encoder:
        def __init__(self):
            self.cache = None

        def clear_graph_cache(self):
            events.append("clear")
            self.cache = None

        def set_graph_cache(self, operators, *, num_edges):
            events.append("set")
            assert num_edges == 2
            self.cache = operators

    class Network:
        def __init__(self):
            self.encoder = Encoder()

        def __call__(self, x, edge_index):
            events.append("forward")
            assert self.encoder.cache == ("hop1", "hop2")
            return torch.asarray([[2.0, 1.0], [1.0, 2.0]])

    logits = NAMESPACE["_forward_logits"](
        Network(),
        torch.zeros((2, 1)),
        torch.asarray([[0, 1], [1, 0]]),
        prepared_h2_operators=("hop1", "hop2"),
    )
    assert events == ["clear", "set", "forward", "clear"]
    assert logits.shape == (2, 2)


def test_models_without_graph_cache_are_supported() -> None:
    class Network:
        encoder = SimpleNamespace()

        def __call__(self, x, edge_index):
            return torch.asarray([[2.0, 1.0], [1.0, 2.0]])

    logits = NAMESPACE["_forward_logits"](
        Network(),
        torch.zeros((2, 1)),
        torch.asarray([[0, 1], [1, 0]]),
    )
    assert logits.argmax(axis=-1).tolist() == [0, 1]


def test_every_forward_rejects_non_finite_logits() -> None:
    class Network:
        encoder = SimpleNamespace()

        def __call__(self, x, edge_index):
            return torch.asarray([[float("nan"), 1.0], [1.0, 2.0]])

    with pytest.raises(RuntimeError, match="non-finite logits"):
        NAMESPACE["_forward_logits"](
            Network(),
            torch.zeros((2, 1)),
            torch.asarray([[0, 1], [1, 0]]),
        )


def test_checkpoint_major_function_loads_network_before_graph_loop_once() -> None:
    source = inspect.getsource(NAMESPACE["evaluate_checkpoint_major"])
    assert source.count("load_network(") == 1
    assert source.index("network = load_network(") < source.index(
        'for graph_entry in catalog["pairs"]:'
    )


def _paired_row(
    run_id: str, replicate: int, percentage: int, *, dataset: str = "pubmed"
) -> dict:
    pair = NAMESPACE["pairing_key"](dataset, replicate, percentage)
    return {
        "run_id": run_id,
        "replicate_index": replicate,
        "p_percent": percentage,
        "pairing_key": pair,
        "checkpoint_sha256": "a" * 64,
        "treatment_prediction_key": f"{run_id}__{pair}__treatment",
        "control_prediction_key": f"{run_id}__{pair}__control",
    }


def _payload(
    *,
    phase: str,
    pair_rows: list[dict],
    dataset: str = "pubmed",
    model: str = "acm-gcn",
    run_id: str = "pubmed_acm-gcn_erm_geomgcn-0_seed-1",
) -> dict:
    return {
        "schema_version": NAMESPACE["SCHEMA_VERSION"],
        "design_id": NAMESPACE["DESIGN_ID"],
        "completed": True,
        "phase": phase,
        "dataset": dataset,
        "model": model,
        "run_id": run_id,
        "checkpoint_sha256": "a" * 64,
        "graph_catalog_sha256": "c" * 64,
        "cuda_peak_allocated_bytes": 0,
        "cuda_peak_reserved_bytes": 0,
        "cuda_peak_measurement_scope": "unit-test",
        "baseline": {
            "run_id": run_id,
            "checkpoint_sha256": "a" * 64,
            "prediction_key": f"{run_id}__original",
            "n_tail_test_prediction_differences_from_parent": 0,
        },
        "pairs": pair_rows,
    }


def test_actual_shard_requires_complete_paired_panel_and_keys() -> None:
    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    rows = [
        _paired_row(run_id, replicate, percentage)
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in NAMESPACE["NONZERO_PERCENTAGES"]
    ]
    validate = NAMESPACE["validate_shard_payload"]
    validate(
        _payload(phase="actual", pair_rows=rows),
        spec=SPECS["pubmed"],
        model="acm-gcn",
        run_id=run_id,
        graph_catalog_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="100 paired rows"):
        validate(
            _payload(phase="actual", pair_rows=rows[:-1]),
            spec=SPECS["pubmed"],
            model="acm-gcn",
            run_id=run_id,
            graph_catalog_sha256="c" * 64,
        )
    rows[0] = {**rows[0], "control_prediction_key": rows[0]["treatment_prediction_key"]}
    with pytest.raises(RuntimeError, match="control prediction key"):
        validate(
            _payload(phase="actual", pair_rows=rows),
            spec=SPECS["pubmed"],
            model="acm-gcn",
            run_id=run_id,
            graph_catalog_sha256="c" * 64,
        )


def test_recovery_shard_is_bound_to_preparation_identity() -> None:
    preparation = "last"
    run_id = "pubmed_acm-gcn_last_geomgcn-0_seed-1"
    rows = [
        {
            **_paired_row(run_id, replicate, percentage),
            "preparation": preparation,
        }
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in NAMESPACE["NONZERO_PERCENTAGES"]
    ]
    payload = _payload(
        phase="actual",
        pair_rows=rows,
        run_id=run_id,
    )
    payload.update(
        {
            "design_id": NAMESPACE["design_id_for_preparation"](preparation),
            "preparation": preparation,
        }
    )
    payload["baseline"]["preparation"] = preparation
    validate = NAMESPACE["validate_shard_payload"]
    kwargs = {
        "spec": SPECS["pubmed"],
        "model": "acm-gcn",
        "run_id": run_id,
        "graph_catalog_sha256": "c" * 64,
        "expected_preparation": preparation,
    }
    validate(payload, **kwargs)
    with pytest.raises(RuntimeError, match="design_id"):
        validate({**payload, "design_id": NAMESPACE["DESIGN_ID"]}, **kwargs)
    with pytest.raises(RuntimeError, match="preparation"):
        validate({**payload, "preparation": "head"}, **kwargs)


def test_legacy_erm_shard_rejects_an_explicit_other_preparation() -> None:
    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    rows = [
        _paired_row(run_id, replicate, percentage)
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in NAMESPACE["NONZERO_PERCENTAGES"]
    ]
    payload = _payload(phase="actual", pair_rows=rows, run_id=run_id)
    payload["preparation"] = "head"
    with pytest.raises(RuntimeError, match="shard preparation differs"):
        NAMESPACE["validate_shard_payload"](
            payload,
            spec=SPECS["pubmed"],
            model="acm-gcn",
            run_id=run_id,
            graph_catalog_sha256="c" * 64,
        )


def test_actual_roman_shard_stops_at_70_percent() -> None:
    run_id = "roman-empire_gcn_erm_platonov-0_seed-1"
    rows = [
        _paired_row(
            run_id,
            replicate,
            percentage,
            dataset="roman-empire",
        )
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in SPECS["roman-empire"].nonzero_percentages
    ]
    NAMESPACE["validate_shard_payload"](
        _payload(
            phase="actual",
            pair_rows=rows,
            dataset="roman-empire",
            model="gcn",
            run_id=run_id,
        ),
        spec=SPECS["roman-empire"],
        model="gcn",
        run_id=run_id,
        graph_catalog_sha256="c" * 64,
    )
    assert len(rows) == 70
    assert max(row["p_percent"] for row in rows) == 70


def test_smoke_shard_accepts_bounded_subset() -> None:
    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    NAMESPACE["validate_shard_payload"](
        _payload(phase="smoke", pair_rows=[_paired_row(run_id, 1, 5)]),
        spec=SPECS["pubmed"],
        model="acm-gcn",
        run_id=run_id,
        graph_catalog_sha256="c" * 64,
    )


def test_cuda_peak_telemetry_is_reset_captured_and_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 456)
    device = torch.device("cuda:3")
    NAMESPACE["reset_cuda_peak_memory"](device)
    telemetry = NAMESPACE["cuda_peak_memory"](
        device, scope="checkpoint-evaluation"
    )
    assert calls == [("reset", "cuda:3")]
    assert telemetry == {
        "cuda_peak_allocated_bytes": 123,
        "cuda_peak_reserved_bytes": 456,
        "cuda_peak_measurement_scope": "checkpoint-evaluation",
    }

    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    payload = _payload(
        phase="smoke", pair_rows=[_paired_row(run_id, 1, 5)]
    )
    del payload["cuda_peak_reserved_bytes"]
    with pytest.raises(RuntimeError, match="CUDA peak reserved bytes"):
        NAMESPACE["validate_shard_payload"](
            payload,
            spec=SPECS["pubmed"],
            model="acm-gcn",
            run_id=run_id,
            graph_catalog_sha256="c" * 64,
        )


def test_evaluation_schedules_reset_cuda_peak_intervals() -> None:
    checkpoint_source = inspect.getsource(NAMESPACE["evaluate_checkpoint_major"])
    assert checkpoint_source.index("reset_cuda_peak_memory(device)") < (
        checkpoint_source.index("network = load_network(")
    )
    h2_source = inspect.getsource(NAMESPACE["evaluate_h2_graph_major"])
    assert h2_source.count("reset_cuda_peak_memory(device)") == 2


def test_existing_shard_is_bound_to_phase_code_and_checkpoint() -> None:
    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    checkpoint_sha = "a" * 64
    rows = [
        {
            **_paired_row(run_id, replicate, percentage),
            "checkpoint_sha256": checkpoint_sha,
        }
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in NAMESPACE["NONZERO_PERCENTAGES"]
    ]
    payload = _payload(phase="actual", pair_rows=rows)
    payload.update(
        {
            "code_version": "current-code",
            "checkpoint_sha256": checkpoint_sha,
        }
    )
    payload["baseline"]["checkpoint_sha256"] = checkpoint_sha
    validate = NAMESPACE["validate_shard_payload"]
    kwargs = {
        "spec": SPECS["pubmed"],
        "model": "acm-gcn",
        "run_id": run_id,
        "graph_catalog_sha256": "c" * 64,
        "expected_phase": "actual",
        "expected_code_version": "current-code",
        "expected_checkpoint_sha256": checkpoint_sha,
    }
    validate(payload, **kwargs)
    with pytest.raises(RuntimeError, match="phase"):
        validate({**payload, "phase": "smoke"}, **kwargs)
    with pytest.raises(RuntimeError, match="code version"):
        validate({**payload, "code_version": "old-code"}, **kwargs)
    with pytest.raises(RuntimeError, match="checkpoint SHA256"):
        validate({**payload, "checkpoint_sha256": "b" * 64}, **kwargs)


def test_validate_command_requires_checkpoint_root() -> None:
    base = [
        "validate",
        "--graph-root",
        "graphs",
        "--catalog-root",
        "catalogs",
        "--output-root",
        "outputs",
    ]
    with pytest.raises(SystemExit):
        NAMESPACE["parse_args"](base)
    args = NAMESPACE["parse_args"](
        [*base, "--checkpoint-root", "checkpoints"]
    )
    assert args.checkpoint_root == Path("checkpoints")
    assert args.preparation == "erm"


def test_validate_resolves_paths_when_release_manifest_is_omitted(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    args = NAMESPACE["parse_args"](
        [
            "validate",
            "--graph-root",
            str(tmp_path / "graphs"),
            "--catalog-root",
            str(tmp_path / "catalogs"),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    NAMESPACE["_resolve_paths"](args)

    assert args.release_provenance_manifest is None
    assert args.graph_root == (tmp_path / "graphs").resolve()
    assert args.catalog_root == (tmp_path / "catalogs").resolve()
    assert args.output_root == (tmp_path / "outputs").resolve()


def test_cli_accepts_recovery_preparation_for_evaluation_and_validation() -> None:
    evaluate = NAMESPACE["parse_args"](
        [
            "evaluate",
            "--phase",
            "actual",
            "--dataset",
            "pubmed",
            "--model",
            "gcn",
            "--graph-root",
            "graphs",
            "--catalog-root",
            "catalogs",
            "--checkpoint-root",
            "checkpoints",
            "--output-root",
            "outputs-last",
            "--data-root",
            "data",
            "--array-root",
            "arrays",
            "--manifest-root",
            "manifests",
            "--preparation",
            "last",
        ]
    )
    assert evaluate.preparation == "last"
    assert "gcn" in NAMESPACE["models_for_preparation"](
        SPECS["pubmed"], evaluate.preparation
    )
    validate = NAMESPACE["parse_args"](
        [
            "validate",
            "--graph-root",
            "graphs",
            "--catalog-root",
            "catalogs",
            "--checkpoint-root",
            "checkpoints",
            "--output-root",
            "outputs-last",
            "--preparation",
            "last",
        ]
    )
    assert validate.preparation == "last"


def test_final_validation_hashes_the_current_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = replace(
        SPECS["pubmed"],
        models=("acm-gcn",),
        split_ids=("geomgcn-0",),
        training_seeds=(1,),
    )
    run_id = "pubmed_acm-gcn_erm_geomgcn-0_seed-1"
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_root / run_id / "checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"current checkpoint")
    checkpoint_sha256 = NAMESPACE["file_sha256"](checkpoint_path)

    rows = [
        _paired_row(run_id, replicate, percentage)
        for replicate in NAMESPACE["REPLICATE_INDICES"]
        for percentage in NAMESPACE["NONZERO_PERCENTAGES"]
    ]
    payload = _payload(phase="actual", pair_rows=rows)
    payload.update(
        {
            "code_version": "current-code",
            "checkpoint_sha256": checkpoint_sha256,
        }
    )
    payload["baseline"]["checkpoint_sha256"] = checkpoint_sha256
    for row in payload["pairs"]:
        row["checkpoint_sha256"] = checkpoint_sha256

    output_root = tmp_path / "outputs"
    shard = output_root / "pubmed" / "acm-gcn" / "shards" / f"{run_id}.json"
    NAMESPACE["atomic_json"](shard, payload)
    function_globals = NAMESPACE["collect_and_validate_outputs"].__globals__
    monkeypatch.setitem(function_globals, "DATASET_SPECS", {"pubmed": spec})
    monkeypatch.setitem(
        function_globals,
        "MODEL_DATASET_CELLS",
        (("pubmed", "acm-gcn"),),
    )
    monkeypatch.setitem(function_globals, "EXPECTED_CHECKPOINTS", 1)
    monkeypatch.setitem(
        function_globals,
        "EXPECTED_PAIRED_ROWS",
        spec.pairs_per_checkpoint,
    )
    monkeypatch.setitem(
        function_globals,
        "load_graph_catalog",
        lambda *args, **kwargs: ({}, "c" * 64),
    )
    args = Namespace(
        graph_root=tmp_path / "graphs",
        catalog_root=tmp_path / "catalogs",
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        code_version="current-code",
    )
    result = NAMESPACE["collect_and_validate_outputs"](args)
    assert result["n_checkpoint_shards"] == 1

    checkpoint_path.write_bytes(b"replacement checkpoint")
    with pytest.raises(RuntimeError, match="shard checkpoint SHA256 is stale"):
        NAMESPACE["collect_and_validate_outputs"](args)


def test_h2_progress_is_atomic_resumable_and_code_bound(tmp_path: Path) -> None:
    checkpoint_sha = "a" * 64
    checkpoint = NAMESPACE["CheckpointSpec"](
        dataset="pubmed",
        model="h2gcn",
        split="geomgcn-0",
        seed=1,
        run_id="pubmed_h2gcn_erm_geomgcn-0_seed-1",
        model_config={},
        checkpoint_path=tmp_path / "checkpoint.pt",
        run_data_path=tmp_path / "run_data.npz",
        checkpoint_sha256=checkpoint_sha,
        training_code_version="parent-code",
        saved_predictions=np.asarray([0, 1]),
    )
    checkpoints = {checkpoint.run_id: checkpoint}
    args = Namespace(phase="actual", code_version="current-code", git_dirty=False)
    row = {
        "run_id": checkpoint.run_id,
        "checkpoint_sha256": checkpoint_sha,
        "n_tail_test_prediction_differences_from_parent": 0,
    }
    path = tmp_path / "progress" / "baseline.json"
    written = NAMESPACE["_write_h2_progress"](
        path,
        args=args,
        spec=SPECS["pubmed"],
        catalog_sha256="c" * 64,
        checkpoints=checkpoints,
        graph_entry=None,
        rows=[row],
        evaluation_seconds=1.0,
        cuda_memory={
            "cuda_peak_allocated_bytes": 123,
            "cuda_peak_reserved_bytes": 456,
            "cuda_peak_measurement_scope": "h2-baseline-progress-unit",
        },
    )
    assert written.rows_by_run == {checkpoint.run_id: row}
    assert written.cuda_peak_allocated_bytes == 123
    assert written.cuda_peak_reserved_bytes == 456
    assert path.is_file()
    assert not path.with_name(f".{path.name}.tmp").exists()
    loaded = NAMESPACE["_load_h2_progress"](
        path,
        args=args,
        spec=SPECS["pubmed"],
        catalog_sha256="c" * 64,
        checkpoints=checkpoints,
        graph_entry=None,
    )
    assert loaded == written
    with pytest.raises(RuntimeError, match="code_version"):
        NAMESPACE["_load_h2_progress"](
            path,
            args=Namespace(
                phase="actual", code_version="new-code", git_dirty=False
            ),
            spec=SPECS["pubmed"],
            catalog_sha256="c" * 64,
            checkpoints=checkpoints,
            graph_entry=None,
        )


def test_graph_catalog_content_has_no_reauthentication_timestamp(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(NAMESPACE["authenticate_graph_dataset"])
    assert "authenticated_at" not in source
    path = tmp_path / "catalog.json"
    catalog = {"dataset": "pubmed", "pairs": [{"pairing_key": "fixed"}]}
    NAMESPACE["atomic_json"](path, catalog)
    first = NAMESPACE["file_sha256"](path)
    NAMESPACE["atomic_json"](path, catalog)
    assert NAMESPACE["file_sha256"](path) == first


def test_graph_manifest_allows_pubmed_v1_omissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "design_id": "pubmed-tail-edge-dose-response-v1",
        "dataset": "pubmed",
        "p_percentages": list(range(0, 101, 5)),
        "graph_replicates": 5,
        "graph_replicate_indices": [1, 2, 3, 4, 5],
        "replicate_seed_base": 2026080100,
        "tail_count": 2696,
        "actual_generation_attempts": 1,
    }
    path = tmp_path / "experiment_manifest.json"
    NAMESPACE["atomic_json"](path, manifest)
    spec = replace(SPECS["pubmed"], reused_manifest_sha256=None)
    observed, digest = NAMESPACE["_validate_graph_manifest"](spec, path)
    assert observed == manifest
    assert len(digest) == 64


def test_graph_file_records_are_content_bound_not_mtime_bound(tmp_path: Path) -> None:
    path = tmp_path / "graphs.npz"
    path.write_bytes(b"portable graph bytes")
    record = NAMESPACE["_file_record"](path)
    assert set(record) == {"bytes", "sha256"}
    NAMESPACE["_validate_bound_file"](path, record)
    path.touch()
    NAMESPACE["_validate_bound_file"](path, record)
    path.write_bytes(b"portable graph byteX")
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        NAMESPACE["_validate_bound_file"](path, record)


def test_legacy_catalog_can_be_relocated(tmp_path: Path) -> None:
    graph_root = tmp_path / "relocated" / "graphs"
    catalog_root = tmp_path / "relocated" / "catalogs"
    spec = replace(
        SPECS["pubmed"],
        percentages=(0, 5),
        reused_manifest_sha256=None,
    )
    dataset_root = graph_root / spec.dataset
    manifest_path = dataset_root / "experiment_manifest.json"
    baseline_path = dataset_root / "baseline" / "metadata.json"
    manifest_path.parent.mkdir(parents=True)
    baseline_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    baseline_path.write_text("{}", encoding="utf-8")
    entries = []
    for replicate in NAMESPACE["REPLICATE_INDICES"]:
        pair_root = dataset_root / f"replicate-{replicate:04d}" / "pct-005"
        pair_root.mkdir(parents=True)
        for name in NAMESPACE["GRAPH_FILES"]:
            (pair_root / name).write_bytes(f"{replicate}:{name}".encode())
        entries.append(
            {
                "replicate_index": replicate,
                "p_percent": 5,
                "relative_dir": pair_root.relative_to(dataset_root).as_posix(),
                "graphs": NAMESPACE["_file_record"](pair_root / "graphs.npz"),
                "swaps": NAMESPACE["_file_record"](pair_root / "swaps.json"),
                "metadata": NAMESPACE["_file_record"](pair_root / "metadata.json"),
            }
        )
    catalog = {
        "schema_version": 1,
        "design_id": NAMESPACE["DESIGN_ID"],
        "dataset": spec.dataset,
        "graph_root_binding": "/old/absolute/cluster/path",
        "source_experiment_manifest_sha256": "not-frozen-in-test",
        "source_experiment_manifest": NAMESPACE["_file_record"](manifest_path),
        "source_baseline_metadata": NAMESPACE["_file_record"](baseline_path),
        "pair_count": 5,
        "pairs": entries,
    }
    catalog_root.mkdir(parents=True)
    NAMESPACE["atomic_json"](catalog_root / "pubmed.json", catalog)
    loaded, digest = NAMESPACE["load_graph_catalog"](
        spec,
        graph_root=graph_root,
        catalog_root=catalog_root,
    )
    assert loaded["graph_root_binding"] == "/old/absolute/cluster/path"
    assert len(digest) == 64


def test_stage_reused_graphs_copies_only_graph_contract_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy" / "pubmed"
    destination = tmp_path / "staged"
    spec = replace(SPECS["pubmed"], percentages=(0, 5))
    (source / "baseline").mkdir(parents=True)
    (source / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    (source / "baseline" / "metadata.json").write_text("{}", encoding="utf-8")
    for replicate in NAMESPACE["REPLICATE_INDICES"]:
        pair = source / f"replicate-{replicate:04d}" / "pct-005"
        pair.mkdir(parents=True)
        for name in NAMESPACE["GRAPH_FILES"]:
            (pair / name).write_bytes(f"{replicate}:{name}".encode())
        (pair / "predictions.npz").write_bytes(b"must not be staged")

    function = NAMESPACE["stage_reused_graph_dataset"]
    monkeypatch.setitem(
        function.__globals__,
        "_validate_graph_manifest",
        lambda *_args, **_kwargs: ({}, "a" * 64),
    )
    monkeypatch.setitem(
        function.__globals__,
        "_validate_source_baseline",
        lambda *_args, **_kwargs: {},
    )
    function(
        Namespace(source_dataset_root=source, graph_root=destination),
        spec,
    )
    staged = destination / "pubmed"
    assert not list(staged.rglob("predictions.npz"))
    assert len(list(staged.rglob("graphs.npz"))) == 5
    assert len(list(staged.rglob("swaps.json"))) == 5
    assert len(list(staged.rglob("metadata.json"))) == 6


def test_cli_accepts_frozen_revision_for_later_validation() -> None:
    args = NAMESPACE["parse_args"](
        [
            "validate",
            "--graph-root",
            "graphs",
            "--catalog-root",
            "catalogs",
            "--checkpoint-root",
            "checkpoints",
            "--output-root",
            "outputs",
            "--experiment-revision",
            "f66a5399",
            "--release-provenance-manifest",
            "release.json",
        ]
    )
    assert args.experiment_revision == "f66a5399"
    assert args.release_provenance_manifest == Path("release.json")


def test_tracked_release_provenance_is_complete_and_locks_are_exact() -> None:
    import hashlib
    import json

    path = REPO_ROOT / "manifests" / "edge-dose-response" / "release-provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["experiment_revision"] == (
        "f66a53999fc01ae52a1e4aeaaf618409abab8fa7"
    )
    assert provenance["release_validation"]["n_checkpoint_shards"] == 720
    assert provenance["release_validation"]["n_paired_rows"] == 66_600
    cells = {
        (record["dataset"], record["model"])
        for record in provenance["accepted_cells"]
    }
    assert cells == set(NAMESPACE["MODEL_DATASET_CELLS"])
    assert all(record["checkpoint_shards"] == 30 for record in provenance["accepted_cells"])
    assert provenance["excluded_outputs"] == [
        {
            "job": "runtime-04",
            "dataset": "arxiv-year",
            "model": "acm-gcn",
            "reason": (
                "20 partial shards belonged to a failed run and were deleted after "
                "the complete replacement result passed"
            ),
            "present_in_release": False,
        }
    ]
    for record in provenance["runtime_environments"].values():
        lock = REPO_ROOT / record["requirements"]
        assert hashlib.sha256(lock.read_bytes()).hexdigest() == record[
            "requirements_sha256"
        ]


def test_accepted_validation_record_is_checked_by_exact_hash(tmp_path: Path) -> None:
    import hashlib
    import json

    output_root = tmp_path / "outputs"
    validation_path = output_root / "final" / "validation.json"
    validation_path.parent.mkdir(parents=True)
    original = b'{"validated": true}\n'
    validation_path.write_bytes(original)
    provenance_path = tmp_path / "release.json"
    provenance_path.write_text(
        json.dumps(
            {
                "release_validation": {
                    "validation_json_sha256": hashlib.sha256(original).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        output_root=output_root,
        release_provenance_manifest=provenance_path,
    )
    loaded, digest = NAMESPACE["load_authenticated_validation"](args)
    assert loaded == {"validated": True}
    assert digest == hashlib.sha256(original).hexdigest()

    validation_path.write_bytes(b'{"validated": false}\n')
    with pytest.raises(RuntimeError, match="SHA256 differs from accepted release"):
        NAMESPACE["load_authenticated_validation"](args)


def test_release_authentication_does_not_rewrite_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    output_root = tmp_path / "outputs"
    validation_path = output_root / "final" / "validation.json"
    validation_path.parent.mkdir(parents=True)
    accepted = {
        "schema_version": 1,
        "design_id": NAMESPACE["DESIGN_ID"],
        "validated": True,
        "validated_at": "2026-08-28T13:28:59.030867+02:00",
        "n_model_dataset_cells": 24,
        "n_checkpoint_shards": 720,
        "n_baseline_rows": 720,
        "n_paired_rows": 66_600,
        "n_unique_treatment_prediction_keys": 66_600,
        "n_unique_control_prediction_keys": 66_600,
        "graph_catalog_sha256": {"pubmed": "a" * 64},
        "experiment_revision": "f66a5399",
        "validation_implementation_revision": "f66a5399",
        "output_sha256": {
            "baseline_rows.jsonl": "b" * 64,
            "paired_rows.jsonl": "c" * 64,
        },
    }
    original = (json.dumps(accepted, indent=2, sort_keys=True) + "\n").encode()
    validation_path.write_bytes(original)
    provenance_path = tmp_path / "release.json"
    provenance_path.write_text(
        json.dumps(
            {
                "release_validation": {
                    "validation_json_sha256": hashlib.sha256(original).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        output_root=output_root,
        release_provenance_manifest=provenance_path,
    )
    calls: list[bool] = []

    def read_only_collect(_args: Namespace, *, write_outputs: bool = True) -> dict:
        calls.append(write_outputs)
        return {
            key: value
            for key, value in accepted.items()
            if key not in {"validated_at", "validation_implementation_revision"}
        }

    function = NAMESPACE["authenticate_existing_release"]
    monkeypatch.setitem(
        function.__globals__, "collect_and_validate_outputs", read_only_collect
    )
    monkeypatch.setitem(
        function.__globals__, "verify_release_provenance", lambda *_args: None
    )
    result = function(args)
    assert result == accepted
    assert calls == [False]
    assert validation_path.read_bytes() == original


@pytest.mark.parametrize(
    (
        "accepted_revision",
        "pinned_revision",
        "requested_revision",
        "recomputed_revision",
        "should_pass",
    ),
    (
        (
            "<absent>",
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            True,
        ),
        (
            "<absent>",
            "wrong-manifest",
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            False,
        ),
        (
            "<absent>",
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            "wrong-revalidation",
            False,
        ),
        ("<absent>", "other", "other", "other", False),
        (
            "wrong-validation",
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            False,
        ),
        (
            None,
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            ACCEPTED_RELEASE_REVISION,
            False,
        ),
    ),
)
def test_legacy_validation_revision_omission_requires_three_way_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_revision: str | None,
    pinned_revision: str,
    requested_revision: str,
    recomputed_revision: str,
    should_pass: bool,
) -> None:
    import hashlib
    import json

    accepted = {
        "schema_version": 1,
        "design_id": NAMESPACE["DESIGN_ID"],
        "validated": True,
        "n_model_dataset_cells": 24,
        "n_checkpoint_shards": 720,
        "n_baseline_rows": 720,
        "n_paired_rows": 66_600,
        "n_unique_treatment_prediction_keys": 66_600,
        "n_unique_control_prediction_keys": 66_600,
        "graph_catalog_sha256": {"pubmed": "a" * 64},
        "output_sha256": {
            "baseline_rows.jsonl": "b" * 64,
            "paired_rows.jsonl": "c" * 64,
        },
    }
    if accepted_revision != "<absent>":
        accepted["experiment_revision"] = accepted_revision
    recomputed = dict(accepted)
    recomputed["experiment_revision"] = recomputed_revision

    output_root = tmp_path / "outputs"
    validation_path = output_root / "final" / "validation.json"
    validation_path.parent.mkdir(parents=True)
    original = (json.dumps(accepted, indent=2, sort_keys=True) + "\n").encode()
    validation_path.write_bytes(original)
    provenance_path = tmp_path / "release.json"
    provenance_path.write_text(
        json.dumps(
            {
                "experiment_revision": pinned_revision,
                "release_validation": {
                    "validation_json_sha256": hashlib.sha256(original).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        code_version=requested_revision,
        output_root=output_root,
        release_provenance_manifest=provenance_path,
    )
    function = NAMESPACE["authenticate_existing_release"]
    monkeypatch.setitem(
        function.__globals__,
        "collect_and_validate_outputs",
        lambda *_args, **_kwargs: recomputed,
    )
    monkeypatch.setitem(
        function.__globals__, "verify_release_provenance", lambda *_args: None
    )

    if should_pass:
        assert function(args) == accepted
    else:
        with pytest.raises(
            RuntimeError,
            match="accepted validation record differs from read-only revalidation",
        ):
            function(args)
    assert validation_path.read_bytes() == original
