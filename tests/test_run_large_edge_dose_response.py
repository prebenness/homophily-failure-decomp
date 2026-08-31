"""Focused tests for the large-graph edge dose-response runner."""

from __future__ import annotations

from argparse import Namespace
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_large_edge_dose_response.py"
NAMESPACE = runpy.run_path(str(SCRIPT))
SPECS = NAMESPACE["DATASET_SPECS"]


def test_dataset_contracts_are_exact() -> None:
    arxiv = SPECS["arxiv-year"]
    ogbn = SPECS["ogbn-arxiv"]

    assert arxiv.tail_count == 21869
    assert arxiv.tail_side == "high-h_v"
    assert arxiv.desired_same is False
    assert arxiv.generator == "singleton"
    assert arxiv.split_ids == tuple(f"lim-{index}" for index in range(5))
    assert arxiv.training_seeds == tuple(range(1, 7))
    assert len(arxiv.checkpoint_identities) == 30

    assert ogbn.tail_count == 30364
    assert ogbn.tail_side == "low-h_v"
    assert ogbn.desired_same is True
    assert ogbn.generator == "coordinated"
    assert ogbn.split_ids == ("ogb-time",)
    assert ogbn.training_seeds == tuple(range(1, 31))
    assert len(ogbn.checkpoint_identities) == 30


@pytest.mark.parametrize(
    ("tail_count", "expected"),
    (
        (21869, {0: 0, 5: 1093, 50: 10934, 100: 21869}),
        (30364, {0: 0, 5: 1518, 50: 15182, 100: 30364}),
    ),
)
def test_target_counts_use_registered_floor_rule(
    tail_count: int, expected: dict[int, int]
) -> None:
    count = NAMESPACE["target_count_for_percentage"]
    for percentage, wanted in expected.items():
        assert count(tail_count, percentage) == wanted
    with pytest.raises(ValueError, match="p_percent"):
        count(tail_count, 7)


def test_replicate_ranking_gives_nested_target_prefixes() -> None:
    tail_mask = np.zeros(20, dtype=bool)
    tail_mask[[1, 3, 5, 7, 9, 11, 13, 15]] = True
    ranking = NAMESPACE["tail_ranking"](tail_mask, 101)
    repeated = NAMESPACE["tail_ranking"](tail_mask, 101)
    targets_25 = NAMESPACE["targets_for_percentage"](ranking, 25)
    targets_50 = NAMESPACE["targets_for_percentage"](ranking, 50)
    targets_100 = NAMESPACE["targets_for_percentage"](ranking, 100)

    assert np.array_equal(ranking, repeated)
    assert set(ranking) == set(np.flatnonzero(tail_mask))
    assert np.array_equal(targets_25, targets_50[: targets_25.size])
    assert np.array_equal(targets_50, targets_100[: targets_50.size])
    assert set(targets_100) == set(np.flatnonzero(tail_mask))


@pytest.mark.parametrize("dataset", ("arxiv-year", "ogbn-arxiv"))
def test_frozen_manifest_records_dataset_contract(dataset: str) -> None:
    spec = SPECS[dataset]
    args = Namespace(code_version="abc123")
    identities = list(spec.checkpoint_identities)
    manifest = NAMESPACE["frozen_manifest"](
        args, spec, identities, spec.default_seed_base
    )

    assert manifest["dataset"] == dataset
    assert manifest["models"] == ["gcn", "graphsage-mean"]
    assert manifest["p_percentages"] == list(range(0, 101, 5))
    assert manifest["graph_replicates"] == 5
    assert manifest["graph_replicate_indices"] == [1, 2, 3, 4, 5]
    assert manifest["tail_side"] == spec.tail_side
    assert manifest["tail_count"] == spec.tail_count
    assert manifest["desired_same"] is spec.desired_same
    assert manifest["generator"] == spec.generator
    assert manifest["checkpoint_identities"] == [
        {"split": split, "seed": seed} for split, seed in identities
    ]
    assert manifest["n_checkpoints_per_model"] == 30
    assert manifest["target_sets"] == (
        "nested prefixes of the replicate target ranking"
    )
    assert manifest["graph_start_at_each_p"] == (
        f"registered original {dataset} graph"
    )
    assert manifest["metric_population"] == "fixed-full-tail-test"
    assert manifest["actual_generation_attempts"] == 1


def test_pair_output_path_encodes_dataset_replicate_and_percentage() -> None:
    path = NAMESPACE["pair_output_path"](
        Path("output") / "actual",
        "arxiv-year",
        replicate_index=4,
        p_percent=35,
    )
    assert path.as_posix().endswith(
        "actual/arxiv-year/replicate-0004/pct-035"
    )


@pytest.mark.parametrize("dataset", ("arxiv-year", "ogbn-arxiv"))
def test_actual_manifest_is_written_below_dataset_root(
    tmp_path: Path, dataset: str
) -> None:
    spec = SPECS[dataset]
    args = Namespace(
        output_root=tmp_path,
        phase="actual",
        code_version="abc123",
    )
    digest = NAMESPACE["prepare_actual_experiment"](
        args,
        spec,
        spec.checkpoint_identities,
        spec.default_seed_base,
    )
    path = tmp_path / "actual" / dataset / "experiment_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert len(digest) == 64
    assert manifest["dataset"] == dataset
    assert manifest["replicate_seed_base"] == spec.default_seed_base
    assert manifest["actual_generation_attempts"] == 1


def test_checkpoint_specs_validate_parent_runs_once_and_stay_lightweight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_validate(
        run_dir: Path,
        *,
        dataset: str,
        model: str,
        split: str,
        seed: int,
        fixed_input_checksums: dict,
    ):
        assert dataset == "arxiv-year"
        assert fixed_input_checksums == {"tail_mask": "registered"}
        calls.append((model, split, seed))
        return (
            {
                "model_config": {"model": model},
                "code_version": "parent-code",
            },
            run_dir / "checkpoint.pt",
            np.asarray([0, 1, 0], dtype=np.int64),
        )

    prepare = NAMESPACE["prepare_checkpoint_specs"]
    monkeypatch.setitem(
        prepare.__globals__, "validate_parent_run", fake_validate
    )
    monkeypatch.setitem(
        prepare.__globals__, "file_sha256", lambda path: "a" * 64
    )
    checkpoint_specs, originals = prepare(
        spec=SPECS["arxiv-year"],
        models=("gcn", "graphsage-mean"),
        identities=(("lim-0", 1), ("lim-0", 2)),
        checkpoint_root=Path("parents"),
        fixed_input_checksums={"tail_mask": "registered"},
        num_nodes=3,
    )

    assert calls == [
        ("gcn", "lim-0", 1),
        ("gcn", "lim-0", 2),
        ("graphsage-mean", "lim-0", 1),
        ("graphsage-mean", "lim-0", 2),
    ]
    assert len(checkpoint_specs) == 4
    assert len(originals) == 4
    assert all(not hasattr(spec, "saved_original") for spec in checkpoint_specs)

    pair_source = inspect.getsource(NAMESPACE["evaluate_pair"])
    assert "validate_parent_run" not in pair_source
    assert "run_data.npz" not in pair_source


def baseline_fixture(
    tmp_path: Path,
) -> tuple[object, object, dict, object, np.ndarray]:
    checkpoint = NAMESPACE["CheckpointSpec"](
        model="gcn",
        split="ogb-time",
        seed=6,
        model_config={},
        checkpoint_path=tmp_path / "checkpoint.pt",
        run_data_path=tmp_path / "run_data.npz",
        checkpoint_sha256="a" * 64,
        training_code_version="parent-code",
    )
    bundle = SimpleNamespace(
        x=np.zeros((4, 2), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        num_classes=2,
        num_nodes=4,
    )
    arrays = {
        "ogb-time.test": np.asarray([0, 1], dtype=np.int64),
        "tail_mask": np.asarray([True, False, False, False]),
    }
    device = SimpleNamespace(type="cpu")
    saved_predictions = np.asarray([0, 0, 0, 0], dtype=np.int64)
    return checkpoint, bundle, arrays, device, saved_predictions


def test_baseline_accepts_numerically_reproduced_off_subset_argmax_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle, arrays, device, saved_predictions = baseline_fixture(
        tmp_path
    )
    saved_logits = np.asarray(
        [[2.0, 1.0], [2.0, 1.0], [1.0000005, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    current_logits = saved_logits.copy()
    current_logits[2] = [1.0, 1.0000005]
    np.savez_compressed(checkpoint.run_data_path, logits=saved_logits)

    evaluate = NAMESPACE["evaluate_baseline"]
    monkeypatch.setitem(
        evaluate.__globals__,
        "load_network",
        lambda *args, **kwargs: (SimpleNamespace(), {}),
    )
    monkeypatch.setitem(
        evaluate.__globals__,
        "predict_logits",
        lambda *args, **kwargs: current_logits,
    )

    rows, predictions = evaluate(
        spec=SPECS["ogbn-arxiv"],
        bundle=bundle,
        arrays=arrays,
        checkpoint_specs=[checkpoint],
        saved_originals={("gcn", "ogb-time", 6): saved_predictions},
        x=SimpleNamespace(),
        original_edges=np.empty((2, 0), dtype=np.int64),
        device=device,
        pause_seconds=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["n_full_graph_prediction_differences_from_parent"] == 1
    assert rows[0]["n_tail_test_prediction_differences_from_parent"] == 0
    assert rows[0]["full_graph_prediction_difference_nodes_from_parent"] == "[2]"
    assert rows[0]["full_graph_logit_reproduction_checked"] is True
    assert predictions["gcn__ogb_time__seed_6__original"][2] == 1


def test_baseline_rejects_argmax_flip_on_fixed_tail_test_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, bundle, arrays, device, saved_predictions = baseline_fixture(
        tmp_path
    )
    current_logits = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    evaluate = NAMESPACE["evaluate_baseline"]
    monkeypatch.setitem(
        evaluate.__globals__,
        "load_network",
        lambda *args, **kwargs: (SimpleNamespace(), {}),
    )
    monkeypatch.setitem(
        evaluate.__globals__,
        "predict_logits",
        lambda *args, **kwargs: current_logits,
    )

    with pytest.raises(RuntimeError, match="fixed tail-test predictions fail"):
        evaluate(
            spec=SPECS["ogbn-arxiv"],
            bundle=bundle,
            arrays=arrays,
            checkpoint_specs=[checkpoint],
            saved_originals={("gcn", "ogb-time", 6): saved_predictions},
            x=SimpleNamespace(),
            original_edges=np.empty((2, 0), dtype=np.int64),
            device=device,
            pause_seconds=0.0,
        )


def test_registered_identity_panel_rejects_manifest_drift() -> None:
    spec = SPECS["ogbn-arxiv"]
    manifest = {
        "splits": [{"split_id": "ogb-time"}],
        "training_seeds": list(range(1, 31)),
        "n_runs": 30,
    }
    identities = NAMESPACE["registered_checkpoint_identities"](
        manifest, spec
    )
    assert identities == list(spec.checkpoint_identities)

    manifest["training_seeds"][-1] = 31
    with pytest.raises(RuntimeError, match="frozen dose-response panel"):
        NAMESPACE["registered_checkpoint_identities"](manifest, spec)


def test_generator_dispatch_uses_registered_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate = NAMESPACE["generate_pair"]
    bundle = SimpleNamespace(
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        y=np.asarray([0, 1], dtype=np.int64),
    )
    arrays = {"tail_mask": np.asarray([True, False])}
    targets = np.asarray([0], dtype=np.int64)
    sentinel = object()
    singleton_calls: list[dict] = []

    def fake_singleton(*args, **kwargs):
        singleton_calls.append(kwargs)
        return sentinel

    monkeypatch.setitem(
        generate.__globals__, "generate_rewire_pair", fake_singleton
    )
    result = generate(
        spec=SPECS["arxiv-year"],
        bundle=bundle,
        arrays=arrays,
        targets=targets,
        pair_seed=10,
        max_attempts=3,
    )
    assert result is sentinel
    assert len(singleton_calls) == 1
    assert singleton_calls[0]["desired_same"] is False
    assert singleton_calls[0]["pair_seed"] == 10
    assert singleton_calls[0]["target_nodes"] is targets
    assert singleton_calls[0]["max_attempts"] == 3

    import hfd.coordinated_edge_perturbation as coordinated

    coordinated_calls: list[dict] = []

    def fake_coordinated(*args, **kwargs):
        coordinated_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        coordinated, "generate_coordinated_rewire_pair", fake_coordinated
    )
    result = generate(
        spec=SPECS["ogbn-arxiv"],
        bundle=bundle,
        arrays=arrays,
        targets=targets,
        pair_seed=20,
        max_attempts=4,
    )
    assert result is sentinel
    assert len(coordinated_calls) == 1
    assert coordinated_calls[0]["target_nodes"] is targets
    assert coordinated_calls[0]["pair_seed"] == 20
    assert coordinated_calls[0]["max_attempts"] == 4
