"""Small synthetic checks for reverse-inference aggregation and portrait plots."""

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
from statistics import fmean, stdev
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_reverse_edge_dose_response.py"
SPEC = importlib.util.spec_from_file_location("analyze_reverse_edge_dose_response", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def implementation(value):
    return {"scripts/runner.py": digest(value)}


def test_current_implementation_provenance_binds_validator_and_producers():
    validator = implementation("validator")
    producers = sorted(
        [implementation("producer-one"), implementation("producer-two")],
        key=analysis._implementation_signature,
    )
    result = analysis._validate_implementation_provenance(
        {
            "validator_implementation_sha256": validator,
            "producer_implementation_sha256": producers,
        },
        [producers[1], producers[0], producers[1]],
        validator,
    )
    assert result == {
        "validator_implementation_sha256": validator,
        "producer_implementation_sha256": producers,
        "implementation_provenance_verified": True,
    }


@pytest.mark.parametrize("field", ["validator_implementation_sha256", "producer_implementation_sha256"])
def test_current_implementation_provenance_rejects_missing_fields(field):
    validator = implementation("validator")
    producer = implementation("producer")
    validation = {
        "validator_implementation_sha256": validator,
        "producer_implementation_sha256": [producer],
    }
    del validation[field]
    with pytest.raises(analysis.ValidationError):
        analysis._validate_implementation_provenance(validation, [producer], validator)


def test_current_implementation_provenance_rejects_changed_validator_hash():
    validator = implementation("validator")
    producer = implementation("producer")
    with pytest.raises(analysis.ValidationError):
        analysis._validate_implementation_provenance(
            {
                "validator_implementation_sha256": implementation("changed-validator"),
                "producer_implementation_sha256": [producer],
            },
            [producer],
            validator,
        )


def test_current_implementation_provenance_rejects_changed_producer_hash():
    validator = implementation("validator")
    producer = implementation("producer")
    with pytest.raises(analysis.ValidationError):
        analysis._validate_implementation_provenance(
            {
                "validator_implementation_sha256": validator,
                "producer_implementation_sha256": [implementation("changed-producer")],
            },
            [producer],
            validator,
        )


def test_current_implementation_provenance_rejects_noncanonical_or_duplicate_producers():
    validator = implementation("validator")
    producers = sorted(
        [implementation("producer-one"), implementation("producer-two")],
        key=analysis._implementation_signature,
    )
    for invalid in (list(reversed(producers)), [producers[0], producers[0]]):
        with pytest.raises(analysis.ValidationError):
            analysis._validate_implementation_provenance(
                {
                    "validator_implementation_sha256": validator,
                    "producer_implementation_sha256": invalid,
                },
                producers,
                validator,
            )


def test_inference_validation_context_checks_optional_source_manifest():
    validation = {
        "schema_version": 1,
        "experiment_revision": "1" * 40,
        "input_snapshot_sha256": "2" * 64,
        "source_hash_manifest_sha256": "3" * 64,
    }
    assert analysis._inference_validation_context(validation) == {
        "inference_schema_version": 1,
        "experiment_revision": "1" * 40,
        "input_snapshot_sha256": "2" * 64,
        "source_hash_manifest_sha256": "3" * 64,
    }
    del validation["source_hash_manifest_sha256"]
    assert analysis._inference_validation_context(validation)["source_hash_manifest_sha256"] is None
    validation["source_hash_manifest_sha256"] = None
    assert analysis._inference_validation_context(validation)["source_hash_manifest_sha256"] is None
    validation["source_hash_manifest_sha256"] = "invalid"
    with pytest.raises(analysis.ValidationError):
        analysis._inference_validation_context(validation)
    validation["source_hash_manifest_sha256"] = None
    del validation["input_snapshot_sha256"]
    with pytest.raises(analysis.ValidationError):
        analysis._inference_validation_context(validation)


@pytest.mark.parametrize("value", ["v100-cuda", "a40-cuda-arm64", "backend1"])
def test_backend_identifier_accepts_portable_slugs(value):
    assert analysis._backend_id(value) == value


@pytest.mark.parametrize("value", ["-v100", "v100-", "v100--cuda", "V100-cuda", "a40_cuda"])
def test_backend_identifier_rejects_non_slugs(value):
    with pytest.raises(analysis.ValidationError):
        analysis._backend_id(value)


def fixture(preparation="erm"):
    dataset, model = "pubmed", "gcn"
    doses = ((134, 100 * 134 / 2696), (175, 100 * 175 / 2696))
    graph_pairs = {
        (count, replicate): {
            "replicate_seed": 2026080100 + replicate,
            "source_graphs_sha256": digest((count, replicate, "archive")),
            "treatment_edge_sha256": digest((count, replicate, "treatment")),
            "control_edge_sha256": digest((count, replicate, "control")),
        }
        for count, _ in doses
        for replicate in range(1, 6)
    }
    contract = analysis.GraphContract(dataset, 2696, False, doses, (("split0", 1), ("split0", 2)), graph_pairs, digest("catalog"))
    baselines, pairs = [], []
    for split, seed in contract.checkpoint_identities:
        run_id = f"{dataset}_{model}_{preparation}_{split}_seed-{seed}"
        baseline = {
            "preparation": preparation, "dataset": dataset, "model": model,
            "split": split, "seed": seed, "run_id": run_id,
            "checkpoint_sha256": digest(run_id), "prediction_sha256": digest(run_id + "original"),
            "prediction_key": run_id + "__original", "tail_test_macro_f1": 0.5 + 0.1 * seed,
            "tail_test_accuracy": 0.6, "n_tail_test": 17, "tail_test_class_composition": {"0": 9, "1": 8},
        }
        baselines.append(baseline)
        for count, percentage in doses:
            for replicate in contract.replicates:
                pairing_key = f"reverse__{dataset}__replicate_{replicate:04d}__targets_{count:08d}"
                treatment = 0.3 + seed * 0.1 + replicate * 0.01
                control = 0.5 + seed * 0.1 + replicate * 0.02
                pair = {name: baseline[name] for name in ("preparation", "dataset", "model", "split", "seed", "run_id", "checkpoint_sha256", "n_tail_test", "tail_test_class_composition")}
                pair.update({
                    "pairing_key": pairing_key, "target_count": count, "n_selected_tail_nodes": count,
                    "original_tail_count": 2696, "desired_same": False,
                    "p_percent": percentage, "actual_pct": percentage, "replicate_index": replicate,
                    **graph_pairs[(count, replicate)],
                    "treatment_macro_f1": treatment, "control_macro_f1": control,
                    "treatment_minus_control_macro_f1": treatment - control,
                    "treatment_accuracy": treatment, "control_accuracy": control,
                    "treatment_minus_control_accuracy": treatment - control,
                })
                for condition in analysis.CONDITIONS:
                    pair[condition + "_prediction_key"] = run_id + "__" + pairing_key + "__" + condition
                    pair[condition + "_prediction_sha256"] = digest((run_id, count, replicate, condition))
                pairs.append(pair)
    return baselines, pairs, {dataset: contract}


@pytest.mark.parametrize("preparation", analysis.PREPARATIONS)
def test_every_preparation_uses_its_own_complete_new_rows(preparation):
    baselines, pairs, contracts = fixture(preparation)
    result = analysis.validate_rows(baselines, pairs, preparation, contracts, models=("gcn",))
    assert result["n_checkpoint_shards"] == 2
    assert result["n_baseline_rows"] == 2
    assert result["n_paired_rows"] == 20
    assert result["pairing_validated"] is True


def test_aggregation_keeps_fractional_original_tail_percentages():
    baselines, pairs, contracts = fixture()
    replicates, summaries = analysis.aggregate_rows(baselines, pairs, "erm", contracts, models=("gcn",))
    assert len(replicates) == 30
    assert len(summaries) == 6
    assert sorted({row["actual_pct"] for row in summaries}) == [0, 100 * 134 / 2696, 100 * 175 / 2696]
    treatment = next(row for row in summaries if row["condition"] == "treatment" and row["target_count"] == 175)
    replicate_means = [fmean((0.3 + 0.1 + r * 0.01, 0.3 + 0.2 + r * 0.01)) for r in range(1, 6)]
    assert treatment["mean_macro_f1"] == pytest.approx(fmean(replicate_means))
    assert treatment["sd_macro_f1_across_graph_replicates"] == pytest.approx(stdev(replicate_means))
    for row in summaries:
        if row["target_count"] == 0:
            assert row["mean_macro_f1"] == pytest.approx(0.65)
            assert row["sd_macro_f1_across_graph_replicates"] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_checkpoint", "wrong_direction", "eligible_denominator", "wrong_graph", "wrong_condition_key", "wrong_population", "invalid_score", "wrong_difference", "wrong_preparation"])
def test_row_validation_rejects_incomplete_or_inconsistent_data(mutation):
    baselines, pairs, contracts = fixture()
    if mutation == "missing":
        pairs.pop()
    elif mutation == "duplicate":
        pairs[-1] = deepcopy(pairs[0])
    elif mutation == "wrong_checkpoint":
        pairs[0]["checkpoint_sha256"] = digest("another checkpoint")
    elif mutation == "wrong_direction":
        pairs[0]["desired_same"] = True
    elif mutation == "eligible_denominator":
        pairs[0]["p_percent"] = pairs[0]["actual_pct"] = 100 * 134 / 175
    elif mutation == "wrong_graph":
        pairs[0]["source_graphs_sha256"] = digest("another graph")
    elif mutation == "wrong_condition_key":
        pairs[0]["treatment_prediction_key"] = pairs[0]["control_prediction_key"]
    elif mutation == "wrong_population":
        pairs[0]["n_tail_test"] = 16
    elif mutation == "invalid_score":
        pairs[0]["treatment_macro_f1"] = float("nan")
    elif mutation == "wrong_difference":
        pairs[0]["treatment_minus_control_macro_f1"] = 0.9
    elif mutation == "wrong_preparation":
        baselines[0]["preparation"] = "last"
    with pytest.raises(analysis.ValidationError):
        analysis.validate_rows(baselines, pairs, "erm", contracts, models=("gcn",))


def test_checkpoint_panel_is_exact_not_just_thirty_arbitrary_runs():
    baselines, pairs, contracts = fixture()
    baselines[0]["seed"] = 99
    with pytest.raises(analysis.ValidationError):
        analysis.validate_rows(baselines, pairs, "erm", contracts, models=("gcn",))


def test_five_graph_replicates_required_in_every_checkpoint():
    baselines, pairs, contracts = fixture()
    pairs[-1]["replicate_index"] = 6
    with pytest.raises(analysis.ValidationError):
        analysis.validate_rows(baselines, pairs, "erm", contracts, models=("gcn",))


def test_portable_paths_reject_absolute_or_parent_paths(tmp_path):
    for name in ("../bad", "/absolute/bad", "C:/absolute/bad", "x\\bad"):
        with pytest.raises(analysis.ValidationError):
            analysis._safe_relative(tmp_path, name)
    assert analysis._safe_relative(tmp_path, "relative/file.json") == tmp_path / "relative" / "file.json"


def plot_fixture():
    counts = {"pubmed": (2696, 175), "ogbn-arxiv": (30364, 9565), "roman-empire": (1012, 943), "amazon-ratings": (2943, 2319), "arxiv-year": (21869, 4670)}
    contracts, summaries = {}, []
    for dataset, (tail, eligible) in counts.items():
        doses = ((max(1, tail * 5 // 100), 100 * (tail * 5 // 100) / tail), (eligible, 100 * eligible / tail))
        contracts[dataset] = analysis.GraphContract(dataset, tail, dataset not in {"pubmed", "ogbn-arxiv"}, doses, (), {}, digest(dataset))
        for model in analysis.MODELS:
            for target_count, percentage in ((0, 0.0), *doses):
                for condition in analysis.CONDITIONS:
                    summaries.append({"preparation": "erm", "dataset": dataset, "model": model, "target_count": target_count, "actual_pct": percentage, "condition": condition, "n_graph_replicates": 5, "n_checkpoints_per_graph": 30, "mean_macro_f1": 0.65 + percentage * (0.001 if condition == "control" else -0.001), "sd_macro_f1_across_graph_replicates": 0 if target_count == 0 else 0.01})
    return contracts, summaries


def test_portrait_plot_has_dataset_specific_axes_and_vertical_model_names():
    import matplotlib.pyplot as plt

    contracts, summaries = plot_fixture()
    figure = analysis.build_preparation_figure(summaries, contracts)
    assert tuple(figure.get_size_inches()) == pytest.approx((5.5, 6.5))
    assert len(figure.axes) == 30
    for row in range(6):
        for column, dataset in enumerate(analysis.DATASETS):
            axis = figure.axes[row * 5 + column]
            assert axis.get_xlim() == pytest.approx((0, contracts[dataset].doses[-1][1]))
            assert max(axis.lines[0].get_xdata()) == pytest.approx(contracts[dataset].doses[-1][1])
    assert not figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[1])
    assert figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[5])
    assert sum(text.get_rotation() == 90 for text in figure.texts) >= 6
    assert figure._supxlabel.get_text() == "Original tail nodes rewired (%)"
    plt.close(figure)


def test_existing_plot_default_remains_zero_to_one_hundred():
    import matplotlib.pyplot as plt
    from plot_edge_dose_response_composite import SummaryRow, build_figure

    _, summaries = plot_fixture()
    rows = [SummaryRow(row["dataset"], row["model"], row["actual_pct"], row["condition"], 5, 30, row["mean_macro_f1"], row["sd_macro_f1_across_graph_replicates"]) for row in summaries]
    figure = build_figure(rows)
    assert all(axis.get_xlim() == (0, 100) for axis in figure.axes)
    assert figure._supxlabel.get_text() == "Tail nodes rewired (%)"
    plt.close(figure)


def test_pdf_smoke_preserves_portrait_size_and_labels(tmp_path):
    import matplotlib.pyplot as plt

    pytest.importorskip("pypdf")
    contracts, summaries = plot_fixture()
    figure = analysis.build_preparation_figure(summaries, contracts)
    path = tmp_path / "reverse-plot-smoke.pdf"
    figure.savefig(path, metadata={"Author": "Anonymous", "CreationDate": None, "ModDate": None})
    plt.close(figure)
    result = analysis.validate_pdf(path)
    assert result["portrait"] is True
    assert result["required_labels_present"] is True
    assert result["pages"] == 1
