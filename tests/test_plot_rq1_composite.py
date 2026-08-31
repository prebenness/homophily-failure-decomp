from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_rq1_composite.py"
)
sys.path.insert(0, str(SCRIPT.parent))
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_additional_panel_leaves_headroom_for_scores_at_one() -> None:
    assert NAMESPACE["score_axis_upper_limit"]("original") == 1.0
    assert NAMESPACE["score_axis_upper_limit"]("additional") == 1.06


def test_composite_uses_twelve_bins_by_default() -> None:
    assert NAMESPACE["DEFAULT_COMPOSITE_N_BINS"] == 12
    assert NAMESPACE["METRIC_CHOICES"] == ("accuracy", "macro-f1")
    assert NAMESPACE["DEFAULT_COMPOSITE_METRIC"] == "accuracy"
    assert NAMESPACE["COMPOSITE_CONFIDENCE_ALPHA"] == 0.20
    assert NAMESPACE["COMPOSITE_SUPPORT_FLOOR"] == 1.0
    assert NAMESPACE["SECONDARY_LEGEND_TITLE"] == "Other"
    assert NAMESPACE["PRIMARY_LEGEND_ANCHOR_Y"] > 1.0
    assert (
        NAMESPACE["PRIMARY_LEGEND_ANCHOR_Y"]
        - NAMESPACE["SECONDARY_LEGEND_ANCHOR_Y"]
    ) >= 1.0


def test_metric_output_stems_preserve_accuracy_and_add_macro_f1() -> None:
    build_stem = NAMESPACE["composite_output_stem"]
    assert build_stem("erm", 12, "accuracy") == (
        "rq1_erm_five-datasets_all-models_"
        "12-bin_accuracy_by_local_homophily_with-ci"
    )
    assert build_stem("erm", 12, "macro-f1") == (
        "rq1_erm_five-datasets_all-models_"
        "12-bin_macro_f1_by_local_homophily_with-ci"
    )
    assert NAMESPACE["metric_metadata_label"]("accuracy") == "accuracy"


def test_composite_layout_reserves_top_right_for_legend() -> None:
    assert NAMESPACE["DATASET_LAYOUT"] == {
        "pubmed": (0, 0),
        "ogbn-arxiv": (0, 1),
        "roman-empire": (1, 0),
        "amazon-ratings": (1, 1),
        "arxiv-year": (1, 2),
    }
    assert (0, 2) not in NAMESPACE["DATASET_LAYOUT"].values()


def test_additional_composite_uses_two_panels_and_a_separate_stem() -> None:
    datasets, layout, legend_cell, rows, height = NAMESPACE["panel_spec"](
        "additional"
    )

    assert datasets == ("actor", "blogcatalog")
    assert layout == {"actor": (0, 0), "blogcatalog": (0, 1)}
    assert legend_cell == (0, 2)
    assert rows == 1
    assert height < NAMESPACE["COMPOSITE_HEIGHT_IN"]
    assert NAMESPACE["composite_output_stem"](
        "erm", 12, "macro-f1", "additional"
    ) == (
        "rq1_erm_additional-datasets_all-models_"
        "12-bin_macro_f1_by_local_homophily_with-ci"
    )
    assert NAMESPACE["dataset_label"]("actor") == "Actor"
    assert NAMESPACE["dataset_label"]("blogcatalog") == "BlogCatalog"


def test_additional_composite_rejects_substituted_registered_run_id() -> None:
    registrations = [
        run
        for run in NAMESPACE["load_registered_runs"](
            NAMESPACE["REPO_ROOT"], "additional", ("actor",)
        )
        if run.model == "mlp" and run.preparation == "erm"
    ]
    records = [SimpleNamespace(run_id=run.run_id) for run in registrations]

    NAMESPACE["validate_registered_record_ids"](
        records, "actor", "mlp", "erm", "additional"
    )
    records[-1] = SimpleNamespace(
        run_id="actor_mlp_erm_geomgcn-999_seed-999"
    )
    with pytest.raises(RuntimeError, match="registered run ID mismatch"):
        NAMESPACE["validate_registered_record_ids"](
            records, "actor", "mlp", "erm", "additional"
        )


def test_composite_is_designed_at_native_paper_size() -> None:
    assert NAMESPACE["PAPER_TEXT_WIDTH_IN"] == 5.5
    assert 3.3 <= NAMESPACE["COMPOSITE_HEIGHT_IN"] <= 3.4
    assert (
        NAMESPACE["ACCURACY_PANEL_HEIGHT_RATIO"]
        / NAMESPACE["SUPPORT_PANEL_HEIGHT_RATIO"]
    ) <= 2.5
    assert NAMESPACE["MINIMUM_TEXT_SIZE_PT"] >= 8.0
    assert NAMESPACE["TICK_LABEL_SIZE_PT"] >= NAMESPACE["MINIMUM_TEXT_SIZE_PT"]
    assert NAMESPACE["LEGEND_TEXT_SIZE_PT"] >= NAMESPACE["MINIMUM_TEXT_SIZE_PT"]
    assert NAMESPACE["MODEL_LINE_WIDTH_PT"] >= 1.0
    assert 2.0 <= NAMESPACE["PLOT_MARKER_SIZE_PT"] <= 2.6
    assert NAMESPACE["LEGEND_MARKER_SIZE_PT"] > NAMESPACE["PLOT_MARKER_SIZE_PT"]


def test_composite_hides_support_below_one_node_per_run() -> None:
    reference = SimpleNamespace(
        centres=np.arange(4, dtype=np.float64),
        mean_run_support=np.array([0.0, 0.7, 1.0, 2.0]),
    )
    mask = NAMESPACE["composite_support_display_mask"](reference)
    assert mask.tolist() == [False, False, True, True]


def test_macro_f1_metric_view_uses_macro_f1_values_and_units() -> None:
    data = SimpleNamespace(
        centres=np.array([0.25, 0.75]),
        macro_f1_units=np.array([[0.2, 0.4], [0.3, np.nan]]),
        mean_macro_f1=np.array([0.25, 0.4]),
        macro_f1_ci_low=np.array([0.1, 0.2]),
        macro_f1_ci_high=np.array([0.4, 0.6]),
    )

    assert NAMESPACE["metric_slug"]("macro-f1") == "macro_f1"
    assert NAMESPACE["metric_label"]("macro-f1") == "Macro-F1"
    np.testing.assert_allclose(
        NAMESPACE["metric_mean"](data, "macro-f1"),
        [0.25, 0.4],
    )
    low, high = NAMESPACE["metric_interval"](data, "macro-f1")
    np.testing.assert_allclose(low, [0.1, 0.2])
    np.testing.assert_allclose(high, [0.4, 0.6])
    assert NAMESPACE["complete_metric_unit_display_mask"](
        data, "macro-f1"
    ).tolist() == [True, False]
