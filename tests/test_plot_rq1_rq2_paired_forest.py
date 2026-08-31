from pathlib import Path
import runpy

import pytest


pytest.importorskip("seaborn")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plot_rq1_rq2_paired_forest.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


def load_cells():
    return NAMESPACE["load_paired_cells"](
        REPO_ROOT
        / "results"
        / "main"
        / "aggregates"
        / "erm_tail_gaps.csv",
        REPO_ROOT
        / "results"
        / "main"
        / "rq2"
        / "class_adjusted_cell_summary.csv",
    )


def test_cli_defaults_target_the_current_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        NAMESPACE["sys"],
        "argv",
        ["plot_rq1_rq2_paired_forest.py"],
    )
    args = NAMESPACE["parse_args"]()

    assert args.output_dir == (
        REPO_ROOT / "results" / "main" / "figures" / "rq1-rq2" / "unequal-y"
    )
    assert args.figure_stem == (
        "rq1_rq2_unadjusted_and_class_adjusted_macro_f1_tail_gaps_unequal_y"
    )
    assert args.y_scale == "dataset"
    assert args.dpi == 400

    readme = tmp_path / "README.md"
    NAMESPACE["write_readme"](
        readme,
        args.figure_stem,
        args.y_scale,
    )
    assert (
        "python scripts/plot_rq1_rq2_paired_forest.py\n"
        in readme.read_text(encoding="utf-8")
    )


def test_paired_panel_is_complete_and_ordered() -> None:
    cells = load_cells()

    assert len(cells) == 35
    assert [
        (cell.dataset, cell.model)
        for cell in cells
    ] == [
        (dataset, model)
        for dataset in NAMESPACE["DATASET_ORDER"]
        for model in NAMESPACE["MODEL_ORDER"]
    ]
    assert all(cell.n_runs == 30 for cell in cells)


def test_zero_inclusion_counts_match_current_tables() -> None:
    cells = load_cells()

    assert sum(
        not cell.unadjusted.ci_excludes_zero for cell in cells
    ) == 3
    assert sum(
        not cell.class_adjusted.ci_excludes_zero for cell in cells
    ) == 1


def test_figure_is_compact_native_paper_width_and_legible() -> None:
    assert NAMESPACE["FIGURE_WIDTH_IN"] == 5.5
    assert 3.0 <= NAMESPACE["FIGURE_HEIGHT_IN"] <= 3.3
    assert (
        NAMESPACE["FIGURE_WIDTH_IN"] / NAMESPACE["FIGURE_HEIGHT_IN"]
    ) >= 1.7
    assert NAMESPACE["MINIMUM_TEXT_SIZE_PT"] >= 8.0
    assert NAMESPACE["MODEL_TEXT_SIZE_PT"] >= 8.0
    assert NAMESPACE["LEGEND_TEXT_SIZE_PT"] >= 7.25
    assert NAMESPACE["MARKER_SIZE_PT"] >= 3.5
    assert NAMESPACE["CI_LINE_WIDTH_PT"] >= 0.7


def test_shared_y_scale_contains_every_interval() -> None:
    cells = load_cells()
    all_bounds = [
        bound
        for cell in cells
        for estimate in (cell.unadjusted, cell.class_adjusted)
        for bound in (estimate.ci_lower, estimate.ci_upper)
    ]

    assert min(all_bounds) > -NAMESPACE["GAP_LIMIT"]
    assert max(all_bounds) < NAMESPACE["GAP_LIMIT"]


def test_dataset_specific_y_scales_contain_every_interval_and_zero() -> None:
    cells = load_cells()
    NAMESPACE["validate_y_axis_specifications"](cells, "dataset")

    shared_span = 2.0 * NAMESPACE["GAP_LIMIT"]
    for dataset in NAMESPACE["DATASET_ORDER"]:
        limits, ticks = NAMESPACE["y_axis_specification"](
            dataset,
            "dataset",
        )
        assert limits[0] < 0.0 < limits[1]
        assert 0.0 in ticks
        assert limits[1] - limits[0] < shared_span


def test_connectors_are_secondary_to_the_zero_reference() -> None:
    assert NAMESPACE["CONNECTOR_ALPHA"] <= 0.30
    assert NAMESPACE["CONNECTOR_LINE_WIDTH_PT"] <= 0.60
    assert (
        NAMESPACE["ZERO_LINE_WIDTH_PT"]
        >= 2.0 * NAMESPACE["CONNECTOR_LINE_WIDTH_PT"]
    )


def test_layout_matches_the_rq1_five_panel_composition() -> None:
    assert NAMESPACE["DATASET_LAYOUT"] == {
        "pubmed": (0, 0),
        "ogbn-arxiv": (0, 1),
        "roman-empire": (1, 0),
        "amazon-ratings": (1, 1),
        "arxiv-year": (1, 2),
    }
    assert NAMESPACE["LEGEND_CELL"] == (0, 2)
    occupied_cells = set(NAMESPACE["DATASET_LAYOUT"].values())
    assert len(occupied_cells) == 5
    assert NAMESPACE["LEGEND_CELL"] not in occupied_cells


def test_additional_layout_and_automatic_y_axis_are_registered() -> None:
    datasets, layout, legend_cell, rows, height = NAMESPACE["panel_spec"](
        "additional"
    )
    assert datasets == ("actor", "blogcatalog")
    assert layout == {"actor": (0, 0), "blogcatalog": (0, 1)}
    assert legend_cell == (0, 2)
    assert rows == 1
    assert height < NAMESPACE["FIGURE_HEIGHT_IN"]

    interval = NAMESPACE["IntervalEstimate"](
        mean=0.08,
        ci_lower=0.03,
        ci_upper=0.14,
        ci_excludes_zero=True,
    )
    cell = NAMESPACE["PairedGapCell"](
        dataset="actor",
        dataset_order=0,
        tail_side="high-h_v",
        model="mlp",
        model_order=0,
        n_runs=30,
        unadjusted=interval,
        class_adjusted=interval,
    )
    limits, ticks = NAMESPACE["y_axis_specification"](
        "actor", "dataset", [cell]
    )
    assert limits[0] < 0.0 < limits[1]
    assert limits[1] > interval.ci_upper
    assert any(abs(tick) < 1e-12 for tick in ticks)


def test_model_encoding_matches_current_rq1_figure() -> None:
    assert NAMESPACE["MODEL_MARKERS"] == {
        "mlp": "o",
        "gcn": "s",
        "graphsage-mean": "^",
        "h2gcn": "D",
        "gpr-gnn": "P",
        "linkx": "X",
        "acm-gcn": "v",
    }
    jitters = NAMESPACE["model_jitters"]()
    assert list(jitters) == list(NAMESPACE["MODEL_ORDER"])
    assert len(set(jitters.values())) == len(NAMESPACE["MODEL_ORDER"])
    assert sum(jitters.values()) == pytest.approx(0.0)


def test_legend_uses_the_shared_models_and_other_language() -> None:
    handles = NAMESPACE["other_legend_handles"]()
    assert [handle.get_label() for handle in handles] == [
        "95% CI",
        "Unfilled: CI includes 0",
        "Zero gap",
    ]
