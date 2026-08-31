from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_rq5_preparations.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))


def make_complete_summaries():
    rows = []
    for dataset_index, dataset in enumerate(NAMESPACE["DATASET_ORDER"]):
        for model_index, model in enumerate(NAMESPACE["PLOT_MODEL_ORDER"]):
            for preparation_index, preparation in enumerate(
                NAMESPACE["PREPARATION_ORDER"]
            ):
                tail = (
                    0.20
                    + 0.08 * dataset_index
                    + 0.01 * model_index
                    + 0.005 * preparation_index
                )
                whole = tail + 0.12 - 0.01 * preparation_index
                rows.append(
                    NAMESPACE["ScoreSummary"](
                        dataset=dataset,
                        model=model,
                        preparation=preparation,
                        n_runs=30,
                        mean_tail_macro_f1=tail,
                        sample_std_tail_macro_f1=0.01,
                        mean_whole_test_macro_f1=whole,
                        sample_std_whole_test_macro_f1=0.01,
                        ci_lower=tail - 0.01,
                        ci_upper=tail + 0.01,
                        run_scores=(tail,) * 30,
                    )
                )
    return rows


def test_compact_figure_has_expected_panels_and_encoding() -> None:
    figure = NAMESPACE["build_tail_vs_whole_compact_figure"](
        make_complete_summaries()
    )
    try:
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "PubMed",
            "ogbn-arxiv",
            "Example for one model",
            "Roman-empire",
            "Amazon-ratings",
            "arxiv-year",
        ]
        assert len(figure.axes[0].lines) == 12
        assert figure.axes[0].lines[0].get_marker() == "o"
        assert figure.axes[0].lines[1].get_marker() == "s"
        assert figure.axes[0].lines[0].get_linestyle() == "-"
        assert figure.axes[0].lines[1].get_linestyle() == "-"
        assert tuple(figure.get_size_inches()) == pytest.approx((5.5, 2.95))
        assert figure.axes[0].lines[0].get_linewidth() == pytest.approx(0.60)
        assert figure.axes[0].lines[0].get_markersize() == pytest.approx(2.3)
        assert figure.axes[0].lines[1].get_markersize() == pytest.approx(2.4)
        assert figure.axes[0].get_xticklabels()[0].get_fontfamily() == ["Arial"]
        assert [
            label.get_text() for label in figure.axes[0].get_xticklabels()
        ] == ["GCN", "SAGE", "H2GCN", "GPR", "ACM", "LINKX"]
        assert [
            label.get_text() for label in figure.axes[2].get_xticklabels()
        ] == [
            r"$A_{\mathrm{erm}}$",
            r"$A_{\mathrm{last}}$",
            r"$A_{\mathrm{head}}$",
            r"$A_{\mathrm{full}}$",
        ]
    finally:
        plt.close(figure)


def test_compact_figure_requires_all_120_cells() -> None:
    rows = make_complete_summaries()

    with pytest.raises(RuntimeError, match="119 compact-figure cells"):
        NAMESPACE["build_tail_vs_whole_compact_figure"](rows[:-1])


def test_compact_figure_writes_requested_format() -> None:
    output_stem = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "test-rq5-compact"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output = output_stem.with_suffix(".png")
    try:
        outputs = NAMESPACE["plot_tail_vs_whole_compact"](
            make_complete_summaries(),
            output_stem,
            formats=("png",),
            dpi=100,
        )

        assert outputs == [output]
        assert output.is_file()
        assert output.stat().st_size > 0
        assert plt.imread(output).shape[:2] == (295, 550)
    finally:
        output.unlink(missing_ok=True)
