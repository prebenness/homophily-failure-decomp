from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_rq4_model_families.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))


def make_run_score(
    run_id: str,
    split: str,
    seed: int,
    *,
    whole: float,
    tail: float,
):
    return NAMESPACE["RunScore"](
        run_id=run_id,
        split=split,
        seed=seed,
        whole_macro_f1=whole,
        tail_macro_f1=tail,
        tail_gap=whole - tail,
        path=Path(f"{run_id}.json"),
        sha256="0" * 64,
    )


def test_pairwise_contrast_matches_split_and_seed() -> None:
    standard = {
        ("split-0", 1): make_run_score(
            "s0", "split-0", 1, whole=0.80, tail=0.50
        ),
        ("split-1", 1): make_run_score(
            "s1", "split-1", 1, whole=0.70, tail=0.50
        ),
    }
    designed = {
        ("split-1", 1): make_run_score(
            "d1", "split-1", 1, whole=0.65, tail=0.55
        ),
        ("split-0", 1): make_run_score(
            "d0", "split-0", 1, whole=0.75, tail=0.55
        ),
    }

    summary = NAMESPACE["summarise_pairwise_contrast"](
        "pubmed",
        "h2gcn",
        "gcn",
        designed,
        standard,
        bootstrap_resamples=200,
        base_seed=123,
    )

    np.testing.assert_allclose(summary.run_contrasts, [-0.10, -0.10])
    assert summary.mean_contrast == pytest.approx(-0.10)
    assert summary.mean_designed_tail_gap == pytest.approx(0.15)
    assert summary.mean_standard_tail_gap == pytest.approx(0.25)


def test_score_summary_uses_requested_scope() -> None:
    records = {
        ("split-0", 1): make_run_score(
            "r0", "split-0", 1, whole=0.80, tail=0.40
        ),
        ("split-1", 1): make_run_score(
            "r1", "split-1", 1, whole=0.60, tail=0.20
        ),
    }

    whole = NAMESPACE["summarise_scores"](
        "pubmed",
        "gcn",
        "whole_test",
        records,
        bootstrap_resamples=200,
        base_seed=123,
    )
    tail = NAMESPACE["summarise_scores"](
        "pubmed",
        "gcn",
        "tail",
        records,
        bootstrap_resamples=200,
        base_seed=123,
    )

    assert whole.mean_macro_f1 == pytest.approx(0.70)
    assert tail.mean_macro_f1 == pytest.approx(0.30)


def test_model_offsets_are_ordered_and_centred() -> None:
    offsets = NAMESPACE["model_offsets"]()
    ordered = [offsets[model] for model in NAMESPACE["MODEL_ORDER"]]

    assert ordered == sorted(ordered)
    assert np.mean(ordered) == pytest.approx(0.0)


def make_complete_score_panel():
    summary_class = NAMESPACE["ScoreSummary"]
    summaries = []
    for dataset_index, dataset in enumerate(NAMESPACE["DATASET_ORDER"]):
        for model_index, model in enumerate(NAMESPACE["MODEL_ORDER"]):
            whole = 0.85 - 0.03 * dataset_index - 0.01 * model_index
            tail = whole - 0.12
            for scope, mean in (
                ("whole_test", whole),
                ("tail", tail),
            ):
                summaries.append(
                    summary_class(
                        dataset=dataset,
                        model=model,
                        scope=scope,
                        n_runs=30,
                        mean_macro_f1=mean,
                        sample_std_macro_f1=0.01,
                        ci_lower=mean - 0.01,
                        ci_upper=mean + 0.01,
                        run_scores=tuple([mean] * 30),
                    )
                )
    return summaries


def test_score_plot_uses_simplified_shared_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_stem = tmp_path / "rq4_scores"
    observed: dict[str, object] = {}
    original_save_figure = NAMESPACE["save_figure"]

    def inspect_then_save(figure, *args, **kwargs):
        axis = figure.axes[0]
        lines = axis.get_lines()
        observed["ylim"] = axis.get_ylim()
        observed["collections"] = len(axis.collections)
        observed["connectors"] = sum(
            line.get_marker() == "None"
            and line.get_linestyle() == "-"
            for line in lines
        )
        observed["whole_markers"] = sum(
            line.get_marker() == "D"
            for line in lines
        )
        observed["tail_markers"] = sum(
            line.get_marker() == "o"
            for line in lines
        )
        return original_save_figure(figure, *args, **kwargs)

    monkeypatch.setitem(
        NAMESPACE["plot_scores"].__globals__,
        "save_figure",
        inspect_then_save,
    )
    outputs = NAMESPACE["plot_scores"](
        make_complete_score_panel(),
        output_stem,
        formats=("png",),
        dpi=100,
    )

    assert outputs == [output_stem.with_suffix(".png")]
    assert outputs[0].is_file()
    assert NAMESPACE["FAMILY_COLOURS"] == {
        "standard": "#a65e3b",
        "heterophily-designed": "#2f6f8f",
    }
    assert NAMESPACE["SCOPE_MARKERS"] == {
        "whole_test": "D",
        "tail": "o",
    }
    assert observed == {
        "ylim": (0.0, 1.0),
        "collections": 0,
        "connectors": 25,
        "whole_markers": 25,
        "tail_markers": 25,
    }


def test_score_plot_rejects_an_incomplete_panel(tmp_path: Path) -> None:
    summaries = make_complete_score_panel()
    summaries.pop()

    with pytest.raises(RuntimeError, match="score panel is incomplete"):
        NAMESPACE["plot_scores"](
            summaries,
            tmp_path / "incomplete",
            formats=("png",),
            dpi=100,
        )

    plt.close("all")


def test_appendix_table_contains_all_pairwise_rows() -> None:
    summary_class = NAMESPACE["PairwiseContrastSummary"]
    summaries = []
    for dataset in NAMESPACE["DATASET_ORDER"]:
        for designed_model in NAMESPACE["DESIGNED_MODELS"]:
            for standard_model in NAMESPACE["STANDARD_MODELS"]:
                summaries.append(
                    summary_class(
                        dataset=dataset,
                        designed_model=designed_model,
                        standard_model=standard_model,
                        n_pairs=30,
                        mean_designed_tail_gap=0.1,
                        mean_standard_tail_gap=0.2,
                        mean_contrast=-0.1,
                        sample_std_contrast=0.01,
                        ci_lower=-0.12,
                        ci_upper=-0.08,
                        smaller_tail_count=0,
                        p_value=0.0,
                        run_contrasts=tuple([-0.1] * 30),
                    )
                )

    table = NAMESPACE["render_appendix_table"](summaries)

    assert "\\label{tab:rq4-designed-vs-standard}" in table
    assert table.count("\\multirow{2}{*}") == 3
    assert table.count("\\gaincell{-0.100}{[-0.120,-0.080]}") == 30
