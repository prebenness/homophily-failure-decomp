import csv
from pathlib import Path
import runpy

import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_rq4_class_adjusted_model_families.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))

FIELDS = [
    "dataset",
    "model",
    "preparation",
    "split",
    "seed",
    "tail_classes",
    "tail_class_ids",
    "whole_reference_weights",
    "whole_macro_f1",
    "adjusted_tail_macro_f1",
    "adjusted_macro_f1_gap",
]


def test_class_adjusted_figure_uses_the_figure_5_model_panel() -> None:
    assert NAMESPACE["MODEL_ORDER"] == (
        "mlp",
        "gcn",
        "graphsage-mean",
        "h2gcn",
        "gpr-gnn",
        "acm-gcn",
        "linkx",
    )
    assert NAMESPACE["MODEL_FAMILIES"] == {
        "mlp": "mlp",
        "gcn": "standard",
        "graphsage-mean": "standard",
        "h2gcn": "heterophily-designed",
        "gpr-gnn": "heterophily-designed",
        "acm-gcn": "heterophily-designed",
        "linkx": "non-aggregation",
    }


def test_additional_panel_leaves_headroom_for_scores_at_one() -> None:
    assert NAMESPACE["score_axis_upper_limit"]("original") == 1.0
    assert NAMESPACE["score_axis_upper_limit"]("additional") == 1.06


def write_adjusted_csv(
    path: Path,
    *,
    models: tuple[str, ...] = ("gcn", "h2gcn"),
    runs: int = 2,
    mismatched_support: bool = False,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for model_index, model in enumerate(models):
            for run_index in range(runs):
                whole = 0.8 - 0.1 * model_index
                tail = 0.5 + 0.05 * run_index
                class_ids = (
                    "0;2"
                    if mismatched_support
                    and model == models[-1]
                    and run_index == 0
                    else "0;1"
                )
                writer.writerow(
                    {
                        "dataset": "pubmed",
                        "model": model,
                        "preparation": "erm",
                        "split": f"split-{run_index}",
                        "seed": 1,
                        "tail_classes": 2,
                        "tail_class_ids": class_ids,
                        "whole_reference_weights": "0.4;0.6",
                        "whole_macro_f1": whole,
                        "adjusted_tail_macro_f1": tail,
                        "adjusted_macro_f1_gap": whole - tail,
                    }
                )


def test_loader_maps_adjusted_components_and_gap(tmp_path: Path) -> None:
    path = tmp_path / "adjusted.csv"
    write_adjusted_csv(path)

    records = NAMESPACE["load_adjusted_run_scores"](
        path,
        datasets=("pubmed",),
        models=("gcn", "h2gcn"),
        expected_runs=2,
    )
    run = records[("pubmed", "gcn")][("split-0", 1)]

    assert run.whole_macro_f1 == pytest.approx(0.8)
    assert run.tail_macro_f1 == pytest.approx(0.5)
    assert run.tail_gap == pytest.approx(0.3)
    assert run.tail_class_ids == (0, 1)


def test_loader_rejects_incomplete_run_panel(tmp_path: Path) -> None:
    path = tmp_path / "adjusted.csv"
    write_adjusted_csv(path, runs=1)

    with pytest.raises(RuntimeError, match="expected 2"):
        NAMESPACE["load_adjusted_run_scores"](
            path,
            datasets=("pubmed",),
            models=("gcn", "h2gcn"),
            expected_runs=2,
        )


def test_support_must_match_across_models(tmp_path: Path) -> None:
    path = tmp_path / "adjusted.csv"
    write_adjusted_csv(path, mismatched_support=True)
    records = NAMESPACE["load_adjusted_run_scores"](
        path,
        datasets=("pubmed",),
        models=("gcn", "h2gcn"),
        expected_runs=2,
    )

    with pytest.raises(RuntimeError, match="tail class IDs differ"):
        NAMESPACE["validate_common_adjustment_support"](
            records,
            datasets=("pubmed",),
            models=("gcn", "h2gcn"),
        )
