from pathlib import Path
import runpy
import csv

import numpy as np
import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_rq3_mlp_contrasts.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))


def make_run_gap(
    run_id: str,
    split: str,
    seed: int,
    gap: float,
    *,
    whole: float = 0.8,
    tail: float | None = None,
    class_ids: tuple[int, ...] | None = None,
    weights: tuple[float, ...] | None = None,
):
    if tail is None:
        tail = whole - gap
    return NAMESPACE["RunGap"](
        run_id=run_id,
        split=split,
        seed=seed,
        whole_macro_f1=whole,
        tail_macro_f1=tail,
        tail_gap=gap,
        path=Path(f"{run_id}.json"),
        sha256="0" * 64,
        tail_class_ids=class_ids,
        whole_reference_weights=weights,
    )


def test_summarise_contrast_matches_split_and_seed() -> None:
    mlp = {
        ("split-0", 1): make_run_gap("m0", "split-0", 1, 0.10),
        ("split-1", 1): make_run_gap("m1", "split-1", 1, 0.20),
    }
    gcn = {
        ("split-1", 1): make_run_gap("g1", "split-1", 1, 0.15),
        ("split-0", 1): make_run_gap("g0", "split-0", 1, 0.30),
    }

    summary = NAMESPACE["summarise_contrast"](
        "pubmed", "gcn", mlp, gcn, 200, 123
    )

    np.testing.assert_allclose(summary.run_contrasts, [0.20, -0.05])
    assert summary.mean_contrast == pytest.approx(0.075)


def test_summarise_components_keeps_all_four_endpoint_means() -> None:
    mlp = {
        ("split-0", 1): make_run_gap(
            "m0", "split-0", 1, 0.20, whole=0.80
        ),
        ("split-1", 1): make_run_gap(
            "m1", "split-1", 1, 0.10, whole=0.90
        ),
    }
    gcn = {
        ("split-0", 1): make_run_gap(
            "g0", "split-0", 1, 0.30, whole=0.75
        ),
        ("split-1", 1): make_run_gap(
            "g1", "split-1", 1, 0.20, whole=0.85
        ),
    }

    summary = NAMESPACE["summarise_components"](
        "pubmed", "gcn", mlp, gcn
    )

    assert summary.mean_mlp_whole_macro_f1 == pytest.approx(0.85)
    assert summary.mean_mlp_tail_macro_f1 == pytest.approx(0.70)
    assert summary.mean_model_whole_macro_f1 == pytest.approx(0.80)
    assert summary.mean_model_tail_macro_f1 == pytest.approx(0.55)
    assert summary.mean_gap_contrast == pytest.approx(0.10)


def test_validate_gap_components_rejects_inconsistent_gap() -> None:
    run = make_run_gap(
        "bad", "split-0", 1, 0.10, whole=0.80, tail=0.65
    )

    with pytest.raises(RuntimeError, match="stored tail gap"):
        NAMESPACE["validate_gap_components"](run, "test")


def test_panel_grid_has_one_small_group_spacer() -> None:
    height_ratios, panel_rows = NAMESPACE["panel_grid_spec"]()

    assert panel_rows == (0, 1, 3, 4, 5)
    assert len(height_ratios) == 6
    assert height_ratios[2] < 0.1
    assert all(height == 1.0 for index, height in enumerate(height_ratios) if index != 2)


def test_adjusted_contrast_requires_identical_class_support() -> None:
    mlp = {
        ("split-0", 1): make_run_gap(
            "m0",
            "split-0",
            1,
            0.10,
            class_ids=(0, 1),
            weights=(0.4, 0.6),
        )
    }
    gcn = {
        ("split-0", 1): make_run_gap(
            "g0",
            "split-0",
            1,
            0.30,
            class_ids=(0, 2),
            weights=(0.4, 0.6),
        )
    }

    with pytest.raises(RuntimeError, match="tail class IDs differ"):
        NAMESPACE["summarise_contrast"](
            "pubmed", "gcn", mlp, gcn, 200, 123
        )


def write_adjusted_rows(path: Path, n_rows: int) -> None:
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(n_rows):
            writer.writerow(
                {
                    "dataset": "pubmed",
                    "model": "gcn",
                    "preparation": "erm",
                    "split": f"split-{index // 3}",
                    "seed": index % 3,
                    "tail_classes": 2,
                    "tail_class_ids": "0;1",
                    "whole_reference_weights": "0.4;0.6",
                    "whole_macro_f1": 0.8,
                    "adjusted_tail_macro_f1": 0.8 - index / 100,
                    "adjusted_macro_f1_gap": index / 100,
                }
            )


def test_load_adjusted_run_gaps_reads_complete_panel(tmp_path: Path) -> None:
    path = tmp_path / "adjusted.csv"
    write_adjusted_rows(path, 30)

    records = NAMESPACE["load_adjusted_run_gaps"](path, "pubmed", "gcn")

    assert len(records) == 30
    assert records[("split-4", 2)].tail_gap == pytest.approx(0.14)
    assert records[("split-4", 2)].whole_macro_f1 == pytest.approx(0.8)
    assert records[("split-4", 2)].tail_macro_f1 == pytest.approx(0.66)
    assert records[("split-4", 2)].tail_class_ids == (0, 1)
    assert records[("split-4", 2)].whole_reference_weights == (0.4, 0.6)


def test_load_adjusted_run_gaps_rejects_incomplete_panel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adjusted.csv"
    write_adjusted_rows(path, 29)

    with pytest.raises(RuntimeError, match="found 29 adjusted ERM runs"):
        NAMESPACE["load_adjusted_run_gaps"](path, "pubmed", "gcn")
