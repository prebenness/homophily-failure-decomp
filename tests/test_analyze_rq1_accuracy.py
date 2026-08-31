from pathlib import Path
import runpy

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_rq1_accuracy.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_compute_run_accuracy_uses_whole_test_and_registered_tail() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0])
    predictions = np.array([0, 0, 1, 1, 1, 0])
    test_idx = np.array([0, 1, 2, 3, 4])
    tail_mask = np.array([False, True, True, False, False, False])

    actual = NAMESPACE["compute_run_accuracy"](
        labels, predictions, test_idx, tail_mask
    )

    assert actual["n_test"] == 5
    assert actual["n_tail_test"] == 2
    assert actual["whole_accuracy"] == pytest.approx(3 / 5)
    assert actual["tail_accuracy"] == pytest.approx(1 / 2)
    assert actual["gap_whole_minus_tail"] == pytest.approx(1 / 10)


def test_compute_run_accuracy_rejects_empty_tail() -> None:
    labels = np.array([0, 1])
    predictions = np.array([0, 1])
    test_idx = np.array([0, 1])
    tail_mask = np.array([False, False])

    with pytest.raises(ValueError, match="test tail is empty"):
        NAMESPACE["compute_run_accuracy"](
            labels, predictions, test_idx, tail_mask
        )


def test_summarise_runs_gives_runs_equal_weight() -> None:
    rows = []
    for dataset in NAMESPACE["DATASET_ORDER"]:
        for model in NAMESPACE["MODEL_ORDER"]:
            for run_index, (whole, tail) in enumerate(
                ((0.2, 0.1), (0.8, 0.5))
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "split": f"split-{run_index}",
                        "seed": 1,
                        "tail_side": "low-h_v",
                        "tail_interval": "[0,0.25)",
                        "whole_accuracy": whole,
                        "tail_accuracy": tail,
                        "n_test": 100,
                        "n_tail_test": 20,
                    }
                )

    summaries = NAMESPACE["summarise_runs"](rows)

    assert len(summaries) == 35
    assert summaries[0]["mean_whole_accuracy"] == pytest.approx(0.5)
    assert summaries[0]["mean_tail_accuracy"] == pytest.approx(0.3)
    assert summaries[0]["mean_gap_whole_minus_tail"] == pytest.approx(0.2)
    assert summaries[0]["bootstrap_ci_lower_gap"] > 0.0
    assert summaries[0]["gap_ci_excludes_zero"] is True


def test_latex_cell_uses_raw_interval_for_grey_shading() -> None:
    conclusive = {
        "mean_gap_whole_minus_tail": 0.2,
        "bootstrap_ci_lower_gap": 0.1,
        "bootstrap_ci_upper_gap": 0.3,
        "gap_ci_excludes_zero": True,
    }
    uncertain = {
        "mean_gap_whole_minus_tail": -0.0001,
        "bootstrap_ci_lower_gap": -0.1,
        "bootstrap_ci_upper_gap": 0.1,
        "gap_ci_excludes_zero": False,
    }

    assert NAMESPACE["latex_gap_cell"](conclusive) == (
        r"\gaincell{+0.200}{[0.100,0.300]}"
    )
    assert NAMESPACE["latex_gap_cell"](uncertain) == (
        r"\uncertaingaincell{0.000}{[-0.100,0.100]}"
    )
