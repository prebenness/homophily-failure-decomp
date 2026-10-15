from __future__ import annotations

import csv
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARATIONS = ("erm", "last", "head", "full")
FORWARD_SIGNS = {
    "pubmed": 1,
    "ogbn-arxiv": 1,
    "roman-empire": -1,
    "amazon-ratings": -1,
    "arxiv-year": -1,
}
MODELS = ("gcn", "graphsage-mean", "h2gcn", "gpr-gnn", "acm-gcn", "linkx")
REVERSE_DOSE_COUNTS = {
    "pubmed": 2,
    "ogbn-arxiv": 7,
    "roman-empire": 19,
    "amazon-ratings": 16,
    "arxiv-year": 5,
}


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(ROOT / "scripts/summarize_edge_sign_agreement.py"))


@pytest.fixture
def results_root(tmp_path):
    root = tmp_path / "results"
    for preparation in PREPARATIONS:
        folder = ("edge_perturbation_dose_response_full" if preparation == "erm"
                  else f"edge_perturbation_dose_response_preparation_{preparation}")
        target = root / folder / "dose_response_summary.csv"
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "results" / folder / target.name, target)
    return root


def write_reverse_summary(results_root, preparation, *, dose_counts=None, shift_pubmed=False):
    """Write a complete synthetic matrix with 49 nonzero dataset/dose points."""
    dose_counts = REVERSE_DOSE_COUNTS if dose_counts is None else dose_counts
    rows = []
    for dataset, number in dose_counts.items():
        doses = list(range(1, number + 1))
        if shift_pubmed and dataset == "pubmed":
            doses = [dose + 1 for dose in doses]
        for model in MODELS:
            for dose in [0, *doses]:
                for condition in ("treatment", "control"):
                    mean = 0.5
                    if dose and condition == "treatment":
                        mean -= 0.1 * FORWARD_SIGNS[dataset]
                    rows.append({
                        "preparation": preparation,
                        "dataset": dataset,
                        "model": model,
                        "target_count": dose,
                        "actual_pct": dose / 10,
                        "condition": condition,
                        "n_graph_replicates": 5,
                        "n_checkpoints_per_graph": 30,
                        "mean_macro_f1": mean,
                        "sd_macro_f1_across_graph_replicates": 0.02 if dose else 0.0,
                    })
    path = results_root / "reverse_edge_perturbation" / preparation / "dose_response_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("dataset,forward_sign", FORWARD_SIGNS.items())
@pytest.mark.parametrize("direction", ("forward", "reverse"))
def test_expected_sign_depends_on_dataset_and_direction(module, dataset, forward_sign, direction):
    expected_sign = forward_sign if direction == "forward" else -forward_sign
    rows = [
        (dataset, "gcn", 0, "treatment", 0.5),
        (dataset, "gcn", 0, "control", 0.5),
        (dataset, "gcn", 1, "treatment", 0.5 + expected_sign * 0.1),
        (dataset, "gcn", 1, "control", 0.5),
        (dataset, "gcn", 2, "treatment", 0.5 - expected_sign * 0.1),
        (dataset, "gcn", 2, "control", 0.5),
        (dataset, "gcn", 3, "treatment", 0.5),
        (dataset, "gcn", 3, "control", 0.5),
    ]
    counts = module["count_signs"](rows, direction)
    assert (counts.agree, counts.opposite, counts.ties, counts.total) == (1, 1, 1, 3)
    assert counts.percentage == pytest.approx(100 / 3)


def test_small_nonzero_difference_is_not_rounded_into_a_tie(module):
    counts = module["count_signs"]([
        ("pubmed", "gcn", 1, "treatment", 0.5 + 1e-14),
        ("pubmed", "gcn", 1, "control", 0.5),
    ], "forward")
    assert (counts.agree, counts.opposite, counts.ties) == (1, 0, 0)


@pytest.mark.parametrize("conditions,message", [
    (("treatment", "control", "treatment"), "Duplicate treatment/control row"),
    (("control",), "Incomplete treatment/control pair"),
])
def test_duplicate_or_missing_pair_is_rejected(module, conditions, message):
    rows = [("pubmed", "gcn", 1, condition, 0.5) for condition in conditions]
    with pytest.raises(ValueError, match=message):
        module["count_signs"](rows, "forward")


def test_dose_zero_cannot_supply_a_comparison(module):
    with pytest.raises(ValueError, match="No nonzero-dose comparisons"):
        module["count_signs"]([
            ("pubmed", "gcn", 0, "treatment", 0.5),
            ("pubmed", "gcn", 0, "control", 0.5),
        ], "forward")


def test_pooling_counts_weights_comparisons_and_keeps_ties(module):
    counts = module["pool_counts"]([
        module["Counts"](1, 0, 0),
        module["Counts"](1, 8, 1),
    ])
    assert (counts.agree, counts.opposite, counts.ties, counts.total) == (2, 8, 1, 11)
    assert counts.percentage == pytest.approx(200 / 11)
    assert counts.percentage != pytest.approx((100 + 10) / 2)


def test_complete_figures_have_correct_comparison_counts_and_global_pool(module, results_root):
    for preparation in PREPARATIONS:
        write_reverse_summary(results_root, preparation)
    counts = module["collect_counts"](results_root)
    for preparation in PREPARATIONS:
        assert counts["forward", preparation].total == 564
        reverse = counts["reverse", preparation]
        assert (reverse.agree, reverse.opposite, reverse.ties, reverse.total) == (294, 0, 0, 294)
    pooled = module["pool_counts"](list(counts.values()))
    assert pooled.total == 3432
    report = module["build_report"](counts)
    assert f"All directions and preparations: {pooled.agree}/3432 = " in report
    assert "All reverse preparations: 1176/1176 = 100.000000%" in report
    assert report.count(".pdf") == 8


@pytest.mark.parametrize("include_one_reverse", (False, True))
def test_missing_reverse_requires_explicit_permission_and_leaves_overall_unavailable(
    module, results_root, include_one_reverse, capsys,
):
    if include_one_reverse:
        write_reverse_summary(results_root, "erm")
    with pytest.raises(FileNotFoundError):
        module["collect_counts"](results_root)
    module["main"](["--results-root", str(results_root), "--allow-missing-reverse"])
    path = results_root / module["REPORT_NAME"]
    report = path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == report
    assert "All forward preparations: " in report
    assert "/2256 = " in report
    assert "All reverse preparations: UNAVAILABLE" in report
    assert "All directions and preparations: UNAVAILABLE" in report
    if include_one_reverse:
        assert "reverse / erm: 294/294 = 100.000000%" in report
    else:
        assert "reverse / erm: UNAVAILABLE" in report


def test_incomplete_reverse_grid_is_not_allowed_even_when_missing_summaries_are(module, results_root):
    incomplete = dict(REVERSE_DOSE_COUNTS, **{"arxiv-year": 4})
    write_reverse_summary(results_root, "erm", dose_counts=incomplete)
    # Every model and condition is present for each remaining dose, so this
    # exercises the total-comparison check beyond the plot loader's matrix check.
    with pytest.raises(ValueError, match="expected 294 nonzero-dose comparisons, found 288"):
        module["collect_counts"](results_root, allow_missing_reverse=True)


def test_reverse_preparations_must_share_the_same_doses(module, results_root):
    write_reverse_summary(results_root, "erm")
    write_reverse_summary(results_root, "last", shift_pubmed=True)
    with pytest.raises(ValueError, match="same graph-dose matrix"):
        module["collect_counts"](results_root, allow_missing_reverse=True)


def test_reverse_preparations_must_share_percentages_as_well_as_target_counts(module, results_root):
    write_reverse_summary(results_root, "erm")
    path = write_reverse_summary(results_root, "last")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row["dataset"] == "pubmed":
            row["actual_pct"] = str(float(row["actual_pct"]) * 2)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # Each file still uses one consistent denominator and identical target
    # counts; only the shared original-tail percentages differ between files.
    with pytest.raises(ValueError, match="same graph-dose matrix"):
        module["collect_counts"](results_root, allow_missing_reverse=True)
