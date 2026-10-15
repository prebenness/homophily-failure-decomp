from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal
from pathlib import Path
import runpy
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARATIONS = ("erm", "last", "head", "full")
FORWARD_COLOURS = {
    "pubmed": "orange",
    "ogbn-arxiv": "orange",
    "roman-empire": "blue",
    "amazon-ratings": "blue",
    "arxiv-year": "blue",
}


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(ROOT / "scripts/summarize_edge_sign_agreement.py"))


def pair(module, dose, treatment_mean, control_mean, treatment_sd=0.0, control_sd=0.0):
    return [
        module["SummaryRow"](
            dataset="pubmed", model="gcn", p_percent=dose,
            condition=condition, n_graph_replicates=5, n_checkpoints_per_graph=30,
            mean_macro_f1=mean, sd_macro_f1=sd,
        )
        for condition, mean, sd in (
            ("treatment", treatment_mean, treatment_sd),
            ("control", control_mean, control_sd),
        )
    ]


@pytest.mark.parametrize("treatment,control,treatment_sd,control_sd,winner", [
    (0.625, 0.5, 0.03125, 0.03125, "orange"),
    (0.5, 0.625, 0.03125, 0.03125, "blue"),
    (0.625, 0.5, 0.125, 0.125, "undecided"),
    (0.5, 0.625, 0.125, 0.125, "undecided"),
    (0.75, 0.5, 0.125, 0.125, "undecided"),
    (0.5, 0.75, 0.125, 0.125, "undecided"),
    (0.5, 0.5, 0.0, 0.0, "undecided"),
])
def test_dose_winner_requires_strict_band_separation(
    module, treatment, control, treatment_sd, control_sd, winner,
):
    subplot, = module["classify_subplots"](
        pair(module, 5, treatment, control, treatment_sd, control_sd)
    )
    assert subplot.total == 1
    assert getattr(subplot, winner) == 1
    assert subplot.winner == winner


def test_zero_dose_does_not_change_the_majority(module):
    rows = [
        *pair(module, 0, 0.25, 0.75),
        *pair(module, 5, 0.75, 0.25),
        *pair(module, 10, 0.75, 0.25),
        *pair(module, 15, 0.5, 0.5),
    ]
    subplot, = module["classify_subplots"](rows)
    assert (subplot.orange, subplot.blue, subplot.undecided, subplot.total) == (2, 0, 1, 3)
    assert subplot.winner == "orange"


@pytest.mark.parametrize("orange,blue,undecided,winner", [
    (2, 1, 1, "undecided"),  # A majority of decided doses is insufficient.
    (3, 2, 2, "undecided"),  # The largest category is insufficient.
    (2, 2, 0, "undecided"),
    (0, 1, 1, "undecided"),
    (3, 1, 1, "orange"),
    (1, 3, 1, "blue"),
])
def test_subplot_requires_more_than_half_of_all_doses(module, orange, blue, undecided, winner):
    subplot = module["Subplot"]("pubmed", "gcn", orange, blue, undecided)
    assert subplot.winner == winner


def test_unequal_dose_counts_still_give_each_subplot_one_vote(module):
    # These have two and nineteen doses, as in the reverse PubMed and
    # Roman-empire grids. Blue supports higher homophily only for PubMed.
    subplots = [
        module["Subplot"]("pubmed", "gcn", 0, 2, 0),
        module["Subplot"]("roman-empire", "gcn", 0, 19, 0),
    ]
    tally = module["subplot_tally"]([("reverse", subplots)])
    assert "orange: 0; blue: 2; undecided: 0" in tally
    assert "supports higher homophily: 1/2 = 50.000000%; opposes: 1" in tally


@pytest.mark.parametrize("dataset,forward_colour", FORWARD_COLOURS.items())
@pytest.mark.parametrize("direction", ("forward", "reverse"))
def test_expected_colour_and_support_flip_with_direction(module, dataset, forward_colour, direction):
    expected = forward_colour if direction == "forward" else (
        "blue" if forward_colour == "orange" else "orange"
    )
    assert module["expected_colour"](direction, dataset) == expected
    subplot = module["Subplot"](
        dataset, "gcn", int(expected == "orange"), int(expected == "blue"), 0,
    )
    tally = module["subplot_tally"]([(direction, [subplot])])
    assert "supports higher homophily: 1/1 = 100.000000%; opposes: 0" in tally


def test_undecided_subplots_stay_in_the_support_denominator(module):
    subplots = [
        module["Subplot"]("pubmed", "gcn", 1, 0, 0),
        module["Subplot"]("pubmed", "linkx", 2, 1, 1),
    ]
    tally = module["subplot_tally"]([("forward", subplots)])
    assert "orange: 1; blue: 0; undecided: 1" in tally
    assert "supports higher homophily: 1/2 = 50.000000%; opposes: 0" in tally


def test_missing_reverse_keeps_overall_subplot_tally_unavailable(module, tmp_path):
    for preparation in PREPARATIONS:
        source = module["summary_path"](ROOT / "results", "forward", preparation)
        target = module["summary_path"](tmp_path, "forward", preparation)
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
    with pytest.raises(FileNotFoundError):
        module["collect_subplots"](tmp_path)
    panels = module["collect_subplots"](tmp_path, allow_missing_reverse=True)
    assert all(len(panels["forward", p]) == 30 for p in PREPARATIONS)
    assert all(panels["reverse", p] is None for p in PREPARATIONS)
    report = module["build_subplot_report"](panels)
    assert "All forward preparations: orange:" in report
    assert "/120 = " in report
    assert "All reverse preparations: UNAVAILABLE" in report
    assert "All directions and preparations: UNAVAILABLE" in report


def independently_count_csv_bands(path):
    """Check the displayed decimal inputs using mean gaps and combined widths."""
    pairs = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            dose = Decimal(row.get("target_count", row.get("p_percent")))
            if dose:
                pairs[row["dataset"], row["model"], dose][row["condition"]] = (
                    Decimal(row["mean_macro_f1"]),
                    Decimal(row["sd_macro_f1_across_graph_replicates"]),
                )
    counts = defaultdict(Counter)
    for (dataset, model, _), conditions in pairs.items():
        treatment_mean, treatment_sd = conditions["treatment"]
        control_mean, control_sd = conditions["control"]
        gap = treatment_mean - control_mean
        combined_width = treatment_sd + control_sd
        colour = ("orange" if gap > combined_width else
                  "blue" if gap < -combined_width else "undecided")
        counts[dataset, model][colour] += 1
    return counts


def test_actual_summaries_supply_240_subplots_with_independently_checked_bands(module):
    panels = module["collect_subplots"](ROOT / "results")
    assert set(panels) == {(direction, p) for direction in ("forward", "reverse")
                           for p in PREPARATIONS}
    independent_colours = Counter()
    support = oppose = 0
    for (direction, preparation), subplots in panels.items():
        assert len(subplots) == 30
        expected_counts = independently_count_csv_bands(
            module["summary_path"](ROOT / "results", direction, preparation)
        )
        assert {(s.dataset, s.model) for s in subplots} == set(expected_counts)
        for subplot in subplots:
            expected = expected_counts[subplot.dataset, subplot.model]
            assert (subplot.orange, subplot.blue, subplot.undecided) == (
                expected["orange"], expected["blue"], expected["undecided"],
            )
            total = sum(expected.values())
            winner = next((colour for colour in ("orange", "blue")
                           if expected[colour] > total / 2), "undecided")
            assert subplot.winner == winner
            independent_colours[winner] += 1
            expected_colour = FORWARD_COLOURS[subplot.dataset]
            if direction == "reverse":
                expected_colour = "blue" if expected_colour == "orange" else "orange"
            if winner != "undecided":
                support += winner == expected_colour
                oppose += winner != expected_colour
    assert sum(independent_colours.values()) == 240
    report = module["build_subplot_report"](panels)
    expected_pooled = (
        f"All directions and preparations: orange: {independent_colours['orange']}; "
        f"blue: {independent_colours['blue']}; undecided: {independent_colours['undecided']}; "
        f"supports higher homophily: {support}/240 = {100 * support / 240:.6f}%; opposes: {oppose}"
    )
    assert expected_pooled in report
