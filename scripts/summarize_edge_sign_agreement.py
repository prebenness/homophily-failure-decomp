"""Count Appendix J treatment/control comparisons supporting higher homophily.

Read the same compact summaries as the eight appendix figures. Each nonzero
dataset/model/dose pair receives one vote within its direction and preparation.
The main-text two-model figure is a subset and is not counted again.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from plot_edge_dose_response_composite import load_summary as load_forward_summary
from plot_reverse_edge_dose_response import load_summary as load_reverse_summary


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARATIONS = ("erm", "last", "head", "full")
FORWARD_SIGN = {
    "pubmed": 1, "ogbn-arxiv": 1,
    "roman-empire": -1, "amazon-ratings": -1, "arxiv-year": -1,
}
REPORT_NAME = "edge_perturbation_sign_agreement.txt"
# Six models over the complete reverse experiment's 49 dataset/dose points.
REVERSE_COMPARISONS = 294


@dataclass(frozen=True)
class Counts:
    agree: int
    opposite: int
    ties: int

    @property
    def total(self) -> int:
        return self.agree + self.opposite + self.ties

    @property
    def percentage(self) -> float:
        return 100.0 * self.agree / self.total


def count_signs(
    rows: Sequence[tuple[str, str, int, str, float]], direction: str,
) -> Counts:
    """Rows are dataset, model, dose (percent or target count), condition, mean."""
    if direction not in ("forward", "reverse"):
        raise ValueError(f"Unknown direction: {direction}")
    pairs: dict[tuple[str, str, int], dict[str, float]] = {}
    for dataset, model, dose, condition, mean in rows:
        if dose == 0:
            continue
        if dataset not in FORWARD_SIGN or dose < 0 or condition not in ("treatment", "control"):
            raise ValueError("Invalid dataset, dose or condition")
        pair = pairs.setdefault((dataset, model, dose), {})
        if condition in pair:
            raise ValueError("Duplicate treatment/control row")
        pair[condition] = mean
    agree = opposite = ties = 0
    for (dataset, _, _), pair in pairs.items():
        if set(pair) != {"treatment", "control"}:
            raise ValueError("Incomplete treatment/control pair")
        expected_sign = FORWARD_SIGN[dataset] * (1 if direction == "forward" else -1)
        difference = expected_sign * (pair["treatment"] - pair["control"])
        if difference > 0:
            agree += 1
        elif difference < 0:
            opposite += 1
        else:
            ties += 1
    if not pairs:
        raise ValueError("No nonzero-dose comparisons")
    return Counts(agree, opposite, ties)


def summary_path(results_root: Path, direction: str, preparation: str) -> Path:
    if direction == "reverse":
        return results_root / "reverse_edge_perturbation" / preparation / "dose_response_summary.csv"
    folder = ("edge_perturbation_dose_response_full" if preparation == "erm"
              else f"edge_perturbation_dose_response_preparation_{preparation}")
    return results_root / folder / "dose_response_summary.csv"


def figure_name(direction: str, preparation: str) -> str:
    if direction == "reverse":
        return f"reverse_edge_perturbation_{preparation}.pdf"
    suffix = "" if preparation == "erm" else f"_{preparation}"
    return f"edge_perturbation_dose_response{suffix}.pdf"


def collect_counts(
    results_root: Path, *, allow_missing_reverse: bool = False,
) -> dict[tuple[str, str], Counts | None]:
    counts = {}
    reverse_grid = None
    for direction in ("forward", "reverse"):
        for preparation in PREPARATIONS:
            path = summary_path(results_root, direction, preparation)
            if direction == "reverse" and not path.is_file() and allow_missing_reverse:
                counts[direction, preparation] = None
                continue
            if direction == "forward":
                rows = [
                    (r.dataset, r.model, int(r.p_percent), r.condition, r.mean_macro_f1)
                    for r in load_forward_summary(path)
                ]
            else:
                load_reverse_summary(path, preparation)
                # Retain integer target counts when pairing reverse doses; the
                # plot's displayed percentages need not be rounded or inferred.
                with path.open(encoding="utf-8", newline="") as stream:
                    raw_rows = list(csv.DictReader(stream))
                rows = [
                    (r["dataset"], r["model"], int(r["target_count"]),
                     r["condition"], float(r["mean_macro_f1"]))
                    for r in raw_rows
                ]
                grid = {
                    (r["dataset"], r["model"], int(r["target_count"]),
                     float(r["actual_pct"]), r["condition"])
                    for r in raw_rows
                }
                if reverse_grid is not None and grid != reverse_grid:
                    raise ValueError("Reverse preparations must use the same graph-dose matrix")
                reverse_grid = grid
            panel = count_signs(rows, direction)
            if direction == "reverse" and panel.total != REVERSE_COMPARISONS:
                raise ValueError(
                    f"Incomplete reverse figure: expected {REVERSE_COMPARISONS} "
                    f"nonzero-dose comparisons, found {panel.total}"
                )
            counts[direction, preparation] = panel
    return counts


def pool_counts(counts: Sequence[Counts]) -> Counts:
    return Counts(*(sum(getattr(c, field) for c in counts)
                    for field in ("agree", "opposite", "ties")))


def format_counts(counts: Counts | None) -> str:
    if counts is None:
        return "UNAVAILABLE (required numerical summaries missing)"
    return (f"{counts.agree}/{counts.total} = {counts.percentage:.6f}% "
            f"(opposite: {counts.opposite}; ties: {counts.ties})")


def build_report(counts: dict[tuple[str, str], Counts | None]) -> str:
    lines = [
        "Appendix J: sign agreement with higher local homophily improving tail macro-F1",
        "",
        "Unit: one treatment/control comparison per dataset, model, direction,",
        "nonzero dose and preparation. Compare the plotted mean macro-F1 values",
        "(30 checkpoints within each of five graph replicates, then average replicates).",
        "Exclude dose zero. Count exact equal means as ties in the denominator.",
        "No significance filter or rounding threshold is applied.",
        "",
        "Expected treatment-minus-control sign:",
        "  Forward: positive for PubMed/ogbn-arxiv; negative for the other three datasets.",
        "  Reverse: negative for PubMed/ogbn-arxiv; positive for the other three datasets.",
        "Pool comparison counts, not figure percentages. The main-text subset is not counted twice.",
        "",
        "Per appendix figure (identified by filename so manuscript renumbering is harmless):",
    ]
    for direction in ("forward", "reverse"):
        for preparation in PREPARATIONS:
            lines.append(f"  {direction} / {preparation}: {format_counts(counts[direction, preparation])}")
            lines.append(f"    {figure_name(direction, preparation)}")
    lines.extend(["", "Pooled comparisons:"])
    for direction in ("forward", "reverse"):
        selected = [counts[direction, p] for p in PREPARATIONS]
        pooled = None if any(c is None for c in selected) else pool_counts(selected)
        lines.append(f"  All {direction} preparations: {format_counts(pooled)}")
    selected = list(counts.values())
    pooled = None if any(c is None for c in selected) else pool_counts(selected)
    lines.append(f"  All directions and preparations: {format_counts(pooled)}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing-reverse", action="store_true",
                        help="report available figures and mark reverse/overall totals unavailable")
    args = parser.parse_args(argv)
    report = build_report(collect_counts(
        args.results_root, allow_missing_reverse=args.allow_missing_reverse,
    ))
    output = args.output or args.results_root / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
