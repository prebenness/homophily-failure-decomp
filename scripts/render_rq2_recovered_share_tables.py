#!/usr/bin/env python3
"""Render main-panel recovered-share tables from saved paired gains.

A ratio is reported only when the full-model gain is at least 0.03 and its
95% paired-bootstrap interval excludes zero, as specified in the registration.
The classifier-only gain's uncertainty does not change eligibility. This
writes portable LaTeX fragments and provenance without training or TeX tools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results/main/aggregates/paired_gains.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paper/generated/tables"
DENOMINATOR_THRESHOLD = 0.03
DATASETS = (
    ("pubmed", "PubMed"),
    ("ogbn-arxiv", r"\texttt{ogbn-arxiv}"),
    ("roman-empire", "Roman-empire"),
    ("amazon-ratings", "Amazon-ratings"),
    ("arxiv-year", r"\texttt{arxiv-year}"),
)
MODELS = (
    ("gcn", "GCN"), ("graphsage-mean", "GraphSAGE-mean"),
    ("h2gcn", "H2GCN-2"), ("gpr-gnn", "GPR-GNN"),
    ("acm-gcn", "ACM-GCN"), ("linkx", "LINKX"),
)
PREPARATIONS = ("last", "head")


@dataclass(frozen=True)
class RecoveredShare:
    dataset: str
    model: str
    preparation: str
    numerator: float
    denominator: float
    denominator_ci_lower: float
    denominator_ci_upper: float
    rho: float | None
    passes_rule: bool


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def compute_recovered_shares(rows: Iterable[dict[str, str]]) -> list[RecoveredShare]:
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["preparation"])
        if key in index:
            raise ValueError(f"Duplicate paired-gain row: {key}")
        index[key] = row
    results = []
    for preparation in PREPARATIONS:
        for dataset, _ in DATASETS:
            for model, _ in MODELS:
                numerator_row = index.get((dataset, model, preparation))
                denominator_row = index.get((dataset, model, "full"))
                if numerator_row is None or denominator_row is None:
                    raise ValueError(
                        f"Missing paired-gain row for {dataset}/{model}/{preparation}"
                    )
                numerator = float(numerator_row["mean_paired_gain"])
                denominator = float(denominator_row["mean_paired_gain"])
                lower = float(denominator_row["bootstrap_ci_lower"])
                upper = float(denominator_row["bootstrap_ci_upper"])
                qualifies = denominator >= DENOMINATOR_THRESHOLD and (
                    lower > 0.0 or upper < 0.0
                )
                results.append(RecoveredShare(
                    dataset=dataset, model=model, preparation=preparation,
                    numerator=numerator, denominator=denominator,
                    denominator_ci_lower=lower, denominator_ci_upper=upper,
                    rho=numerator / denominator if qualifies else None,
                    passes_rule=qualifies,
                ))
    return results


def render_latex(preparation: str, shares: Iterable[RecoveredShare]) -> str:
    """Return one table environment suitable for inclusion in the paper."""
    if preparation not in PREPARATIONS:
        raise ValueError(f"Unknown preparation: {preparation}")
    cell_index = {
        (share.dataset, share.model): share
        for share in shares if share.preparation == preparation
    }
    expected = {(dataset, model) for dataset, _ in DATASETS for model, _ in MODELS}
    if set(cell_index) != expected:
        raise ValueError(f"Incomplete paper table for {preparation}")
    rq = "RQ2a" if preparation == "last" else "RQ2b"
    refit = "final-layer" if preparation == "last" else "head"
    caption = (
        rf"\textbf{{{rq}}}. Recovered share \(\rho_{{\mathrm{{{preparation}}}}}\) "
        f"after {refit} refitting. An em dash indicates that the full-model "
        r"gain is below 0.03 or its 95\% paired-bootstrap interval includes zero."
    )
    lines = [
        r"\begin{table}[H]", r"  \centering", rf"  \caption{{{caption}}}",
        rf"  \label{{tab:rho-{preparation}-recovered-share}}",
        r"  \setlength{\tabcolsep}{5.5pt}",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"    \toprule", r"    & \multicolumn{2}{c}{Low-homophily tail}",
        r"    & \multicolumn{3}{c}{High-homophily tail} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        "    Model & " + " & ".join(label for _, label in DATASETS) + r" \\",
        r"    \midrule",
    ]
    for model, label in MODELS:
        cells = []
        for dataset, _ in DATASETS:
            share = cell_index[(dataset, model)]
            value = r"\textemdash" if share.rho is None else f"${share.rho:.3f}$"
            cells.append("$0.000$" if value == "$-0.000$" else value)
        lines.append("    " + label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def build_manifest(input_path: Path, rendered: dict[str, str],
                   shares: list[RecoveredShare]) -> str:
    payload = {
        "generator": "scripts/render_rq2_recovered_share_tables.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input": {"path": portable_path(input_path),
                  "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()},
        "eligibility": (
            "full-model mean gain >= 0.03 and its 95% paired-bootstrap "
            "interval excludes zero; numerator significance is not required"
        ),
        "ineligible_display": "em dash",
        "eligible_counts": {
            p: sum(s.passes_rule for s in shares if s.preparation == p)
            for p in PREPARATIONS
        },
        "shares_above_0.7": {
            p: sum(s.rho is not None and s.rho > 0.7 for s in shares
                   if s.preparation == p) for p in PREPARATIONS
        },
        "outputs": {
            name: {"sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
            for name, content in rendered.items()
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="Compare without writing.")
    args = parser.parse_args()
    shares = compute_recovered_shares(read_rows(args.input))
    rendered = {
        f"rq2_{p}_recovered_share.tex": render_latex(p, shares) for p in PREPARATIONS
    }
    manifest = build_manifest(args.input, rendered, shares)
    rendered["recovered_share_analysis_manifest.json"] = manifest
    if args.check:
        for name, content in rendered.items():
            path = args.output_dir / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Generated output differs: {portable_path(path)}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (args.output_dir / name).write_bytes(content.encode("utf-8"))


if __name__ == "__main__":
    main()
