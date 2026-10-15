from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/render_rq2_recovered_share_tables.py"
SPEC = importlib.util.spec_from_file_location("render_rq2_rho", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rows():
    return MODULE.read_rows(PROJECT_ROOT / "results/main/aggregates/paired_gains.csv")


def test_canonical_counts_keep_uncertain_classifier_gains() -> None:
    shares = MODULE.compute_recovered_shares(_rows())
    assert len(shares) == 60
    for preparation, above in (("last", 8), ("head", 13)):
        selected = [s for s in shares if s.preparation == preparation]
        assert sum(s.passes_rule for s in selected) == 19
        assert sum(s.rho is not None and s.rho > 0.7 for s in selected) == above
    index = {(s.preparation, s.dataset, s.model): s for s in shares}
    assert index[("last", "roman-empire", "graphsage-mean")].passes_rule
    assert index[("head", "amazon-ratings", "linkx")].passes_rule
    assert index[("head", "arxiv-year", "gcn")].rho is None


@pytest.mark.parametrize(
    "denominator,lower,upper,eligible",
    [(0.03, 0.01, 0.05, True), (0.029, 0.01, 0.05, False),
     (0.1, 0.0, 0.2, False), (0.1, -0.01, 0.2, False),
     (0.0, -0.02, 0.02, False), (-0.1, -0.2, -0.01, False)],
)
def test_denominator_rule_handles_threshold_zero_and_uncertainty(
    denominator, lower, upper, eligible
) -> None:
    rows = _rows()
    for row in rows:
        if row["dataset"] == "pubmed" and row["model"] == "gcn":
            if row["preparation"] == "full":
                row.update(mean_paired_gain=str(denominator),
                           bootstrap_ci_lower=str(lower),
                           bootstrap_ci_upper=str(upper))
            else:
                row.update(mean_paired_gain="0", bootstrap_ci_lower="-0.1",
                           bootstrap_ci_upper="0.1")
    shares = MODULE.compute_recovered_shares(rows)
    selected = [s for s in shares if s.dataset == "pubmed" and s.model == "gcn"]
    assert all(s.passes_rule == eligible for s in selected)
    assert all(s.rho == (0.0 if eligible else None) for s in selected)


def test_main_fragments_match_canonical_artifacts_and_use_em_dashes() -> None:
    shares = MODULE.compute_recovered_shares(_rows())
    for preparation in MODULE.PREPARATIONS:
        latex = MODULE.render_latex(preparation, shares)
        path = MODULE.DEFAULT_OUTPUT_DIR / f"rq2_{preparation}_recovered_share.tex"
        assert latex == path.read_text(encoding="utf-8")
        assert "MLP" not in latex
        assert r"\cellcolor" not in latex
        assert latex.count(r"\textemdash") == 11
        assert r"\begin{table}[H]" in latex
        assert r"95\% paired-bootstrap interval includes zero" in latex


def test_manifest_has_input_output_hashes_and_eligibility_counts() -> None:
    shares = MODULE.compute_recovered_shares(_rows())
    rendered = {f"rq2_{p}_recovered_share.tex": MODULE.render_latex(p, shares)
                for p in MODULE.PREPARATIONS}
    payload = json.loads(MODULE.build_manifest(MODULE.DEFAULT_INPUT, rendered, shares))
    assert payload["eligible_counts"] == {"last": 19, "head": 19}
    assert payload["shares_above_0.7"] == {"last": 8, "head": 13}
    assert payload["input"]["path"] == "results/main/aggregates/paired_gains.csv"
    assert set(payload["outputs"]) == set(rendered)
    assert len(payload["input"]["sha256"]) == 64


def test_incomplete_panel_is_rejected() -> None:
    rows = [r for r in _rows() if not (
        r["dataset"] == "pubmed" and r["model"] == "gcn"
        and r["preparation"] == "full"
    )]
    with pytest.raises(ValueError, match="Missing paired-gain row"):
        MODULE.compute_recovered_shares(rows)
