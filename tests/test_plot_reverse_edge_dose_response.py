from __future__ import annotations

import csv
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(ROOT / "scripts/plot_reverse_edge_dose_response.py"))


def summary_rows(module):
    return [
        {"preparation": "erm", "dataset": dataset, "model": model,
         "target_count": count, "actual_pct": percentage, "condition": condition,
         "n_graph_replicates": 5, "n_checkpoints_per_graph": 30,
         "mean_macro_f1": 0.5, "sd_macro_f1_across_graph_replicates": 0.0 if count == 0 else 0.02}
        for dataset, _ in module["DATASETS"] for model, _ in module["MODELS"]
        for count, percentage in ((0, 0.0), (13, 6.5))
        for condition in module["CONDITIONS"]
    ]


def write_summary(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compact_plot_preserves_fractional_original_tail_percentages(module, tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path, summary_rows(module))
    rows, endpoints = module["load_summary"](path, "erm")
    assert len(rows) == 120
    assert set(endpoints.values()) == {6.5}
    assert set(row.p_percent for row in rows) == {0.0, 6.5}


def test_compact_plot_rejects_an_incomplete_model_condition(module, tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path, summary_rows(module)[:-1])
    with pytest.raises(ValueError, match="incomplete six-model"):
        module["load_summary"](path, "erm")


def test_missing_summary_skip_is_reported_without_creating_outputs(module, tmp_path, capsys):
    module["main"](["--summary-root", str(tmp_path), "--output-root", str(tmp_path / "out"),
                    "--skip-missing"])
    assert "not regenerated" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
