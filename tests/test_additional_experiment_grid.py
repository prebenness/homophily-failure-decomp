from collections import Counter
from pathlib import Path

from hfd.experiment_grid import grid_jsonl_text, read_experiment_grid


REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "manifests" / "additional-dataset-runs.jsonl"


def test_additional_grid_is_complete_separate_and_deterministic() -> None:
    rows = read_experiment_grid(GRID_PATH)

    assert len(rows) == 1680
    assert grid_jsonl_text(rows) == GRID_PATH.read_text(encoding="utf-8")
    assert Counter(row.dataset for row in rows) == {
        "actor": 840,
        "blogcatalog": 840,
    }
    assert Counter(row.model for row in rows) == {
        "mlp": 240,
        "gcn": 240,
        "graphsage-mean": 240,
        "h2gcn": 240,
        "gpr-gnn": 240,
        "linkx": 240,
        "acm-gcn": 240,
    }
    assert rows[0].run_id == "actor_mlp_erm_geomgcn-0_seed-1"
    assert rows[-1].run_id == (
        "blogcatalog_acm-gcn_full_gaug-public_seed-30"
    )
    assert all(
        (row.preparation in {"last", "head"}) == bool(row.parent_run_id)
        for row in rows
    )
