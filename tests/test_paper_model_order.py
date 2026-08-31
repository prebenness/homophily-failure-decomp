import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
PAPER_GRAPH_MODEL_ORDER = PAPER_MODEL_ORDER[1:]
ORDER_DEFINITIONS = (
    ("scripts/plot_rq1_model_comparison.py", "MODEL_ORDER", PAPER_MODEL_ORDER),
    ("scripts/plot_rq1_rq2_paired_forest.py", "MODEL_ORDER", PAPER_MODEL_ORDER),
    ("scripts/plot_compact_main_figures.py", "MODEL_ORDER", PAPER_MODEL_ORDER),
    ("scripts/render_paper_tables.py", "MODEL_ORDER", PAPER_MODEL_ORDER),
    (
        "scripts/analyze_rq5_preparations.py",
        "PLOT_MODEL_ORDER",
        PAPER_GRAPH_MODEL_ORDER,
    ),
    (
        "scripts/plot_rq5_tail_vs_non_tail.py",
        "MODEL_ORDER",
        PAPER_GRAPH_MODEL_ORDER,
    ),
)


def literal_assignment(path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{path}: no literal assignment for {name}")


@pytest.mark.parametrize("relative_path,name,expected", ORDER_DEFINITIONS)
def test_paper_generators_share_canonical_model_order(
    relative_path: str,
    name: str,
    expected: tuple[str, ...],
) -> None:
    assert literal_assignment(REPO_ROOT / relative_path, name) == expected
