from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_paper_artifacts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_paper_artifacts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_release_artifact_inventory_is_complete() -> None:
    module = load_module()
    counts = module.verify_inventory(REPO_ROOT / "paper" / "artifacts.json")

    assert counts["figures"] == 23
    assert counts["tables"] == 25


def test_inventory_rejects_undeclared_release_figure(tmp_path: Path) -> None:
    module = load_module()
    module.REPO_ROOT = tmp_path

    figures = []
    figure_directory = tmp_path / "paper" / "figures"
    figure_directory.mkdir(parents=True)
    for index in range(11):
        relative_path = f"paper/figures/figure-{index}.pdf"
        (tmp_path / relative_path).write_bytes(b"pdf")
        figures.append(
            {
                "label": f"fig:{index}",
                "artifact_path": relative_path,
                "kind": "versioned-source",
            }
        )

    tables = []
    table_directory = tmp_path / "paper" / "generated" / "tables"
    table_directory.mkdir(parents=True)
    for index in range(13):
        label = f"tab:{index}"
        relative_path = f"paper/generated/tables/table-{index}.tex"
        (tmp_path / relative_path).write_text(
            f"\\label{{{label}}}\n",
            encoding="utf-8",
        )
        tables.append(
            {
                "label": label,
                "kind": "generated",
                "generated_path": relative_path,
            }
        )

    manifest = tmp_path / "paper" / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "figures": figures,
                "tables": tables,
            }
        ),
        encoding="utf-8",
    )
    (figure_directory / "leftover.pdf").write_bytes(b"old")

    with pytest.raises(RuntimeError, match="undeclared release figure"):
        module.verify_inventory(manifest)
