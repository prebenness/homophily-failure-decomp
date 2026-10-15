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

    assert counts["figures"] == 28
    assert counts["tables"] == 27


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


def minimal_inventory(tmp_path: Path):
    module = load_module()
    module.REPO_ROOT = tmp_path
    (tmp_path / "paper/figures").mkdir(parents=True)
    (tmp_path / "paper/figures/current.pdf").write_bytes(b"%PDF-fresh-run")
    payload = {
        "schema_version": 1,
        "expected_counts": {"figures": 1, "tables": 0},
        "figures": [{"label": "fig:current", "artifact_path": "paper/figures/current.pdf",
                     "sha256": "0" * 64}],
        "tables": [],
    }
    manifest = tmp_path / "paper/artifacts.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return module, manifest, payload


def test_fresh_scientific_output_does_not_require_reference_hash(tmp_path: Path) -> None:
    module, manifest, _ = minimal_inventory(tmp_path)
    assert module.verify_inventory(manifest)["figures"] == 1
    with pytest.raises(RuntimeError, match="release asset hash differs"):
        module.verify_inventory(manifest, verify_reference_hashes=True)


def test_manuscript_check_detects_asset_missing_from_repo_and_inventory(tmp_path: Path) -> None:
    module, manifest, _ = minimal_inventory(tmp_path)
    manuscript = tmp_path / "manuscript"
    (manuscript / "figures").mkdir(parents=True)
    (manuscript / "figures/current.pdf").write_bytes(b"%PDF-reference")
    (manuscript / "figures/new.pdf").write_bytes(b"%PDF-new")
    (manuscript / "main.tex").write_text(
        "\\input{results}\n% \\input{inactive}\n", encoding="utf-8")
    (manuscript / "results.tex").write_text(
        "\\includegraphics{figures/current.pdf}\\label{fig:current}\n"
        "\\includegraphics{figures/new.pdf}\\label{fig:new}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unregistered manuscript figure or table: fig:new"):
        module.verify_inventory(manifest, manuscript_root=manuscript)


def test_reference_only_exception_is_explicit(tmp_path: Path) -> None:
    module, manifest, payload = minimal_inventory(tmp_path)
    payload["figures"][0].update(
        generated_path="results/current.pdf", reference_only_until_regenerated=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing regenerated artifact"):
        module.verify_inventory(manifest, require_generated=True)
    counts = module.verify_inventory(manifest, require_generated=True, allow_reference_only=True)
    assert counts["reference_only"] == 1
