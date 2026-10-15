from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sync_main_paper_figures", REPO_ROOT / "scripts" / "sync_main_paper_figures.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_repo(tmp_path: Path):
    module = load_module()
    module.REPO_ROOT = tmp_path
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/artifacts.json").write_bytes(
        (REPO_ROOT / "paper/artifacts.json").read_bytes()
    )
    for source, destination in module.figure_mappings().items():
        (tmp_path / source).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / source).write_bytes(b"fresh-" + source.name.encode())
        (tmp_path / destination).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / destination).write_bytes(b"old-publication-copy")
    return module


def test_full_sync_replaces_all_eight_old_publication_figures(isolated_repo) -> None:
    module = isolated_repo
    assert module.sync_figures() == 8
    for source, destination in module.figure_mappings().items():
        assert (module.REPO_ROOT / destination).read_bytes() == (
            module.REPO_ROOT / source
        ).read_bytes()


def test_derived_sync_preserves_the_three_raw_only_figures(isolated_repo) -> None:
    module = isolated_repo
    derived = module.figure_mappings(derived_only=True)
    for source in module.figure_mappings().keys() - derived.keys():
        (module.REPO_ROOT / source).unlink()
    assert module.sync_figures(derived_only=True) == 5
    for source, destination in module.figure_mappings().items():
        actual = (module.REPO_ROOT / destination).read_bytes()
        if source in derived:
            assert actual == (module.REPO_ROOT / source).read_bytes()
        else:
            assert actual == b"old-publication-copy"


def test_missing_generated_figure_fails_before_replacing_any_copy(isolated_repo) -> None:
    module = isolated_repo
    mappings = module.figure_mappings()
    (module.REPO_ROOT / list(mappings)[-1]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing generated figures"):
        module.sync_figures()
    assert all(
        (module.REPO_ROOT / destination).read_bytes() == b"old-publication-copy"
        for destination in mappings.values()
    )
