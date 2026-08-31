from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = runpy.run_path(
    str(REPO_ROOT / "scripts" / "sync_additional_paper_artifacts.py")
)


def test_release_copies_match_every_declared_additional_artifact() -> None:
    mappings = NAMESPACE["artifact_mappings"]()

    assert len(NAMESPACE["FIGURE_MAPPINGS"]) == 9
    assert len(NAMESPACE["TABLE_FILENAMES"]) == 12
    assert len(mappings) == 21
    assert NAMESPACE["sync_artifacts"](check=True) == 21
