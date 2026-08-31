from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "reproduce_paper.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reproduce_paper", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_build_contains_every_current_analysis_stage(tmp_path: Path) -> None:
    module = load_module()
    dfr_root = tmp_path / "dfr"
    fg_root = tmp_path / "fg"
    commands = module.build_commands(
        tmp_path,
        additional_raw_root=tmp_path / "additional",
        edge_dose_response_final_root=tmp_path / "edge-final",
        edge_dose_response_recovery_root=tmp_path / "edge-recovery",
        derived_only=False,
        dfr_raw_root=dfr_root,
        fg_raw_root=fg_root,
    )
    scripts = [Path(invocation[1]).name for invocation in commands]

    assert scripts[0] == "recalculate_macro_f1.py"
    assert "verify_raw_store.py" not in scripts
    assert "recalculate_macro_f1.py" in scripts
    assert "plot_rq1_composite.py" in scripts
    rq1_commands = [
        invocation
        for invocation in commands
        if Path(invocation[1]).name == "plot_rq1_composite.py"
    ]
    assert len(rq1_commands) == 4
    assert {
        invocation[invocation.index("--metric") + 1]
        for invocation in rq1_commands
    } == {"accuracy", "macro-f1"}
    assert "analyze_rq2_within_class.py" in scripts
    assert any(
        "--dataset-mode" in invocation
        and invocation[invocation.index("--dataset-mode") + 1] == "additional"
        for invocation in rq1_commands
    )
    assert "analyze_dfr_pubmed.py" in scripts
    dfr_command = commands[scripts.index("analyze_dfr_pubmed.py")]
    assert str(dfr_root) in dfr_command
    assert str(tmp_path) in dfr_command
    assert "analyze_fg_ccdb.py" in scripts
    fg_command = commands[scripts.index("analyze_fg_ccdb.py")]
    assert str(fg_root) in fg_command
    assert str(tmp_path) in fg_command
    assert "plot_rq3_mlp_contrasts.py" in scripts
    assert "analyze_rq4_model_families.py" in scripts
    assert "plot_rq5_tail_vs_non_tail.py" in scripts
    assert "generate_local_homophily_bin_figure.py" in scripts
    assert scripts.count("analyze_edge_dose_response_full.py") == 4
    edge_commands = [
        invocation
        for invocation in commands
        if Path(invocation[1]).name == "analyze_edge_dose_response_full.py"
    ]
    assert str(tmp_path / "edge-final") in edge_commands[0]
    for preparation, invocation in zip(
        module.RECOVERY_PREPARATIONS, edge_commands[1:], strict=True
    ):
        assert invocation[invocation.index("--preparation") + 1] == preparation
        assert str(
            tmp_path / "edge-recovery" / "actual" / preparation / "final"
        ) in invocation
    edge_plot_commands = [
        invocation
        for invocation in commands
        if Path(invocation[1]).name == "plot_edge_dose_response_composite.py"
    ]
    assert len(edge_plot_commands) == 4
    for preparation, invocation in zip(
        module.RECOVERY_PREPARATIONS, edge_plot_commands[1:], strict=True
    ):
        assert str(
            module.REPO_ROOT
            / "results"
            / f"edge_perturbation_dose_response_preparation_{preparation}"
            / "dose_response_summary.csv"
        ) in invocation
        assert invocation[-1].endswith(
            f"edge_perturbation_dose_response_{preparation}.pdf"
        )
    assert scripts[-5:] == [
        "render_paper_tables.py",
        "generate_local_homophily_bin_figure.py",
        "render_paper_tables.py",
        "sync_additional_paper_artifacts.py",
        "verify_paper_artifacts.py",
    ]


def test_derived_build_does_not_read_raw_archive() -> None:
    module = load_module()
    commands = module.build_commands(None, derived_only=True)
    scripts = [Path(invocation[1]).name for invocation in commands]

    assert "recalculate_macro_f1.py" not in scripts
    assert "plot_rq5_tail_vs_non_tail.py" not in scripts
    assert "analyze_dfr_pubmed.py" not in scripts
    assert "analyze_fg_ccdb.py" not in scripts
    assert scripts.count("plot_rq3_mlp_contrasts.py") == 2
    assert scripts.count("render_paper_tables.py") == 2
    assert "sync_additional_paper_artifacts.py" in scripts
    assert "generate_local_homophily_bin_figure.py" in scripts
    assert scripts.count("plot_edge_dose_response_composite.py") == 4
    assert scripts[-1] == "verify_paper_artifacts.py"


def test_full_build_can_use_tracked_dfr_summary(
    tmp_path: Path,
) -> None:
    module = load_module()
    commands = module.build_commands(
        tmp_path,
        additional_raw_root=tmp_path / "additional",
        edge_dose_response_final_root=tmp_path / "edge-final",
        edge_dose_response_recovery_root=tmp_path / "edge-recovery",
        derived_only=False,
    )
    scripts = [Path(invocation[1]).name for invocation in commands]

    assert "analyze_dfr_pubmed.py" not in scripts
    assert "analyze_fg_ccdb.py" not in scripts


def test_full_build_can_run_the_optional_array_audit(
    tmp_path: Path,
) -> None:
    module = load_module()
    commands = module.build_commands(
        tmp_path,
        additional_raw_root=tmp_path / "additional",
        edge_dose_response_final_root=tmp_path / "edge-final",
        edge_dose_response_recovery_root=tmp_path / "edge-recovery",
        derived_only=False,
        validate_raw_arrays=True,
    )

    assert [Path(commands[index][1]).name for index in (0, 1)] == [
        "verify_raw_store.py",
        "verify_raw_store.py",
    ]
    assert commands[0][-2:] == ["--level", "arrays"]
    assert commands[1][-2:] == ["--level", "arrays"]
    assert "additional-dataset-runs.jsonl" in str(commands[1])


def test_full_build_requires_raw_root() -> None:
    module = load_module()
    with pytest.raises(ValueError, match="raw-root"):
        module.build_commands(None, derived_only=False)


def test_full_build_requires_additional_raw_root(tmp_path: Path) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="additional-raw-root"):
        module.build_commands(tmp_path, derived_only=False)


def test_full_build_requires_edge_dose_response_final_root(
    tmp_path: Path,
) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="edge-dose-response-final-root"):
        module.build_commands(
            tmp_path,
            additional_raw_root=tmp_path / "additional",
            edge_dose_response_recovery_root=tmp_path / "edge-recovery",
            derived_only=False,
        )


def test_full_build_requires_edge_dose_response_recovery_root(
    tmp_path: Path,
) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="edge-dose-response-recovery-root"):
        module.build_commands(
            tmp_path,
            additional_raw_root=tmp_path / "additional",
            edge_dose_response_final_root=tmp_path / "edge-final",
            derived_only=False,
        )
