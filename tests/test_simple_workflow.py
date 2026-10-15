from argparse import Namespace
from pathlib import Path
import runpy
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> dict:
    return runpy.run_path(str(REPO_ROOT / "scripts" / name))


def test_download_entry_point_runs_registered_dataset_verification() -> None:
    module = load_script("download_datasets.py")
    command = module["build_command"]()

    assert Path(command[1]).name == "compute_dataset_stats.py"
    assert command[2:] == ["--verify"]


def test_download_entry_point_prepares_both_dataset_panels() -> None:
    module = load_script("download_datasets.py")
    commands = module["build_commands"]()

    assert [Path(command[1]).name for command in commands] == [
        "compute_dataset_stats.py",
        "run_additional_datasets.py",
    ]
    assert commands[1][2:] == ["prepare"]


def test_download_entry_point_does_not_duplicate_an_explicit_mode() -> None:
    module = load_script("download_datasets.py")

    command = module["build_command"](["--update-registration"])

    assert command[2:] == ["--update-registration"]


def test_training_entry_point_covers_main_and_appendix_panels() -> None:
    module = load_script("train_models.py")
    args = Namespace(
        device="cuda",
        main_output_root=module["DEFAULT_MAIN_ROOT"],
        additional_output_root=module["DEFAULT_ADDITIONAL_ROOT"],
        dfr_output_root=module["DEFAULT_DFR_ROOT"],
        fg_output_root=module["DEFAULT_FG_ROOT"],
        main_only=False,
        dataset=[],
        model=[],
        preparation=[],
        split=[],
        seed=[],
        max_runs=None,
        no_require_clean=False,
        dry_run=False,
    )
    commands = module["build_commands"](args)
    scripts = [Path(command[1]).name for command in commands]

    assert scripts == [
        "build_experiment_grid.py",
        "run_experiment_grid.py",
        "run_additional_datasets.py",
        "run_dfr_pubmed.py",
        "run_fg_ccdb_pubmed.py",
    ]
    assert str(module["DEFAULT_MAIN_ROOT"]) in commands[1]
    assert str(module["DEFAULT_ADDITIONAL_ROOT"]) in commands[2]
    assert str(module["DEFAULT_DFR_ROOT"]) in commands[3]
    assert str(module["DEFAULT_FG_ROOT"]) in commands[4]
    assert all("--device" in command for command in commands[1:])


def test_analysis_entry_point_uses_repository_local_run_roots() -> None:
    module = load_script("analyse_results.py")
    args = Namespace(
        raw_root=module["DEFAULT_MAIN_ROOT"],
        additional_raw_root=module["DEFAULT_ADDITIONAL_ROOT"],
        dfr_raw_root=module["DEFAULT_DFR_ROOT"],
        fg_raw_root=module["DEFAULT_FG_ROOT"],
        edge_dose_response_final_root=module[
            "DEFAULT_EDGE_DOSE_RESPONSE_FINAL_ROOT"
        ],
        edge_dose_response_recovery_root=module[
            "DEFAULT_EDGE_DOSE_RESPONSE_RECOVERY_ROOT"
        ],
        validate_raw_arrays=False,
        derived_only=False,
        dry_run=False,
    )
    command = module["build_command"](args)

    assert Path(command[1]).name == "reproduce_paper.py"
    assert command[2:] == [
        "--raw-root",
        str(module["DEFAULT_MAIN_ROOT"]),
        "--additional-raw-root",
        str(module["DEFAULT_ADDITIONAL_ROOT"]),
        "--dfr-raw-root",
        str(module["DEFAULT_DFR_ROOT"]),
        "--fg-raw-root",
        str(module["DEFAULT_FG_ROOT"]),
        "--edge-dose-response-final-root",
        str(module["DEFAULT_EDGE_DOSE_RESPONSE_FINAL_ROOT"]),
        "--edge-dose-response-recovery-root",
        str(module["DEFAULT_EDGE_DOSE_RESPONSE_RECOVERY_ROOT"]),
        *[
            value
            for name, default in module["EXTRA_ROOT_DEFAULTS"].items()
            for value in (f"--{name.replace('_', '-')}", str(default))
        ],
    ]


def test_analysis_dry_run_does_not_require_existing_run_roots(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyse_results.py"),
            "--raw-root",
            str(tmp_path / "main"),
            "--additional-raw-root",
            str(tmp_path / "additional"),
            "--dfr-raw-root",
            str(tmp_path / "dfr"),
            "--fg-raw-root",
            str(tmp_path / "fg"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "recalculate_macro_f1.py" in result.stdout
