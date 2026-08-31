from argparse import Namespace
from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script() -> dict:
    return runpy.run_path(str(REPO_ROOT / "scripts" / "run_additional_datasets.py"))


def test_prepare_is_data_only_and_checks_the_registered_panel() -> None:
    module = load_script()
    commands = module["prepare_commands"]()

    assert [Path(command[1]).name for command in commands] == [
        "compute_dataset_stats.py",
        "check_additional_datasets.py",
        "build_experiment_grid.py",
    ]
    assert all("train_run.py" not in command for command in commands)
    assert "--verify" in commands[0]
    assert "--check" in commands[2]


def test_run_command_uses_separate_registry_manifest_and_output() -> None:
    module = load_script()
    args = Namespace(
        command="run",
        output_root=Path("artifacts/extra-runs"),
        dataset=["actor"],
        model=["gcn"],
        preparation=[],
        split=[],
        seed=[],
        device="cuda:0",
        max_runs=None,
        dry_run=True,
        no_require_clean=False,
        rerun_invalid=False,
    )

    command = module["grid_command"](args)

    assert Path(command[1]).name == "run_experiment_grid.py"
    assert command[2] == "run"
    assert str(module["RUN_MANIFEST"]) in command
    assert str(module["DATASET_CONFIG"]) in command
    assert str(Path("artifacts/extra-runs")) in command
    assert command[-1] == "--dry-run"
