"""Download the registered datasets and reconstruct their fixed inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_command(extra_arguments: list[str] | None = None) -> list[str]:
    extra_arguments = list(extra_arguments or [])
    mode = (
        []
        if {"--verify", "--update-registration"} & set(extra_arguments)
        else ["--verify"]
    )
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compute_dataset_stats.py"),
        *mode,
        *extra_arguments,
    ]


def build_commands(extra_arguments: list[str] | None = None) -> list[list[str]]:
    """Build the complete dataset-preparation workflow.

    Extra low-level arguments apply only to the original-panel registration;
    the no-argument public entry point also prepares Actor and BlogCatalog.
    """

    extra_arguments = list(extra_arguments or [])
    commands = [build_command(extra_arguments)]
    if not extra_arguments:
        commands.append(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_additional_datasets.py"),
                "prepare",
            ]
        )
    return commands


def main() -> None:
    for invocation in build_commands(sys.argv[1:]):
        subprocess.run(invocation, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
