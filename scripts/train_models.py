"""Run every training panel required by the paper.

With no arguments, this runs the 4,200 original-dataset experiments, the
1,680 Actor/BlogCatalog experiments, and then the PubMed DFR and FG-CCDB
appendix panels. All outputs are resumable and are written below
``artifacts/`` in this repository.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROOT = REPO_ROOT / "artifacts" / "runs"
DEFAULT_ADDITIONAL_ROOT = REPO_ROOT / "artifacts" / "additional-dataset-runs"
DEFAULT_DFR_ROOT = REPO_ROOT / "artifacts" / "dfr-pubmed-runs"
DEFAULT_FG_ROOT = REPO_ROOT / "artifacts" / "fg-ccdb-pubmed-runs"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = REPO_ROOT / "artifacts" / "data"
ARRAY_ROOT = REPO_ROOT / "artifacts" / "manifests"


def command(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        *(str(argument) for argument in arguments),
    ]


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    main_command = command(
        "run_experiment_grid.py",
        "run",
        "--output-root",
        args.main_output_root,
    )
    main_command.extend(["--device", args.device])
    if args.max_runs is not None:
        main_command.extend(["--max-runs", str(args.max_runs)])
    if args.no_require_clean:
        main_command.append("--no-require-clean")
    for option, values in (
        ("--dataset", args.dataset),
        ("--model", args.model),
        ("--preparation", args.preparation),
        ("--split", args.split),
        ("--seed", args.seed),
    ):
        for value in values:
            main_command.extend([option, str(value)])

    commands = [
        command("build_experiment_grid.py", "--check"),
        main_command,
    ]
    if args.main_only:
        return commands

    commands.append(
        command(
            "run_additional_datasets.py",
            "run",
            "--output-root",
            args.additional_output_root,
            "--device",
            args.device,
        )
    )

    common = [
        "--data-root",
        DATA_ROOT,
        "--array-root",
        ARRAY_ROOT,
        "--parent-run-root",
        args.main_output_root,
    ]
    dfr_command = command(
        "run_dfr_pubmed.py",
        *common,
        "--output-root",
        args.dfr_output_root,
    )
    fg_command = command(
        "run_fg_ccdb_pubmed.py",
        *common,
        "--output-root",
        args.fg_output_root,
    )
    dfr_command.extend(["--device", args.device])
    fg_command.extend(["--device", args.device])
    commands.extend([dfr_command, fg_command])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"PyTorch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--main-output-root",
        type=Path,
        default=DEFAULT_MAIN_ROOT,
    )
    parser.add_argument(
        "--additional-output-root",
        type=Path,
        default=DEFAULT_ADDITIONAL_ROOT,
    )
    parser.add_argument(
        "--dfr-output-root",
        type=Path,
        default=DEFAULT_DFR_ROOT,
    )
    parser.add_argument(
        "--fg-output-root",
        type=Path,
        default=DEFAULT_FG_ROOT,
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="run only the original registered 4,200-run panel",
    )
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--preparation",
        action="append",
        choices=("erm", "last", "head", "full"),
        default=[],
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--no-require-clean",
        action="store_true",
        help="allow a main-panel smoke run from a dirty worktree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without executing them",
    )
    args = parser.parse_args()

    selected = any(
        (args.dataset, args.model, args.preparation, args.split, args.seed)
    )
    if (selected or args.max_runs is not None or args.no_require_clean) and (
        not args.main_only
    ):
        parser.error(
            "selection, --max-runs, and --no-require-clean require --main-only"
        )
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for index, invocation in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(invocation)}", flush=True)
        if not args.dry_run:
            subprocess.run(invocation, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
