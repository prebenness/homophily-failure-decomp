"""Prepare, inspect, or run the Actor and BlogCatalog extension panel.

``prepare`` downloads data, verifies the tracked fixed-input registrations,
checks tail support, and verifies the 1,680-run panel. It fits no model.
``status`` and ``run`` delegate to the resumable experiment-grid runner.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = REPO_ROOT / "configs" / "datasets-additional.yaml"
EXPERIMENT_SPEC = REPO_ROOT / "configs" / "experiment-additional-datasets.yaml"
DATASET_MANIFEST_DIR = REPO_ROOT / "manifests" / "additional-datasets"
RUN_MANIFEST = REPO_ROOT / "manifests" / "additional-dataset-runs.jsonl"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "additional-dataset-runs"


def command(script: str, *arguments: object) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        *(str(argument) for argument in arguments),
    ]


def prepare_commands() -> list[list[str]]:
    return [
        command(
            "compute_dataset_stats.py",
            "--config",
            DATASET_CONFIG,
            "--verify",
        ),
        command(
            "check_additional_datasets.py",
            "--config",
            DATASET_CONFIG,
        ),
        command(
            "build_experiment_grid.py",
            "--spec",
            EXPERIMENT_SPEC,
            "--dataset-manifest-dir",
            DATASET_MANIFEST_DIR,
            "--output",
            RUN_MANIFEST,
            "--check",
        ),
    ]


def grid_command(args: argparse.Namespace) -> list[str]:
    invocation = command(
        "run_experiment_grid.py",
        args.command,
        "--manifest",
        RUN_MANIFEST,
        "--spec",
        EXPERIMENT_SPEC,
        "--dataset-config",
        DATASET_CONFIG,
        "--output-root",
        args.output_root,
    )
    for option, values in (
        ("--dataset", args.dataset),
        ("--model", args.model),
        ("--preparation", args.preparation),
        ("--split", args.split),
        ("--seed", args.seed),
    ):
        for value in values:
            invocation.extend([option, str(value)])
    if args.command == "status":
        invocation.extend(["--show", str(args.show)])
        if args.require_complete:
            invocation.append("--require-complete")
    else:
        invocation.extend(["--device", args.device])
        if args.max_runs is not None:
            invocation.extend(["--max-runs", str(args.max_runs)])
        if args.dry_run:
            invocation.append("--dry-run")
        if args.no_require_clean:
            invocation.append("--no-require-clean")
        if args.rerun_invalid:
            invocation.append("--rerun-invalid")
    return invocation


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", action="append", choices=("actor", "blogcatalog"), default=[]
    )
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--preparation",
        action="append",
        choices=("erm", "last", "head", "full"),
        default=[],
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--dry-run", action="store_true", help="print preparation commands only"
    )

    status = subparsers.add_parser("status")
    add_selection_arguments(status)
    status.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    status.add_argument("--show", type=int, default=20)
    status.add_argument("--require-complete", action="store_true")

    run = subparsers.add_parser("run")
    add_selection_arguments(run)
    run.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    run.add_argument("--device", default="cuda")
    run.add_argument("--max-runs", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-require-clean", action="store_true")
    run.add_argument("--rerun-invalid", action="store_true")

    args = parser.parse_args()
    if getattr(args, "show", 0) < 0:
        parser.error("--show must be non-negative")
    if getattr(args, "max_runs", None) is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    invocations = prepare_commands() if args.command == "prepare" else [grid_command(args)]
    for index, invocation in enumerate(invocations, start=1):
        print(f"[{index}/{len(invocations)}] {shlex.join(invocation)}", flush=True)
        if not getattr(args, "dry_run", False):
            subprocess.run(invocation, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
