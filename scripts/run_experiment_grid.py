"""Inspect or execute the tracked experiment grid.

The executor is local and sequential. It is intended as a reproducible
reference implementation and as a building block for machine-specific batch
schedulers. Completed runs are skipped. Selecting a recovery preparation also
selects its required ERM parent.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hfd.experiment_grid import (  # noqa: E402
    RunState,
    execution_plan,
    inspect_run,
    inspect_runs,
    read_experiment_grid,
    select_runs,
    state_counts,
    train_command,
)

DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "main-runs.jsonl"
DEFAULT_SPEC = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_DATASET_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "runs"
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "train_run.py"


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
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


def _load_output_contract(spec_path: Path) -> tuple[str, ...]:
    import yaml

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    files = spec.get("run_output_files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise ValueError(f"{spec_path}: invalid run_output_files")
    return tuple(files)


def _load_default_device(spec_path: Path) -> str:
    import yaml

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    device = spec.get("default_device")
    if not isinstance(device, str) or not device:
        raise ValueError(f"{spec_path}: invalid default_device")
    return device


def _selected(args: argparse.Namespace):
    rows = read_experiment_grid(args.manifest)
    selected = select_runs(
        rows,
        datasets=args.dataset,
        models=args.model,
        preparations=args.preparation,
        splits=args.split,
        seeds=args.seed,
        include_dependencies=True,
    )
    if not selected:
        raise SystemExit("selection matched no experiment runs")
    return selected


def _print_summary(inspections) -> Counter[RunState]:
    counts = state_counts(inspections)
    total = len(inspections)
    print(
        f"selected={total:,} "
        + " ".join(f"{state.value}={counts[state]:,}" for state in RunState)
    )
    return counts


def status(args: argparse.Namespace) -> None:
    rows = _selected(args)
    required_files = _load_output_contract(args.spec)
    inspections = inspect_runs(rows, args.output_root, required_files)
    counts = _print_summary(inspections)

    if args.show:
        shown = 0
        for inspection in inspections:
            if inspection.state is RunState.COMPLETE:
                continue
            suffix = f" ({inspection.detail})" if inspection.detail else ""
            print(f"{inspection.state.value:>10}  {inspection.run.run_id}{suffix}")
            shown += 1
            if shown == args.show:
                remaining = sum(
                    item.state is not RunState.COMPLETE for item in inspections
                ) - shown
                if remaining:
                    print(f"... {remaining:,} further non-complete runs")
                break

    if args.require_complete and counts[RunState.COMPLETE] != len(inspections):
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
    rows = _selected(args)
    required_files = _load_output_contract(args.spec)
    inspections = inspect_runs(rows, args.output_root, required_files)
    _print_summary(inspections)
    plan = execution_plan(rows, inspections, rerun_invalid=args.rerun_invalid)
    initial_states = {
        inspection.run.run_id: inspection.state for inspection in inspections
    }
    device = args.device or _load_default_device(args.spec)
    if args.max_runs is not None:
        plan = plan[: args.max_runs]
    print(f"to_execute={len(plan):,}")

    completed_this_call: set[str] = set()
    for index, row in enumerate(plan, start=1):
        if row.parent_run_id:
            parent = inspect_run(
                next(item for item in rows if item.run_id == row.parent_run_id),
                args.output_root,
                required_files,
            )
            if (
                parent.state is not RunState.COMPLETE
                and row.parent_run_id not in completed_this_call
                and not args.dry_run
            ):
                raise RuntimeError(
                    f"{row.run_id}: parent ceased to be complete before execution"
                )
        command = train_command(
            sys.executable,
            TRAINING_SCRIPT,
            row,
            output_root=args.output_root,
            device=device,
            dataset_config=args.dataset_config,
            experiment_manifest=args.manifest,
            replace_existing=initial_states[row.run_id] is not RunState.MISSING,
            require_clean=args.require_clean,
        )
        rendered = shlex.join(command)
        print(f"[{index:,}/{len(plan):,}] {rendered}")
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        inspection = inspect_run(row, args.output_root, required_files)
        if inspection.state is not RunState.COMPLETE:
            raise RuntimeError(
                f"{row.run_id}: training command returned without a complete "
                f"run ({inspection.state.value}: {inspection.detail})"
            )
        completed_this_call.add(row.run_id)


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    common.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    common.add_argument(
        "--dataset-config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG,
        help="dataset registry passed to each atomic training run",
    )
    common.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory containing one subdirectory per run",
    )
    _add_selection_arguments(common)

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", parents=[common])
    status_parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="maximum number of non-complete run identifiers to print",
    )
    status_parser.add_argument("--require-complete", action="store_true")
    status_parser.set_defaults(function=status)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument(
        "--device",
        help=(
            "PyTorch device; defaults to configs/experiment.yaml:"
            "default_device"
        ),
    )
    run_parser.add_argument(
        "--require-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require a clean Git tree for training provenance (default: true)",
    )
    run_parser.add_argument("--max-runs", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--rerun-invalid",
        action="store_true",
        help="allow an existing run whose config conflicts with the manifest",
    )
    run_parser.set_defaults(function=run)

    args = parser.parse_args()
    if getattr(args, "show", 1) < 0:
        parser.error("--show must be non-negative")
    max_runs = getattr(args, "max_runs", None)
    if max_runs is not None and max_runs < 0:
        parser.error("--max-runs must be non-negative")
    args.output_root = args.output_root.resolve()
    args.manifest = args.manifest.resolve()
    args.spec = args.spec.resolve()
    args.dataset_config = args.dataset_config.resolve()
    args.function(args)


if __name__ == "__main__":
    main()
