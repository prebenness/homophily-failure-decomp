"""Regenerate every paper analysis, figure, and data-derived table."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_ROOT = REPO_ROOT / "artifacts" / "runs"
DEFAULT_ADDITIONAL_ROOT = REPO_ROOT / "artifacts" / "additional-dataset-runs"
DEFAULT_DFR_ROOT = REPO_ROOT / "artifacts" / "dfr-pubmed-runs"
DEFAULT_FG_ROOT = REPO_ROOT / "artifacts" / "fg-ccdb-pubmed-runs"
DEFAULT_EDGE_DOSE_RESPONSE_FINAL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "edge-dose-response-expansion"
    / "actual"
    / "final"
)
DEFAULT_EDGE_DOSE_RESPONSE_RECOVERY_ROOT = (
    REPO_ROOT / "outputs" / "edge-dose-response-recovery"
)
EXTRA_ROOT_DEFAULTS = {
    "edge_dose_response_pubmed_root": REPO_ROOT / "artifacts" / "edge-dose-response" / "legacy" / "pubmed" / "actual",
    "edge_dose_response_large_root": REPO_ROOT / "artifacts" / "edge-dose-response" / "legacy" / "large" / "actual",
    "reverse_edge_output_root": REPO_ROOT / "artifacts" / "reverse-edge-inference",
    "reverse_edge_graph_root": REPO_ROOT / "artifacts" / "reverse-edge-graphs",
}


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reproduce_paper.py"),
    ]
    if args.derived_only:
        command.append("--derived-only")
    else:
        command.extend(
            [
                "--raw-root",
                str(args.raw_root),
                "--additional-raw-root",
                str(args.additional_raw_root),
                "--dfr-raw-root",
                str(args.dfr_raw_root),
                "--fg-raw-root",
                str(args.fg_raw_root),
            ]
        )
        edge_dose_response_final_root = getattr(
            args, "edge_dose_response_final_root", None
        )
        if edge_dose_response_final_root is not None:
            command.extend(
                [
                    "--edge-dose-response-final-root",
                    str(edge_dose_response_final_root),
                ]
            )
        edge_dose_response_recovery_root = getattr(
            args, "edge_dose_response_recovery_root", None
        )
        if edge_dose_response_recovery_root is not None:
            command.extend(
                [
                    "--edge-dose-response-recovery-root",
                    str(edge_dose_response_recovery_root),
                ]
            )
        if args.validate_raw_arrays:
            command.append("--validate-raw-arrays")
        for name, default in EXTRA_ROOT_DEFAULTS.items():
            command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name, default))])
        if getattr(args, "include_training_times", False):
            command.append("--include-training-times")
    if args.dry_run:
        command.append("--dry-run")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_MAIN_ROOT,
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        default=DEFAULT_ADDITIONAL_ROOT,
    )
    parser.add_argument(
        "--dfr-raw-root",
        type=Path,
        default=DEFAULT_DFR_ROOT,
    )
    parser.add_argument(
        "--fg-raw-root",
        type=Path,
        default=DEFAULT_FG_ROOT,
    )
    parser.add_argument(
        "--edge-dose-response-final-root",
        type=Path,
        default=DEFAULT_EDGE_DOSE_RESPONSE_FINAL_ROOT,
        help=(
            "validated consolidated Appendix J expansion outputs produced "
            "by run_edge_dose_response_expansion.py"
        ),
    )
    parser.add_argument(
        "--edge-dose-response-recovery-root",
        type=Path,
        default=DEFAULT_EDGE_DOSE_RESPONSE_RECOVERY_ROOT,
        help=(
            "validated three-preparation Appendix J output produced by the "
            "portable recovery evaluator"
        ),
    )
    for name, default in EXTRA_ROOT_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--include-training-times", action="store_true")
    parser.add_argument(
        "--validate-raw-arrays",
        action="store_true",
        help="run the optional full node-array audit before analysis",
    )
    parser.add_argument(
        "--derived-only",
        action="store_true",
        help="rebuild outputs that use the tracked compact summaries",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.derived_only and (args.validate_raw_arrays or args.include_training_times):
        parser.error("raw-array validation and training times cannot use --derived-only")
    return args


def main() -> None:
    args = parse_args()
    subprocess.run(
        build_command(args),
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
