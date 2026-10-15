"""Regenerate the released analyses, data-derived tables, and figures.

The tracked release figures and tables are stored under ``paper/``. This
command regenerates the corresponding analysis products from the registered
raw run stores and the validated Appendix J expansion output. Use
``--derived-only`` to rebuild outputs that depend only on the tracked corrected
summaries. The full node-array audit is optional because it reads the complete
raw store; request it with ``--validate-raw-arrays``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RECOVERY_PREPARATIONS = ("last", "head", "full")


def command(script: str, *arguments: object) -> list[str]:
    return [
        PYTHON,
        str(REPO_ROOT / "scripts" / script),
        *(str(argument) for argument in arguments),
    ]


def build_commands(
    raw_root: Path | None,
    *,
    additional_raw_root: Path | None = None,
    edge_dose_response_final_root: Path | None = None,
    edge_dose_response_recovery_root: Path | None = None,
    edge_dose_response_pubmed_root: Path | None = None,
    edge_dose_response_large_root: Path | None = None,
    reverse_edge_output_root: Path | None = None,
    reverse_edge_graph_root: Path | None = None,
    derived_only: bool,
    dfr_raw_root: Path | None = None,
    fg_raw_root: Path | None = None,
    validate_raw_arrays: bool = False,
    include_training_times: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = []
    if not derived_only:
        if raw_root is None:
            raise ValueError("--raw-root is required unless --derived-only is used")
        if additional_raw_root is None:
            raise ValueError(
                "--additional-raw-root is required unless --derived-only is used"
            )
        if edge_dose_response_final_root is None:
            raise ValueError(
                "--edge-dose-response-final-root is required unless "
                "--derived-only is used"
            )
        if edge_dose_response_recovery_root is None:
            raise ValueError(
                "--edge-dose-response-recovery-root is required unless "
                "--derived-only is used"
            )
        for name, root in (
            ("dfr-raw-root", dfr_raw_root),
            ("fg-raw-root", fg_raw_root),
            ("edge-dose-response-pubmed-root", edge_dose_response_pubmed_root),
            ("edge-dose-response-large-root", edge_dose_response_large_root),
            ("reverse-edge-output-root", reverse_edge_output_root),
            ("reverse-edge-graph-root", reverse_edge_graph_root),
        ):
            if root is None:
                raise ValueError(f"--{name} is required unless --derived-only is used")
        if validate_raw_arrays:
            commands.extend(
                [
                    command(
                        "verify_raw_store.py",
                        "--raw-root",
                        raw_root,
                        "--level",
                        "arrays",
                    ),
                    command(
                        "verify_raw_store.py",
                        "--raw-root",
                        additional_raw_root,
                        "--panel",
                        REPO_ROOT / "manifests" / "additional-dataset-runs.jsonl",
                        "--dataset-manifest-dir",
                        REPO_ROOT / "manifests" / "additional-datasets",
                        "--level",
                        "arrays",
                    ),
                ]
            )
        commands.extend(
            [
                command("recalculate_macro_f1.py", "--raw-root", raw_root),
                command("analyze_rq1_accuracy.py", "--raw-root", raw_root),
                command(
                    "plot_rq1_composite.py",
                    "--raw-root",
                    raw_root,
                    "--bins",
                    12,
                    "--metric",
                    "accuracy",
                ),
                command(
                    "plot_rq1_composite.py",
                    "--raw-root",
                    raw_root,
                    "--bins",
                    12,
                    "--metric",
                    "macro-f1",
                ),
                command("analyze_rq2_within_class.py", "--raw-root", raw_root),
                command(
                    "recalculate_macro_f1.py",
                    "--dataset-mode",
                    "additional",
                    "--additional-raw-root",
                    additional_raw_root,
                ),
                command(
                    "analyze_rq1_accuracy.py",
                    "--dataset-mode",
                    "additional",
                    "--additional-raw-root",
                    additional_raw_root,
                ),
                command(
                    "plot_rq1_composite.py",
                    "--dataset-mode",
                    "additional",
                    "--additional-raw-root",
                    additional_raw_root,
                    "--bins",
                    12,
                    "--metric",
                    "accuracy",
                ),
                command(
                    "plot_rq1_composite.py",
                    "--dataset-mode",
                    "additional",
                    "--additional-raw-root",
                    additional_raw_root,
                    "--bins",
                    12,
                    "--metric",
                    "macro-f1",
                ),
                command(
                    "analyze_rq2_within_class.py",
                    "--dataset-mode",
                    "additional",
                    "--additional-raw-root",
                    additional_raw_root,
                ),
            ]
        )
        if include_training_times:
            commands.extend([
                command("analyze_training_compute.py", "--raw-dir", raw_root),
                command("analyze_training_compute.py", "--dataset-mode", "additional",
                        "--additional-raw-root", additional_raw_root),
            ])
        if dfr_raw_root is not None:
            commands.append(
                command(
                    "analyze_dfr_pubmed.py",
                    "--dfr-raw-root",
                    dfr_raw_root,
                    "--canonical-raw-root",
                    raw_root,
                    "--output-root",
                    REPO_ROOT
                    / "results"
                    / "main"
                    / "rq5"
                    / "dfr_pubmed_sensitivity",
                    "--models",
                    "acm-gcn",
                    "gcn",
                    "graphsage-mean",
                    "--modes",
                    "val",
                )
            )
        if fg_raw_root is not None:
            commands.append(
                command(
                    "analyze_fg_ccdb.py",
                    "--fg-raw-root",
                    fg_raw_root,
                    "--canonical-raw-root",
                    raw_root,
                )
            )
        # Regenerate all six two-model cells from this run before combining
        # them with the 24-cell expansion. Never silently reuse release scores.
        pubmed_summary_root = REPO_ROOT / "results" / "edge_perturbation_dose_response_pubmed"
        large_summary_root = REPO_ROOT / "results" / "edge_perturbation_dose_response_large"
        commands.append(command(
            "analyze_pubmed_edge_dose_response.py", "--input-root",
            edge_dose_response_pubmed_root, "--output-root", pubmed_summary_root,
        ))
        assert edge_dose_response_large_root is not None
        for dataset in ("ogbn-arxiv", "arxiv-year"):
            commands.append(command(
                "analyze_large_edge_dose_response.py", "--dataset", dataset,
                "--input-root", edge_dose_response_large_root / dataset,
                "--output-root", large_summary_root / dataset,
            ))
        commands.append(
            command(
                "analyze_edge_dose_response_full.py",
                "--final-root",
                edge_dose_response_final_root,
                "--pubmed-summary-root", pubmed_summary_root,
                "--large-summary-root", large_summary_root,
            )
        )

        commands.append(command(
            "analyze_reverse_edge_dose_response.py",
            "--input-root", reverse_edge_output_root,
            "--graph-root", reverse_edge_graph_root,
            "--output-root", REPO_ROOT / "results" / "reverse_edge_perturbation",
            "--paper-output-dir", REPO_ROOT / "paper" / "figures",
        ))
        for preparation in RECOVERY_PREPARATIONS:
            commands.append(
                command(
                    "analyze_edge_dose_response_full.py",
                    "--preparation",
                    preparation,
                    "--final-root",
                    edge_dose_response_recovery_root
                    / "actual"
                    / preparation
                    / "final",
                    "--output-root",
                    REPO_ROOT
                    / "results"
                    / f"edge_perturbation_dose_response_preparation_{preparation}",
                )
            )

    commands.extend(
        [
            command(
                "plot_rq1_rq2_paired_forest.py",
                "--output-dir",
                REPO_ROOT / "results" / "main" / "figures" / "rq1-rq2"
                / "unequal-y",
                "--figure-stem",
                "rq1_rq2_unadjusted_and_class_adjusted_macro_f1_tail_gaps_unequal_y",
                "--y-scale",
                "dataset",
                "--dpi",
                400,
            ),
            command("plot_rq3_mlp_contrasts.py"),
            command("analyze_rq4_model_families.py"),
            command("plot_rq4_class_adjusted_model_families.py"),
            command("plot_compact_main_figures.py"),
            command("analyze_rq5_preparations.py"),
            command(
                "plot_rq1_rq2_paired_forest.py",
                "--dataset-mode",
                "additional",
                "--y-scale",
                "dataset",
                "--dpi",
                400,
            ),
            command(
                "plot_rq3_mlp_contrasts.py",
                "--dataset-mode",
                "additional",
            ),
            command(
                "analyze_rq4_model_families.py",
                "--dataset-mode",
                "additional",
            ),
            command(
                "plot_rq4_class_adjusted_model_families.py",
                "--dataset-mode",
                "additional",
            ),
            command(
                "plot_compact_main_figures.py",
                "--dataset-mode",
                "additional",
            ),
            command(
                "analyze_rq5_preparations.py",
                "--dataset-mode",
                "additional",
            ),
        ]
    )

    if not derived_only:
        assert raw_root is not None
        commands.append(
            command("plot_rq5_tail_vs_non_tail.py", "--raw-root", raw_root)
        )
        assert additional_raw_root is not None
        commands.append(
            command(
                "plot_rq5_tail_vs_non_tail.py",
                "--dataset-mode",
                "additional",
                "--raw-root",
                additional_raw_root,
            )
        )

    edge_figure_commands = [command("plot_edge_dose_response_composite.py")]
    for preparation in RECOVERY_PREPARATIONS:
        analysis_root = (
            REPO_ROOT
            / "results"
            / f"edge_perturbation_dose_response_preparation_{preparation}"
        )
        edge_figure_commands.append(
            command(
                "plot_edge_dose_response_composite.py",
                "--summary",
                analysis_root / "dose_response_summary.csv",
                "--output-dir",
                REPO_ROOT
                / "results"
                / f"edge_perturbation_dose_response_composite_{preparation}",
                "--paper-output",
                REPO_ROOT
                / "paper"
                / "figures"
                / f"edge_perturbation_dose_response_{preparation}.pdf",
            )
        )
    edge_figure_commands.append(command("plot_edge_dose_response_composite.py", "--main-text"))
    if derived_only:
        edge_figure_commands.append(command(
            "plot_reverse_edge_dose_response.py", "--skip-missing",
        ))

    commands.extend(
        [
            command("plot_rq5_pubmed_gcn_recovery_panel.py"),
            command("generate_local_homophily_bin_figure.py"),
            *edge_figure_commands,
            command("render_paper_tables.py"),
            command("render_rq2_recovered_share_tables.py"),
            command(
                "generate_local_homophily_bin_figure.py",
                "--dataset-mode",
                "additional",
            ),
            command(
                "render_paper_tables.py",
                "--dataset-mode",
                "additional",
            ),
            command("sync_additional_paper_artifacts.py"),
            command(
                "verify_paper_artifacts.py",
                "--require-generated",
                *(["--allow-reference-only"] if derived_only else []),
            ),
        ]
    )
    return commands


def run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for index, invocation in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(invocation)}", flush=True)
        if not dry_run:
            subprocess.run(invocation, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="directory containing the 4,200 registered raw run directories",
    )
    parser.add_argument(
        "--additional-raw-root",
        type=Path,
        help="directory containing the 1,680 Actor/BlogCatalog raw runs",
    )
    parser.add_argument(
        "--edge-dose-response-final-root",
        type=Path,
        help=(
            "directory containing the validated baseline_rows.jsonl, "
            "paired_rows.jsonl, and validation.json expansion outputs"
        ),
    )
    parser.add_argument(
        "--edge-dose-response-recovery-root",
        type=Path,
        help=(
            "top-level validated recovery output containing "
            "actual/{last,head,full}/final"
        ),
    )
    parser.add_argument(
        "--derived-only",
        action="store_true",
        help=(
            "rebuild summary-dependent figures and tables from tracked "
            "corrected results; skip all node-level raw-store analyses"
        ),
    )
    parser.add_argument(
        "--dfr-raw-root",
        type=Path,
        help=(
            "directory containing the 90 PubMed "
            "DFR runs"
        ),
    )
    parser.add_argument(
        "--fg-raw-root",
        type=Path,
        help=(
            "directory containing the 270 PubMed "
            "FG-CCDB runs"
        ),
    )
    for name in (
        "edge-dose-response-pubmed-root", "edge-dose-response-large-root",
        "reverse-edge-output-root", "reverse-edge-graph-root",
    ):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument(
        "--include-training-times", action="store_true",
        help="also summarize recorded training times, if training histories are available",
    )
    parser.add_argument(
        "--validate-raw-arrays",
        action="store_true",
        help="run the optional full node-array audit before analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ordered commands without executing them",
    )
    args = parser.parse_args()

    root_names = (
        "raw_root", "additional_raw_root", "dfr_raw_root", "fg_raw_root",
        "edge_dose_response_final_root", "edge_dose_response_recovery_root",
        "edge_dose_response_pubmed_root", "edge_dose_response_large_root",
        "reverse_edge_output_root", "reverse_edge_graph_root",
    )
    roots = {
        name: getattr(args, name).expanduser().resolve() if getattr(args, name) else None
        for name in root_names
    }
    if args.derived_only and any(root is not None for root in roots.values()):
        parser.error("raw-run roots cannot be used with --derived-only")
    if args.derived_only and (args.validate_raw_arrays or args.include_training_times):
        parser.error("raw-array validation and training times require raw-run roots")
    if not args.dry_run:
        for name, root in roots.items():
            if root is not None and not root.is_dir():
                raise FileNotFoundError(f"{name.replace('_', '-')} not found: {root}")
    commands = build_commands(
        **roots,
        derived_only=args.derived_only,
        validate_raw_arrays=args.validate_raw_arrays,
        include_training_times=args.include_training_times,
    )
    run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
