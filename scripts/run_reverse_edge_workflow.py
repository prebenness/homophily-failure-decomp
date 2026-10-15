"""Run the portable reverse graph and checkpoint-inference workflow.

This sequential reference runner uses existing checkpoints and never trains a
model. It may take substantial compute time. Inspect --dry-run first; the
individual commands can also be scheduled at the execution site.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("pubmed", "ogbn-arxiv", "roman-empire", "amazon-ratings", "arxiv-year")
MODELS = ("gcn", "graphsage-mean", "h2gcn", "gpr-gnn", "acm-gcn", "linkx")
PREPARATIONS = ("erm", "last", "head", "full")


def command(script, *args):
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *(str(value) for value in args)]


def build_commands(args):
    commands = []
    inputs = ["--data-root", args.data_root, "--array-root", args.array_root,
              "--manifest-root", args.manifest_root]
    for dataset in DATASETS:
        graph_args = ["--dataset", dataset, "--output-root", args.graph_root]
        commands.append(command("run_reverse_edge_graphs.py", "prepare", *graph_args, *inputs))
        for stage in ("smoke", "generate", "verify"):
            commands.append(command("run_reverse_edge_graphs.py", stage, *graph_args))
    snapshot = args.output_root / "input-authentication.json"
    commands.append(command("authenticate_reverse_edge_inputs.py", "authenticate",
                            "--checkpoint-root", args.checkpoint_root,
                            "--graph-root", args.graph_root, *inputs, "--snapshot", snapshot))
    common = ["--graph-root", args.graph_root, "--output-root", args.output_root,
              "--experiment-revision", args.experiment_revision]
    commands.append(command("run_reverse_edge_inference.py", "authenticate", *common, *inputs))
    for preparation in PREPARATIONS:
        checkpoint_args = [*common, "--checkpoint-root", args.checkpoint_root,
                           "--preparation", preparation, "--input-snapshot", snapshot]
        for dataset in DATASETS:
            for model in MODELS:
                evaluate_args = [*checkpoint_args, *inputs, "--dataset", dataset,
                                 "--model", model, "--backend-id", args.backend_id,
                                 "--device", args.device]
                commands.append(command("run_reverse_edge_inference.py", "evaluate",
                                        *evaluate_args, "--phase", "smoke",
                                        "--max-checkpoints", 1, "--max-pairs", 1))
                commands.append(command("run_reverse_edge_inference.py", "evaluate",
                                        *evaluate_args, "--phase", "actual"))
        commands.append(command("run_reverse_edge_inference.py", "validate", *checkpoint_args))
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--graph-root", type=Path, default=Path("artifacts/reverse-edge-graphs"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/reverse-edge-inference"))
    parser.add_argument("--data-root", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--array-root", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/datasets"))
    parser.add_argument("--experiment-revision", help="defaults to the current Git revision")
    parser.add_argument("--backend-id", default="pytorch",
                        help="portable name distinguishing the evaluation environment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.experiment_revision is None:
        args.experiment_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if len(args.experiment_revision) != 40 or any(c not in "0123456789abcdef" for c in args.experiment_revision):
        parser.error("experiment revision must be a complete lowercase Git commit")
    for name, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    commands = build_commands(args)
    for index, invocation in enumerate(commands, 1):
        if Path(invocation[1]).name == "authenticate_reverse_edge_inputs.py":
            print("Checking the complete planned graph panel before checkpoint inference.", flush=True)
            if not args.dry_run:
                from analyze_reverse_edge_dose_response import load_graph_contracts

                load_graph_contracts(args.graph_root)
        print(f"[{index}/{len(commands)}] {shlex.join(invocation)}", flush=True)
        if not args.dry_run:
            subprocess.run(invocation, cwd=REPO_ROOT, check=True)
    if not args.dry_run:
        print("Reverse inference is complete. Run scripts/analyse_results.py to rebuild the paper analyses.")


if __name__ == "__main__":
    main()
