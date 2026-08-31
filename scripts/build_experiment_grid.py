"""Build or verify the complete 4,200-run experiment manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hfd.experiment_grid import build_experiment_grid, grid_jsonl_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument(
        "--dataset-manifest-dir",
        type=Path,
        default=REPO_ROOT / "manifests" / "datasets",
    )
    parser.add_argument(
        "--model-config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "manifests" / "main-runs.jsonl",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output is byte-identical instead of rewriting it",
    )
    args = parser.parse_args()

    rows, output_files = build_experiment_grid(
        args.spec,
        args.dataset_manifest_dir,
        args.model_config_dir,
    )
    generated = grid_jsonl_text(rows)
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"missing experiment manifest: {args.output}")
        current = args.output.read_text(encoding="utf-8")
        if current != generated:
            raise SystemExit(
                f"experiment manifest is stale: {args.output}; "
                "run this command without --check"
            )
        action = "verified"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated, encoding="utf-8", newline="")
        action = "wrote"

    print(
        f"{action} {len(rows):,} runs in {args.output} "
        f"({len(output_files)} required output files per run)"
    )


if __name__ == "__main__":
    main()
