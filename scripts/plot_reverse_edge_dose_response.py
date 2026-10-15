"""Plot the four reverse-intervention figures from compact numerical summaries.

This is a figure-only convenience. The raw analysis and its checkpoint/graph
checks remain in analyze_reverse_edge_dose_response.py. No reference PDF hash
is required for a fresh scientific reproduction.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil

from plot_edge_dose_response_composite import (
    CONDITIONS, DATASETS, MODELS, REPO_ROOT, SummaryRow, build_figure, plt,
)

PREPARATIONS = ("erm", "last", "head", "full")
DEFAULT_ROOT = REPO_ROOT / "results" / "reverse_edge_perturbation"


def load_summary(path: Path, preparation: str) -> tuple[list[SummaryRow], dict[str, float]]:
    """Require a complete shared dose grid for all models and both conditions."""
    required = {
        "preparation", "dataset", "model", "target_count", "actual_pct",
        "condition", "n_graph_replicates", "n_checkpoints_per_graph",
        "mean_macro_f1", "sd_macro_f1_across_graph_replicates",
    }
    rows = []
    keys = set()
    doses: dict[str, dict[int, float]] = {}
    baselines = {}
    datasets = {name for name, _ in DATASETS}
    models = {name for name, _ in MODELS}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"{path}: unexpected reverse-summary columns")
        for raw in reader:
            dataset, model, condition = raw["dataset"], raw["model"], raw["condition"]
            if (raw["preparation"] != preparation or dataset not in datasets
                    or model not in models or condition not in CONDITIONS):
                raise ValueError(f"{path}: unexpected preparation, dataset, model or condition")
            count = int(raw["target_count"])
            percentage = float(raw["actual_pct"])
            mean = float(raw["mean_macro_f1"])
            sd = float(raw["sd_macro_f1_across_graph_replicates"])
            if (count < 0 or not math.isfinite(percentage) or not 0 <= percentage <= 100
                    or (count == 0) != (percentage == 0)
                    or not math.isfinite(mean) or not 0 <= mean <= 1
                    or not math.isfinite(sd) or sd < 0
                    or int(raw["n_graph_replicates"]) != 5
                    or int(raw["n_checkpoints_per_graph"]) != 30):
                raise ValueError(f"{path}: invalid dose, score or replicate count")
            key = (dataset, model, count, condition)
            if key in keys:
                raise ValueError(f"{path}: duplicate summary row {key}")
            keys.add(key)
            grid = doses.setdefault(dataset, {})
            if count in grid and grid[count] != percentage:
                raise ValueError(f"{path}: inconsistent original-tail percentage")
            grid[count] = percentage
            if count == 0:
                if sd != 0:
                    raise ValueError(f"{path}: baseline standard deviation must be zero")
                baseline_key = (dataset, model)
                if baseline_key in baselines and baselines[baseline_key] != mean:
                    raise ValueError(f"{path}: treatment/control baselines differ")
                baselines[baseline_key] = mean
            rows.append(SummaryRow(dataset, model, percentage, condition, 5, 30, mean, sd))
    if set(doses) != datasets or any(0 not in grid or len(grid) < 2 for grid in doses.values()):
        raise ValueError(f"{path}: missing dataset, baseline or nonzero dose")
    expected = {
        (dataset, model, count, condition)
        for dataset in datasets for model in models
        for count in doses[dataset] for condition in CONDITIONS
    }
    if keys != expected:
        raise ValueError(f"{path}: incomplete six-model/five-dataset dose matrix")
    for grid in doses.values():
        nonzero = sorted((count, percentage) for count, percentage in grid.items() if count)
        denominator = nonzero[0][0] / nonzero[0][1]
        if any(not math.isclose(count / percentage, denominator, rel_tol=1e-10)
               for count, percentage in nonzero):
            raise ValueError(f"{path}: percentages do not use one fixed original-tail denominator")
    return rows, {dataset: max(grid.values()) for dataset, grid in doses.items()}


def plot_summaries(summary_root: Path, output_root: Path, paper_output_dir: Path) -> None:
    # Load and check every input before replacing any output figure.
    inputs = {
        preparation: load_summary(summary_root / preparation / "dose_response_summary.csv", preparation)
        for preparation in PREPARATIONS
    }
    grids = [
        {(r.dataset, r.model, r.p_percent, r.condition) for r in rows}
        for rows, _ in inputs.values()
    ]
    if any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("reverse preparations must use the same graph-dose matrix")
    for preparation, (rows, endpoints) in inputs.items():
        figure = build_figure(rows, x_limits_by_dataset=endpoints,
                              xlabel="Original tail nodes rewired (%)")
        destination = output_root / preparation
        destination.mkdir(parents=True, exist_ok=True)
        name = f"reverse_edge_perturbation_{preparation}"
        pdf_path = destination / f"{name}.pdf"
        figure.savefig(pdf_path, metadata={
            "Title": f"Reverse-direction edge perturbation: {preparation}",
            "Author": "Anonymous", "Creator": "plot_reverse_edge_dose_response.py",
            "CreationDate": None, "ModDate": None,
        })
        figure.savefig(destination / f"{name}.png", dpi=300)
        plt.close(figure)
        paper_output_dir.mkdir(parents=True, exist_ok=True)
        target = paper_output_dir / pdf_path.name
        if target.resolve() != pdf_path.resolve():
            shutil.copyfile(pdf_path, target)
        print(f"Replotted reverse figure: {preparation}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--paper-output-dir", type=Path, default=REPO_ROOT / "paper" / "figures")
    parser.add_argument("--skip-missing", action="store_true",
                        help="report unavailable summaries and retain reference PDFs")
    args = parser.parse_args(argv)
    missing = [p for p in PREPARATIONS
               if not (args.summary_root / p / "dose_response_summary.csv").is_file()]
    if missing:
        message = ("Reverse figures were not regenerated: compact summaries are missing for "
                   + ", ".join(missing) + ". Existing PDFs remain reference copies.")
        if args.skip_missing:
            print(message)
            return
        raise FileNotFoundError(message)
    plot_summaries(args.summary_root, args.output_root, args.paper_output_dir)


if __name__ == "__main__":
    main()
