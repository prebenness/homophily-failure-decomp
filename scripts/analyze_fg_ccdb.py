"""Validate and summarise the focused PubMed FG-CCDB recovery panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from hfd.fg_ccdb import compute_fg_ccdb_weights  # noqa: E402
from hfd.metrics import gap_audit_metrics  # noqa: E402


MODELS = ["graphsage-mean", "gcn", "acm-gcn"]
MODEL_LABELS = {
    "graphsage-mean": "GraphSAGE-mean",
    "gcn": "GCN",
    "acm-gcn": "ACM-GCN",
}
PREPARATIONS = ["last", "head", "full"]
SPLITS = [f"geomgcn-{index}" for index in range(10)]
SEEDS = [1, 2, 3]
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20_260_716
TOLERANCE = 1e-12
PUBLIC_RAW_ROOTS = {
    "fg": "PATH-TO-FG-CCDB-RUNS",
    "canonical_erm_and_groupdro": "PATH-TO-CANONICAL-RUNS",
}
SCRIPT_SHA256 = hashlib.sha256(
    Path(__file__)
    .read_text(encoding="utf-8")
    .replace("\r\n", "\n")
    .encode("utf-8")
).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bootstrap_seed(model: str, comparison: str) -> int:
    key = f"{BOOTSTRAP_SEED}:pubmed:{model}:{comparison}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def paired_bootstrap(
    values: np.ndarray, model: str, comparison: str
) -> tuple[float, float, float]:
    rng = np.random.default_rng(bootstrap_seed(model, comparison))
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size)
    )
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    )
    smaller_tail_count = min(
        int(np.count_nonzero(bootstrap_means <= 0.0)),
        int(np.count_nonzero(bootstrap_means >= 0.0)),
    )
    p_value = min(
        1.0, 2.0 * smaller_tail_count / BOOTSTRAP_RESAMPLES
    )
    return float(lower), float(upper), float(p_value)


def expected_id(
    model: str,
    preparation: str,
    split: str,
    seed: int,
    *,
    fg: bool,
) -> str:
    tag = f"fg-ccdb-{preparation}" if fg else preparation
    return f"pubmed_{model}_{tag}_{split}_seed-{seed}"


def validate_config(
    config: dict,
    *,
    run_id: str,
    model: str,
    preparation: str,
    split: str,
    seed: int,
    fg: bool,
) -> None:
    expected = {
        "run_id": run_id,
        "dataset": "pubmed",
        "model": model,
        "preparation": preparation,
        "split": split,
        "seed": seed,
    }
    if fg:
        expected["recovery_objective"] = "fg-ccdb"
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{run_id}: configuration mismatch: {mismatches}")


def score_run(
    run_dir: Path,
    *,
    manifest: dict,
    model: str,
    preparation: str,
    split: str,
    seed: int,
    fg: bool,
) -> tuple[dict, dict | None]:
    run_id = expected_id(model, preparation, split, seed, fg=fg)
    if run_dir.name != run_id:
        raise RuntimeError(f"Expected {run_id}, got directory {run_dir.name}")
    required = {
        "config.json",
        "metrics.json",
        "history.json",
        "checkpoint.pt",
        "run_data.npz",
    }
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        raise RuntimeError(f"{run_id}: missing files {missing}")

    config = read_json(run_dir / "config.json")
    validate_config(
        config,
        run_id=run_id,
        model=model,
        preparation=preparation,
        split=split,
        seed=seed,
        fg=fg,
    )
    expected_checksums = manifest["array_checksums_sha256"]
    recorded_checksums = config["fixed_input_checksums"]
    for key in ("labels", "merged_bin", "tail_mask", f"{split}.test"):
        if recorded_checksums.get(key) != expected_checksums[key]:
            raise RuntimeError(f"{run_id}: recorded checksum mismatch for {key}")

    with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        predictions = saved["predictions"]
        train_idx = saved["train_idx"]
        test_idx = saved["test_idx"]
        merged_bin = saved["merged_bin"]
        tail_mask = saved["tail_mask"]
        if fg:
            objective_train_idx = saved["objective_train_idx"]
            objective_train_weights = saved["objective_train_weights"]
        else:
            objective_train_idx = None
            objective_train_weights = None

    arrays_to_check = {
        "labels": labels,
        "merged_bin": merged_bin,
        "tail_mask": tail_mask,
        f"{split}.test": test_idx,
    }
    for key, array in arrays_to_check.items():
        if array_sha256(array) != expected_checksums[key]:
            raise RuntimeError(f"{run_id}: saved array checksum mismatch for {key}")

    bin_intervals = [
        definition["interval"]
        for definition in manifest["local_homophily"]["merged_bins"]
    ]
    scores = gap_audit_metrics(
        y_true=labels,
        y_pred=predictions,
        test_idx=test_idx,
        merged_bin=merged_bin,
        tail_mask=tail_mask,
        bin_intervals=bin_intervals,
        standard_metric="accuracy",
    )
    stored_metrics = read_json(run_dir / "metrics.json")
    for key in ("whole_test_macro_f1", "tail_macro_f1", "tail_gap"):
        if abs(float(stored_metrics[key]) - float(scores[key])) > TOLERANCE:
            raise RuntimeError(f"{run_id}: stored {key} differs from predictions")

    weight_diagnostics = None
    if fg:
        eligible = merged_bin[train_idx] >= 0
        expected_train_idx = train_idx[eligible]
        if not np.array_equal(objective_train_idx, expected_train_idx):
            raise RuntimeError(f"{run_id}: objective training indices differ")
        recomputed = compute_fg_ccdb_weights(
            labels[objective_train_idx],
            merged_bin[objective_train_idx],
            num_classes=int(manifest["graph"]["num_classes"]),
            num_bins=len(manifest["local_homophily"]["merged_bins"]),
        )
        if not np.allclose(
            objective_train_weights, recomputed.weights, rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(f"{run_id}: saved FG weights differ from recomputation")
        recorded = config["objective_diagnostics"]
        if recorded["cell_counts"] != recomputed.diagnostics["cell_counts"]:
            raise RuntimeError(f"{run_id}: recorded FG contingency table differs")
        if (
            recorded["balance_audit"]["max_abs_joint_target_error"]
            > 1e-12
        ):
            raise RuntimeError(f"{run_id}: FG balance audit exceeds tolerance")
        weight_diagnostics = recomputed.diagnostics

    checkpoint = stored_metrics["checkpoint"]
    return (
        {
            "run_id": run_id,
            "model": model,
            "preparation": preparation,
            "split": split,
            "seed": seed,
            "whole_test_macro_f1": float(scores["whole_test_macro_f1"]),
            "tail_macro_f1": float(scores["tail_macro_f1"]),
            "tail_gap": float(scores["tail_gap"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "train_seconds": float(checkpoint["train_seconds"]),
            "training_code_version": config["code_version"],
        },
        weight_diagnostics,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    fg_rows: list[dict],
    erm_rows: dict[tuple[str, str, int], dict],
    groupdro_rows: dict[tuple[str, str, str, int], dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    by_cell = defaultdict(list)
    for row in fg_rows:
        key = (row["model"], row["preparation"])
        erm = erm_rows[(row["model"], row["split"], row["seed"])]
        groupdro = groupdro_rows[
            (row["model"], row["preparation"], row["split"], row["seed"])
        ]
        by_cell[key].append(
            {
                **row,
                "erm_tail_macro_f1": erm["tail_macro_f1"],
                "tail_paired_gain": row["tail_macro_f1"] - erm["tail_macro_f1"],
                "whole_test_paired_change": (
                    row["whole_test_macro_f1"] - erm["whole_test_macro_f1"]
                ),
                "tail_gap_paired_change": row["tail_gap"] - erm["tail_gap"],
                "groupdro_tail_macro_f1": groupdro["tail_macro_f1"],
                "fg_minus_groupdro_tail_macro_f1": (
                    row["tail_macro_f1"] - groupdro["tail_macro_f1"]
                ),
            }
        )

    run_rows = []
    aggregate_rows = []
    for model in MODELS:
        for preparation in PREPARATIONS:
            rows = sorted(
                by_cell[(model, preparation)],
                key=lambda row: (row["split"], row["seed"]),
            )
            if len(rows) != 30:
                raise RuntimeError(
                    f"{model}/{preparation}: found {len(rows)} pairs; expected 30"
                )
            run_rows.extend(rows)
            tail_gains = np.asarray([row["tail_paired_gain"] for row in rows])
            whole_changes = np.asarray(
                [row["whole_test_paired_change"] for row in rows]
            )
            gap_changes = np.asarray(
                [row["tail_gap_paired_change"] for row in rows]
            )
            objective_differences = np.asarray(
                [row["fg_minus_groupdro_tail_macro_f1"] for row in rows]
            )
            gain_lower, gain_upper, gain_p = paired_bootstrap(
                tail_gains, model, f"fg-ccdb-{preparation}-gain"
            )
            diff_lower, diff_upper, diff_p = paired_bootstrap(
                objective_differences,
                model,
                f"fg-ccdb-minus-groupdro-{preparation}",
            )
            aggregate_rows.append(
                {
                    "model": model,
                    "preparation": preparation,
                    "n_pairs": 30,
                    "mean_fg_tail_macro_f1": float(
                        np.mean([row["tail_macro_f1"] for row in rows])
                    ),
                    "mean_tail_paired_gain_vs_erm": float(tail_gains.mean()),
                    "sample_std_tail_paired_gain": float(
                        tail_gains.std(ddof=1)
                    ),
                    "gain_bootstrap_ci_lower": gain_lower,
                    "gain_bootstrap_ci_upper": gain_upper,
                    "gain_bootstrap_p_value": gain_p,
                    "gain_ci_excludes_zero": (
                        gain_lower > 0.0 or gain_upper < 0.0
                    ),
                    "mean_whole_test_paired_change_vs_erm": float(
                        whole_changes.mean()
                    ),
                    "mean_tail_gap_paired_change_vs_erm": float(
                        gap_changes.mean()
                    ),
                    "mean_fg_minus_groupdro_tail_macro_f1": float(
                        objective_differences.mean()
                    ),
                    "objective_difference_ci_lower": diff_lower,
                    "objective_difference_ci_upper": diff_upper,
                    "objective_difference_p_value": diff_p,
                    "objective_difference_ci_excludes_zero": (
                        diff_lower > 0.0 or diff_upper < 0.0
                    ),
                }
            )

    aggregate_by_cell = {
        (row["model"], row["preparation"]): row for row in aggregate_rows
    }
    recovered_rows = []
    for model in MODELS:
        full = aggregate_by_cell[(model, "full")]
        denominator = float(full["mean_tail_paired_gain_vs_erm"])
        qualifies = (
            denominator >= 0.03 and bool(full["gain_ci_excludes_zero"])
        )
        for preparation in ("last", "head"):
            numerator = float(
                aggregate_by_cell[(model, preparation)][
                    "mean_tail_paired_gain_vs_erm"
                ]
            )
            recovered_rows.append(
                {
                    "model": model,
                    "preparation": preparation,
                    "mean_tail_paired_gain_vs_erm": numerator,
                    "full_mean_tail_paired_gain_vs_erm": denominator,
                    "full_gain_bootstrap_ci_lower": full[
                        "gain_bootstrap_ci_lower"
                    ],
                    "full_gain_bootstrap_ci_upper": full[
                        "gain_bootstrap_ci_upper"
                    ],
                    "qualifies": qualifies,
                    "recovered_share": (
                        numerator / denominator if qualifies else None
                    ),
                }
            )
    return run_rows, aggregate_rows, recovered_rows


def markdown_report(
    aggregate_rows: list[dict],
    recovered_rows: list[dict],
    weight_rows: list[dict],
) -> str:
    lines = [
        "# FG-CCDB PubMed recovery sensitivity",
        "",
        "This exploratory sensitivity repeats A_last, A_head, and A_full for "
        "PubMed/GraphSAGE-mean, PubMed/GCN, and PubMed/ACM-GCN. The training "
        "objective is deterministic full-batch importance-weighted "
        "cross-entropy with weights fitted on each training split:",
        "",
        "`w(bin, class) = n_bin * n_class / (N * n_bin,class)`.",
        "",
        "This targets independence between the fixed local-homophily bin and "
        "the class label. It does not target independence for continuous "
        "local homophily. No sampler, clipping, smoothing, or additional "
        "cell-count factor is used.",
        "",
        "## Paired results",
        "",
        "| Model | Preparation | FG gain over ERM [95% CI] | FG minus GroupDRO [95% CI] | Whole-test change vs ERM |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['preparation']} | "
            f"{row['mean_tail_paired_gain_vs_erm']:.4f} "
            f"[{row['gain_bootstrap_ci_lower']:.4f}, "
            f"{row['gain_bootstrap_ci_upper']:.4f}] | "
            f"{row['mean_fg_minus_groupdro_tail_macro_f1']:.4f} "
            f"[{row['objective_difference_ci_lower']:.4f}, "
            f"{row['objective_difference_ci_upper']:.4f}] | "
            f"{row['mean_whole_test_paired_change_vs_erm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recovered shares",
            "",
            "| Model | Preparation | Recovered share |",
            "|---|---|---:|",
        ]
    )
    for row in recovered_rows:
        value = (
            f"{row['recovered_share']:.4f}"
            if row["recovered_share"] is not None
            else "not interpreted"
        )
        lines.append(
            f"| {MODEL_LABELS[row['model']]} | {row['preparation']} | {value} |"
        )

    minimum_ess = min(row["effective_sample_size_fraction"] for row in weight_rows)
    maximum_weight = max(row["maximum_weight"] for row in weight_rows)
    minimum_cell = min(row["minimum_cell_count"] for row in weight_rows)
    lines.extend(
        [
            "",
            "## Weight audit",
            "",
            f"All ten splits have complete support. The minimum class-by-bin "
            f"training count is {minimum_cell}. The largest weight is "
            f"{maximum_weight:.4f}. Effective sample size is at least "
            f"{minimum_ess:.2%} of the eligible training set.",
            "",
            "The local-homophily bins are oracle quantities constructed from "
            "true node and neighbour labels over the complete graph. This is "
            "therefore a retrospective decomposition diagnostic, rather than "
            "a deployable semi-supervised debiasing procedure.",
            "",
            "The raw run archive is supplied at runtime with "
            "`--fg-raw-root`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fg-raw-root", type=Path, required=True)
    parser.add_argument("--canonical-raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "fg_ccdb",
    )
    args = parser.parse_args()
    fg_root = args.fg_raw_root.resolve()
    canonical_root = args.canonical_raw_root.resolve()
    output_root = args.output_root.resolve()
    if not fg_root.is_dir() or not canonical_root.is_dir():
        parser.error("Both raw roots must exist")

    manifest = read_json(REPO_ROOT / "manifests" / "datasets" / "pubmed.json")
    fg_rows = []
    erm_rows = {}
    groupdro_rows = {}
    weight_by_split = {}

    for model in MODELS:
        for split in SPLITS:
            for seed in SEEDS:
                erm_id = expected_id(model, "erm", split, seed, fg=False)
                erm_row, _ = score_run(
                    canonical_root / erm_id,
                    manifest=manifest,
                    model=model,
                    preparation="erm",
                    split=split,
                    seed=seed,
                    fg=False,
                )
                erm_rows[(model, split, seed)] = erm_row

        for preparation in PREPARATIONS:
            for split in SPLITS:
                for seed in SEEDS:
                    fg_id = expected_id(
                        model, preparation, split, seed, fg=True
                    )
                    fg_row, diagnostics = score_run(
                        fg_root / fg_id,
                        manifest=manifest,
                        model=model,
                        preparation=preparation,
                        split=split,
                        seed=seed,
                        fg=True,
                    )
                    fg_rows.append(fg_row)

                    signature = {
                        "cell_counts": diagnostics["cell_counts"],
                        "cell_weights": diagnostics["cell_weights"],
                        "weight_summary": diagnostics["weight_summary"],
                    }
                    if split in weight_by_split and weight_by_split[split] != signature:
                        raise RuntimeError(
                            f"{split}: FG weights vary by model, preparation, or seed"
                        )
                    weight_by_split[split] = signature

                    groupdro_id = expected_id(
                        model, preparation, split, seed, fg=False
                    )
                    groupdro_row, _ = score_run(
                        canonical_root / groupdro_id,
                        manifest=manifest,
                        model=model,
                        preparation=preparation,
                        split=split,
                        seed=seed,
                        fg=False,
                    )
                    groupdro_rows[(model, preparation, split, seed)] = groupdro_row

    expected_fg_directories = {
        expected_id(model, preparation, split, seed, fg=True)
        for model in MODELS
        for preparation in PREPARATIONS
        for split in SPLITS
        for seed in SEEDS
    }
    observed_fg_directories = {
        path.name for path in fg_root.iterdir() if path.is_dir()
    }
    if observed_fg_directories != expected_fg_directories:
        raise RuntimeError(
            "FG raw panel differs from the expected 270 runs: "
            f"missing={sorted(expected_fg_directories - observed_fg_directories)}, "
            f"unexpected={sorted(observed_fg_directories - expected_fg_directories)}"
        )

    run_rows, aggregate_rows, recovered_rows = aggregate(
        fg_rows, erm_rows, groupdro_rows
    )
    weight_rows = []
    for split in SPLITS:
        signature = weight_by_split[split]
        summary = signature["weight_summary"]
        positive_counts = np.asarray(signature["cell_counts"])
        positive_counts = positive_counts[positive_counts > 0]
        weight_rows.append(
            {
                "split": split,
                "n_training_nodes": int(sum(map(sum, signature["cell_counts"]))),
                "minimum_cell_count": int(positive_counts.min()),
                "minimum_weight": float(summary["minimum"]),
                "maximum_weight": float(summary["maximum"]),
                "effective_sample_size": float(summary["effective_sample_size"]),
                "effective_sample_size_fraction": float(
                    summary["effective_sample_size_fraction"]
                ),
            }
        )

    write_csv(output_root / "run_metrics.csv", run_rows)
    write_csv(output_root / "paired_results.csv", aggregate_rows)
    write_csv(output_root / "recovered_shares.csv", recovered_rows)
    write_csv(output_root / "weight_diagnostics.csv", weight_rows)
    (output_root / "README.md").write_text(
        markdown_report(aggregate_rows, recovered_rows, weight_rows),
        encoding="utf-8",
    )
    analysis_manifest = {
        "schema_version": 1,
        "analysis": "fg-ccdb-pubmed-recovery-sensitivity",
        "analysis_code": {
            "script": "scripts/analyze_fg_ccdb.py",
            "script_sha256": SCRIPT_SHA256,
            "numpy_version": np.__version__,
        },
        "scope": {
            "dataset": "pubmed",
            "models": MODELS,
            "preparations": PREPARATIONS,
            "splits": SPLITS,
            "seeds": SEEDS,
            "fg_runs": len(fg_rows),
            "erm_pairs": len(erm_rows),
            "groupdro_comparison_runs": len(groupdro_rows),
        },
        "metric": {
            "primary": "tail macro-F1",
            "label_set": (
                "classes present in each evaluated subset's true labels, "
                "fixed before predictions are evaluated"
            ),
            "zero_division": 0,
        },
        "fg_objective": {
            "spurious_variable": "fixed merged local-homophily bin",
            "target": "empirical P(bin) * empirical P(label) on eligible training nodes",
            "weight_formula": "n_bin * n_class / (N * n_bin_class)",
            "implementation": (
                "deterministic full-batch importance-weighted cross-entropy"
            ),
            "sampler": None,
            "support_policy": "fail on any empty active bin-by-class cell",
        },
        "paired_bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "two-sided percentile 95%",
            "quantile_method": "linear",
            "global_seed": BOOTSTRAP_SEED,
            "multiple_comparison_correction": None,
        },
        "raw_roots": PUBLIC_RAW_ROOTS,
        "training_code_versions": sorted(
            {row["training_code_version"] for row in fg_rows}
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated 270 FG runs and wrote the focused analysis to {output_root}"
    )


if __name__ == "__main__":
    main()
