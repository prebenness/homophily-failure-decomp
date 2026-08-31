"""Baseline validation debug runs: ERM only, standard test metric only.

Trains one ERM run per requested dataset (first split, given seed) using the
exact training path of the experiment, evaluates ONLY the dataset's standard
whole-test metric, and compares it against the reference table in
manifests/reference_numbers.yaml. No local-homophily data (bins, tails,
per-bin scores) is computed or stored: these runs debug that models train
correctly and are excluded from the scientific record.

Usage:
    python scripts/debug_baseline.py --model gcn [--datasets pubmed ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from hfd.datasets import DATASET_NAMES, load_dataset  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.training import TrainConfig, evaluate_split, set_seed, train_erm  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    refs = yaml.safe_load(
        (REPO_ROOT / "manifests" / "reference_numbers.yaml").read_text(encoding="utf-8")
    )
    margin = refs["gate_margin_points"]
    model_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / f"{args.model}.yaml").read_text(encoding="utf-8")
    )
    train_cfg = TrainConfig(**model_cfg["train"])
    ds_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8")
    )
    device = torch.device(args.device)
    # Tracked: debug results are part of the record (unanchored results
    # become their cells' recorded references).
    out_dir = REPO_ROOT / "results" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.datasets:
        print(f"=== {name} / {args.model} / seed {args.seed} ===", flush=True)
        bundle = load_dataset(name, REPO_ROOT / ds_config["data_root"])
        split = bundle.splits[0]

        simple = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
        x = bundle.x.to(device)
        edge_index = torch.from_numpy(simple).to(device)
        y = torch.from_numpy(bundle.y).to(device)
        train_idx = torch.from_numpy(split.train_idx).to(device)
        val_idx = torch.from_numpy(split.val_idx).to(device)
        test_idx = torch.from_numpy(split.test_idx).to(device)

        set_seed(args.seed)
        model = build_model(
            args.model,
            int(x.shape[1]),
            bundle.num_classes,
            model_cfg,
            num_nodes=bundle.num_nodes,
        ).to(device)
        start = time.time()
        result = train_erm(model, x, edge_index, y, train_idx, val_idx, train_cfg)
        model.load_state_dict(result.best_state)
        _, test_acc = evaluate_split(model, x, edge_index, y, test_idx)
        seconds = time.time() - start

        test_pct = 100.0 * test_acc
        anchor = refs["anchors"].get(name, {}).get(args.model)
        majority_pct = 100.0 * float(
            np.bincount(bundle.y[split.test_idx]).max() / split.test_idx.size
        )
        if anchor is None:
            # Unanchored cell: sanity bounds only; parity tests and
            # cross-checking cover correctness. Result becomes the
            # recorded reference for this cell.
            flag = test_pct <= majority_pct + 5.0
            deviation = None
            grade = "unanchored"
            verdict = "sanity-FLAG" if flag else "sanity-ok (becomes reference)"
        else:
            deviation = test_pct - anchor["value"]
            flag = abs(deviation) > margin
            grade = "anchored"
            verdict = "FLAG" if flag else "ok"

        row = {
            "dataset": name,
            "model": args.model,
            "split": split.split_id,
            "seed": args.seed,
            "test_metric_pct": round(test_pct, 2),
            "reference_pct": None if anchor is None else anchor["value"],
            "grade": grade,
            "deviation_pct": None if deviation is None else round(deviation, 2),
            "gate_margin": margin if anchor else None,
            "majority_class_pct": round(majority_pct, 2),
            "verdict": verdict,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "train_seconds": round(seconds, 1),
            "code_version": git_commit(),
        }
        rows.append(row)
        (out_dir / f"{name}_{args.model}_seed-{args.seed}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, indent=2), flush=True)

    print("\nsummary:")
    print(f"{'dataset':<16}{'test %':>8}{'ref %':>8}{'dev':>7}  {'grade':<6}{'verdict'}")
    for r in rows:
        ref = "-" if r["reference_pct"] is None else f"{r['reference_pct']:.2f}"
        dev = "-" if r["deviation_pct"] is None else f"{r['deviation_pct']:+.1f}"
        print(
            f"{r['dataset']:<16}{r['test_metric_pct']:>8.2f}{ref:>8}{dev:>7}  "
            f"{r['grade']:<6}{r['verdict']}"
        )


if __name__ == "__main__":
    main()
