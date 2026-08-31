"""Recovery-code validation debug runs (work-plan Step 2).

For one dataset-model cell (first split, given seed), trains the parent ERM
run and then each recovery preparation (last, head, full) once, through the
exact training path of the experiment (same fixed-input verification, group
construction, seeding order, and train_recovery code as scripts/train_run.py).

Records ONLY: that each run ran, its runtime, and, for the recovery
preparations alone, whether the run converged within its step budget; ERM
parents have no convergence criterion. No evaluation is performed and no
validation or test score is stored or printed: recovery checkpoint selection uses tail macro-F1
internally, and those numbers must not be seen before the pre-analysis
registration is filed. These runs are excluded from the scientific record.

Usage:
    python scripts/debug_recovery.py --model gcn [--dataset roman-empire]
"""

from __future__ import annotations

import argparse
import copy
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

from hfd.datasets import load_dataset  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.stats import _sha256  # noqa: E402
from hfd.training import (  # noqa: E402
    RecoveryConfig,
    TrainConfig,
    _reset_module,
    set_seed,
    train_erm,
    train_recovery,
)

# A recovery preparation "converged within budget" when its selected
# checkpoint does not sit in the final tenth of the budget
# (still-improving-at-the-edge is what the exploratory runs showed for an
# underbudgeted preparation). Recovery preparations only: ERM runs have no
# convergence criterion. Uses only the checkpoint's epoch index, never its
# score.
CONVERGENCE_TAIL_FRACTION = 0.9


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def verify_fixed_inputs(name: str, arrays, manifest: dict) -> None:
    recorded = manifest["array_checksums_sha256"]
    for key in ("labels", "merged_bin", "tail_mask", "eligible"):
        if recorded[key] != _sha256(arrays[key]):
            raise RuntimeError(
                f"{name}: checksum mismatch for frozen array {key!r}; "
                "regenerate manifests before training."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="roman-empire")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / f"{args.model}.yaml").read_text(encoding="utf-8")
    )
    train_cfg = TrainConfig(**model_cfg["train"])
    ds_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "datasets.yaml").read_text(encoding="utf-8")
    )
    device = torch.device(args.device)
    out_dir = REPO_ROOT / "results" / "debug_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- fixed inputs (verified against the frozen manifest) -----------------
    bundle = load_dataset(args.dataset, REPO_ROOT / ds_config["data_root"])
    manifest = json.loads(
        (REPO_ROOT / ds_config["manifest_dir"] / f"{args.dataset}.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = np.load(REPO_ROOT / ds_config["array_dir"] / f"{args.dataset}.npz")
    verify_fixed_inputs(args.dataset, arrays, manifest)
    split = bundle.splits[0]

    simple = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
    x = bundle.x.to(device)
    edge_index = torch.from_numpy(simple).to(device)
    y = torch.from_numpy(bundle.y).to(device)
    train_idx = torch.from_numpy(split.train_idx).to(device)
    val_idx = torch.from_numpy(split.val_idx).to(device)

    # GroupDRO groups: training mask intersected with the fixed bin masks
    # (same construction as scripts/train_run.py).
    merged_bin = arrays["merged_bin"]
    num_groups = len(manifest["local_homophily"]["merged_bins"])
    grp = merged_bin[split.train_idx]
    in_group = grp >= 0
    train_dro_idx = torch.from_numpy(split.train_idx[in_group]).to(device)
    train_group_ids = torch.from_numpy(grp[in_group]).to(device)
    tail_mask = arrays["tail_mask"]
    val_tail_idx = torch.from_numpy(split.val_idx[tail_mask[split.val_idx]]).to(device)

    def fresh_model():
        set_seed(args.seed)
        return build_model(
            args.model, int(x.shape[1]), bundle.num_classes, model_cfg,
            num_nodes=bundle.num_nodes,
        ).to(device)

    rows = []

    def record(preparation: str, result, seconds: float, budget: int) -> None:
        row = {
            "dataset": args.dataset,
            "model": args.model,
            "preparation": preparation,
            "split": split.split_id,
            "seed": args.seed,
            "ran": True,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "budget": budget,
        }
        if preparation != "erm-parent":
            row["converged_within_budget"] = (
                result.best_epoch < CONVERGENCE_TAIL_FRACTION * budget
            )
        row["train_seconds"] = round(seconds, 1)
        row["code_version"] = git_commit()
        row["note"] = "no val/test score recorded (pre-registration)"
        rows.append(row)
        (out_dir / f"{args.dataset}_{args.model}_{preparation}_seed-{args.seed}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, indent=2), flush=True)

    # --- parent ERM (needed for last/head; itself validated in Step 1) -------
    print(f"=== {args.dataset} / {args.model} / erm parent ===", flush=True)
    model = fresh_model()
    t0 = time.time()
    erm_result = train_erm(model, x, edge_index, y, train_idx, val_idx, train_cfg)
    record("erm-parent", erm_result, time.time() - t0, train_cfg.max_epochs)
    parent_state = copy.deepcopy(erm_result.best_state)

    # --- recovery preparations ------------------------------------------------
    for preparation in ("last", "head", "full"):
        print(f"=== {args.dataset} / {args.model} / {preparation} ===", flush=True)
        model = fresh_model()
        if preparation in ("last", "head"):
            model.load_state_dict(parent_state)
            model.to(device)
            _reset_module(model.head.out if preparation == "last" else model.head)
        cfg = RecoveryConfig.from_erm(train_cfg, preparation)
        t0 = time.time()
        result = train_recovery(
            model, preparation, x, edge_index, y,
            train_dro_idx, train_group_ids, val_tail_idx, num_groups, cfg,
        )
        record(preparation, result, time.time() - t0, cfg.max_epochs)

    print("\nsummary:")
    print(f"{'preparation':<12}{'best_ep':>8}{'budget':>8}{'converged':>10}{'seconds':>9}")
    for r in rows:
        print(
            f"{r['preparation']:<12}{r['best_epoch']:>8}{r['budget']:>8}"
            f"{str(r.get('converged_within_budget', '-')):>10}{r['train_seconds']:>9.1f}"
        )


if __name__ == "__main__":
    main()
