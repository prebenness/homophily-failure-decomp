"""Run the fixed three-cell PubMed FG-CCDB recovery panel.

The panel is:

- models: GraphSAGE-mean, GCN, ACM-GCN;
- preparations: A_last, A_head, A_full;
- pairs: ten Geom-GCN splits by three training seeds.

This is 270 atomic runs. Existing canonical ERM checkpoints are read from a
separate parent root. FG outputs must use a separate output root.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
# Finish the shortest complete model cell first so a usable partial result is
# available while the two longer cells continue.
MODELS = ["acm-gcn", "gcn", "graphsage-mean"]
PREPARATIONS = ["last", "head", "full"]
SPLITS = [f"geomgcn-{index}" for index in range(10)]
SEEDS = [1, 2, 3]
REQUIRED_OUTPUTS = {
    "config.json",
    "metrics.json",
    "history.json",
    "checkpoint.pt",
    "run_data.npz",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_id(model: str, preparation: str, split: str, seed: int) -> str:
    return f"pubmed_{model}_fg-ccdb-{preparation}_{split}_seed-{seed}"


def validate_complete_run(
    run_dir: Path,
    *,
    model: str,
    preparation: str,
    split: str,
    seed: int,
    code_version: str,
) -> tuple[bool, str | None]:
    missing = sorted(
        filename for filename in REQUIRED_OUTPUTS if not (run_dir / filename).is_file()
    )
    if missing:
        return False, f"missing files: {missing}"
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        with np.load(run_dir / "run_data.npz", allow_pickle=False) as saved:
            saved_keys = set(saved.files)
    except Exception as error:
        return False, f"unreadable output: {error}"

    expected = {
        "run_id": run_dir.name,
        "dataset": "pubmed",
        "model": model,
        "preparation": preparation,
        "split": split,
        "seed": seed,
        "recovery_objective": "fg-ccdb",
        "code_version": code_version,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        return False, f"config mismatch: {mismatches}"
    if metrics.get("training_objective", {}).get("name") != "fg-ccdb":
        return False, "metrics do not identify the FG-CCDB objective"
    if not history or any(row.get("objective") != "fg-ccdb" for row in history):
        return False, "history is empty or contains another objective"
    required_arrays = {
        "logits",
        "predictions",
        "labels",
        "train_idx",
        "val_idx",
        "test_idx",
        "merged_bin",
        "tail_mask",
        "objective_train_idx",
        "objective_train_weights",
    }
    if not required_arrays.issubset(saved_keys):
        return False, f"run_data.npz lacks {sorted(required_arrays - saved_keys)}"
    if (run_dir / "checkpoint.pt").stat().st_size == 0:
        return False, "checkpoint.pt is empty"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--preparations",
        nargs="+",
        choices=PREPARATIONS,
        default=PREPARATIONS,
    )
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    array_root = args.array_root.resolve()
    parent_root = args.parent_run_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == parent_root or parent_root in output_root.parents:
        parser.error("FG output root must be separate from the canonical parent root")
    for name, path in (
        ("data root", data_root),
        ("array root", array_root),
        ("parent run root", parent_root),
    ):
        if not path.is_dir():
            parser.error(f"{name} does not exist: {path}")

    code_version = git_output("rev-parse", "HEAD")
    tracked_changes = git_output("status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        parser.error(
            "The worktree has tracked changes. Commit the implementation before "
            "running so every raw result has an exact code version."
        )

    panel = [
        (model, preparation, split, seed)
        for model in args.models
        for preparation in args.preparations
        for split in args.splits
        for seed in args.seeds
    ]
    if args.dry_run:
        for model, preparation, split, seed in panel:
            print(run_id(model, preparation, split, seed))
        print(f"{len(panel)} runs")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "panel_status.json"
    status = {
        "schema_version": 1,
        "experiment": "fg-ccdb-pubmed-recovery",
        "code_version": code_version,
        "started_utc": now_utc(),
        "updated_utc": now_utc(),
        "completed_utc": None,
        "scope": {
            "dataset": "pubmed",
            "models": list(args.models),
            "preparations": list(args.preparations),
            "splits": list(args.splits),
            "seeds": list(args.seeds),
            "expected_runs": len(panel),
        },
        "paths": {
            "data_root": str(data_root),
            "array_root": str(array_root),
            "parent_run_root": str(parent_root),
            "output_root": str(output_root),
        },
        "completed_runs": [],
        "failed_run": None,
    }

    # Resume only results made by this exact committed implementation.
    for model, preparation, split, seed in panel:
        identifier = run_id(model, preparation, split, seed)
        complete, _ = validate_complete_run(
            output_root / identifier,
            model=model,
            preparation=preparation,
            split=split,
            seed=seed,
            code_version=code_version,
        )
        if complete:
            status["completed_runs"].append(identifier)
    atomic_json(status_path, status)

    for index, (model, preparation, split, seed) in enumerate(panel, start=1):
        identifier = run_id(model, preparation, split, seed)
        run_dir = output_root / identifier
        complete, reason = validate_complete_run(
            run_dir,
            model=model,
            preparation=preparation,
            split=split,
            seed=seed,
            code_version=code_version,
        )
        if complete:
            print(f"[{index}/{len(panel)}] complete: {identifier}", flush=True)
            continue
        if run_dir.exists():
            print(
                f"[{index}/{len(panel)}] replacing incomplete output "
                f"({reason}): {identifier}",
                flush=True,
            )
        else:
            print(f"[{index}/{len(panel)}] running: {identifier}", flush=True)

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_run.py"),
            "--dataset",
            "pubmed",
            "--model",
            model,
            "--preparation",
            preparation,
            "--split",
            split,
            "--seed",
            str(seed),
            "--recovery-objective",
            "fg-ccdb",
            "--device",
            args.device,
            "--data-root",
            str(data_root),
            "--array-root",
            str(array_root),
            "--parent-run-root",
            str(parent_root),
            "--output-root",
            str(output_root),
        ]
        if run_dir.exists():
            command.append("--replace-existing")
        process = subprocess.run(command, cwd=REPO_ROOT)
        complete, reason = validate_complete_run(
            run_dir,
            model=model,
            preparation=preparation,
            split=split,
            seed=seed,
            code_version=code_version,
        )
        if process.returncode != 0 or not complete:
            status["failed_run"] = {
                "run_id": identifier,
                "return_code": process.returncode,
                "validation_error": reason,
                "failed_utc": now_utc(),
            }
            status["updated_utc"] = now_utc()
            atomic_json(status_path, status)
            raise SystemExit(
                f"Atomic run failed or produced invalid output: {identifier}: {reason}"
            )
        if identifier not in status["completed_runs"]:
            status["completed_runs"].append(identifier)
        status["updated_utc"] = now_utc()
        atomic_json(status_path, status)

    status["completed_utc"] = now_utc()
    status["updated_utc"] = status["completed_utc"]
    atomic_json(status_path, status)
    print(f"Completed and validated {len(panel)} FG-CCDB runs.", flush=True)


if __name__ == "__main__":
    main()
