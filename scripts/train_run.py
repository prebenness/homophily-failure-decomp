"""Execute one atomic run: (dataset, model, preparation, split, seed).

Preparations: erm (average-loss parent), and recovery preparations last /
head / full under either GroupDRO or FG-CCDB importance weighting. last and
head require the matching erm run's checkpoint (same dataset, model, split,
seed).

Writes the contract's run outputs to the selected raw output root:
- config.json   resolved configuration, code version, library versions
- metrics.json  gap-audit metric block + checkpoint metadata
- history.json  per-epoch training curve
- checkpoint.pt best model state dict
- run_data.npz  logits, predictions, masks, labels, group assignments

Usage:
    python scripts/train_run.py --dataset roman-empire --model gcn \
        --preparation erm --split platonov-0 --seed 1

    python scripts/train_run.py --dataset pubmed --model gcn \
        --preparation head --split geomgcn-0 --seed 1 \
        --recovery-objective fg-ccdb --parent-run-root <canonical-runs> \
        --output-root <separate-fg-runs>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from hfd.datasets import load_dataset  # noqa: E402
from hfd.fg_ccdb import compute_fg_ccdb_weights  # noqa: E402
from hfd.homophily import to_simple_undirected  # noqa: E402
from hfd.metrics import gap_audit_metrics  # noqa: E402
from hfd.models import build_model  # noqa: E402
from hfd.stats import _sha256  # noqa: E402
from hfd.training import (  # noqa: E402
    RecoveryConfig,
    TrainConfig,
    _reset_module,
    evaluate_split,
    set_seed,
    train_erm,
    train_recovery,
)


def git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except OSError:
        return "unknown", None
    except subprocess.CalledProcessError:
        return "unknown", None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def verify_fixed_inputs(name: str, arrays: np.lib.npyio.NpzFile, manifest: dict) -> None:
    """Check loaded fixed-input arrays against the manifest checksums."""
    recorded = manifest["array_checksums_sha256"]
    if set(arrays.files) != set(recorded):
        missing = sorted(set(recorded) - set(arrays.files))
        unexpected = sorted(set(arrays.files) - set(recorded))
        raise RuntimeError(
            f"{name}: frozen-array keys differ from the registered manifest; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(recorded):
        actual = _sha256(arrays[key])
        if recorded[key] != actual:
            raise RuntimeError(
                f"{name}: checksum mismatch for frozen array {key!r}; "
                "regenerate manifests before training."
            )


def resolve_input_path(path: Path | None, default: str) -> Path:
    selected = path if path is not None else Path(default)
    return selected.resolve() if selected.is_absolute() else (REPO_ROOT / selected).resolve()


def validate_parent_config(
    config: dict,
    *,
    dataset: str,
    model: str,
    split: str,
    seed: int,
    model_config: dict,
    fixed_input_checksums: dict,
) -> None:
    expected_fields = {
        "dataset": dataset,
        "model": model,
        "preparation": "erm",
        "split": split,
        "seed": seed,
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected_fields.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Parent ERM configuration mismatch: {mismatches}")
    if config.get("model_config") != model_config:
        raise RuntimeError("Parent ERM model configuration differs from this run")
    if config.get("fixed_input_checksums") != fixed_input_checksums:
        raise RuntimeError("Parent ERM fixed-input checksums differ from this run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--preparation", default="erm", choices=["erm", "last", "head", "full"]
    )
    parser.add_argument("--split", required=True, help="split id, e.g. platonov-0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "datasets.yaml"))
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=REPO_ROOT / "manifests" / "main-runs.jsonl",
        help="run-panel manifest used to schedule this atomic run",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--recovery-objective",
        default="groupdro",
        choices=["groupdro", "fg-ccdb"],
        help="training objective for last/head/full; ERM always uses average loss",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="downloaded dataset root; defaults to the dataset configuration",
    )
    parser.add_argument(
        "--array-root",
        type=Path,
        help="frozen node-array root; defaults to the dataset configuration",
    )
    parser.add_argument(
        "--parent-run-root",
        type=Path,
        help="root containing canonical ERM parent run directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "runs",
        help="directory that will contain one completed directory per run",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="atomically replace an existing run directory after a successful rerun",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to start when the Git working tree has uncommitted changes",
    )
    args = parser.parse_args()
    if args.preparation == "erm" and args.recovery_objective != "groupdro":
        parser.error("--recovery-objective applies only to last/head/full")

    ds_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "models" / f"{args.model}.yaml").read_text(encoding="utf-8")
    )
    train_cfg = TrainConfig(**model_cfg["train"])

    preparation_tag = (
        f"fg-ccdb-{args.preparation}"
        if args.recovery_objective == "fg-ccdb"
        else args.preparation
    )
    run_id = f"{args.dataset}_{args.model}_{preparation_tag}_{args.split}_seed-{args.seed}"
    output_root = args.output_root.expanduser().resolve()
    parent_run_root = (
        args.parent_run_root.expanduser().resolve()
        if args.parent_run_root is not None
        else output_root
    )
    data_root = resolve_input_path(args.data_root, ds_config["data_root"])
    array_root = resolve_input_path(args.array_root, ds_config["array_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists() and not args.replace_existing:
        raise FileExistsError(
            f"Run output already exists: {final_dir}. The experiment-grid "
            "runner skips valid completed runs; pass --replace-existing only "
            "for a deliberate rerun."
        )
    code_version, git_dirty = git_state()
    if args.require_clean and git_dirty is not False:
        state = "unknown" if git_dirty is None else "dirty"
        raise RuntimeError(
            f"Refusing a registered run from a {state} Git working tree."
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    print(f"run: {run_id}")

    # --- fixed inputs -------------------------------------------------------
    bundle = load_dataset(args.dataset, data_root)
    manifest = json.loads(
        (REPO_ROOT / ds_config["manifest_dir"] / f"{args.dataset}.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = np.load(array_root / f"{args.dataset}.npz")
    verify_fixed_inputs(args.dataset, arrays, manifest)

    try:
        bundle_split = next(s for s in bundle.splits if s.split_id == args.split)
    except StopIteration as exc:
        registered = [row["split_id"] for row in manifest["splits"]]
        raise ValueError(
            f"split {args.split!r} is not registered for {args.dataset}; "
            f"expected one of {registered}"
        ) from exc
    if args.seed not in bundle.training_seeds:
        raise ValueError(f"seed {args.seed} not in panel {bundle.training_seeds}")
    split_arrays = {
        part: arrays[f"{args.split}.{part}"]
        for part in ("train", "val", "test")
    }
    for part in ("train", "val", "test"):
        live = getattr(bundle_split, f"{part}_idx")
        if not np.array_equal(live, split_arrays[part]):
            raise RuntimeError(
                f"{args.dataset}/{args.split}: live {part} split differs from "
                "the frozen registered array"
            )
    if not np.array_equal(bundle.y, arrays["labels"]):
        raise RuntimeError(
            f"{args.dataset}: live labels differ from the frozen registered array"
        )

    # --- tensors ------------------------------------------------------------
    device = torch.device(args.device)
    simple_edges = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
    graph_checksums = manifest.get("graph_checksums_sha256", {})
    if "features" in graph_checksums:
        feature_array = bundle.x.detach().cpu().numpy()
        if _sha256(feature_array) != graph_checksums["features"]:
            raise RuntimeError(
                f"{args.dataset}: live feature matrix differs from registration"
            )
    if "simple_edge_index" in graph_checksums:
        if _sha256(simple_edges) != graph_checksums["simple_edge_index"]:
            raise RuntimeError(
                f"{args.dataset}: live simple edge index differs from registration"
            )
    x = bundle.x.to(device)
    edge_index = torch.from_numpy(simple_edges).to(device)
    y = torch.from_numpy(bundle.y).to(device)
    train_idx = torch.from_numpy(split_arrays["train"]).to(device)
    val_idx = torch.from_numpy(split_arrays["val"]).to(device)
    test_idx = torch.from_numpy(split_arrays["test"]).to(device)

    # --- train --------------------------------------------------------------
    set_seed(args.seed)
    model = build_model(
        args.model,
        int(x.shape[1]),
        bundle.num_classes,
        model_cfg,
        num_nodes=bundle.num_nodes,
    ).to(device)

    parent_run_id = None
    parent_provenance = None
    parent_checkpoint_sha256 = None
    recovery_cfg = None
    objective_diagnostics = None
    objective_train_idx = None
    objective_train_weights = None
    train_group_ids = None
    train_weights = None
    if args.preparation == "erm":
        result = train_erm(model, x, edge_index, y, train_idx, val_idx, train_cfg)
    else:
        # Both recovery objectives use training nodes with a defined fixed bin.
        merged_bin = arrays["merged_bin"]
        num_groups = len(manifest["local_homophily"]["merged_bins"])
        grp = merged_bin[split_arrays["train"]]
        in_group = grp >= 0
        objective_train_idx = split_arrays["train"][in_group]
        train_objective_idx = torch.from_numpy(objective_train_idx).to(device)
        if args.recovery_objective == "groupdro":
            train_group_ids = torch.from_numpy(grp[in_group]).to(device)
        else:
            fg_weights = compute_fg_ccdb_weights(
                bundle.y[objective_train_idx],
                grp[in_group],
                num_classes=bundle.num_classes,
                num_bins=num_groups,
            )
            objective_train_weights = fg_weights.weights
            train_weights = torch.from_numpy(objective_train_weights).to(
                device=device, dtype=x.dtype
            )
            objective_diagnostics = {
                **fg_weights.diagnostics,
                "implementation": (
                    "deterministic full-batch importance-weighted cross-entropy"
                ),
                "sampler": None,
                "n_split_training_nodes": int(split_arrays["train"].size),
                "n_eligible_training_nodes": int(objective_train_idx.size),
                "n_excluded_undefined_bin": int(np.count_nonzero(~in_group)),
                "fit_scope": (
                    "bin-by-class counts use eligible training nodes only; "
                    "validation and test nodes do not enter the weights"
                ),
                "oracle_status": (
                    "the fixed local-homophily bins were constructed from true "
                    "node and neighbour labels over the full graph"
                ),
            }
        tail_mask = arrays["tail_mask"]
        val_tail_idx = torch.from_numpy(
            split_arrays["val"][tail_mask[split_arrays["val"]]]
        ).to(device)

        if args.preparation in ("last", "head"):
            parent_run_id = f"{args.dataset}_{args.model}_erm_{args.split}_seed-{args.seed}"
            parent_dir = parent_run_root / parent_run_id
            parent_ckpt = parent_dir / "checkpoint.pt"
            parent_config_path = parent_dir / "config.json"
            if not parent_ckpt.is_file() or not parent_config_path.is_file():
                raise FileNotFoundError(
                    f"Parent ERM checkpoint or config missing below {parent_dir}"
                )
            parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
            validate_parent_config(
                parent_config,
                dataset=args.dataset,
                model=args.model,
                split=args.split,
                seed=args.seed,
                model_config=model_cfg,
                fixed_input_checksums=manifest["array_checksums_sha256"],
            )
            parent_checkpoint_sha256 = file_sha256(parent_ckpt)
            parent_provenance = {
                "run_id": parent_run_id,
                "training_code_version": parent_config["code_version"],
                "checkpoint_sha256": parent_checkpoint_sha256,
            }
            model.load_state_dict(
                torch.load(parent_ckpt, map_location="cpu", weights_only=True)
            )
            model.to(device)
            # Randomly re-initialise the retrained part (seeded via set_seed).
            _reset_module(model.head.out if args.preparation == "last" else model.head)

        recovery_cfg = RecoveryConfig.from_erm(train_cfg, args.preparation)
        result = train_recovery(
            model,
            args.preparation,
            x,
            edge_index,
            y,
            train_objective_idx,
            train_group_ids,
            val_tail_idx,
            num_groups,
            recovery_cfg,
            objective=args.recovery_objective,
            train_weights=(
                train_weights if args.recovery_objective == "fg-ccdb" else None
            ),
        )

    # --- evaluate best checkpoint -------------------------------------------
    model.load_state_dict(result.best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index).cpu().numpy()
    y_pred = logits.argmax(axis=-1)
    test_loss, test_acc = evaluate_split(model, x, edge_index, y, test_idx)

    bin_intervals = [b["interval"] for b in manifest["local_homophily"]["merged_bins"]]
    metrics = gap_audit_metrics(
        y_true=bundle.y,
        y_pred=y_pred,
        test_idx=split_arrays["test"],
        merged_bin=arrays["merged_bin"],
        tail_mask=arrays["tail_mask"],
        bin_intervals=bin_intervals,
        standard_metric=bundle.metric,
    )
    metrics["checkpoint"] = {
        "best_epoch": result.best_epoch,
        "best_val_metric": result.best_val_metric,
        "checkpoint_selection": (
            "standard validation metric"
            if args.preparation == "erm"
            else "validation macro-F1 on fixed minority-pattern tail"
        ),
        "epochs_run": result.epochs_run,
        "train_seconds": result.seconds,
        "test_loss": test_loss,
        "parent_run_id": parent_run_id,
    }
    metrics["training_objective"] = {
        "name": (
            "erm" if args.preparation == "erm" else args.recovery_objective
        ),
        "fg_ccdb_diagnostics": objective_diagnostics,
    }

    # --- write outputs atomically --------------------------------------------
    experiment_manifest = args.experiment_manifest.expanduser().resolve()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    accelerator = {
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }
    config = {
        "run_id": run_id,
        "dataset": args.dataset,
        "model": args.model,
        "preparation": args.preparation,
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "model_config": model_cfg,
        "train_config": asdict(train_cfg),
        "recovery_config": asdict(recovery_cfg) if recovery_cfg else None,
        "recovery_objective": (
            None if args.preparation == "erm" else args.recovery_objective
        ),
        "objective_diagnostics": objective_diagnostics,
        "parent_run_id": parent_run_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_provenance": parent_provenance,
        "code_version": code_version,
        "git_dirty": git_dirty,
        "command": command,
        "experiment_manifest": (
            str(experiment_manifest.relative_to(REPO_ROOT))
            if experiment_manifest.is_file()
            and experiment_manifest.is_relative_to(REPO_ROOT)
            else str(experiment_manifest) if experiment_manifest.is_file() else None
        ),
        "experiment_manifest_sha256": (
            file_sha256(experiment_manifest)
            if experiment_manifest.is_file()
            else None
        ),
        "versions": {
            "torch": torch.__version__,
            "torch_geometric": package_version("torch-geometric"),
            "ogb": package_version("ogb"),
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "scikit_learn": package_version("scikit-learn"),
            "pyyaml": package_version("PyYAML"),
            "python": platform.python_version(),
        },
        "accelerator": accelerator,
        "graph_convention": "simple undirected (symmetrised, no self-loops, deduped)",
        "fixed_input_checksums": manifest["array_checksums_sha256"],
        "input_roots": {
            "data": str(data_root),
            "fixed_arrays": str(array_root),
            "parent_runs": str(parent_run_root),
        },
        "graph_checksums": manifest.get("graph_checksums_sha256"),
    }
    run_arrays = {
        "logits": logits.astype(np.float32),
        "predictions": y_pred.astype(np.int64),
        "labels": bundle.y,
        "train_idx": split_arrays["train"],
        "val_idx": split_arrays["val"],
        "test_idx": split_arrays["test"],
        "merged_bin": arrays["merged_bin"],
        "tail_mask": arrays["tail_mask"],
    }
    if objective_train_idx is not None:
        run_arrays["objective_train_idx"] = objective_train_idx
    if objective_train_weights is not None:
        run_arrays["objective_train_weights"] = objective_train_weights.astype(
            np.float64
        )
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.in-progress-", dir=output_root)
    )
    try:
        (temporary_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        (temporary_dir / "history.json").write_text(
            json.dumps(result.history) + "\n", encoding="utf-8"
        )
        torch.save(result.best_state, temporary_dir / "checkpoint.pt")
        np.savez_compressed(
            temporary_dir / "run_data.npz",
            **run_arrays,
        )

        backup_dir = None
        if final_dir.exists():
            backup_dir = output_root / f".{run_id}.superseded-{os.getpid()}"
            if backup_dir.exists():
                raise FileExistsError(f"Backup path already exists: {backup_dir}")
            final_dir.rename(backup_dir)
        try:
            temporary_dir.rename(final_dir)
        except Exception:
            if backup_dir is not None and not final_dir.exists():
                backup_dir.rename(final_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    compact = {
        k: v
        for k, v in metrics.items()
        if k not in ("per_bin", "within_class_tail_contrast")
    }
    print(json.dumps(compact, indent=2))
    print("per-bin macro-F1:")
    for row in metrics["per_bin"]:
        print(f"  {row['bin']:>10}  n={row['n_test']:>6}  macro_f1={row['macro_f1']:.4f}")
    if metrics["within_class_tail_contrast"]:
        print("within-class tail contrast (correct fraction):")
        for row in metrics["within_class_tail_contrast"]:
            print(
                f"  class {row['class']:>3}  tail {row['tail_correct_fraction']:.3f} "
                f"(n={row['n_tail_test']})  non-tail "
                f"{row['non_tail_correct_fraction']:.3f} (n={row['n_non_tail_test']})"
            )
    print(f"outputs: {final_dir}")


if __name__ == "__main__":
    main()
