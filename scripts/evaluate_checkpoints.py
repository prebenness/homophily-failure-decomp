"""Regenerate predictions and evaluation metrics from supplied checkpoints.

Place the original config.json and checkpoint.pt files in the run directories
used by train_models.py. DFR runs instead contain config.json and
coefficients.npz, and use their matching ERM checkpoint from --main-root.
Run download_datasets.py first. Existing training metadata are preserved;
models are neither trained nor refitted by this command.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml

from hfd.datasets import load_dataset
from hfd.dfr import DFRFitResult
from hfd.fg_ccdb import compute_fg_ccdb_weights
from hfd.homophily import to_simple_undirected
from hfd.metrics import accuracy, gap_audit_metrics
from hfd.models import build_model
from hfd.stats import _sha256
from hfd.training import evaluate_split
from train_run import file_sha256, git_state, validate_parent_config, verify_fixed_inputs
from run_dfr_pubmed import extract_classifier_features, non_tail_metrics, read_parent_provenance


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def evaluation_revision() -> str:
    return git_state()[0]


def panel_runs(panel: str) -> list[dict]:
    """Use the existing registered panels and appendix run identifiers."""
    if panel in ("main", "additional"):
        filename = "main-runs.jsonl" if panel == "main" else "additional-dataset-runs.jsonl"
        return [json.loads(line) for line in (REPO_ROOT / "manifests" / filename).read_text().splitlines() if line.strip()]
    runs = []
    for model in ("acm-gcn", "gcn", "graphsage-mean"):
        for split_index in range(10):
            split = f"geomgcn-{split_index}"
            for seed in (1, 2, 3):
                for preparation in (("dfr",) if panel == "dfr" else ("last", "head", "full")):
                    tag = "dfr-val" if panel == "dfr" else f"fg-ccdb-{preparation}"
                    runs.append(dict(dataset="pubmed", model=model, preparation=preparation,
                                     split=split, seed=seed,
                                     run_id=f"pubmed_{model}_{tag}_{split}_seed-{seed}"))
    return runs


def check_config(config: dict, expected: dict, manifest: dict, model_config: dict, panel: str) -> None:
    for key in ("run_id", "dataset", "model", "preparation", "split", "seed"):
        if config.get(key) != expected[key]:
            raise ValueError(f"{expected['run_id']}: configuration differs for {key}")
    if config.get("fixed_input_checksums") != manifest["array_checksums_sha256"]:
        raise ValueError(f"{expected['run_id']}: registered inputs differ")
    if panel == "dfr":
        if config.get("dfr_mode") != "val" or config.get("feature_cut") != "classifier_features":
            raise ValueError(f"{expected['run_id']}: unexpected DFR mode or feature cut")
        if config.get("protocol_id") != "dfr_l1_joint_balanced_v1":
            raise ValueError(f"{expected['run_id']}: unexpected DFR protocol")
    else:
        if config.get("model_config") != model_config:
            raise ValueError(f"{expected['run_id']}: model configuration differs")
        if expected["preparation"] != "erm":
            objective = "fg-ccdb" if panel == "fg" else "groupdro"
            if config.get("recovery_objective", "groupdro") != objective:
                raise ValueError(f"{expected['run_id']}: unexpected recovery objective")


def load_inputs(dataset: str, panel: str, args: argparse.Namespace):
    additional = panel == "additional"
    manifest_dir = "additional-datasets" if additional else "datasets"
    manifest = read_json(REPO_ROOT / "manifests" / manifest_dir / f"{dataset}.json")
    array_root = args.additional_array_root if additional else args.array_root
    with np.load(array_root / f"{dataset}.npz", allow_pickle=False) as saved:
        verify_fixed_inputs(dataset, saved, manifest)
        arrays = {key: saved[key] for key in saved.files}
    bundle = load_dataset(dataset, args.data_root)
    if not np.array_equal(bundle.y, arrays["labels"]):
        raise ValueError(f"{dataset}: dataset labels differ from registered inputs")
    for split in bundle.splits:
        for part in ("train", "val", "test"):
            if not np.array_equal(getattr(split, f"{part}_idx"), arrays[f"{split.split_id}.{part}"]):
                raise ValueError(f"{dataset}/{split.split_id}: {part} split differs")
    edges = to_simple_undirected(bundle.edge_index, bundle.num_nodes)
    for key, value in (("features", bundle.x.detach().cpu().numpy()), ("simple_edge_index", edges)):
        recorded = manifest.get("graph_checksums_sha256", {}).get(key)
        if recorded is not None and _sha256(value) != recorded:
            raise ValueError(f"{dataset}: {key} differ from registered inputs")
    return bundle, arrays, manifest, bundle.x.to(args.device), torch.from_numpy(edges).to(args.device)


def check_parent(config: dict, parent_root: Path, manifest: dict, model_config: dict) -> None:
    identifier = f"{config['dataset']}_{config['model']}_erm_{config['split']}_seed-{config['seed']}"
    if config.get("parent_run_id") != identifier:
        raise ValueError(f"{config['run_id']}: mismatched ERM parent")
    parent_dir = parent_root / identifier
    validate_parent_config(read_json(parent_dir / "config.json"), dataset=config["dataset"],
                           model=config["model"], split=config["split"], seed=config["seed"],
                           model_config=model_config, fixed_input_checksums=manifest["array_checksums_sha256"])
    if config.get("parent_checkpoint_sha256") != file_sha256(parent_dir / "checkpoint.pt"):
        raise ValueError(f"{config['run_id']}: parent checkpoint differs")


def evaluate_run(run_dir: Path, config: dict, *, panel: str, inputs: tuple,
                 model_config: dict, parent_root: Path, device: torch.device) -> tuple[dict, dict]:
    bundle, fixed, manifest, x, edges = inputs
    split = config["split"]
    arrays = {f"{part}_idx": fixed[f"{split}.{part}"] for part in ("train", "val", "test")}
    arrays.update(labels=fixed["labels"], merged_bin=fixed["merged_bin"], tail_mask=fixed["tail_mask"])
    previous = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else {}
    if panel == "dfr":
        check_parent(config, parent_root, manifest, model_config)
        parent = read_parent_provenance(parent_root, model=config["model"], split=split,
                                       seed=config["seed"], model_config=model_config,
                                       fixed_input_checksums=manifest["array_checksums_sha256"])
        features, _ = extract_classifier_features(model_name=config["model"], model_config=model_config,
                                                  parent=parent, bundle=bundle, x=x, edge_index=edges, device=device)
        with np.load(run_dir / "coefficients.npz", allow_pickle=False) as saved:
            fitted = DFRFitResult(classes_=saved["classes"], coef_=saved["coef"],
                                  intercept_=saved["intercept"], scaler_mean_=saved["scaler_mean"],
                                  scaler_scale_=saved["scaler_scale"], diagnostics={})
        probabilities = fitted.predict_proba(features)
        predictions = fitted.predict(features).astype(np.int64)
        arrays.update(probabilities=probabilities.astype(np.float32), probability_classes=fitted.classes_,
                      eligible=fixed["merged_bin"] >= 0,
                      refit_idx=arrays["val_idx"][fixed["merged_bin"][arrays["val_idx"]] >= 0],
                      external_tuning_idx=np.empty(0, dtype=np.int64))
        objective = dict(previous.get("training_objective", {}))
        objective.update(name="dfr", mode="val", protocol=config["protocol_id"])
    else:
        if config["preparation"] in ("last", "head"):
            check_parent(config, parent_root, manifest, model_config)
        model = build_model(config["model"], int(x.shape[1]), bundle.num_classes,
                            model_config, num_nodes=bundle.num_nodes).to(device)
        model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = model(x, edges).cpu().numpy()
        predictions = logits.argmax(axis=-1).astype(np.int64)
        test_loss, _ = evaluate_split(model, x, edges, torch.from_numpy(bundle.y).to(device),
                                      torch.from_numpy(arrays["test_idx"]).to(device))
        arrays["logits"] = logits.astype(np.float32)
        previous["checkpoint"] = {**previous.get("checkpoint", {}), "test_loss": test_loss,
                                  "parent_run_id": config.get("parent_run_id")}
        objective = dict(previous.get("training_objective", {}))
        objective["name"] = "erm" if config["preparation"] == "erm" else config.get("recovery_objective", "groupdro")
        if config["preparation"] != "erm":
            train = arrays["train_idx"]
            eligible_train = train[fixed["merged_bin"][train] >= 0]
            arrays["objective_train_idx"] = eligible_train
            if panel == "fg":
                weights = compute_fg_ccdb_weights(bundle.y[eligible_train], fixed["merged_bin"][eligible_train],
                                                  num_classes=bundle.num_classes,
                                                  num_bins=len(manifest["local_homophily"]["merged_bins"]))
                arrays["objective_train_weights"] = weights.weights.astype(np.float64)
                objective["fg_ccdb_diagnostics"] = config.get("objective_diagnostics")
    if not np.isfinite(arrays.get("logits", arrays.get("probabilities"))).all():
        raise ValueError(f"{config['run_id']}: evaluation produced non-finite values")
    arrays["predictions"] = predictions
    metrics = gap_audit_metrics(y_true=bundle.y, y_pred=predictions, test_idx=arrays["test_idx"],
                                merged_bin=fixed["merged_bin"], tail_mask=fixed["tail_mask"],
                                bin_intervals=[row["interval"] for row in manifest["local_homophily"]["merged_bins"]],
                                standard_metric=bundle.metric)
    if panel == "dfr":
        non_tail = non_tail_metrics(bundle.y, predictions, arrays["test_idx"], fixed["merged_bin"], fixed["tail_mask"])
        tail = arrays["test_idx"][fixed["tail_mask"][arrays["test_idx"]]]
        metrics.update(whole_test_accuracy=metrics["whole_test_standard"],
                       tail_accuracy=accuracy(bundle.y[tail], predictions[tail]),
                       non_tail_macro_f1=non_tail["macro_f1"], non_tail_accuracy=non_tail["accuracy"], non_tail=non_tail)
    previous.pop("canonicalization", None)
    previous.update(metrics)
    previous["training_objective"] = objective
    previous["evaluation"] = {"code_version": evaluation_revision(), "device": str(device),
                              "source": "saved DFR coefficients" if panel == "dfr" else "saved model checkpoint"}
    return previous, arrays


def save_evaluation(run_dir: Path, metrics: dict, arrays: dict) -> None:
    """Finish both exports before replacing either existing evaluation file."""
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=run_dir) as temporary:
        temporary = Path(temporary)
        np.savez_compressed(temporary / "run_data.npz", **arrays)
        (temporary / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary / "run_data.npz", run_dir / "run_data.npz")
        os.replace(temporary / "metrics.json", run_dir / "metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=("main", "additional", "dfr", "fg"),
                        default=["main", "additional", "dfr", "fg"])
    for panel, folder in (("main", "runs"), ("additional", "additional-dataset-runs"),
                          ("dfr", "dfr-pubmed-runs"), ("fg", "fg-ccdb-pubmed-runs")):
        parser.add_argument(f"--{panel}-root", type=Path, default=REPO_ROOT / "artifacts" / folder)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "artifacts" / "data")
    parser.add_argument("--array-root", type=Path, default=REPO_ROOT / "artifacts" / "manifests")
    parser.add_argument("--additional-array-root", type=Path, default=REPO_ROOT / "artifacts" / "additional-dataset-manifests")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-id", action="append", help="evaluate only these registered runs (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="check supplied files and list runs without inference")
    args = parser.parse_args()
    args.device = torch.device(args.device)
    selected = set(args.run_id or [])
    jobs = []
    for panel in dict.fromkeys(args.panels):
        for run in panel_runs(panel):
            if not selected or run["run_id"] in selected:
                jobs.append((panel, run, getattr(args, f"{panel}_root") / run["run_id"]))
    unknown = selected - {run["run_id"] for _, run, _ in jobs}
    if unknown:
        parser.error(f"run IDs absent from selected panels: {sorted(unknown)}")
    missing = []
    for panel, run, directory in jobs:
        for filename in ("config.json", "coefficients.npz" if panel == "dfr" else "checkpoint.pt"):
            if not (directory / filename).is_file():
                missing.append(str(directory / filename))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required checkpoint/configuration files; first: {missing[:5]}")
    current = None
    inputs = None
    for number, (panel, run, directory) in enumerate(jobs, start=1):
        print(f"[{number}/{len(jobs)}] {run['run_id']}", flush=True)
        if args.dry_run:
            continue
        key = (panel, run["dataset"])
        if key != current:
            inputs = None
            if args.device.type == "cuda":
                torch.cuda.empty_cache()
            inputs = load_inputs(run["dataset"], panel, args)
            current = key
        config = read_json(directory / "config.json")
        model_config = yaml.safe_load((REPO_ROOT / "configs" / "models" / f"{run['model']}.yaml").read_text())
        check_config(config, run, inputs[2], model_config, panel)
        parent_root = args.additional_root if panel == "additional" else args.main_root
        metrics, arrays = evaluate_run(directory, config, panel=panel, inputs=inputs,
                                       model_config=model_config, parent_root=parent_root, device=args.device)
        save_evaluation(directory, metrics, arrays)


if __name__ == "__main__":
    main()
