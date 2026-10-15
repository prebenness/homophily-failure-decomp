"""Authenticate the frozen inputs for all four reverse-rewiring preparations.

Paths are supplied at runtime; the written inventory contains only relative
run identities, input hashes, and portable experiment metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_edge_dose_response_expansion as base

DESIGN_ID = "appendix-j-reverse-edge-inputs-v1"
RUN_FILES = ("checkpoint.pt", "config.json", "run_data.npz", "metrics.json")
OPTIONAL_RUN_FILES = ("history.json",)


def expected_runs():
    for dataset, spec in base.DATASET_SPECS.items():
        for preparation in base.PREPARATIONS:
            for model in base.FULL_MODELS:
                for split, seed in spec.checkpoint_identities:
                    yield {
                        "dataset": dataset, "preparation": preparation,
                        "model": model, "split": split, "seed": seed,
                        "run_id": base.parent_run_id(spec, model, split, seed, preparation),
                    }


def record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": base.file_sha256(path)}


def validate_inventory(payload: dict) -> None:
    if payload.get("design_id") != DESIGN_ID or payload.get("authenticated") is not True:
        raise RuntimeError("Input authentication identity or success flag differs")
    expected = {item["run_id"]: item for item in expected_runs()}
    observed = {}
    for row in payload.get("runs", []):
        run_id = row.get("run_id")
        if run_id in observed or run_id not in expected:
            raise RuntimeError(f"Unexpected or duplicate input run: {run_id}")
        if any(row.get(key) != value for key, value in expected[run_id].items()):
            raise RuntimeError(f"Input identity differs: {run_id}")
        files = set(row.get("files", {}))
        if not set(RUN_FILES) <= files or files - set(RUN_FILES + OPTIONAL_RUN_FILES):
            raise RuntimeError(f"Incomplete or unexpected checkpoint run files: {run_id}")
        for bound in row["files"].values():
            digest = bound.get("sha256", "")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise RuntimeError(f"Invalid input hash: {run_id}")
            if not isinstance(bound.get("bytes"), int) or bound["bytes"] <= 0:
                raise RuntimeError(f"Invalid input size: {run_id}")
        observed[run_id] = row
    if set(observed) != set(expected) or payload.get("run_count") != 3600:
        raise RuntimeError("Input inventory must contain exactly 3,600 registered runs")
    if set(payload.get("datasets", {})) != set(base.DATASET_SPECS):
        raise RuntimeError("Input inventory must contain exactly five datasets")


def authenticate(args) -> dict:
    payload = {
        "schema_version": 1, "design_id": DESIGN_ID, "authenticated": True,
        "preparations": list(base.PREPARATIONS), "models": list(base.FULL_MODELS),
        "run_count": 3600, "run_record_files": list(RUN_FILES),
        "optional_run_record_files": list(OPTIONAL_RUN_FILES),
        "datasets": {}, "runs": [],
    }
    runs = list(expected_runs())
    for dataset, spec in base.DATASET_SPECS.items():
        bundle, arrays, manifest = base.load_registered_inputs(
            spec, data_root=args.data_root, array_root=args.array_root,
            manifest_root=args.manifest_root,
        )
        try:
            graph_actual = args.graph_root / dataset / "actual"
            validation = json.loads((graph_actual / "validation.json").read_text())
            catalog_hash = base.file_sha256(graph_actual / "catalog.json")
            if (validation.get("validated") is not True
                    or validation.get("catalog", {}).get("sha256") != catalog_hash):
                raise RuntimeError(f"{dataset}: graph validation does not bind the current catalog")
            payload["datasets"][dataset] = {
                "manifest": record(args.manifest_root / f"{dataset}.json"),
                "frozen_arrays": record(args.array_root / f"{dataset}.npz"),
                "array_checksums_sha256": manifest["array_checksums_sha256"],
                "graph_checksums_sha256": manifest["graph_checksums_sha256"],
                "reverse_graph_catalog": record(graph_actual / "catalog.json"),
                "reverse_graph_validation": record(graph_actual / "validation.json"),
            }
            for item in runs:
                if item["dataset"] != dataset:
                    continue
                checkpoint = base.load_checkpoint_spec(
                    spec, model=item["model"], split=item["split"], seed=item["seed"],
                    checkpoint_root=args.checkpoint_root,
                    fixed_input_checksums=manifest["array_checksums_sha256"],
                    num_nodes=bundle.num_nodes, preparation=item["preparation"],
                )
                run_dir = args.checkpoint_root / item["run_id"]
                names = (*RUN_FILES, *(name for name in OPTIONAL_RUN_FILES
                                      if (run_dir / name).is_file()))
                files = {name: record(run_dir / name) for name in names}
                if files["checkpoint.pt"]["sha256"] != checkpoint.checkpoint_sha256:
                    raise RuntimeError(f"Input checkpoint changed during authentication: {item['run_id']}")
                payload["runs"].append({
                    **item, "files": files,
                    "training_code_version": checkpoint.training_code_version,
                })
                if len(payload["runs"]) % 30 == 0:
                    print(f"AUTHENTICATED {item['preparation']}/{dataset}/{item['model']} "
                          f"total={len(payload['runs'])}/3600", flush=True)
        finally:
            arrays.close()
    validate_inventory(payload)
    return payload


def verify_outputs(snapshot: dict, output_root: Path) -> dict:
    validate_inventory(snapshot)
    expected = {row["run_id"]: row for row in snapshot["runs"]}
    observed = set()
    pairs = 0
    for path in sorted((output_root / "actual").glob("*/*/*/shards/*.json")):
        value = json.loads(path.read_text())
        run_id = value.get("run_id")
        if run_id not in expected or run_id in observed:
            raise RuntimeError(f"Unexpected or duplicate output run: {run_id}")
        source = expected[run_id]
        for key in ("preparation", "dataset", "model", "run_id"):
            if value.get(key) != source[key]:
                raise RuntimeError(f"Output identity differs for {run_id}: {key}")
        if value.get("checkpoint_sha256") != source["files"]["checkpoint.pt"]["sha256"]:
            raise RuntimeError(f"Output used a different checkpoint: {run_id}")
        if value.get("completed") is not True or value.get("phase") != "actual":
            raise RuntimeError(f"Incomplete or non-actual output: {run_id}")
        for row in value.get("pairs", []):
            if row.get("checkpoint_sha256") != source["files"]["checkpoint.pt"]["sha256"]:
                raise RuntimeError(f"Pair checkpoint binding differs: {run_id}")
        pairs += len(value.get("pairs", []))
        observed.add(run_id)
    if observed != set(expected) or pairs != 176400:
        raise RuntimeError(f"Incomplete outputs: {len(observed)}/3600 runs and {pairs}/176400 pairs")
    return {"verified": True, "checkpoint_count": len(observed), "paired_row_count": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authenticate", "verify", "verify-outputs"))
    for name in ("checkpoint", "data", "array", "manifest", "graph"):
        parser.add_argument(f"--{name}-root", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.mode == "verify-outputs":
        if args.output_root is None:
            parser.error("verify-outputs requires --output-root")
        print(json.dumps(verify_outputs(json.loads(args.snapshot.read_text()), args.output_root)))
        return
    if any(getattr(args, name + "_root") is None for name in ("checkpoint", "data", "array", "manifest", "graph")):
        parser.error("authenticate and verify require all five input roots")
    payload = authenticate(args)
    if args.mode == "verify" or args.snapshot.exists():
        if json.loads(args.snapshot.read_text()) != payload:
            raise RuntimeError("Input snapshot differs; refusing to replace authenticated inputs")
    else:
        base.atomic_json(args.snapshot, payload)
    print(f"AUTHENTICATION_COMPLETE runs=3600 sha256={base.file_sha256(args.snapshot)}", flush=True)


if __name__ == "__main__":
    main()
