"""Check additional datasets without fitting any model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hfd.dataset_readiness import inspect_dataset_readiness  # noqa: E402
from hfd.datasets import load_dataset  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "datasets-additional.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output; the check otherwise only prints",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    registered = config.get("datasets")
    if not isinstance(registered, list) or not registered:
        raise ValueError(f"{args.config}: datasets must be a non-empty list")
    selected = args.dataset or registered
    outside = sorted(set(selected) - set(registered))
    if outside:
        raise ValueError(
            "selected datasets are absent from the registry: "
            + ", ".join(outside)
        )
    data_root = REPO_ROOT / config["data_root"]

    reports = []
    for dataset in selected:
        print(f"=== {dataset} ===", flush=True)
        report = inspect_dataset_readiness(load_dataset(dataset, data_root))
        reports.append(report)
        graph = report["graph"]
        homophily = report["local_homophily"]
        test_tail = [split["test"]["n_tail"] for split in report["splits"]]
        supported = [
            split["test"]["n_tail_classes_ge_10"] for split in report["splits"]
        ]
        print(
            f"ready={report['ready']} nodes={graph['num_nodes']:,} "
            f"edges={graph['num_undirected_simple_edges']:,} "
            f"classes={graph['num_classes']} mean_h={homophily['mean_h']:.4f} "
            f"tail={homophily['tail_count']:,} "
            f"({homophily['tail_fraction']:.1%})"
        )
        print(
            "test-tail nodes min/median/max="
            f"{min(test_tail)}/{int(round(float(np.median(test_tail))))}/"
            f"{max(test_tail)}; classes with >=10 tail nodes min/max="
            f"{min(supported)}/{max(supported)}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")

    document = {"schema_version": 1, "datasets": reports}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")
    if not all(report["ready"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
