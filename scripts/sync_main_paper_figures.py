"""Copy regenerated main-dataset PDFs to their declared publication paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ARRAY_PLOTTERS = {
    "scripts/plot_rq1_composite.py",
    "scripts/plot_rq5_tail_vs_non_tail.py",
}


def figure_mappings(*, derived_only: bool = False) -> dict[Path, Path]:
    inventory = json.loads(
        (REPO_ROOT / "paper" / "artifacts.json").read_text(encoding="utf-8")
    )
    mappings = {}
    for figure in inventory["figures"]:
        source = figure.get("generated_path") or ""
        if not source.startswith("results/main/figures/") or not source.endswith(".pdf"):
            continue
        if derived_only and figure["generator"] in RAW_ARRAY_PLOTTERS:
            continue
        mappings[Path(source)] = Path(figure["artifact_path"])
    return mappings


def sync_figures(*, derived_only: bool = False) -> int:
    mappings = figure_mappings(derived_only=derived_only)
    missing = [str(source) for source in mappings if not (REPO_ROOT / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing generated figures: " + ", ".join(missing))
    for source, destination in mappings.items():
        target = REPO_ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / source, target)
    return len(mappings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-only", action="store_true",
        help="copy only figures rebuilt without node-level arrays",
    )
    args = parser.parse_args()
    count = sync_figures(derived_only=args.derived_only)
    print(f"Updated {count} main-dataset publication figures")


if __name__ == "__main__":
    main()
