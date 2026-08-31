"""Copy generated Actor/BlogCatalog outputs into the release inventory.

The analysis scripts write their canonical outputs below ``results/additional``.
This deterministic step gives the paper-facing copies stable names under
``paper/``. The paper repository uses these same files without manual edits.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURE_MAPPINGS = {
    Path("results/additional/figures/datasets/local_homophily_bin.tex"):
        Path("paper/figures/additional-datasets/local_homophily_bin.tex"),
    Path(
        "results/additional/figures/rq1/"
        "rq1_erm_additional-datasets_all-models_12-bin_"
        "macro_f1_by_local_homophily_with-ci.pdf"
    ): Path(
        "paper/figures/additional-datasets/"
        "rq1_macro_f1_by_local_homophily.pdf"
    ),
    Path(
        "results/additional/figures/rq1-rq2/unequal-y/"
        "rq1_rq2_additional_datasets_unadjusted_and_class_adjusted_"
        "macro_f1_tail_gaps_unequal_y.pdf"
    ): Path("paper/figures/additional-datasets/rq1_rq2_tail_gaps.pdf"),
    Path(
        "results/additional/figures/rq3-rq4/"
        "rq3_rq4_whole_test_and_tail_macro_f1_by_model_family.pdf"
    ): Path("paper/figures/additional-datasets/rq3_rq4_whole_tail_scores.pdf"),
    Path(
        "results/additional/figures/rq5/"
        "rq5_tail_vs_non_tail_macro_f1_compact.pdf"
    ): Path("paper/figures/additional-datasets/rq5_preparations_overview.pdf"),
    Path(
        "results/additional/figures/rq1/"
        "rq1_erm_additional-datasets_all-models_12-bin_"
        "accuracy_by_local_homophily_with-ci.pdf"
    ): Path(
        "paper/figures/additional-datasets/"
        "rq1_accuracy_by_local_homophily.pdf"
    ),
    Path(
        "results/additional/figures/rq3/"
        "rq3_aggregation_vs_mlp_tail_gap_contrast.pdf"
    ): Path("paper/figures/additional-datasets/rq3_tail_gap_contrast.pdf"),
    Path(
        "results/additional/figures/rq3/"
        "rq3_aggregation_vs_mlp_class_adjusted_tail_gap_contrast.pdf"
    ): Path(
        "paper/figures/additional-datasets/"
        "rq3_class_adjusted_tail_gap_contrast.pdf"
    ),
    Path(
        "results/additional/figures/rq4/"
        "rq4_class_adjusted_whole_test_and_tail_macro_f1_by_model_family.pdf"
    ): Path(
        "paper/figures/additional-datasets/"
        "rq4_class_adjusted_whole_tail_scores.pdf"
    ),
}

TABLE_FILENAMES = (
    "local_homophily_groups.tex",
    "rq1_macro_f1_tail_gap.tex",
    "rq1_accuracy_tail_gap.tex",
    "rq2_within_class_accuracy.tex",
    "rq2_adjusted_accuracy.tex",
    "rq2_adjusted_macro_f1.tex",
    "rq4_designed_vs_standard.tex",
    "rq5_last_tail_gain.tex",
    "rq5_head_tail_gain.tex",
    "rq5_full_tail_gain.tex",
    "rq2_last_recovered_share.tex",
    "rq2_head_recovered_share.tex",
)


def artifact_mappings() -> dict[Path, Path]:
    mappings = dict(FIGURE_MAPPINGS)
    for filename in TABLE_FILENAMES:
        mappings[
            Path("results/additional/tables") / filename
        ] = Path("paper/generated/additional-datasets") / filename
    return mappings


def sync_artifacts(*, check: bool = False) -> int:
    mismatches: list[str] = []
    mappings = artifact_mappings()
    for relative_source, relative_destination in mappings.items():
        source = REPO_ROOT / relative_source
        destination = REPO_ROOT / relative_destination
        if not source.is_file():
            mismatches.append(f"missing source {relative_source.as_posix()}")
            continue
        if check:
            if not destination.is_file():
                mismatches.append(
                    f"missing release file {relative_destination.as_posix()}"
                )
            elif source.read_bytes() != destination.read_bytes():
                mismatches.append(
                    f"stale release file {relative_destination.as_posix()}"
                )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if mismatches:
        raise RuntimeError("\n".join(mismatches))
    return len(mappings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the release copies match without writing files",
    )
    args = parser.parse_args()
    count = sync_artifacts(check=args.check)
    verb = "Verified" if args.check else "Updated"
    print(f"{verb} {count} additional-dataset paper artifacts")


if __name__ == "__main__":
    main()
