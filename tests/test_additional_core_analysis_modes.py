from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / relative_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rq1_additional_mode_uses_separate_input_and_output(
    tmp_path: Path,
) -> None:
    module = load_script("rq1_additional_mode", "scripts/analyze_rq1_accuracy.py")

    args = module.parse_args(
        [
            "--dataset-mode",
            "additional",
            "--additional-raw-root",
            str(tmp_path),
        ]
    )

    assert args.dataset_mode is module.AnalysisDatasetMode.ADDITIONAL
    assert args.input_root == tmp_path
    assert args.output_root == (
        REPO_ROOT / "results" / "additional" / "rq1" / "accuracy"
    )


def test_rq1_additional_table_contains_only_new_datasets() -> None:
    module = load_script("rq1_additional_table", "scripts/analyze_rq1_accuracy.py")
    module.DATASET_ORDER = module.ADDITIONAL_DATASET_ORDER
    summaries = [
        {
            "dataset": dataset,
            "model": model,
            "mean_gap_whole_minus_tail": 0.1,
            "bootstrap_ci_lower_gap": 0.05,
            "bootstrap_ci_upper_gap": 0.15,
            "gap_ci_excludes_zero": True,
        }
        for dataset in module.DATASET_ORDER
        for model in module.MODEL_ORDER
    ]

    rendered = module.build_appendix_table(summaries)

    assert "Model & Actor & BlogCatalog" in rendered
    assert "lcc@{}" in rendered
    assert "PubMed" not in rendered
    assert r"\label{tab:additional-erm-tail-gap-accuracy}" in rendered
    assert "ERM accuracy gap, whole-test accuracy minus tail accuracy." in rendered
    assert r"\gaincell{+.100}{[.050,.150]}" in rendered


def test_rq2_additional_mode_uses_additional_prerequisites(
    tmp_path: Path,
) -> None:
    module = load_script(
        "rq2_additional_mode", "scripts/analyze_rq2_within_class.py"
    )

    args = module.parse_args(
        [
            "--dataset-mode",
            "additional",
            "--additional-raw-root",
            str(tmp_path),
        ]
    )

    assert args.input_root == tmp_path
    assert args.output_dir == REPO_ROOT / "results" / "additional" / "rq2"
    assert args.gap_table == (
        REPO_ROOT
        / "results"
        / "additional"
        / "aggregates"
        / "erm_tail_gaps.csv"
    )
    assert args.accuracy_gap_table == (
        REPO_ROOT
        / "results"
        / "additional"
        / "rq1"
        / "accuracy"
        / "whole_vs_tail_accuracy.csv"
    )


def test_rq2_records_explicit_external_output_path(tmp_path: Path) -> None:
    module = load_script(
        "rq2_external_output", "scripts/analyze_rq2_within_class.py"
    )
    output = tmp_path / "summary.csv"

    assert module.recorded_path(output) == str(output.resolve())


def test_rq2_additional_table_has_separate_label_and_two_columns() -> None:
    module = load_script(
        "rq2_additional_table", "scripts/analyze_rq2_within_class.py"
    )
    module.DATASET_ORDER = list(module.ADDITIONAL_DATASET_ORDER)
    rows = [
        {
            "dataset": dataset,
            "model": model,
            "preparation": "erm",
            "mean_class_macro_accuracy_contrast": 0.1,
            "bootstrap_ci_lower_class_macro_accuracy": 0.05,
            "bootstrap_ci_upper_class_macro_accuracy": 0.15,
            "class_macro_accuracy_ci_excludes_zero": True,
        }
        for dataset in module.DATASET_ORDER
        for model in module.MODEL_ORDER
    ]

    rendered = "\n".join(module.latex_table(rows, "within_accuracy"))

    assert "Model & Actor & BlogCatalog" in rendered
    assert r"\label{tab:additional-rq2-class-macro-accuracy}" in rendered
    assert "PubMed" not in rendered
    assert r"\gaincell{+.100}{[.050,.150]}" in rendered
    assert (
        "Per-class non-tail accuracy minus tail accuracy, averaged equally "
        "over classes."
    ) in rendered
