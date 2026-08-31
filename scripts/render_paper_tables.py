#!/usr/bin/env python3
"""Render the released result and compute tables from canonical outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

from hfd.analysis_datasets import (
    ORIGINAL_DATASET_ORDER,
    AnalysisDatasetMode,
    DATASET_REGISTRY,
    dataset_order,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "main"
DEFAULT_FG_RESULTS_ROOT = REPO_ROOT / "results" / "fg_ccdb"
DEFAULT_DATASET_MANIFEST_ROOT = REPO_ROOT / "manifests" / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "generated" / "tables"

DATASET_ORDER = ORIGINAL_DATASET_ORDER
DATASET_MODE = AnalysisDatasetMode.ORIGINAL
MODEL_ORDER = (
    "mlp",
    "gcn",
    "graphsage-mean",
    "h2gcn",
    "gpr-gnn",
    "acm-gcn",
    "linkx",
)
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage-mean": "GraphSAGE-mean",
    "h2gcn": "H2GCN-2",
    "gpr-gnn": "GPR-GNN",
    "acm-gcn": "ACM-GCN",
    "linkx": "LINKX",
}
DESIGNED_MODELS = ("h2gcn", "gpr-gnn", "acm-gcn")
STANDARD_MODELS = ("gcn", "graphsage-mean")
RETRAINING_PREPARATIONS = ("last", "head", "full")
DFR_SENSITIVITY_MODELS = ("acm-gcn", "gcn", "graphsage-mean")
DFR_SENSITIVITY_METHODS = ("erm", "last", "head", "full", "dfr-val")
FG_MODEL_ORDER = ("graphsage-mean", "gcn", "acm-gcn")
DATASET_LABELS = {
    name: metadata.display_label for name, metadata in DATASET_REGISTRY.items()
}

INPUT_PATHS = {
    "gpu_hours": Path("compute/gpu_hours_by_preparation.csv"),
    "rq1_macro_f1": Path("aggregates/erm_tail_gaps.csv"),
    "rq1_accuracy": Path("rq1/accuracy/whole_vs_tail_accuracy.csv"),
    "rq2_within_class": Path(
        "rq2/within_class_accuracy_cell_summary.csv"
    ),
    "rq2_adjusted": Path("rq2/class_adjusted_cell_summary.csv"),
    "rq4_pairwise": Path("rq4/pairwise_tail_gap_contrasts.csv"),
    "rq5_gains": Path("aggregates/paired_gains.csv"),
    "rq5_dfr_absolute": Path(
        "rq5/dfr_pubmed_sensitivity/absolute_score_summary.csv"
    ),
}
FG_INPUT_PATHS = {
    "fg_ccdb": Path("paired_results.csv"),
}
DATASET_MANIFEST_PATHS = {
    f"dataset_manifest_{dataset}": Path(f"{dataset}.json")
    for dataset in DATASET_ORDER
}
ADDITIONAL_INPUT_PATHS = {
    "recovered_shares": Path("aggregates/recovered_shares.csv"),
}

OUTPUT_FILENAMES = (
    "local_homophily_groups.tex",
    "training_gpu_hours.tex",
    "rq1_macro_f1_tail_gap.tex",
    "rq1_accuracy_tail_gap.tex",
    "rq2_within_class_accuracy.tex",
    "rq2_adjusted_accuracy.tex",
    "rq2_adjusted_macro_f1.tex",
    "rq4_designed_vs_standard.tex",
    "rq5_last_tail_gain.tex",
    "rq5_head_tail_gain.tex",
    "rq5_full_tail_gain.tex",
    "rq5_dfr_tail_macro_f1.tex",
    "fg_ccdb_pubmed.tex",
)
ADDITIONAL_OUTPUT_FILENAMES = (
    "local_homophily_groups.tex",
    "training_gpu_hours.tex",
    "rq1_macro_f1_tail_gap.tex",
    "rq1_accuracy_tail_gap.tex",
    "rq2_within_class_accuracy.tex",
    "rq2_adjusted_accuracy.tex",
    "rq2_adjusted_macro_f1.tex",
    "rq2_last_recovered_share.tex",
    "rq2_head_recovered_share.tex",
    "rq4_designed_vs_standard.tex",
    "rq5_last_tail_gain.tex",
    "rq5_head_tail_gain.tex",
    "rq5_full_tail_gain.tex",
)


def configure_dataset_mode(mode: AnalysisDatasetMode | str) -> None:
    """Select the dataset panel rendered by all table helpers."""

    global DATASET_MODE, DATASET_ORDER
    DATASET_MODE = AnalysisDatasetMode(mode)
    DATASET_ORDER = dataset_order(DATASET_MODE)


def table_label(label: str) -> str:
    """Use distinct labels when additional tables share a paper with originals."""

    if DATASET_MODE is AnalysisDatasetMode.ORIGINAL:
        return label
    if label.startswith("tab:"):
        return "tab:additional-" + label.removeprefix("tab:")
    return f"additional-{label}"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render every released result and compute table from the canonical "
            "analysis CSV files."
        )
    )
    parser.add_argument(
        "--dataset-mode",
        choices=(
            AnalysisDatasetMode.ORIGINAL.value,
            AnalysisDatasetMode.ADDITIONAL.value,
        ),
        default=AnalysisDatasetMode.ORIGINAL.value,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Root containing the canonical results/main analysis outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the generated standalone LaTeX tables.",
    )
    parser.add_argument(
        "--fg-results-root",
        type=Path,
        help="Root containing the exploratory FG-CCDB analysis outputs.",
    )
    parser.add_argument(
        "--dataset-manifest-root",
        type=Path,
        help="Root containing the five registered dataset manifests.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Check that existing outputs equal a fresh rendering without "
            "writing any files."
        ),
    )
    args = parser.parse_args()
    mode = AnalysisDatasetMode(args.dataset_mode)
    result_root = REPO_ROOT / "results" / (
        "main" if mode is AnalysisDatasetMode.ORIGINAL else mode.value
    )
    args.dataset_mode = mode
    args.results_root = args.results_root or result_root
    args.output_dir = args.output_dir or (
        DEFAULT_OUTPUT_DIR
        if mode is AnalysisDatasetMode.ORIGINAL
        else result_root / "tables"
    )
    args.fg_results_root = args.fg_results_root or DEFAULT_FG_RESULTS_ROOT
    args.dataset_manifest_root = args.dataset_manifest_root or (
        DEFAULT_DATASET_MANIFEST_ROOT
        if mode is AnalysisDatasetMode.ORIGINAL
        else REPO_ROOT / "manifests" / "additional-datasets"
    )
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required table input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Required table input is empty: {path}")
    return rows


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required table input does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON table input is not an object: {path}")
    return payload


def parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must contain True or False, found {value!r}")


def sha256_text_file(path: Path) -> str:
    """Hash UTF-8 text after normalising platform line endings."""
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def relative_to_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def format_signed(value: str | float) -> str:
    formatted = f"{float(value):+.3f}"
    formatted = "0.000" if formatted in {"+0.000", "-0.000"} else formatted
    if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL:
        formatted = formatted.replace("+0.", "+.", 1).replace("-0.", "-.", 1)
        if formatted.startswith("0."):
            formatted = formatted[1:]
    return formatted


def format_endpoint(
    value: str | float,
    *,
    explicit_positive: bool = False,
) -> str:
    number = float(value)
    formatted = (
        f"{number:+.3f}" if explicit_positive else f"{number:.3f}"
    )
    if formatted in {"+0.000", "-0.000"}:
        formatted = "0.000"
    if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL:
        formatted = formatted.replace("+0.", "+.", 1).replace("-0.", "-.", 1)
        if formatted.startswith("0."):
            formatted = formatted[1:]
    return formatted


def index_complete_panel(
    rows: Iterable[dict[str, str]],
    *,
    key_fields: tuple[str, ...],
    expected_keys: set[tuple[str, ...]],
    name: str,
    predicate: Callable[[dict[str, str]], bool] | None = None,
) -> dict[tuple[str, ...], dict[str, str]]:
    panel: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        try:
            key = tuple(row[field] for field in key_fields)
        except KeyError as exc:
            raise ValueError(
                f"{name} is missing required column {exc.args[0]!r}"
            ) from exc
        if key in panel:
            raise ValueError(f"{name} contains duplicate cell {key}")
        panel[key] = row
    actual_keys = set(panel)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{name} does not contain the expected panel; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return panel


def matrix_expected_keys() -> set[tuple[str, str]]:
    return {
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    }


def latex_cell(
    row: Mapping[str, str],
    *,
    mean_field: str,
    lower_field: str,
    upper_field: str,
    significance_field: str,
    dagger_field: str | None = None,
    explicit_interval_sign: bool = False,
) -> str:
    try:
        excludes_zero = parse_bool(
            row[significance_field],
            field=significance_field,
        )
        mean_text = format_signed(row[mean_field])
        if dagger_field is not None and not parse_bool(
            row[dagger_field],
            field=dagger_field,
        ):
            mean_text += r"^{\dagger}"
        lower = format_endpoint(
            row[lower_field],
            explicit_positive=explicit_interval_sign,
        )
        upper = format_endpoint(
            row[upper_field],
            explicit_positive=explicit_interval_sign,
        )
    except KeyError as exc:
        raise ValueError(
            f"Table input is missing required column {exc.args[0]!r}"
        ) from exc
    macro = "gaincell" if excludes_zero else "uncertaingaincell"
    return f"\\{macro}{{{mean_text}}}{{[{lower},{upper}]}}"


def render_matrix_table(
    panel: Mapping[tuple[str, str], Mapping[str, str]],
    *,
    caption_lines: tuple[str, ...],
    label: str,
    model_width: int,
    cell_builder: Callable[[Mapping[str, str]], str],
) -> str:
    if DATASET_MODE is AnalysisDatasetMode.ORIGINAL:
        column_spec = r"@{\extracolsep{\fill}}lccccc@{}"
        header_lines = [
            r"    & \multicolumn{2}{c}{Low-homophily tail}",
            r"    & \multicolumn{3}{c}{High-homophily tail} \\",
            r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
            (
                r"    Model & PubMed & ogbn-arxiv & Roman-empire "
                r"& Amazon-ratings & arxiv-year \\"
            ),
        ]
    else:
        column_spec = (
            r"@{\extracolsep{\fill}}l"
            + "c" * len(DATASET_ORDER)
            + "@{}"
        )
        family = DATASET_REGISTRY[DATASET_ORDER[0]].family
        if any(
            DATASET_REGISTRY[dataset].family != family
            for dataset in DATASET_ORDER
        ):
            raise ValueError(
                "Non-original matrix tables require one contiguous dataset family"
            )
        family_title = (
            "Low-homophily tail"
            if family == "homophilic"
            else "High-homophily tail"
        )
        end_column = 1 + len(DATASET_ORDER)
        header_lines = [
            "    & "
            f"\\multicolumn{{{len(DATASET_ORDER)}}}{{c}}{{{family_title}}} \\\\",
            f"    \\cmidrule(lr){{2-{end_column}}}",
            "    Model & "
            + " & ".join(DATASET_LABELS[dataset] for dataset in DATASET_ORDER)
            + r" \\",
        ]
    size_lines = (
        [r"  \scriptsize"]
        if DATASET_MODE is AnalysisDatasetMode.ORIGINAL
        else []
    )
    if DATASET_MODE is AnalysisDatasetMode.ORIGINAL:
        begin_tabular = f"  \\begin{{tabular*}}{{\\linewidth}}{{{column_spec}}}"
        end_tabular = r"  \end{tabular*}"
    else:
        begin_tabular = f"  \\begin{{tabular}}{{{column_spec}}}"
        end_tabular = r"  \end{tabular}"
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        *caption_lines,
        f"  \\label{{{table_label(label)}}}",
        *size_lines,
        begin_tabular,
        r"    \toprule",
        *header_lines,
        r"    \midrule",
    ]
    for model in MODEL_ORDER:
        cells = [
            cell_builder(panel[(dataset, model)])
            for dataset in DATASET_ORDER
        ]
        lines.append(
            f"    {MODEL_LABELS[model]:<{model_width}} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            end_tabular,
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_gpu_hours(rows: list[dict[str, str]]) -> str:
    expected = {("erm",), ("last",), ("head",), ("full",), ("total",)}
    panel = index_complete_panel(
        rows,
        key_fields=("preparation",),
        expected_keys=expected,
        name="GPU-hours input",
    )
    labels = {
        "erm": r"\(\Aerm\)",
        "last": r"\(\Alast\)",
        "head": r"\(\Ahead\)",
        "full": r"\(\Afull\)",
        "total": "Total",
    }
    row_lines = []
    for preparation in ("erm", "last", "head", "full", "total"):
        row = panel[(preparation,)]
        try:
            run_count = int(row["run_count"])
            gpu_hours = float(row["gpu_hours"])
        except KeyError as exc:
            raise ValueError(
                f"GPU-hours input is missing column {exc.args[0]!r}"
            ) from exc
        if preparation == "erm":
            prefix = f"    {labels[preparation]}  "
        elif preparation == "total":
            prefix = f"    {labels[preparation]:<11}"
        else:
            prefix = f"    {labels[preparation]} "
        row_lines.append(
            f"{prefix}& {run_count:,} & {gpu_hours:.3f} \\\\"
        )
        if preparation == "full":
            row_lines.append(r"    \midrule")
    caption = (
        r"  \caption{Total GPU-hours by training procedure.}"
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else r"  \caption{Total GPU-hours by preparation.}"
    )
    header = (
        r"    Procedure & Runs & GPU-hours \\"
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else r"    Preparation & Runs & GPU-hours \\"
    )
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        caption,
        f"  \\label{{{table_label('tab:training-gpu-hours')}}}",
        *(
            [r"  \small"]
            if DATASET_MODE is AnalysisDatasetMode.ORIGINAL
            else []
        ),
        r"  \begin{tabular}{@{}lrr@{}}",
        r"    \toprule",
        header,
        r"    \midrule",
        *row_lines,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def render_rq1_macro_f1(rows: list[dict[str, str]]) -> str:
    panel = index_complete_panel(
        rows,
        key_fields=("dataset", "model"),
        expected_keys=matrix_expected_keys(),
        name="RQ1 macro-F1 input",
    )
    return render_matrix_table(
        panel,
        caption_lines=(
            (
                r"  \caption{ERM signed tail gap: whole-test macro-F1 "
                r"minus tail macro-F1."
            ),
            (
                r"  Each cell gives the mean paired difference over 30 "
                r"matched split--seed runs"
            ),
            (
                r"  and its 95\% percentile-bootstrap interval. Positive "
                r"values indicate lower"
            ),
            r"  tail performance; grey cells contain zero.}",
        ),
        label="tab:erm-tail-gap",
        model_width=15,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_tail_gap",
            lower_field="bootstrap_ci_lower",
            upper_field="bootstrap_ci_upper",
            significance_field="ci_excludes_zero",
        ),
    )


def render_rq1_accuracy(rows: list[dict[str, str]]) -> str:
    panel = index_complete_panel(
        rows,
        key_fields=("dataset", "model"),
        expected_keys=matrix_expected_keys(),
        name="RQ1 accuracy input",
    )
    return render_matrix_table(
        panel,
        caption_lines=(
            r"  \caption{ERM accuracy gap, whole-test accuracy minus tail",
            (
                r"  accuracy. Each cell gives the mean paired difference "
                r"over 30 matched"
            ),
            (
                r"  split--seed runs and its 95\% percentile-bootstrap "
                r"interval. Positive"
            ),
            r"  values indicate lower tail accuracy; grey cells contain zero.}",
        ),
        label="tab:erm-tail-gap-accuracy",
        model_width=16,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_gap_whole_minus_tail",
            lower_field="bootstrap_ci_lower_gap",
            upper_field="bootstrap_ci_upper_gap",
            significance_field="gap_ci_excludes_zero",
        ),
    )


def erm_matrix_panel(
    rows: list[dict[str, str]],
    *,
    name: str,
) -> dict[tuple[str, str], dict[str, str]]:
    return index_complete_panel(
        rows,
        key_fields=("dataset", "model"),
        expected_keys=matrix_expected_keys(),
        name=name,
        predicate=lambda row: row.get("preparation") == "erm",
    )


def render_rq2_within_class(rows: list[dict[str, str]]) -> str:
    panel = erm_matrix_panel(rows, name="RQ2 within-class input")
    caption_lines = (
        (
            r"  \caption{",
            (
                r"    Per-class non-tail accuracy minus tail accuracy, "
                r"averaged equally over classes. Each cell reports the mean "
                r"over the 30 runs and the 95\% percentile-bootstrap interval. "
                r"Grey cells denote confidence intervals that contain zero."
            ),
            r"  }",
        )
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else (
            (
                r"  \caption{ERM within-class accuracy contrast, non-tail "
                r"minus tail. Within"
            ),
            (
                r"  each run, class contrasts are averaged equally over "
                r"true classes represented in both"
            ),
            (
                r"  the tail and non-tail subsets; cells report the mean "
                r"of the 30 run-level"
            ),
            (
                r"  contrasts and the 95\% percentile-bootstrap interval. "
                r"Positive values"
            ),
            r"  indicate lower tail accuracy; grey cells contain zero.}",
        )
    )
    return render_matrix_table(
        panel,
        caption_lines=caption_lines,
        label="tab:rq2-class-macro-accuracy",
        model_width=16,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_class_macro_accuracy_contrast",
            lower_field="bootstrap_ci_lower_class_macro_accuracy",
            upper_field="bootstrap_ci_upper_class_macro_accuracy",
            significance_field="class_macro_accuracy_ci_excludes_zero",
        ),
    )


def render_rq2_adjusted_accuracy(
    rows: list[dict[str, str]],
) -> str:
    panel = erm_matrix_panel(rows, name="RQ2 adjusted input")
    caption_lines = (
        (
            r"  \caption{",
            (
                r"    Class-adjusted accuracy tail gaps for all models on "
                r"Actor and BlogCatalog. Entries are the mean and 95\% "
                r"bootstrap interval across 30 runs. Grey cells include zero, "
                r"and $\dagger$ marks cells whose unadjusted accuracy tail gap "
                r"includes zero."
            ),
            r"  }",
        )
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else (
            (
                r"  \caption{Class-composition-adjusted ERM accuracy gap, "
                r"whole test minus tail. The tail is reweighted to the "
                r"whole-test proportions of true classes present in the "
                r"tail; the whole-test score uses the same class set. "
                r"Entries are the mean and 95\% bootstrap "
                r"interval across 30 runs. Positive values indicate lower "
                r"tail accuracy; grey cells include zero; $\dagger$ marks "
                r"cells whose unadjusted RQ1 accuracy interval includes zero.}"
            ),
        )
    )
    return render_matrix_table(
        panel,
        caption_lines=caption_lines,
        label="tab:rq2-adjusted-accuracy",
        model_width=16,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_adjusted_accuracy_gap",
            lower_field="bootstrap_ci_lower_adjusted_accuracy_gap",
            upper_field="bootstrap_ci_upper_adjusted_accuracy_gap",
            significance_field="adjusted_accuracy_gap_ci_excludes_zero",
            dagger_field="erm_accuracy_gap_ci_excludes_zero",
        ),
    )


def render_rq2_adjusted_macro_f1(
    rows: list[dict[str, str]],
) -> str:
    panel = erm_matrix_panel(rows, name="RQ2 adjusted input")
    caption_lines = (
        (
            r"  \caption{",
            (
                r"    Class-adjusted macro-F1 tail gaps for all models on "
                r"Actor and BlogCatalog. Entries are the mean and 95\% "
                r"bootstrap interval across 30 runs. Grey cells include zero, "
                r"and $\dagger$ marks cells whose unadjusted macro-F1 tail gap "
                r"includes zero."
            ),
            r"  }",
        )
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else (
            (
                r"  \caption{Class-composition-adjusted ERM macro-F1 gap, "
                r"whole test minus tail. The tail is reweighted to the "
                r"whole-test proportions of true classes present in the "
                r"tail; the whole-test score uses the same class set. "
                r"Entries are the mean and 95\% bootstrap "
                r"interval across 30 runs. Positive values indicate lower "
                r"tail macro-F1; grey cells include zero; $\dagger$ marks "
                r"cells whose unadjusted RQ1 macro-F1 interval includes zero.}"
            ),
        )
    )
    return render_matrix_table(
        panel,
        caption_lines=caption_lines,
        label="tab:rq2-adjusted-macro-f1",
        model_width=16,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_adjusted_macro_f1_gap",
            lower_field="bootstrap_ci_lower_adjusted_macro_f1_gap",
            upper_field="bootstrap_ci_upper_adjusted_macro_f1_gap",
            significance_field="adjusted_macro_f1_gap_ci_excludes_zero",
            dagger_field="erm_tail_gap_ci_excludes_zero",
        ),
    )


def render_rq4_pairwise(rows: list[dict[str, str]]) -> str:
    expected = {
        (dataset, designed, standard)
        for dataset in DATASET_ORDER
        for designed in DESIGNED_MODELS
        for standard in STANDARD_MODELS
    }
    panel = index_complete_panel(
        rows,
        key_fields=("dataset", "designed_model", "standard_model"),
        expected_keys=expected,
        name="RQ4 pairwise input",
    )
    if DATASET_MODE is AnalysisDatasetMode.ORIGINAL:
        column_spec = "@{}llccccc@{}"
        header_lines = [
            r"    & & \multicolumn{2}{c}{Low-homophily tail}",
            r"    & \multicolumn{3}{c}{High-homophily tail} \\",
            r"    \cmidrule(lr){3-4}\cmidrule(lr){5-7}",
            (
                r"    Designed model & Standard baseline & PubMed & "
                r"ogbn-arxiv & Roman-empire & Amazon-ratings & arxiv-year \\"
            ),
        ]
    else:
        column_spec = "@{}ll" + "c" * len(DATASET_ORDER) + "@{}"
        family = DATASET_REGISTRY[DATASET_ORDER[0]].family
        if any(
            DATASET_REGISTRY[dataset].family != family
            for dataset in DATASET_ORDER
        ):
            raise ValueError(
                "Non-original RQ4 tables require one contiguous dataset family"
            )
        family_title = (
            "Low-homophily tail"
            if family == "homophilic"
            else "High-homophily tail"
        )
        end_column = 2 + len(DATASET_ORDER)
        header_lines = [
            "    & & "
            f"\\multicolumn{{{len(DATASET_ORDER)}}}{{c}}{{{family_title}}} \\\\",
            f"    \\cmidrule(lr){{3-{end_column}}}",
            "    Designed model & Standard baseline & "
            + " & ".join(DATASET_LABELS[dataset] for dataset in DATASET_ORDER)
            + r" \\",
        ]
    caption = (
        (
            r"  \caption{\textbf{RQ1c}. Macro-F1 tail-gap comparison between "
            r"each heterophily-designed model and each standard aggregation "
            r"baseline. Each cell is heterophily-model gap minus standard-baseline "
            r"gap, with its 95\% percentile-bootstrap interval over the 30 matched "
            r"split--seed runs. Negative values indicate a smaller tail gap for "
            r"the heterophily-designed model; grey cells contain zero and are not "
            r"statistically significant.}"
        )
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else (
            r"  \caption{Paired ERM tail-gap contrast between each "
            r"heterophily-designed model and each standard aggregation "
            r"baseline. Each cell is designed-model gap minus "
            r"standard-baseline gap, with its 95\% percentile-bootstrap "
            r"interval over 30 matched split--seed runs. Negative values "
            r"indicate a smaller signed tail gap for the "
            r"heterophily-designed model; grey cells contain zero.}"
        )
    )
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        caption,
        f"  \\label{{{table_label('tab:rq4-designed-vs-standard')}}}",
        *(
            [r"  \scriptsize"]
            if DATASET_MODE is AnalysisDatasetMode.ORIGINAL
            else []
        ),
        (
            r"  \setlength{\tabcolsep}{2.5pt}"
            if DATASET_MODE is AnalysisDatasetMode.ORIGINAL
            else r"  \setlength{\tabcolsep}{5pt}"
        ),
        f"  \\begin{{tabular}}{{{column_spec}}}",
        r"    \toprule",
        *header_lines,
        r"    \midrule",
    ]
    for designed_index, designed_model in enumerate(DESIGNED_MODELS):
        for standard_index, standard_model in enumerate(STANDARD_MODELS):
            designed_label = (
                rf"\multirow{{2}}{{*}}{{{MODEL_LABELS[designed_model]}}}"
                if standard_index == 0
                else ""
            )
            cells = []
            for dataset in DATASET_ORDER:
                row = panel[(dataset, designed_model, standard_model)]
                cells.append(
                    latex_cell(
                        row,
                        mean_field="mean_tail_gap_contrast",
                        lower_field="bootstrap_ci_lower",
                        upper_field="bootstrap_ci_upper",
                        significance_field="ci_excludes_zero",
                        explicit_interval_sign=True,
                    )
                )
            lines.append(
                "    "
                + designed_label
                + " & "
                + MODEL_LABELS[standard_model]
                + " & "
                + " & ".join(cells)
                + r" \\"
            )
        if designed_index < len(DESIGNED_MODELS) - 1:
            lines.append(r"    \addlinespace[2pt]")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_rq5_gain(
    rows: list[dict[str, str]],
    *,
    preparation: str,
) -> str:
    if preparation not in RETRAINING_PREPARATIONS:
        raise ValueError(f"Unknown RQ5 preparation: {preparation}")
    panel = index_complete_panel(
        rows,
        key_fields=("dataset", "model"),
        expected_keys=matrix_expected_keys(),
        name=f"RQ5 {preparation} input",
        predicate=lambda row: row.get("preparation") == preparation,
    )
    additional_captions = {
        "last": (
            r"  \caption{",
            (
                r"    \textbf{RQ2a}. Change in tail macro-F1 after last layer "
                r"retraining, \(\Alast-\Aerm\). Each cell gives the mean "
                r"difference over the 30 runs and its 95\% percentile-bootstrap "
                r"interval. Grey cells contain zero and hence are not "
                r"statistically significant.}"
            ),
        ),
        "head": (
            r"  \caption{",
            (
                r"    \textbf{RQ2b}. Change in tail macro-F1 after head "
                r"retraining, \(\Ahead-\Aerm\). Each cell gives the mean "
                r"difference over the 30 runs and its 95\% percentile-bootstrap "
                r"interval. Grey cells contain zero and hence are not "
                r"statistically significant.}"
            ),
        ),
        "full": (
            r"  \caption{",
            (
                r"    \textbf{RQ2c}. Change in tail macro-F1 after full model "
                r"retraining, \(\Afull-\Aerm\). Each cell gives the mean "
                r"difference over the 30 runs and its 95\% percentile-bootstrap "
                r"interval. Grey cells contain zero and hence are not "
                r"statistically significant."
            ),
            r"  }",
        ),
    }
    original_captions = {
        "last": (
            r"  \caption{Final-layer change in tail macro-F1, "
            r"\(\Alast-\Aerm\). Each cell",
            (
                r"  gives the mean paired difference over 30 matched "
                r"split--seed runs and its"
            ),
            (
                r"  95\% percentile-bootstrap interval. Positive values "
                r"indicate improvement;"
            ),
            r"  grey cells contain zero.}",
        ),
        "head": (
            r"  \caption{Head change in tail macro-F1, "
            r"\(\Ahead-\Aerm\). Each cell gives the",
            (
                r"  mean paired difference over 30 matched split--seed runs "
                r"and its 95\%"
            ),
            (
                r"  percentile-bootstrap interval. Positive values indicate "
                r"improvement; grey"
            ),
            r"  cells contain zero.}",
        ),
        "full": (
            r"  \caption{Full-model change in tail macro-F1, "
            r"\(\Afull-\Aerm\). Each cell",
            (
                r"  gives the mean paired difference over 30 matched "
                r"split--seed runs and its"
            ),
            (
                r"  95\% percentile-bootstrap interval. Positive values "
                r"indicate improvement;"
            ),
            r"  grey cells contain zero.}",
        ),
    }
    captions = (
        additional_captions
        if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL
        else original_captions
    )
    labels = {
        "last": "tab:last-tail-gain",
        "head": "tab:head-tail-gain",
        "full": "tab:full-tail-gain",
    }
    return render_matrix_table(
        panel,
        caption_lines=captions[preparation],
        label=labels[preparation],
        model_width=15,
        cell_builder=lambda row: latex_cell(
            row,
            mean_field="mean_paired_gain",
            lower_field="bootstrap_ci_lower",
            upper_field="bootstrap_ci_upper",
            significance_field="ci_excludes_zero",
        ),
    )


def render_rq2_recovered_share(
    recovered_rows: list[dict[str, str]],
    gain_rows: list[dict[str, str]],
    *,
    preparation: str,
) -> str:
    """Render the registered RQ2 recovered-share table for one refit."""

    if preparation not in {"last", "head"}:
        raise ValueError(f"Unknown recovered-share preparation: {preparation}")
    graph_models = tuple(model for model in MODEL_ORDER if model != "mlp")
    expected = {
        (dataset, model)
        for dataset in DATASET_ORDER
        for model in graph_models
    }
    recovered_panel = index_complete_panel(
        recovered_rows,
        key_fields=("dataset", "model"),
        expected_keys=expected,
        name=f"RQ2 {preparation} recovered-share input",
        predicate=lambda row: (
            row.get("preparation") == preparation
            and row.get("model") in graph_models
        ),
    )
    gain_panel = index_complete_panel(
        gain_rows,
        key_fields=("dataset", "model"),
        expected_keys=expected,
        name=f"RQ2 {preparation} gain input",
        predicate=lambda row: (
            row.get("preparation") == preparation
            and row.get("model") in graph_models
        ),
    )
    rq_label = "RQ2a" if preparation == "last" else "RQ2b"
    refit = "final-layer" if preparation == "last" else "head"
    rho = "last" if preparation == "last" else "head"
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        (
            f"  \\caption{{\\textbf{{{rq_label}}}. Recovered share "
            f"\\(\\rho_{{\\mathrm{{{rho}}}}}\\) after {refit} refitting; "
            "cells are grey when the 95\\% paired-bootstrap interval for the "
            "numerator contains zero or "
            "\\(\\hat A_{\\mathrm{full}}-\\hat A_{\\mathrm{ERM}}<0.03\\).}"
        ),
        (
            "  \\label{"
            + table_label(f"tab:rho-{rho}-recovered-share")
            + "}"
        ),
        r"  \setlength{\tabcolsep}{5.5pt}",
        (
            r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l"
            + "c" * len(DATASET_ORDER)
            + "@{}}"
        ),
        r"    \toprule",
    ]
    family = DATASET_REGISTRY[DATASET_ORDER[0]].family
    if any(
        DATASET_REGISTRY[dataset].family != family
        for dataset in DATASET_ORDER
    ):
        raise ValueError(
            "Recovered-share table requires one contiguous dataset family"
        )
    family_title = (
        "Low-homophily tail"
        if family == "homophilic"
        else "High-homophily tail"
    )
    end_column = 1 + len(DATASET_ORDER)
    lines.extend(
        [
            "    & "
            f"\\multicolumn{{{len(DATASET_ORDER)}}}{{c}}{{{family_title}}} \\\\",
            f"    \\cmidrule(lr){{2-{end_column}}}",
            "    Model & "
            + " & ".join(DATASET_LABELS[dataset] for dataset in DATASET_ORDER)
            + r" \\",
            r"    \midrule",
        ]
    )
    for model in graph_models:
        cells: list[str] = []
        for dataset in DATASET_ORDER:
            recovered = recovered_panel[(dataset, model)]
            gain = gain_panel[(dataset, model)]
            denominator = float(recovered["full_mean_paired_gain"])
            if denominator == 0.0:
                raise ValueError(
                    f"{dataset}/{model}: full-model mean gain is zero"
                )
            value = float(recovered["mean_paired_gain"]) / denominator
            qualified = parse_bool(
                recovered["qualifies"], field="qualifies"
            ) and parse_bool(
                gain["ci_excludes_zero"], field="ci_excludes_zero"
            )
            cell = "$" + f"{value:.3f}" + "$"
            if not qualified:
                cell = r"\cellcolor{black!10}" + cell
            cells.append(cell)
        lines.append(
            f"    {MODEL_LABELS[model]} & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular*}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_local_homophily_groups(
    manifests: Mapping[str, Mapping],
) -> str:
    if set(manifests) != set(DATASET_ORDER):
        raise ValueError(
            "Local-homophily table requires exactly the selected datasets"
        )

    centering = (
        r"    \centering\footnotesize"
        if DATASET_MODE is AnalysisDatasetMode.ORIGINAL
        else r"    \centering"
    )
    lines = [
        r"\begin{table}[H]",
        centering,
        r"    \begin{tabular}{@{}llrr}",
        r"        \toprule",
        r"        Dataset & Bin & Nodes & \% of nodes\\",
        r"        \midrule",
    ]
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        manifest = manifests[dataset]
        if manifest.get("dataset") != dataset:
            raise ValueError(
                f"Dataset manifest key {dataset!r} contains "
                f"{manifest.get('dataset')!r}"
            )
        homophily = manifest["local_homophily"]
        bins = homophily["merged_bins"]
        if not bins or sum(int(row["is_tail"]) for row in bins) != 1:
            raise ValueError(f"{dataset}: expected exactly one tail bin")
        mean_h = float(homophily["mean_h"])
        direction = homophily["direction"]
        for bin_index, bin_row in enumerate(bins):
            interval = str(bin_row["interval"])
            merged = len(bin_row["base_bins"]) > 1
            if bin_row["is_tail"]:
                bin_text = rf"$\mathbf{{{interval}}}$"
            else:
                bin_text = rf"${interval}$"
            if merged:
                bin_text += " (merged)"

            if bin_index == 0:
                dataset_text = DATASET_LABELS[dataset]
            elif bin_index == 1:
                dataset_text = (
                    f"({direction}, mean $h_v$ {mean_h:.3f})"
                )
            else:
                dataset_text = ""
            count = f"{int(bin_row['count']):,}".replace(",", "{,}")
            percentage = 100.0 * float(bin_row["fraction_of_eligible"])
            lines.append(
                f"        {dataset_text} & {bin_text} & {count} & "
                f"{percentage:.1f}\\\\"
            )
        if dataset_index < len(DATASET_ORDER) - 1:
            lines.append(r"        \midrule")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{",
            (
                r"        The local homophily bins used throughout. "
                r"\textbf{Bold} type marks the rare \(h_v\) tail---the "
                r"low \(h_v\) bin on homophilic datasets and the high-\(h_v\) "
                r"bin on heterophilic datasets."
            ),
            r"    }",
            f"    \\label{{{table_label('tab:groups')}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_fg_ccdb_pubmed(
    fg_rows: list[dict[str, str]],
    groupdro_rows: list[dict[str, str]],
) -> str:
    expected = {
        (model, preparation)
        for model in FG_MODEL_ORDER
        for preparation in RETRAINING_PREPARATIONS
    }
    fg_panel = index_complete_panel(
        fg_rows,
        key_fields=("model", "preparation"),
        expected_keys=expected,
        name="FG-CCDB PubMed input",
    )
    groupdro_panel = index_complete_panel(
        groupdro_rows,
        key_fields=("model", "preparation"),
        expected_keys=expected,
        name="GroupDRO PubMed input",
        predicate=lambda row: (
            row.get("dataset") == "pubmed"
            and row.get("tail_side") == "low-h_v"
            and row.get("model") in FG_MODEL_ORDER
            and row.get("preparation") in RETRAINING_PREPARATIONS
        ),
    )
    preparation_labels = {
        "last": r"\(\Alast\)",
        "head": r"\(\Ahead\)",
        "full": r"\(\Afull\)",
    }
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        (
            r"  \caption{PubMed tail macro-F1 gain relative to \(\Aerm\) "
            r"under FG-CCDB and GroupDRO. Values are means over 30 matched "
            r"split--seed runs. FG-CCDB entries include 95\% "
            r"percentile-bootstrap intervals; grey intervals contain zero.}"
        ),
        r"  \label{tab:fg-ccdb-pubmed}",
        r"  \small",
        r"  \begin{tabular}{@{}llcc@{}}",
        r"    \toprule",
        (
            r"    Model & Preparation & FG-CCDB gain [95\% CI] "
            r"& GroupDRO gain \\"
        ),
        r"    \midrule",
    ]
    for model_index, model in enumerate(FG_MODEL_ORDER):
        for preparation_index, preparation in enumerate(
            RETRAINING_PREPARATIONS
        ):
            fg_row = fg_panel[(model, preparation)]
            groupdro_row = groupdro_panel[(model, preparation)]
            if int(fg_row["n_pairs"]) != 30 or int(
                groupdro_row["n_pairs"]
            ) != 30:
                raise ValueError(
                    "FG-CCDB and GroupDRO table inputs must each contain "
                    f"30 pairs for {(model, preparation)}"
                )
            model_cell = (
                rf"\multirow{{3}}{{*}}{{{MODEL_LABELS[model]}}}"
                if preparation_index == 0
                else ""
            )
            fg_cell = latex_cell(
                fg_row,
                mean_field="mean_tail_paired_gain_vs_erm",
                lower_field="gain_bootstrap_ci_lower",
                upper_field="gain_bootstrap_ci_upper",
                significance_field="gain_ci_excludes_zero",
            )
            groupdro_cell = format_signed(
                groupdro_row["mean_paired_gain"]
            )
            lines.append(
                f"    {model_cell} & {preparation_labels[preparation]} "
                f"& {fg_cell} & {groupdro_cell} \\\\"
            )
        if model_index < len(FG_MODEL_ORDER) - 1:
            lines.append(r"    \addlinespace[2pt]")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_rq5_dfr_tail_macro_f1(rows: list[dict[str, str]]) -> str:
    expected = {
        (model, method)
        for model in DFR_SENSITIVITY_MODELS
        for method in DFR_SENSITIVITY_METHODS
    }
    panel = index_complete_panel(
        rows,
        key_fields=("model", "method"),
        expected_keys=expected,
        name="RQ5 DFR absolute-score input",
        predicate=lambda row: (
            row.get("dataset") == "pubmed"
            and row.get("dfr_mode") == "val"
            and row.get("metric") == "tail_macro_f1"
        ),
    )
    method_labels = {
        "erm": r"\(\Aerm\)",
        "last": r"\(\Alast\)",
        "head": r"\(\Ahead\)",
        "full": r"\(\Afull\)",
        "dfr-val": r"\(A_{\mathrm{DFR}}\)",
    }
    row_lines = []
    for model in DFR_SENSITIVITY_MODELS:
        cells = []
        for method in DFR_SENSITIVITY_METHODS:
            row = panel[(model, method)]
            try:
                n_matched_runs = int(row["n_matched_runs"])
                mean = float(row["mean"])
                lower = float(row["bootstrap_ci_lower"])
                upper = float(row["bootstrap_ci_upper"])
            except KeyError as exc:
                raise ValueError(
                    "RQ5 DFR absolute-score input is missing column "
                    f"{exc.args[0]!r}"
                ) from exc
            if n_matched_runs != 30:
                raise ValueError(
                    "RQ5 DFR absolute-score input must contain 30 matched "
                    f"runs per cell; found {n_matched_runs} for "
                    f"{model}/{method}"
                )
            cells.append(
                rf"\shortstack{{${mean:.3f}$\\[-1pt]"
                rf"{{\tiny $[{lower:.3f},{upper:.3f}]$}}}}"
            )
        row_lines.append(
            f"    {MODEL_LABELS[model]} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        (
            r"  \caption{PubMed tail macro-F1 for ERM, the three registered "
            r"retraining preparations, and a DFR-style frozen-representation "
            r"refit. Each cell gives the mean over 30 matched split--seed "
            r"runs and its 95\% percentile-bootstrap interval.}"
        ),
        r"  \label{tab:rq5-dfr-tail-macro-f1}",
        r"  \scriptsize",
        (
            r"  \begin{tabular*}{\linewidth}"
            r"{@{\extracolsep{\fill}}lccccc@{}}"
        ),
        r"    \toprule",
        (
            r"    Model & "
            + " & ".join(
                method_labels[method]
                for method in DFR_SENSITIVITY_METHODS
            )
            + r" \\"
        ),
        r"    \midrule",
        *row_lines,
        r"    \bottomrule",
        r"  \end{tabular*}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def render_all(
    results_root: Path,
    fg_results_root: Path = DEFAULT_FG_RESULTS_ROOT,
    dataset_manifest_root: Path = DEFAULT_DATASET_MANIFEST_ROOT,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> tuple[dict[str, str], dict[str, Path]]:
    configure_dataset_mode(dataset_mode)
    selected_inputs = dict(INPUT_PATHS)
    if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL:
        selected_inputs.pop("rq5_dfr_absolute")
        selected_inputs.update(ADDITIONAL_INPUT_PATHS)
    input_paths = {
        name: results_root / relative
        for name, relative in selected_inputs.items()
    }
    if DATASET_MODE is AnalysisDatasetMode.ORIGINAL:
        input_paths.update(
            {
                name: fg_results_root / relative
                for name, relative in FG_INPUT_PATHS.items()
            }
        )
    input_paths.update(
        {
            f"dataset_manifest_{dataset}": (
                dataset_manifest_root / f"{dataset}.json"
            )
            for dataset in DATASET_ORDER
        }
    )
    inputs = {
        name: (
            read_json(path)
            if name.startswith("dataset_manifest_")
            else read_csv(path)
        )
        for name, path in input_paths.items()
    }
    dataset_manifests = {
        dataset: inputs[f"dataset_manifest_{dataset}"]
        for dataset in DATASET_ORDER
    }
    core_rendered = {
        "local_homophily_groups.tex": render_local_homophily_groups(
            dataset_manifests
        ),
        "training_gpu_hours.tex": render_gpu_hours(inputs["gpu_hours"]),
        "rq1_macro_f1_tail_gap.tex": render_rq1_macro_f1(
            inputs["rq1_macro_f1"]
        ),
        "rq1_accuracy_tail_gap.tex": render_rq1_accuracy(
            inputs["rq1_accuracy"]
        ),
        "rq2_within_class_accuracy.tex": render_rq2_within_class(
            inputs["rq2_within_class"]
        ),
        "rq2_adjusted_accuracy.tex": render_rq2_adjusted_accuracy(
            inputs["rq2_adjusted"]
        ),
        "rq2_adjusted_macro_f1.tex": render_rq2_adjusted_macro_f1(
            inputs["rq2_adjusted"]
        ),
    }
    if DATASET_MODE is AnalysisDatasetMode.ADDITIONAL:
        rendered = {
            **core_rendered,
            "rq2_last_recovered_share.tex": render_rq2_recovered_share(
                inputs["recovered_shares"],
                inputs["rq5_gains"],
                preparation="last",
            ),
            "rq2_head_recovered_share.tex": render_rq2_recovered_share(
                inputs["recovered_shares"],
                inputs["rq5_gains"],
                preparation="head",
            ),
            "rq4_designed_vs_standard.tex": render_rq4_pairwise(
                inputs["rq4_pairwise"]
            ),
            "rq5_last_tail_gain.tex": render_rq5_gain(
                inputs["rq5_gains"],
                preparation="last",
            ),
            "rq5_head_tail_gain.tex": render_rq5_gain(
                inputs["rq5_gains"],
                preparation="head",
            ),
            "rq5_full_tail_gain.tex": render_rq5_gain(
                inputs["rq5_gains"],
                preparation="full",
            ),
        }
        expected_outputs = ADDITIONAL_OUTPUT_FILENAMES
    else:
        rendered = {
            **core_rendered,
            "rq4_designed_vs_standard.tex": render_rq4_pairwise(
                inputs["rq4_pairwise"]
            ),
        "rq5_last_tail_gain.tex": render_rq5_gain(
            inputs["rq5_gains"],
            preparation="last",
        ),
        "rq5_head_tail_gain.tex": render_rq5_gain(
            inputs["rq5_gains"],
            preparation="head",
        ),
        "rq5_full_tail_gain.tex": render_rq5_gain(
            inputs["rq5_gains"],
            preparation="full",
        ),
        "rq5_dfr_tail_macro_f1.tex": render_rq5_dfr_tail_macro_f1(
            inputs["rq5_dfr_absolute"]
        ),
        "fg_ccdb_pubmed.tex": render_fg_ccdb_pubmed(
            inputs["fg_ccdb"],
            inputs["rq5_gains"],
        ),
        }
        expected_outputs = OUTPUT_FILENAMES
    if tuple(rendered) != expected_outputs:
        raise AssertionError("Rendered table order does not match contract")
    return rendered, input_paths


def build_manifest(
    rendered: Mapping[str, str],
    input_paths: Mapping[str, Path],
    output_dir: Path,
    dataset_mode: AnalysisDatasetMode | str = AnalysisDatasetMode.ORIGINAL,
) -> str:
    mode = AnalysisDatasetMode(dataset_mode)
    payload = {
        "schema_version": 1,
        "analysis": "released-result-and-compute-table-rendering",
        "generator": "scripts/render_paper_tables.py",
        "generator_sha256": sha256_text_file(Path(__file__)),
        "inputs": {
            name: {
                "path": relative_to_repo(path),
                "sha256": sha256_text_file(path),
            }
            for name, path in input_paths.items()
        },
        "outputs": {
            filename: {
                "path": relative_to_repo(output_dir / filename),
                "sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
            }
            for filename, content in rendered.items()
        },
    }
    if mode is not AnalysisDatasetMode.ORIGINAL:
        payload["dataset_mode"] = mode.value
        payload["datasets"] = list(dataset_order(mode))
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def check_outputs(
    rendered: Mapping[str, str],
    manifest: str,
    output_dir: Path,
) -> None:
    expected = dict(rendered)
    expected["analysis_manifest.json"] = manifest
    mismatches = []
    for filename, content in expected.items():
        path = output_dir / filename
        if not path.is_file():
            mismatches.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != content:
            mismatches.append(f"stale {path}")
    if mismatches:
        raise RuntimeError(
            "Generated quantitative tables are not current:\n"
            + "\n".join(f"- {item}" for item in mismatches)
        )


def write_outputs(
    rendered: Mapping[str, str],
    manifest: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in rendered.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    (output_dir / "analysis_manifest.json").write_text(
        manifest,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_dataset_mode(args.dataset_mode)
    rendered, input_paths = render_all(
        args.results_root,
        args.fg_results_root,
        args.dataset_manifest_root,
        dataset_mode=args.dataset_mode,
    )
    manifest = build_manifest(
        rendered,
        input_paths,
        args.output_dir,
        dataset_mode=args.dataset_mode,
    )
    if args.check:
        check_outputs(rendered, manifest, args.output_dir)
        print(f"Verified {len(rendered)} generated quantitative tables.")
    else:
        write_outputs(rendered, manifest, args.output_dir)
        print(f"Wrote {len(rendered)} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
