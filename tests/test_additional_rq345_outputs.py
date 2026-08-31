from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


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


@pytest.mark.parametrize(
    ("name", "relative_path", "extra_args", "expected_paths"),
    [
        (
            "additional_rq3",
            "scripts/plot_rq3_mlp_contrasts.py",
            (),
            {
                "run_root": REPO_ROOT / "results" / "additional" / "runs",
                "analysis_dir": REPO_ROOT / "results" / "additional" / "rq3",
                "figure_dir": (
                    REPO_ROOT / "results" / "additional" / "figures" / "rq3"
                ),
            },
        ),
        (
            "additional_rq4",
            "scripts/analyze_rq4_model_families.py",
            (),
            {
                "run_root": REPO_ROOT / "results" / "additional" / "runs",
                "analysis_dir": REPO_ROOT / "results" / "additional" / "rq4",
                "figure_dir": (
                    REPO_ROOT / "results" / "additional" / "figures" / "rq4"
                ),
            },
        ),
        (
            "additional_rq5",
            "scripts/analyze_rq5_preparations.py",
            (),
            {
                "run_root": REPO_ROOT / "results" / "additional" / "runs",
                "analysis_dir": REPO_ROOT / "results" / "additional" / "rq5",
                "figure_dir": (
                    REPO_ROOT / "results" / "additional" / "figures" / "rq5"
                ),
            },
        ),
        (
            "additional_compact",
            "scripts/plot_compact_main_figures.py",
            (),
            {
                "rq3_components": (
                    REPO_ROOT
                    / "results"
                    / "additional"
                    / "rq3"
                    / "mlp_tail_gap_components.csv"
                ),
                "output_dir": (
                    REPO_ROOT
                    / "results"
                    / "additional"
                    / "figures"
                    / "rq3-rq4"
                ),
            },
        ),
        (
            "additional_tail_non_tail",
            "scripts/plot_rq5_tail_vs_non_tail.py",
            ("--raw-root", str(REPO_ROOT)),
            {
                "analysis_dir": REPO_ROOT / "results" / "additional" / "rq5",
                "figure_dir": (
                    REPO_ROOT / "results" / "additional" / "figures" / "rq5"
                ),
            },
        ),
    ],
)
def test_additional_mode_defaults_stay_under_additional_results(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    relative_path: str,
    extra_args: tuple[str, ...],
    expected_paths: dict[str, Path],
) -> None:
    module = load_script(name, relative_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [relative_path, "--dataset-mode", "additional", *extra_args],
    )
    args = module.parse_args()

    assert args.dataset_mode.value == "additional"
    for attribute, expected in expected_paths.items():
        assert getattr(args, attribute) == expected


def test_rq5_additional_mode_rejects_substituted_registered_run_id() -> None:
    module = load_script(
        "additional_tail_non_tail_registry",
        "scripts/plot_rq5_tail_vs_non_tail.py",
    )
    module.configure_dataset_mode("additional")
    registrations = module.load_registered_runs(
        REPO_ROOT, "additional", module.DATASET_ORDER
    )
    rows = [
        {"run_id": run.run_id}
        for run in registrations
        if run.model in module.MODEL_ORDER
        and run.preparation in module.PREPARATION_ORDER
    ]

    module.validate_registered_row_ids(rows, "additional")
    rows[-1] = {"run_id": "blogcatalog_linkx_full_gaug-public_seed-999"}
    with pytest.raises(RuntimeError, match="registered run ID mismatch"):
        module.validate_registered_row_ids(rows, "additional")


def test_rq5_additional_panel_leaves_headroom_at_a_perfect_score() -> None:
    module = load_script(
        "additional_tail_non_tail_axis",
        "scripts/plot_rq5_tail_vs_non_tail.py",
    )

    _, upper, ticks = module.y_axis_spec(
        [0.50, 1.0],
        upper_cap=1.06,
    )

    assert upper == 1.06
    assert ticks[-1] == pytest.approx(1.0)


def test_rq3_additional_panel_has_two_rows_without_family_spacer() -> None:
    module = load_script(
        "additional_rq3_panel", "scripts/plot_rq3_mlp_contrasts.py"
    )
    module.configure_dataset_mode("additional")

    assert module.DATASET_ORDER == ("actor", "blogcatalog")
    assert module.panel_grid_spec() == ((1.0, 1.0), (0, 1))


def test_rq4_additional_table_has_only_actor_and_blogcatalog() -> None:
    module = load_script(
        "additional_rq4_table", "scripts/analyze_rq4_model_families.py"
    )
    module.configure_dataset_mode("additional")
    summaries = [
        module.PairwiseContrastSummary(
            dataset=dataset,
            designed_model=designed,
            standard_model=standard,
            n_pairs=30,
            mean_designed_tail_gap=0.1,
            mean_standard_tail_gap=0.2,
            mean_contrast=-0.1,
            sample_std_contrast=0.01,
            ci_lower=-0.12,
            ci_upper=-0.08,
            smaller_tail_count=1_000,
            p_value=0.0,
            run_contrasts=tuple([-0.1] * 30),
        )
        for dataset in module.DATASET_ORDER
        for designed in module.DESIGNED_MODELS
        for standard in module.STANDARD_MODELS
    ]

    rendered = module.render_appendix_table(summaries)

    assert "Actor & BlogCatalog" in rendered
    assert "PubMed" not in rendered
    assert r"\label{tab:additional-rq4-designed-vs-standard}" in rendered
    assert r"\begin{tabular}{@{}llcc@{}}" in rendered
    assert (
        r"\caption{\textbf{RQ1c}. Macro-F1 tail-gap comparison"
        in rendered
    )
    assert r"\scriptsize" not in rendered
    assert r"\setlength{\tabcolsep}{5pt}" in rendered
    assert r"\gaincell{-.100}{[-.120,-.080]}" in rendered


def test_additional_recovered_share_table_uses_registered_grey_rule() -> None:
    module = load_script(
        "additional_recovered_share", "scripts/render_paper_tables.py"
    )
    module.configure_dataset_mode("additional")
    graph_models = tuple(
        model for model in module.MODEL_ORDER if model != "mlp"
    )
    recovered_rows = []
    gain_rows = []
    for dataset in module.DATASET_ORDER:
        for model in graph_models:
            recovered_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "preparation": "last",
                    "mean_paired_gain": "0.05",
                    "full_mean_paired_gain": "0.10",
                    "qualifies": "True",
                }
            )
            gain_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "preparation": "last",
                    "ci_excludes_zero": (
                        "False" if (dataset, model) == ("actor", "gcn") else "True"
                    ),
                }
            )

    rendered = module.render_rq2_recovered_share(
        recovered_rows, gain_rows, preparation="last"
    )

    assert "Model & Actor & BlogCatalog" in rendered
    assert r"\label{tab:additional-rho-last-recovered-share}" in rendered
    assert r"GCN & \cellcolor{black!10}$0.500$ & $0.500$" in rendered


def test_additional_tables_follow_current_paper_style() -> None:
    module = load_script(
        "additional_paper_tables", "scripts/render_paper_tables.py"
    )
    module.configure_dataset_mode("additional")

    assert module.format_signed(0.036) == "+.036"
    assert module.format_signed(-0.011) == "-.011"
    assert module.format_endpoint(0.022) == ".022"
    assert module.format_endpoint(0.05, explicit_positive=True) == "+.050"

    matrix_rows = [
        {
            "dataset": dataset,
            "model": model,
            "preparation": "erm",
            "mean_class_macro_accuracy_contrast": "0.036",
            "bootstrap_ci_lower_class_macro_accuracy": "0.022",
            "bootstrap_ci_upper_class_macro_accuracy": "0.050",
            "class_macro_accuracy_ci_excludes_zero": "True",
            "mean_adjusted_accuracy_gap": "0.036",
            "bootstrap_ci_lower_adjusted_accuracy_gap": "0.022",
            "bootstrap_ci_upper_adjusted_accuracy_gap": "0.050",
            "adjusted_accuracy_gap_ci_excludes_zero": "True",
            "erm_accuracy_gap_ci_excludes_zero": "True",
            "mean_adjusted_macro_f1_gap": "0.036",
            "bootstrap_ci_lower_adjusted_macro_f1_gap": "0.022",
            "bootstrap_ci_upper_adjusted_macro_f1_gap": "0.050",
            "adjusted_macro_f1_gap_ci_excludes_zero": "True",
            "erm_tail_gap_ci_excludes_zero": "True",
        }
        for dataset in module.DATASET_ORDER
        for model in module.MODEL_ORDER
    ]
    within_class = module.render_rq2_within_class(matrix_rows)
    adjusted_accuracy = module.render_rq2_adjusted_accuracy(matrix_rows)
    adjusted_macro_f1 = module.render_rq2_adjusted_macro_f1(matrix_rows)

    assert (
        "Per-class non-tail accuracy minus tail accuracy, averaged equally "
        "over classes."
    ) in within_class
    assert (
        "Class-adjusted accuracy tail gaps for all models on Actor and "
        "BlogCatalog."
    ) in adjusted_accuracy
    assert (
        "Class-adjusted macro-F1 tail gaps for all models on Actor and "
        "BlogCatalog."
    ) in adjusted_macro_f1
    assert r"\gaincell{+.036}{[.022,.050]}" in within_class
    for rendered in (within_class, adjusted_accuracy, adjusted_macro_f1):
        assert r"\scriptsize" not in rendered
        assert r"\small" not in rendered
        assert r"\footnotesize" not in rendered
        assert (
            r"\begin{tabular}{@{\extracolsep{\fill}}lcc@{}}"
            in rendered
        )

    gpu_rows = [
        {
            "preparation": preparation,
            "run_count": "30",
            "gpu_hours": "1.25",
        }
        for preparation in ("erm", "last", "head", "full", "total")
    ]
    gpu_table = module.render_gpu_hours(gpu_rows)
    assert "Total GPU-hours by training procedure." in gpu_table
    assert "Procedure & Runs & GPU-hours" in gpu_table
    assert r"\small" not in gpu_table

    rq4_rows = [
        {
            "dataset": dataset,
            "designed_model": designed,
            "standard_model": standard,
            "mean_tail_gap_contrast": "-0.036",
            "bootstrap_ci_lower": "-0.050",
            "bootstrap_ci_upper": "-0.022",
            "ci_excludes_zero": "True",
        }
        for dataset in module.DATASET_ORDER
        for designed in module.DESIGNED_MODELS
        for standard in module.STANDARD_MODELS
    ]
    rq4_table = module.render_rq4_pairwise(rq4_rows)
    assert (
        r"\caption{\textbf{RQ1c}. Macro-F1 tail-gap comparison"
        in rq4_table
    )
    assert r"\setlength{\tabcolsep}{5pt}" in rq4_table
    assert r"\scriptsize" not in rq4_table

    rq5_rows = [
        {
            "dataset": dataset,
            "model": model,
            "preparation": "last",
            "mean_paired_gain": "0.036",
            "bootstrap_ci_lower": "0.022",
            "bootstrap_ci_upper": "0.050",
            "ci_excludes_zero": "True",
        }
        for dataset in module.DATASET_ORDER
        for model in module.MODEL_ORDER
    ]
    rq5_table = module.render_rq5_gain(rq5_rows, preparation="last")
    assert (
        r"\textbf{RQ2a}. Change in tail macro-F1 after last layer retraining"
        in rq5_table
    )
