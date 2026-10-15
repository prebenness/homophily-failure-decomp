import json
from pathlib import Path
import runpy

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "render_paper_tables.py"
NAMESPACE = runpy.run_path(str(SCRIPT))

def test_all_quantitative_tables_match_tracked_release_artifacts() -> None:
    rendered, input_paths = NAMESPACE["render_all"](
        REPO_ROOT / "results" / "main"
    )

    assert tuple(rendered) == NAMESPACE["OUTPUT_FILENAMES"]
    assert set(input_paths) == (
        set(NAMESPACE["INPUT_PATHS"])
        | set(NAMESPACE["FG_INPUT_PATHS"])
        | set(NAMESPACE["DATASET_MANIFEST_PATHS"])
    )
    for filename, content in rendered.items():
        tracked_path = (
            REPO_ROOT / "paper" / "generated" / "tables" / filename
        )
        assert content == tracked_path.read_text(encoding="utf-8")


def test_additional_tables_match_tracked_release_artifacts() -> None:
    rendered, _ = NAMESPACE["render_all"](
        REPO_ROOT / "results" / "additional",
        dataset_manifest_root=(
            REPO_ROOT / "manifests" / "additional-datasets"
        ),
        dataset_mode="additional",
    )

    release_dir = REPO_ROOT / "paper" / "generated" / "additional-datasets"
    released = {
        path.name for path in release_dir.glob("*.tex")
    }
    assert released == set(NAMESPACE["ADDITIONAL_OUTPUT_FILENAMES"]) - {
        "training_gpu_hours.tex"
    }
    for filename in released:
        assert rendered[filename] == (release_dir / filename).read_text(
            encoding="utf-8"
        )


def test_additional_matrix_tables_use_the_accepted_paper_width() -> None:
    NAMESPACE["configure_dataset_mode"]("additional")
    rows = NAMESPACE["read_csv"](
        REPO_ROOT / "results" / "additional" / "aggregates" / "erm_tail_gaps.csv"
    )

    rendered = NAMESPACE["render_rq1_macro_f1"](rows)

    assert r"\begin{tabular}{@{\extracolsep{\fill}}lcc@{}}" in rendered


def test_additional_recovered_shares_keep_uncertain_numerators() -> None:
    NAMESPACE["configure_dataset_mode"]("additional")
    rows = NAMESPACE["read_csv"](
        REPO_ROOT / "results/additional/aggregates/recovered_shares.csv"
    )
    try:
        last = NAMESPACE["render_rq2_recovered_share"](rows, preparation="last")
        head = NAMESPACE["render_rq2_recovered_share"](rows, preparation="head")
        assert r"H2GCN-2 & $0.155$ & \textemdash" in last
        assert r"GCN & $0.140$ & \textemdash" in head
        for rendered in (last, head):
            assert rendered.count(r"\textemdash") == 5
            assert r"\cellcolor" not in rendered
    finally:
        NAMESPACE["configure_dataset_mode"]("original")


@pytest.mark.parametrize(
    "gain,lower,upper,expected",
    [(0.03, 0.01, 0.05, "$0.000$"),
     (0.029, 0.01, 0.05, r"\textemdash"),
     (0.1, 0.0, 0.2, r"\textemdash"),
     (0.0, -0.01, 0.01, r"\textemdash")],
)
def test_additional_recovered_share_denominator_controls_eligibility(
    gain, lower, upper, expected
) -> None:
    NAMESPACE["configure_dataset_mode"]("additional")
    rows = NAMESPACE["read_csv"](
        REPO_ROOT / "results/additional/aggregates/recovered_shares.csv"
    )
    for row in rows:
        if row["dataset"] == "actor" and row["model"] == "gcn":
            row.update(
                mean_paired_gain="0",
                full_mean_paired_gain=str(gain),
                full_bootstrap_ci_lower=str(lower),
                full_bootstrap_ci_upper=str(upper),
            )
    try:
        rendered = NAMESPACE["render_rq2_recovered_share"](
            rows, preparation="head"
        )
        assert f"GCN & {expected} & " + r"\textemdash" in rendered
    finally:
        NAMESPACE["configure_dataset_mode"]("original")


def test_manifest_records_every_input_and_output() -> None:
    rendered, input_paths = NAMESPACE["render_all"](
        REPO_ROOT / "results" / "main"
    )
    manifest = NAMESPACE["build_manifest"](
        rendered,
        input_paths,
        REPO_ROOT / "paper" / "generated" / "tables",
    )
    payload = json.loads(manifest)

    assert set(payload["inputs"]) == (
        set(NAMESPACE["INPUT_PATHS"])
        | set(NAMESPACE["FG_INPUT_PATHS"])
        | set(NAMESPACE["DATASET_MANIFEST_PATHS"])
    )
    assert set(payload["outputs"]) == set(NAMESPACE["OUTPUT_FILENAMES"])
    assert all(
        len(record["sha256"]) == 64
        for record in payload["inputs"].values()
    )
    assert all(
        len(record["sha256"]) == 64
        for record in payload["outputs"].values()
    )


def test_renderer_rejects_an_incomplete_matrix() -> None:
    rows = NAMESPACE["read_csv"](
        REPO_ROOT
        / "results"
        / "main"
        / "aggregates"
        / "erm_tail_gaps.csv"
    )

    with pytest.raises(ValueError, match="missing="):
        NAMESPACE["render_rq1_macro_f1"](rows[:-1])


def test_rq5_dfr_table_reports_absolute_tail_macro_f1() -> None:
    rows = NAMESPACE["read_csv"](
        REPO_ROOT
        / "results"
        / "main"
        / "rq5"
        / "dfr_pubmed_sensitivity"
        / "absolute_score_summary.csv"
    )

    rendered = NAMESPACE["render_rq5_dfr_tail_macro_f1"](rows)

    assert (
        r"Model & \(\Aerm\) & \(\Alast\) & \(\Ahead\) & \(\Afull\) "
        r"& \(A_{\mathrm{DFR}}\) \\"
    ) in rendered
    expected_rows = (
        r"ACM-GCN & \shortstack{$0.557$\\[-1pt]"
        r"{\tiny $[0.550,0.564]$}} & "
        r"\shortstack{$0.562$\\[-1pt]{\tiny $[0.555,0.569]$}} & "
        r"\shortstack{$0.575$\\[-1pt]{\tiny $[0.568,0.581]$}} & "
        r"\shortstack{$0.658$\\[-1pt]{\tiny $[0.652,0.665]$}} & "
        r"\shortstack{$0.593$\\[-1pt]{\tiny $[0.586,0.600]$}}",
        r"GCN & \shortstack{$0.512$\\[-1pt]"
        r"{\tiny $[0.506,0.518]$}} & "
        r"\shortstack{$0.537$\\[-1pt]{\tiny $[0.531,0.543]$}} & "
        r"\shortstack{$0.620$\\[-1pt]{\tiny $[0.614,0.626]$}} & "
        r"\shortstack{$0.634$\\[-1pt]{\tiny $[0.628,0.639]$}} & "
        r"\shortstack{$0.551$\\[-1pt]{\tiny $[0.542,0.560]$}}",
        r"GraphSAGE-mean & \shortstack{$0.532$\\[-1pt]"
        r"{\tiny $[0.526,0.538]$}} & "
        r"\shortstack{$0.648$\\[-1pt]{\tiny $[0.635,0.660]$}} & "
        r"\shortstack{$0.688$\\[-1pt]{\tiny $[0.678,0.697]$}} & "
        r"\shortstack{$0.694$\\[-1pt]{\tiny $[0.689,0.700]$}} & "
        r"\shortstack{$0.629$\\[-1pt]{\tiny $[0.621,0.636]$}}",
    )
    positions = [rendered.index(row) for row in expected_rows]
    assert positions == sorted(positions)


def test_local_homophily_table_is_derived_from_the_manifests() -> None:
    manifests = {
        dataset: NAMESPACE["read_json"](
            REPO_ROOT / "manifests" / "datasets" / f"{dataset}.json"
        )
        for dataset in NAMESPACE["DATASET_ORDER"]
    }
    rendered = NAMESPACE["render_local_homophily_groups"](manifests)
    assert r"PubMed & $\mathbf{[0,0.25)}$ & 2{,}696 & 13.7" in rendered
    assert (
        r"Roman-empire & $[0,0.25)$ & 20{,}297 & 89.6"
        in rendered
    )
    assert r"\begin{tabular}{@{}llrr}" in rendered
    assert "Tail bin" not in rendered
    assert r"\leftarrow" not in rendered
    assert (
        r"The local homophily bins used throughout. \textbf{Bold} type "
        r"marks the rare \(h_v\) tail---the low \(h_v\) bin on homophilic "
        r"datasets and the high-\(h_v\) bin on heterophilic datasets."
        in rendered
    )


def test_fg_ccdb_table_has_the_complete_requested_panel() -> None:
    fg_rows = NAMESPACE["read_csv"](
        REPO_ROOT / "results" / "fg_ccdb" / "paired_results.csv"
    )
    groupdro_rows = NAMESPACE["read_csv"](
        REPO_ROOT
        / "results"
        / "main"
        / "aggregates"
        / "paired_gains.csv"
    )
    rendered = NAMESPACE["render_fg_ccdb_pubmed"](
        fg_rows,
        groupdro_rows,
    )
    assert rendered.count(r"\multirow{3}{*}") == 3
    assert r"\gaincell{+0.043}{[0.039,0.048]}" in rendered
    assert r"\uncertaingaincell{+0.002}{[-0.005,0.008]}" in rendered
    assert r"\label{tab:fg-ccdb-pubmed}" in rendered


def test_fg_ccdb_table_rejects_an_incomplete_panel() -> None:
    fg_rows = NAMESPACE["read_csv"](
        REPO_ROOT / "results" / "fg_ccdb" / "paired_results.csv"
    )
    groupdro_rows = NAMESPACE["read_csv"](
        REPO_ROOT
        / "results"
        / "main"
        / "aggregates"
        / "paired_gains.csv"
    )

    with pytest.raises(ValueError, match="missing="):
        NAMESPACE["render_fg_ccdb_pubmed"](
            fg_rows[:-1],
            groupdro_rows,
        )
