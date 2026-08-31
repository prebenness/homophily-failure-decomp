from __future__ import annotations

from pathlib import Path
import runpy

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "plot_edge_dose_response_composite.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_validated_summary_forms_complete_full_matrix(tmp_path: Path) -> None:
    rows = NAMESPACE["load_summary"](NAMESPACE["DEFAULT_SUMMARY"])
    assert len(rows) == 1188

    figure = NAMESPACE["build_figure"](rows)
    assert len(figure.axes) == 30
    assert [len(axis.lines) for axis in figure.axes] == [2] * 30
    assert [axis.get_title() for axis in figure.axes[:5]] == [
        "PubMed",
        "ogbn-arxiv",
        "Roman-empire",
        "Amazon-ratings",
        "arxiv-year",
    ]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Treatment",
        "Control",
    ]
    expected_vertical_labels = [
        "GCN",
        "GraphSAGE-mean",
        "H2GCN-2",
        "GPR-GNN",
        "ACM-GCN",
        "LINKX",
    ]
    vertical_labels = [
        text.get_text()
        for text in figure.texts
        if text.get_text() in expected_vertical_labels
    ]
    assert vertical_labels == expected_vertical_labels

    roman_axes = figure.axes[2::5]
    assert len(roman_axes) == 6
    assert all(
        max(line.get_xdata()) == 70
        for axis in roman_axes
        for line in axis.lines
    )

    outputs = NAMESPACE["save_figure"](figure, tmp_path)
    plt.close(figure)
    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    assert outputs["png"].read_bytes().startswith(b"\x89PNG")


def test_release_pdf_is_exact_copy_of_generated_pdf(tmp_path: Path) -> None:
    rows = NAMESPACE["load_summary"](NAMESPACE["DEFAULT_SUMMARY"])
    figure = NAMESPACE["build_figure"](rows)
    paper_output = tmp_path / "paper" / "figure.pdf"
    outputs = NAMESPACE["save_figure"](
        figure,
        tmp_path / "generated",
        paper_output=paper_output,
    )
    plt.close(figure)

    assert outputs["pdf"].read_bytes() == paper_output.read_bytes()
