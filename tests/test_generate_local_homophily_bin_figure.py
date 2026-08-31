from copy import deepcopy
from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_local_homophily_bin_figure.py"
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_generated_figure_matches_checked_in_figure_exactly(tmp_path: Path) -> None:
    output = tmp_path / "local_homophily_bin.tex"

    NAMESPACE["generate_figure"](
        REPO_ROOT / "manifests" / "datasets",
        output,
    )

    expected = (
        REPO_ROOT / "paper" / "figures" / "local_homophily_bin.tex"
    ).read_bytes()
    assert output.read_bytes() == expected


def test_rendered_values_are_derived_from_manifests() -> None:
    manifests = NAMESPACE["load_manifests"](
        REPO_ROOT / "manifests" / "datasets"
    )
    changed = deepcopy(manifests)
    changed[0]["local_homophily"]["mean_h"] = 0.1234
    changed[0]["local_homophily"]["merged_bins"][0][
        "fraction_of_eligible"
    ] = 0.2222

    rendered = NAMESPACE["render_figure"](changed)

    assert "    (0,22.2)" in rendered
    assert "(axis cs:0.123,0) -- (axis cs:0.123,100);" in rendered
    assert "at (axis cs:0.123,98) {0.123};" in rendered


def test_histogram_axes_use_compact_paper_height() -> None:
    manifests = NAMESPACE["load_manifests"](
        REPO_ROOT / "manifests" / "datasets"
    )

    rendered = NAMESPACE["render_figure"](manifests)

    assert "    height=1.92cm," in rendered


def test_additional_mode_renders_only_actor_and_blogcatalog() -> None:
    datasets = NAMESPACE["ADDITIONAL_DATASET_ORDER"]
    manifests = NAMESPACE["load_manifests"](
        REPO_ROOT / "manifests" / "additional-datasets",
        datasets,
    )

    rendered = NAMESPACE["render_figure"](manifests, datasets)

    assert "group size=2 by 1," in rendered
    assert "title={Actor}" in rendered
    assert "title={BlogCatalog}" in rendered
    assert "title={PubMed}" not in rendered
    assert (
        r"    ylabel={Nodes (\%)}," + "\n"
        r"    xlabel={Local homophily \(h_v\)},"
    ) in rendered
    assert r"\label{fig:additional-local-homophily-bins-tikz}" in rendered
    assert rendered.startswith(r"\begin{figure}[H]")
    assert (
        "Distribution of nodes across local-homophily bins. Bar\n"
        "widths show the bin intervals and heights show within-dataset "
        "percentages."
    ) in rendered
