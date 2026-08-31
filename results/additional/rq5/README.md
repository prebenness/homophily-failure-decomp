# RQ5 absolute preparation scores

`preparation_tail_macro_f1.csv` reports mean tail macro-F1 and its 95%
percentile-bootstrap interval for every dataset, model, and preparation.
Each interval resamples the 30 complete split--seed runs.

`preparation_tail_vs_whole_macro_f1.csv` reports the tail and whole-test
macro-F1 means plotted in the comparison figures. It contains the six graph
models and omits MLP.

Regenerate both sets of PDF/PNG figures and their tables with:

```powershell
python scripts/analyze_rq5_preparations.py --dataset-mode additional
```

The tail-only figures show one point for `A_erm`, `A_last`, `A_head`, and
`A_full` at each graph model. Preparation points share an x-coordinate within
a model. The dashed vertical segment joins the lowest and highest preparation
means.

The comparison figures place the four preparations from left to right within
each model and connect them with dashed visual guides. Blue filled circles
show tail macro-F1; rust open circles show whole-test macro-F1. Changes in
the orange series relative to `A_erm` show whole-test performance gained or
sacrificed. The vertical separation between the two series is not itself a
measure of sacrificed performance.

`rq5_tail_vs_whole_macro_f1_compact` combines Actor and BlogCatalog with the key in a single row. It uses short model names, solid lines,
blue circles for tail macro-F1, and rust squares for whole-test macro-F1. Each
dataset panel uses its own y-axis range.

MLP remains in the tail summary table but is omitted from all figures because
it is not a graph model.

The two separate dataset figures prefer TeX Gyre Termes, with Times New
Roman, STIXGeneral, and DejaVu Serif as fallbacks. The compact main-text figure
uses Arial, STIX Sans mathematics, and the seaborn colourblind blue and orange
used by Figure 3. It is generated at its final 5.5-by-1.65-inch paper size.
