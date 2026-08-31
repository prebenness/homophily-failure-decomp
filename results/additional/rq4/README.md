# RQ4 heterophily-designed versus standard aggregation models

`pairwise_tail_gap_contrasts.csv` contains the 12 direct RQ4 comparisons:
three heterophily-designed models by two standard baselines by 2 datasets.
Runs are matched by split and seed before calculating

```text
heterophily-designed-model tail gap - standard-baseline tail gap
```

Negative values mean that the heterophily-designed model has the smaller
signed ERM tail gap.

`erm_whole_tail_macro_f1.csv` reports the separate whole-test and tail
macro-F1 means and 95% bootstrap intervals. The RQ4 figure plots the means
without confidence intervals.
`appendix_table.tex` is the generated LaTeX table used in the paper appendix.

Regenerate every RQ4 output with:

```powershell
python scripts/analyze_rq4_model_families.py --dataset-mode additional
```

The figure is written to `results/additional/figures/rq4/` as PDF and PNG.
