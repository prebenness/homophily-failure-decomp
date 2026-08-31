# RQ5 accuracy figures

`preparation_accuracy_run_values.csv` contains the recomputed tail and
whole-test accuracy for all 4,200 runs. Each value is calculated from the
saved labels and predictions and checked against the canonical raw metrics.

`preparation_tail_vs_whole_accuracy.csv` reports the means plotted in the
five figures. It contains the six graph models and omits MLP.

Regenerate the CSV files and the PDF/PNG figures with:

```powershell
python scripts/analyze_rq5_accuracy.py `
  --raw-root C:\path\to\heterophilic-graph-paper
```

The four preparations are ordered from left to right within each model:
`A_erm`, `A_last`, `A_head`, and `A_full`. Dashed blue lines with filled
circles show tail accuracy. Dashed rust lines with open circles show
whole-test accuracy. MLP is retained in the run-level calculations but
omitted from the figures because it is not a graph model.

Figure text uses the same font stack and dimensions as the corresponding RQ5
macro-F1 figures.
