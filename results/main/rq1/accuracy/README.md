# RQ1 whole-test and tail accuracy

This directory contains the accuracy comparison for the 30 ERM runs in every
dataset-model cell. Accuracy is calculated separately within each run. The
reported cell values are unweighted means of the 30 split-seed run values.
Whole-test accuracy uses every node in `test_idx`. Tail accuracy uses the
intersection of `test_idx` and the tail mask registered in the corresponding
dataset manifest.

`whole_vs_tail_accuracy_wide.csv` is the model-by-dataset comparison table.
`whole_vs_tail_accuracy.csv` also reports the paired whole-minus-tail gap,
its paired-bootstrap confidence interval, sample standard deviations, and mean
test support. `appendix_table.tex` is the corresponding paper table in the
same format as the main macro-F1 gap table. `run_accuracy.csv` contains
the 1,050 run-level values and input checksums. `analysis_manifest.json`
records the calculation, aggregation, tail definitions, and validation audit.

Regenerate the files from the raw archive with:

```powershell
python scripts/analyze_rq1_accuracy.py `
  --raw-root PATH-TO-CANONICAL-RUNS
```
