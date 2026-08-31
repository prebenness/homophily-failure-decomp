# RQ2 within-class and class-composition analyses

This directory contains the direct within-class accuracy comparison stated in
RQ2 and class-composition-adjusted versions of the two RQ1
whole-test-minus-tail gaps. The direct comparison includes classes represented
among both tail and non-tail test nodes. The class-composition adjustment
includes every class present in the tail.

Outputs:

- `within_class_accuracy_run_contrasts.csv`: one row per ERM split-seed run,
  containing the equal-class mean of non-tail accuracy minus tail accuracy;
- `within_class_accuracy_cell_summary.csv`: the 30-run means and
  paired-bootstrap intervals used in the direct RQ2 table;
- `class_adjusted_run_metrics.csv`: one row per ERM split-seed run, containing
  the whole-test and class-composition-adjusted tail scores and their gaps;
- `class_adjusted_cell_summary.csv`: the 30-run means and paired-bootstrap
  intervals for every ERM dataset-model cell;
- `appendix_tables.tex`: the direct RQ2 table followed by the two adjusted RQ1
  tables, generated from the cell summaries.

The direct RQ2 contrast is the unweighted mean across included true classes of
non-tail accuracy minus tail accuracy. A positive value means that tail
nodes have lower accuracy within class. The table contains all 14 cells; the
RQ2 condition selects cells whose unadjusted RQ1 macro-F1 gap is positive and
has an interval wholly above zero.

For the class-composition adjustment, the tail confusion matrix is reweighted
to the whole-test class proportions over the tail-present classes. The
whole-test score is calculated over the same class set. These gaps are the
whole-test score minus the class-reweighted tail score. Predictions outside
the retained class set remain errors for retained true classes.

The adjusted tables retain all cells. A dagger is based on the corresponding
unadjusted RQ1 table: `results/additional/rq1/accuracy/whole_vs_tail_accuracy.csv`
for adjusted accuracy and `results/additional/aggregates/erm_tail_gaps.csv` for
adjusted macro-F1.

Regenerate the table from the canonical raw archive:

```powershell
python scripts/analyze_rq2_within_class.py `
  --dataset-mode additional `
  --additional-raw-root PATH-TO-ADDITIONAL-RUNS
```

The raw archive is never modified. `analysis_manifest.json` records the
calculation, bootstrap definition, panel checks, combined input hash,
software versions, and output hashes.
