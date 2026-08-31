# RQ1 and RQ2 grouped tail-gap plot

`rq1_rq2_additional_datasets_unadjusted_and_class_adjusted_macro_f1_tail_gaps_unequal_y.pdf` and `rq1_rq2_additional_datasets_unadjusted_and_class_adjusted_macro_f1_tail_gaps_unequal_y.png` show the 14 ERM
dataset-model cells in 2 dataset panels using panel-specific vertical scales, all of which include zero.
Within each panel, each coloured model line joins its unadjusted RQ1
macro-F1 tail gap to its class-composition-adjusted RQ2 macro-F1 tail gap.
Endpoints carry 95% paired-bootstrap confidence intervals.

Filled markers denote intervals that exclude zero. Open markers denote
intervals that contain zero. The faint connector joins the two point estimates
only; it is not an interval for their difference.

Regenerate from the tracked summary tables with:

```powershell
python scripts/plot_rq1_rq2_paired_forest.py --dataset-mode additional
```
