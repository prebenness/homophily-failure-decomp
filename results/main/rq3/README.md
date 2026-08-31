# RQ3 aggregation-model contrasts against MLP

`mlp_tail_gap_contrasts.csv` compares each aggregation model's corrected ERM
tail gap with the MLP tail gap on the same dataset. Runs are matched by split
and seed before calculating

```text
aggregation-model tail gap - MLP tail gap
```

Positive values mean that the aggregation model has a larger tail gap. The
figure shows the mean paired contrast and its 95% paired-bootstrap interval.

`class_adjusted_mlp_tail_gap_contrasts.csv` repeats the same matched comparison
using the class-composition-adjusted macro-F1 gaps in
`../rq2/class_adjusted_run_metrics.csv`. For each split and seed, both models
use the same tail-present true classes and the same whole-test class weights.
The adjusted and unadjusted figures use identical bootstrap resampling indices.

`mlp_tail_gap_components.csv` retains the unadjusted whole-test and tail means
used to generate the compact main-text comparison. Its manifest records the
matched source runs and the exact analysis script.

Regenerate all CSV files, manifests, and figures with:

```powershell
python scripts/plot_rq3_mlp_contrasts.py
```

The two contrast figures are written to `results/main/figures/rq3/` as PDF and
PNG. Each analysis has a separate provenance manifest in this directory.
