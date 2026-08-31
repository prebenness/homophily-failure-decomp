# PubMed DFR frozen-representation sensitivity

This post-registration sensitivity tests whether the RQ5 result depends on
the particular optimisation used for `A_last`. It applies a DFR-style
classifier refit to the same frozen ERM representation.

## Primary panel

- Dataset: PubMed.
- Models: ACM-GCN, GCN, and GraphSAGE-mean.
- Repetitions: ten Geom-GCN splits by three ERM training seeds.
- Refit data: the canonical validation split.
- Representation: `model.head.classifier_features(model.encoder(...))`, which
  is the input to the ERM model's final affine layer and therefore matches the
  `A_last` cut.
- Groups: the Cartesian product of class and fixed local-homophily bin.
- Classifier: L1-regularised logistic regression on standardised frozen
  features.
- Hyperparameter selection: the validation split is divided into two
  deterministic joint-stratified halves. The first half fits the classifier;
  the second selects `C` by worst joint-cell accuracy.
- Final refit: ten equal-cell subsamples of the full validation split are
  fitted at the selected `C`; their coefficients and intercepts are averaged.

The source DFR experiments include binary tasks. PubMed has three classes, so
this implementation uses multinomial logistic regression with the `saga`
solver and L1 regularisation.

## Reproduction

Run from a clean committed checkout. The runner validates existing outputs and
skips them, so the same command resumes an interrupted panel.

```powershell
python scripts/run_dfr_pubmed.py `
  --data-root C:\path\to\homophily-failure-decomp\artifacts\data `
  --array-root C:\path\to\homophily-failure-decomp\artifacts\manifests `
  --parent-run-root D:\path\to\canonical-runs `
  --output-root D:\path\to\dfr-pubmed-runs `
  --modes val
```

The runner uses one process, limits numerical libraries to one CPU thread, and
sets a per-process CUDA memory fraction. Each completed run is first written to
a temporary directory, validated, and then promoted atomically.

Analyse the complete panel and regenerate the tracked summaries with:

```powershell
python scripts/analyze_dfr_pubmed.py `
  --dfr-raw-root D:\path\to\dfr-pubmed-runs `
  --canonical-raw-root D:\path\to\canonical-runs `
  --output-root results\main\rq5\dfr_pubmed_sensitivity `
  --models acm-gcn gcn graphsage-mean `
  --modes val
```

The analysis validates and pairs all 90 DFR runs with the corresponding
`A_erm`, `A_last`, `A_head`, and `A_full` runs. It recomputes the metrics from
saved predictions and uses 10,000 percentile-bootstrap resamples of the 30
complete split--seed runs. `absolute_score_summary.csv` is the source for the
appendix table. `paired_contrasts.csv` and `recovered_shares.csv` contain the
supporting comparisons.

Render the LaTeX table from the tracked summary with:

```powershell
python scripts/render_paper_tables.py
```

The result is
`paper/generated/tables/rq5_dfr_tail_macro_f1.tex`. The renderer requires the
complete set of three models, five preparations, and their tail macro-F1
means and confidence intervals.

A restricted diagnostic can select models, splits, and seeds with `--models`,
`--splits`, and `--seeds`; it must also use `--allow-partial`. Such an output
does not satisfy the paper-table contract.

The raw DFR directory remains separate from the canonical run archive and is
not stored in Git. The runner reconstructs it from the fixed data arrays and
the canonical ERM checkpoints.
