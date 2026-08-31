# True-class support across equal-width local-homophily bins

This directory records whether a class-composition-adjusted version of the
accuracy-versus-\(h_v\) figure has adequate common class support.

For each registered test split, the interval \([0,1]\) is divided into 4, 6,
8, 10, or 20 equal-width bins. Bins are left-closed and right-open, except
that \(h_v=1\) belongs to the final bin. A true class is counted only when at
least one test node of that class occurs in every bin of the split.

Outputs:

- `split_class_support.csv` records the common-class count and class IDs for
  every dataset, split, and bin count. It also records the class and node
  counts in each individual bin.
- `dataset_class_support_summary.csv` gives the minimum, maximum, and complete
  split sequence of common-class counts for each dataset and bin count.
- `analysis_manifest.json` records the definitions, validated frozen inputs,
  checksums, software versions, output checksums, and complete summary.

Regenerate the outputs from the frozen manifests:

```powershell
python scripts/analyze_hv_bin_class_support.py
```

The threshold of one node establishes class presence. It does not establish
that the resulting class-specific accuracy estimate is precise.
