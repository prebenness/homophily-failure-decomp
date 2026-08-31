# Corrected main-run metrics

`runs/` contains one corrected macro-F1 record per run. Each score uses the
classes present in that evaluated subset's true labels; the class set is fixed
before predictions are evaluated. `aggregates/` contains prepared scores,
paired gains, ERM tail gaps, and recovered shares. Macro-F1 is recalculated
from `run_data.npz`; stored `metrics.json` values are used only to audit the
correction.

Regenerate from the raw archive with:

```powershell
python scripts/recalculate_macro_f1.py `
  --raw-root PATH-TO-CANONICAL-RUNS
```

The current snapshot contains all 4,200 registered runs across the five datasets.
The raw store's `historical-records/` directory contains superseded files and is
explicitly excluded from run discovery and analysis.
Binary predictions, histories, and checkpoints are not stored in Git.
