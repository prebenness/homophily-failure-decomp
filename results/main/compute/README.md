# Main-panel training compute

These files summarise GPU use for the 4,200 confirmatory main-panel runs.

Regenerate them from the raw run archive with:

```text
python scripts/analyze_training_compute.py \
    --raw-dir /path/to/raw-runs \
    --output-dir results/main/compute
```

GPU-hours are calculated by summing `checkpoint.train_seconds` from the
retained runs and dividing by 3,600. Every run used one CUDA GPU. The recorded
duration covers the optimisation loop and excludes data loading, preprocessing,
the one-off frozen-feature computation for the `last` and `head` preparations,
final test evaluation, and serialisation.

The hardware labels contain GPU model and memory only. They contain no host,
facility, account, queue, or filesystem identifiers.

Outputs:

- `gpu_hours_by_preparation.csv`: paper-table totals by preparation.
- `hardware_inventory.csv`: the two GPU models and their memory capacities.
- `analysis_manifest.json`: definitions and validation counts.

The final raw records do not retain a reliable run-to-GPU-model assignment.
The GPU-hour totals are therefore not partitioned by GPU model.
