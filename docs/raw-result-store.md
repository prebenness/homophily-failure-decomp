# Canonical raw-result store

The five-dataset experiment panel is registered in `manifests/main-runs.jsonl`.
The Actor/BlogCatalog extension is registered separately in
`manifests/additional-dataset-runs.jsonl`. Each row names one atomic run. A
complete store contains one directory for every registered `run_id` in its
panel and no additional active run directories.

The raw stores are intentionally outside Git because they contain checkpoints
and node-level arrays for 4,200 original runs and 1,680 additional-dataset
runs. They can be created by executing the registered panels, or supplied as
a data deposit. The tracked files under `results/main/` and
`results/additional/` are derived summaries. They do not replace the raw
stores when an analysis needs logits, predictions, histories or checkpoints.

## Directory contract

The active layout is:

```text
<raw-root>/
  README.md
  <run_id>/
    config.json
    metrics.json
    history.json
    checkpoint.pt
    run_data.npz
  historical-records/
```

`historical-records/` contains superseded derived files retained for the audit
trail of the macro-F1 correction. It is excluded from every current analysis.
The machine-readable form of this contract is
`manifests/raw_store_contract.json`.

`config.json` records the exact dataset, model, preparation, split and seed,
the parent ERM run used by `last` and `head`, the Git revision, package
versions, and the registered fixed-input checksums. `run_data.npz` contains
the arrays from which the reported metrics and figures can be regenerated.
Fresh recovery runs also record the SHA256 checksum of the parent ERM
checkpoint. The historical archive predates that field. Its Roman-empire
provenance limitation is stated in `docs/reference-environment.md`.

## Fixed-input authority

`manifests/datasets/<dataset>.json` is the authority for the original panel;
`manifests/additional-datasets/<dataset>.json` has the same role for Actor and
BlogCatalog. Each manifest records:

- the graph dimensions and number of classes;
- the split identifiers and training seeds;
- the SHA256 checksum of labels, each train/validation/test split, local
  homophily groups and the tail mask.

The corresponding regenerable arrays live under
`artifacts/manifests/<dataset>.npz` for the original panel and
`artifacts/additional-dataset-manifests/<dataset>.npz` for the extension. The
validator compares each raw run with the tracked dataset manifest. This
detects a run produced with a different split, label vector, group definition
or tail definition.

## Verification

Metadata verification is fast and should be run before any paper analysis:

```powershell
python scripts/verify_raw_store.py --raw-root D:\path\to\raw-store
```

It checks the exact run panel, directory names, required files, resolved
model and training configuration, parent-run relationships, parent
checkpoint hashes when recorded, checkpoint archive structure, Git revision
format, reference version provenance, fixed-input checksums and the structure
of metrics and histories. Historical runs made with an earlier recorded
environment remain valid inputs and are reported as a separate environment
stratum. Add
`--require-reference-environment` when verifying a newly generated panel that
is expected to use the current lock throughout.

Array verification reads every `run_data.npz`:

```powershell
python scripts/verify_raw_store.py `
  --raw-root D:\path\to\raw-store `
  --level arrays `
  --report results/main/raw_store_verification.json
```

It additionally checks registered array checksums, shapes, index ranges,
finite logits, agreement between stored predictions and the argmax of the
stored logits, and agreement between the node arrays and the stored evaluation
metrics. A zero exit status means that every registered run passed.

The array check is deliberately separate because reading all compressed
node-level arrays is much slower than checking metadata. Paper reproduction
should use the array level at least once for each acquired or newly generated
raw store.

Validate the Actor/BlogCatalog store against its separate panel and manifests:

```powershell
python scripts/verify_raw_store.py `
  --raw-root artifacts/additional-dataset-runs `
  --panel manifests/additional-dataset-runs.jsonl `
  --dataset-manifest-dir manifests/additional-datasets `
  --model-config-dir configs/models `
  --level arrays `
  --report results/additional/raw_store_verification.json
```

## Recreating the store

Create and inspect the panel before using compute resources:

```powershell
python scripts/build_experiment_grid.py
python scripts/run_experiment_grid.py run --dry-run
python scripts/run_additional_datasets.py prepare
python scripts/run_additional_datasets.py run --dry-run
```

Run the complete panel with a suitable scheduler or the supplied resumable
runner. ERM runs precede dependent `last` and `head` runs. Successful run
directories can be copied between machines because all paths recorded in the
contract are relative to the chosen raw root.

Fresh training uses stochastic GPU operations. The registered seeds,
environment lock, code revision and fixed arrays constrain the computation,
but different GPU models or CUDA libraries can still produce small numerical
differences. The deposited raw store and the tracked paper artefacts are the
exact numerical record used for the manuscript.

## Appendix J inference subset

The edge-perturbation figures use the 3,600 graph-model runs from the original
panel: 900 runs for each of `erm`, `last`, `head`, and `full`. This subset is a
valid checkpoint source for the four Appendix J panels, but it is not a complete
replacement for the 4,200-run original store because it omits the 600 MLP runs.
Keep the subset outside Git together with the authenticated perturbed graphs;
the compact summaries, provenance manifests, and rendered figures are the
tracked release.
