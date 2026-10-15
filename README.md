# Decomposing failure on rare homophily nodes

Experiment code, registered inputs, numerical results, and figure and table
generators for the paper.

## Setup

```bash
conda env create -f environment.yml
conda activate hfd-repro
python -m pip install -e .
python -m pip install -r requirements-reverse-inference.txt
python scripts/download_datasets.py
```

The dataset command downloads all seven datasets and reconstructs the registered
splits and local-homophily groups. Run the commands below from the repository
root. Full training and graph evaluation require substantial compute; the
experiment guides describe the resource requirements.

## Reproduce from training

```bash
python scripts/train_models.py --device cuda
```

This trains the 4,200 main runs, the 1,680 Actor/BlogCatalog runs, and the PubMed
Deep Feature Reweighting (DFR) and fixed-group class-conditional balancing
(FG-CCDB) sensitivity panels. Run directories are resumable and written under
`artifacts/`.

## Reproduce from checkpoints

Place the supplied run directories in `artifacts/runs/`,
`artifacts/additional-dataset-runs/`, `artifacts/dfr-pubmed-runs/`, and
`artifacts/fg-ccdb-pubmed-runs/`, then regenerate the predictions and metrics:

```bash
python scripts/evaluate_checkpoints.py --device cuda
```

Each model run needs its original `config.json` and `checkpoint.pt`. DFR runs
need their saved `coefficients.npz` (including the fitted scaler) and their
parent ERM checkpoint. The evaluation command uses the registered dataset
inputs, does not train or refit models, and writes `run_data.npz` and
`metrics.json` in each supplied run directory. The root options shown by
`--help` support checkpoint stores in other locations. Checkpoint packages and
large raw outputs are separate inputs; they are not included in Git.

## Generate the figures and tables

After either route, run the [forward edge-perturbation
experiment](docs/edge-dose-response.md) and the [reverse edge-perturbation
experiment](docs/reverse-edge-inference.md). These guides generate the
graph pairs and evaluate the checkpoints on them. The reverse guide's complete
workflow is also available as `python scripts/run_reverse_edge_workflow.py
--device cuda`. Then run:

```bash
python scripts/analyse_results.py
```

This recomputes the analyses from the supplied run stores and graph evaluations,
including all forward and reverse panels. It writes numerical summaries under
`results/`, publication figures under `paper/figures/`, and LaTeX table fragments
under `paper/generated/tables/`. The forward analysis regenerates all 30
model-dataset combinations from the same run. Fresh training may give different
numerical results while following the same experimental method.

The workflow also writes `results/edge_perturbation_sign_agreement.txt`, with
the percentage of nonzero-dose treatment/control comparisons supporting higher
homophily for each Appendix J figure and pooled across both directions.
It also tallies one vote per subplot: a line must have non-overlapping plotted
mean-plus/minus-one-standard-deviation bands above the other line at more than
half of all nonzero doses; otherwise the subplot is undecided.

For a quick rebuild from the compact summaries already in Git:

```bash
python scripts/analyse_results.py --derived-only
```

This does not rerun inference. Figures requiring node-level arrays keep their
published copies. The included reverse compact summaries regenerate all four
reverse figures and their sign-agreement percentages.

The lower-level scripts expose filters for individual datasets, models,
preparations, splits, and seeds. See [the raw-result
layout](docs/raw-result-store.md) for run contents. Machine-specific scheduler
scripts and runtime state belong outside the repository.
