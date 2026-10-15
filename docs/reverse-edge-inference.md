# Inference on reverse-direction perturbed graphs

This extension evaluates the existing ERM, last-layer, classification-head,
and full-model checkpoints on the graph pairs described in
[reverse-edge-graphs.md](reverse-edge-graphs.md). It does not retrain models
or mix reverse-direction results with the original Appendix J results.

Each preparation includes all six graph models and all five datasets, with
30 registered training runs per combination. MLP is excluded because its
predictions do not depend on graph edges. Checkpoint identities and the
original frozen tail-test populations are unchanged.

## Complete workflow

Run graph generation and the complete workflow in Linux or another POSIX
environment. The bounded graph-generation checks use POSIX signal timers.
Install `requirements-reverse-inference.txt` along with the PyTorch setup
described in the main README before running this workflow.

After training or exporting predictions from the supplied checkpoints, the
sequential reference runner prepares the five graph panels, runs bounded
smoke checks, evaluates all four preparations, and validates their results:

```sh
python scripts/run_reverse_edge_workflow.py --dry-run
python scripts/run_reverse_edge_workflow.py --device cuda
```

Its defaults are `artifacts/runs` for checkpoints,
`artifacts/reverse-edge-graphs` for graph pairs, and
`artifacts/reverse-edge-inference` for results. The same roots are used by
the main analysis command. Use the options shown by `--help` to use other
locations. This is a substantial inference workload and should run on
suitable compute workers. The lower-level commands below support dividing
it into independent tasks; no scheduler configuration is included.

The runner checks that every planned graph dose has all five paired
replicates before starting checkpoint inference. A graph-generation timeout
or construction limit stops this workflow early. Review the recorded cutoff
and use the lower-level graph commands with a suitable time or attempt budget
to construct the full panel. Existing terminal cutoff records are retained;
use a new graph output directory when changing the construction budget.
The final analysis requires the complete planned dose range.

## Inputs and counts

Use one checkpoint directory containing `checkpoint.pt`, `config.json`,
`run_data.npz`, and `metrics.json` for each registered run. Training or the
checkpoint evaluation/export command produces these records. Training
history is not needed for these results; if `history.json` is present, its
hash is also retained. The authentication command checks
the dataset, model, preparation, split, seed, registered-input hashes, and
the required file hashes. It requires 3,600 distinct runs, including 900 for each
preparation.

The accepted reverse graph panel contains 10 PubMed pairs, 35 ogbn-arxiv
pairs, 25 arxiv-year pairs, 95 Roman-empire pairs, and 80 Amazon-ratings
pairs. Each nonzero dose has five complete treatment/control pairs.
The four preparations therefore require 3,600 baseline records and
176,400 paired result rows. A paired row contains both treatment and
control results for one checkpoint and graph pair.

The horizontal axis uses the actual fraction of the original tail selected
for rewiring, including fractional endpoints. Integer target counts identify
graphs and resume records; percentages are never rounded to identify files.

## Evaluation and numerical checks

GCN, GraphSAGE-mean, ACM-GCN, GPR-GNN, and LINKX load each checkpoint once
per evaluation invocation and evaluate it on all retained graph pairs.
H2GCN constructs each graph's operators once and shares them across the
pending checkpoints. Completed checkpoint outputs and complete H2GCN graph
pairs can be resumed after authentication.

Every invocation recomputes original-graph predictions for its pending
checkpoints. The saved logits must reproduce the authenticated saved
predictions. The new forward pass must then give exactly the same predicted
class at every fixed tail-test node. This also requires exact agreement in
tail-test accuracy, macro-F1, prediction hashes, and confusion counts. Raw
logit differences and any prediction differences outside the fixed tail-test
population are recorded, but neither is an acceptance threshold. When moving
an evaluation to a different hardware or software environment, run the
original-graph check three times in fresh processes and require all three
fixed-tail prediction vectors and metrics to agree.

Smoke outputs are separate from actual outputs. A baseline-only smoke check
can test all 30 checkpoints before inference on perturbed graphs. Source,
input, graph, preparation, and backend identities are recorded in outputs;
resume rejects incompatible identities. Choose a backend identifier that
distinguishes GPU family and numerical implementation, without including
usernames or hostnames.

By default, final validation accepts only outputs produced by the current
portable source hashes. If a resumable experiment spans more than one pinned
portable source snapshot, provide a source-hash manifest containing an
`accepted_implementation_sha256` list. Validation checks every checkpoint
output against that explicit list and records both the producer hash maps and
the manifest hash. The manifest contains repository-relative paths only.

## Portable commands

Install `requirements-reverse-inference.txt` in the experiment environment.
Construct and verify the graphs using `run_reverse_edge_graphs.py` first.
Set `GRAPH_ROOT`, `CHECKPOINT_ROOT`, and `OUTPUT_ROOT` to separate input and
output directories, `REVISION` to the complete experiment Git revision, and
`BACKEND` to the portable backend identifier. Run these commands on suitable
compute workers; scheduling and machine-specific paths remain outside the
research repository.

```sh
python scripts/authenticate_reverse_edge_inputs.py authenticate \
  --checkpoint-root "$CHECKPOINT_ROOT" --graph-root "$GRAPH_ROOT" \
  --data-root artifacts/data --array-root artifacts/manifests \
  --manifest-root manifests/datasets \
  --snapshot "$OUTPUT_ROOT/input-authentication.json"

python scripts/run_reverse_edge_inference.py authenticate \
  --graph-root "$GRAPH_ROOT" --output-root "$OUTPUT_ROOT" \
  --data-root artifacts/data --array-root artifacts/manifests \
  --manifest-root manifests/datasets --experiment-revision "$REVISION"
```

For each preparation, dataset, and model, run a bounded smoke test followed
by actual evaluation. The following example evaluates ERM/GCN on PubMed:

```sh
python scripts/run_reverse_edge_inference.py evaluate \
  --phase smoke --max-checkpoints 30 --baseline-only \
  --preparation erm --dataset pubmed --model gcn \
  --graph-root "$GRAPH_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$OUTPUT_ROOT" --data-root artifacts/data \
  --array-root artifacts/manifests --manifest-root manifests/datasets \
  --input-snapshot "$OUTPUT_ROOT/input-authentication.json" \
  --experiment-revision "$REVISION" --backend-id "$BACKEND" --device cuda:0

python scripts/run_reverse_edge_inference.py evaluate \
  --phase actual --preparation erm --dataset pubmed --model gcn \
  --graph-root "$GRAPH_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --output-root "$OUTPUT_ROOT" --data-root artifacts/data \
  --array-root artifacts/manifests --manifest-root manifests/datasets \
  --input-snapshot "$OUTPUT_ROOT/input-authentication.json" \
  --experiment-revision "$REVISION" --backend-id "$BACKEND" --device cuda:0
```

Also run a small perturbed-graph smoke test using `--phase smoke`,
`--max-checkpoints 1`, and `--max-pairs 1`, without `--baseline-only`.
Use a separate output directory when testing a different backend.
Only one process may write a given preparation/model/dataset combination.

After all combinations finish, validate each of `erm`, `last`, `head`, and
`full`, then generate the four figures:

```sh
for preparation in erm last head full; do
  python scripts/run_reverse_edge_inference.py validate \
    --preparation "$preparation" --graph-root "$GRAPH_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" --output-root "$OUTPUT_ROOT" \
    --input-snapshot "$OUTPUT_ROOT/input-authentication.json" \
    --experiment-revision "$REVISION"
done
python scripts/analyze_reverse_edge_dose_response.py \
  --input-root "$OUTPUT_ROOT" --graph-root "$GRAPH_ROOT" \
  --output-root results/reverse_edge_perturbation \
  --paper-output-dir paper/figures
```

The analyzer checks exact checkpoint and treatment/control pairing before
aggregation. It first averages macro-F1 over the 30 checkpoints within each
graph replicate, then computes the mean and sample standard deviation over
the five graph replicates. Each figure has six model rows and five dataset
columns. Summary CSVs, four portrait PDFs, and file-hash validation records
are written together. Cluster runtime records and submission controls are
not part of these portable results.

## Replot from compact results

After analysis, the four `dose_response_summary.csv` files under
`results/reverse_edge_perturbation/{erm,last,head,full}/` are sufficient to
redraw the four plots without loading checkpoints or graph arrays:

```sh
python scripts/plot_reverse_edge_dose_response.py
```

The supplied PDFs are reference copies from the manuscript. Their accepted
compact summaries are not included in this checkout. Until those summaries
are supplied or a fresh run is completed, the command reports missing inputs.
The overall compact-only reproduction workflow uses `--skip-missing`, which
explicitly reports that these four reference PDFs were retained. It does not
claim to have regenerated or validated their numerical contents.

Fresh runs use the same registered datasets, checkpoint panel, graph
interventions and aggregation. Their scores and PDF bytes need not equal the
published reference. Baseline validation compares each checkpoint with its
own unperturbed predictions, rather than requiring the published scores.
