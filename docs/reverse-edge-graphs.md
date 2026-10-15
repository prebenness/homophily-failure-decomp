# Reverse-direction edge perturbations

This extension constructs graph pairs for the opposite direction of the
Appendix J intervention. It does not train models, run checkpoint inference,
or modify the existing results. The graph pairs can later be shared by the
ERM, last-layer, classification-head, and full-model checkpoint evaluations.

## Direction and target population

Treatment decreases local homophily in PubMed and ogbn-arxiv, and increases
it in Roman-empire, Amazon-ratings, and arxiv-year. Each selected target loses
one neighbour and gains one neighbour of the opposite label relation. Its
local homophily therefore changes by exactly minus or plus `1 / degree`.
The matched control changes one neighbour while preserving that target's
local homophily. Both graphs preserve every node's degree, the edge count,
and the absence of self-loops and duplicate undirected edges.

The original registered tail remains the denominator and the population for
any later performance evaluation. A node is eligible for selection only if
it has a neighbour whose label relation can be changed in the requested
direction: at least one same-label neighbour for a decrease, or at least one
different-label neighbour for an increase. Isolated nodes are not eligible.
Thus, nodes already at zero cannot be selected for a decrease, and nodes
already at one cannot be selected for an increase.

Only selected targets are protected against unintended homophily changes.
Other nodes, including unselected tail nodes and saturated nodes, may take
part in swaps and have their local homophily changed indirectly. The output
records these collateral changes. Degree preservation applies to all nodes.
This is the existing ordinary paired-swap generator's behaviour, not a new
rule that protects the entire tail.

## Percentages and replicates

The horizontal axis is the percentage of the **original tail** selected for
rewiring. It is not rescaled to the eligible population. Regular requested
doses are 5%, 10%, and so on, stopping before the number of requested targets
would exceed the eligible count. Integer target counts are rounded down.
The eligible endpoint is also attempted when it differs from the last
regular dose. Its actual percentage is `100 * eligible_count / tail_count`;
for example, 175 eligible nodes out of 2,696 tail nodes gives an endpoint of
approximately 6.491%. The manifest records both requested and actual
percentages. The unchanged graph supplies the zero-dose baseline.

There are five independently seeded treatment/control pairs at each
nonzero dose. The dataset-specific seed bases are unchanged from the original
experiment; replicate indices are 1 through 5, and each replicate's seed is
the base plus its index. For each replicate, the original seeded permutation
of all tail nodes is filtered to eligible nodes. Doses use nested prefixes
of this filtered ranking. No failed target is replaced with another node.

Every dose starts from the original graph. Increasing the dose increases the
number of selected targets; it does not continue rewiring the preceding
dose's graph. The ordinary paired-swap generator is used for all five
datasets, including the reverse ogbn-arxiv intervention. The earlier
coordinated increase generator is not used in this extension.

## Acceptance and stopping

A percentage is accepted only after all five graph pairs have been
generated and authenticated. At the first dose without five complete pairs,
generation stops for that dataset and retains the earlier complete doses.
An incomplete set of replicates is not a plottable result. The actual run
attempts the eligible endpoint only if all preceding doses are complete;
a separate smoke test may probe it earlier.

The retry cap and time limit are recorded operational bounds, not part of
the experimental definition. Reaching either limit does not prove that the
requested graph is mathematically impossible. The stop record distinguishes
these cases and records the achieved cutoff without claiming a proven
maximum. Later doses are not silently skipped or replaced. Reaching 100%
is not required.

Authentication checks the registered inputs, target identities, seeds,
checksums, exact treatment and control changes, and graph invariants. It
replays the recorded swaps from the original graph and compares the result
with the saved graphs. Resume uses only authenticated complete pairs;
partial files do not establish completion. The final catalog includes only
percentages with all five accepted pairs.

## Portable entry point

`scripts/run_reverse_edge_graphs.py` provides `prepare`, `smoke`, `generate`,
and `verify` commands. For example, from the repository root with the
registered inputs available:

```sh
python scripts/run_reverse_edge_graphs.py prepare --dataset pubmed \
  --data-root artifacts/data --array-root artifacts/manifests \
  --manifest-root manifests/datasets --output-root artifacts/reverse-edge-graphs
python scripts/run_reverse_edge_graphs.py smoke --dataset pubmed \
  --output-root artifacts/reverse-edge-graphs
python scripts/run_reverse_edge_graphs.py generate --dataset pubmed \
  --output-root artifacts/reverse-edge-graphs
python scripts/run_reverse_edge_graphs.py verify --dataset pubmed \
  --output-root artifacts/reverse-edge-graphs
```

Use `--help` for the full arguments and operational bounds. Input
preparation, graph generation, and verification should run on a suitable
compute worker. Runtime locations and scheduling remain outside the
research repository.

Each dataset has separate `prepared`, `smoke`, and `actual` output
directories. The actual outputs retain the experiment manifest, baseline,
graph arrays, swap records, per-pair metadata, accepted catalog, validation
report, and any stop record. These reverse-direction outputs are separate
from the existing forward-direction results.
