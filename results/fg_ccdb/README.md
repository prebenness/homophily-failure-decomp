# FG-CCDB PubMed recovery sensitivity

This exploratory sensitivity repeats A_last, A_head, and A_full for PubMed/GraphSAGE-mean, PubMed/GCN, and PubMed/ACM-GCN. The training objective is deterministic full-batch importance-weighted cross-entropy with weights fitted on each training split:

`w(bin, class) = n_bin * n_class / (N * n_bin,class)`.

This targets independence between the fixed local-homophily bin and the class label. It does not target independence for continuous local homophily. No sampler, clipping, smoothing, or additional cell-count factor is used.

## Paired results

| Model | Preparation | FG gain over ERM [95% CI] | FG minus GroupDRO [95% CI] | Whole-test change vs ERM |
|---|---|---:|---:|---:|
| GraphSAGE-mean | last | 0.0067 [0.0037, 0.0099] | -0.1090 [-0.1195, -0.0981] | 0.0012 |
| GraphSAGE-mean | head | 0.0433 [0.0386, 0.0476] | -0.1124 [-0.1176, -0.1066] | -0.0028 |
| GraphSAGE-mean | full | 0.0129 [0.0060, 0.0196] | -0.1492 [-0.1572, -0.1420] | -0.0042 |
| GCN | last | -0.0151 [-0.0190, -0.0115] | -0.0404 [-0.0494, -0.0310] | -0.0010 |
| GCN | head | 0.0116 [0.0067, 0.0160] | -0.0968 [-0.1054, -0.0888] | 0.0018 |
| GCN | full | 0.0017 [-0.0049, 0.0083] | -0.1202 [-0.1262, -0.1144] | -0.0072 |
| ACM-GCN | last | -0.0007 [-0.0035, 0.0020] | -0.0060 [-0.0103, -0.0018] | -0.0005 |
| ACM-GCN | head | 0.0084 [0.0042, 0.0127] | -0.0095 [-0.0134, -0.0059] | -0.0041 |
| ACM-GCN | full | 0.0030 [-0.0014, 0.0083] | -0.0982 [-0.1035, -0.0931] | -0.0036 |

## Recovered shares

| Model | Preparation | Recovered share |
|---|---|---:|
| GraphSAGE-mean | last | not interpreted |
| GraphSAGE-mean | head | not interpreted |
| GCN | last | not interpreted |
| GCN | head | not interpreted |
| ACM-GCN | last | not interpreted |
| ACM-GCN | head | not interpreted |

## Weight audit

All ten splits have complete support. The minimum class-by-bin training count is 50. The largest weight is 1.8791. Effective sample size is at least 96.69% of the eligible training set.

The local-homophily bins are oracle quantities constructed from true node and neighbour labels over the complete graph. This is therefore a retrospective decomposition diagnostic, rather than a deployable semi-supervised debiasing procedure.

The raw run archive is supplied at runtime with `--fg-raw-root`.
