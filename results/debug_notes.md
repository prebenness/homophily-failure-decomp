# Debug-pass notes

Hand-written observations from the debug passes. Run records live in
`debug/`; the generated summary table is `debug_baseline.md`. Everything
noted here was established from whole-split training curves only (training
loss and whole-validation score); no tail, bin, within-class, or
decomposition numbers were computed or seen.

## Late best checkpoints (2026-07-09)

Several cells record their best validation checkpoint late in the training
budget (see the best-epoch column of `debug_baseline.md`). The training
logs of some of these runs were inspected: in every run inspected, loss and
validation score plateau at a reasonable point, and the validation score
oscillates randomly through the later part of training; a best checkpoint
landing near the end is chance placement within that oscillation. All
inspected runs looked fine.
