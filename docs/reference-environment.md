# Reference software environment

The verified reference environment for rerunning the experiment and paper
analysis uses Python 3.11.15, NumPy 2.2.6 and PyTorch 2.6.0 with CUDA 12.4.
Its full installed Python package set is locked in
`requirements-reproducibility.txt`. `environment.yml` creates that environment
and installs this repository in editable mode.

From the repository root:

```powershell
conda env create -f environment.yml
conda activate hfd-repro
python scripts/verify_environment.py
pytest
```

The lock records all transitive Python packages observed in the reference
environment. The operating-system NVIDIA driver and GPU firmware are outside
the Python lock. The recorded raw-run configurations provide the training
Git revision and core package versions for each run.

The 4,200 retained paper runs record three core environment tuples:

| Runs | Python | NumPy | PyTorch |
|---:|---:|---:|---:|
| 2,240 | 3.10.12 | 2.2.6 | 2.6.0+cu124 |
| 1,860 | 3.11.15 | 2.2.6 | 2.6.0+cu124 |
| 100 | 3.11.15 | 2.4.6 | 2.5.1 |

Those historical run records do not contain full transitive package lists.
The exact historical environments therefore cannot be reconstructed from the
archive. The reference lock supplies one complete environment for a fresh,
uniform rerun. Hardware-dependent numerical differences remain possible.

The 630 Roman-empire recovery directories record
`00bc07934e50c94cbd4415ffa974fc704dd0aafd` as their base Git revision.
That revision used the dataset-wide class set for validation-tail macro-F1,
whereas the active histories and metrics use true-label support. The external
correction manifest records the exact migration-script checksum, and Git commit
`17e58488` retains that script. The external archive retains the superseded
files and records that the selected checkpoint epoch did not change. The exact working-tree state used for the
reported Roman-empire reruns was not recorded separately, so the historical
`code_version` field alone is insufficient provenance for those 630 runs.
The current training code implements the corrected criterion used by the
paper.

`tensorflow-cpu` is included because the H2GCN parity test uses the authors'
TensorFlow implementation. It is not used to train the main experiment panel.

Figure PDF byte identity depends on the rendering libraries and installed
fonts. The tracked PDFs are therefore the canonical released artefacts.
Numerical reproduction remains checkable when a different environment
produces harmless PDF metadata or font-subsetting differences.
