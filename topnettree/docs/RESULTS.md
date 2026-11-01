# Results

No full S645 cross-validation has been run. The numbers below are the published targets. They are not claimed as reproductions.

| Quantity | Published | This package | Error | Gate | Milestone |
|---|---|---|---|---|---|
| 10-fold Pearson r | 0.65 | not run | — | ±0.05, nested only | M4 |
| 10-fold RMSE (kcal/mol) | 1.57 | not run | — | report | M4 |
| 10-fold r without 27 non-binders | 0.68 | not run | — | report | M4 |
| 10-fold RMSE without non-binders | 1.06 | not run | — | report | M4 |
| Homology-model blind r | 0.55 | not run | — | report | M4 |
| Leave-one-complex-out mean r | 0.51 | not run | — | report | M4 |

## Why M4 was not scored

- `X_h12` (756-d in the legacy script) is not in the capsule. Any model fit here is a **no-H12** ablation. The task allows that as a documented miss against 0.65, and it is not a pass.
- The GBDT settings in `default.yaml` are unverified. The uploaded CV script uses a different set (`legacy_ab_cv.yaml`). Neither file is a record of the paper's 10×10-fold search.
- Nested 10-fold CV refits the CNN for every fold (2000 epochs, batch 8). This machine has a CPU build of PyTorch (`torch 2.10.0+cpu`). That run was not started.
- `leaky_original` is implemented and tested for leakage behavior. It is optimistic and is not a headline number. The nested-versus-leaky gap is therefore not yet a measured correlation gap.

## What did run

- Data contract: shapes, the importance permutation, 27 non-binders at +8, 87 homology rows, 29 complexes, aux alignment 3/36 site matches.
- Layer A on `antibodies_BFEs.csv`: disrupted counts `[185, 163, 143, 124, 106, 78, 60, 33, 15, 5, 2]`, fold reduction about 20.94, ACE2 sum about 2.602. K417N × LY-CoV016 is strongly negative and N501Y × ACE2 is positive in the published table.
- CNN flatten order, normalizer, nested-fit spy, H0 invariance.
- The 64-row overfit check passes: after 900 epochs (Adam, lr 1e-4, legacy dropout) the eval-mode training MSE is below 0.05. At 400 epochs it was 0.065. Eval mode is the check, because dropout is off when features are extracted.
- Keras equivalence is skipped when TensorFlow is not installed. Gudhi MST cross-check is skipped when Gudhi is not installed.

## Ablations and substitutions

| Piece | Status |
|---|---|
| CNN on the shipped H0 tensor | implemented |
| Top-300 CNN features, re-ranked per training fold | implemented |
| `X_h12` | missing; models named `cnn_aux`, `cnn`, `aux` |
| `X_aux` | shipped array used as-is; generator refused |
| H0 generator | MST + legacy binning; not matched back to `X_unnormed.npy` (no mutant PDBs) |
| H1/H2 generator | legacy 7 statistics; needs Gudhi to fill bars |
| Panel rescoring of BA.1 | not run; needs per-complex features and a trained bundle |
| Published panel vs itself | agreement metrics are tested; that is not a model score |

Runtime per fold: not measured. Fill this table from `runs/<stamp>_*/environment.json` and `metrics.json` after `make cv-full` or `make cv-fast`.
