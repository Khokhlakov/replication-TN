# Results

Nested 10-fold `cnn_aux` has been run with `configs/legacy_ab_cv.yaml` (2000 CNN epochs, 20000 trees, 3 seeds, no H1/H2). It misses the published gate. The aux-only log was not in the pasted output.

| Quantity | Published | This package | Error | Gate | Milestone |
|---|---|---|---|---|---|
| 10-fold Pearson r | 0.65 | 0.467 ± 0.049 | 0.183 | fail (outside ±0.05) | M4 |
| 10-fold RMSE (kcal/mol) | 1.57 | 1.791 ± 0.067 | 0.221 | report | M4 |
| 10-fold r without 27 non-binders | 0.68 | 0.480 ± 0.012 | 0.200 | report | M4 |
| 10-fold RMSE without non-binders | 1.06 | 1.421 ± 0.025 | 0.361 | report | M4 |
| Homology-model blind r | 0.55 | not run | — | report | M4 |
| Leave-one-complex-out mean r | 0.51 | not run | — | report | M4 |

## Fast 10-fold, one seed (`configs/fast.yaml`, CUDA server, 2026-11-01)

Runs: `runs/20261101T081918Z_nested_cnn_aux_kfold10` and `runs/20261101T082840Z_leaky_original_cnn_aux_kfold10`.

Predicting the label mean would give RMSE 1.96 on all 645 rows and 1.40 on the 618 binders.

| | Pearson r | RMSE | r without non-binders | RMSE without non-binders |
|---|---|---|---|---|
| Nested (headline) | 0.500 | 1.734 | 0.494 | 1.378 |
| Leaky, labeled optimistic | 0.831 | 1.143 | 0.754 | 0.928 |
| Gap (leaky − nested) | 0.332 | −0.590 | 0.260 | −0.450 |

The nested result beats the null RMSE. The leaky result is far above the published 0.65 because the CNN and the feature ranking saw the test rows. It is not a TopNetTree score.

## Full nested 10-fold (`configs/legacy_ab_cv.yaml`, 3 seeds, 2026-11-01)

Runs: `runs/20261101T084813Z_nested_cnn_aux_kfold10` and `runs/20261101T084813Z_nested_cnn_kfold10`. Null RMSE is 1.96 on all rows and 1.40 on the 618 binders.

| Feature set | Pearson r | RMSE | r without non-binders | RMSE without non-binders |
|---|---|---|---|---|
| `cnn_aux` | 0.467 ± 0.049 | 1.791 ± 0.067 | 0.480 ± 0.012 | 1.421 ± 0.025 |
| `cnn` | 0.464 ± 0.047 | 1.786 ± 0.061 | 0.484 ± 0.017 | 1.396 ± 0.012 |
| `aux` | log not pasted | | | |

`cnn_aux` and `cnn` agree within the seed noise, so the shipped auxiliary block is not improving this fit. Binder RMSE does not beat predicting the mean (1.40). The longer protocol is also slightly worse than the 200-epoch nested run (r 0.500, RMSE 1.734). The miss against 0.65 is recorded as a failure of this no-H12 ablation, not hidden.

## Still open

- Homology-model blind test and leave-one-complex-out.
- The aux-only row above.
- `X_h12` is still absent, so none of these runs is TopNetTree.

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

Runtime per fold: not measured. The fast pair finished; wall time was not recorded in `metrics.json`.
