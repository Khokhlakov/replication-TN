# Data notes

Measurements below were computed from the files in this package and from `AB_645.csv`. Items marked **inferred** are not confirmed by a second source.

## Labels and sign

`data/labels/ab_bind_s645.csv` is the AB-Bind S645 table (Sirin et al.) in file order.

- 645 rows, 29 `complex_id` values (24 crystal PDBs plus 5 `HM_*` homology models).
- 87 rows have `is_homology = 1` (558 do not). That is the homology-blind split.
- 27 rows have `ddg == 8`. Those are the non-binders. AB-Bind encodes them as **+8**, not -8.
- `code/crossvalidation/Y.npy` matches this `ddg` column exactly, same order.

**Sign decision.** The task text expected "positive = binding strengthened" and non-binders at -8. The table and `Y.npy` use the opposite convention: positive `ddg` means weaker binding, and non-binders are +8. Example: `1AK4` chain D `A488G` is +2.49 in both files. Training uses the table sign. Negating every label would swap the meaning of the prediction and would move non-binders to -8; Pearson r and RMSE of a retrained model would be unchanged. This is recorded so it can be reversed if a later check disagrees.

`X_unnormed.npy` / `X_aux.npy` are not self-describing. The order above is used because `Y.npy` matches the table and the duplicate experimental rows line up with duplicate aux vectors (next section).

## Identical auxiliary rows

36 adjacent `X_aux` pairs are exactly equal.

| | count |
|---|---|
| same complex, chain and residue | 3 |
| same complex | 36 |
| both rows the all-zero vector | most of the rest |

The 3 site matches are duplicate experimental entries (`1DVF` `A:Y49A`, `1N8Z` `H:Y100aF`, `1N8Z` `L:H91F`) with different `ddg` and identical features. 30 of the 36 pairs are all-zero aux vectors on consecutive rows of the same PDB (`2NYY`, `2NZ9`, `HM_2NYY`, `HM_2NZ9`, and a few short runs). Those zeros are failed feature rows, not evidence that the sites are the same. Site match rate: **3/36**.

`X_aux` has 707 columns, 2 of them constant (669 and 676), 48 columns with an absolute value above 1e6, and a maximum near 3.7e20. The last column takes integer values from -4 to 7. Trees see the raw aux block. `signed_log1p` is available and is not used in training.

## H0 tensor layout

Verified from `gen_cnn_feature.py`, and consistent with the column means from `scripts/inspect_tensor.py`:

- Cutoff 12 Å, bin width 0.25 Å, 48 bins. A death exactly on a boundary is assigned to the higher bin. Deaths at or above 12 Å are dropped. Bars shorter than 0.01 Å are dropped.
- Four blocks of 27 channels, concatenated in this order:
  1. binding-site death counts
  2. binding-site persistence counts (how many bars are still alive in earlier bins)
  3. mutation-site death counts
  4. mutation-site persistence counts
- Inside each block, 9 element pairs in order CC, CN, CO, NC, NN, NO, OC, ON, OO, then wild, mutant, and wild minus mutant.
- Column means of the first block: wild and mutant nearly equal (max absolute gap about 0.04), difference columns near 0. The same WT / MUT / DIFF pattern repeats in the other three blocks. Persistence blocks have much larger counts than death blocks, which matches "count the bar in every bin before its death".

8.87% of the (bin, channel) cells have zero standard deviation over the 645 rows, and those cells are identically 0.

The reference `importance_idx.npy` puts most of its top 300 indices in bins 12-20, with smaller clusters around 24-26, 31-32 and 42, plus a few counts outside those clusters (including bins 35, 43 and 46). Filter index within a bin is not comparable across trainings.

## What is still missing

- `X_h12` is not in the capsule. The legacy script builds a **756-d** vector: H1 and H2, binding and mutation site, 9 element pairs, 7 statistics, and a mutant-minus-wild slot (`3*3*7*3*4`). **Inferred from code**, not from an array we can hash. The 7 statistics are sum/max/mean of bar length and min/max of birth and death. The task text also lists min and std; that wider set is implemented as `spec_bar_stats` and is not the training vector.
- H0 difference is wild minus mutant. H12 difference in the legacy script is mutant minus wild.
- Mutant PDBs are not in the capsule, so `X_unnormed.npy` has not been rebuilt row by row. The H0 generator is covered by invariance tests only.
- Auxiliary features are not regenerated. `mutation_aux` raises rather than invent a 707-d stand-in.
- The legacy PDB reader uses `line[23:26]` for the residue number. Below 1000 that only drops a space. At or above 1000 it drops the thousands digit. The default reader keeps the full resSeq plus insertion code.

## Open point

If a structure-level audit shows that `X_unnormed.npy` is not in `AB_645.csv` order, the label file has to be realigned before any reported correlation is meaningful. The `Y.npy` match is necessary evidence and not a proof about the CNN tensor by itself. The duplicate-mutation aux pairs are the additional check, and they agree.
