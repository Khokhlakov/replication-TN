# Capsule pipeline

Read-only notes from `../capsule-2202829/code/`. Line numbers are from that tree. `docs/ref/SI_methods.pdf` is present (about 507 KB). No PDF library is installed here, and none was installed, so sections S3-S5 were not read. SI methods not available as text.

## 1. Mutation tool

`gen_mut_structure.py` builds the mutant with Jackal SCAP and repairs the complex with profix.

- SCAP: `./scap -ini 20 -min 4 <pdb> ./tmp_scap.list` (`gen_mut_structure.py:24`). The list is one line `chain,resid,mutname` (`gen_mut_structure.py:20-21`). Output is renamed from `<pdb>_scap.pdb` to `<pdb>_mut.pdb` (`gen_mut_structure.py:25`).
- profix: `./profix -fix 0 /<workdir>/complex.pdb`, then `complex_fix.pdb` replaces `complex.pdb` (`gen_mut_structure.py:31-34`).
- `modify_pdb.py:11-13` copies only `ATOM` lines before that repair.
- The topology driver sets the site cutoff to 12.0 and passes it through (`Gen_feature_topo.py:27-28`).
- Mutant feature directories are named `feature/<pdb>_<chain>_<wild>_<resid>_<mut>` (`Gen_feature_topo.py:13`, `Gen_feature_aux.py:13`). The published prediction file is `<pdb>_<chain>_<wild>_<resid>_<mut>.ddg` (`run_topnettree.sh:52`). The script first writes `result.ddg` (`predict.py:70`).

`run_topnettree.sh` is bash. The server runbook uses tcsh and does not call it.

## 2. Element groups, site cutoff, hydrogens, waters, altlocs

- Element groups are C, N, O (`Gen_pointclouds.py:9`, `Construct_jplex_binding.m:14`). The element letter is PDB column 14 (`Gen_pointclouds.py:18`).
- Binding and mutation neighborhoods are VMD selections, cutoff from the command line: `within <cutoff> of chain <chain>` and `within <cutoff> of resid <resid> and chain <chain>` (`gen_mut_structure.py:46-48`). The topology entry point passes 12 (`Gen_feature_topo.py:27`).
- The aux pairwise cutoff is a different number, 10 Å (`MutationModule.py` in `gen_feature_aux`, `construct_index_list` / `construct_feature` default `CutNear = 10.0`).
- Hydrogens: only column-14 letters C, N, O are kept (`Gen_pointclouds.py:21`). Waters are `HETATM` and are not read (`Gen_pointclouds.py:17`).
- Alternate locations are not filtered. Every `ATOM` line is kept.

## 3. H0

`Construct_jplex_binding.m` (the mutation script is the same geometry with `wild_mutation_` names):

- Distance is the Euclidean norm, not a squared radius (`Construct_jplex_binding.m:28`).
- Same-set pairs stay at 100, so they never enter a 15 Å complex (`Construct_jplex_binding.m:24` and the `A(1,ii)+A(1,jj) == 1` test at line 27). Cross pairs are the finite distances.
- Stream: `createVietorisRipsStream(m_space, 1, 15.0, 200)` (`Construct_jplex_binding.m:36`). Dimension 1, max filtration 15, 200 divisions.
- H0 endpoints: `getEndpoints(intervals, 0, false)` (`Construct_jplex_binding.m:39`).

`gen_cnn_feature.py` bins those deaths:

- Cut 12 Å, width 0.25, 48 bins (`gen_cnn_feature.py:19-22`). Bars with persistence below 0.01 are dropped (`gen_cnn_feature.py:53`). Deaths at or above 12 are dropped (`gen_cnn_feature.py:69`).
- A death on a boundary is assigned to the higher bin because `BinID` overwrites the index (`gen_cnn_feature.py:27-30`).
- Each element pair writes wild (`j=0`) and mutant (`j=1`) into columns `Ip*3+Is+j*9` (`gen_cnn_feature.py:71-73`). Columns 18:27 are wild minus mutant (`gen_cnn_feature.py:75-76`).
- The 108-channel tensor is binding deaths, binding persistence, mutation deaths, mutation persistence (`gen_cnn_feature.py:139-141`). Death and persistence counts are non-negative. The difference blocks are signed.

This confirms Rips (Vietoris-Rips on Euclidean distance) for H0. It refutes squared radii for this step.

## 4. Alpha complex

`PH_Alpha.R:2-8` loads the R package `TDA` and calls `alphaShapeDiag`. The commented alternative is `alphaComplexDiag` (line 5). The diagram is printed with no `sqrt` and no factor of two.

`gen_feature_12.py:43-50` stores, for dimension 1 and again for dimension 2, sum / max / mean of bar length and min / max of birth and death. The third slot is mutant minus wild, assigned inside the element loop (`gen_feature_12.py:51`). The saved vector is the concatenation of binding H1, binding H2, mutation H1, mutation H2 (`gen_feature_12.py:68-75`).

Shape: `3 * 3 * 7 * 3` per homology dimension per site, times 2 dimensions times 2 sites = 756. That dimension is read from this script. It is not read from Wang, Cang and Wei, Nat. Mach. Intell. 2020. The published r about 0.65 and RMSE about 1.57 were not checked against that paper. Both stay unverified as paper claims. The 756 count itself is verified from `gen_feature_12.py`.

The H12 path does not apply `sqrt`. Whether `alphaShapeDiag` already returns radii or squared radii is not written in `PH_Alpha.R`. Unverified. A `sqrt` does appear in the unused Dionysus helper `MutationModule.py` `construct_PH_feature` (topo copy, lines 515-518). `construct_mutation_feature` does not call it; the PH block is commented out (`gen_feature_aux/MutationModule.py:877-880`).

## 5. How the matrices are assembled

- `X_unnormed.npy` in `code/src/model/` is the CNN tensor. `gen_cnn_feature.py:142` saves one complex as `Feature_cnn.npy`. The stack of 645 rows is not built by a script in this tree.
- `X_aux.npy` is `construct_mutation_feature` (`gen_feature_aux/MutationModule.py:837`): wild features, mutant features, mutant minus wild, then pKa summaries, then the wild environment block. Chimera, PH, PSSM and secondary-structure blocks in that function are commented out.
- `crossvalidation/X.npy` is what `crossvalidation_ab.py:9` loads. No script in the capsule concatenates aux and the CNN tensor into that file.
- G0 (see `docs/ALIGNMENT_G0.md`) found no column of `X.npy` with |Pearson r| above 0.9 against `X_aux` or flattened `X_unnormed`. Column standard deviations of `X.npy` are about 0.024 to 0.080, not 1. The normalized-concatenation hypothesis is not supported. There is no matched block and therefore no remainder to call an original H1/H2 block.
- `importanceidx.npy` is shape `(5888,)`. `crossvalidation_ab.py:16-20` slices `X` to `idx[:300]` before `KFold`. Every row is visible to that selection. The script that fit the importances is not in the capsule. One global index applied before the split is a leak if it is used for selection. This repo does not use it that way.

Aux constants: G0 found 2 (`X_aux` columns 669 and 676), not 3. Flattened `X_unnormed` has 460 constant columns. `5184 + 704` is not the constant-dropped width of these two arrays.

## 6. Insertion codes and identical aux rows

`Gen_pointclouds.py:20` reads the residue as `line[23:26]`. That drops the thousands digit and drops the insertion code in column 27. Two residues that differ only by an insertion code, or only by the thousands digit, become the same site key. `cutoff_mut` then compares that key to the resid argument (`Gen_pointclouds.py:52`).

The aux PQR reader stores `int(line[20:26])` (`gen_feature_aux/MutationModule.py:86`). Column 27, the insertion code, is outside that slice. DSSP is looked up with a blank insertion code (`MutationModule.py:814`).

There is no `try`/`except` in `gen_feature_aux/` or `gen_feature_topo/` that substitutes a zero vector. An empty near-residue list would hit a typo, `self.FeatureEnviroment` (`MutationModule.py:743`), and raise rather than write zeros. The all-zero aux rows in 2NYY, 2NZ9, HM_2NYY and HM_2NZ9 are therefore not produced by a fallback in the uploaded code. The parser collisions explain identical site keys. They do not, by themselves, explain a stored zero vector. Those rows are left as shipped. `X_aux` was not modified.

`X_cnn` can still differ for those BoNT rows because the binding-site cloud depends on the mutant coordinates, not only on the residue key.

## 7. Original CV versus `cv_mode=leaky_original`

`crossvalidation_ab.py`:

- Loads `X.npy` and `Y.npy` (`lines 9-11`). `Y` is cast with `.astype(float)`.
- `crosslabel.npy` is loaded only in a comment (`lines 14-15`). It does not define folds.
- Applies `importanceidx[:300]` on all rows (`lines 16-20`).
- `KFold(n_splits=10, shuffle=True)` with no `random_state` (`line 24`). Folds change every run.
- One `GradientBoostingRegressor`: 20000 trees, `max_features="sqrt"`, `learning_rate=0.001`, `max_depth=6`, `min_samples_split=3`, `subsample=0.4` (`line 44`). No CNN is trained here. The matrix is already built.
- Iteration count is `sys.argv[1]` (`line 28`). `note_for_reproducible_run.txt` says 50 repeats.
- The output file is opened `"wb"` and then written as text (`lines 69-74`). That raises on Python 3 at the end of a successful run.

`cv_mode=leaky_original` in this repo:

- Trains the CNN, the normalizer and the importance ranking once on all rows, then cross-validates the tree.
- Uses `KFold(..., random_state=seed)` so seeds 0, 1, 2 match the nested baseline's folds. They do not match the capsule's unseeded folds.
- Selects 300 CNN units from a model trained in the fold's leaky pass. It does not load `importanceidx.npy`.
- The tree hyperparameters in `configs/legacy_ab_cv.yaml` match line 44. The CNN (2000 epochs, the architecture in `train_model_cnn.py:31-45`) is extra relative to `crossvalidation_ab.py`, because that script never sees the raw tensor.

Differences, listed:

| Capsule `crossvalidation_ab.py` | This repo's `leaky_original` |
|---|---|
| Prebuilt `X.npy` (5888 columns), then 300 by `importanceidx` | CNN features of `X_unnormed` plus raw `X_aux`; top 300 from a ranking fit inside the run |
| No CNN | One CNN on all rows |
| Unseeded `KFold` | Seeded `KFold`, same seeds as the nested baseline |
| `crosslabel` unused | `crosslabel` unused |
| Tree only | Tree after the leaky CNN |
| Text written to a binary file at the end | CSV and JSON under `runs/` |

## 8. Pretrained files

`train_model_cnn.py:18-24` loads `X_cnn.npy` and `Y_cnn.npy`, trains 2000 epochs (`line 45`), and saves `cnn_model.h5` (`line 58`). Those two input names are not in `code/src/model/`. The directory does contain `cnn_model.h5`, `X_unnormed.npy` and `importance_idx.npy` (1536, the CNN permutation, not the 5888-vector). `predict.py:36-50` reloads `cnn_model.h5`, concatenates the new tensor with `X_unnormed.npy`, and normalizes using that concatenation. `predict.py:55` and `predict.py:64` load `../../data/gbdt1.pickle` (option 0, CNN + H12 + aux) and `gbdt2.pickle` (option 1, CNN + H12). Both pickles exist under `capsule-2202829/data/`. Nothing in the uploaded scripts records whether those artifacts were fit on AB-Bind, SKEMPI, or the SARS-CoV-2 DMS set. Unverified. The readme says the package training set is AB-Bind 645 (`readme.txt:3`). That sentence is not a training log for the pickle files.

## 9. Unverified lines, confirmed or refuted

| Claim | Result |
|---|---|
| r about 0.65 and RMSE about 1.57 on S645 | Unverified. The Nat. Mach. Intell. 2020 paper was not read. |
| 756-dimensional H12 block | Confirmed from `gen_feature_12.py`. Not confirmed from that paper. |
| Rips for H0, Euclidean distance | Confirmed. `Construct_jplex_binding.m:28` and `:36`. |
| Alpha complex for H1-H2 | Confirmed. `PH_Alpha.R:6`, consumed by `gen_feature_12.py`. |
| `sqrt` filtration, no extra factor of 2 | Refuted as a step in the H12 path. The TDA output scale is unverified. |
| TopNetmAb was trained on SKEMPI 2.0 plus SARS-CoV-2 DMS | Not decided by the capsule scripts. The readme names AB-Bind 645. Treat a SKEMPI/AB-Bind-only model as the early model, as in the task statement. |
