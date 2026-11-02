# Runbook

Shell is tcsh. Run from `topnettree/`. `setenv CAP ../capsule-2202829` is used when a command reads the capsule. Do not edit the capsule or `data/raw/`.

Training labels are `data/labels/ab_bind_s645.csv` column `ddg`. Positive means binding weakened. Non-binders are `is_nonbinder`, value +8. Do not flip the sign. Multiply a prediction by -1 only when comparing with a JPCL number (positive = strengthened). The supplement `ddg` column is the opposite sign and is not a target.

S645 numbers below are validation of this re-implementation. They are not results from Chen and Wei, J. Phys. Chem. Lett. 2022.

## What changed in code

No `src/` or `scripts/` path treats non-binders as -8, clips at -7, or finds them by the label value. `metrics.py` and `compare_runs.py` use `is_nonbinder`. Nothing was edited for a -8 assumption.

| File | Change |
|---|---|
| `src/topnet/data.py` | `assert_label_tables`: same length, same pdb/chain/wt/mut, `resid` matches `resid_str` up to case, canonical `ddg` matches `ddg_abbind`. |
| `src/topnet/cv.py` | `homology_blind_strict`, `loco_parent`, dup-group check on those new splits, `--row-subset` via `restrict_splits`, per-fold GBDT from `gbdt_by_fold`, metrics on rows that were actually tested. `kfold10` is unchanged. |
| `src/topnet/gbdt.py` | Feature sets `h12`, `cnn_h12`, `cnn_aux_h12`, `h12_original`. Order for `cnn_aux_h12` is CNN, then H1/H2, then aux. |
| `src/topnet/pipeline.py` | Those sets are labeled `regenerated H12 (not original)`. |
| `src/topnet/features/pdbio.py` | `read_pdb_records` keeps chain, resseq, icode, altloc. |
| `scripts/run_cv.py` | Label assert, supplement `parent_pdb` and `dup_group`, `--seeds`, `--row-subset`. |
| `scripts/train_final.py` | Label assert. With `--h12`, also fits `h12`, `cnn_h12`, `cnn_aux_h12`. |
| `scripts/tune_gbdt.py` | Inner-only random search, `staged_predict`, 1 SE rule. Option A retrains the CNN inside the inner fold. Option B is `tune.cnn_option: B`. |
| `scripts/inspect_tensor.py` | Distinct aux rows per complex. |
| `tests/test_predict_and_tune.py` | Spy on `GradientBoostingRegressor.fit`. Held-out labels are row ids and must not appear. |

New scripts: `check_capsule_alignment.py`, `check_dup_folds.py`, `check_paper_arithmetic.py`, `compare_runs.py`, `report_split_metrics.py`, `validate_abbind_structures.py`, `reproduce_h0.py`.

## Recommended order

1. G0 and T0. G0 passed. Do not stop.
2. T5a. Already run on this workstation. The command below reprints it.
3. T1 and T1b on GPU 0. T2 tuning on CPU. Then T2 nested runs on GPU 0 and GPU 1.
4. T3: homology on GPU 0, LOCO on GPU 1, after T2 has written `configs/tuned.yaml`.
5. T4.1–T4.3. The H0 pilot failed. Do not start T4.4–T4.7.
6. T5b only after an explicit yes.

## G0

Already run. Verdict: **PASS**. Y.npy equals `ddg_abbind` at 645/645 positions in order. Report: `docs/ALIGNMENT_G0.md`.

```
python scripts/check_capsule_alignment.py
```

CPU, about 20 seconds. Rewrites `docs/ALIGNMENT_G0.md`.

What (c) found: no `X.npy` column has |Pearson r| above 0.9 against `X_aux` or flattened `X_unnormed` (max |r| 0.672, median 0.193). Column std of `X.npy` is 0.024–0.080, not 1. That rejects a column-wise z-scored concatenation. It does not identify an H1/H2 block, because no source block matched. T4.8 is skipped. `X.npy` does not independently verify the `X_aux` row order. The pass is the Y match.

Aux constant columns: 2 (669 and 676), not 3. `crosslabel.npy` is not used by `crossvalidation_ab.py` (the load is commented out). 640 distinct values. It is not a fold id. `AB_645.csv` matches the supplement in order after case-folding; 22 insertion-code rows differ only by `a` versus `A`. Repo `X_aux.npy` and `X_unnormed.npy` are byte-identical to `code/src/model/`.

## T0

No compute. Write-up: `docs/CAPSULE_PIPELINE.md`.

Confirmed from the capsule: Rips on Euclidean distance for H0 (`Construct_jplex_binding.m:28`, stream max 15 with 200 divisions at line 36); alpha complex via `TDA::alphaShapeDiag` (`PH_Alpha.R:6`); 756 H12 numbers from `gen_feature_12.py`. Refuted: an explicit `sqrt` in that H12 path. Unverified: the numeric scale `alphaShapeDiag` returns, the Nat. Mach. Intell. 2020 r/RMSE, and which dataset the shipped `cnn_model.h5` and `gbdt1/2.pickle` were fit on. SI methods not available as text (`docs/ref/SI_methods.pdf` is in the tree; no PDF reader was installed).

Identical aux rows: the point-cloud reader uses `line[23:26]` and the PQR reader uses `int(line[20:26])`, so insertion codes and the thousands digit are dropped. There is no `try/except` that writes a zero vector. The zero blocks are unexplained by a fallback. They were not edited.

## T5a

CPU, under a minute. Already run.

```
python scripts/check_paper_arithmetic.py
```

| Check | Result | Provenance |
|---|---|---|
| exp(2.98-2.60) = 1.462 vs about 1.5 | PASS | Prompt figures, not recomputed from a structure |
| exp(2.98) = 19.69 vs about 20 | PASS | Same |
| exp(1.55) vs 4.2x | FAIL | 4.711 vs 4.2. The two prompt figures disagree |
| 27.1/20.9 = 1.297 vs 30% | PASS | Prompt figures |
| 163/185, 168/185, 164/185 | PASS | 0.881, 0.908, 0.886 |
| Fig 3e vs Fig 3f Pearson | PASS | 0.899 vs about 0.9. Prompt figures |
| Header equals `antibodies.csv` | FAIL as exact text | The BFE header is the name after the underscore (`6M0J_ACE2` vs `ACE2`). Those suffixes match |
| ACE2 sum | PASS | 2.6018 from `antibodies_BFEs.csv` |
| H15 anchors | PASS | Row 5 LY-CoV016 = -2.016; row 10 LY-CoV555 = -2.793; row 11 LY-CoV555 = -1.670; row 7 REGN10987 = -0.787 |
| BA.1 fold | PASS | 20.9445 vs 20.9, from this file, paper sign |
| Unclipped 15-mutation sum < -0.3 | 115 antibodies | Not 163. The count 163 is the most-negative single mutation < -0.3 |
| PDB ids | 184 unique 4-character ids | Not PDB ids: `RN33_REGN10933`, `RN87_REGN10987` |

BA.2, BA.3, Delta and the earlier variants are not in this file. The task states the SI ZIP has only these two files. No other per-mutation table was used.

## T1

GPU 0. Same tree, epochs and seeds as `configs/legacy_ab_cv.yaml`. Folds come from `kfold_splits` with `random_state=seed`, which is the nested baseline's generator. The capsule's own `KFold` has no seed.

```
mkdir -p logs
setenv CUDA_VISIBLE_DEVICES 0
nohup python scripts/run_cv.py --config configs/leaky_cnn_aux.yaml >& logs/t1_leaky_cnn_aux.log &
```

Runtime: unverified. One 2000-epoch CNN on 645 rows, then 3 seeds times 10 folds of a 20000-tree model. The nested 3-seed pair on 2026-11-01 finished the same day and did not record wall time. Expect on the order of a few hours, shorter than that nested pair because the CNN is trained once.

Output: `runs/<timestamp>_leaky_original_cnn_aux_kfold10/predictions.csv` and `metrics.json`. The log prints the path.

Read it only after the nested baseline `runs/20261101T084813Z_nested_cnn_aux_kfold10` is the paired partner. Gap = leaky minus nested. Labeled optimistic. Not a headline.

```
python scripts/compare_runs.py --nested runs/20261101T084813Z_nested_cnn_aux_kfold10 --leaky runs/<timestamp>_leaky_original_cnn_aux_kfold10
```

This appends `## OPTIMISTIC - not a headline number` to `docs/RESULTS.md`. Run it once. Non-binders are dropped with `is_nonbinder`.

If leaky Pearson r is about 0.65, the protocol is a credible account of the gap to the published figure (that figure itself is unverified against the 2020 paper). If it stays about 0.5, the features or the tree are the likelier account. The fast one-seed pair already in `docs/RESULTS.md` was 0.831 versus 0.500 at 200 epochs and is a different protocol length.

Duplicate groups under this same generator, sklearn 1.9.1 on the workstation that prepared this file (re-run on the server; a different sklearn could shuffle differently):

```
python scripts/check_dup_folds.py --config configs/legacy_ab_cv.yaml
```

Seeds 0, 1, 2: 3, 2 and 3 of the 3 groups straddle. Seed 1 keeps dup_group 228 in one fold. Seeds 0–4 under `configs/fast_nested.yaml`: 3, 2, 3, 3, 3. The old folds were not regenerated.

### T1b

`crossvalidation_ab.py` uses Python 3 `print` calls. numpy, scipy and sklearn import on this workstation. It was not executed. A full run is `argv` repeats (the note says 50) of 10-fold times 20000 trees, and the script then writes text into a file opened `"wb"` (`crossvalidation_ab.py:69-74`), which raises on Python 3 after the loop. Do not treat a partial log as the original result.

On the server, copy only, then stop unless you intend the long run:

```
mkdir -p scratch/t1b
cp $CAP/code/crossvalidation/crossvalidation_ab.py scratch/t1b/
cp $CAP/code/crossvalidation/X.npy scratch/t1b/
cp $CAP/code/crossvalidation/Y.npy scratch/t1b/
cp $CAP/code/crossvalidation/importanceidx.npy scratch/t1b/
cd scratch/t1b
python -m py_compile crossvalidation_ab.py
```

If that compile works and you accept a multi-hour CPU job, the as-is command is `python crossvalidation_ab.py 1` for one repeat, not 50. Any edit, including the binary-file write, has to be saved as a diff under `scratch/t1b/` before the rerun. Report that r and RMSE as "original protocol on original features". It is not an S645 headline for this package and not a JPCL number.

## T2

Tuning is CPU. Option A is the default (`configs/fast_nested.yaml` sets `tune.cnn_option: A`). The search forces the CNN onto CPU.

```
mkdir -p logs runs/t2_tune
python scripts/tune_gbdt.py --config configs/fast_nested.yaml --protocol random_inner --out runs/t2_tune >& logs/t2_tune.log
```

Runtime, unverified: 5 seeds times 10 outer folds times 12 trials times 3 inner folds is 1800 CNNs of 200 epochs, about 36 times one 5-seed 200-epoch nested run. If that nested run is a few hours, this search is on the order of a few days. Each trial fits up to 3000 trees and picks the tree count with `staged_predict`. Outer test rows are not read.

Option B, only if A is too slow: set `cnn_option: B` in a copy `configs/fast_nested_optimistic.yaml` (do not edit `fast_nested.yaml` in place if a run is using it) and repeat the command with `--config` pointing at that copy. The CNN is then fit once per outer-training split, so inner validation rows were in the CNN. The written winner is tagged `mildly optimistic`.

Outputs: `runs/t2_tune/trials.csv`, `runs/t2_tune/chosen_by_fold.csv`, and `configs/tuned.yaml` (`tune_status` becomes `complete` or `per_fold`). `run_cv.py` applies `gbdt_by_fold` inside nested mode. Until that file is rewritten, `tune_status: not_run` and T3 prints a warning and uses the untuned tree.

Then the 200-epoch nested runs. Default tree is the legacy_ab_cv tree (20000 estimators), not the tuned one.

```
setenv CUDA_VISIBLE_DEVICES 0
nohup python scripts/run_cv.py --config configs/fast_nested.yaml --feature-set cnn_aux >& logs/t2_cnn_aux.log &
setenv CUDA_VISIBLE_DEVICES 1
nohup python scripts/run_cv.py --config configs/fast_nested.yaml --feature-set cnn >& logs/t2_cnn.log &
```

After `configs/tuned.yaml` exists and those two have finished:

```
setenv CUDA_VISIBLE_DEVICES 0
nohup python scripts/run_cv.py --config configs/tuned.yaml --feature-set cnn_aux >& logs/t2_tuned_cnn_aux.log &
setenv CUDA_VISIBLE_DEVICES 1
nohup python scripts/run_cv.py --config configs/tuned.yaml --feature-set cnn >& logs/t2_tuned_cnn.log &
```

Each process is 5 seeds times 10 folds times 200 CNN epochs plus the tree. Unverified, on the order of a few hours per process. The tuned tree may be shorter because `n_estimators` is at most 3000.

Compare each new `predictions.csv` to `runs/20261101T084813Z_nested_cnn_aux_kfold10` with `compare_runs.py`. Seeds 3 and 4 have no 2000-epoch partner; the script still emits their rows, and the mean gap mixes protocols. Prefer the per-seed rows for seeds 0, 1 and 2. Say so in any note you add. The 2000-epoch baseline is no-H12 `cnn_aux`. A `cnn` or 200-epoch run is not the same feature set or the same training length.

Known bias: under option A the inner CNN is honest given the outer-training tensor, and that tensor's CNN features in a later nested run are still fit on the outer-training rows. The search does not peek at the outer test fold. Option B is the mildly optimistic one.

## T3

Do this after T2 has replaced `configs/tuned.yaml`. Otherwise the warning in the log is the result: untuned legacy tree, 200 epochs.

Homology, GPU 0. LOCO, GPU 1. One seed.

```
setenv CUDA_VISIBLE_DEVICES 0
nohup python scripts/run_cv.py --config configs/homology.yaml >& logs/t3_homology.log &
setenv CUDA_VISIBLE_DEVICES 1
nohup python scripts/run_cv.py --config configs/loco.yaml >& logs/t3_loco.log &
```

Strict homology drops crystal templates 1KTZ, 1YY9, 3BN9, 2NYY, 2NZ9 from training. They are in neither split. 87 HM rows are the test.

```
setenv CUDA_VISIBLE_DEVICES 0
nohup python scripts/run_cv.py --config configs/homology_strict.yaml >& logs/t3_homology_strict.log &
```

Parent-PDB LOCO (the leakage-safe grouping; `HM_x` and `x` share a template):

```
setenv CUDA_VISIBLE_DEVICES 1
nohup python scripts/run_cv.py --config configs/loco_parent.yaml >& logs/t3_loco_parent.log &
```

Seeds 1 and 2, after seed 0 looks sane:

```
python scripts/run_cv.py --config configs/homology.yaml --seeds 1,2
python scripts/run_cv.py --config configs/loco_parent.yaml --seeds 1,2
```

Runtime, unverified: homology is one CNN on 558 rows for 200 epochs. LOCO is 29 CNNs. Each is much smaller than the 5-seed 10-fold run.

Read:

```
python scripts/report_split_metrics.py --predictions runs/<homology>/predictions.csv --bootstrap 1000
python scripts/report_split_metrics.py --predictions runs/<loco>/predictions.csv --group complex_id
python scripts/report_split_metrics.py --predictions runs/<loco_parent>/predictions.csv --group parent_pdb
```

Report the blind r and RMSE with the printed 95% interval. For LOCO report mean and median per-complex r, the pooled r, and the line that drops complexes with n<5 or undefined r. The 0.55 and 0.51 targets are for the full published model. A miss without H12 is not a failure. These runs are no-H12 unless `tuned.yaml` somehow gained an H12 matrix, which it does not.

Preconditions checked in `run_cv.py`: 87 homology rows and 29 `complex_id` values.

## T4

### T4.1

Done. `python scripts/inspect_tensor.py`. 48 bins, four blocks of 27, each block nine element pairs times WT / MUT / WT-MUT. Death and persistence channels are non-negative counts. DIFF channels are signed. Constant aux columns: 2. Per-complex distinct aux rows are in `docs/DATA_NOTES.md`. This matches T0 item 3.

### T4.2

Done on the workstation. N = 645, no complex excluded, no empty file. Copies and checksums: `data/structures/abbind/` and `MANIFEST_abbind.txt`. Kept rows: `data/structures/abbind_keep_rows.txt`.

```
python scripts/validate_abbind_structures.py
```

Secondary RCSB arm: stopped. `data/structures/raw` has no PDB files, and `data/structures/chain_map.csv` is not in the repo. `chain_map_suggestions.csv` was not used.

### T4.3

Pilot already run. Command, for a server reprint:

```
python scripts/reproduce_h0.py --complexes 1KTZ,2JEL,3BDY,3BE1,3HFM --exclude-icode-rows
```

CPU. About two minutes here. Settings follow the capsule: cutoff 12, all altlocs, collapsed insertion codes, deaths quantized like the 200-division JavaPlex stream, 48 bins. WT channels only.

Result: 145 rows, exact integer match 0/145 on every WT block. Best single-row cell agreement 0.616 (mutation death). Continuous MST on 1KTZ alone was also 0/22 exact (best mean cell agreement 0.785 on binding persistence). **Pilot failed.** T4.4, T4.5, T4.6, T4.7 and T4.8 were not run. A later model that used a hand-built H1/H2 vector would be H1/H2-style, not original-equivalent.

Gudhi cross-check stopped. `gudhi` does not import. Do not install it from this session. The command, if you want the 1e-9 MST check later:

```
pip install gudhi
```

MATLAB / JavaPlex were not available. `reproduce_h0.py` is the Python port of the distance and the binning. The quantization formula is the stream arguments (max 15, 200 divisions); the Java rounding is unverified.

### T4.8

Skipped. G0 did not find a block of `X.npy` that matches neither source, because it found no matching block at all.

The names `h12`, `cnn_h12` and `cnn_aux_h12` exist in `run_cv.py` so a later passing pilot can call them. Do not launch them now. `--row-subset data/structures/abbind_keep_rows.txt` would be a no-op at N = 645.

## T5b

Plan only. Not run.

| Item | Status |
|---|---|
| 26 RBD mutations times (ACE2 + 185 complexes), about 4800 mutants | MISSING as structures. The BFE file has 15 rows times 186 columns |
| 185 antibody-RBD structures | MISSING. 184 ids in `antibodies.csv` look like PDB ids; `RN33_REGN10933` and `RN87_REGN10987` do not. The list is the token before the underscore |
| BA.2 / BA.3 / earlier-variant per-mutation predictions | MISSING. Compare only the published aggregates if a model is run later: Fig. 1g sums 2.60 / 2.98 / 2.88, Fig. 3e folds 20.9 / 27.1 / 18.1, Fig. 3a2–c2 counts 163 / 168 / 164 |
| AB-Bind S645 labels and tensors | AVAILABLE |
| SKEMPI-1131 precomputed features in the capsule | AVAILABLE as `code/crossvalidation/X_skempi_feature1.npy`, `X_skempi_feature2.npy`, `Y_skempi.npy` |
| Raw SKEMPI 2.0 | MISSING |
| SARS-CoV-2 DMS training data | MISSING |
| `cnn_model.h5`, `gbdt1.pickle`, `gbdt2.pickle` | AVAILABLE as files. Training set unverified (T0 item 8) |
| Ref. 41 escape fractions | MISSING |
| Ref. 5 serum folds | Only the seven Fig. 3f numbers typed in the T5a prompt |

Pilot to approve before any download or mutant build: fetch 6M0J, build the 19 Omicron RBD mutants (the 15 BA.1 names in `check_paper_arithmetic.py` plus S371F, T376A, D405N, R408S) with the capsule's scap/profix, and run the capsule `predict.py` with the shipped weights. Compare the 15 BA.1 ACE2 predictions to column ACE2 of `antibodies_BFEs.csv` (Pearson, RMSE, sign agreement; the published sum is 2.6018). Compare a BA.2 ACE2 sum (12 shared plus the four BA.2-specific) to 2.98 and a BA.3 sum (12 shared plus S371F, D405N, G446S) to 2.88. A model without the DMS set is expected to agree only partly. Low agreement is informative.

Jackal was not executed. On the server, before any mutant build:

```
file $CAP/data/software/jackal_64bit/bin/scap
ldd $CAP/data/software/jackal_64bit/bin/scap
```

## Provenance

| Number | Where it comes from |
|---|---|
| Nested r 0.467 ± 0.049 | Existing `docs/RESULTS.md`. No-H12, 2000 epochs, 3 seeds. Not re-fit here |
| G0 correlations, 2 constant aux columns | `scripts/check_capsule_alignment.py` on the capsule `X.npy` and the repo arrays |
| Dup-group straddle counts | `check_dup_folds.py`, sklearn 1.9.1 |
| T5a arithmetic | Prompt figures, or `antibodies_BFEs.csv` where the table says so |
| H0 0/145 | `reproduce_h0.py`. Python port, not JavaPlex. Gudhi not used |
| 0.65, 1.57, 0.55, 0.51 | Unverified against the 2020 paper. Not results of this run |
| 756 | `gen_feature_12.py` only |
