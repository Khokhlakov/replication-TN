# G0 alignment

Verdict: **PASS**

Y.npy equals supplement ddg_abbind and canonical ddg at 645/645 positions, in order. G0 PASS. T1-T5 may proceed. This re-runs the check already done; it does not re-derive the labels.

## a. Y.npy versus ddg_abbind

Equal positions: 645 / 645. allclose to canonical ddg: True.
Provenance: capsule `code/crossvalidation/Y.npy`, decoded from byte strings and cast to float.

## b. crosslabel.npy

shape (645,) dtype float64
640 unique values, min 0, max 644.
Values that occur more than once: 0 x2, 80 x2, 124 x2, 230 x2, 239 x2
Integers missing between 0 and the max: [79, 229, 238, 565, 638]
Equals 0..644: False.
Three of the doubled values are the second row of a known duplicate pair, and the missing integers include that pair's first row: 79 missing and 80 doubled (1DVF rows 79/80), 229 missing and 230 doubled, 238 missing and 239 doubled. That is only a partial account. 0 and 124 are also doubled, and 565 and 638 are also missing. Meaning: unknown. It is not a fold id.
crossvalidation_ab.py lines 14-15 load crosslabel.npy and cast it to int, and both lines are commented out. The script that actually runs does not read this array.

## c. X.npy columns

X.npy shape (645, 5888). Constant columns: 0. Column std min 0.0244, max 0.0800 (a z-scored column would have std 1).
X_aux constant columns (2): [669, 676].
X_unnormed flattened constant columns: 460 of 5184.
Exact column identity was already 0. This section is correlation after dropping constant source columns.
Best |Pearson r| summary: max 0.6721, median 0.1925, 95th percentile 0.3358.
Best |Pearson r| of each X.npy column against the remaining sources: |r|>0.99999: 0; >0.999: 0; >0.99: 0; <0.9: 5888
Best |Spearman rho| summary: max 0.3514, median 0.1543.
Best |Spearman rho|: |r|>0.99999: 0; >0.999: 0; >0.99: 0; <0.9: 5888
Columns with |Spearman|>0.999 and |Pearson|<=0.99 (monotone but not linear): 0.
Assigned to aux: 547. Assigned to flattened X_unnormed: 5341.
Every X.npy column has best |r| < 0.9 against both sources. That rejects a column-wise affine copy of X_aux or X_unnormed, including a z-scored concatenation. It is not a located H1/H2 block: no source block matched, so there is no remainder to cut out. T4.8 is skipped.
Aux assignment unique: False (208 distinct / 547 columns, source width 705).
CNN assignment unique: False (1321 distinct / 5341 columns, source width 4724).
Aux-assigned X columns, in X order: permuted.
CNN-assigned X columns, in X order: permuted.
CNN-assigned X index range 0:5887. Aux-assigned X index range 5:5883.
importanceidx.npy shape (5888,). crossvalidation_ab.py:16-20 applies idx[:300] to all 645 rows before KFold. That selection sees every row. The script that computed the ranking is not in the capsule, so 'fit on all 645 rows' is the implication of one global index, not a logged training call. Using it for feature selection is a leak either way.
Columns with |r|>0.99999: 0 / 5888. None is an affine match of a source column.

The aux block does not correlate at about 1 with X_aux (max |r| 0.672). X.npy therefore does not independently verify the X_aux row order. Y.npy still matches `ddg_abbind` in order, and that is the G0 pass. The expected 3 constant aux columns were not found: there are 2 (columns 669 and 676). Dropping those does not make 5888 = 5184 + 704.

## d. AB_645.csv versus the supplement

Row-by-row equality after case-folding `(pdb, mutation, ddg)`: True. Permutation is the identity.
Before case-folding, 22 rows did not match. That is the insertion-code case (`100a` in the canonical `resid` versus `100A` in AB_645 / `resid_str`). The site and the value are the same.

## e. Repo arrays versus capsule model arrays

X_aux.npy byte-identical: True.
X_unnormed.npy byte-identical: True.

## Decision

Y.npy equals supplement ddg_abbind and canonical ddg at 645/645 positions, in order. G0 PASS. T1-T5 may proceed. This re-runs the check already done; it does not re-derive the labels.
