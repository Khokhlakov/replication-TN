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
crossvalidation_ab.py lines 14-15 load crosslabel.npy and cast it to int, and both lines are commented out. The script that actually runs does not read this array. It is not a fold id used by the CV loop.
The value pattern is not a 10-fold id and not an obvious two-class label. Meaning: unknown.

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
Columns with |r|>0.99999: 0 / 5888. A value this close to 1 is an affine match (z-scoring), not a second feature family.

If the aux block of X.npy correlates at about 1 with X_aux, the capsule's X.npy/Y.npy pairing independently verifies the X_aux row order.

## d. AB_645.csv versus the supplement

Row-by-row (pdb, mutation, rounded ddg) equality: True.
Permutation is the identity. Insertion codes are inside the mutation string.

## e. Repo arrays versus capsule model arrays

X_aux.npy byte-identical: True.
X_unnormed.npy byte-identical: True.

## Decision

Y.npy equals supplement ddg_abbind and canonical ddg at 645/645 positions, in order. G0 PASS. T1-T5 may proceed. This re-runs the check already done; it does not re-derive the labels.
