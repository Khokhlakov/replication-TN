# Deviations from the paper (edit by hand; scripts/70_data_card.py embeds this file)

Items marked [FILL] depend on a step that has not been run yet; delete the marker once you have the number.

## Training data
- SKEMPI 2.0 single-point set rebuilt from the downloaded CSV: repeated measurements averaged, mutations whose repeats span >= 2.0 kcal/mol dropped.
  Result: 4173 rows / 317 complexes / 895 tighter / 3278 weaker-or-neutral, against the published S4169 (4169 / 319 / 901 / 3268).
  No filter in the sweep (mean or median; no discard, >= 2.0, >= 1.0; no averaging) reproduces the published counts. Keeping the discordant
  repeats gives 4187 rows, 319 complexes, 897 tighter, 3290 weaker. The two complexes lost by the 2.0 rule are 3NCB and 3NCC (one mutation each),
  the reverse-mutation structures of two 3MZG mutations (ddG exactly negated; the 3MZG repeats span about 3 kcal/mol).
  Tested and rejected: counting the 141 exact-zero ddG values as tighter (gives 1038 tighter, not 901). The remaining gap is 4 rows (0.1%) and unexplained.
  Kept: mean, 2.0 kcal/mol rule. Leakage note: reverse-mutation PDBs are merged into one group in data/splits (61_make_splits.py).
- Sign: repo ddg is positive = weaker binding; the paper's BFE change is the negative of it. The shipped Y_skempi.npy has the same sign as repo ddg
  (positive = weaker): 1126 of 1131 S1131 rows join to SKEMPI 2.0 (cleaned numbering), Pearson 0.995, mean absolute difference 0.068 kcal/mol,
  81% within 0.05. The 5 unjoined rows are listed in data/raw/skempi/s1131_unjoined.csv. [FILL: what they are]
- SKEMPI rows overlapping AB-Bind S645: 170 identical mutations, 623 rows on shared PDB ids. Handling (data/splits/labels_with_folds.csv):
  any model evaluated on AB-Bind S645 is trained with excl_for_s645_eval == 0 only (all SKEMPI rows on S645 PDBs removed); SKEMPI-internal CV keeps them.
- Starr 2020 RBD/ACE2 scan: 3802 mutations from single_mut_effects.csv (bind_avg), converted with ddg = -RT ln10 * score. 133 rows dropped by
  60_merge_labels.py (residues 331, 332, 527-531 are not in 6M0J chain E); 3669 kept. Wild-type agreement before dropping: 0.965.
- Chan 2020 ACE2 scan: 2223 substitutions, ddg is a PROXY (ddg = -RT ln2 * mean log2 enrichment of the two High sorts), arbitrary scale; raw scores kept.
  Wild-type agreement with 6M0J chain A: 1.000. Values are censored at -5 (flagged in chan2020_labels.csv).
- Linsky 2020 (CTC-445.2): NOT available. The supplement PDF has no per-mutation table (enrichments appear only in plots). The scan covers 75 RBD
  positions; six of the paper's mutation sites (339, 346, 371, 373, 375, 376) are outside it. The CTC-445.2 structure is PDB 7KL9
  ("S 6P trimer in complex with the ACE2 protein decoy, CTC-445.2, state 4"); all 75 scanned positions are present in RBD chain A.
  RBD A pairs with CTC-445.2 chain F (141 atom pairs, F touches no other RBD copy); B-E (218) and C-D (188) also exist but E also touches C.
- Validation protocol: the SI says the ACE2 and CTC-445.2 scans were held out as test sets (1539 and 1500 samples); the main text says they are used
  in training. Both protocols are defined in data/splits (column protocol_A; protocol B = all rows). A DMS number from a model that trained on DMS is not validation.

## Structures and mutation sets
- 186 complexes from data/raw/antibodies.csv. RN33 (REGN10933) and RN87 (REGN10987) resolved to 6XDG, 1061 (AZD1061) to 7L7E.
  Chain assignments are by contact analysis: 6XDG RBD = E, REGN10933 = B+D, REGN10987 = C+A (A has < 5 contact pairs and is set by override);
  7L7E RBD copy K (no shared partners), AZD8895 = I+J, AZD1061 = O+P. 7B27: nanobody = C (B and D do not touch the RBD), only valid on the shifted PDB.
  [FILL: complex_map_final.csv QC result from 37_freeze_complex_map.py]
- 5 entries were .cif-only (no legacy PDB file) and were converted with 32_cif_to_pdb.py; every converted file auto-detects an RBD chain. [FILL: which five]
- 11 of the 26 mutation rows (the 7 earlier-variant and the 4 BA.2-specific mutations) have no antibody or ACE2 BFE values: they need a trained model
  applied to the 185 antibody structures and the ACE2 complex (data/predict/rows_26x186.csv lists the rows). [FILL: modellable rows and the
  exact-zero check from 62_make_prediction_manifest.py]

## Features and model
- The capsule's aux generator (gen_feature_aux: PrepareStructure.py, Run_MIBPB.py, Construct_Feature.py, MutationModule.py) is Python 2 and needs profix,
  scap, MIBPB and MS_Intersection binaries; the repo's src/topnet/features/aux.py is a stub that refuses to invent features. Features exist only for
  AB-Bind S645 (X_unnormed, X_aux) and S1131 (X_skempi_feature1/2). [FILL: result of the generator smoke test and of 35_compare_features.py]
- X_skempi_feature1 has 324 columns; what it contains is not yet identified. [FILL]
- H1/H2 (alpha-complex) features not reproduced; the AB-Bind runs are the no-H12 ablation (nested 10-fold Pearson 0.467 +/- 0.049).
- Starr vs the shipped ACE2 BFE for the 15 known BA.1 mutations: Pearson 0.51, Spearman 0.44. The ACE2 values the paper pins down for missing
  mutations (inferred from printed sums, assuming additivity) are far from Starr's measurements, so Starr values are not used to fill any missing row.
