# Deviations from the paper (edit by hand; scripts/70_data_card.py embeds this file)

Items marked [FILL] depend on a step that has not been run yet; delete the marker once you have the number.

## Training data
- SKEMPI 2.0 single-point set rebuilt from the downloaded CSV: repeated measurements averaged, mutations whose repeats span >= 2.0 kcal/mol dropped.
  Result: 4173 rows / 317 complexes / 895 tighter / 3278 weaker-or-neutral, against the published S4169 (4169 / 319 / 901 / 3268). No filter in the sweep reproduces the published counts exactly. Keeping the discordant repeats gives 4187 rows and 319 complexes; the two complexes lost by the 2.0 rule are 3NCB and 3NCC, one mutation each, which are the reverse-mutation structures of two 3MZG mutations (values are exact negatives of the 3MZG ones; the 3MZG repeats span about 3 kcal/mol). Counting ddG == 0 as tighter does not close the gap (141 exact zeros). Kept: mean, 2.0 kcal/mol rule.
- Sign: repo ddg is positive = weaker binding; the paper's BFE change is the negative of it. The shipped Y_skempi.npy has the same sign as repo ddg (positive = weaker): 1126 of 1131 S1131 rows join to SKEMPI 2.0 (cleaned numbering), Pearson 0.995, mean absolute difference 0.068 kcal/mol, 81% within 0.05.
- SKEMPI rows overlapping AB-Bind S645: 170 identical mutations, 623 rows on shared PDB ids. [FILL: how they are handled in each evaluation]
- Starr 2020 RBD/ACE2 scan: 3802 mutations from single_mut_effects.csv (bind_avg), converted with ddg = -RT ln10 * score. Rows whose residue is not in 6M0J chain E are dropped by 60_merge_labels.py. 133 rows dropped (residues 331, 332, 527-531 are not in 6M0J chain E); 3669 kept.
- Chan 2020 ACE2 scan: 2223 substitutions, ddg is a PROXY (ddg = -RT ln2 * mean log2 enrichment of the two High sorts), arbitrary scale; raw scores kept.
- Linsky 2020 (CTC-445.2): NOT available. The supplement PDF has no per-mutation table. The scan covers 75 RBD positions; six of the paper's mutation sites (339, 346, 371, 373, 375, 376) are outside it. The CTC-445.2 structure (7KL9, 75/75 scanned positions present in chain A) is available.
- The SI says the ACE2 and CTC-445.2 scans were held out as test sets for validation (1539 and 1500 samples). [FILL: held-out validation or part of the training mix]

## Structures and mutation sets
- 186 complexes from data/raw/antibodies.csv. RN33 (REGN10933) and RN87 (REGN10987) resolved to 6XDG, 1061 (AZD1061) to 7L7E (provisional copy, confirm by contacts). [FILL: complex_map_final.csv QC result]
- 5 entries were .cif-only (no legacy PDB file); the chain check found at least 10 of 21 RBD sites in every structure. [FILL: which five]
- 11 of the 26 mutation rows (the 7 earlier-variant and the 4 BA.2-specific mutations) have no antibody or ACE2 BFE values: they need a trained model applied to the 185 antibody structures and the ACE2 complex.

## Features and model
- H1/H2 (alpha-complex) features not reproduced; the AB-Bind runs are the no-H12 ablation (nested 10-fold Pearson 0.467 +/- 0.049).
- [FILL: results of 35_compare_features.py for aux and H0 on the S1131 rows]
- Starr vs the shipped ACE2 BFE for the 15 known BA.1 mutations: Pearson 0.51, Spearman 0.44. The ACE2 values the paper pins down for missing mutations (inferred from printed sums, assuming additivity) are far from Starr's measurements, so Starr values are not used to fill any missing row.
