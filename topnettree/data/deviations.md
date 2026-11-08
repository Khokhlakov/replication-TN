# Deviations from the paper (edit by hand; scripts/70_data_card.py embeds this file)

Items marked [FILL] depend on a step that has not been run yet; delete the marker once you have the number.

## Training data
- SKEMPI 2.0 single-point set rebuilt from the downloaded CSV: repeated measurements averaged, mutations whose repeats span >= 2.0 kcal/mol dropped.
  Result: 4173 rows / 317 complexes / 895 tighter / 3278 weaker-or-neutral, against the published S4169 (4169 / 319 / 901 / 3268). [FILL: closest option from 38_s4169_report.py]
- Sign: repo ddg is positive = weaker binding; the paper's BFE change is the negative of it. [FILL: sign of the shipped Y_skempi.npy relative to repo ddg, from 34_join_s1131.py]
- SKEMPI rows overlapping AB-Bind S645: 170 identical mutations, 623 rows on shared PDB ids. [FILL: how they are handled in each evaluation]
- Starr 2020 RBD/ACE2 scan: 3802 mutations from single_mut_effects.csv (bind_avg), converted with ddg = -RT ln10 * score. Rows whose residue is not in 6M0J chain E are dropped by 60_merge_labels.py. [FILL: count dropped]
- Chan 2020 ACE2 scan: 2223 substitutions, ddg is a PROXY (ddg = -RT ln2 * mean log2 enrichment of the two High sorts), arbitrary scale; raw scores kept.
- Linsky 2020 (CTC-445.2): NOT available. The supplement PDF has no per-mutation table. [FILL: say whether a data file was found later]
- The SI says the ACE2 and CTC-445.2 scans were held out as test sets for validation (1539 and 1500 samples). [FILL: how this reconstruction uses them]

## Structures and mutation sets
- 186 complexes from data/raw/antibodies.csv. RN33 (REGN10933) and RN87 (REGN10987) resolved to 6XDG, 1061 (AZD1061) to 7L7E (provisional copy, confirm by contacts). [FILL: complex_map_final.csv QC result]
- 5 entries were .cif-only (no legacy PDB file). [FILL: which, and whether chain detection succeeded]
- 11 of the 26 mutation rows (all BA.2-specific and earlier-variant ones except those in the shipped BA.1 file) have no antibody BFE values: they need a trained model on the 185 structures.

## Features and model
- H1/H2 (alpha-complex) features not reproduced; the AB-Bind runs are the no-H12 ablation (nested 10-fold Pearson 0.467 +/- 0.049).
- [FILL: results of 35_compare_features.py for aux and H0 on the S1131 rows]
