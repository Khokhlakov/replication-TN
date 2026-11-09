# Data card

Generated 2026-11-09 20:49 UTC from `data`; 1176 files hashed (see MANIFEST.sha256; verify with `sha256sum -c`).

## Files

| file | bytes | rows | cols | sha256[:12] |
|---|---|---|---|---|
| labels/ab_bind_s645.csv | 27206 | 645 | 12 | 1272e63f3686 |
| labels/ab_bind_s645_supplement.csv | 44209 | 645 | 17 | 6bb2e094977a |
| labels/reconstructed_labels.csv | 919015 | 10065 | 12 | 29bc383cb9b2 |
| labels/reconstructed_labels_dropped.csv | 12840 | 133 | 12 | cbb63378eb92 |
| mutation_sets/antibody_bfe_26x186.csv | 24308 | 26 | 187 | 2fbbbbca682d |
| mutation_sets/mutations_26.csv | 1598 | 26 | 18 | 365c662b3900 |
| mutation_sets/starr_vs_shipped_ace2.csv | 810 | 26 | 3 | 83f22756564b |
| mutation_sets/variant_sets.csv | 509 | 11 | 3 | dff9ccc4a3db |
| predict/rows_26x186.csv | 493514 | 4836 | 16 | a4940d34b9c6 |
| raw/X_unnormed.npy | 26749568 |  |  | e0c3a2546330 |
| raw/antibodies.csv | 2581 | 185 | 2 | 74b21b815c2f |
| raw/antibodies_BFEs.csv | 22097 | 15 | 186 | 21b1ceec35c4 |
| raw/antibody_aliases.csv | 391 | 3 | 5 | 6c2addbf2036 |
| raw/complex_extra.csv | 100 | 1 | 5 | ce0a864acd90 |
| raw/complex_overrides.csv | 3945 | 34 | 5 | a4759f9057fc |
| raw/dms/chan2020_labels.csv | 265229 | 2223 | 16 | 5661ddf0afa6 |
| raw/dms/chan2020_tidy.csv | 294605 | 2223 | 15 | f60a4f3ae818 |
| raw/dms/starr2020_rbd_ace2_labels.csv | 331701 | 3802 | 14 | 6cfd01d68ec0 |
| raw/skempi/s1131_joined.csv | 48248 | 1126 | 8 | 540a3bc42113 |
| raw/skempi/s1131_unjoined.csv | 171 | 5 | 8 | 6f9e0467ed59 |
| raw/skempi/skempi_v2.csv | 1602208 | 7085 | 29 | 76f8e60d09ea |
| raw/skempi/skempi_v2_entries.csv | 862309 | 7085 | 14 | 0821068d70f8 |
| raw/skempi/skempi_v2_single_labels.csv | 568950 | 4173 | 23 | c97a88bc26c9 |
| splits/labels_with_folds.csv | 1284981 | 10065 | 23 | b73ef0c99852 |
| structures/antibody_rbd/7L7E.pdb | 5490342 |  |  | 23176313dba9 |
| structures/antibody_rbd/complex_chain_review.csv | 17300 | 187 | 7 | 7080f4f2fde9 |
| structures/antibody_rbd/complex_map.csv | 10366 | 186 | 4 | 1091baf5b586 |
| structures/antibody_rbd/complex_map_final.csv | 22455 | 187 | 13 | 4a3e908ad04f |
| structures/antibody_rbd/manifest.csv | 122584 | 186 | 15 | 314e1e4fcc7c |
| structures/chain_map_suggestions.csv | 1890 | 45 | 10 | 4f35af101db3 |
| structures/download_log.csv | 2147 | 24 | 5 | a5f49a266d8d |
| structures/validation_rows.csv | 16946 | 645 | 7 | e1184e5b97ce |
| structures/validation_rows_mapped.csv | 16234 | 645 | 7 | b04d4cdf9cc8 |
| structures/validation_summary.csv | 863 | 29 | 10 | 886773f6ee99 |
| structures/validation_summary_mapped.csv | 820 | 29 | 9 | 852c54cc80c9 |

## File counts per folder

- `.`: 1
- `labels`: 4
- `mutation_sets`: 6
- `predict`: 1
- `raw`: 9
- `raw/dms`: 3
- `raw/dms/chan2020`: 1
- `raw/rcsb_entities`: 184
- `raw/skempi`: 7
- `raw/skempi/mappings`: 345
- `raw/skempi/pdbs`: 345
- `splits`: 1
- `structures`: 9
- `structures/abbind`: 30
- `structures/antibody_rbd`: 187
- `structures/antibody_rbd_converted`: 10
- `structures/ctc445`: 2
- `structures/homology`: 5
- `structures/raw`: 24
- `supplement`: 2

## Machine-written reports

### raw/skempi/numbering_check.json
```json
{
  "n_checked": 4173,
  "wt_match_rate": {
    "clean": 1.0,
    "pdb": 0.38126048406422236
  },
  "pdbs_missing": [],
  "n_pdbs_missing": 0
}
```
### raw/skempi/s1131_join_report.json
```json
{
  "mapping": {
    "pdb": 0,
    "chain": 1,
    "wt": 2,
    "resid": 3,
    "mutaa": 4
  },
  "best_scheme": "cleaned",
  "n_s1131": 1131,
  "n_parsed": 1131,
  "n_joined": 1126,
  "join_rate": 0.995579133510168,
  "pearson_y_ddg": 0.9947962589997124,
  "pearson_negated": -0.9947962589997124,
  "y_sign_vs_repo_ddg": "same (positive = weaker)",
  "spearman_y_ddg": 0.992121396627547,
  "abs_pearson": 0.9947962589997124,
  "mean_abs_diff_after_sign": 0.06787880696939665,
  "frac_abs_diff_lt_0.05_after_sign": 0.8134991119005328,
  "gate_pass": true
}
```
### mutation_sets/target_report.json
```json
[
  {
    "variant": "BA.1",
    "n_mut": 15,
    "missing": [],
    "counts": [
      2,
      5,
      15,
      33,
      60,
      78,
      106,
      124,
      143,
      163,
      185
    ],
    "fold": 20.945,
    "complete": true,
    "ace2_sum": 2.6018,
    "counts_match": true,
    "counts_max_abs_err": 0,
    "fold_ok": true,
    "ace2_ok": true
  },
  {
    "variant": "BA.2",
    "n_mut": 16,
    "missing": [
      "S371F",
      "T376A",
      "D405N",
      "R408S"
    ],
    "counts": [
      2,
      5,
      15,
      33,
      60,
      78,
      106,
      122,
      138,
      161,
      185
    ],
    "fold": 15.044,
    "complete": false,
    "ace2_sum": 2.4115,
    "counts_bound_ok": true,
    "counts_still_needed": [
      0,
      0,
      1,
      2,
      5,
      7,
      9,
      10,
      12,
      7,
      0
    ],
    "fold_bound_ok": true,
    "mean_negative_still_needed": -0.589
  },
  {
    "variant": "BA.3",
    "n_mut": 15,
    "missing": [
      "S371F",
      "D405N"
    ],
    "counts": [
      2,
      5,
      15,
      33,
      60,
      78,
      106,
      123,
      140,
      162,
      185
    ],
    "fold": 15.787,
    "complete": false,
    "ace2_sum": 2.4161,
    "counts_bound_ok": true,
    "counts_still_needed": [
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      2,
      0
    ],
    "fold_bound_ok": true,
    "mean_negative_still_needed": -0.137
  },
  {
    "variant": "Delta",
    "n_mut": 2,
    "missing": [
      "L452R"
    ],
    "counts": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      3,
      52
    ],
    "fold": 1.036,
    "complete": false,
    "counts_bound_ok": true,
    "counts_still_needed": [
      0,
      0,
      0,
      5,
      10,
      16,
      20,
      31,
      47,
      67,
      89
    ],
    "fold_bound_ok": true,
    "mean_negative_still_needed": -0.37
  },
  {
    "variant": "Alpha",
    "n_mut": 1,
    "missing": [],
    "counts": [
      0,
      0,
      1,
      2,
      7,
      11,
      13,
      23,
      27,
      38,
      82
    ],
    "fold": 1.292,
    "complete": true,
    "fold_ok": true
  },
  {
    "variant": "Beta",
    "n_mut": 3,
    "missing": [
      "E484K"
    ],
    "counts": [
      0,
      1,
      10,
      19,
      34,
      44,
      56,
      66,
      71,
      85,
      132
    ],
    "fold": 2.513,
    "complete": false,
    "fold_bound_ok": true,
    "mean_negative_still_needed": -0.514
  },
  {
    "variant": "Gamma",
    "n_mut": 3,
    "missing": [
      "K417T",
      "E484K"
    ],
    "counts": [
      0,
      0,
      1,
      2,
      7,
      11,
      13,
      23,
      27,
      38,
      82
    ],
    "fold": 1.292,
    "complete": false,
    "fold_bound_ok": true,
    "mean_negative_still_needed": -1.202
  },
  {
    "variant": "Theta",
    "n_mut": 3,
    "missing": [
      "K417T",
      "E484K"
    ],
    "counts": [
      0,
```
### mutation_sets/ace2_constraints.json
```json
{
  "known": {
    "shared12_sum": 2.4114999999999998,
    "G446S": 0.0046,
    "T478K": 0.9994,
    "BA1_sum": 2.6018
  },
  "intervals_kcal_per_mol": {
    "S371F+T376A+D405N+R408S": [
      0.5635000000000003,
      0.5735000000000001
    ],
    "S371F+D405N": [
      0.45890000000000025,
      0.46890000000000004
    ],
    "T376A+R408S": [
      0.0946000000000003,
      0.11459999999999987
    ],
    "L452R": [
      0.5286810170636748,
      0.562491665757393
    ]
  },
  "consistency_checks": {
    "infectivity_BA2_vs_original": [
      19.589623249595995,
      19.786502236370332,
      "paper: about 20"
    ],
    "infectivity_BA2_vs_BA1": [
      1.4523747857443652,
      1.4669713950099283,
      "paper: about 1.5"
    ]
  },
  "note": "Intervals assume additivity of per-mutation ACE2 BFE (the paper's definition of accumulated BFE) and are widened only by two-decimal rounding of the printed sums and the ratio 4.2."
}
```
### complex_map_final.csv
- rows 187, QC pass 187, rules {'clean_copy': 150, 'override': 37}

## Deviations from the paper

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
- 186 complexes from data/raw/antibodies.csv. RN33 (REGN10933) and RN87 (REGN10987) resolved to 6XDG, 1061 (AZD1061) to 7L7E (provisional copy, confirm by contacts). Antibody chains are defined by contact: chains with >= 5 heavy-atom pairs within 4.5 A of the chosen RBD copy; a partner chain touching several RBD copies is assigned to the copy it touches most (37b_resolve_shared_partners.py, rows kept in data/raw/complex_overrides.csv). AZD1061 uses 7L7E RBD copy S with Fab chains M+N. QC thresholds relaxed to --weak-pairs 1 and --min-total 40. [FILL: final QC pass count and the rows with ambiguous chains]
- 5 entries were .cif-only (no legacy PDB file); the chain check found at least 10 of 21 RBD sites in every structure. 6ZBP (H11-H4), 7B27 (NM1230), 7OAY (Nanobody F2), 7OAP (Nanobody H3 C1), 7OAO (Nanobody C5).
- 11 of the 26 mutation rows (the 7 earlier-variant and the 4 BA.2-specific mutations) have no antibody or ACE2 BFE values: they need a trained model applied to the 185 antibody structures and the ACE2 complex.

## Features and model
- H1/H2 (alpha-complex) features not reproduced; the AB-Bind runs are the no-H12 ablation (nested 10-fold Pearson 0.467 +/- 0.049).
- [FILL: results of 35_compare_features.py for aux and H0 on the S1131 rows]
- Starr vs the shipped ACE2 BFE for the 15 known BA.1 mutations: Pearson 0.51, Spearman 0.44. The ACE2 values the paper pins down for missing mutations (inferred from printed sums, assuming additivity) are far from Starr's measurements, so Starr values are not used to fill any missing row.
- Antibody chains per complex: chosen by RBD-copy contact (heavy-atom pairs within 4.5 A), then checked against RCSB entity descriptions. A Fab keeps its heavy and light chain when the light chain touches the RBD with at least 1 pair; a Fab chain with no RBD contact is excluded (7MZM PDI215 and 7BNV ION-300 keep the heavy chain only; 7JWB is a VH-only binder). Spike chains that the geometry picked as partners were removed by hand (7LSS, 7LAA). Hand-set rows: 6XDG cocktail, 7L7E (copy S), 7LSS, 7LAA, 7BYR, 7MF1, 7D0B, 7D0D, 7B27. Rows set from 37d proposals: 7C01 (B: C+D), 6ZER (D: F+G), 7K8X (B: D+F), 7NP1 (A: H+L). [FILL: 7RKU]
- Bridging antibodies (7K90 C144, 6XKP CV07-270, 7K43 S2M11, 7KM5 Nanosota-1) touch two RBD copies; only the chain set on the chosen copy is used, so the second contact is not represented.
- 6Z2M (CR3022/H11-D4): the chosen RBD copy carries nanobody D4 (chain D) and one CR3022 Fab (B+C); the other CR3022 Fab (H+L) sits on the second RBD copy.
- Five entries were converted from .cif to .pdb; 7B27 notes that the converted PDB has shifted residue numbering. [FILL: wild-type agreement at the 21 mutation sites for 6ZBP, 7B27, 7OAY, 7OAP, 7OAO]
- 7RKU C022: RBD B carries Fab J+K; the light chain L of a second Fab (H+L) touches B with 157 pairs but its heavy chain touches only RBD C, so that Fab is not assigned to B. RBD A is the mirror case (G+I, with light chain F of E+F on A).

