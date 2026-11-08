#!/usr/bin/env python3
"""Step 9. The rows the trained model will be asked to predict: 26 mutations x 186 complexes, with everything the feature
generator and the later table-filling step need. Built from complex_map_final.csv (step 3) and mutations_26.csv (step 4).

Per row: structure file, chosen RBD chain, antibody chains, mutation, and
  site_present / wt_ok   is residue <resid> present in the chosen RBD chain, and is it the expected wild-type residue?
  known_shipped          mutation is one of the 15 BA.1 mutations whose BFE is in antibodies_BFEs.csv
  shipped_bfe            that value (paper sign: positive = tighter), NaN for the 11 missing mutations
Rows with wt_ok == 0 cannot be modelled (the paper's Fig. 2 shows such cells in grey) and must stay NaN in the final table.

Two consistency checks are printed, using only data you already have:
  1. On the 15 known mutations, are exact-zero shipped values concentrated where the site is unresolved (wt_ok == 0)?
     If yes, zeros mean "no prediction", the complex/copy choice is confirmed, and zeros must be treated as missing, not as 0.
  2. Sites present but with the WRONG wild-type residue (numbering problem), listed per complex.

    python scripts/62_make_prediction_manifest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import BA1_ROW_ORDER  # noqa: E402
from topnet.sources import read_pdb_residues  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--mutations", default="data/mutation_sets/mutations_26.csv")
    ap.add_argument("--shipped", default="data/raw/antibodies_BFEs.csv")
    ap.add_argument("--out", default="data/predict/rows_26x186.csv")
    args = ap.parse_args()

    cmap = pd.read_csv(args.map, dtype=str, keep_default_na=False)
    cmap["col_index"] = cmap["col_index"].astype(int)
    cmap = cmap[cmap["col_index"] >= 0].sort_values("col_index").reset_index(drop=True)  # drop 'extra' structures (e.g. 7KL9)
    muts = pd.read_csv(args.mutations, dtype={"resid": str})
    bf = pd.read_csv(args.shipped)
    if list(bf.columns) != list(cmap["label"]):
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(bf.columns, cmap["label"])) if a != b][:5]
        sys.exit(f"Column order of {args.shipped} differs from the complex map labels. First differences: {bad}")
    shipped = {(BA1_ROW_ORDER[i], j): float(bf.iloc[i, j]) for i in range(len(BA1_ROW_ORDER)) for j in range(bf.shape[1])}

    rows, wrong = [], {}
    for c in cmap.itertuples():
        path = Path(c.file) if c.file else None
        res = {}
        if path is not None and path.is_file() and c.rbd_chain:
            res = {k.upper(): v for k, v in read_pdb_residues(path).get(c.rbd_chain, {}).items()}
        for m in muts.itertuples():
            aa = res.get(str(m.resid).upper())
            site_present = aa is not None
            wt_ok = aa == m.wt
            if site_present and not wt_ok:
                wrong.setdefault(c.label, []).append(f"{m.mutation}(found {aa})")
            known = m.mutation in BA1_ROW_ORDER
            rows.append({
                "pair_id": f"{c.col_index:03d}_{m.mutation}", "col_index": c.col_index, "label": c.label, "resolved_id": c.resolved_id,
                "file": c.file, "rbd_chain": c.rbd_chain, "ab_chains": c.ab_chains, "qc_pass": c.qc_pass,
                "mutation": m.mutation, "wt": m.wt, "resid": m.resid, "mut": m.mut,
                "site_present": int(site_present), "wt_ok": int(wt_ok), "known_shipped": int(known),
                "shipped_bfe": shipped[(m.mutation, c.col_index)] if known else np.nan,
            })
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    n_c = out["col_index"].nunique()
    print(f"wrote {args.out}: {len(out)} rows = {n_c} complexes x {out['mutation'].nunique()} mutations")
    print(f"modellable rows (wt_ok): {int(out.wt_ok.sum())}; unresolved sites: {int((out.site_present == 0).sum())}; "
          f"wrong wild-type residue: {int(((out.site_present == 1) & (out.wt_ok == 0)).sum())}")
    print(f"rows to predict (missing 11 mutations, wt_ok): {int(((out.known_shipped == 0) & (out.wt_ok == 1)).sum())}")
    print(f"QC-failed complexes still in the manifest: {sorted(out.loc[~out.qc_pass.astype(str).str.lower().eq('true'), 'label'].unique())[:10]}")

    kn = out[out.known_shipped == 1]
    tab = pd.crosstab(kn["wt_ok"].map({1: "site resolved, wt ok", 0: "site unresolved / wrong wt"}),
                      (kn["shipped_bfe"] == 0).map({True: "shipped value == 0", False: "shipped value != 0"}))
    print("\nconsistency check 1 (known mutations): resolved site vs exact-zero shipped value")
    print(tab.to_string())
    if "site unresolved / wrong wt" in tab.index and "shipped value == 0" in tab.columns:
        z_unres = tab.loc["site unresolved / wrong wt", "shipped value == 0"]
        z_all = tab["shipped value == 0"].sum()
        print(f"  {z_unres}/{z_all} exact zeros fall on unresolved sites"
              f" -> {'zeros look like missing predictions' if z_all and z_unres / z_all > 0.8 else 'zeros are mostly NOT explained by missing residues: inspect before treating them as missing'}")
    if wrong:
        print(f"\nconsistency check 2: wrong wild-type residue at the expected site, {len(wrong)} complexes (numbering or chain problem):")
        for label, items in list(wrong.items())[:15]:
            print(f"  {label}: {', '.join(items[:6])}")
    else:
        print("\nconsistency check 2: no wrong-wild-type residues found")


if __name__ == "__main__":
    main()
