#!/usr/bin/env python3
"""Feature-rows table -> build_homology_features manifest, same row order (so reference rows line up).

    python scripts/make_manifest_from_rows.py --rows data/raw/s645_for_features.csv --prefix S645 --out data/features/s645_manifest.csv
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--wild-col", default="structure")
ap.add_argument("--resid-col", default="resid_struct")
ap.add_argument("--keep-col", help="optional column with keep_chains such as H+L+E")
args = ap.parse_args()
r = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
ids = [f"{args.prefix}_{i:04d}_{p}_{c}{w}{n}{u}"
       for i, (p, c, w, n, u) in enumerate(zip(r["pdb"], r["chain"], r["wt"], r[args.resid_col], r["mut"]))]
m = pd.DataFrame({"row_id": ids, "wild_pdb": r[args.wild_col], "mutant_pdb": r["mutant"], "chain": r["chain"],
                  "resid": r[args.resid_col], "wt": r["wt"], "mut": r["mut"], "icode": "", "keep_chains": ""})
if args.keep_col:
    m["keep_chains"] = r[args.keep_col]
if m["row_id"].duplicated().any():
    raise SystemExit("duplicate row_id values")
m.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(m)} rows")