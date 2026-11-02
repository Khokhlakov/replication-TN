#!/usr/bin/env python3
"""
build_labels_from_supplement.py

Builds the S645 table directly from the AB-Bind supplementary workbook
(PRO-25-393-s002.xlsx, sheet 'AB-Bind_experimental_data.csv') and, optionally,
compares it with the label table your pipeline is already using.

S645 = every row of the workbook whose 'Mutation' field has exactly one mutation
       (no comma).  Verified on the supplied file: 645 rows, 29 PDB groups
       (24 crystal structures + 5 homology models named HM_<pdb>), 27 rows with
       ddG = 8 (non-binders), 87 rows on homology models.

Columns written
  row_idx          position in the workbook's own order (0..644)
  pdb              workbook '#PDB' (e.g. 1AK4 or HM_1KTZ)
  parent_pdb       pdb without the HM_ prefix (the template entry on RCSB)
  is_homology_model
  chain, wt, resid (int), icode (PDB insertion code, upper case), mut
  resid_str        resid + icode, e.g. '52A'
  ddg_abbind       as published (positive = binding WEAKENED)
  ddg              -ddg_abbind  (positive = binding STRENGTHENED, AGENTS.md convention;
                   non-binders therefore -8)
  is_nonbinder     ddg_abbind == 8
  complex_id       0..28, one per '#PDB' group (HM_ groups are their own complex)
  partners, assay, dup_group

Usage
  python scripts/build_labels_from_supplement.py \
      --xlsx PRO-25-393-s002.xlsx --out data/labels/ab_bind_s645_supplement.csv
  python scripts/build_labels_from_supplement.py --xlsx ... --compare data/labels/ab_bind_s645.csv
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

MUT_RE = re.compile(r"^([A-Za-z0-9]):([A-Z])(-?\d+)([A-Za-z]?)([A-Z])$")

# Supplementary Table S3 (homology models). Chain names after ':' are the template's PDB chains.
HM_TEMPLATES = {
    "HM_1YY9": {"target": "Hu-225", "template": "Erbitux Fab (1YY9)", "template_chains": "H<-D, L<-C", "identity": "H:81%, L:80%"},
    "HM_3BN9": {"target": "Epithin", "template": "MT-SP1 (3BN9)", "template_chains": "A<-A", "identity": "87%"},
    "HM_2NYY": {"target": "BoNT/A2", "template": "BoNT/A1 (2NYY)", "template_chains": "A<-A", "identity": "90%"},
    "HM_2NZ9": {"target": "BoNT/A2", "template": "BoNT/A1 (2NZ9)", "template_chains": "A<-A", "identity": "90%"},
    "HM_1KTZ": {"target": "TGF-beta", "template": "TGF-beta (1KTZ)", "template_chains": "A<-A", "identity": "76%"},
}


def load_s645(xlsx):
    raw = pd.read_excel(xlsx, sheet_name=0)
    need = ["#PDB", "Partners(A_B)", "Mutation", "ddG(kcal/mol)"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        sys.exit(f"ERROR: workbook lacks columns {miss}; found {list(raw.columns)}")
    print(f"[xlsx] {len(raw)} rows total, {raw['#PDB'].nunique()} PDB groups")
    s = raw[~raw["Mutation"].astype(str).str.contains(",")].copy().reset_index(drop=True)
    m = s["Mutation"].astype(str).str.strip().str.extract(MUT_RE)
    if m.isna().any().any():
        bad = s.loc[m.isna().any(axis=1), "Mutation"].tolist()
        sys.exit(f"ERROR: unparsable single mutations: {bad[:10]}")
    m.columns = ["chain", "wt", "resid", "icode", "mut"]
    out = pd.DataFrame({
        "row_idx": np.arange(len(s)),
        "pdb": s["#PDB"].astype(str).str.strip(),
    })
    out["parent_pdb"] = out["pdb"].str.replace(r"^HM_", "", regex=True)
    out["is_homology_model"] = out["pdb"].str.startswith("HM_")
    out["chain"] = m["chain"]
    out["wt"] = m["wt"]
    out["resid"] = m["resid"].astype(int)
    out["icode"] = m["icode"].str.upper()
    out["mut"] = m["mut"]
    out["resid_str"] = out["resid"].astype(str) + out["icode"]
    out["ddg_abbind"] = s["ddG(kcal/mol)"].astype(float)
    out["ddg"] = -out["ddg_abbind"]
    out["is_nonbinder"] = out["ddg_abbind"] >= 8.0
    groups = list(dict.fromkeys(out["pdb"]))
    out["complex_id"] = out["pdb"].map({g: i for i, g in enumerate(groups)})
    out["partners"] = s["Partners(A_B)"].astype(str)
    out["assay"] = s["Assay Name"].astype(str) if "Assay Name" in s.columns else ""
    key = out["pdb"] + "|" + out["chain"] + out["wt"] + out["resid_str"] + out["mut"]
    out["dup_group"] = np.where(key.duplicated(keep=False), key.factorize()[0], -1)
    return out


def report(t):
    print(f"[S645] rows={len(t)}  complexes={t['pdb'].nunique()}  "
          f"homology-model rows={int(t['is_homology_model'].sum())}  "
          f"non-binders={int(t['is_nonbinder'].sum())}")
    print(f"[S645] crystal-structure PDB IDs to download "
          f"({t.loc[~t.is_homology_model, 'pdb'].nunique()}): "
          f"{sorted(t.loc[~t.is_homology_model, 'pdb'].unique())}")
    print("[S645] homology models (not on RCSB):")
    for hm, g in t[t.is_homology_model].groupby("pdb"):
        info = HM_TEMPLATES.get(hm, {})
        print(f"   {hm}: {len(g)} rows, chains {sorted(g.chain.unique())}, "
              f"resid {g.resid.min()}..{g.resid.max()}, template: {info.get('template', '?')} "
              f"[{info.get('template_chains', '?')}], identity {info.get('identity', '?')}")
    n_ic = int((t.icode != "").sum())
    print(f"[S645] rows with an insertion code: {n_ic} (e.g. {t.loc[t.icode != '', 'resid_str'].head(5).tolist()})")
    d = t[t.dup_group >= 0]
    if len(d):
        print(f"[S645] {len(d)} rows belong to {d.dup_group.nunique()} duplicated (pdb, mutation) groups "
              f"(same mutation listed more than once, different assays/values):")
        print(d[["row_idx", "pdb", "chain", "wt", "resid_str", "mut", "ddg_abbind", "assay"]].to_string(index=False))
    print(f"[S645] adjacent rows sharing a site: {int(((t.pdb + t.chain + t.resid_str) == (t.pdb + t.chain + t.resid_str).shift()).sum())} "
          "(informational; the 36 identical X_aux pairs in AGENTS.md need not correspond to this)")


def compare(new, path):
    """Compare against the label table the pipeline currently uses."""
    old = pd.read_csv(path)
    print(f"\n[compare] existing table: {path}  rows={len(old)}  columns={list(old.columns)}")

    def col(df, pats):
        for p in pats:
            for c in df.columns:
                if re.search(p, str(c).lower()):
                    return c
        return None

    c = {k: col(old, v) for k, v in {
        "pdb": [r"^pdb"], "chain": [r"^chain"], "wt": [r"^wt$", r"^wt", r"wild"],
        "resid": [r"^resid", r"resnum", r"^resi", r"^pos"], "mut": [r"^mut"],
        "ddg": [r"^ddg$", r"^ddg", r"^y$", r"bfe", r"label"]}.items()}
    print(f"[compare] detected columns: {c}")
    if any(v is None for k, v in c.items() if k != "mut"):
        print("[compare] cannot map the columns automatically; pass a table with pdb/chain/wt/resid/mut/ddg columns.")
        return
    o = pd.DataFrame({
        "pdb": old[c["pdb"]].astype(str).str.strip(),
        "chain": old[c["chain"]].astype(str).str.strip(),
        "wt": old[c["wt"]].astype(str).str.strip().str.upper().str[:1],
        "resid": old[c["resid"]].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True),
        "mut": old[c["mut"]].astype(str).str.strip().str.upper().str[:1] if c["mut"] else "",
        "ddg": pd.to_numeric(old[c["ddg"]], errors="coerce"),
    })
    n = new.assign(resid=new["resid_str"])
    key = lambda d: d["pdb"] + "|" + d["chain"] + "|" + d["wt"] + d["resid"] + d["mut"]  # noqa: E731
    ko, kn = key(o), key(n)
    print(f"[compare] same row count: {len(o) == len(n)}  ({len(o)} vs {len(n)})")
    only_old, only_new = set(ko) - set(kn), set(kn) - set(ko)
    print(f"[compare] mutations only in existing table: {len(only_old)}   only in supplement: {len(only_new)}")
    if only_old:
        print("   e.g. existing-only:", sorted(only_old)[:6])
    if only_new:
        print("   e.g. supplement-only:", sorted(only_new)[:6])
    if len(o) == len(n):
        same_order = bool((ko.values == kn.values).all())
        print(f"[compare] identical row ORDER with the workbook: {same_order}")
        if not same_order:
            first = int(np.argmax(ko.values != kn.values))
            print(f"   first difference at row {first}: existing={ko.iloc[first]!r} vs workbook={kn.iloc[first]!r}")
    # sign / value check on the common keys (handles duplicates by position within key)
    o2 = o.assign(k=ko, rep=ko.groupby(ko).cumcount())
    n2 = n.assign(k=kn, rep=kn.groupby(kn).cumcount())
    j = o2.merge(n2[["k", "rep", "ddg", "ddg_abbind"]], on=["k", "rep"], suffixes=("_old", "_new"))
    if len(j):
        same = np.isclose(j["ddg_old"], j["ddg_new"], atol=1e-6)
        flip = np.isclose(j["ddg_old"], -j["ddg_new"], atol=1e-6)
        print(f"[compare] on {len(j)} matched mutations: same sign/value {int(same.sum())}, "
              f"sign flipped {int((flip & ~same).sum())}, other {int((~same & ~flip).sum())}")
        if flip.sum() > same.sum():
            print("   -> the existing table uses the AB-Bind convention (positive = weakened). "
                  "AGENTS.md expects positive = strengthened, non-binders -8.")
        nb = j.loc[np.isclose(j["ddg_abbind"], 8.0), "ddg_old"]
        print(f"[compare] non-binder labels in existing table: {sorted(set(np.round(nb, 3)))[:5]} (n={len(nb)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default="PRO-25-393-s002.xlsx")
    ap.add_argument("--out", default="data/labels/ab_bind_s645_supplement.csv")
    ap.add_argument("--compare", default=None, help="existing label table to check against")
    a = ap.parse_args()
    if not Path(a.xlsx).exists():
        sys.exit(f"ERROR: {a.xlsx} not found")
    t = load_s645(a.xlsx)
    assert len(t) == 645, f"expected 645 single-point rows, got {len(t)}"
    assert t["pdb"].nunique() == 29, "expected 29 complexes"
    assert int(t["is_nonbinder"].sum()) == 27, "expected 27 non-binders"
    assert int(t["is_homology_model"].sum()) == 87, "expected 87 homology-model rows"
    report(t)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    print(f"[out] wrote {a.out}  (row order = workbook order; NOT yet verified against X_unnormed.npy)")
    if a.compare:
        compare(t, a.compare)


if __name__ == "__main__":
    main()
