#!/usr/bin/env python3
"""Per complex: atoms in the wild file but not in the SCAP mutant (and the reverse), ignoring hydrogens, HETATM,
altloc copies and the mutated residue. Distances use the sites of every row of the complex.

    python scripts/explain_atom_diffs.py --manifest data/features/panel_manifest_profix_all.csv --out data/features/panel_atom_diffs.csv
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HYD = re.compile(r"^[0-9]?H")


def read(path):
    seen, out = set(), []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not ln.startswith("ATOM") or len(ln) < 54:
            continue
        name = ln[12:16].strip()
        if HYD.match(name):
            continue
        k = (ln[21], ln[22:26].strip() + ln[26].strip().upper(), name)
        if k in seen:
            continue
        seen.add(k)
        out.append({"k": k, "resname": ln[17:20].strip(), "xyz": (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))})
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
man["cx"] = man["row_id"].str.split("_").str[0]
res = []
for cx, g in man.groupby("cx", sort=True):
    first = g.iloc[0]
    wild, mut = read(first["wild_pdb"]), read(first["mutant_pdb"])
    keep = set(re.findall(r"[A-Za-z0-9]", first.get("keep_chains", "") or ""))
    site0 = (first["chain"], (first["resid"] + (first.get("icode", "") or "")).upper())
    sites = {(r["chain"], (r["resid"] + (r.get("icode", "") or "")).upper()) for _, r in g.iterrows()}
    wk, mk = {a["k"] for a in wild}, {a["k"] for a in mut}
    ow = [a for a in wild if a["k"] not in mk and a["k"][:2] != site0 and (not keep or a["k"][0] in keep)]
    om = [a for a in mut if a["k"] not in wk and a["k"][:2] != site0 and (not keep or a["k"][0] in keep)]
    oxt = [a for a in ow if a["k"][2] == "OXT"]
    other = [a for a in ow if a["k"][2] != "OXT"]
    site_xyz = np.asarray([a["xyz"] for a in wild if a["k"][:2] in sites])
    miss = np.asarray([a["xyz"] for a in ow])
    near = float(np.linalg.norm(miss[:, None, :] - site_xyz[None, :, :], axis=2).min()) if len(miss) and len(site_xyz) else float("nan")
    res.append({
        "complex": cx, "wild_only_OXT": len(oxt), "wild_only_other": len(other), "mutant_only": len(om),
        "nearest_missing_to_a_site_A": round(near, 1), "other_residues": len({a["k"][:2] for a in other}),
        "other_resnames": " ".join(f"{c}x{n}" for n, c in Counter(a["resname"] for a in other).most_common(6)),
        "other_chains": " ".join(f"{c}x{n}" for n, c in Counter(a["k"][0] for a in other).most_common(4)),
        "example_atoms": " ".join(f"{a['k'][0]}:{a['resname']}{a['k'][1]}.{a['k'][2]}" for a in other[:4]),
        "wild_pdb": first["wild_pdb"],
    })
df = pd.DataFrame(res)
df.to_csv(args.out, index=False)
clean = int(((df.wild_only_OXT == 0) & (df.wild_only_other == 0) & (df.mutant_only == 0)).sum())
oxt_only = int(((df.wild_only_OXT > 0) & (df.wild_only_other == 0) & (df.mutant_only == 0)).sum())
print(f"{len(df)} complexes: {clean} identical, {oxt_only} differ only by OXT, {len(df) - clean - oxt_only} differ in other atoms")
cols = ["complex", "wild_only_OXT", "wild_only_other", "mutant_only", "nearest_missing_to_a_site_A", "other_resnames", "other_chains", "example_atoms"]
print(df[(df.wild_only_other > 0) | (df.mutant_only > 0)].sort_values("wild_only_other", ascending=False)[cols].to_string(index=False))