#!/usr/bin/env python3
"""Per structure: heavy ATOM atoms in the wild file but not in the SCAP mutant (and the reverse), ignoring hydrogens,
HETATM, altloc copies and the mutated residue. Also reports atoms that share a (chain, residue, name) key and
therefore get collapsed, so a clean result can be cross-checked against check_report.csv.

    python scripts/explain_atom_diffs.py --manifest data/features/panel_manifest_profix_all.csv --out data/features/panel_atom_diffs.csv
    python scripts/explain_atom_diffs.py --manifest data/features/s645_manifest.csv --group wild --out data/features/s645_atom_diffs.csv
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HYD = re.compile(r"^[0-9]?H")


def read(path):
    seen, out, raw, het = set(), [], 0, 0
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("HETATM"):
            het += 1
            continue
        if not ln.startswith("ATOM") or len(ln) < 54:
            continue
        name = ln[12:16].strip()
        if HYD.match(name):
            continue
        raw += 1
        k = (ln[21], ln[22:26].strip() + ln[26].strip().upper(), name)
        if k in seen:
            continue
        seen.add(k)
        out.append({"k": k, "resname": ln[17:20].strip(), "xyz": (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))})
    return out, raw, het


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--group", choices=["prefix", "wild"], default="prefix",
                help="prefix: complex = text before the first _ in row_id; wild: one group per wild_pdb file")
args = ap.parse_args()
man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
man["cx"] = man["row_id"].str.split("_").str[0] if args.group == "prefix" else man["wild_pdb"]
res = []
for cx, g in man.groupby("cx", sort=True):
    first = g.iloc[0]
    (wild, w_raw, w_het), (mut, m_raw, m_het) = read(first["wild_pdb"]), read(first["mutant_pdb"])
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
        "group": cx, "wild_only_OXT": len(oxt), "wild_only_other": len(other), "mutant_only": len(om),
        "nearest_missing_to_a_site_A": round(near, 1),
        "wild_dups": w_raw - len(wild), "mut_dups": m_raw - len(mut), "wild_hetatm": w_het, "mut_hetatm": m_het,
        "other_resnames": " ".join(f"{c}x{n}" for n, c in Counter(a["resname"] for a in other).most_common(6)),
        "other_chains": " ".join(f"{c}x{n}" for n, c in Counter(a["k"][0] for a in other).most_common(4)),
        "example_atoms": " ".join(f"{a['k'][0]}:{a['resname']}{a['k'][1]}.{a['k'][2]}" for a in other[:4]),
        "wild_pdb": first["wild_pdb"],
    })
df = pd.DataFrame(res)
df.to_csv(args.out, index=False)
clean = int(((df.wild_only_OXT == 0) & (df.wild_only_other == 0) & (df.mutant_only == 0)).sum())
oxt_only = int(((df.wild_only_OXT > 0) & (df.wild_only_other == 0) & (df.mutant_only == 0)).sum())
print(f"{len(df)} groups: {clean} identical, {oxt_only} differ only by OXT, {len(df) - clean - oxt_only} differ in other atoms")
cols = ["group", "wild_only_OXT", "wild_only_other", "mutant_only", "nearest_missing_to_a_site_A", "wild_dups", "mut_dups",
        "wild_hetatm", "mut_hetatm", "other_resnames", "other_chains", "example_atoms"]
show = df[(df.wild_only_other > 0) | (df.mutant_only > 0) | (df.wild_dups > 0) | (df.mut_dups > 0)]
print(show.sort_values(["wild_only_other", "wild_dups"], ascending=False)[cols].head(40).to_string(index=False))