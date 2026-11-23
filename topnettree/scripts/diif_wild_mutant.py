#!/usr/bin/env python3
"""Why do wild and mutant differ outside the site? Lists atoms found in only one file, by residue.

    python scripts/diff_wild_mutant.py --manifest data/features/panel_manifest_profix_all.csv --row 000_D405N
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HYD = re.compile(r"^[0-9]?H")


def read(path):
    atoms = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        atoms.append({"rec": rec, "name": line[12:16].strip(), "alt": line[16].strip(), "resname": line[17:20].strip(),
                      "chain": line[21], "res": line[22:26].strip() + line[26].strip(),
                      "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54]))})
    return atoms


def key(a):
    return (a["chain"], a["res"], a["name"])


def only_in(a, b):
    left = Counter(key(x) for x in b)
    out = []
    for x in a:
        if left[key(x)] > 0:
            left[key(x)] -= 1
        else:
            out.append(x)
    return out


def report(label, atoms, site_xyz, site):
    groups = {}
    for x in atoms:
        groups.setdefault((x["chain"], x["res"], x["resname"], x["rec"]), []).append(x)
    print(f"\n{label}: {len(atoms)} atoms in {len(groups)} residues (first 25)")
    for (chain, res, resname, rec), xs in list(groups.items())[:25]:
        pts = np.asarray([x["xyz"] for x in xs])
        d = np.linalg.norm(pts[:, None, :] - site_xyz[None, :, :], axis=2).min() if len(site_xyz) else float("nan")
        mark = "  <- SITE" if (chain, res) == site else ""
        print(f"  {rec:6} {chain}:{resname}{res:<6} {len(xs):2d} atoms  altlocs {sorted({x['alt'] or '-' for x in xs})}  "
              f"names {' '.join(x['name'] for x in xs[:6])}  nearest site atom {d:.1f} A{mark}")


def first_keys(atoms):
    seen, out = set(), []
    for a in atoms:
        if key(a) not in seen:
            seen.add(key(a))
            out.append(key(a))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--row", required=True)
args = ap.parse_args()
man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
sel = man[man["row_id"] == args.row]
if sel.empty:
    raise SystemExit(f"row_id {args.row} not in {args.manifest}")
row = sel.iloc[0]
wild, mut = read(row["wild_pdb"]), read(row["mutant_pdb"])
for lab, lst in (("wild", wild), ("mutant", mut)):
    print(f"{lab}: {len(lst)} atoms, {sum(1 for a in lst if HYD.match(a['name']))} hydrogen-like, records "
          f"{dict(Counter(a['rec'] for a in lst))}, altloc labels {dict(Counter(a['alt'] or '-' for a in lst))}")
wild = [a for a in wild if not HYD.match(a["name"])]
mut = [a for a in mut if not HYD.match(a["name"])]
site = (row["chain"], row["resid"] + (row.get("icode", "") or ""))
site_xyz = np.asarray([a["xyz"] for a in wild if (a["chain"], a["res"]) == site])
report("only in wild", only_in(wild, mut), site_xyz, site)
report("only in mutant", only_in(mut, wild), site_xyz, site)
kw, km = first_keys(wild), first_keys(mut)
sw, sm = set(kw), set(km)
cw, cm = [k for k in kw if k in sm], [k for k in km if k in sw]
print(f"\nshared atoms in the same order: {cw == cm}")
if cw != cm:
    i = next(i for i, (a, b) in enumerate(zip(cw, cm)) if a != b)
    print(f"  first divergence at shared atom {i}: wild {cw[i:i+3]} vs mutant {cm[i:i+3]}")