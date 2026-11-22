#!/usr/bin/env python3
"""Which chains of a structure have given residues resolved, and with which letter?

Use it when a site is "unresolved" in the RBD copy you chose: spike structures hold several RBD
copies, and a copy that is "down" often has a disordered receptor-binding loop (residues ~475-486)
that an "up" copy shows. The chain you chose by antibody contact may not be the copy that has the
residue, and the chain table tells you whether another copy does.

    python scripts/site_by_chain.py --ids 6WPS 7CAH --resid 446 477 478 484

A chain is listed if it holds at least --min-res residues (default 100, so short peptides and
single-residue chains stay out). `*` marks the RBD chain chosen in complex_map_final.csv.
Each cell is the one-letter residue, or `.` if the residue is not in that chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def residues(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 27:
            continue
        chain = line[21]
        key = line[22:26].strip() + line[26].strip()
        out.setdefault(chain, {})[key] = AA3.get(line[17:20].strip(), "?")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", required=True, help="resolved ids, e.g. 6WPS")
    ap.add_argument("--resid", nargs="+", required=True)
    ap.add_argument("--dir", default="data/structures/antibody_rbd")
    ap.add_argument("--map", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--min-res", type=int, default=100)
    args = ap.parse_args()

    chosen = {}
    if Path(args.map).is_file():
        cmap = pd.read_csv(args.map, dtype=str, keep_default_na=False)
        for r in cmap.itertuples():
            chosen.setdefault(r.resolved_id, set()).add(r.rbd_chain)

    for pid in args.ids:
        path = Path(args.dir) / f"{pid}.pdb"
        print(f"\n{pid}  ({path})")
        if not path.is_file():
            print("  file not found")
            continue
        res = residues(path)
        print(f"  {'chain':>5} {'residues':>8}  " + " ".join(f"{r:>5}" for r in args.resid))
        for chain in sorted(res):
            if len(res[chain]) < args.min_res:
                continue
            mark = "*" if chain in chosen.get(pid, set()) else " "
            cells = [res[chain].get(r.upper(), ".") for r in args.resid]
            print(f"  {chain:>4}{mark} {len(res[chain]):>8}  " + " ".join(f"{c:>5}" for c in cells))


if __name__ == "__main__":
    main()
