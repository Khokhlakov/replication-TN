#!/usr/bin/env python3
"""Which chains touch which RBD copy?   python scripts/31_contacts.py FILE.pdb [--rbd-chains E,G] [--cutoff 4.5]
RBD chains are auto-detected (>= 15 of the paper's mutation-site wild-type residues) unless --rbd-chains is given.
Heavy-atom contacts, RBD residues 319-541 only. PDB format only (convert .cif first)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import FIG2_ORDER, split  # noqa: E402
from topnet.sources import read_pdb_residues  # noqa: E402

SITES = sorted({(wt, pos) for wt, pos, _ in map(split, FIG2_ORDER)})
WINDOW = (319, 541)


def heavy_atoms(path):
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] == "MSE")):
            continue
        if line[16] not in (" ", "A"):
            continue
        if line[12:16].strip().lstrip("0123456789").startswith("H"):
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        out.setdefault(line[21], []).append((line[22:27].strip(), xyz))
    return out


def molecules(path):
    names, cur = {}, ""
    for line in open(path, errors="replace"):
        if not line.startswith("COMPND"):
            continue
        body = line[10:].strip()
        if body.startswith("MOLECULE:"):
            cur = body[9:].strip(" ;")
        elif body.startswith("CHAIN:"):
            for c in body[6:].strip(" ;").split(","):
                names[c.strip()] = cur
    return names


def contacts(a, b, cutoff):
    A = np.array([x for _, x in a])
    B = np.array([x for _, x in b])
    rb = [r for r, _ in b]
    hit, pairs = set(), 0
    for i in range(0, len(A), 200):
        close = ((A[i:i + 200, None, :] - B[None, :, :]) ** 2).sum(-1) < cutoff ** 2
        pairs += int(close.sum())
        hit.update(rb[j] for j in np.where(close.any(axis=0))[0])
    return pairs, len(hit)


def in_window(rid):
    m = re.match(r"-?\d+", rid)
    return bool(m) and WINDOW[0] <= int(m.group(0)) <= WINDOW[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("--rbd-chains")
    ap.add_argument("--cutoff", type=float, default=4.5)
    args = ap.parse_args()
    path = Path(args.pdb)
    atoms, mol = heavy_atoms(path), molecules(path)
    if args.rbd_chains:
        rbd = args.rbd_chains.split(",")
    else:
        res = read_pdb_residues(path)
        rbd = [c for c, r in res.items() if sum(r.get(str(p)) == w for w, p in SITES) >= 15]
    print(f"{path.name}: chains {sorted(atoms)}; RBD chains {rbd}")
    for r in rbd:
        core = [a for a in atoms[r] if in_window(a[0])]
        print(f"\nRBD chain {r} ({mol.get(r, '?')}), {len(core)} RBD heavy atoms")
        rows = []
        for c, a in atoms.items():
            if c == r or c in rbd:
                continue
            pairs, nres = contacts(core, a, args.cutoff)
            if pairs >= 5:
                rows.append((pairs, nres, c))
        for pairs, nres, c in sorted(rows, reverse=True):
            print(f"  chain {c:>3s}  atom pairs {pairs:5d}  residues {nres:3d}  {mol.get(c, '?')}")


if __name__ == "__main__":
    main()