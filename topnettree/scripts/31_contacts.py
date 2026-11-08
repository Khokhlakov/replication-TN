#!/usr/bin/env python3
"""Heavy-atom contact helpers. 37_freeze_complex_map.py loads this file by path and uses:

    heavy_atoms(path)            -> {chain: [atom, ...]}, atom = (resnum:int, resname, atom_name, x, y, z)
    in_window(resnum)            -> True for RBD residues 319-541
    contacts(core, other, cutoff)-> (n_atom_pairs, n_residue_pairs) between two atom lists within `cutoff` angstrom
    SITES                        -> [(wild-type one-letter, position), ...] for the paper's RBD mutation sites (21)

Reads .pdb or .cif. Author chain ids and author residue numbers (the numbering the paper's mutation names use),
first model only, alternate location ' ', '.' or 'A' only, hydrogens, waters and ligands skipped (MSE kept).

As a command it prints which chains look like RBD copies and which partner chains touch each copy, which is what you need
to pick the right Fab copy in a multi-copy entry such as 7L7E:

    python scripts/31_contacts.py data/structures/antibody_rbd/7L7E.pdb
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import FIG2_ORDER, split  # noqa: E402
from topnet.sources import _CIF_TOKEN, read_pdb_residues  # noqa: E402

RBD_WINDOW = (319, 541)
SITES = sorted({(wt, pos) for wt, pos, _ in map(split, FIG2_ORDER)})
SKIP_RESNAMES = {"HOH", "WAT", "DOD"}


def in_window(resnum) -> bool:
    return RBD_WINDOW[0] <= int(resnum) <= RBD_WINDOW[1]


def _pdb_atoms(path: Path) -> dict:
    out: dict = {}
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            if resname in SKIP_RESNAMES or (line.startswith("HETATM") and resname != "MSE"):
                continue
            name = line[12:16].strip()
            element = line[76:78].strip().upper() or name.lstrip("0123456789")[:1].upper()
            if element in ("H", "D"):
                continue
            try:
                resnum = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            out.setdefault(line[21], []).append((resnum, resname, name, *xyz))
    return out


def _cif_atoms(path: Path) -> dict:
    out: dict = {}
    cols: list[str] = []
    header = in_atoms = False
    first_model = None
    need = ("_atom_site.group_PDB", "_atom_site.label_atom_id", "_atom_site.label_comp_id", "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id", "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z")
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("loop_"):
                cols, header, in_atoms = [], True, False
                continue
            if header:
                if line.startswith("_"):
                    cols.append(line.split()[0])
                    continue
                header = False
                in_atoms = bool(cols) and cols[0].startswith("_atom_site.")
            if not in_atoms:
                continue
            if line.startswith("#"):
                in_atoms = False
                continue
            if line.startswith(("loop_", "_")):
                continue
            tok = [t[1:-1] if len(t) > 1 and t[0] in "'\"" and t[-1] == t[0] else t for t in _CIF_TOKEN.findall(line)]
            if len(tok) != len(cols):
                continue
            rec = dict(zip(cols, tok))
            if any(k not in rec for k in need):
                continue
            model = rec.get("_atom_site.pdbx_PDB_model_num", "1")
            first_model = first_model or model
            if model != first_model:
                break
            if rec.get("_atom_site.label_alt_id", ".") not in (".", "A"):
                continue
            resname = rec["_atom_site.label_comp_id"].upper()
            if resname in SKIP_RESNAMES or (rec["_atom_site.group_PDB"] == "HETATM" and resname != "MSE"):
                continue
            if rec.get("_atom_site.type_symbol", "C").upper() in ("H", "D"):
                continue
            try:
                resnum = int(rec["_atom_site.auth_seq_id"])
                xyz = (float(rec["_atom_site.Cartn_x"]), float(rec["_atom_site.Cartn_y"]), float(rec["_atom_site.Cartn_z"]))
            except ValueError:
                continue
            out.setdefault(rec["_atom_site.auth_asym_id"], []).append((resnum, resname, rec["_atom_site.label_atom_id"], *xyz))
    return out


def heavy_atoms(path) -> dict:
    path = Path(path)
    return _cif_atoms(path) if path.suffix.lower() == ".cif" else _pdb_atoms(path)


def contacts(core, other, cutoff: float = 4.5):
    """(heavy-atom pairs, residue pairs) within `cutoff`. Residues are identified by (chain-local) residue number."""
    if not core or not other:
        return 0, 0
    a = np.array([t[3:6] for t in core], dtype=float)
    b = np.array([t[3:6] for t in other], dtype=float)
    hits = cKDTree(a).query_ball_tree(cKDTree(b), cutoff)
    n_atoms = sum(len(h) for h in hits)
    residue_pairs = {(core[i][0], other[j][0]) for i, h in enumerate(hits) for j in h}
    return n_atoms, len(residue_pairs)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    atoms = heavy_atoms(path)
    res = read_pdb_residues(path)
    rbd = [c for c, d in res.items() if sum(d.get(str(p)) == w for w, p in SITES) >= 15]
    print(f"{path.name}: chains {sorted(atoms)}; RBD-like chains (>= 15 of {len(SITES)} sites with the expected residue): {rbd}")
    for rc in rbd:
        core = [a for a in atoms[rc] if in_window(a[0])]
        row = {}
        for c, at in atoms.items():
            if c == rc or c in rbd:
                continue
            p, r = contacts(core, at, 4.5)
            if p >= 5:
                row[c] = (p, r)
        print(f"  RBD copy {rc}: partners (atom pairs, residue pairs) {dict(sorted(row.items(), key=lambda kv: -kv[1][0]))}")


if __name__ == "__main__":
    main()
