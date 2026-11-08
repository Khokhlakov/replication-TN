#!/usr/bin/env python3
"""One row per column of antibodies.csv: structure file, RBD chain, antibody chains, ambiguity flags.
Heavy-atom contacts (4.5 A) between each RBD copy (residues 319-541) and every other chain.
data/raw/antibody_aliases.csv overrides with columns antibody_chains (like B+D) and rbd_chain."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import read_pdb_residues  # noqa: E402

spec = importlib.util.spec_from_file_location("contacts31", HERE / "31_contacts.py")
c31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c31)

MAN = Path("data/structures/antibody_rbd/manifest.csv")
CONV = Path("data/structures/antibody_rbd_converted")
ALIAS = Path("data/raw/antibody_aliases.csv")
OUT = Path("data/structures/antibody_rbd/complex_map.csv")
CUT, MIN_PAIRS, MIN_SITES = 4.5, 5, 15

man = pd.read_csv(MAN, dtype=str, keep_default_na=False)
alias = {}
if ALIAS.is_file():
    a = pd.read_csv(ALIAS, dtype=str, keep_default_na=False)
    alias = {row["label"]: row for _, row in a.iterrows() if row.get("pdb")}

rows = []
for _, r in man.iterrows():
    rec = {"col_index": int(r["col_index"]), "label": r["label"], "resolved_id": r["resolved_id"]}
    flags = []
    path = Path(r["path"]) if r["path"] else None
    if path is not None and path.suffix == ".cif":
        conv = CONV / f"{r['resolved_id']}.pdb"
        if conv.is_file():
            path = conv
            flags.append("converted_from_cif")
    if path is None or path.suffix != ".pdb" or not path.is_file():
        rec.update(file="", flags="no_pdb_file")
        rows.append(rec)
        continue
    atoms = c31.heavy_atoms(path)
    res = read_pdb_residues(path)
    rbd = [c for c, d in res.items() if sum(d.get(str(p)) == w for w, p in c31.SITES) >= MIN_SITES]
    if not rbd and r["rbd_chain_guess"] in atoms:
        rbd = [r["rbd_chain_guess"]]
        flags.append("rbd_chain_from_manifest_guess")
    per_copy = {}
    for rc in rbd:
        core = [x for x in atoms.get(rc, []) if c31.in_window(x[0])]
        partners = {}
        if core:
            for c, at in atoms.items():
                if c in rbd:
                    continue
                pairs, _ = c31.contacts(core, at, CUT)
                if pairs >= MIN_PAIRS:
                    partners[c] = pairs
        per_copy[rc] = partners
    best = max(per_copy, key=lambda k: sum(per_copy[k].values())) if per_copy else ""
    partners = per_copy.get(best, {})
    ab = [c for c, _ in sorted(partners.items(), key=lambda kv: -kv[1])]
    if len(rbd) > 1:
        flags.append(f"{len(rbd)}_rbd_copies")
    shared = [c for c in partners if sum(c in p for p in per_copy.values()) > 1]
    if shared:
        flags.append("partner_touches_several_rbd_copies:" + "".join(shared))
    big = [c for c in ab if len(res.get(c, {})) > 400]
    if big:
        flags.append("partner_over_400_residues:" + "".join(big))
    if not rbd:
        flags.append("no_rbd_chain")
    elif not ab:
        flags.append("no_partner_chain")
    al = alias.get(r["label"])
    if al is not None:
        if al.get("rbd_chain"):
            best = al["rbd_chain"]
        if al.get("antibody_chains"):
            ab = al["antibody_chains"].split("+")
        flags.append("alias_override")
    rec.update(file=str(path), rbd_chain=best, ab_chains="+".join(ab), n_rbd_copies=len(rbd),
               contacts_json=json.dumps(partners), flags=";".join(flags))
    rows.append(rec)

out = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
flagged = out[out["flags"].fillna("") != ""]
print(f"wrote {OUT}: {len(out)} rows, {int((out['ab_chains'].fillna('') != '').sum())} with antibody chains, "
      f"{len(flagged)} flagged")
print(flagged[["col_index", "label", "resolved_id", "rbd_chain", "ab_chains", "flags"]].to_string(index=False))