#!/usr/bin/env python3
"""Step 3. Freeze data/structures/antibody_rbd/complex_map.csv into complex_map_final.csv.

For every complex it recomputes heavy-atom contacts (4.5 A, RBD residues 319-541) per RBD copy, then picks ONE RBD copy with a
fixed rule and records which rule fired:
  1. override        a row in --overrides (matched on resolved_id, optionally label_contains) or --aliases (matched on label)
  2. clean_copy      among copies whose partner chains are not shared with any other copy, the one with most contact pairs
  3. max_contacts    no clean copy exists: copy with most contact pairs, flagged shared_partner_unresolved
Antibody chains = partners of the chosen copy with >= --min-pairs contacts, unless overridden. RBD copies are never partners.

QC (qc_pass): exactly one RBD chain, >= 1 antibody chain, chosen-copy contacts >= --min-total (default 50), no flag starting with
'shared_partner_unresolved' or 'no_'. ACE2's >400-residue partner is expected and only noted.

--overrides CSV columns: resolved_id,label_contains,rbd_chain,antibody_chains,note      (antibody_chains like I+J)
--extra CSV columns (structures outside the 186, e.g. 7KL9): label,resolved_id,file,rbd_chain,antibody_chains

    python scripts/37_freeze_complex_map.py [--overrides data/raw/complex_overrides.csv] [--extra data/raw/complex_extra.csv]
"""

from __future__ import annotations

import argparse
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


def pairs_between(atoms, rbd_chain, other):
    core = [a for a in atoms.get(rbd_chain, []) if c31.in_window(a[0])]
    if not core or other not in atoms:
        return 0
    return c31.contacts(core, atoms[other], 4.5)[0]


def load_overrides(path: Path | None):
    if path is None or not path.is_file():
        return pd.DataFrame(columns=["resolved_id", "label_contains", "rbd_chain", "antibody_chains", "note"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def find_override(ov: pd.DataFrame, rid: str, label: str):
    for r in ov.itertuples():
        if r.resolved_id.upper() == rid.upper() and (not r.label_contains or r.label_contains.lower() in label.lower()):
            return r
    return None


def analyse(path: Path, min_pairs: int, forced_rbd=None, forced_ab=None):
    atoms = c31.heavy_atoms(path)
    res = read_pdb_residues(path)
    rbd = [c for c, d in res.items() if sum(d.get(str(p)) == w for w, p in c31.SITES) >= 15]
    flags = []
    if forced_rbd and forced_rbd not in rbd:
        flags.append("forced_rbd_chain_not_auto_detected")
        rbd = sorted(set(rbd) | {forced_rbd})
    per_copy = {}
    for rc in rbd:
        core = [a for a in atoms.get(rc, []) if c31.in_window(a[0])]
        per_copy[rc] = {}
        if core:
            for c, at in atoms.items():
                if c in rbd:
                    continue
                p, _ = c31.contacts(core, at, 4.5)
                if p >= min_pairs:
                    per_copy[rc][c] = p
    if not rbd:
        return None, None, [], per_copy, "none", ["no_rbd_chain"], res, atoms
    shared = {c for rc in per_copy for c in per_copy[rc] if sum(c in per_copy[o] for o in per_copy) > 1}
    if forced_rbd:
        chosen, rule = forced_rbd, "override"
    else:
        clean = [rc for rc in per_copy if per_copy[rc] and not (set(per_copy[rc]) & shared)]
        if clean:
            chosen, rule = max(sorted(clean), key=lambda rc: sum(per_copy[rc].values())), "clean_copy"
        else:
            chosen = max(sorted(per_copy), key=lambda rc: sum(per_copy[rc].values()))
            rule = "max_contacts"
            if per_copy[chosen]:
                flags.append("shared_partner_unresolved:" + "".join(sorted(set(per_copy[chosen]) & shared)))
    if forced_ab:
        ab = forced_ab
        rule = "override"
    else:
        ab = [c for c, _ in sorted(per_copy[chosen].items(), key=lambda kv: -kv[1])]
    return atoms, chosen, ab, per_copy, rule, flags, res, atoms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="data/structures/antibody_rbd/complex_map.csv")
    ap.add_argument("--overrides", default="data/raw/complex_overrides.csv")
    ap.add_argument("--aliases", default="data/raw/antibody_aliases.csv")
    ap.add_argument("--extra", default="data/raw/complex_extra.csv")
    ap.add_argument("--out", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--min-total", type=int, default=50)
    ap.add_argument("--weak-pairs", type=int, default=5,
                    help="a FORCED antibody chain (alias/override/extra) with fewer contact pairs than this is flagged weak_antibody_chain; "
                         "use 1 to accept light chains that barely touch the RBD while still catching a wrong RBD copy (0 pairs)")
    args = ap.parse_args()

    cmap = pd.read_csv(args.map, dtype=str, keep_default_na=False)
    ov = load_overrides(Path(args.overrides))
    alias = {}
    if Path(args.aliases).is_file():
        a = pd.read_csv(args.aliases, dtype=str, keep_default_na=False)
        alias = {r["label"]: r for _, r in a.iterrows() if r.get("pdb")}
    rows = []
    todo = [dict(r, extra=0) for r in cmap.to_dict("records")]
    if Path(args.extra).is_file():
        for k, r in enumerate(pd.read_csv(args.extra, dtype=str, keep_default_na=False).to_dict("records")):
            todo.append({"col_index": str(-1 - k), "label": r["label"], "resolved_id": r["resolved_id"], "file": r["file"],
                         "rbd_chain": r.get("rbd_chain", ""), "ab_chains": r.get("antibody_chains", ""), "extra": 1,
                         "_forced": True})
    for r in todo:
        rec = {"col_index": int(r["col_index"]), "label": r["label"], "resolved_id": r["resolved_id"], "extra": r["extra"]}
        path = Path(r.get("file", ""))
        if not r.get("file") or not path.is_file():
            rec.update(file=r.get("file", ""), rbd_chain="", ab_chains="", rule="none", chosen_pairs=0, n_rbd_copies=0,
                       flags="no_pdb_file", qc_pass=False)
            rows.append(rec)
            continue
        f_rbd = f_ab = None
        source = ""
        al = alias.get(r["label"])
        if al is not None:
            f_rbd = al.get("rbd_chain") or None
            f_ab = al.get("antibody_chains", "").split("+") if al.get("antibody_chains") else None
            source = "alias"
        o = find_override(ov, r["resolved_id"], r["label"])
        if o is not None:
            f_rbd = o.rbd_chain or f_rbd
            f_ab = o.antibody_chains.split("+") if o.antibody_chains else f_ab
            source = "override_csv"
        if r.get("_forced"):
            f_rbd = r["rbd_chain"] or None
            f_ab = r["ab_chains"].split("+") if r["ab_chains"] else None
            source = "extra_csv"
        atoms, chosen, ab, per_copy, rule, flags, res, _ = analyse(path, args.min_pairs, f_rbd, f_ab)
        if chosen is None:
            rec.update(file=str(path), rbd_chain="", ab_chains="", rule=rule, chosen_pairs=0, n_rbd_copies=0, flags=";".join(flags), qc_pass=False)
            rows.append(rec)
            continue
        pairs = {c: pairs_between(atoms, chosen, c) for c in ab}
        total = sum(pairs.values())
        weak = [c for c, p in pairs.items() if p < args.weak_pairs]
        if weak:
            flags.append("weak_antibody_chain:" + "".join(weak))
        big = [c for c in ab if len(res.get(c, {})) > 400]
        if big:
            flags.append("partner_over_400_residues:" + "".join(big) + ("(expected for ACE2)" if r["label"] == "ACE2" else ""))
        if not ab:
            flags.append("no_antibody_chain")
        if total < args.min_total:
            flags.append(f"low_total_contacts:{total}")
        if source:
            flags.append(f"source:{source}")
        bad = any(f.startswith(("shared_partner_unresolved", "no_", "low_total", "weak_")) for f in flags)
        rec.update(file=str(path), rbd_chain=chosen, ab_chains="+".join(ab), rule=rule, chosen_pairs=total, n_rbd_copies=len(per_copy),
                   contacts_json=json.dumps(pairs), flags=";".join(flags), qc_pass=not bad)
        rows.append(rec)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    fails = out[~out["qc_pass"]]
    print(f"wrote {args.out}: {len(out)} rows ({int(out.extra.sum())} extra), QC pass {int(out.qc_pass.sum())}, fail {len(fails)}")
    print("rule counts:", out["rule"].value_counts().to_dict())
    if len(fails):
        print("\nQC failures (add rows to the overrides CSV, then rerun):")
        print(fails[["col_index", "label", "resolved_id", "rbd_chain", "ab_chains", "chosen_pairs", "flags"]].to_string(index=False))
    print("\nGATE G3:", "PASS" if len(fails) == 0 else f"FAIL ({len(fails)} rows)")


if __name__ == "__main__":
    main()
