#!/usr/bin/env python3
"""Retry failed SCAP mutants on structures cut down to the complex's own chains.

42 gives SCAP the whole PDB. Cryo-EM spike files carry other protomers and extra Fabs, and
SCAP can crash (scap_exit_-11) on incomplete residues in chains that the features never use.
The features keep only keep_chains (the RBD chain plus its antibody chains), so SCAP can be run
on exactly those chains.

The first retry (trimming only) still crashed on 7JMP, and the SCAP warnings name residues with missing
atoms ("not complete standard residue"). With --prune-incomplete the trimmed file also drops every residue
that is missing heavy atoms AND lies farther than --min-dist (15 A) from every atom of the mutated chain.
The features only use atoms within 12 A of that chain or of the mutated residue, so a residue that far away
cannot enter any of them. An incomplete residue closer than that is never dropped; it is reported, and
needs a different repair. --altloc first keeps one alternate location per atom, as SCAP expects.

The pruned file is meant to be BOTH the SCAP input and the wild-type of the homology manifest, so wild and
mutant are cut from the same atoms. --out-manifest writes that manifest.

For every mutant listed as failed in a sharded SCAP log, this script
  1. finds its row in the homology manifest (wild_pdb, keep_chains, chain, resid, mut);
  2. writes <out-dir>/<structure>__<chains>.pdb with only the ATOM records of keep_chains
     (one file per structure and chain set, shared by all mutations of it);
  3. writes a rows file for 42 / run_scap_sharded.py whose `structure` is that trimmed file and
     whose `mutant` is the SAME path that failed, so a successful retry fills the gap.

    python scripts/make_trimmed_retry.py \
        --failed-log data/features/rows/panel.sharded_scap_log.csv \
        --manifest data/features/panel_manifest.csv \
        --out-rows data/features/rows/panel_retry_trimmed.csv \
        --prune-incomplete --out-manifest data/features/panel_manifest_pruned.csv
    python scripts/run_scap_sharded.py --rows data/features/rows/panel_retry_trimmed.csv \
        --scap $CAP/data/software/jackal_64bit/bin/scap --shards 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


BACKBONE = {"N", "CA", "C", "O"}
SIDE = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2", "ASP": "CB CG OD1 OD2", "CYS": "CB SG",
    "GLN": "CB CG CD OE1 NE2", "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ", "MET": "CB CG SD CE",
    "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "CB CG CD", "SER": "CB OG", "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
}
TEMPLATE = {k: BACKBONE | set(v.split()) for k, v in SIDE.items()}


def is_h(name: str) -> bool:
    return name.strip().lstrip("0123456789").startswith("H")


def read_atoms(src: Path, chains: set, altloc: str):
    """ATOM lines of the wanted chains. altloc='first' keeps the first alternate of each atom, blanked."""
    atoms, seen = [], {}
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 54 or line[21] not in chains:
            continue
        alt = line[16]
        if altloc == "first" and alt != " ":
            key = (line[21], line[22:27], line[12:16])
            if seen.setdefault(key, alt) != alt:
                continue
            line = line[:16] + " " + line[17:]
        atoms.append(
            {
                "line": line, "chain": line[21], "res": (line[21], line[22:26].strip(), line[26].strip()),
                "resname": line[17:20].strip(), "name": line[12:16].strip(),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            }
        )
    return atoms


def find_incomplete(atoms):
    """{residue key: (resname, [missing heavy atom names])} for standard residues missing heavy atoms."""
    by = {}
    for a in atoms:
        by.setdefault(a["res"], {"resname": a["resname"], "names": set()})["names"].add(a["name"])
    out = {}
    for key, v in by.items():
        want = TEMPLATE.get(v["resname"])
        if want is None:
            continue
        have = {n for n in v["names"] if not is_h(n)}
        missing = sorted(want - have)
        if missing:
            out[key] = (v["resname"], missing)
    return out


def trim(src: Path, dest: Path, chains: set, mut_chain: str, altloc: str, prune: bool, min_dist: float):
    """Write the trimmed file. Returns (atoms kept, report dict)."""
    atoms = read_atoms(src, chains, altloc)
    incomplete = find_incomplete(atoms)
    report = {"incomplete": len(incomplete), "pruned": [], "near": []}
    drop = set()
    if prune and incomplete:
        rbd = np.asarray([a["xyz"] for a in atoms if a["chain"] == mut_chain and not is_h(a["name"])])
        tree = cKDTree(rbd)
        by_res = {}
        for a in atoms:
            by_res.setdefault(a["res"], []).append(a["xyz"])
        for key, (resname, missing) in incomplete.items():
            dist = float(tree.query(np.asarray(by_res[key]), k=1)[0].min())
            label = f"{key[0]}:{resname}{key[1]}{key[2]}({dist:.0f}A)"
            if dist > min_dist:
                drop.add(key)
                report["pruned"].append(label)
            else:
                report["near"].append(label)
    kept = [a["line"] for a in atoms if a["res"] not in drop]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(l.rstrip("\n") for l in kept) + "\nEND\n", encoding="utf-8")
    return len(kept), report

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failed-log", required=True, help="a *_scap_log.csv with columns mutant,status")
    ap.add_argument("--manifest", required=True, help="homology manifest with mutant_pdb, wild_pdb, chain, resid, mut, keep_chains")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-dir", default="data/structures/antibody_rbd_trimmed")
    ap.add_argument("--status", default="scap_exit", help="retry rows whose status contains this text")
    ap.add_argument("--prune-incomplete", action="store_true")
    ap.add_argument("--min-dist", type=float, default=15.0, help="A from the mutated chain beyond which an incomplete residue may be dropped")
    ap.add_argument("--altloc", choices=["all", "first"], default="first")
    ap.add_argument("--out-manifest", help="copy of --manifest whose wild_pdb is the trimmed file for the retried rows")
    args = ap.parse_args()

    log = pd.read_csv(args.failed_log, dtype=str, keep_default_na=False)
    failed = log[log["status"].str.contains(args.status)]["mutant"].tolist()
    # The log may hold absolute paths and the manifest relative ones: match on the path under data/.
    def key(p):
        p = str(p)
        return p[p.index("data/"):] if "data/" in p else p

    failed_keys = {key(p) for p in failed}
    man = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    man["k"] = man["mutant_pdb"].map(key)
    sel = man[man["k"].isin(failed_keys)].drop_duplicates("k")
    missing = failed_keys - set(sel["k"])
    print(f"{len(failed_keys)} failed mutants in the log, {len(sel)} found in the manifest, {len(missing)} not found")
    for m in sorted(missing)[:5]:
        print(f"  not in manifest: {m}")
    if (sel["keep_chains"] == "").any():
        raise SystemExit("some failed rows have no keep_chains; there is nothing to trim to")

    out_dir = Path(args.out_dir)
    made, rows = {}, []
    for r in sel.itertuples():
        chains = r.keep_chains.split("+")
        src = Path(r.wild_pdb)
        if not src.is_absolute():
            src = ROOT / src
        name = f"{src.stem}__{''.join(chains)}.pdb"
        dest = out_dir / name
        if name not in made:
            n_kept, rep = trim(src, ROOT / dest if not dest.is_absolute() else dest, set(chains), r.chain,
                               args.altloc, args.prune_incomplete, args.min_dist)
            if n_kept == 0:
                raise SystemExit(f"no ATOM records for chains {chains} in {src}")
            made[name] = (n_kept, rep)
        rows.append(
            {
                "structure": str(dest),
                "mutant": key(r.mutant_pdb),
                "chain": r.chain,
                "resid_struct": r.resid,
                "wt": r.wt,
                "mut": r.mut,
                "keep_chains": r.keep_chains,
            }
        )
    pd.DataFrame(rows).to_csv(args.out_rows, index=False, lineterminator="\n")
    print(f"wrote {len(made)} trimmed structures to {out_dir}")
    for name, (n, rep) in made.items():
        print(f"  {name}: {n} ATOM records, {rep['incomplete']} incomplete residues")
        if rep["pruned"]:
            print(f"      pruned (far from the mutated chain): {' '.join(rep['pruned'][:12])}{' ...' if len(rep['pruned']) > 12 else ''}")
        if rep["near"]:
            print(f"      NEAR, kept, needs another repair: {' '.join(rep['near'][:12])}")
        elif rep["incomplete"] and not args.prune_incomplete:
            print("      (run with --prune-incomplete to drop the far ones)")
    if args.out_manifest:
        lookup = {str(r["mutant"]): r["structure"] for r in rows}
        man["wild_pdb"] = [lookup.get(k, w) for k, w in zip(man["k"], man["wild_pdb"])]
        man.drop(columns=["k"]).to_csv(args.out_manifest, index=False, lineterminator="\n")
        print(f"wrote {args.out_manifest}: wild_pdb replaced for {len(lookup)} retried rows")
    print(f"wrote {args.out_rows}: {len(rows)} rows")


if __name__ == "__main__":
    main()
