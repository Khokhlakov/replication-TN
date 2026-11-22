#!/usr/bin/env python3
"""Retry failed SCAP mutants on structures cut down to the complex's own chains.

42 gives SCAP the whole PDB. Cryo-EM spike files carry other protomers and extra Fabs, and
SCAP can crash (scap_exit_-11) on incomplete residues in chains that the features never use.
The features keep only keep_chains (the RBD chain plus its antibody chains), so SCAP can be run
on exactly those chains.

For every mutant listed as failed in a sharded SCAP log, this script
  1. finds its row in the homology manifest (wild_pdb, keep_chains, chain, resid, mut);
  2. writes <out-dir>/<structure>__<chains>.pdb with only the ATOM records of keep_chains
     (one file per structure and chain set, shared by all mutations of it);
  3. writes a rows file for 42 / run_scap_sharded.py whose `structure` is that trimmed file and
     whose `mutant` is the SAME path that failed, so a successful retry fills the gap.

    python scripts/make_trimmed_retry.py \
        --failed-log data/features/rows/panel.sharded_scap_log.csv \
        --manifest data/features/panel_manifest.csv \
        --out-rows data/features/rows/panel_retry_trimmed.csv
    python scripts/run_scap_sharded.py --rows data/features/rows/panel_retry_trimmed.csv \
        --scap $CAP/data/software/jackal_64bit/bin/scap --shards 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def trim(src: Path, dest: Path, chains: set) -> int:
    kept = 0
    lines = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        if line.startswith("ATOM") and len(line) > 21 and line[21] in chains:
            lines.append(line if line.endswith("\n") else line + "\n")
            kept += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines) + "END\n", encoding="utf-8")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--failed-log", required=True, help="a *_scap_log.csv with columns mutant,status")
    ap.add_argument("--manifest", required=True, help="homology manifest with mutant_pdb, wild_pdb, chain, resid, mut, keep_chains")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-dir", default="data/structures/antibody_rbd_trimmed")
    ap.add_argument("--status", default="scap_exit", help="retry rows whose status contains this text")
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
            made[name] = trim(src, ROOT / dest if not dest.is_absolute() else dest, set(chains))
            if made[name] == 0:
                raise SystemExit(f"no ATOM records for chains {chains} in {src}")
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
    for name, n in list(made.items())[:10]:
        print(f"  {name}: {n} ATOM records")
    print(f"wrote {args.out_rows}: {len(rows)} rows")


if __name__ == "__main__":
    main()
