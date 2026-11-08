#!/usr/bin/env python3
"""Step 3. The 186 complexes in data/raw/antibodies.csv (ACE2 + 185 antibodies).

For each line "<ID>_<label>" the script
  1. asks RCSB whether <ID> is a real, released entry and records its title and status,
  2. downloads <ID>.pdb (falls back to .cif and says so),
  3. reads every chain: residue count, overlap with the RBD numbering window, and how many of the paper's
     mutation sites carry the expected wild-type residue. That identifies the RBD chain without assuming a letter.

Non-PDB names (RN33_REGN10933, RN87_REGN10987, 1061_AZD1061) are not guessed. They are written to
data/raw/antibody_aliases.csv with an empty 'pdb' column. Fill it (and optionally the chain columns) after
looking at the RCSB entity listing printed by 00_probe_sources.py, then rerun.

    python scripts/30_fetch_antibody_structures.py [--local-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import FIG2_ORDER, split  # noqa: E402
from topnet.sources import get_json, read_pdb_residues, try_get  # noqa: E402

ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pid}"
PDB_URL = "https://files.rcsb.org/download/{pid}.pdb"
CIF_URL = "https://files.rcsb.org/download/{pid}.cif"
ID_FORMAT = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
RBD_WINDOW = (319, 541)
SITES = sorted({(wt, pos) for wt, pos, _ in map(split, FIG2_ORDER)})
SPIKE_WORDS = re.compile(r"spike|sars|coronavirus|rbd|receptor|ace2|nanobody|antibod|fab|neutraliz", re.I)


def analyse_chains(path: Path) -> dict:
    chains = read_pdb_residues(path)
    info = {}
    for ch, res in chains.items():
        nums = []
        for key in res:
            m = re.match(r"-?\d+", key)
            if m:
                nums.append(int(m.group(0)))
        in_win = sum(RBD_WINDOW[0] <= n <= RBD_WINDOW[1] for n in nums)
        present = [(wt, pos) for wt, pos in SITES if str(pos) in res]
        match = sum(res.get(str(pos)) == wt for wt, pos in present)
        info[ch] = {
            "n_res": len(res),
            "rbd_window_frac": round(in_win / max(len(nums), 1), 3),
            "sites_present": len(present),
            "wt_match": match,
        }
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="data/raw/antibodies.csv")
    ap.add_argument("--out-dir", default="data/structures/antibody_rbd")
    ap.add_argument("--aliases", default="data/raw/antibody_aliases.csv")
    ap.add_argument("--local-dir", help="folder with <ID>.pdb files already downloaded")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [ln.rstrip("\n") for ln in open(args.names, encoding="utf-8") if ln.strip()]
    alias = {}
    if Path(args.aliases).is_file():
        a = pd.read_csv(args.aliases, dtype=str, keep_default_na=False)
        alias = {r.label: r for r in a.itertuples() if r.pdb}

    rows, unresolved = [], []
    for idx, line in enumerate(labels):
        listed, label = line.split("_", 1)
        pid = alias[label].pdb.upper() if label in alias else listed.upper()
        rec = {"col_index": idx, "label": label, "listed_id": listed, "resolved_id": pid,
               "via_alias": int(label in alias), "alias_chains": alias[label].antibody_chains if label in alias and hasattr(alias[label], "antibody_chains") else ""}
        entry, status = (None, "bad id format") if not ID_FORMAT.match(pid) else get_json(ENTRY.format(pid=pid))
        time.sleep(args.sleep)
        if entry is None:
            rec.update(status=status, title="", suspect_title=1, format="", path="")
            unresolved.append((label, listed, status))
            rows.append(rec)
            continue
        title = (entry.get("struct") or {}).get("title", "")
        rec.update(status=(entry.get("rcsb_accession_info") or {}).get("status_code", "?"), title=title,
                   suspect_title=int(not SPIKE_WORDS.search(title)))
        dest = None
        for fmt, url in (("pdb", PDB_URL), ("cif", CIF_URL)):
            cand = out_dir / f"{pid}.{fmt}"
            if cand.is_file() and cand.stat().st_size > 1000:
                dest = cand
                break
            if args.local_dir and (Path(args.local_dir) / f"{pid}.{fmt}").is_file():
                cand.write_bytes((Path(args.local_dir) / f"{pid}.{fmt}").read_bytes())
                dest = cand
                break
            data, st = try_get(url.format(pid=pid), timeout=120, retries=2)
            time.sleep(args.sleep)
            if data is not None and len(data) > 1000:
                cand.write_bytes(data)
                dest = cand
                break
        if dest is None:
            rec.update(format="", path="", status=rec["status"] + " | download failed")
            rows.append(rec)
            continue
        rec.update(format=dest.suffix[1:], path=str(dest))
        if dest.suffix in (".pdb", ".cif"):
            chains = analyse_chains(dest)
            best = max(chains, key=lambda c: chains[c]["wt_match"], default=None)
            rec["chains_json"] = json.dumps(chains)
            rec["rbd_chain_guess"] = best or ""
            rec["rbd_wt_match"] = chains[best]["wt_match"] if best else 0
            rec["rbd_sites_present"] = chains[best]["sites_present"] if best else 0
        else:
            rec["rbd_chain_guess"] = "needs .cif reader"
        rows.append(rec)
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(labels)} processed")

    man = pd.DataFrame(rows)
    for col in ("rbd_wt_match", "rbd_sites_present"):
        man[col] = man[col].fillna(0).astype(int) if col in man else 0
    man.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nmanifest: {out_dir / 'manifest.csv'}")
    print(f"downloaded as .pdb: {(man['format'] == 'pdb').sum()} | .cif only: {(man['format'] == 'cif').sum()} | "
          f"unresolved or failed: {(man['format'] == '').sum()}")
    weak = man[(man["format"] == "pdb") & (man["rbd_wt_match"] < 10)]
    print(f"entries where no chain matches at least 10 of {len(SITES)} RBD sites (check numbering or chain): {len(weak)}")
    if len(weak):
        print(weak[["col_index", "label", "resolved_id", "rbd_wt_match", "rbd_sites_present"]].to_string(index=False))
    sus = man[man["suspect_title"] == 1]
    if len(sus):
        print("titles that do not look like SARS-CoV-2 spike/antibody entries (verify the ID):")
        print(sus[["col_index", "label", "resolved_id", "title"]].to_string(index=False))

    if unresolved and not alias:
        tpl = pd.DataFrame([{"label": l, "pdb": "", "antibody_chains": "", "note": f"listed id {i}: {s}"} for l, i, s in unresolved])
        Path(args.aliases).parent.mkdir(parents=True, exist_ok=True)
        tpl.to_csv(args.aliases, index=False)
        print(f"\nunresolved names written to {args.aliases}: {[u[0] for u in unresolved]}\n"
              "Fill the 'pdb' column (parent entry) and 'antibody_chains', then rerun. Nothing was guessed.")


if __name__ == "__main__":
    main()
