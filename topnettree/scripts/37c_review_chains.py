#!/usr/bin/env python3
"""Step 3d. Review complex_map_final.csv against RCSB entity descriptions (does not change anything).

37/37b choose chains by geometry only. This adds the names: for every complex it lists the description of each selected
antibody chain and flags
  heavy_light_mismatch      selected heavy-chain count differs from selected light-chain count (a Fab missing one chain)
  omitted_in_contact        an antibody-like chain that touches the chosen RBD copy (>= --omit-pairs heavy-atom pairs at 4.5 A) was not selected
  selected_spike_or_receptor a selected chain described as spike/RBD/ACE2 (not checked for ACE2 or CTC labels)
  no_description            RCSB gave no description for a selected chain
Entity listings are cached as JSON in --cache (one file per PDB id), so a rerun is offline.

    python scripts/37c_review_chains.py [--final data/structures/antibody_rbd/complex_map_final.csv]
Fix a flagged row by editing antibody_chains in data/raw/complex_overrides.csv (chains joined by +), rerun 37, rerun this.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import get_json  # noqa: E402

spec = importlib.util.spec_from_file_location("c31", HERE / "31_contacts.py")
c31 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c31)

ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pid}"
ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/{eid}"
HEAVY = re.compile(r"heavy|heigh|\bvh\b|\bhc\b", re.I)
LIGHT = re.compile(r"light|\bvl\b|\blc\b|kappa|lambda", re.I)
ABLIKE = re.compile(r"heavy|light|\bfab\b|antibod|nanobod|\bvhh\b|\bvh\b|\bvl\b|immunoglobulin|\bscfv\b|binder|sybody|\bigg", re.I)
SPIKE = re.compile(r"spike|\bs1\b|receptor.binding|\brbd\b|angiotensin|ace2", re.I)


def entity_chains(pid: str, cache: Path) -> dict:
    f = cache / f"{pid.upper()}.json"
    if f.is_file():
        return json.loads(f.read_text())
    entry, status = get_json(ENTRY.format(pid=pid))
    if entry is None:
        return {}
    out = {}
    for eid in (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []:
        ent, _ = get_json(ENTITY.format(pid=pid, eid=eid))
        time.sleep(0.1)
        if ent is None:
            continue
        desc = (ent.get("rcsb_polymer_entity") or {}).get("pdbx_description") or ""
        for ch in (ent.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []:
            out[ch] = desc
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="data/structures/antibody_rbd/complex_map_final.csv")
    ap.add_argument("--cache", default="data/raw/rcsb_entities")
    ap.add_argument("--out", default="data/structures/antibody_rbd/complex_chain_review.csv")
    ap.add_argument("--omit-pairs", type=int, default=20, help="report an unselected antibody-like chain only if it touches the chosen RBD copy with at least this many pairs")
    args = ap.parse_args()
    final = pd.read_csv(args.final, dtype=str, keep_default_na=False)
    rows = []
    for r in final.itertuples():
        pid = r.resolved_id.upper()
        desc = entity_chains(pid, Path(args.cache))
        sel = [c for c in r.ab_chains.split("+") if c]
        flags = []
        if not desc:
            flags.append("no_rcsb_entities")
        n_h = sum(bool(HEAVY.search(desc.get(c, ""))) for c in sel)
        n_l = sum(bool(LIGHT.search(desc.get(c, ""))) for c in sel)
        if (n_h or n_l) and n_h != n_l:
            flags.append(f"heavy_light_mismatch:H={n_h},L={n_l}")
        lab = r.label.lower()
        if "ace2" not in lab and "ctc" not in lab:
            for c in sel:
                d = desc.get(c, "")
                if not d:
                    flags.append(f"no_description:{c}")
                elif SPIKE.search(d):
                    flags.append(f"selected_spike_or_receptor:{c}({d[:40]})")
        if r.file and Path(r.file).is_file() and r.rbd_chain:
            atoms = c31.heavy_atoms(r.file)
            core = [a for a in atoms.get(r.rbd_chain, []) if c31.in_window(a[0])]
            for c, d in desc.items():
                if c in sel or c == r.rbd_chain or c not in atoms or not ABLIKE.search(d) or SPIKE.search(d):
                    continue
                p, _ = c31.contacts(core, atoms[c], 4.5)
                if p >= args.omit_pairs:
                    flags.append(f"omitted_in_contact:{c}({d[:35]},{p})")
        rows.append({"col_index": r.col_index, "label": r.label, "id": pid, "rbd": r.rbd_chain, "antibody_chains": r.ab_chains,
                     "selected": "; ".join(f"{c}={desc.get(c, '?')[:38]}" for c in sel), "flags": " | ".join(flags)})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    bad = out[out["flags"] != ""]
    print(f"wrote {args.out}: {len(out)} complexes, {len(bad)} with flags")
    if len(bad):
        with pd.option_context("display.max_colwidth", 120, "display.width", 250):
            print(bad[["col_index", "label", "id", "rbd", "antibody_chains", "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
