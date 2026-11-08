#!/usr/bin/env python3
"""Step 4b. Where does the remaining S4169 gap sit?

38_s4169_report.py found no exact match. The closest option (mean, discard repeats spanning >= 2 kcal/mol) gives
4173 / 317 / 895 / 3278 against the published 4169 / 319 / 901 / 3268; keeping the discordant repeats gives
4187 / 319 / 897 / 3290. This script prints the facts needed to see which undocumented rule would close the gap:

  * how many mutations have ddG exactly 0 (Kd_mut == Kd_wt) and how the tighter/weaker split changes if they count as tighter,
  * the discordant mutations (all repeated values) and the complexes that vanish when they are dropped,
  * mutations that appear under more than one SKEMPI entry (different partner chains) and are merged by the grouping,
  * the spread of repeat counts.

Nothing is written back. Read the output; the gap is 4 rows (0.1%) and does not need to block the replication.

    python scripts/38b_s4169_diagnose.py [--entries data/raw/skempi/skempi_v2_entries.csv]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
spec = importlib.util.spec_from_file_location("s38", HERE / "38_s4169_report.py")
s38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s38)

KEYS = ["pdb", "chain", "wt", "resid", "mut"]
PUB = s38.PUBLISHED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", default="data/raw/skempi/skempi_v2_entries.csv")
    args = ap.parse_args()
    entries = pd.read_csv(args.entries, dtype={"temperature_K": float})
    base = s38.build_singles(entries)
    g = base.groupby(KEYS, sort=False)["ddg"].agg(mean="mean", lo="min", hi="max", n="size").reset_index()
    g["rng"] = g["hi"] - g["lo"]
    kept = g[g["rng"] < 2.0]
    print(f"measurements {len(base)}; unique mutations {len(g)} (published {PUB['variants']}); after dropping repeats spanning >= 2.0: {len(kept)}")

    print("\n1) sign split and exact zeros (published: tighter 901, weaker 3268)")
    for name, d in (("all unique (no discard)", g), ("discard >= 2.0", kept)):
        v = d["mean"]
        print(f"  {name:24s} n={len(d)}  ddg<0: {int((v < 0).sum())}  ddg==0 exactly: {int((v == 0).sum())}  "
              f"|ddg|<0.01: {int((v.abs() < 0.01).sum())}  ddg<=0: {int((v <= 0).sum())}  ddg>0: {int((v > 0).sum())}")

    print("\n2) repeat counts per mutation:", g["n"].value_counts().sort_index().to_dict())
    disc = g[g["rng"] >= 2.0].sort_values("rng", ascending=False)
    print(f"\n3) discordant mutations (range >= 2.0): {len(disc)}")
    vals = base.groupby(KEYS, sort=False)["ddg"].apply(lambda s: ", ".join(f"{x:+.2f}" for x in s)).rename("values")
    print(disc.merge(vals.reset_index(), on=KEYS)[KEYS + ["n", "rng", "values"]].to_string(index=False))
    lost = sorted(set(g["pdb"]) - set(kept["pdb"]))
    print(f"\n   complexes that vanish when these are dropped: {lost}")
    for pid in lost:
        print(f"   {pid}: {int((g.pdb == pid).sum())} unique mutation(s) in total")

    part = base.groupby(KEYS, sort=False)["partners"].nunique()
    multi = part[part > 1]
    print(f"\n4) mutations reported under more than one SKEMPI entry (different partners), merged by the grouping: {len(multi)}")
    if len(multi):
        print(multi.reset_index().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
