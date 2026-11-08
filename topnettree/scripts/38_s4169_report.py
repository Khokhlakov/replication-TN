#!/usr/bin/env python3
"""Step 4. Which SKEMPI 2.0 filtering gives the published S4169 (4169 variants, 319 complexes, 901 tighter, 3268 weaker)?

Reads data/raw/skempi/skempi_v2_entries.csv (written by 10_fetch_skempi.py) and sweeps the choices that are not documented in
the paper: how repeated measurements of one mutation are combined (mean, median, or kept separate) and the discordance threshold. Prints a table and the closest option. Nothing is written back to the labels.

    python scripts/38_s4169_report.py [--entries data/raw/skempi/skempi_v2_entries.csv]

Reading the result: if one row matches 4169/319/901/3268 exactly, adopt its options in 10_fetch_skempi.py. If none does, report the
table back; the nearest option and the size of the gap say what is still unknown (for example an extra complex-level filter).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import parse_skempi_token  # noqa: E402

PUBLISHED = {"variants": 4169, "complexes": 319, "tighter": 901, "weaker": 3268}


def build_singles(entries: pd.DataFrame) -> pd.DataFrame:
    sel = entries[(entries["n_mut"] == 1) & entries["ddg"].notna()].copy()
    tok = sel["mut_clean"].map(parse_skempi_token)
    sel = sel[tok.notna()].copy()
    tok = tok[tok.notna()]
    sel["chain"], sel["wt"], sel["resid"], sel["mut"] = (tok.map(lambda t: t[1]), tok.map(lambda t: t[0]),
                                                         tok.map(lambda t: t[2]), tok.map(lambda t: t[3]))
    return sel


def combine(sel: pd.DataFrame, how: str, thr: float) -> pd.DataFrame:
    keys = ["pdb", "chain", "wt", "resid", "mut"]
    if how == "none":
        out = sel.copy()
        return out
    g = sel.groupby(keys, sort=False)["ddg"]
    agg = g.agg(ddg="mean" if how == "mean" else "median", lo="min", hi="max").reset_index()
    agg["rng"] = agg["hi"] - agg["lo"]
    return agg[agg["rng"] < thr].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", default="data/raw/skempi/skempi_v2_entries.csv")
    args = ap.parse_args()
    entries = pd.read_csv(args.entries, dtype={"temperature_K": float})
    print(f"entries {len(entries)}; with ddG {int(entries.ddg.notna().sum())}; single-point with ddG "
          f"{int(((entries.n_mut == 1) & entries.ddg.notna()).sum())}; multi-point {int((entries.n_mut > 1).sum())}")
    base = build_singles(entries)
    print(f"parsed single-point measurements {len(base)}; unique (pdb, chain, wt, resid, mut) {base.drop_duplicates(['pdb','chain','wt','resid','mut']).shape[0]}")

    rows = []
    for how, thr in itertools.product(["mean", "median", "none"], [np.inf, 2.0, 1.0]):
        if how == "none" and thr != np.inf:
            continue
        out = combine(base, how, thr)
        tighter = int((out["ddg"] < 0).sum())
        rows.append({"combine": how, "discordance_thr": thr, "variants": len(out), "complexes": int(out["pdb"].nunique()),
                     "tighter": tighter, "weaker_or_neutral": len(out) - tighter})
    tab = pd.DataFrame(rows).drop_duplicates()
    tab["L1_gap"] = (abs(tab.variants - PUBLISHED["variants"]) + abs(tab.complexes - PUBLISHED["complexes"])
                     + abs(tab.tighter - PUBLISHED["tighter"]) + abs(tab.weaker_or_neutral - PUBLISHED["weaker"]))
    print(f"\npublished: {PUBLISHED}\n")
    print(tab.sort_values("L1_gap").to_string(index=False))
    best = tab.sort_values("L1_gap").iloc[0]
    print("\nclosest option: combine=%s, discordance_thr=%s, gap %d" % (best["combine"], best["discordance_thr"], int(best["L1_gap"])))
    if best["L1_gap"] == 0:
        print("EXACT MATCH. Adopt these options in 10_fetch_skempi.py.")
    else:
        print("No exact match. Send this table back; the remaining gap indicates an undocumented complex- or row-level filter.")
    # Where could the remaining gap sit?  Complexes by number of rows, smallest first.
    per = base.groupby("pdb").size().sort_values()
    print(f"\ncomplexes with the fewest single-point rows (candidates for an unreported minimum-size filter): "
          f"{per.head(8).to_dict()}")


if __name__ == "__main__":
    main()
