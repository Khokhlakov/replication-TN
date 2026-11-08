#!/usr/bin/env python3
"""Step 5. Close out the DMS sets.

 a) Starr 2020 labels vs 6M0J chain E, and Chan 2020 labels vs 6M0J chain A: wild-type agreement must be ~1.0.
 b) Linsky 2020 (CTC-445.2): reports whether any data file is present under data/raw/dms/linsky2020/ (and prints its columns),
    and checks that the 75 RBD positions Linsky's SM lists exist in the chosen 7KL9 RBD chain. No per-mutation Linsky table was in
    the supplement PDF, so nothing is invented here.
 c) Independent plausibility check (NOT a training or filling step): Starr measured binding of ACE2 for every one of the paper's
    26 RBD substitutions. For the 15 mutations whose shipped ACE2 BFE is known, correlate it with Starr's measurement
    (bfe_change = -ddg; positive = tighter). A weak correlation would mean the shipped model and the experiment disagree
    before any of our reconstruction is involved.

    python scripts/39_dms_closeout.py [--ctc-pdb data/structures/ctc445/7KL9.pdb --ctc-rbd-chain A]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.mutations import BA1_ROW_ORDER, FIG2_ORDER, split  # noqa: E402
from topnet.sources import read_pdb_residues  # noqa: E402

LINSKY_POSITIONS = (
    [401, 402, 403, 404, 405, 406, 408, 409, 410, 416, 417, 418, 419, 420, 421, 422, 437, 439, 440]
    + list(range(442, 460)) + list(range(470, 481)) + list(range(482, 509))
)  # transcribed from the Linsky SM "Deep mutational scanning" paragraph; 75 positions expected


def wt_agreement(df: pd.DataFrame, pdb: Path, chain: str) -> float:
    res = {k.upper(): v for k, v in read_pdb_residues(pdb).get(chain, {}).items()}
    if not res:
        print(f"  chain {chain} not found in {pdb.name}")
        return float("nan")
    return float(np.mean([res.get(str(r).upper()) == w for r, w in zip(df["resid"], df["wt"])]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starr", default="data/raw/dms/starr2020_rbd_ace2_labels.csv")
    ap.add_argument("--chan", default="data/raw/dms/chan2020_labels.csv")
    ap.add_argument("--six", default="data/structures/antibody_rbd/6M0J.pdb")
    ap.add_argument("--bfes", default="data/raw/antibodies_BFEs.csv")
    ap.add_argument("--linsky-dir", default="data/raw/dms/linsky2020")
    ap.add_argument("--ctc-pdb", default="data/structures/ctc445/7KL9.pdb")
    ap.add_argument("--ctc-rbd-chain", default="A")
    args = ap.parse_args()
    assert len(LINSKY_POSITIONS) == 75, len(LINSKY_POSITIONS)

    print("a) numbering checks")
    starr = pd.read_csv(args.starr, dtype=str, keep_default_na=False)
    chan = pd.read_csv(args.chan, dtype=str, keep_default_na=False)
    six = Path(args.six)
    if six.is_file():
        print(f"  Starr vs 6M0J chain E: {wt_agreement(starr, six, 'E'):.3f} ({len(starr)} rows)")
        print(f"  Chan  vs 6M0J chain A: {wt_agreement(chan, six, 'A'):.3f} ({len(chan)} rows)")
    else:
        print(f"  {six} missing")

    print("\nb) Linsky 2020")
    lin = Path(args.linsky_dir)
    files = sorted(p for p in lin.rglob("*") if p.is_file()) if lin.is_dir() else []
    print(f"  data files in {lin}: {[str(f.name) for f in files] or 'none'}")
    for f in files[:3]:
        if f.suffix.lower() in (".csv", ".tsv", ".txt"):
            print(f"  columns of {f.name}: {list(pd.read_csv(f, sep=None, engine='python', nrows=3).columns)}")
        elif f.suffix.lower() in (".xlsx", ".xls"):
            print(f"  columns of {f.name}: {list(pd.read_excel(f, nrows=3).columns)}")
    ctc = Path(args.ctc_pdb)
    if ctc.is_file():
        all_chains = read_pdb_residues(ctc)
        res = {k.upper(): v for k, v in all_chains.get(args.ctc_rbd_chain, {}).items()}
        if not res:
            print(f"  chain {args.ctc_rbd_chain} not found in {ctc.name}; chains present: {sorted(all_chains)}")
        present = [p for p in LINSKY_POSITIONS if str(p) in res]
        print(f"  {ctc.name} chain {args.ctc_rbd_chain}: {len(present)}/{len(LINSKY_POSITIONS)} Linsky positions present; "
              f"missing {sorted(set(LINSKY_POSITIONS) - set(present))}")
    paper_sites = sorted({split(m)[1] for m in FIG2_ORDER})
    print(f"  paper mutation sites covered by Linsky's scanned positions: {[s for s in paper_sites if s in LINSKY_POSITIONS]}")
    print(f"  paper mutation sites NOT covered: {[s for s in paper_sites if s not in LINSKY_POSITIONS]}")

    print("\nc) Starr measurement vs shipped ACE2 BFE for the paper's mutations (plausibility check only)")
    st = starr.assign(key=starr["wt"] + starr["resid"] + starr["mut"], bfe=pd.to_numeric(starr["bfe_change"]))
    lookup = dict(zip(st["key"], st["bfe"]))
    ace2 = pd.read_csv(args.bfes)["ACE2"].to_numpy()
    shipped = dict(zip(BA1_ROW_ORDER, ace2))
    tab = pd.DataFrame({"mutation": FIG2_ORDER, "starr_bfe_change": [lookup.get(m, np.nan) for m in FIG2_ORDER],
                        "shipped_ace2_bfe": [shipped.get(m, np.nan) for m in FIG2_ORDER]})
    print(tab.to_string(index=False))
    both = tab.dropna()
    if len(both) >= 5:
        print(f"  n={len(both)}  Pearson {stats.pearsonr(both.starr_bfe_change, both.shipped_ace2_bfe)[0]:+.2f}  "
              f"Spearman {stats.spearmanr(both.starr_bfe_change, both.shipped_ace2_bfe)[0]:+.2f}")
    print("  Do NOT use Starr values to fill the 11 missing rows: it would make the final table circular.")
    out = Path("data/mutation_sets")
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "starr_vs_shipped_ace2.csv", index=False)


if __name__ == "__main__":
    main()
