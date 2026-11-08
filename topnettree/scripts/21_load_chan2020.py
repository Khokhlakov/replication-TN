#!/usr/bin/env python3
"""Step 2b. Chan et al. 2020 (Science, ACE2 deep mutational scan vs SARS-CoV-2 RBD): Data File S1 -> label table.

Data File S1 is a one-sheet workbook, not a flat table: residue number and wild-type amino acid appear only on the
first row of each 20-row block (the rest are blank), a "Nonsynonymous" / "Nonsense" marker sits in column A, and the
wild-type self row has the text "WT" instead of a read count. 117 ACE2 residues (19..518) x 20 rows = 2,340
nonsynonymous rows, of which 117 are the WT self rows, so 2,223 are real substitutions; plus 117 nonsense rows.
Columns: log2 enrichment, nCoV-S-High sort (rep 1, rep 2), nCoV-S-Low sort (rep 1, rep 2). WT is 0 by construction
and maximum depletion is floored at -5 (censored values).

Outputs (data/raw/dms/):
  chan2020_tidy.csv      every substitution with all four replicates, naive read counts, floor flags
  chan2020_labels.csv    repo label schema (pdb 6M0J, chain A). ddg is a PROXY (ddg_is_proxy = 1), raw_score kept.

Score used for the label (--score):
  high   mean of the two nCoV-S-High replicates (default). Higher = tighter binding.
  net    mean over replicates of (High - Low)/2. Uses both sorts; High and Low correlate about -0.8.
Proxy conversion: ddg = -RT * ln(2) * score  (positive = weaker, repo convention); bfe_change = -ddg.
The scale is arbitrary: enrichment is not a free energy and the original authors' scaling is not documented, so
Pearson/Spearman are unaffected but absolute kcal/mol-like magnitudes are not meaningful. --scale lets you change it.

    python scripts/21_load_chan2020.py --xlsx data/raw/dms/chan2020/abc0870_data_file_s1.xlsx \\
        [--structure data/structures/antibody_rbd/6M0J.pdb]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import R_KCAL, T_DEFAULT, read_pdb_residues  # noqa: E402

FLOOR = -5.0
# Residue identities that Chan et al. print in the SM (Table S1 and Fig. S3): wild-type -> position.
SM_WT_CHECK = {
    19: "S", 24: "Q", 25: "A", 26: "K", 27: "T", 29: "L", 31: "K", 33: "N", 34: "H", 39: "L", 40: "F", 41: "Y",
    42: "Q", 69: "W", 72: "F", 75: "E", 76: "Q", 79: "L", 89: "Q", 90: "N", 91: "L", 92: "T", 324: "T", 325: "Q",
    330: "N", 351: "L", 386: "A", 389: "P", 393: "R", 518: "R",
}
# Combinatorial variants in Table S1 (Chan SM): the validated single substitutions that sit inside them.
SM_VALIDATED = [(27, "Y"), (79, "T"), (330, "Y"), (386, "L"), (34, "A"), (92, "Q"), (325, "P"), (25, "V"), (42, "L")]


def read_blocks(xlsx: Path) -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    rows = list(wb[wb.sheetnames[0]].iter_rows(values_only=True))
    head = [i for i, r in enumerate(rows) if r and any(str(c).strip() == "Residue #" for c in r if c is not None)]
    if not head:
        sys.exit("Header row ('Residue #') not found; the workbook layout differs from the one this loader expects.")
    section = resid = wt = None
    recs = []
    for r in rows[head[0] + 1:]:
        r = list(r) + [None] * (9 - len(r))
        if r[0] in ("Nonsynonymous", "Nonsense"):
            section = r[0]
        if r[1] is not None:
            resid = int(r[1])
        if r[2] is not None:
            wt = str(r[2]).strip()
        if r[3] is None:
            continue
        recs.append((section, resid, wt, str(r[3]).strip(), r[4], r[5], r[6], r[7], r[8]))
    df = pd.DataFrame(recs, columns=["section", "resid", "wt", "mut", "naive_reads", "high_r1", "high_r2", "low_r1", "low_r2"])
    for c in ("naive_reads", "high_r1", "high_r2", "low_r1", "low_r2"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    n_all = len(df)
    sub = df[(df["section"] == "Nonsynonymous") & (df["mut"] != df["wt"])].copy()
    print(f"rows read {n_all}: nonsynonymous {int((df.section == 'Nonsynonymous').sum())}, nonsense {int((df.section == 'Nonsense').sum())}; "
          f"WT self rows dropped {int(((df.section == 'Nonsynonymous') & (df.mut == df.wt)).sum())}; substitutions kept {len(sub)}")
    bad = sub[["high_r1", "high_r2", "low_r1", "low_r2"]].isna().any(axis=1)
    if bad.any():
        print(f"dropping {int(bad.sum())} rows with a missing replicate value")
        sub = sub[~bad]
    sub["at_floor_high"] = ((sub["high_r1"] <= FLOOR) | (sub["high_r2"] <= FLOOR)).astype(int)
    sub["at_floor_low"] = ((sub["low_r1"] <= FLOOR) | (sub["low_r2"] <= FLOOR)).astype(int)
    sub["high_mean"] = sub[["high_r1", "high_r2"]].mean(axis=1)
    sub["low_mean"] = sub[["low_r1", "low_r2"]].mean(axis=1)
    sub["net_mean"] = (sub["high_mean"] - sub["low_mean"]) / 2.0
    return sub.reset_index(drop=True)


def internal_checks(sub: pd.DataFrame) -> None:
    ok = [sub[sub.resid == r].wt.iloc[0] == w for r, w in SM_WT_CHECK.items() if (sub.resid == r).any()]
    print(f"wild-type identity vs residues named in the Chan SM: {sum(ok)}/{len(ok)} agree")
    reps = sub[["high_r1", "high_r2"]].corr().iloc[0, 1], sub[["low_r1", "low_r2"]].corr().iloc[0, 1], sub[["high_mean", "low_mean"]].corr().iloc[0, 1]
    print(f"replicate correlation High {reps[0]:.2f}, Low {reps[1]:.2f}; High vs Low {reps[2]:.2f} (paper Fig. S2: R2 0.77 / 0.85 / 0.67)")
    q = sub.set_index(["resid", "mut"])
    vals = [q.loc[k, "high_mean"] for k in SM_VALIDATED if k in q.index]
    print(f"validated improving substitutions (Table S1/Fig. S3): mean High score {np.mean(vals):+.2f} over {len(vals)} "
          f"vs library mean {sub.high_mean.mean():+.2f}")
    print(f"censored at the -5 floor: High {int(sub.at_floor_high.sum())}, Low {int(sub.at_floor_low.sum())} of {len(sub)}")


def check_structure(sub: pd.DataFrame, structure: Path, chain: str) -> int:
    res = read_pdb_residues(structure).get(chain, {})
    if not res:
        print(f"chain {chain} not found in {structure.name}; chains present: {sorted(read_pdb_residues(structure))}")
        return 0
    best, best_rate = 0, -1.0
    for off in range(-40, 41):
        rate = float(np.mean([res.get(str(r + off)) == w for r, w in zip(sub["resid"], sub["wt"])]))
        if rate > best_rate:
            best, best_rate = off, rate
    zero = float(np.mean([res.get(str(r)) == w for r, w in zip(sub["resid"], sub["wt"])]))
    print(f"{structure.name} chain {chain}: wild-type agreement at offset 0 = {zero:.3f}; best offset {best:+d} gives {best_rate:.3f}")
    if zero < 0.9:
        print("WARNING: offset 0 is below 90% agreement. Do not build features until the numbering is resolved.")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/raw/dms/chan2020/abc0870_data_file_s1.xlsx")
    ap.add_argument("--out-dir", default="data/raw/dms")
    ap.add_argument("--structure", help="6M0J PDB file, to verify ACE2 numbering (chain A)")
    ap.add_argument("--pdb", default="6M0J")
    ap.add_argument("--chain", default="A")
    ap.add_argument("--score", choices=["high", "net"], default="high")
    ap.add_argument("--scale", type=float, default=math.log(2.0), help="nats per score unit (default ln 2: score is log2 enrichment)")
    args = ap.parse_args()

    sub = tidy(read_blocks(Path(args.xlsx)))
    internal_checks(sub)
    offset = 0
    if args.structure:
        offset = check_structure(sub, Path(args.structure), args.chain)
        if offset:
            print(f"applying residue offset {offset:+d}")
    sub["resid_struct"] = sub["resid"] + offset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "chan2020_tidy.csv", index=False)

    score = sub["high_mean"] if args.score == "high" else sub["net_mean"]
    ddg = -(R_KCAL * T_DEFAULT) * args.scale * score
    labels = pd.DataFrame(
        {
            "row_idx": np.arange(len(sub)), "pdb": args.pdb, "chain": args.chain, "wt": sub["wt"], "resid": sub["resid_struct"].astype(str),
            "mut": sub["mut"], "ddg": ddg.values, "is_nonbinder": 0, "complex_id": f"{args.pdb}_{args.chain}", "is_homology": 0,
            "dataset": "chan2020", "ddg_is_proxy": 1, "bfe_change": (-ddg).values, "raw_score": score.values,
            "score_kind": f"log2_enrichment:{args.score}", "at_floor": (sub["at_floor_high"] | sub["at_floor_low"]).values,
        }
    )
    labels.to_csv(out_dir / "chan2020_labels.csv", index=False)
    print(f"wrote {out_dir}/chan2020_tidy.csv ({len(sub)} rows) and chan2020_labels.csv; "
          f"ddg proxy range {labels.ddg.min():.2f}..{labels.ddg.max():.2f} (arbitrary scale, see docstring)")


if __name__ == "__main__":
    main()
