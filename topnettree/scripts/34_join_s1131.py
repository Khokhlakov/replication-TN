#!/usr/bin/env python3
"""Step 1. Join the capsule's SKEMPI-1131 identifiers to the reconstructed SKEMPI 2.0 single-mutation table.

Run in two passes, because the layout of skempi_1131.npy (1131 x 6 byte strings) has not been seen:

  1) python scripts/34_join_s1131.py --s1131 CAPSULE/data/skempi_1131.npy --y CAPSULE/code/crossvalidation/Y_skempi.npy --inspect
     prints the first rows and, per column, what it looks like (PDB id? mutation token? single character?).
  2) python scripts/34_join_s1131.py --s1131 ... --y ... [--map pdb=0 chain=1 wt=2 resid=3 mutaa=4]
     joins on (pdb, wt, resid, mut) [+ chain when known], trying SKEMPI's cleaned and original numbering.

--map overrides the auto-detection. Keys: pdb, mut, chain, wt, resid, mutaa (use mut for a combined token such as
LI45G or L45G; use wt/resid/mutaa when the three parts are separate columns).

Gate: join rate >= 0.95 and Pearson(y, ddg) >= 0.95 on joined rows. The singles table drops mutations whose repeated
measurements span >= 2 kcal/mol; if the join rate is low only because of that, rerun 10_fetch_skempi.py --keep-discordant
into another --out-dir and pass it with --singles.

Outputs (data/raw/skempi/): s1131_joined.csv (s1131_row = row index into the shipped .npy files), s1131_unjoined.csv,
s1131_join_report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import parse_plain_token, parse_skempi_token  # noqa: E402

PDB_RX = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
SK_RX = re.compile(r"^[A-Z][A-Za-z0-9]-?\d+[A-Za-z]?[A-Z]$")
PL_RX = re.compile(r"^[A-Z]-?\d+[A-Za-z]?[A-Z]$")
INT_RX = re.compile(r"^-?\d+[A-Za-z]?$")


def decode(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape, dtype=object)
    for idx, v in np.ndenumerate(arr):
        out[idx] = v.decode("utf-8", "replace").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
    return out


def column_profile(col: np.ndarray) -> dict:
    n = len(col)
    frac = lambda rx: float(np.mean([bool(rx.match(x)) for x in col]))

    def isfloat(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    return {
        "n_unique": int(len(set(col))), "example": col[0], "pdb_like": frac(PDB_RX), "skempi_token": frac(SK_RX),
        "plain_token": frac(PL_RX), "single_char": float(np.mean([len(x) == 1 for x in col])),
        "float_like": float(np.mean([isfloat(x) for x in col])), "int_like": frac(INT_RX), "n": n,
    }


def autodetect(table: np.ndarray) -> dict:
    """Combined-token layout (LI45G / L45G in one column) first; otherwise the separate-column layout seen in the capsule:
    [pdb, chain, wt, resid, mutaa, value]. In that layout the single-letter columns before the residue column are
    (chain, wt) and the one after it is the mutant amino acid. The join rate is the check on this assumption."""
    prof = [column_profile(table[:, j]) for j in range(table.shape[1])]
    mapping: dict[str, int] = {}
    pdb = max(range(len(prof)), key=lambda j: prof[j]["pdb_like"])
    if prof[pdb]["pdb_like"] > 0.9:
        mapping["pdb"] = pdb
    tok = max(range(len(prof)), key=lambda j: max(prof[j]["skempi_token"], prof[j]["plain_token"]))
    if max(prof[tok]["skempi_token"], prof[tok]["plain_token"]) > 0.9 and tok != mapping.get("pdb"):
        mapping["mut"] = tok
        chain = [j for j in range(len(prof)) if prof[j]["single_char"] > 0.95 and j not in mapping.values()]
        if chain and prof[mapping.get("mut", 0)]["plain_token"] > 0.9:
            mapping["chain"] = chain[0]
        return mapping
    ints = [j for j in range(len(prof)) if prof[j]["int_like"] > 0.99 and j != mapping.get("pdb")]
    if "pdb" in mapping and ints:
        resid = ints[0]
        singles = [j for j in range(len(prof)) if prof[j]["single_char"] > 0.95 and j != mapping["pdb"]]
        before = [j for j in singles if j < resid]
        after = [j for j in singles if j > resid]
        if after and before:
            mapping["resid"] = resid
            mapping["mutaa"] = after[0]
            mapping["wt"] = before[-1]
            if len(before) >= 2:
                mapping["chain"] = before[-2]
    return mapping


def tokens_from_row(row: np.ndarray, m: dict):
    """Returns (chain or None, wt, resid, mut) or None.

    'L38G' (plain) also matches SKEMPI's wt+chain+resid+mut pattern (chain '3', residue '8'), so the two forms are
    told apart explicitly: with a separate chain column the token is plain; without one, a token whose second
    character is a letter is a SKEMPI token (LI45G), otherwise it is plain."""
    if "mut" in m:
        s = row[m["mut"]]
        has_chain_col = "chain" in m
        sk = parse_skempi_token(s)
        pl = parse_plain_token(s)
        if sk and not has_chain_col and (s[1].isalpha() or not pl):
            return sk[1], sk[0], sk[2], sk[3]
        if pl:
            return (row[m["chain"]] if has_chain_col else None), pl[0], pl[1], pl[2]
        if sk:
            return sk[1], sk[0], sk[2], sk[3]
        return None
    if all(k in m for k in ("wt", "resid", "mutaa")):
        return (row[m["chain"]] if "chain" in m else None), row[m["wt"]], row[m["resid"]], row[m["mutaa"]]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1131", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--singles", default="data/raw/skempi/skempi_v2_single_labels.csv")
    ap.add_argument("--out-dir", default="data/raw/skempi")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--map", nargs="*", default=[], help="e.g. pdb=0 mut=3 chain=1")
    args = ap.parse_args()

    table = decode(np.load(args.s1131, allow_pickle=True))
    y = np.array([float(v) for v in decode(np.load(args.y, allow_pickle=True)).ravel()])
    print(f"s1131 table {table.shape}; Y {y.shape}; fraction y>0 = {np.mean(y > 0):.3f}")
    if len(y) != len(table):
        sys.exit("Row counts differ between the identifier table and Y.")
    print("first rows:")
    for i in range(min(8, len(table))):
        print(f"  {i}: {list(table[i])}  y={y[i]}")
    print("column profiles:")
    for j in range(table.shape[1]):
        print(f"  col {j}: {column_profile(table[:, j])}")
    if args.inspect:
        print("\nauto-detected mapping:", autodetect(table), "\nRerun without --inspect (add --map if this is wrong).")
        return

    mapping = {k: int(v) for k, v in (kv.split("=") for kv in args.map)} or autodetect(table)
    print("using mapping:", mapping)
    if "pdb" not in mapping or not ("mut" in mapping or "mutaa" in mapping):
        sys.exit("Could not find a PDB column and a mutation column. Pass --map explicitly.")

    sk = pd.read_csv(args.singles, dtype=str, keep_default_na=False)
    sk["ddg"] = pd.to_numeric(sk["ddg"])
    sk["pdb"] = sk["pdb"].str.upper()
    parsed = [tokens_from_row(table[i], mapping) for i in range(len(table))]
    rows = [
        {"s1131_row": i, "pdb": table[i, mapping["pdb"]].upper(), "chain_s": p[0], "wt": p[1], "resid_s": str(p[2]).upper(),
         "mut": p[3], "y": y[i]}
        for i, p in enumerate(parsed) if p is not None
    ]
    left = pd.DataFrame(rows)
    print(f"parsed {len(left)} of {len(table)} identifier rows")

    best = None
    for scheme, col in (("cleaned", "resid"), ("original_pdb", "resid_pdb")):
        right = sk.assign(resid_s=sk[col].str.upper())
        if left["chain_s"].notna().all():
            keys = ["pdb", "chain_s", "wt", "resid_s", "mut"]
            right = right.rename(columns={"chain": "chain_s"})
        else:
            keys = ["pdb", "wt", "resid_s", "mut"]
        merged = left.merge(right[keys + ["ddg", "chain"] if "chain" in right.columns else keys + ["ddg"]].drop_duplicates(keys),
                            on=keys, how="left")
        rate = float(merged["ddg"].notna().mean())
        print(f"  scheme {scheme:13s} join rate {rate:.3f} ({int(merged['ddg'].notna().sum())}/{len(left)}) on keys {keys}")
        if best is None or rate > best[1]:
            best = (scheme, rate, merged)
    scheme, rate, merged = best
    joined = merged[merged["ddg"].notna()].copy()
    report = {"mapping": mapping, "best_scheme": scheme, "n_s1131": int(len(table)), "n_parsed": int(len(left)),
              "n_joined": int(len(joined)), "join_rate": float(len(joined) / len(table))}
    if len(joined) > 3:
        r_pos = float(stats.pearsonr(joined["y"], joined["ddg"])[0])
        sign = 1.0 if r_pos >= 0 else -1.0  # -1: shipped Y is positive = tighter, opposite to the repo's ddg
        yc = sign * joined["y"]
        report["pearson_y_ddg"] = r_pos
        report["pearson_negated"] = -r_pos
        report["y_sign_vs_repo_ddg"] = "same (positive = weaker)" if sign > 0 else "opposite (Y positive = tighter): negate Y to get repo ddg"
        report["spearman_y_ddg"] = float(stats.spearmanr(joined["y"], joined["ddg"])[0])
        report["abs_pearson"] = abs(r_pos)
        report["mean_abs_diff_after_sign"] = float((yc - joined["ddg"]).abs().mean())
        report["frac_abs_diff_lt_0.05_after_sign"] = float(((yc - joined["ddg"]).abs() < 0.05).mean())
    report["gate_pass"] = bool(report["join_rate"] >= 0.95 and report.get("abs_pearson", 0) >= 0.95)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "s1131_joined.csv", index=False)
    merged[merged["ddg"].isna()].to_csv(out / "s1131_unjoined.csv", index=False)
    (out / "s1131_join_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("GATE G1:", "PASS" if report["gate_pass"] else "FAIL (inspect s1131_unjoined.csv; see the docstring for common causes)")


if __name__ == "__main__":
    main()
