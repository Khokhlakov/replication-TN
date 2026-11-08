#!/usr/bin/env python3
"""Step 2. Deep-mutational-scanning tables in the repo's label schema.

Starr 2020 (RBD mutations, binding to ACE2) is fetched automatically. Chan 2020 (ACE2 mutations) and
Linsky 2020 (CTC-445.2 / RBD) are not hard-coded; load them with --dms-file. Run --dms-file with no column
arguments first and the script only prints the file's columns and exits.

Starr's per-mutation table lives in Git LFS. A plain raw.githubusercontent.com request returns a pointer file.
The script detects that, tries media.githubusercontent.com, and falls back to the non-LFS MaveDB upload in the
same repository (per-library rows; the 'average' column is the two-library mean).

Units: Starr scores are delta log10(Ka,app) vs wild type. ddg = -RT ln(10) * score, positive = weaker binding
(repo convention); bfe_change = -ddg. The sign is checked from the data (N501Y should bind tighter, proline
substitutions should not) instead of being assumed.

Enrichment-style tables (Chan, Linsky) are NOT free energies. They are converted to a proxy and marked
ddg_is_proxy=1; how the original model scaled them is not documented, so keep the raw score column.

    python scripts/20_fetch_dms.py --starr
    python scripts/20_fetch_dms.py --dms-file data/raw/dms/chan2020/table.xlsx            # inspect only
    python scripts/20_fetch_dms.py --dms-file ... --dms-name chan2020 --pdb 6M0J --chain A \\
        --mutation-col mutation --score-col log2_enrichment --score-kind log2_enrichment
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import (  # noqa: E402
    AA3, R_KCAL, RT_LN10, T_DEFAULT, find_col, is_lfs_pointer, parse_plain_token, read_pdb_residues, try_get,
)

RAW = "https://raw.githubusercontent.com/jbloomlab/SARS-CoV-2-RBD_DMS/{branch}/{path}"
MEDIA = "https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/{branch}/{path}"
MAIN = "results/single_mut_effects/single_mut_effects.csv"
MAVEDB = "MaveDB_upload/score-set_binding_mut.csv"
SITES = "data/RBD_sites.csv"
HGVS = re.compile(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*|=)$")
LABEL_COLUMNS = ["row_idx", "pdb", "chain", "wt", "resid", "mut", "ddg", "is_nonbinder", "complex_id", "is_homology"]


def _fetch(path: str, local_dir: Path | None):
    """Returns (bytes | None, note). Local clone first, then raw, then media (LFS)."""
    if local_dir is not None:
        f = local_dir / path
        if f.is_file():
            data = f.read_bytes()
            return (None, f"{path}: LFS pointer in local clone") if is_lfs_pointer(data) else (data, f"{path}: local clone")
        return None, f"{path}: not in local clone"
    for branch in ("master", "main"):
        data, status = try_get(RAW.format(branch=branch, path=path), timeout=120, retries=2)
        if data is None:
            continue
        if is_lfs_pointer(data):
            media, mstatus = try_get(MEDIA.format(branch=branch, path=path), timeout=300, retries=2)
            if media is not None and not is_lfs_pointer(media):
                return media, f"{path}: LFS via media.githubusercontent.com ({branch})"
            return None, f"{path}: LFS pointer, media download failed ({mstatus})"
        return data, f"{path}: raw ({branch})"
    return None, f"{path}: unreachable"


def load_starr(local_dir: Path | None) -> pd.DataFrame:
    data, note = _fetch(MAIN, local_dir)
    print("starr main table ->", note)
    if data is not None:
        df = pd.read_csv(io.BytesIO(data))
        c_wt, c_mut = find_col(df.columns, r"^wildtype$"), find_col(df.columns, r"^mutant$")
        c_site, c_score = find_col(df.columns, r"^site_SARS2$"), find_col(df.columns, r"^bind_avg$")
        out = pd.DataFrame({"wt": df[c_wt], "resid": df[c_site].astype(int), "mut": df[c_mut], "score": df[c_score]})
        out["source"] = "single_mut_effects.csv:bind_avg"
    else:
        print("falling back to the MaveDB upload (same measurements, per-library rows)")
        mdata, mnote = _fetch(MAVEDB, local_dir)
        sdata, snote = _fetch(SITES, local_dir)
        print("  ", mnote, "|", snote)
        if mdata is None:
            sys.exit("Neither Starr source could be read. Clone the repo with git-lfs, or pass --starr-dir.")
        mv = pd.read_csv(io.BytesIO(mdata))
        recs = []
        for h, avg in zip(mv["hgvs_pro"], mv["average"]):
            m = HGVS.match(str(h))
            if m and m.group(3) not in ("*", "="):
                recs.append((AA3.get(m.group(1).capitalize()), int(m.group(2)), AA3.get(m.group(3).capitalize()), avg))
        out = pd.DataFrame(recs, columns=["wt", "site_rbd", "mut", "score"]).dropna()
        out = out.groupby(["wt", "site_rbd", "mut"], as_index=False)["score"].mean()
        if sdata is not None:
            sites = pd.read_csv(io.BytesIO(sdata))
            mp = dict(zip(sites["site_RBD"], sites["site_SARS2"]))
            aa = dict(zip(sites["site_RBD"], sites["amino_acid_SARS2"]))
            agree = np.mean([aa.get(s) == w for s, w in zip(out["site_rbd"], out["wt"])])
            out["resid"] = out["site_rbd"].map(mp)
            print(f"   site mapping via RBD_sites.csv, wild-type agreement {agree:.3f}")
        else:
            out["resid"] = out["site_rbd"] + 330
            print("   RBD_sites.csv unavailable: using offset +330 (verify against 6M0J below)")
        out = out.drop(columns="site_rbd").dropna(subset=["resid"])
        out["resid"] = out["resid"].astype(int)
        out["source"] = "MaveDB score-set_binding_mut.csv:average"
    out = out[(out["wt"] != out["mut"]) & out["score"].notna()].reset_index(drop=True)
    return out


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Make score mean 'higher = tighter' using N501Y and proline substitutions as sign controls."""
    n501y = df[(df.resid == 501) & (df.mut == "Y")]["score"]
    pro = df[(df.mut == "P") & (df.resid > 340) & (df.resid < 520)]["score"]
    a = float(n501y.iloc[0]) if len(n501y) else float("nan")
    b = float(pro.mean())
    print(f"sign controls: N501Y score {a:+.3f} (tighter expected), mean Pro score {b:+.3f} (weaker expected)")
    if a > 0 and b < 0:
        df["score_higher_tighter"] = df["score"]
    elif a < 0 and b > 0:
        print("scores are 'higher = weaker'; flipping")
        df["score_higher_tighter"] = -df["score"]
    else:
        sys.exit("Sign controls disagree; inspect the table by hand before converting.")
    return df


def check_wt(df: pd.DataFrame, structure: Path, chain: str) -> float:
    res = read_pdb_residues(structure).get(chain, {})
    rate = float(np.mean([res.get(str(r)) == w for r, w in zip(df["resid"], df["wt"])]))
    print(f"wild-type agreement with {structure.name} chain {chain}: {rate:.3f}")
    return rate


def to_label_table(df: pd.DataFrame, pdb: str, chain: str, name: str, ddg: pd.Series, proxy: bool) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row_idx": np.arange(len(df)), "pdb": pdb, "chain": chain, "wt": df["wt"].values,
            "resid": df["resid"].astype(str).values, "mut": df["mut"].values, "ddg": ddg.values,
            "is_nonbinder": 0, "complex_id": f"{pdb}_{chain}", "is_homology": 0,
        }
    )
    out["dataset"] = name
    out["ddg_is_proxy"] = int(proxy)
    out["bfe_change"] = -out["ddg"]
    out["raw_score"] = df["score"].values
    return out[LABEL_COLUMNS + ["dataset", "ddg_is_proxy", "bfe_change", "raw_score"]]


def read_table(path: Path, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path, sep=None, engine="python")


def best_offset(df: pd.DataFrame, structure: Path, chain: str) -> int:
    res = read_pdb_residues(structure).get(chain, {})
    best, best_rate = 0, -1.0
    for off in range(-600, 601):
        rate = np.mean([res.get(str(r + off)) == w for r, w in zip(df["resid"], df["wt"])])
        if rate > best_rate:
            best, best_rate = off, rate
    print(f"best residue offset {best:+d} (wild-type agreement {best_rate:.3f})")
    return best


def run_user_table(args) -> None:
    path = Path(args.dms_file)
    raw = read_table(path, args.sheet)
    print(f"{path.name}: {raw.shape[0]} rows x {raw.shape[1]} columns\n{list(raw.columns)}\n{raw.head(3).to_string()}")
    if not (args.mutation_col and args.score_col and args.pdb and args.chain and args.dms_name):
        print("\ninspect-only. Rerun with --dms-name --pdb --chain --mutation-col --score-col [--score-kind ...].")
        return
    muts = raw[args.mutation_col].astype(str)
    recs = []
    for i, m in enumerate(muts):
        tok = parse_plain_token(m.strip())
        if tok:
            recs.append((tok[0], int(re.sub(r"\D", "", tok[1])), tok[2], raw[args.score_col].iloc[i]))
            continue
        h = HGVS.match(m.strip())
        if h and h.group(3) not in ("*", "="):
            recs.append((AA3.get(h.group(1).capitalize()), int(h.group(2)), AA3.get(h.group(3).capitalize()), raw[args.score_col].iloc[i]))
    df = pd.DataFrame(recs, columns=["wt", "resid", "mut", "score"]).dropna()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna()
    print(f"parsed {len(df)} of {len(raw)} rows as single substitutions (others skipped)")
    if args.structure:
        off = args.resid_offset if args.resid_offset is not None else best_offset(df, Path(args.structure), args.chain)
        df["resid"] = df["resid"] + off
        check_wt(df, Path(args.structure), args.chain)
    elif args.resid_offset:
        df["resid"] = df["resid"] + args.resid_offset
    base = {"log2_enrichment": math.log(2.0), "log10_enrichment": math.log(10.0), "ln_enrichment": 1.0}[args.score_kind]
    rt = R_KCAL * T_DEFAULT
    sign = 1.0 if args.lower_is_tighter else -1.0  # ddg positive = weaker; higher enrichment is read as tighter
    ddg = sign * rt * base * df["score"]
    out = to_label_table(df, args.pdb, args.chain, args.dms_name, ddg, proxy=True)
    dest = Path(args.out_dir) / f"{args.dms_name}_labels.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} rows). ddg is a proxy; raw_score is kept.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/dms")
    ap.add_argument("--starr", action="store_true")
    ap.add_argument("--starr-dir", help="local clone of jbloomlab/SARS-CoV-2-RBD_DMS")
    ap.add_argument("--structure", help="PDB file used to verify residue numbering (6M0J for Starr)")
    ap.add_argument("--dms-file")
    ap.add_argument("--sheet")
    ap.add_argument("--dms-name")
    ap.add_argument("--pdb")
    ap.add_argument("--chain")
    ap.add_argument("--mutation-col")
    ap.add_argument("--score-col")
    ap.add_argument("--score-kind", choices=["log2_enrichment", "log10_enrichment", "ln_enrichment"], default="log2_enrichment")
    ap.add_argument("--lower-is-tighter", action="store_true", help="set when a higher score means weaker binding")
    ap.add_argument("--resid-offset", type=int)
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.starr:
        df = orient(load_starr(Path(args.starr_dir) if args.starr_dir else None))
        struct = Path(args.structure) if args.structure else Path("data/structures/antibody_rbd/6M0J.pdb")
        if struct.is_file():
            check_wt(df, struct, "E")
        else:
            print(f"(no {struct}; run 30_fetch_antibody_structures.py first to verify numbering against 6M0J chain E)")
        ddg = -RT_LN10 * df["score_higher_tighter"]
        out = to_label_table(df.assign(score=df["score_higher_tighter"]), "6M0J", "E", "starr2020", ddg, proxy=False)
        dest = Path(args.out_dir) / "starr2020_rbd_ace2_labels.csv"
        out.to_csv(dest, index=False)
        print(f"wrote {dest}: {len(out)} rows; ddg range {out.ddg.min():.2f}..{out.ddg.max():.2f} kcal/mol; "
              f"source {df['source'].iloc[0]}")
    if args.dms_file:
        run_user_table(args)
    if not args.starr and not args.dms_file:
        ap.print_help()


if __name__ == "__main__":
    main()
