#!/usr/bin/env python3
"""Step 1. SKEMPI 2.0: download, parse affinities into ddG, export in the repo's label schema.

Outputs (data/raw/skempi/):
  skempi_v2.csv                  raw download
  skempi_v2_entries.csv          every entry with parsed ddG (single and multi-point)
  skempi_v2_single_labels.csv    single-point mutations, one row per (pdb, mutation), ddG averaged over measurements
  pdbs/<PDBID>.pdb               only with --structures (or --pdb-tgz-local)
  numbering_check.json           which residue numbering the structure files follow

Sign: ddg = RT ln(Kd_mut / Kd_wt), positive = weaker binding. This is the AB-Bind / data.py convention.
The paper's BFE change is the negative of it, so bfe_change = -ddg is written alongside.

    python scripts/10_fetch_skempi.py --ab-bind-labels data/labels/ab_bind_s645.csv --structures
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import (  # noqa: E402
    R_KCAL, T_DEFAULT, UA, find_col, http_get, looks_like_html, parse_skempi_token, read_pdb_residues,
)

CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
TGZ_URL = "https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz"
SI_REFERENCE = {"entries": 7085, "single_or_element_count": 4169, "complexes": 319}  # SI section S3.1


def get_csv(out_dir: Path, local: str | None) -> Path:
    dest = out_dir / "skempi_v2.csv"
    if local:
        dest.write_bytes(Path(local).read_bytes())
    elif not dest.is_file():
        data = http_get(CSV_URL, timeout=120)
        if looks_like_html(data):
            sys.exit("The SKEMPI URL returned HTML. Download skempi_v2.csv manually and rerun with --csv-local.")
        dest.write_bytes(data)
    return dest


def parse_temperature(text: str) -> float:
    m = re.search(r"\d+(\.\d+)?", str(text))
    if not m:
        return T_DEFAULT
    t = float(m.group(0))
    return t + 273.15 if t < 100 else t


def load_entries(csv_path: Path) -> pd.DataFrame:
    head = csv_path.open(encoding="utf-8", errors="replace").readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    raw = pd.read_csv(csv_path, sep=sep, dtype=str, keep_default_na=False)
    cols = raw.columns
    c_pdb = find_col(cols, r"^#?\s*pdb$")
    c_mut_pdb = find_col(cols, r"mutation.*pdb")
    c_mut_clean = find_col(cols, r"mutation.*clean")
    c_mut_aff = find_col(cols, r"affinity_mut_parsed")
    c_wt_aff = find_col(cols, r"affinity_wt_parsed")
    c_temp = find_col(cols, r"^temperature", required=False)
    c_hold = find_col(cols, r"hold_out_type", required=False)
    c_ref = find_col(cols, r"^reference", required=False)
    c_method = find_col(cols, r"^method", required=False)
    print(f"columns used: pdb={c_pdb!r} mut_pdb={c_mut_pdb!r} mut_clean={c_mut_clean!r} "
          f"aff_mut={c_mut_aff!r} aff_wt={c_wt_aff!r} temp={c_temp!r}")

    out = pd.DataFrame(
        {
            "entry": raw[c_pdb],
            "mut_pdb": raw[c_mut_pdb],
            "mut_clean": raw[c_mut_clean],
            "kd_mut": pd.to_numeric(raw[c_mut_aff], errors="coerce"),
            "kd_wt": pd.to_numeric(raw[c_wt_aff], errors="coerce"),
            "temperature_K": raw[c_temp].map(parse_temperature) if c_temp else T_DEFAULT,
            "hold_out_type": raw[c_hold] if c_hold else "",
            "reference": raw[c_ref] if c_ref else "",
            "method": raw[c_method] if c_method else "",
        }
    )
    out["pdb"] = out["entry"].str.split("_").str[0].str.upper()
    out["partners"] = out["entry"].str.split("_", n=1).str[1].fillna("")
    out["n_mut"] = out["mut_clean"].map(lambda s: len([t for t in str(s).split(",") if t.strip()]))
    ok = (out["kd_mut"] > 0) & (out["kd_wt"] > 0)
    out["ddg"] = np.where(ok, R_KCAL * out["temperature_K"] * np.log(out["kd_mut"] / out["kd_wt"]), np.nan)
    out["bfe_change"] = -out["ddg"]
    return out


def single_labels(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in entries[(entries["n_mut"] == 1) & entries["ddg"].notna()].itertuples():
        tok = parse_skempi_token(rec.mut_clean)
        tok_pdb = parse_skempi_token(rec.mut_pdb)
        if tok is None:
            continue
        wt, chain, resid, mut = tok
        rows.append(
            {
                "pdb": rec.pdb, "chain": chain, "wt": wt, "resid": resid, "mut": mut,
                "resid_pdb": tok_pdb[2] if tok_pdb else "", "ddg": rec.ddg,
                "partners": rec.partners, "hold_out_type": rec.hold_out_type,
            }
        )
    frame = pd.DataFrame(rows)
    keys = ["pdb", "chain", "wt", "resid", "mut"]
    grouped = frame.groupby(keys, sort=False)
    agg = grouped.agg(
        resid_pdb=("resid_pdb", "first"), ddg_mean=("ddg", "mean"), ddg_sd=("ddg", "std"),
        n_meas=("ddg", "size"), partners=("partners", "first"), hold_out_type=("hold_out_type", "first"),
    ).reset_index()
    agg = agg.rename(columns={"ddg_mean": "ddg"})
    agg.insert(0, "row_idx", np.arange(len(agg)))
    agg["is_nonbinder"] = 0
    agg["complex_id"] = agg["pdb"]
    agg["is_homology"] = 0
    agg["dup_group"] = -1
    agg["bfe_change"] = -agg["ddg"]
    return agg


def flag_ab_bind_overlap(singles: pd.DataFrame, labels_csv: str) -> pd.DataFrame:
    ab = pd.read_csv(labels_csv, dtype=str)
    ab = ab[~ab["pdb"].str.startswith("HM_")]
    key = lambda d: d["pdb"].str.upper() + "|" + d["chain"] + "|" + d["wt"] + "|" + d["resid"].astype(str) + "|" + d["mut"]
    singles["ab_bind_overlap"] = key(singles).isin(set(key(ab))).astype(int)
    singles["ab_bind_pdb_overlap"] = singles["pdb"].isin(set(ab["pdb"].str.upper())).astype(int)
    print(f"AB-Bind S645 overlap: {int(singles['ab_bind_overlap'].sum())} identical mutations, "
          f"{int(singles['ab_bind_pdb_overlap'].sum())} rows on shared PDB ids. Exclude them from any train/test mix with S645.")
    return singles


def get_structures(out_dir: Path, local: str | None) -> Path:
    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    stream = open(local, "rb") if local else urllib.request.urlopen(urllib.request.Request(TGZ_URL, headers=UA), timeout=120)
    seen, kept = [], 0
    with stream as handle, tarfile.open(fileobj=handle, mode="r|gz") as tar:
        for member in tar:
            if len(seen) < 8:
                seen.append(member.name)
            if not member.isfile() or not member.name.lower().endswith(".pdb"):
                continue
            pid = Path(member.name).stem.upper()
            data = tar.extractfile(member).read()
            (pdb_dir / f"{pid}.pdb").write_bytes(data)
            kept += 1
    print(f"tar members (first 8): {seen}\nextracted {kept} .pdb files to {pdb_dir}")
    return pdb_dir


def check_numbering(singles: pd.DataFrame, pdb_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Which numbering do the structure files follow: SKEMPI's cleaned numbering or the original PDB one?"""
    cache: dict[str, dict] = {}
    hit = {"clean": 0, "pdb": 0}
    total = 0
    missing = set()
    for rec in singles.itertuples():
        path = pdb_dir / f"{rec.pdb}.pdb"
        if not path.is_file():
            missing.add(rec.pdb)
            continue
        res = cache.setdefault(rec.pdb, read_pdb_residues(path))
        chain = {k.upper(): v for k, v in res.get(rec.chain, {}).items()}
        total += 1
        for scheme, rid in (("clean", rec.resid), ("pdb", rec.resid_pdb)):
            if rid and chain.get(str(rid).upper()) == rec.wt:
                hit[scheme] += 1
    rates = {k: (v / total if total else float("nan")) for k, v in hit.items()}
    report = {"n_checked": total, "wt_match_rate": rates, "pdbs_missing": sorted(missing)[:50], "n_pdbs_missing": len(missing)}
    (out_dir / "numbering_check.json").write_text(json.dumps(report, indent=2))
    print("numbering check:", report)
    best = max(rates, key=lambda k: -1 if math.isnan(rates[k]) else rates[k]) if total else None
    if best and rates[best] > 0.9:
        singles["resid_struct"] = singles["resid"] if best == "clean" else singles["resid_pdb"]
        print(f"structure files follow the '{best}' numbering: use column resid_struct for feature generation.")
    else:
        print("WARNING: neither numbering reaches 90% wild-type agreement. Do not generate features until this is resolved.")
    return singles


def check_1131(singles: pd.DataFrame, y_path: str) -> None:
    """Fingerprint test of the shipped S1131 labels against SKEMPI 2.0 single-mutation ddG."""
    y = np.load(y_path).astype(float).ravel()
    pool = np.sort(singles["ddg"].to_numpy())

    def frac(values):
        idx = np.clip(np.searchsorted(pool, values), 1, len(pool) - 1)
        near = np.minimum(np.abs(pool[idx] - values), np.abs(pool[idx - 1] - values))
        return float((near < 0.005).mean())

    print(f"Y_skempi.npy: n={len(y)}  exact-ish matches (+/-0.005) in SKEMPI2 singles: as-is {frac(y):.2f}, "
          f"negated {frac(-y):.2f}, control shift {frac(y + 0.0137):.2f}")
    print("as-is >> negated means Y_skempi uses the same sign as ddg here; control should sit well below both.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw/skempi")
    ap.add_argument("--csv-local")
    ap.add_argument("--structures", action="store_true", help="download SKEMPI2_PDBs.tgz and extract the .pdb files")
    ap.add_argument("--pdb-tgz-local")
    ap.add_argument("--ab-bind-labels", help="data/labels/ab_bind_s645.csv, to flag overlapping rows")
    ap.add_argument("--check-1131", help="path to Y_skempi.npy from the capsule")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(get_csv(out_dir, args.csv_local))
    entries.to_csv(out_dir / "skempi_v2_entries.csv", index=False)
    singles = single_labels(entries)
    print(f"entries {len(entries)} (SI: {SI_REFERENCE['entries']}) | with ddG {int(entries['ddg'].notna().sum())} | "
          f"single-point rows after averaging {len(singles)} | complexes {entries['pdb'].nunique()} (SI: {SI_REFERENCE['complexes']}) "
          f"| SI quotes {SI_REFERENCE['single_or_element_count']} for the second figure; compare and decide the filter yourself.")
    if args.ab_bind_labels:
        singles = flag_ab_bind_overlap(singles, args.ab_bind_labels)
    pdb_dir = out_dir / "pdbs"
    if args.structures or args.pdb_tgz_local:
        pdb_dir = get_structures(out_dir, args.pdb_tgz_local)
    if pdb_dir.is_dir() and any(pdb_dir.glob("*.pdb")):
        singles = check_numbering(singles, pdb_dir, out_dir)
    singles.to_csv(out_dir / "skempi_v2_single_labels.csv", index=False)
    if args.check_1131:
        check_1131(singles, args.check_1131)
    print(f"wrote {out_dir}/skempi_v2_entries.csv and skempi_v2_single_labels.csv")


if __name__ == "__main__":
    main()
