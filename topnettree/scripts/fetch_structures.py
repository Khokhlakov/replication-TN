#!/usr/bin/env python3
"""
fetch_structures.py  --  Part A of the TopNetTree M5 workflow.

What it does
  1. Reads the label table (data/labels/ab_bind_s645.csv) and derives the list of
     PDB entries. Entries that do not look like RCSB IDs (or that the table flags as
     homology models) are written to a separate list and NOT downloaded.
  2. (optional) Cross-checks the PDB list against the AB-Bind supplementary table.
  3. Downloads each RCSB entry (.pdb, falling back to .cif), with retries, and writes
     a sha256 manifest + download log.
  4. Validates every one of the 645 rows against the downloaded structure:
     chain exists, residue (resseq + insertion code) exists, residue name == wt.
     For failing (pdb, chain) groups it searches for a constant numbering offset.
  5. Writes a per-row report and a per-complex summary.

It never edits raw structure files and never invents data. Needs: python>=3.8, pandas.
Optional: openpyxl (xlsx supplement), biopython (validating .cif fallbacks).

Usage
  python scripts/fetch_structures.py --labels data/labels/ab_bind_s645.csv \
         --outdir data/structures
  python scripts/fetch_structures.py ... --supplement AB-Bind_supplement.xlsx
  python scripts/fetch_structures.py ... --offline        # validate only, no network
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified residues that appear in these entries
    "MSE": "M", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CSO": "C",
}
RCSB_ID = re.compile(r"^[0-9][A-Z0-9]{3}$")
URL = "https://files.rcsb.org/download/{pid}.{ext}"
RESID_RE = re.compile(r"^\s*(-?\d+)\s*([A-Za-z]?)\s*$")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_col(df, patterns, explicit=None):
    """Return the first column whose lowercase name matches any regex in patterns."""
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"ERROR: column '{explicit}' not in {list(df.columns)}")
        return explicit
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, str(c).lower()):
                return c
    return None


def to_one_letter(x):
    s = str(x).strip().upper()
    if len(s) == 3:
        return AA3TO1.get(s, "?")
    return s[:1] if s else "?"


def parse_resid(x):
    """Return (resseq:int, icode:str) from 52, 52.0, '52', '52A', '-3'."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    m = RESID_RE.match(str(x))
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).upper()


def row_resid(r, cols):
    """(resseq, icode) for a label row; uses a separate 'icode' column when present."""
    resseq, ic = parse_resid(r[cols["resid"]])
    if resseq is not None and not ic and cols.get("icode"):
        v = r[cols["icode"]]
        if isinstance(v, str) and v.strip():
            ic = v.strip().upper()
    return resseq, ic


# Supplementary Table S3 of Sirin et al. 2016: homology models are NOT on RCSB.
HM_TEMPLATES = {
    "HM_1YY9": "template 1YY9 chains D->H, C->L (Hu-225 from Erbitux Fab; identity H 81%, L 80%)",
    "HM_3BN9": "template 3BN9 chain A (Epithin from MT-SP1; identity 87%)",
    "HM_2NYY": "template 2NYY chain A (BoNT/A2 from BoNT/A1; identity 90%)",
    "HM_2NZ9": "template 2NZ9 chain A (BoNT/A2 from BoNT/A1; identity 90%)",
    "HM_1KTZ": "template 1KTZ chain A (TGF-beta; identity 76%)",
}


# ----------------------------------------------------------------------------
# step 1: PDB list from the label table
# ----------------------------------------------------------------------------
def load_labels(path, args):
    if not Path(path).exists():
        sys.exit(f"ERROR: label table not found: {path}\n"
                 "Expected columns: row_idx, pdb, chain, wt, resid, mut, ... (see AGENTS.md sec. 2)")
    df = pd.read_csv(path)
    cols = {
        "pdb": find_col(df, [r"^pdb"], args.pdb_col),
        "chain": find_col(df, [r"^chain"], args.chain_col),
        "wt": find_col(df, [r"^wt$", r"wild", r"^wt"], args.wt_col),
        "resid": find_col(df, [r"^resid", r"resnum", r"^resi", r"position", r"^pos"], args.resid_col),
    }
    cols["icode"] = find_col(df, [r"^icode", r"insertion"])          # optional
    missing = [k for k, v in cols.items() if v is None and k != "icode"]
    if missing:
        sys.exit(f"ERROR: could not find columns for {missing} in {list(df.columns)}.\n"
                 "Pass them explicitly: --pdb-col --chain-col --wt-col --resid-col")
    homol = find_col(df, [r"homolog"], args.homology_col)
    if "row_idx" not in df.columns:
        df.insert(0, "row_idx", range(len(df)))
    print(f"[labels] {len(df)} rows; columns used: {cols}; homology flag column: {homol}")
    if len(df) != 645:
        print(f"[warn] expected 645 rows, found {len(df)}")
    return df, cols, homol


def split_ids(df, cols, homol):
    df = df.copy()
    df["_pid"] = df[cols["pdb"]].astype(str).str.strip().str.upper()
    flagged = set()
    if homol:
        flag = df[homol].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        flagged = set(df.loc[flag, "_pid"])
    rcsb, other = [], []
    for pid in sorted(df["_pid"].unique()):
        if pid in flagged or not RCSB_ID.match(pid):
            other.append(pid)
        else:
            rcsb.append(pid)
    return df, rcsb, other


# ----------------------------------------------------------------------------
# step 2: optional supplement cross-check
# ----------------------------------------------------------------------------
def check_supplement(path, sheet, table_ids):
    p = Path(path)
    if not p.exists():
        print(f"[supplement] file not found: {p} (skipping cross-check)")
        return
    try:
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            sheets = pd.read_excel(p, sheet_name=sheet if sheet else None)
            if isinstance(sheets, pd.DataFrame):
                sheets = {sheet or "sheet": sheets}
        else:
            sheets = {"csv": pd.read_csv(p, sep=None, engine="python")}
    except Exception as e:  # noqa: BLE001
        print(f"[supplement] could not read {p}: {e}")
        return
    found = False
    for name, sdf in sheets.items():
        pc = find_col(sdf, [r"pdb"])
        print(f"[supplement] sheet '{name}': shape={sdf.shape}; columns={list(sdf.columns)[:12]}")
        if pc is None:
            continue
        ids = set(sdf[pc].dropna().astype(str).str.strip().str.upper().str[:4])
        ids = {i for i in ids if RCSB_ID.match(i)}
        found = True
        print(f"[supplement] sheet '{name}': {len(ids)} distinct PDB-like IDs in column '{pc}'")
        print(f"[supplement]   in label table but not in supplement: {sorted(set(table_ids) - ids)}")
        print(f"[supplement]   in supplement but not in label table : {sorted(ids - set(table_ids))}"
              "  (expected: the complexes outside S645)")
    if not found:
        print("[supplement] no sheet with a PDB column was found; paste the column names back to me.")


# ----------------------------------------------------------------------------
# step 3: download
# ----------------------------------------------------------------------------
def download_one(pid, rawdir, formats, retries=3, pause=1.5):
    """Return (path or None, ext, status). Skips files already present."""
    for ext in formats:
        dest = rawdir / f"{pid}.{ext}"
        if dest.exists() and dest.stat().st_size > 0:
            return dest, ext, "exists"
        url = URL.format(pid=pid, ext=ext)
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "topnet-replication/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if b"ATOM" not in data and b"_atom_site" not in data:
                    raise ValueError("response has no atom records")
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.rename(dest)
                return dest, ext, "downloaded"
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # try next format
                time.sleep(pause * attempt)
            except Exception:  # noqa: BLE001
                time.sleep(pause * attempt)
    return None, None, "FAILED"


# ----------------------------------------------------------------------------
# step 4: parse + validate
# ----------------------------------------------------------------------------
def parse_pdb(path):
    """First MODEL only. Returns {chain: {(resseq, icode): resname}}."""
    res = defaultdict(dict)
    with open(path, errors="replace") as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if rec == "HETATM" and line[17:20].strip() not in AA3TO1:
                continue  # skip waters, ligands
            alt = line[16]
            if alt not in (" ", "A", "1"):
                continue
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            key = (resseq, line[26].strip().upper())
            res[chain].setdefault(key, line[17:20].strip().upper())
    return res


def parse_cif(path):
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser
    except ImportError:
        return None
    s = MMCIFParser(QUIET=True).get_structure("x", str(path))
    res = defaultdict(dict)
    for ch in next(iter(s)):
        for r in ch:
            het, num, ic = r.id
            if het.strip() == "W":
                continue
            res[ch.id].setdefault((num, ic.strip().upper()), r.get_resname().upper())
    return res


def load_structure(path):
    return parse_cif(path) if str(path).endswith(".cif") else parse_pdb(path)


def validate_rows(df, cols, structures):
    rows = []
    for _, r in df.iterrows():
        pid = r["_pid"]
        out = {"row_idx": r["row_idx"], "pdb": pid, "chain": r[cols["chain"]],
               "resid": r[cols["resid"]], "wt": r[cols["wt"]]}
        s = structures.get(pid)
        if s is None:
            out.update(status="NO_STRUCTURE", found_resname="")
            rows.append(out)
            continue
        resseq, icode = row_resid(r, cols)
        wt = to_one_letter(r[cols["wt"]])
        chains = [c for c in str(r[cols["chain"]]).strip()] or [" "]
        if resseq is None:
            out.update(status="BAD_RESID", found_resname="")
            rows.append(out)
            continue
        status, found = "CHAIN_MISSING", ""
        for c in chains:
            if c not in s:
                continue
            status = "RESIDUE_MISSING"
            rn = s[c].get((resseq, icode))
            if rn is not None:
                found = rn
                status = "OK" if AA3TO1.get(rn, "?") == wt else "WT_MISMATCH"
                if status == "OK":
                    break
        out.update(status=status, found_resname=found)
        rows.append(out)
    return pd.DataFrame(rows)


def offset_diagnostic(df, cols, rep, structures, max_off=30):
    """For (pdb, chain) groups with failures, look for a constant resseq offset."""
    notes = {}
    bad = rep[rep["status"] != "OK"]
    for (pid, ch), g in bad.groupby(["pdb", "chain"]):
        s = structures.get(pid)
        if s is None:
            continue
        sub = df[(df["_pid"] == pid) & (df[cols["chain"]] == ch)]
        best = (0, 0)
        for off in range(-max_off, max_off + 1):
            hit = 0
            for _, r in sub.iterrows():
                resseq, _ic = row_resid(r, cols)
                if resseq is None or _ic:
                    continue
                wt = to_one_letter(r[cols["wt"]])
                for c in str(ch):
                    rn = s.get(c, {}).get((resseq + off, ""))
                    if rn and AA3TO1.get(rn, "?") == wt:
                        hit += 1
                        break
            if hit > best[1]:
                best = (off, hit)
        if best[0] != 0 and best[1] >= 0.9 * len(sub):
            notes[(pid, ch)] = f"possible constant offset {best[0]:+d} ({best[1]}/{len(sub)} rows)"
    return notes


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/labels/ab_bind_s645.csv")
    ap.add_argument("--outdir", default="data/structures")
    ap.add_argument("--formats", default="pdb,cif", help="download order, comma separated")
    ap.add_argument("--supplement", default=None, help="AB-Bind supplementary table (csv/xlsx), optional")
    ap.add_argument("--supplement-sheet", default=None)
    ap.add_argument("--offline", action="store_true", help="do not download; validate existing files")
    ap.add_argument("--pdb-col"); ap.add_argument("--chain-col")
    ap.add_argument("--wt-col"); ap.add_argument("--resid-col"); ap.add_argument("--homology-col")
    args = ap.parse_args()

    out = Path(args.outdir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 1 -- ids
    df, cols, homol = load_labels(args.labels, args)
    df, rcsb_ids, other_ids = split_ids(df, cols, homol)
    Path("pdb_ids.txt").write_text("\n".join(rcsb_ids) + "\n")
    Path("pdb_ids_nonstandard.txt").write_text("\n".join(other_ids) + ("\n" if other_ids else ""))
    print(f"[ids] {len(rcsb_ids)} RCSB-style entries -> pdb_ids.txt")
    print(f"[ids] {len(other_ids)} non-RCSB / homology-flagged entries -> pdb_ids_nonstandard.txt: {other_ids}")
    for hm in other_ids:
        if hm in HM_TEMPLATES:
            print(f"[ids]   {hm}: homology model, {HM_TEMPLATES[hm]}")
    print(f"[ids] rows per complex:\n{df.groupby('_pid').size().to_string()}")

    # 2 -- supplement
    if args.supplement:
        check_supplement(args.supplement, args.supplement_sheet, rcsb_ids + other_ids)

    # 3 -- download
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    log, paths = [], {}
    for pid in rcsb_ids:
        if args.offline:
            hit = next((raw / f"{pid}.{e}" for e in fmts if (raw / f"{pid}.{e}").exists()), None)
            p, ext, st = (hit, hit.suffix[1:], "offline-found") if hit else (None, None, "MISSING")
        else:
            p, ext, st = download_one(pid, raw, fmts)
            time.sleep(0.3)  # be polite to RCSB
        if p:
            paths[pid] = p
        log.append({"pdb": pid, "format": ext, "status": st,
                    "bytes": p.stat().st_size if p else 0,
                    "sha256": sha256(p) if p else ""})
        print(f"[get] {pid}: {st}")
    pd.DataFrame(log).to_csv(out / "download_log.csv", index=False)

    with open(out / "MANIFEST.txt", "w") as fh:
        fh.write(f"# generated {dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat()}Z from {args.labels}\n")
        for e in log:
            if e["sha256"]:
                fh.write(f"{e['sha256']}  raw/{e['pdb']}.{e['format']}\n")
    (out / "manifest.json").write_text(json.dumps(
        {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z", "source": "files.rcsb.org",
         "entries": log}, indent=2))

    # 4 -- validate
    structures = {}
    for pid, p in paths.items():
        s = load_structure(p)
        if s is None:
            print(f"[warn] {pid}: .cif parsing needs biopython (pip install biopython); skipped")
            continue
        structures[pid] = s
    rep = validate_rows(df, cols, structures)
    rep.to_csv(out / "validation_rows.csv", index=False)

    summ = rep.groupby("pdb")["status"].value_counts().unstack(fill_value=0)
    for c in ["OK", "WT_MISMATCH", "RESIDUE_MISSING", "CHAIN_MISSING", "BAD_RESID", "NO_STRUCTURE"]:
        if c not in summ:
            summ[c] = 0
    summ["n_rows"] = summ.sum(axis=1)
    summ["frac_ok"] = (summ["OK"] / summ["n_rows"]).round(3)
    notes = offset_diagnostic(df, cols, rep, structures)
    summ["note"] = [
        "; ".join(v for (p, ch), v in notes.items() if p == pid) for pid in summ.index]
    summ.to_csv(out / "validation_summary.csv")

    n_ok = int((rep["status"] == "OK").sum())
    print("\n================ SUMMARY ================")
    print(summ.to_string())
    print(f"\nrows validated OK: {n_ok}/{len(rep)}")
    print(f"rows with no structure (homology models / failed downloads): "
          f"{int((rep['status'] == 'NO_STRUCTURE').sum())}")
    print("\nFiles written:", out / "MANIFEST.txt", out / "download_log.csv",
          out / "validation_rows.csv", out / "validation_summary.csv", sep="\n  ")
    if n_ok < len(rep):
        print("\nNEXT: read validation_summary.csv. Complexes with frac_ok < 1 need a numbering fix or the "
              "AB-Bind prepared structure before any mutant building.")


if __name__ == "__main__":
    main()
