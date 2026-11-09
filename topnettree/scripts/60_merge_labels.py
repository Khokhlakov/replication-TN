#!/usr/bin/env python3
"""Step 6. Merge SKEMPI 2.0 singles, Starr 2020 and Chan 2020 into one label table with provenance.
Keeps only rows whose (chain, residue) exists in the structure file with the expected wild-type residue."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import read_pdb_residues  # noqa: E402

SKEMPI = Path("data/raw/skempi")
SIX = Path("data/structures/antibody_rbd/6M0J.pdb")
OUT = Path("data/labels/reconstructed_labels.csv")
_cache = {}


def residues(path):
    key = str(path)
    if key not in _cache:
        raw = read_pdb_residues(path) if path.is_file() else {}
        _cache[key] = {c: {k.upper(): v for k, v in d.items()} for c, d in raw.items()}
    return _cache[key]


def present(df):
    ok = []
    for r in df.itertuples():
        ok.append(residues(Path(r.structure)).get(r.chain, {}).get(str(r.resid).upper()) == r.wt)
    return pd.Series(ok, index=df.index)


def rd(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
 

sk = rd(SKEMPI / "skempi_v2_single_labels.csv")
sk_out = pd.DataFrame({
    "dataset": "skempi2", "pdb": sk["pdb"], "chain": sk["chain"], "wt": sk["wt"],
    "resid": sk["resid_struct"] if "resid_struct" in sk.columns else sk["resid"],
    "mut": sk["mut"], "ddg": pd.to_numeric(sk["ddg"]), "ddg_is_proxy": 0, "raw_score": float("nan"),
    "ab_bind_overlap": pd.to_numeric(sk["ab_bind_overlap"]) if "ab_bind_overlap" in sk.columns else 0,
    "structure": [str(SKEMPI / "pdbs" / (p + ".pdb")) for p in sk["pdb"]],
})


def dms(path):
    d = rd(path)
    return pd.DataFrame({
        "dataset": d["dataset"], "pdb": d["pdb"], "chain": d["chain"], "wt": d["wt"], "resid": d["resid"],
        "mut": d["mut"], "ddg": pd.to_numeric(d["ddg"]), "ddg_is_proxy": pd.to_numeric(d["ddg_is_proxy"]),
        "raw_score": pd.to_numeric(d["raw_score"]), "ab_bind_overlap": 0, "structure": str(SIX),
    })


allrows = pd.concat([sk_out, dms("data/raw/dms/starr2020_rbd_ace2_labels.csv"),
                     dms("data/raw/dms/chan2020_labels.csv")], ignore_index=True)
allrows["in_structure"] = present(allrows)
print(allrows.groupby("dataset")["in_structure"].agg(rows="size", in_structure="sum"))
dropped = allrows[~allrows["in_structure"]]
for name, g in dropped.groupby("dataset"):
    print(f"dropped from {name}: {len(g)} rows; residues (first 15): {sorted(set(g['resid']))[:15]}")
kept = allrows[allrows["in_structure"]].drop(columns="in_structure").reset_index(drop=True)
kept.insert(0, "row_idx", range(len(kept)))
OUT.parent.mkdir(parents=True, exist_ok=True)
kept.to_csv(OUT, index=False)
dropped.to_csv(OUT.with_name("reconstructed_labels_dropped.csv"), index=False)
print(f"wrote {OUT} ({len(kept)} rows); SKEMPI rows overlapping AB-Bind S645: "
      f"{int(kept.loc[kept.dataset == 'skempi2', 'ab_bind_overlap'].sum())}")