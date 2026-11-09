#!/usr/bin/env python3
"""Build the row tables the feature generator consumes.

Panel structures come from complex_map_final.csv column ``file``. Five of the
186 complexes point at data/structures/antibody_rbd_converted/. The residue
number stored in that file is the number passed to the generator. A chain-map
shift is used only when the paper residue number is absent and the shifted
number carries the paper wild-type letter. On 7B27 the paper numbers are
already present (GLY 339 is residue 339), so the recorded shift of +12 is not
applied a second time.

SKEMPI, Starr, and Chan rows come from data/labels/reconstructed_labels.csv.
That table already dropped residues that are not in the structure. Its
``structure`` column is copied through. S1131 rows keep the joined order
required by scripts/35_compare_features.py.

Does not build mutant coordinates and does not drop a row whose wild-type
letter fails. Those rows are marked wt_ok=0 and fail later in the generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from topnet.mutations import FIG2_ORDER, split  # noqa: E402
from topnet.sources import read_pdb_residues  # noqa: E402

CONVERTED = (
    "6ZBP",
    "7B27",
    "7OAY",
    "7OAP",
    "7OAO",
)
SITES = [split(mutation) for mutation in FIG2_ORDER]


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    rooted = ROOT / path
    if rooted.is_file():
        return rooted
    return rooted


def chain_residues(path: Path) -> dict[str, dict[str, str]]:
    return {chain: {key.upper(): value for key, value in residues.items()} for chain, residues in read_pdb_residues(path).items()}


def chainmap_shift(resolved_id: str, rbd_chain: str) -> int:
    path = ROOT / "data" / "structures" / "antibody_rbd_converted" / f"{resolved_id}.chainmap.json"
    if not path.is_file():
        return 0
    meta = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for original, new in meta.get("original_to_new", {}).items():
        if new == rbd_chain and original in meta.get("shifts", {}):
            found.append(int(meta["shifts"][original]))
    if len(found) > 1:
        raise SystemExit(f"{resolved_id} chain {rbd_chain} has more than one chain-map shift: {found}")
    return found[0] if found else 0


def numbering_offset(resolved_id: str, rbd_chain: str, residues: dict[str, str]) -> tuple[int, str, int]:
    """Offset added to the paper residue number. Prefer 0 when the wild type is already there."""
    shift = chainmap_shift(resolved_id, rbd_chain)
    offsets = [0] if shift == 0 else [0, shift, -shift]

    def score(offset: int) -> int:
        return sum(residues.get(str(resid + offset), "") == wt for wt, resid, _mut in SITES)

    best = max(offsets, key=lambda offset: (score(offset), -abs(offset)))
    rule = "paper_resid" if best == 0 else "paper_plus_recorded_shift" if best == shift else "paper_minus_recorded_shift"
    return best, rule, score(best)


def mutant_path(pdb: str, chain: str, wt: str, resid: int, mut: str) -> str:
    return f"data/features/mutants/{pdb}_{chain}_{wt}_{resid}_{mut}.pdb"


def panel_rows(cmap_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cmap = pd.read_csv(cmap_path, dtype=str, keep_default_na=False)
    library = cmap[(cmap["extra"] == "0") & (cmap["qc_pass"].isin(["True", "true", "1"]))].copy()
    if len(library) != 186:
        raise SystemExit(f"expected 186 QC-passing library rows, found {len(library)}")
    if library["label"].duplicated().any():
        raise SystemExit("complex_map_final labels are not unique")
    converted = sorted(library.loc[library["file"].str.contains("antibody_rbd_converted"), "resolved_id"])
    if converted != sorted(CONVERTED):
        raise SystemExit(f"converted PDB ids are {converted}, expected {list(CONVERTED)}")

    rows = []
    audits = []
    for rec in library.itertuples(index=False):
        path = resolve(rec.file)
        residues = chain_residues(path).get(rec.rbd_chain, {})
        offset, rule, matched = numbering_offset(rec.resolved_id, rec.rbd_chain, residues)
        audits.append({
            "label": rec.label,
            "resolved_id": rec.resolved_id,
            "file": rec.file,
            "rbd_chain": rec.rbd_chain,
            "offset": offset,
            "rule": rule,
            "wt_matches": matched,
            "n_sites": len(SITES),
        })
        for mutation, (wt, resid, mut) in zip(FIG2_ORDER, SITES):
            file_resid = resid + offset
            rows.append({
                "mutation": mutation,
                "label": rec.label,
                "pdb": rec.resolved_id,
                "chain": rec.rbd_chain,
                "wt": wt,
                "resid": resid,
                "resid_struct": file_resid,
                "mut": mut,
                "structure": rec.file,
                "mutant": mutant_path(rec.resolved_id, rec.rbd_chain, wt, file_resid, mut),
                "wt_ok": int(residues.get(str(file_resid), "") == wt),
                "numbering_rule": rule,
            })
    return pd.DataFrame(rows), pd.DataFrame(audits)


def dataset_rows(labels: pd.DataFrame, dataset: str) -> pd.DataFrame:
    part = labels[labels["dataset"] == dataset].copy()
    if part.empty:
        raise SystemExit(f"{dataset} has no rows in reconstructed_labels.csv")
    part["resid_struct"] = part["resid"]
    part["complex_id"] = part["pdb"]
    part["mutant"] = [
        mutant_path(pdb, chain, wt, int(resid), mut)
        for pdb, chain, wt, resid, mut in zip(part["pdb"], part["chain"], part["wt"], part["resid"], part["mut"])
    ]
    part["wt_ok"] = 1
    keep = ["row_idx", "dataset", "pdb", "complex_id", "chain", "wt", "resid", "resid_struct", "mut", "ddg",
            "ddg_is_proxy", "ab_bind_overlap", "structure", "mutant", "wt_ok"]
    return part[keep].reset_index(drop=True)


def s1131_rows(joined_path: Path, singles_path: Path) -> pd.DataFrame:
    joined = pd.read_csv(joined_path, dtype=str, keep_default_na=False)
    singles = pd.read_csv(singles_path, dtype=str, keep_default_na=False)
    keys = ["pdb", "wt", "mut"]
    right = singles.rename(columns={"chain": "chain_s", "resid": "resid_s"})
    merged = joined.merge(
        right[keys + ["chain_s", "resid_s", "resid_struct", "ab_bind_overlap", "ddg"]],
        on=keys + ["chain_s", "resid_s"],
        how="left",
        suffixes=("", "_singles"),
    )
    if len(merged) != len(joined):
        raise SystemExit(f"S1131 join expanded from {len(joined)} to {len(merged)} rows")
    missing = merged["resid_struct"].eq("").sum()
    if missing:
        raise SystemExit(f"{int(missing)} joined S1131 rows have no resid_struct")
    merged["structure"] = merged["pdb"].map(lambda pdb: f"data/raw/skempi/pdbs/{pdb}.pdb")
    merged["chain"] = merged["chain_s"]
    merged["resid"] = merged["resid_struct"]
    merged["mutant"] = [
        mutant_path(pdb, chain, wt, resid, mut)
        for pdb, chain, wt, resid, mut in zip(
            merged["pdb"], merged["chain"], merged["wt"], merged["resid_struct"], merged["mut"]
        )
    ]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/features/rows")
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel, audit = panel_rows(ROOT / "data/structures/antibody_rbd/complex_map_final.csv")
    panel.to_csv(out / "panel.csv", index=False)
    audit.to_csv(out / "panel_numbering.csv", index=False)
    labels = pd.read_csv(ROOT / "data/labels/reconstructed_labels.csv", dtype=str, keep_default_na=False)
    for dataset in ("skempi2", "starr2020", "chan2020"):
        dataset_rows(labels, dataset).to_csv(out / f"{dataset}.csv", index=False)
    s1131 = s1131_rows(
        ROOT / "data/raw/skempi/s1131_joined.csv",
        ROOT / "data/raw/skempi/skempi_v2_single_labels.csv",
    )
    s1131.to_csv(ROOT / "data/raw/skempi/s1131_for_features.csv", index=False)
    failed = panel[panel["wt_ok"] == 0]
    print(f"panel {len(panel)} rows, wt mismatches {len(failed)}, numbering rules {audit['rule'].value_counts().to_dict()}")
    print(f"s1131 feature rows {len(s1131)}")
    for dataset in ("skempi2", "starr2020", "chan2020"):
        print(dataset, sum(1 for _ in open(out / f"{dataset}.csv", encoding="utf-8")) - 1)


if __name__ == "__main__":
    main()
