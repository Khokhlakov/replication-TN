"""Copy AB-Bind-prepared PDBs and check chain, resseq, icode and wild-type residue.

A complex with any failing row is excluded. Nothing in the PDB is rewritten.
The secondary RCSB arm runs only when data/structures/raw has files and
data/structures/chain_map.csv exists.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topnet.config import load_config, repo_root
from topnet.data import assert_label_tables
from topnet.features.pdbio import AA1, read_pdb_records

CAPSULE = Path(__file__).resolve().parents[2] / "capsule-2202829" / "data" / "AB-Bind-Database"
CRYSTAL = [
    "1AK4", "1BJ1", "1CZ8", "1DQJ", "1DVF", "1FFW", "1JRH", "1JTG", "1KTZ", "1MHP",
    "1MLC", "1N8Z", "1VFB", "1YY9", "2JEL", "2NYY", "2NZ9", "3BDY", "3BE1", "3BN9",
    "3HFM", "3K2M", "3NGB", "3NPS",
]
HOMOLOGY = ["HM_1KTZ", "HM_1YY9", "HM_2NYY", "HM_2NZ9", "HM_3BN9"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _index(records):
    found = {}
    for rec in records:
        key = (rec["chain"], rec["resseq_int"], rec["icode"])
        found.setdefault(key, set()).add(rec["resname"])
    return found


def main() -> int:
    cfg = load_config("configs/default.yaml")
    _y, meta, sup = assert_label_tables(cfg)
    root = repo_root()
    dest = root / "data" / "structures" / "abbind"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in CRYSTAL + HOMOLOGY:
        source = CAPSULE / f"{name}.pdb"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = dest / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, target)
        manifest.append(f"{_sha256(target)}  {target.name}")
    (dest / "MANIFEST_abbind.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    merged = meta.copy()
    merged["icode"] = sup["icode"].astype(str).to_numpy()
    merged["resid_num"] = sup["resid"].to_numpy()
    excluded = []
    checked = 0
    empty = []
    for pdb, part in merged.groupby(merged["pdb"].astype(str), sort=False):
        path = dest / f"{pdb}.pdb"
        if path.stat().st_size == 0:
            empty.append(str(pdb))
            excluded.append(str(pdb))
            continue
        found = _index(read_pdb_records(path))
        bad = []
        for rec in part.itertuples(index=False):
            icode = "" if rec.icode in ("", "nan") else str(rec.icode)
            key = (str(rec.chain), int(rec.resid_num), icode)
            names = found.get(key, set())
            expected = AA1[str(rec.wt)]
            accepted = {expected}
            if expected == "HIS":
                accepted.update({"HSE", "HSD", "HSP", "HID", "HIE", "HIP"})
            if names.isdisjoint(accepted):
                bad.append(int(rec.row_idx))
            else:
                checked += 1
        if bad:
            excluded.append(str(pdb))
            print(f"FAIL {pdb}: {len(bad)} rows, first row_idx {bad[:8]}")
        else:
            print(f"OK {pdb}: {len(part)} rows")
    keep = merged.loc[~merged["pdb"].astype(str).isin(excluded), "row_idx"]
    keep_path = root / "data" / "structures" / "abbind_keep_rows.txt"
    keep_path.write_text("\n".join(str(int(i)) for i in keep) + "\n", encoding="utf-8")
    print(f"validated rows {checked}. excluded complexes {excluded or 'none'}. empty files {empty or 'none'}.")
    print(f"N kept {len(keep)}. expected 645 if nothing failed.")
    raw = root / "data" / "structures" / "raw"
    chain_map = root / "data" / "structures" / "chain_map.csv"
    n_raw = len(list(raw.glob("*.pdb"))) if raw.is_dir() else 0
    if n_raw == 0:
        print("SECONDARY RCSB arm skipped: data/structures/raw has no PDB files.")
    elif not chain_map.is_file():
        print("SECONDARY RCSB arm skipped: data/structures/chain_map.csv is missing. chain_map_suggestions.csv is not used.")
    else:
        print(f"SECONDARY RCSB arm not implemented in this run beyond the file check. raw pdbs {n_raw}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
