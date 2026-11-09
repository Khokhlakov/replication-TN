#!/usr/bin/env python3
"""Write one H0 vector for a mutation.

Geometry is scripts/reproduce_h0.py: 12 A neighborhood, every alternate location,
collapsed insertion codes, JavaPlex-style quantization, 48 bins. The wild-type
PDB and a prebuilt mutant PDB are both required. A missing file, a missing
site, or a wild-type letter that does not match --wt is a failed row. This
script does not write a zero vector in those cases.

The residue passed in --resid must be the number printed in the PDB (the
complex_map file's numbering). For the five converted PDBs that number is
already the paper residue: 7B27.chainmap.json records a shift that was applied
when the PDB was written, and GLY 339 is residue 339 in that file.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rh0", HERE / "reproduce_h0.py")
rh0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh0)

sys.path.insert(0, str(HERE.parent / "src"))
from topnet.sources import read_pdb_residues  # noqa: E402


def legacy_key(resid: str) -> str:
    """PDB columns 24-26, matching reproduce_h0._legacy_key."""
    text = str(resid).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    suffix = "".join(ch for ch in text if ch.isalpha())
    if not digits:
        return text
    return f"{int(digits):4d}"[1:].strip() + suffix


def prepare(path: str) -> list[dict]:
    records = rh0.read_pdb_records(path, altloc="all")
    for rec in records:
        rec["site"] = rec["legacy_resid"]
    return records


def parts(records: list[dict], chain: str, site: str):
    neighborhood = rh0.binding_atoms(records, chain, 12.0)
    mine = [rec for rec in neighborhood if rec["chain"] == chain]
    partner = [rec for rec in neighborhood if rec["chain"] != chain]
    death_b, per_b = rh0.nine(mine, partner, True)
    mutated = [rec for rec in neighborhood if rec["chain"] == chain and rec["site"] == site]
    rest = [rec for rec in neighborhood if not (rec["chain"] == chain and rec["site"] == site)]
    if not mutated:
        raise SystemExit(f"no atoms for chain {chain} site {site}")
    death_m, per_m = rh0.nine(mutated, rest, True)
    return death_b, per_b, death_m, per_m


def pack(wild: np.ndarray, mutant: np.ndarray) -> np.ndarray:
    return np.concatenate([wild, mutant, wild - mutant], axis=1)


def tensor(wild_path: str, mutant_path: str, chain: str, site: str) -> np.ndarray:
    wild = parts(prepare(wild_path), chain, site)
    mutant = parts(prepare(mutant_path), chain, site)
    built = np.concatenate([pack(wild[i], mutant[i]) for i in range(4)], axis=1)
    if built.shape != (48, 108) or not np.isfinite(built).all():
        raise SystemExit(f"bad tensor {built.shape}")
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wild", required=True)
    parser.add_argument("--mutant", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--resid", required=True)
    parser.add_argument("--wt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not Path(args.mutant).is_file():
        raise SystemExit(f"missing mutant structure: {args.mutant}")
    if not Path(args.wild).is_file():
        raise SystemExit(f"missing wild structure: {args.wild}")
    site = legacy_key(args.resid)
    if args.wt:
        found = read_pdb_residues(Path(args.wild)).get(args.chain, {}).get(str(args.resid).strip())
        if found != args.wt:
            raise SystemExit(f"wild type at {args.chain}:{args.resid} is {found}, expected {args.wt}")
    np.save(args.out, tensor(args.wild, args.mutant, args.chain, site).ravel())


if __name__ == "__main__":
    main()
