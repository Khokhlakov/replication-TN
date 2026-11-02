"""PDB atom reader aligned with Gen_pointclouds.py.

The legacy reader keeps element character ``line[13]``, chain ``line[21]``,
and residue text ``line[23:26]`` (it drops the first character of the 4-column
resSeq). For residue numbers below 1000 that dropped character is a space.
For residue numbers >= 1000 it drops the thousands digit. The default here
keeps the full resSeq plus the insertion code. Pass ``legacy_resid=True`` to
reproduce the old slice.
"""

from __future__ import annotations

from pathlib import Path

ELEMENTS = ("C", "N", "O")
AA1 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def read_pdb_records(path: str | Path, altloc: str = "all") -> list[dict]:
    """ATOM records with chain, resseq, icode and the column-14 element letter.

    ``altloc='all'`` keeps every alternate, which is what Gen_pointclouds.py does.
    ``altloc='blank'`` keeps only a blank alternate-location character.
    Waters and other HETATM lines are not read.
    """
    records = []
    seen_alt = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            element = line[13:14]
            alternate = line[16:17]
            if altloc == "blank" and alternate not in ("", " "):
                continue
            if altloc == "first" and alternate not in ("", " "):
                key = (line[21:22], line[22:26], line[26:27], line[12:16])
                if key in seen_alt:
                    continue
                seen_alt[key] = alternate
            records.append(
                {
                    "chain": line[21:22],
                    "resseq": line[22:26],
                    "resseq_int": int(line[22:26]),
                    "icode": line[26:27].strip(),
                    "resname": line[17:20].strip(),
                    "element": element,
                    "xyz": np_xyz(line),
                    "legacy_resid": line[23:26].strip(),
                }
            )
    return records


def np_xyz(line: str):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def read_pdb(path: str | Path, legacy_resid: bool = False) -> list[tuple]:
    atoms = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or len(line) < 54:
                continue
            element = line[13:14]
            if element not in ELEMENTS:
                continue
            chain = line[21:22]
            if legacy_resid:
                resid = line[23:26].strip()
            else:
                resid = line[22:26].strip()
                insertion = line[26:27].strip() if len(line) > 26 else ""
                if insertion and not insertion.isdigit():
                    resid = f"{resid}{insertion}"
            atoms.append(
                (
                    chain,
                    resid,
                    element,
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return atoms
