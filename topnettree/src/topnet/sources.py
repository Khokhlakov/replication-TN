"""Shared helpers for the data-recovery scripts (network, LFS detection, mutation parsing, units)."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "topnet-data-recovery/0.1"}
R_KCAL = 0.0019872041  # kcal / (mol K)
T_DEFAULT = 298.15
RT_LN10 = R_KCAL * T_DEFAULT * math.log(10.0)  # ~1.364 kcal/mol per log10 unit at 298.15 K

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
RES3TO1 = {k.upper(): v for k, v in AA3.items()}
RES3TO1.update({"MSE": "M", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C"})

SKEMPI_TOKEN = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")
PLAIN_TOKEN = re.compile(r"^([A-Z])(\d+[A-Za-z]?)([A-Z])$")


def http_get(url: str, timeout: int = 60, retries: int = 3, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = dict(UA)
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read() if byte_range is None else resp.read(byte_range[1] - byte_range[0] + 1)
        except urllib.error.HTTPError as err:
            if err.code in (400, 401, 403, 404):
                raise
            last = err
        except Exception as err:  # network flake
            last = err
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last!r}")


def try_get(url: str, **kw) -> tuple[bytes | None, str]:
    """Returns (data, status). status is 'ok' or a short error string. Never raises."""
    try:
        return http_get(url, **kw), "ok"
    except urllib.error.HTTPError as err:
        return None, f"http {err.code}"
    except Exception as err:
        return None, f"{type(err).__name__}: {err}"[:160]


def get_json(url: str):
    data, status = try_get(url, timeout=30, retries=2)
    if data is None:
        return None, status
    try:
        return json.loads(data), "ok"
    except Exception as err:
        return None, f"bad json: {err}"


def is_lfs_pointer(data: bytes) -> bool:
    return data[:40].startswith(b"version https://git-lfs")


def looks_like_html(data: bytes) -> bool:
    head = data[:200].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def parse_skempi_token(tok: str):
    """'LI45G' -> ('L', 'I', '45', 'G'). Returns None when it does not parse."""
    m = SKEMPI_TOKEN.match(tok.strip())
    return m.groups() if m else None


def parse_plain_token(tok: str):
    """'N501Y' -> ('N', '501', 'Y')."""
    m = PLAIN_TOKEN.match(tok.strip())
    return m.groups() if m else None


def find_col(columns, *patterns, required: bool = True):
    """First column matching any regex (case-insensitive). Lists the columns on failure."""
    cols = list(columns)
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(str(c)):
                return c
    if required:
        raise KeyError(f"none of {patterns} matched. Columns are: {cols}")
    return None


_CIF_TOKEN = re.compile(r"""'(?:[^']|'(?!\s))*'|"(?:[^"]|"(?!\s))*"|\S+""")


def read_cif_residues(path: Path) -> dict[str, dict[str, str]]:
    """Same output as read_pdb_residues for an mmCIF file. Uses author chain ids and author residue numbers
    (the numbering the paper's mutation names follow). First model, CA atoms, altloc '.'/A, HETATM only MSE."""
    out: dict[str, dict[str, str]] = {}
    cols: list[str] = []
    in_loop_header = False
    in_atoms = False
    first_model = None
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("loop_"):
                cols, in_loop_header, in_atoms = [], True, False
                continue
            if in_loop_header:
                if line.startswith("_"):
                    cols.append(line.split()[0])
                    continue
                in_loop_header = False
                in_atoms = bool(cols) and cols[0].startswith("_atom_site.")
            if not in_atoms:
                continue
            if line.startswith(("#", "loop_", "_")):
                if line.startswith("#"):
                    in_atoms = False
                continue
            tok = [t.strip("'\"") if len(t) > 1 and t[0] in "'\"" else t for t in _CIF_TOKEN.findall(line)]
            if len(tok) != len(cols):
                continue
            rec = dict(zip(cols, tok))
            model = rec.get("_atom_site.pdbx_PDB_model_num", "1")
            first_model = first_model or model
            if model != first_model:
                break
            if rec["_atom_site.label_atom_id"] != "CA" or rec.get("_atom_site.label_alt_id", ".") not in (".", "A"):
                continue
            resname = rec["_atom_site.label_comp_id"].upper()
            if rec["_atom_site.group_PDB"] == "HETATM" and resname != "MSE":
                continue
            ins = rec.get("_atom_site.pdbx_PDB_ins_code", "?")
            key = rec["_atom_site.auth_seq_id"] + ("" if ins in ("?", ".") else ins)
            out.setdefault(rec["_atom_site.auth_asym_id"], {})[key] = RES3TO1.get(resname, "X")
    return out


def read_pdb_residues(path: Path) -> dict[str, dict[str, str]]:
    """chain -> {'resseq+icode': one-letter}. Reads CA records of the first model only (.pdb or .cif)."""
    if str(path).lower().endswith(".cif"):
        return read_cif_residues(path)
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            resname = line[17:20].strip().upper()
            if line.startswith("HETATM") and resname != "MSE":
                continue
            key = line[22:26].strip() + line[26].strip()
            out.setdefault(line[21], {})[key] = RES3TO1.get(resname, "X")
    return out
