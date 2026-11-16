"""H0, H1 and H2 blocks from a wild-type PDB and a mutant PDB.

This is the topology half of ``capsule-2202829/code/src/gen_feature_topo``
in Python. MATLAB, JavaPlex, R and VMD are not called. Nothing in this
module writes a feature for a residue it cannot find, and it does not build
the mutant coordinates. Pass the mutant structure in yourself.

    from topnet.features.homology import homology_blocks

    blocks = homology_blocks("wild.pdb", "mutant.pdb", chain="H", resid=100, icode="A")
    blocks.h0    # (48, 108) float64. This is the CNN input.
    blocks.h12   # (756,) float64, or None if you passed compute_h12=False.

The structures are protein complexes (antibody-antigen, spike-receptor, or
any other complex). The groups come from C, N and O atom coordinates, not
from the sequence.

Neighborhoods
-------------
The capsule asks VMD for two subsets, both at 12 angstroms
(``gen_mut_structure.py``):

- Binding: ``within 12 of chain <chain>``. Every atom of that chain, plus
  every atom within 12 A of one of them. The cloud is then split into
  "on the chain" and "not on the chain".
- Mutation: ``within 12 of resid <resid> and chain <chain>``. Every atom of
  that residue, plus every atom within 12 A of one of them. The cloud is
  split into "that residue" and "everything else in the subset".

Hydrogens and other elements take part in the distance test, because VMD
sees the whole PDB. Only C, N and O (PDB column 14, the atom-name letter)
enter the complexes. Waters and other HETATM records are ignored. Every
alternate location is kept, which is what ``Gen_pointclouds.py`` does.

H0
--
For each of the nine element pairs CC, CN, CO, NC, NN, NO, OC, ON, OO, the
two sides are a bipartite Vietoris-Rips complex: a distance is kept only
when the two atoms are on opposite sides. Finite deaths are the edge lengths
of a minimum spanning tree. Deaths longer than 15 A are dropped, and the
rest are snapped to ``k * 15/200`` angstroms, then to 4 decimal places.
That is the JavaPlex stream in ``Construct_jplex_binding.m``.

Those deaths are counted in 48 bins of width 0.25 A on [0, 12). A death on
a bin edge goes in the higher bin. The tensor columns are:

- 0:27    binding deaths
- 27:54   binding persistence (how many bars are still alive before that bin)
- 54:81   mutation deaths
- 81:108  mutation persistence

Inside each group of 27 columns: 9 wild-type pairs, 9 mutant pairs, 9
wild-type minus mutant. ``quantize=False`` skips the 15 A grid and keeps
the raw lengths. Binning still drops anything at or above 12 A.

H1 and H2
---------
The same neighborhoods and the same nine pairs, but the two sides are
concatenated into one point cloud and the alpha complex is computed with
Gudhi. Bars in dimension 1 and dimension 2 are reduced to the seven numbers
``gen_feature_12.py`` stores. Gudhi must be installed
(``pip install gudhi``). A missing install raises ImportError. It does not
substitute a zero vector.

The 756 numbers are binding H1, binding H2, mutation H1, mutation H2.
Inside each of those, the last of every three entries is mutant minus wild.

``legacy_resid=True`` matches the capsule's residue key, PDB columns 24-26,
which drops the thousands digit and the insertion code. The default keeps
the full residue number and the insertion code, so residue 100 and 100A
stay distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from topnet.features.h0 import ELEMENTS, PAIRS, capsule_h0_deaths, histogram_deaths
from topnet.features.h12 import H12_DIM, h12_from_neighborhoods
from topnet.features.pdbio import coords_of, read_pdb_records

H0_SHAPE = (48, 108)
BIND_DEATH = slice(0, 27)
BIND_PERSISTENCE = slice(27, 54)
MUT_DEATH = slice(54, 81)
MUT_PERSISTENCE = slice(81, 108)
# Inside one 27-column block.
WILD_PAIRS = slice(0, 9)
MUTANT_PAIRS = slice(9, 18)
DIFF_PAIRS = slice(18, 27)


@dataclass(frozen=True)
class HomologyBlocks:
    """One mutation.

    ``h12`` is None when ``compute_h12`` was false. It is never an array of
    zeros standing in for a diagram that was not computed.
    """

    h0: np.ndarray
    h12: np.ndarray | None
    chain: str
    resid: int
    icode: str
    cutoff: float
    quantized: bool
    keep_chains: tuple = ()


def parse_site(resid: int | str, icode: str | None = None) -> tuple[int, str]:
    """Residue number plus insertion code.

    ``resid=100, icode="A"`` and ``resid="100A"`` are the same site.
    ``icode=""`` means no insertion code.
    """
    if icode is None:
        text = str(resid).strip()
        digits = []
        suffix = []
        for char in text:
            if char.isdigit() or (char == "-" and not digits and not suffix):
                if suffix:
                    raise ValueError(f"Cannot read residue {resid!r}")
                digits.append(char)
            else:
                suffix.append(char)
        if not digits:
            raise ValueError(f"Cannot read residue {resid!r}")
        return int("".join(digits)), "".join(suffix).strip()
    return int(str(resid).strip()), str(icode).strip()


def legacy_residue_key(resseq: int) -> str:
    """PDB columns 24-26: the residue number without its thousands digit."""
    return f"{int(resseq):4d}"[1:].strip()


def _heavy(records: list[dict]) -> list[dict]:
    return [rec for rec in records if rec["element"] in ELEMENTS]


def atoms_within(records: list[dict], query: list[dict], cutoff: float) -> list[dict]:
    """Query atoms, plus every record within ``cutoff`` angstroms of one of them.

    This is VMD's ``within <cutoff> of <selection>``, which includes the
    selection itself. Comparison is ``<= cutoff``. One vectorised KD-tree
    query for all records.
    """
    if not query or not records:
        return []
    query_ids = {id(rec) for rec in query}
    tree = cKDTree(np.asarray([rec["xyz"] for rec in query], dtype=np.float64))
    dist, _ = tree.query(np.asarray([rec["xyz"] for rec in records], dtype=np.float64), k=1)
    return [
        rec for rec, d in zip(records, dist) if id(rec) in query_ids or float(d) <= cutoff
    ]


def _is_site(rec: dict, chain: str, resid: int, icode: str, legacy_resid: bool) -> bool:
    if rec["chain"] != chain:
        return False
    if legacy_resid:
        return rec["legacy_resid"] == legacy_residue_key(resid)
    return rec["resseq_int"] == resid and rec["icode"] == icode


def binding_neighborhood(records: list[dict], chain: str, cutoff: float) -> list[dict]:
    query = [rec for rec in records if rec["chain"] == chain]
    return atoms_within(records, query, cutoff)


def mutation_neighborhood(
    records: list[dict],
    chain: str,
    resid: int,
    icode: str,
    cutoff: float,
    legacy_resid: bool,
) -> list[dict]:
    query = [rec for rec in records if _is_site(rec, chain, resid, icode, legacy_resid)]
    return atoms_within(records, query, cutoff)


def split_binding(neighborhood: list[dict], chain: str) -> tuple[list[dict], list[dict]]:
    mine = [rec for rec in neighborhood if rec["chain"] == chain]
    partner = [rec for rec in neighborhood if rec["chain"] != chain]
    return mine, partner


def split_mutation(
    neighborhood: list[dict],
    chain: str,
    resid: int,
    icode: str,
    legacy_resid: bool,
) -> tuple[list[dict], list[dict]]:
    site = [rec for rec in neighborhood if _is_site(rec, chain, resid, icode, legacy_resid)]
    rest = [rec for rec in neighborhood if not _is_site(rec, chain, resid, icode, legacy_resid)]
    return site, rest


def _require_sides(name: str, structure: str, side_a: list[dict], side_b: list[dict]) -> None:
    """Refuse an empty interface. A zero block would look like a real diagram."""
    if not _heavy(side_a) or not _heavy(side_b):
        raise ValueError(
            f"{structure} {name} neighborhood has no C/N/O atoms on one side "
            f"({len(_heavy(side_a))} and {len(_heavy(side_b))}). "
            "No homology block was written."
        )


def _nine_counts(side_a: list, side_b: list, quantize: bool) -> tuple[np.ndarray, np.ndarray]:
    death = np.zeros((48, 9), dtype=np.float64)
    persist = np.zeros((48, 9), dtype=np.float64)
    for index, (e1, e2) in enumerate(PAIRS):
        deaths = capsule_h0_deaths(coords_of(side_a, e1), coords_of(side_b, e2), quantize=quantize)
        death[:, index], persist[:, index] = histogram_deaths(deaths)
    return death, persist


def _pack(wild: np.ndarray, mutant: np.ndarray) -> np.ndarray:
    """(48, 9) wild, mutant, and wild minus mutant -> (48, 27)."""
    return np.concatenate([wild, mutant, wild - mutant], axis=1)


def h0_from_neighborhoods(
    wild_binding: tuple[list, list],
    mutant_binding: tuple[list, list],
    wild_mutation: tuple[list, list],
    mutant_mutation: tuple[list, list],
    *,
    quantize: bool = True,
) -> np.ndarray:
    """(48, 108) tensor. Each argument is ``(side_one, side_two)``."""
    wild_b = _nine_counts(*wild_binding, quantize)
    mut_b = _nine_counts(*mutant_binding, quantize)
    wild_m = _nine_counts(*wild_mutation, quantize)
    mut_m = _nine_counts(*mutant_mutation, quantize)
    tensor = np.concatenate(
        [
            _pack(wild_b[0], mut_b[0]),
            _pack(wild_b[1], mut_b[1]),
            _pack(wild_m[0], mut_m[0]),
            _pack(wild_m[1], mut_m[1]),
        ],
        axis=1,
    )
    if tensor.shape != H0_SHAPE:
        raise RuntimeError(f"H0 tensor has shape {tensor.shape}, expected {H0_SHAPE}")
    if not np.isfinite(tensor).all():
        raise RuntimeError("H0 tensor has a non-finite value. No file was written.")
    return tensor


def parse_keep_chains(keep_chains: Union[None, str, Iterable[str]]) -> tuple:
    """``"H+L+E"``, ``"HLE"`` (single-letter ids only), or an iterable of ids."""
    if keep_chains is None:
        return ()
    if isinstance(keep_chains, str):
        text = keep_chains.strip()
        if not text:
            return ()
        parts = text.replace(",", "+").split("+") if ("+" in text or "," in text) else list(text)
    else:
        parts = list(keep_chains)
    out = []
    for part in parts:
        part = str(part)
        if len(part) != 1:
            raise ValueError(f"PDB chain ids are one character, got {part!r} in {keep_chains!r}")
        if part not in out:
            out.append(part)
    return tuple(out)


def restrict_to_chains(records: list[dict], keep: tuple, chain: str, label: str) -> list[dict]:
    """Drop every record whose chain is not in ``keep``.

    The binding block is "this chain against everything within 12 A". When the
    PDB holds more than the complex you mean (other spike protomers, a second
    Fab, ACE2 next to an antibody), those atoms would count as partner. Call
    this with the frozen chain set of the complex. ``keep`` must contain the
    mutated chain, and every chain in it must exist in the file.
    """
    if not keep:
        return records
    if chain not in keep:
        raise ValueError(f"{label}: mutated chain {chain!r} is not in keep_chains {keep}")
    present = {rec["chain"] for rec in records}
    missing = [c for c in keep if c not in present]
    if missing:
        raise ValueError(f"{label}: chains {missing} not in file (file has {sorted(present)})")
    return [rec for rec in records if rec["chain"] in keep]


def _as_records(source, altloc: str, label: str = "Wild") -> list[dict]:
    """A path (parsed here) or a record list you already parsed with read_pdb_records."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"{label} structure not found: {path}")
        return read_pdb_records(path, altloc=altloc)
    return source


def _prepared_neighborhoods(
    wild_pdb: str | Path,
    mutant_pdb: str | Path,
    chain: str,
    resid: int | str,
    icode: str | None,
    cutoff: float,
    legacy_resid: bool,
    altloc: str,
    keep_chains: tuple = (),
):
    number, insertion = parse_site(resid, icode)
    chain = str(chain)
    if len(chain) != 1:
        raise ValueError(f"chain must be one PDB chain id, got {chain!r}")
    wild = _as_records(wild_pdb, altloc, "Wild")
    mutant = _as_records(mutant_pdb, altloc, "Mutant")
    wild = restrict_to_chains(wild, keep_chains, chain, "wild")
    mutant = restrict_to_chains(mutant, keep_chains, chain, "mutant")
    sides = {}
    for label, records in (("wild", wild), ("mutant", mutant)):
        binding = binding_neighborhood(records, chain, cutoff)
        mutation = mutation_neighborhood(records, chain, number, insertion, cutoff, legacy_resid)
        bind_sides = split_binding(binding, chain)
        mut_sides = split_mutation(mutation, chain, number, insertion, legacy_resid)
        _require_sides("binding", label, *bind_sides)
        _require_sides("mutation", label, *mut_sides)
        sides[label] = (bind_sides, mut_sides)
    return number, insertion, sides


def homology_blocks(
    wild_pdb: str | Path,
    mutant_pdb: str | Path,
    chain: str,
    resid: int | str,
    icode: str | None = None,
    *,
    cutoff: float = 12.0,
    legacy_resid: bool = False,
    quantize: bool = True,
    compute_h12: bool = True,
    altloc: str = "all",
    keep_chains: Union[None, str, Sequence[str]] = None,
) -> HomologyBlocks:
    """Build the H0 tensor and, unless turned off, the 756-d H1/H2 vector.

    ``cutoff`` is the VMD neighborhood radius in angstroms. The capsule uses
    12. ``quantize`` snaps H0 deaths to the JavaPlex grid. ``altloc`` is
    ``"all"`` (capsule), ``"blank"`` or ``"first"``.

    ``wild_pdb`` and ``mutant_pdb`` are paths or record lists from
    ``read_pdb_records`` (parse a wild-type file once and reuse it across the
    mutations of one complex). ``keep_chains`` (``"H+L+E"``) restricts both
    structures to the chains of the complex before any neighborhood is cut.
    """
    keep = parse_keep_chains(keep_chains)
    number, insertion, sides = _prepared_neighborhoods(
        wild_pdb, mutant_pdb, chain, resid, icode, cutoff, legacy_resid, altloc, keep
    )
    wild_b, wild_m = sides["wild"]
    mut_b, mut_m = sides["mutant"]
    h0 = h0_from_neighborhoods(wild_b, mut_b, wild_m, mut_m, quantize=quantize)
    h12 = None
    if compute_h12:
        h12 = h12_from_neighborhoods(wild_b, mut_b, wild_m, mut_m)
        if h12.shape != (H12_DIM,) or not np.isfinite(h12).all():
            raise RuntimeError(f"H1/H2 vector is unusable (shape {h12.shape}). No file was written.")
    return HomologyBlocks(
        h0=h0,
        h12=h12,
        chain=str(chain),
        resid=number,
        icode=insertion,
        cutoff=float(cutoff),
        quantized=bool(quantize),
        keep_chains=keep,
    )
