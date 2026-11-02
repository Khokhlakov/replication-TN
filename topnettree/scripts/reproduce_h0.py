"""WT H0 channels from AB-Bind-prepared PDBs, compared with X_unnormed.npy.

This is a Python port of the geometry in Construct_jplex_binding.m and
Gen_pointclouds.py. MATLAB and JavaPlex are not called. Gudhi is used only
as a cross-check when it imports.

The code-following defaults are: cutoff 12 A, ATOM records, elements C/N/O,
all alternate locations, insertion codes collapsed, residue keys from PDB
columns 24-26, Vietoris-Rips deaths quantized to 200 divisions over 0..15,
then the 48 bins of gen_cnn_feature.py. Only wild-type channels are compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.spatial import cKDTree

from topnet.config import load_config
from topnet.data import assert_label_tables, load_raw_cnn
from topnet.features.h0 import histogram_deaths, mst_lengths, pairwise_distances
from topnet.features.pdbio import read_pdb_records

WT_SLICES = {
    "binding_death": (0, 9),
    "binding_persistence": (27, 36),
    "mutation_death": (54, 63),
    "mutation_persistence": (81, 90),
}
PAIRS = [(a, b) for a in ("C", "N", "O") for b in ("C", "N", "O")]


def quantize(deaths: np.ndarray, max_filtration: float = 15.0, divisions: int = 200) -> np.ndarray:
    """Smallest JavaPlex division at least the Euclidean length. The Java rounding is unverified."""
    step = max_filtration / float(divisions)
    values = np.asarray(deaths, dtype=np.float64)
    if values.size == 0:
        return values
    ticks = np.ceil(np.maximum(values, 0.0) / step - 1e-12) * step
    return ticks


def bipartite_deaths(part_a: np.ndarray, part_b: np.ndarray, do_quantize: bool) -> np.ndarray:
    if len(part_a) == 0 or len(part_b) == 0:
        return np.zeros(0, dtype=np.float64)
    points = np.vstack([part_a, part_b])
    dist = pairwise_distances(points)
    n_a = len(part_a)
    cross = dist.copy()
    cross[:n_a, :n_a] = 0.0
    cross[n_a:, n_a:] = 0.0
    finite = cross[np.isfinite(cross)]
    big = float(np.max(finite) + 10.0) if finite.size else 1.0
    # Same-set distances are 100 in the MATLAB script. 100 is outside the 15 A stream.
    dist[:n_a, :n_a] = 100.0
    dist[n_a:, n_a:] = 100.0
    np.fill_diagonal(dist, 0.0)
    lengths = mst_lengths(dist)
    lengths = lengths[lengths < 50.0]
    if do_quantize:
        lengths = quantize(lengths)
    return lengths


def binding_atoms(records, chain: str, cutoff: float):
    chain_xyz = np.asarray([rec["xyz"] for rec in records if rec["chain"] == chain], dtype=np.float64)
    if len(chain_xyz) == 0:
        return []
    tree = cKDTree(chain_xyz)
    kept = []
    for rec in records:
        if rec["chain"] == chain:
            kept.append(rec)
            continue
        if tree.query(rec["xyz"], k=1)[0] <= cutoff:
            kept.append(rec)
    return kept


def coords(records, element: str) -> np.ndarray:
    rows = [rec["xyz"] for rec in records if rec["element"] == element]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def nine(records_a, records_b, do_quantize: bool):
    death = np.zeros((48, 9), dtype=np.float64)
    persist = np.zeros((48, 9), dtype=np.float64)
    for index, (e1, e2) in enumerate(PAIRS):
        deaths = bipartite_deaths(coords(records_a, e1), coords(records_b, e2), do_quantize)
        death[:, index], persist[:, index] = histogram_deaths(deaths)
    return death, persist


def gudhi_check() -> str:
    try:
        import gudhi
    except ImportError:
        return "Gudhi is not installed. Cross-check stopped. pip install gudhi"
    rng = np.random.default_rng(0)
    points = rng.normal(size=(12, 3))
    dist = pairwise_distances(points)
    np.fill_diagonal(dist, 0.0)
    ours = np.sort(mst_lengths(dist))
    complex_ = gudhi.RipsComplex(points=points, max_edge_length=float(dist.max() + 1.0))
    tree = complex_.create_simplex_tree(max_dimension=1)
    tree.compute_persistence()
    intervals = tree.persistence_intervals_in_dimension(0)
    finite = intervals[np.isfinite(intervals[:, 1])]
    theirs = np.sort(finite[:, 1].astype(np.float64))
    gap = float(np.max(np.abs(ours - theirs))) if len(ours) == len(theirs) and len(ours) else float("inf")
    return f"gudhi Rips H0 versus scipy MST max abs gap {gap:.3e} (gate 1e-9, {'PASS' if gap < 1e-9 else 'FAIL'})"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbs", default="data/structures/abbind")
    parser.add_argument("--complexes", default="1KTZ,2JEL,3BDY,3BE1,3HFM")
    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--altloc", choices=["all", "blank", "first"], default="all")
    parser.add_argument("--icode", choices=["collapsed", "respected"], default="collapsed")
    parser.add_argument("--quantize", choices=["javaplex", "continuous"], default="javaplex")
    parser.add_argument("--exclude-icode-rows", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config("configs/default.yaml")
    _y, meta, sup = assert_label_tables(cfg)
    print(gudhi_check())
    x = load_raw_cnn()
    merged = meta.copy()
    merged["icode"] = sup["icode"].astype(str).to_numpy()
    merged["resid_num"] = sup["resid"].astype(int).to_numpy()
    wanted = [item for item in args.complexes.split(",") if item]
    merged = merged[merged["pdb"].isin(wanted)]
    if args.exclude_icode_rows:
        merged = merged[merged["icode"].isin(["", "nan"])]
    pdb_dir = Path(args.pdbs)
    cache = {}
    rates = {name: [] for name in WT_SLICES}
    cells = {name: [] for name in WT_SLICES}
    for pdb, part in merged.groupby("pdb", sort=False):
        path = pdb_dir / f"{pdb}.pdb"
        records = read_pdb_records(path, altloc=args.altloc)
        if args.icode == "collapsed":
            for rec in records:
                rec["site"] = rec["legacy_resid"]
        else:
            for rec in records:
                rec["site"] = f"{rec['resseq_int']}{rec['icode']}"
        for chain, chain_part in part.groupby("chain", sort=False):
            key = (str(pdb), str(chain))
            if key not in cache:
                site = binding_atoms(records, str(chain), args.cutoff)
                partner = [rec for rec in site if rec["chain"] != chain]
                mine = [rec for rec in site if rec["chain"] == chain]
                cache[key] = (nine(mine, partner, args.quantize == "javaplex"), site)
            (death_b, per_b), neighborhood = cache[key]
            for rec in chain_part.itertuples(index=False):
                if args.icode == "collapsed":
                    site_key = _legacy_key(int(rec.resid_num))
                else:
                    site_key = str(rec.resid)
                mut = [atom for atom in neighborhood if atom["chain"] == chain and atom["site"] == site_key]
                rest = [atom for atom in neighborhood if not (atom["chain"] == chain and atom["site"] == site_key)]
                death_m, per_m = nine(mut, rest, args.quantize == "javaplex")
                built = {
                    "binding_death": death_b,
                    "binding_persistence": per_b,
                    "mutation_death": death_m,
                    "mutation_persistence": per_m,
                }
                row = int(rec.row_idx)
                for name, (start, stop) in WT_SLICES.items():
                    got = np.rint(built[name]).astype(np.int64)
                    ref = np.rint(x[row, :, start:stop]).astype(np.int64)
                    rates[name].append(bool(np.array_equal(got, ref)))
                    cells[name].append(float(np.mean(got == ref)))
        print(f"finished {pdb} rows {len(part)}", flush=True)
    print("icode setting", args.icode, "quantize", args.quantize, "altloc", args.altloc, "cutoff", args.cutoff)
    any_pass = False
    for name, flags in rates.items():
        rate = float(np.mean(flags)) if flags else float("nan")
        cell = cells[name]
        best = float(np.max(cell)) if cell else float("nan")
        print(
            f"{name}: exact rows {int(np.sum(flags))}/{len(flags)} rate {rate:.4f} "
            f"mean cell agreement {float(np.mean(cell)) if cell else float('nan'):.4f} best row {best:.4f}"
        )
        if flags and rate >= 0.95:
            any_pass = True
    if not any_pass:
        print("PILOT FAIL: no WT channel block matched at least 95% of these rows. Stop T4.4-T4.7.")
        print("Downstream features would be H1/H2-style, not original-equivalent.")
        return 2
    print("PILOT PASS on at least one WT channel block.")
    return 0


def _legacy_key(resseq: int) -> str:
    """PDB columns 24-26: the last three digits, without the thousands digit."""
    return f"{resseq:4d}"[1:].strip()


if __name__ == "__main__":
    raise SystemExit(main())
