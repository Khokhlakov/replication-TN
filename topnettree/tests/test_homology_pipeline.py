"""Synthetic checks for the homology blocks and the batch driver. No real PDB needed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topnet.features import homology as hom  # noqa: E402
from topnet.features.h0 import bipartite_deaths  # noqa: E402

gudhi = pytest.importorskip("gudhi")

BACKBONE = ("N", "CA", "C", "O", "CB")


def _line(serial, name, resname, chain, resseq, icode, xyz, element):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  {name:<3s} {resname:>3s} {chain}{resseq:4d}{icode or ' '}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
    )


def make_structure(path, chains, mutate=None, seed=0, hydrogens=False):
    """chains: {chain_id: (center, n_residues)}. mutate: (chain, resseq, icode, new_resname).

    Coordinates are drawn per residue from their own generator, so a mutant differs
    from the wild type only at the mutated residue.
    """
    lines, serial = [], 1
    for chain, (center, n_res) in chains.items():
        for r in range(1, n_res + 1):
            rng = np.random.default_rng([seed, ord(chain), r])
            base = np.asarray(center) + rng.normal(scale=6.0, size=3)
            offsets = rng.normal(scale=1.3, size=(7, 3))
            h_offsets = rng.normal(scale=1.6, size=(2, 3))  # always drawn, so heavy atoms never move
            resname = "ALA"
            icode = ""
            if mutate and (chain, r) == (mutate[0], mutate[1]):
                resname = mutate[3]
                icode = mutate[2]
            names = BACKBONE if resname == "ALA" else BACKBONE + ("CG", "OD1")
            for k, name in enumerate(names):
                lines.append(_line(serial, name, resname, chain, r, icode, base + offsets[k], name[0]))
                serial += 1
            if hydrogens:
                for k, name in enumerate(("HA", "HB1")):
                    lines.append(_line(serial, name, resname, chain, r, icode, base + h_offsets[k], "H"))
                    serial += 1
    path.write_text("".join(lines) + "END\n")
    return path


COMPLEX = {"E": ((0, 0, 0), 40), "H": ((12, 0, 0), 40), "L": ((12, 12, 0), 25)}
WITH_EXTRA = dict(COMPLEX, Y=((6, 6, 6), 30))  # a near chain that is not part of the complex


def _pair(tmp_path, chains, name="a"):
    wild = make_structure(tmp_path / f"{name}_wild.pdb", chains, seed=3)
    mutant = make_structure(tmp_path / f"{name}_mut.pdb", chains, mutate=("E", 7, "", "ASP"), seed=3)
    return wild, mutant


def test_keep_chains_equals_file_without_the_extra_chain(tmp_path):
    w_full, m_full = _pair(tmp_path, WITH_EXTRA, "full")
    w_clean, m_clean = _pair(tmp_path, COMPLEX, "clean")
    kept = hom.homology_blocks(w_full, m_full, "E", 7, keep_chains="E+H+L")
    clean = hom.homology_blocks(w_clean, m_clean, "E", 7)
    leaky = hom.homology_blocks(w_full, m_full, "E", 7)
    assert np.array_equal(kept.h0, clean.h0)
    assert np.allclose(kept.h12, clean.h12)
    assert not np.array_equal(leaky.h0, clean.h0), "the extra chain should change the binding block"
    assert kept.keep_chains == ("E", "H", "L")


def test_keep_chains_errors(tmp_path):
    w, m = _pair(tmp_path, COMPLEX)
    with pytest.raises(ValueError, match="not in keep_chains"):
        hom.homology_blocks(w, m, "E", 7, keep_chains="H+L", compute_h12=False)
    with pytest.raises(ValueError, match="not in file"):
        hom.homology_blocks(w, m, "E", 7, keep_chains="E+H+Q", compute_h12=False)
    with pytest.raises(ValueError, match="one character"):
        hom.parse_keep_chains("E+HH")


def test_records_and_paths_agree(tmp_path):
    from topnet.features.pdbio import read_pdb_records

    w, m = _pair(tmp_path, COMPLEX)
    a = hom.homology_blocks(w, m, "E", 7, compute_h12=False)
    b = hom.homology_blocks(read_pdb_records(w), read_pdb_records(m), "E", 7, compute_h12=False)
    assert np.array_equal(a.h0, b.h0)


def _kruskal_cross(a, b):
    """Independent MST of the complete bipartite graph."""
    edges = sorted(
        (float(np.linalg.norm(p - q)), i, len(a) + j) for i, p in enumerate(a) for j, q in enumerate(b)
    )
    parent = list(range(len(a) + len(b)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            out.append(d)
    return np.sort(np.asarray(out))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bipartite_deaths_match_kruskal(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(17, 3)) * 4
    b = rng.normal(size=(23, 3)) * 4 + 3
    assert np.allclose(np.sort(bipartite_deaths(a, b)), _kruskal_cross(a, b))


def _driver():
    spec = importlib.util.spec_from_file_location("bhf", ROOT / "scripts" / "build_homology_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bhf"] = mod  # multiprocessing must be able to import it by name
    spec.loader.exec_module(mod)
    return mod


def test_driver_check_build_validate(tmp_path, capsys):
    drv = _driver()
    w, m = _pair(tmp_path, WITH_EXTRA)
    bad_m = make_structure(tmp_path / "wrong.pdb", WITH_EXTRA, mutate=("E", 7, "", "GLY"), seed=3)
    table = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "wild_pdb": [str(w)] * 4,
            "mutant_pdb": [str(m), str(m), str(bad_m), str(tmp_path / "nope.pdb")],
            "chain": ["E"] * 4,
            "resid": ["7", "7", "7", "7"],
            "wt": ["A", "A", "A", "A"],
            "mut": ["D", "D", "D", "D"],
            "keep_chains": ["E+H+L", "", "E+H+L", "E+H+L"],
        }
    )
    csv = tmp_path / "m.csv"
    table.to_csv(csv, index=False)
    out = tmp_path / "out"

    rc = drv.main(["check", "--manifest", str(csv), "--out-dir", str(out)])
    report = pd.read_csv(out / "check_report.csv").set_index("row_id")
    assert rc == 1
    assert report.loc["r1", "status"] == "ok"
    assert "Y" in report.loc["r2", "partner_chains"], "unfiltered partner chains should be visible"
    assert report.loc["r3", "status"] == "fail" and "table says D" in report.loc["r3", "messages"]
    assert report.loc["r4", "status"] == "fail" and "not found" in report.loc["r4", "messages"]

    rc = drv.main(["build", "--manifest", str(csv), "--out-dir", str(out), "--jobs", "2"])
    assert rc == 1  # r4 has no mutant file
    ok = np.load(out / "ok_mask.npy")
    h0 = np.load(out / "h0.npy")
    h12 = np.load(out / "h12.npy")
    assert ok.tolist() == [True, True, True, False]
    assert h0.shape == (4, 48, 108) and h12.shape == (4, 756)
    assert np.isnan(h0[3]).all() and np.isnan(h12[3]).all(), "a failed row must be NaN, never zeros"
    assert not np.array_equal(h0[0], h0[1]), "row 2 keeps the extra chain and must differ"

    rc = drv.main(["validate", "--out-dir", str(out), "--reference", str(out / "h0.npy")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "OK" in text


def test_invariants_catch_a_corrupted_tensor(tmp_path):
    drv = _driver()
    w, m = _pair(tmp_path, COMPLEX)
    blocks = hom.homology_blocks(w, m, "E", 7, compute_h12=True)
    h0 = blocks.h0[None].copy()
    assert sum(drv.h0_invariants(h0).values()) == 0
    assert sum(drv.h12_invariants(blocks.h12[None]).values()) == 0
    h0[0, 5, 3] += 1.0  # break a persistence column
    assert drv.h0_invariants(h0)["persistence_not_tail_sum_of_deaths"] == 1


def test_hydrogen_option(tmp_path, capsys):
    from topnet.features.pdbio import is_hydrogen, read_pdb_records

    drv = _driver()
    wild = make_structure(tmp_path / "w.pdb", COMPLEX, seed=3)
    clean_mut = make_structure(tmp_path / "m_clean.pdb", COMPLEX, mutate=("E", 7, "", "ASP"), seed=3)
    h_mut = make_structure(tmp_path / "m_h.pdb", COMPLEX, mutate=("E", 7, "", "ASP"), seed=3, hydrogens=True)
    recs = read_pdb_records(h_mut)
    assert sum(is_hydrogen(r) for r in recs) == 2 * 105
    assert not any(is_hydrogen(r) for r in read_pdb_records(wild))

    table = pd.DataFrame(
        {"row_id": ["a"], "wild_pdb": [str(wild)], "mutant_pdb": [str(h_mut)], "chain": ["E"], "resid": ["7"],
         "wt": ["A"], "mut": ["D"], "keep_chains": ["E+H+L"]}
    )
    csv = tmp_path / "m.csv"
    table.to_csv(csv, index=False)
    drv.main(["check", "--manifest", str(csv), "--out-dir", str(tmp_path / "keep")])
    rep = pd.read_csv(tmp_path / "keep" / "check_report.csv").iloc[0]
    assert rep["status"] == "warn" and "hydrogens only in one file" in rep["messages"]
    assert rep["h_wild"] == 0 and rep["h_mutant"] == 2 * 105

    drv.main(["check", "--manifest", str(csv), "--out-dir", str(tmp_path / "drop"), "--hydrogens", "drop"])
    rep = pd.read_csv(tmp_path / "drop" / "check_report.csv").iloc[0]
    assert rep["status"] == "ok"

    drv.main(["build", "--manifest", str(csv), "--out-dir", str(tmp_path / "drop"), "--hydrogens", "drop", "--h0-only"])
    dropped = np.load(tmp_path / "drop" / "h0.npy")[0]
    clean = hom.homology_blocks(wild, clean_mut, "E", 7, compute_h12=False, keep_chains="E+H+L").h0
    assert np.array_equal(dropped, clean)
