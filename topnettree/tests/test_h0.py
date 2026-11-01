import numpy as np
import pytest

from topnet.features.h0 import bipartite_deaths, euclidean_mst_lengths, gudhi_rips_h0_deaths, h0_tensor, histogram_deaths
from topnet.features.pdbio import read_pdb
from topnet.config import repo_root


def _atoms(seed=1):
    rng = np.random.default_rng(seed)
    atoms = []
    for _ in range(4):
        atoms.append(("A", "10", "C", *rng.normal(size=3)))
    for _ in range(5):
        atoms.append(("B", "3", "N", *(rng.normal(size=3) + np.array([4.0, 0.0, 0.0]))))
    return atoms


def test_bin_edges_follow_the_legacy_loop():
    death, per = histogram_deaths([0.02, 0.25, 11.75, 12.0, 0.001])
    assert death[0] == 1
    assert death[1] == 1
    assert death[47] == 1
    assert death.sum() == 3
    assert per[0] == 2
    assert per[1] == 1


def test_bipartite_mst_count_and_invariance():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(4, 3)) + 2
    deaths = bipartite_deaths(a, b)
    assert len(deaths) == len(a) + len(b) - 1
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = np.sort(bipartite_deaths(a @ rotation, b @ rotation))
    assert np.allclose(np.sort(deaths), rotated, atol=1e-8)
    order = rng.permutation(len(a))
    assert np.allclose(np.sort(deaths), np.sort(bipartite_deaths(a[order], b)), atol=1e-8)
    cloud = rng.normal(size=(6, 3))
    assert len(euclidean_mst_lengths(cloud)) == 5


def test_noop_mutation_diff_is_zero_and_repeatable():
    atoms = _atoms()
    first = h0_tensor(atoms, atoms, "A", "10")
    second = h0_tensor(atoms, atoms, "A", "10")
    assert first.shape == (48, 108)
    assert np.array_equal(first, second)
    for sl in (slice(18, 27), slice(45, 54), slice(72, 81), slice(99, 108)):
        assert np.all(first[:, sl] == 0)
    mutant = list(atoms)
    moved = list(mutant[0])
    moved[3] = moved[3] + 3.0
    mutant[0] = tuple(moved)
    changed = h0_tensor(atoms, mutant, "A", "10")
    assert np.any(changed[:, 18:27] != 0)
    rng = np.random.default_rng(3)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    def spin(rows):
        spun = []
        for atom in rows:
            xyz = np.asarray(atom[3:], dtype=np.float64) @ rotation
            spun.append((atom[0], atom[1], atom[2], float(xyz[0]), float(xyz[1]), float(xyz[2])))
        return spun

    assert np.allclose(changed, h0_tensor(spin(atoms), spin(mutant), "A", "10"), atol=1e-6)
    shuffled = atoms.copy()
    rng.shuffle(shuffled)
    assert np.allclose(h0_tensor(atoms, mutant, "A", "10"), h0_tensor(shuffled, mutant, "A", "10"))


def test_pdb_reader_matches_1ak4_columns():
    path = repo_root().parent / "capsule-2202829" / "data" / "AB-Bind-Database" / "1AK4.pdb"
    if not path.is_file():
        pytest.skip("1AK4.pdb is not next to the package")
    atoms = read_pdb(path)
    assert atoms[0][0] == "A"
    assert atoms[0][1] == "1"
    assert atoms[0][2] == "N"
    legacy = read_pdb(path, legacy_resid=True)
    assert legacy[0][1] == "1"


def test_gudhi_rips_matches_mst():
    pytest.importorskip("gudhi")
    cloud = np.random.default_rng(4).normal(size=(8, 3))
    left = euclidean_mst_lengths(cloud)
    right = gudhi_rips_h0_deaths(cloud)
    assert np.allclose(left, right, atol=1e-9)
