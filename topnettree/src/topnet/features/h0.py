"""H0 Rips features on a bipartite atom set.

Within each part the distance is infinite, so the finite deaths are the edge
lengths of a minimum spanning tree of the complete bipartite graph
(``scipy.sparse.csgraph.minimum_spanning_tree``). Binning follows
``gen_cnn_feature.py``: 0.25 A bins on [0, 12), boundary deaths assigned to the
higher bin, bars shorter than 0.01 A dropped, deaths at or beyond 12 A dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ELEMENTS = ("C", "N", "O")
PAIRS = [(a, b) for a in ELEMENTS for b in ELEMENTS]


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist


def mst_lengths(dist: np.ndarray) -> np.ndarray:
    tree = minimum_spanning_tree(np.asarray(dist, dtype=np.float64))
    data = tree.tocoo().data
    return np.asarray(data, dtype=np.float64)


def euclidean_mst_lengths(points: np.ndarray) -> np.ndarray:
    return np.sort(mst_lengths(pairwise_distances(points)))


def bipartite_deaths(part_a: np.ndarray, part_b: np.ndarray) -> np.ndarray:
    part_a = np.atleast_2d(np.asarray(part_a, dtype=np.float64))
    part_b = np.atleast_2d(np.asarray(part_b, dtype=np.float64))
    if part_a.size == 0 or part_b.size == 0 or len(part_a) == 0 or len(part_b) == 0:
        return np.zeros((0,), dtype=np.float64)
    points = np.vstack([part_a, part_b])
    dist = pairwise_distances(points)
    n_a = len(part_a)
    cross = dist.copy()
    cross[:n_a, :n_a] = 0.0
    cross[n_a:, n_a:] = 0.0
    finite = cross[np.isfinite(cross)]
    big = float(np.max(finite) + 10.0) if finite.size else 1.0
    dist[:n_a, :n_a] = big
    dist[n_a:, n_a:] = big
    np.fill_diagonal(dist, 0.0)
    lengths = mst_lengths(dist)
    return lengths[lengths <= (big - 5.0)]


def histogram_deaths(
    deaths: np.ndarray,
    n_bins: int = 48,
    interval: float = 0.25,
    cut: float = 12.0,
    small: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin death counts and the persistence count used by the legacy code."""
    death_count = np.zeros(n_bins, dtype=np.float64)
    persistence = np.zeros(n_bins, dtype=np.float64)
    edges = [i * interval for i in range(n_bins + 1)]
    for death in np.asarray(deaths, dtype=np.float64):
        if (death - 0.0) < small or death >= cut:
            continue
        bid = None
        for i in range(n_bins):
            if death >= edges[i] and death <= edges[i + 1]:
                bid = i
        if bid is None:
            continue
        death_count[bid] += 1.0
        persistence[:bid] += 1.0
    return death_count, persistence


def _coords(atoms: list[tuple], element: str) -> np.ndarray:
    rows = [[a[3], a[4], a[5]] for a in atoms if a[2] == element]
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _pair_block(wild: list[tuple], mutant: list[tuple], select) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bins = 48
    death_wt = np.zeros((n_bins, 9), dtype=np.float64)
    death_mut = np.zeros((n_bins, 9), dtype=np.float64)
    per_wt = np.zeros((n_bins, 9), dtype=np.float64)
    per_mut = np.zeros((n_bins, 9), dtype=np.float64)
    for index, (e1, e2) in enumerate(PAIRS):
        w1, w2 = select(wild)
        m1, m2 = select(mutant)
        wild_deaths = bipartite_deaths(_coords(w1, e1), _coords(w2, e2))
        mut_deaths = bipartite_deaths(_coords(m1, e1), _coords(m2, e2))
        death_wt[:, index], per_wt[:, index] = histogram_deaths(wild_deaths)
        death_mut[:, index], per_mut[:, index] = histogram_deaths(mut_deaths)
    return death_wt, death_mut, per_wt, per_mut


def _pack(wild: np.ndarray, mutant: np.ndarray) -> np.ndarray:
    return np.concatenate([wild, mutant, wild - mutant], axis=1)


def h0_tensor(wild: list[tuple], mutant: list[tuple], chain: str, resid: str) -> np.ndarray:
    """(48, 108) tensor: binding death, binding persistence, mutation death, mutation persistence."""
    resid = str(resid)

    def bind(atoms):
        first = [a for a in atoms if a[0] == chain]
        second = [a for a in atoms if a[0] != chain]
        return first, second

    def site(atoms):
        first = [a for a in atoms if a[0] == chain and a[1] == resid]
        second = [a for a in atoms if not (a[0] == chain and a[1] == resid)]
        return first, second

    bind_block = _pair_block(wild, mutant, bind)
    site_block = _pair_block(wild, mutant, site)
    parts = [
        _pack(bind_block[0], bind_block[1]),
        _pack(bind_block[2], bind_block[3]),
        _pack(site_block[0], site_block[1]),
        _pack(site_block[2], site_block[3]),
    ]
    return np.concatenate(parts, axis=1)


def gudhi_rips_h0_deaths(points: np.ndarray, max_edge_length: float | None = None) -> np.ndarray:
    """Finite H0 deaths of a Euclidean Rips complex. Requires gudhi."""
    import gudhi

    pts = np.asarray(points, dtype=np.float64)
    if max_edge_length is None:
        max_edge_length = float(np.max(pairwise_distances(pts)) + 1.0)
    complex_ = gudhi.RipsComplex(points=pts, max_edge_length=max_edge_length)
    tree = complex_.create_simplex_tree(max_dimension=1)
    tree.compute_persistence()
    intervals = tree.persistence_intervals_in_dimension(0)
    finite = intervals[np.isfinite(intervals[:, 1])]
    return np.sort(finite[:, 1].astype(np.float64))
