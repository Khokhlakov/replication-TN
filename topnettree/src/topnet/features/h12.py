"""H1 and H2 statistics from an alpha complex.

``gen_feature_12.py`` stores, for each element pair and for wild and mutant,
seven numbers: sum, max and mean of bar length, and the minimum and maximum
of birth and of death. The third slot along that axis is mutant minus wild.
One site (binding, or mutation) is 3 x 3 x 7 x 3 x 2 = 378 numbers. Binding
then mutation is 756.

The point cloud for one element pair is the C/N/O atoms of that pair from
both sides of the neighborhood, concatenated. It is not the bipartite complex
used for H0. ``PH_Alpha.R`` computes the diagram with ``TDA::alphaShapeDiag``
and does not take a square root. Gudhi's filtration value is the squared
radius, so ``alpha_persistence`` reports ``sqrt(filtration)`` and does not
multiply by two. There is no shipped ``X_h12.npy`` in the capsule to check
that scale against.

The task text also mentions min and std of lengths, births and deaths. That
is not what the uploaded script computes. ``legacy_bar_stats`` follows the
script. ``spec_bar_stats`` follows the task text and is not part of the
756-vector.

Call ``topnet.features.homology.homology_blocks`` on a wild PDB and a mutant
PDB. ``h12_from_neighborhoods`` is the same 756-vector once the two
neighborhoods have already been cut.
"""

from __future__ import annotations

import numpy as np

from topnet.features.h0 import PAIRS
from topnet.features.pdbio import coords_of

H12_DIM = 756
SMALL = 0.01


def filter_bars(births: np.ndarray, deaths: np.ndarray, small: float = SMALL):
    births = np.asarray(births, dtype=np.float64)
    deaths = np.asarray(deaths, dtype=np.float64)
    keep = (deaths - births) >= small
    return births[keep], deaths[keep]


def legacy_bar_stats(births: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    births, deaths = filter_bars(births, deaths)
    if len(births) == 0:
        return np.zeros(7, dtype=np.float64)
    length = deaths - births
    return np.array(
        [
            np.sum(length),
            np.max(length),
            np.mean(length),
            np.min(births),
            np.max(births),
            np.min(deaths),
            np.max(deaths),
        ],
        dtype=np.float64,
    )


def spec_bar_stats(births: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """sum, max, min, mean, std of lengths, births and deaths. Not the legacy vector."""
    births, deaths = filter_bars(births, deaths)
    if len(births) == 0:
        return np.zeros(15, dtype=np.float64)
    length = deaths - births
    pieces = []
    for values in (length, births, deaths):
        pieces.extend([np.sum(values), np.max(values), np.min(values), np.mean(values), np.std(values)])
    return np.asarray(pieces, dtype=np.float64)


def alpha_persistence(points: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Alpha-complex bars on the radius scale: sqrt(filtration), no extra factor of 2.

    Gudhi's alpha filtration value is the squared radius. Requires the optional
    ``gudhi`` package.
    """
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError("H1/H2 alpha features need the optional gudhi package") from exc
    pts = np.asarray(points, dtype=np.float64)
    empty = {1: (np.zeros(0), np.zeros(0)), 2: (np.zeros(0), np.zeros(0))}
    if len(pts) < 1:
        return empty
    simplex = gudhi.AlphaComplex(points=pts).create_simplex_tree()
    # Z/2 coefficients, as in the SI (Gudhi's default is Z/11).
    simplex.compute_persistence(homology_coeff_field=2)
    out = {}
    for dim in (1, 2):
        intervals = np.asarray(simplex.persistence_intervals_in_dimension(dim), dtype=np.float64)
        if intervals.size == 0:
            out[dim] = (np.zeros(0), np.zeros(0))
            continue
        finite = np.isfinite(intervals[:, 1])
        birth = np.sqrt(np.maximum(intervals[finite, 0], 0.0))
        death = np.sqrt(np.maximum(intervals[finite, 1], 0.0))
        out[dim] = filter_bars(birth, death)
    return out


def stack_legacy(wild_h1, mut_h1, wild_h2, mut_h2) -> np.ndarray:
    """One site: H1 wild/mutant/difference, then H2 the same way. Shape (378,).

    Each argument is nine stat vectors (one per element pair CC, CN, CO, NC,
    NN, NO, OC, ON, OO), shape (9, 7). The difference is mutant minus wild,
    which is the opposite sign from the H0 tensor.
    """

    def block(wild, mutant):
        wild = np.asarray(wild, dtype=np.float64).reshape(3, 3, 7)
        mutant = np.asarray(mutant, dtype=np.float64).reshape(3, 3, 7)
        diff = mutant - wild
        stacked = np.stack([wild, mutant, diff], axis=-1)
        return stacked.reshape(-1)

    return np.concatenate([block(wild_h1, mut_h1), block(wild_h2, mut_h2)])


def _pair_stats(part_a: list, part_b: list) -> tuple[np.ndarray, np.ndarray]:
    """(9, 7) H1 stats and (9, 7) H2 stats for one structure and one neighborhood."""
    h1 = np.zeros((9, 7), dtype=np.float64)
    h2 = np.zeros((9, 7), dtype=np.float64)
    for index, (e1, e2) in enumerate(PAIRS):
        left = coords_of(part_a, e1)
        right = coords_of(part_b, e2)
        pieces = [cloud for cloud in (left, right) if len(cloud)]
        if not pieces:
            continue
        points = pieces[0] if len(pieces) == 1 else np.vstack(pieces)
        diagram = alpha_persistence(points)
        h1[index] = legacy_bar_stats(*diagram[1])
        h2[index] = legacy_bar_stats(*diagram[2])
    return h1, h2


def h12_from_neighborhoods(
    wild_binding: tuple[list, list],
    mutant_binding: tuple[list, list],
    wild_mutation: tuple[list, list],
    mutant_mutation: tuple[list, list],
) -> np.ndarray:
    """756-vector. Each argument is ``(side_one, side_two)`` of one neighborhood.

    Side one is the chain (binding) or the mutated residue (mutation). Side
    two is the rest of that neighborhood. Order of the vector, matching
    ``gen_feature_12.py``: binding H1, binding H2, mutation H1, mutation H2.
    """
    wild_b = _pair_stats(*wild_binding)
    mut_b = _pair_stats(*mutant_binding)
    wild_m = _pair_stats(*wild_mutation)
    mut_m = _pair_stats(*mutant_mutation)
    binding = stack_legacy(wild_b[0], mut_b[0], wild_b[1], mut_b[1])
    mutation = stack_legacy(wild_m[0], mut_m[0], wild_m[1], mut_m[1])
    vector = np.concatenate([binding, mutation])
    if vector.shape != (H12_DIM,):
        raise RuntimeError(f"H1/H2 vector has shape {vector.shape}, expected {(H12_DIM,)}")
    return vector
