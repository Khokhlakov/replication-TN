"""H1/H2 statistics.

``gen_feature_12.py`` stores, for each element pair and for wild and mutant,
seven numbers: sum / max / mean of bar lengths, min / max birth, min / max death.
The third slot along that axis is mutant minus wild. Dimension is
3 x 3 x 7 x 3 x 4 = 756 (H1 and H2, binding and mutation sites).

The task text also mentions min and std of lengths, births and deaths. That is
not what the uploaded script computes. ``legacy_bar_stats`` follows the script.
``spec_bar_stats`` follows the task text and is not wired into the 756-vector.
"""

from __future__ import annotations

import numpy as np

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
    simplex.compute_persistence()
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
    """Each argument is a list of 9 stat vectors (one per element pair), or an array (9, 7)."""
    def block(wild, mutant):
        wild = np.asarray(wild, dtype=np.float64).reshape(3, 3, 7)
        mutant = np.asarray(mutant, dtype=np.float64).reshape(3, 3, 7)
        diff = mutant - wild
        stacked = np.stack([wild, mutant, diff], axis=-1)
        return stacked.reshape(-1)

    return np.concatenate([block(wild_h1, mut_h1), block(wild_h2, mut_h2)])
