import numpy as np

from topnet.cv import homology_blind_splits, loco_splits
from topnet.data import load_labels


def test_loco_and_homology_counts():
    _y, meta = load_labels()
    loco = list(loco_splits(meta["complex_id"].to_numpy()))
    assert len(loco) == 29
    covered = []
    for train, test in loco:
        assert len(np.intersect1d(train, test)) == 0
        covered.extend(test.tolist())
    assert sorted(covered) == list(range(len(meta)))
    blind = list(homology_blind_splits(meta["is_homology"].to_numpy()))
    train, test = blind[0]
    assert len(test) == 87
    assert len(train) == 558
    assert len(np.intersect1d(train, test)) == 0
