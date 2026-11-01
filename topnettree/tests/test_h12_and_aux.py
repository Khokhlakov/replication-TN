import numpy as np
import pytest

from topnet.features.aux import mutation_aux
from topnet.features.h12 import H12_DIM, legacy_bar_stats, spec_bar_stats, stack_legacy
from topnet.gbdt import assemble_features


def test_legacy_stats_drop_short_bars_and_diff_sign():
    assert np.all(legacy_bar_stats(np.array([0.0]), np.array([0.001])) == 0)
    stats = legacy_bar_stats(np.array([0.0, 1.0]), np.array([2.0, 1.5]))
    assert stats.shape == (7,)
    assert np.isclose(stats[0], 2.5)
    wild = np.arange(63, dtype=np.float64).reshape(9, 7)
    mutant = wild + 1
    vector = stack_legacy(wild, mutant, wild, mutant)
    assert vector.shape == (378,)
    assert np.allclose(vector[:189][2::3], 1.0)
    full = np.concatenate([vector, vector])
    assert full.shape == (H12_DIM,)
    assert spec_bar_stats(np.array([0.0]), np.array([1.0])).shape == (15,)


def test_feature_set_names_and_missing_h12():
    cnn = np.zeros((2, 5))
    aux = np.zeros((2, 4))
    h12 = np.zeros((2, 3))
    assert assemble_features(cnn, aux, None, "cnn_aux").shape == (2, 9)
    assert assemble_features(cnn, aux, None, "cnn").shape == (2, 5)
    assert assemble_features(cnn, aux, None, "aux").shape == (2, 4)
    assert assemble_features(cnn, aux, h12, "topo_aux").shape == (2, 12)
    assert assemble_features(cnn, None, h12, "topo").shape == (2, 8)
    with pytest.raises(FileNotFoundError):
        assemble_features(cnn, aux, None, "topo_aux")


def test_aux_generator_refuses():
    with pytest.raises(NotImplementedError):
        mutation_aux()
