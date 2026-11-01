import numpy as np

from topnet.analysis.layer_a import (
    BA1_MUTATIONS,
    THRESH,
    disrupted_counts,
    fold_reduction,
    infectivity_ratio,
    jaccard_disrupted,
    load_published_panel,
    panel_comparison,
    sign_agreement,
)
from topnet.config import resolve_path, load_config


def test_threshold_list_is_exact():
    assert THRESH == [0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.1, -2.4, -2.7, -3.0]
    assert len(BA1_MUTATIONS) == 15
    assert BA1_MUTATIONS[4] == "K417N"
    assert BA1_MUTATIONS[13] == "N501Y"


def test_published_panel_matches_the_paper_counts():
    cfg = load_config("configs/default.yaml")
    frame, antibodies = load_published_panel(resolve_path(cfg, "antibodies_bfes"))
    assert frame.shape == (15, 186)
    assert antibodies.shape == (15, 185)
    assert not frame.isna().any().any()
    assert disrupted_counts(antibodies) == [185, 163, 143, 124, 106, 78, 60, 33, 15, 5, 2]
    assert abs(fold_reduction(antibodies) - 20.94) < 0.05
    assert abs(float(frame["ACE2"].sum()) - 2.602) < 1e-3
    assert float(frame.loc[4, "LY-CoV016"]) < -1.5
    assert float(frame.loc[13, "ACE2"]) > 0
    names = (cfg and open(resolve_path(cfg, "antibodies"), encoding="utf-8").read().splitlines())
    assert len(names) == 186
    assert names[0] == "6M0J_ACE2"


def test_identity_comparison_and_strict_count():
    matrix = np.array([[0.0, -0.3, -0.31], [-1.0, 0.2, -4.0]])
    # mins are -1, -0.3, -4. Strict <: three complexes at 0, two at -0.3.
    assert disrupted_counts(matrix, [0, -0.3]) == [3, 2]
    assert sign_agreement(matrix, matrix, 0.3) == 1.0
    assert jaccard_disrupted(matrix, matrix, -0.3) == 1.0
    report = panel_comparison(matrix, matrix)
    assert report["pearson_r"] == 1.0
    assert abs(infectivity_ratio(1.0, 0.0) - np.e) < 1e-12
