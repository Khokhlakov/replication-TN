"""Inner-fold GBDT search. The outer test rows are never passed in."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from topnet.config import load_config
from topnet.cv import FitLog
from topnet.gbdt import make_gbdt


def search_gbdt(features: np.ndarray, y: np.ndarray, train_idx: np.ndarray, cfg: dict, seed: int, fit_log: FitLog | None = None):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    tune = cfg["tune"]
    inner = KFold(n_splits=int(tune["inner_splits"]), shuffle=True, random_state=int(seed))
    grid = itertools.product(tune["max_depth"], tune["subsample"], tune["n_estimators"])
    trials = []
    best = None
    for depth, subsample, n_estimators in grid:
        local = {**cfg, "gbdt": {**cfg["gbdt"]}}
        local["gbdt"]["max_depth"] = int(depth)
        local["gbdt"]["subsample"] = float(subsample)
        local["gbdt"]["n_estimators"] = int(n_estimators)
        scores = []
        for fold, (tr, te) in enumerate(inner.split(np.arange(len(train_idx)))):
            tr_idx = train_idx[tr]
            te_idx = train_idx[te]
            if fit_log is not None:
                fit_log.record("tune", tr_idx, te_idx, seed, fold)
            model = make_gbdt(local, seed + fold)
            model.fit(features[tr_idx], y[tr_idx])
            pred = model.predict(features[te_idx])
            scores.append(float(np.sqrt(mean_squared_error(y[te_idx], pred))))
        row = {
            "max_depth": int(depth),
            "subsample": float(subsample),
            "n_estimators": int(n_estimators),
            "rmse": float(np.mean(scores)),
        }
        trials.append(row)
        print(row)
        if best is None or row["rmse"] < best["rmse"]:
            best = row
    return best, trials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--features", required=True, help="Training-row feature matrix .npy")
    parser.add_argument("--labels", required=True, help="Training-row targets .npy")
    parser.add_argument("--out", default="tune_trials.csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    features = np.load(args.features)
    y = np.load(args.labels)
    best, trials = search_gbdt(features, y, np.arange(len(y)), cfg, int(cfg["seed"]))
    pd.DataFrame(trials).to_csv(args.out, index=False)
    print("best", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
