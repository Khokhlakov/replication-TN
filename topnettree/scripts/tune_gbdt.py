"""Inner-fold GBDT search. The outer test rows are never passed in.

``search_gbdt`` is the original small grid. ``random_search_outer`` is the
T2 random search: about 12 trials, 3 inner folds, ``n_estimators`` chosen with
``staged_predict``. Option A retrains the CNN inside each inner fold. Option B
tunes on features from a CNN fit on the whole outer-training split and is
labeled mildly optimistic.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from topnet.config import load_config, repo_root
from topnet.cv import FitLog, kfold_splits
from topnet.gbdt import make_gbdt

RANDOM_SPACE = {
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.4, 0.5, 0.7],
    "learning_rate": [0.01, 0.03],
    "max_features": ["sqrt", 0.3],
    "min_samples_leaf": [1, 3, 5],
    "n_keep": [100, 300],
}


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


def _space(cfg: dict) -> dict:
    raw = (cfg.get("tune") or {}).get("random_space")
    return raw or RANDOM_SPACE


def _draw_trials(space: dict, n_trials: int, seed: int):
    keys = list(space.keys())
    combos = list(itertools.product(*[space[key] for key in keys]))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(combos))
    picked = [combos[int(i)] for i in order[: min(n_trials, len(combos))]]
    return keys, picked


def _simplicity(row: dict):
    features = 0 if row["max_features"] == "sqrt" else 1
    return (
        int(row["max_depth"]),
        -int(row["min_samples_leaf"]),
        int(row["n_estimators"]),
        int(row["n_keep"]),
        float(row["learning_rate"]),
        features,
        -float(row["subsample"]),
    )


def select_within_one_se(trials: list[dict]) -> dict:
    """Lowest inner RMSE, then the simpler trial within one standard error."""
    best_rmse = min(float(row["rmse"]) for row in trials)
    best = min(trials, key=lambda row: float(row["rmse"]))
    se = float(best["rmse_se"])
    pool = [row for row in trials if float(row["rmse"]) <= best_rmse + se]
    return min(pool, key=_simplicity)


def _staged_curve(model, x_test, y_test) -> np.ndarray:
    curve = []
    for pred in model.staged_predict(x_test):
        curve.append(float(np.sqrt(mean_squared_error(y_test, pred))))
    return np.asarray(curve, dtype=np.float64)


def _gbdt_for_trial(row: dict, n_estimators: int, seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(row["learning_rate"]),
        max_depth=int(row["max_depth"]),
        min_samples_leaf=int(row["min_samples_leaf"]),
        subsample=float(row["subsample"]),
        max_features=row["max_features"],
        loss="squared_error",
        random_state=int(seed),
    )


def _matrices_for_inner(x_cnn, y, x_aux, x_h12, inner_train, inner_test, cfg, seed, n_keep, feature_set, fit_log):
    """Option A. The CNN, the ranking and the matrix use inner-training rows only."""
    from topnet.cnn import extract_features, train_cnn
    from topnet.gbdt import assemble_features
    from topnet.normalize import Normalizer
    from topnet.select import rank_cnn_features, take_top

    if fit_log is not None:
        fit_log.record("tune_cnn", inner_train, inner_test, seed, 0)
    normalizer = Normalizer().fit(x_cnn[inner_train])
    model = train_cnn(
        normalizer.transform(x_cnn[inner_train]),
        y[inner_train],
        cfg,
        seed=seed,
        device="cpu",
    )
    train_feat = extract_features(model, normalizer.transform(x_cnn[inner_train]), device="cpu")
    test_feat = extract_features(model, normalizer.transform(x_cnn[inner_test]), device="cpu")
    if fit_log is not None:
        fit_log.record("tune_rank", inner_train, inner_test, seed, 0)
    order = rank_cnn_features(train_feat, y[inner_train], cfg, seed=seed + 1)
    keep = min(int(n_keep), int(order.shape[0]))
    chosen = take_top(order, keep)
    train_matrix = assemble_features(
        train_feat[:, chosen], None if x_aux is None else x_aux[inner_train], None if x_h12 is None else x_h12[inner_train], feature_set
    )
    test_matrix = assemble_features(
        test_feat[:, chosen], None if x_aux is None else x_aux[inner_test], None if x_h12 is None else x_h12[inner_test], feature_set
    )
    return train_matrix, test_matrix


def random_search_outer(
    x_cnn,
    y,
    x_aux,
    outer_train,
    outer_test,
    cfg,
    seed: int,
    fit_log: FitLog | None = None,
    x_h12=None,
    feature_set: str = "cnn_aux",
    precomputed=None,
):
    """Random search on one outer-training split. ``outer_test`` is never indexed."""
    outer_train = np.asarray(outer_train, dtype=np.int64)
    outer_test = np.asarray(outer_test, dtype=np.int64)
    if np.intersect1d(outer_train, outer_test).size:
        raise RuntimeError("outer train and test overlap")
    tune = cfg["tune"]
    n_trials = int(tune.get("n_trials", 12))
    max_estimators = int(tune.get("max_estimators", 3000))
    inner = KFold(n_splits=int(tune["inner_splits"]), shuffle=True, random_state=int(seed))
    keys, picked = _draw_trials(_space(cfg), n_trials, seed)
    trials = []
    option = "B" if precomputed is not None else str(tune.get("cnn_option", "A"))
    for trial_id, values in enumerate(picked):
        row = {key: value for key, value in zip(keys, values)}
        curves = []
        for fold, (tr, te) in enumerate(inner.split(np.arange(len(outer_train)))):
            inner_train = outer_train[tr]
            inner_test = outer_train[te]
            if np.intersect1d(inner_train, outer_test).size or np.intersect1d(inner_test, outer_test).size:
                raise RuntimeError("inner fold touched the outer test rows")
            if precomputed is None:
                x_tr, x_te = _matrices_for_inner(
                    x_cnn, y, x_aux, x_h12, inner_train, inner_test, cfg, seed + fold, row["n_keep"], feature_set, fit_log
                )
                y_tr, y_te = y[inner_train], y[inner_test]
            else:
                from topnet.gbdt import assemble_features
                from topnet.select import rank_cnn_features, take_top

                if fit_log is not None:
                    fit_log.record("tune_rank", inner_train, inner_test, seed, fold)
                order = rank_cnn_features(precomputed[inner_train], y[inner_train], cfg, seed=seed + fold)
                keep = min(int(row["n_keep"]), int(order.shape[0]))
                chosen = take_top(order, keep)
                x_tr = assemble_features(
                    precomputed[inner_train][:, chosen],
                    None if x_aux is None else x_aux[inner_train],
                    None if x_h12 is None else x_h12[inner_train],
                    feature_set,
                )
                x_te = assemble_features(
                    precomputed[inner_test][:, chosen],
                    None if x_aux is None else x_aux[inner_test],
                    None if x_h12 is None else x_h12[inner_test],
                    feature_set,
                )
                y_tr, y_te = y[inner_train], y[inner_test]
            model = _gbdt_for_trial(row, max_estimators, seed + fold)
            model.fit(x_tr, y_tr)
            curves.append(_staged_curve(model, x_te, y_te))
        length = min(len(curve) for curve in curves)
        stacked = np.vstack([curve[:length] for curve in curves])
        mean_curve = stacked.mean(axis=0)
        best_k = int(np.argmin(mean_curve)) + 1
        per_fold = stacked[:, best_k - 1]
        record = dict(row)
        record.update(
            {
                "trial": trial_id,
                "n_estimators": best_k,
                "rmse": float(mean_curve[best_k - 1]),
                "rmse_se": float(per_fold.std(ddof=1) / np.sqrt(len(per_fold))) if len(per_fold) > 1 else 0.0,
                "cnn_option": option,
                "max_estimators_fit": max_estimators,
            }
        )
        trials.append(record)
        print(record)
    winner = select_within_one_se(trials)
    winner = dict(winner)
    winner["selection"] = "min inner RMSE, simpler model within 1 SE"
    if option == "B":
        winner["optimism"] = "mildly optimistic: CNN saw inner-validation rows"
    else:
        winner["optimism"] = "strict: CNN retrained inside each inner fold"
    return winner, trials


def _plain_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def write_tuned_config(per_fold: list[dict], path: Path) -> None:
    """Summary config. Per-fold parameters stay in the run directory and in gbdt_by_fold."""
    path = Path(path)
    per_fold = [_plain_row(row) for row in per_fold]
    gbdt = {
        "n_estimators": int(per_fold[0]["n_estimators"]),
        "learning_rate": float(per_fold[0]["learning_rate"]),
        "max_depth": int(per_fold[0]["max_depth"]),
        "min_samples_leaf": int(per_fold[0]["min_samples_leaf"]),
        "min_samples_split": 2,
        "subsample": float(per_fold[0]["subsample"]),
        "max_features": per_fold[0]["max_features"],
        "loss": "squared_error",
        "backend": "sklearn",
    }
    same = all(
        row["max_depth"] == per_fold[0]["max_depth"] and row["n_estimators"] == per_fold[0]["n_estimators"]
        for row in per_fold
    )
    document = {
        "extends": "fast_nested.yaml",
        "tune_status": "complete" if same else "per_fold",
        "select": {"n_keep": int(per_fold[0]["n_keep"]), "n_estimators": None},
        "gbdt": gbdt,
        "gbdt_by_fold": per_fold,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def tune_all_outer_folds(cfg: dict, out_dir: Path, feature_set: str = "cnn_aux"):
    """Option A by default. This is the long T2 search. Do not start it from a smoke test."""
    from topnet.data import assert_label_tables, load_arrays

    y, meta, _sup = assert_label_tables(cfg)
    x_cnn, x_aux, _idx = load_arrays(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    option = str(cfg["tune"].get("cnn_option", "A"))
    chosen_rows = []
    all_trials = []
    for seed in [int(item) for item in cfg["cv"]["seeds"]]:
        splits = kfold_splits(len(y), int(cfg["cv"]["n_splits"]), seed)
        for fold, (train, test) in enumerate(splits):
            precomputed = None
            if option == "B":
                from topnet.cnn import extract_features, train_cnn
                from topnet.normalize import Normalizer

                normalizer = Normalizer().fit(x_cnn[train])
                model = train_cnn(
                    normalizer.transform(x_cnn[train]), y[train], cfg, seed=seed + fold, device="cpu"
                )
                precomputed = np.zeros((len(y), 48 * 32), dtype=np.float32)
                precomputed[train] = extract_features(model, normalizer.transform(x_cnn[train]), device="cpu")
            winner, trials = random_search_outer(
                x_cnn,
                y,
                x_aux,
                train,
                test,
                cfg,
                seed=seed + 1000 * fold,
                feature_set=feature_set,
                precomputed=precomputed,
            )
            winner["seed"] = seed
            winner["fold"] = fold
            chosen_rows.append(winner)
            for trial in trials:
                trial["seed"] = seed
                trial["fold"] = fold
            all_trials.extend(trials)
            print("chosen", winner)
    pd.DataFrame(all_trials).to_csv(out_dir / "trials.csv", index=False)
    pd.DataFrame(chosen_rows).to_csv(out_dir / "chosen_by_fold.csv", index=False)
    write_tuned_config(chosen_rows, repo_root() / "configs" / "tuned.yaml")
    return chosen_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--protocol", choices=["grid", "random_inner"], default="grid")
    parser.add_argument("--features", default=None, help="Training-row feature matrix .npy (grid protocol)")
    parser.add_argument("--labels", default=None, help="Training-row targets .npy (grid protocol)")
    parser.add_argument("--out", default="tune_trials.csv")
    parser.add_argument("--feature-set", default="cnn_aux")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.protocol == "random_inner":
        tune_all_outer_folds(cfg, Path(args.out), feature_set=args.feature_set)
        print("wrote", repo_root() / "configs" / "tuned.yaml")
        return 0
    if not args.features or not args.labels:
        raise SystemExit("grid protocol requires --features and --labels")
    features = np.load(args.features)
    y = np.load(args.labels)
    best, trials = search_gbdt(features, y, np.arange(len(y)), cfg, int(cfg["seed"]))
    pd.DataFrame(trials).to_csv(args.out, index=False)
    print("best", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
