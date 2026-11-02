"""Cross-validation splitters and the nested / leaky runners.

In nested mode every fit (normalizer, CNN, importance ranking, GBDT) receives
training rows only. leaky_original trains the CNN and the ranking once on all
rows, then cross-validates the final tree. That result is optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from topnet.cnn import extract_features, resolve_device, train_cnn
from topnet.gbdt import assemble_features, make_gbdt
from topnet.metrics import metrics_with_nonbinders, per_complex_metrics, repeat_summary
from topnet.normalize import Normalizer, count_large
from topnet.select import rank_cnn_features, take_top


class FitLog:
    def __init__(self):
        self.events: list[dict] = []

    def record(self, stage: str, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, fold: int) -> None:
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        if np.intersect1d(train_idx, test_idx).size:
            raise RuntimeError(f"{stage} fit received held-out rows")
        self.events.append(
            {
                "stage": stage,
                "seed": int(seed),
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_idx": train_idx.copy(),
                "test_idx": test_idx.copy(),
            }
        )


def kfold_splits(n: int, n_splits: int, seed: int):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(splitter.split(np.arange(n)))


def loco_splits(complex_id: np.ndarray):
    complex_id = np.asarray(complex_id)
    order = list(dict.fromkeys(complex_id.tolist()))
    for cid in order:
        test = np.flatnonzero(complex_id == cid)
        train = np.flatnonzero(complex_id != cid)
        yield train, test


def homology_blind_splits(is_homology: np.ndarray):
    flag = np.asarray(is_homology).astype(bool)
    if flag.sum() == 0 or (~flag).sum() == 0:
        raise ValueError("homology_blind needs both crystal and homology-model rows")
    train = np.flatnonzero(~flag)
    test = np.flatnonzero(flag)
    yield train, test


def homology_blind_strict_splits(meta: pd.DataFrame, drop_pdb):
    """Train on crystal rows whose pdb is not a homology template. Test on HM rows.

    Dropped template crystals are in neither split. ``drop_pdb`` comes from the config.
    """
    flag = meta["is_homology"].to_numpy().astype(bool)
    pdb = meta["pdb"].astype(str).to_numpy()
    dropped = np.isin(pdb, [str(item) for item in drop_pdb]) & ~flag
    train = np.flatnonzero(~flag & ~dropped)
    test = np.flatnonzero(flag)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("homology_blind_strict produced an empty train or test split")
    yield train, test


def assert_dup_groups_intact(splits, dup_group: np.ndarray) -> None:
    """A new fold scheme must keep every dup_group entirely in one side of the split."""
    dup_group = np.asarray(dup_group)
    for train, test in splits:
        train = np.asarray(train)
        test = np.asarray(test)
        for group in set(dup_group.tolist()):
            if int(group) < 0:
                continue
            members = np.flatnonzero(dup_group == group)
            in_train = int(np.isin(members, train).sum())
            in_test = int(np.isin(members, test).sum())
            if in_train and in_test:
                raise RuntimeError(f"dup_group {group} straddles train and test")
            if in_train + in_test not in (0, len(members)):
                raise RuntimeError(f"dup_group {group} is split by a row exclusion")


def make_splits(splitter: str, n: int, meta: pd.DataFrame | None, cfg: dict, seed: int):
    if splitter == "kfold10":
        return kfold_splits(n, int(cfg["cv"]["n_splits"]), seed)
    if meta is None:
        raise ValueError(f"{splitter} needs the label metadata")
    if splitter == "loco":
        splits = list(loco_splits(meta["complex_id"].to_numpy()))
    elif splitter == "loco_parent":
        if "parent_pdb" not in meta.columns:
            raise ValueError(
                "loco_parent needs parent_pdb joined from the supplement on row_idx. Do not train on supplement ddg."
            )
        splits = list(loco_splits(meta["parent_pdb"].to_numpy()))
    elif splitter == "homology_blind":
        if "is_homology" not in meta.columns:
            raise ValueError("homology_blind needs an is_homology column in the label table")
        splits = list(homology_blind_splits(meta["is_homology"].to_numpy()))
    elif splitter == "homology_blind_strict":
        drop = cfg["cv"].get("drop_pdb")
        if not drop:
            raise ValueError("homology_blind_strict needs cv.drop_pdb in the config")
        splits = list(homology_blind_strict_splits(meta, drop))
    else:
        raise ValueError(f"Unknown splitter {splitter}")
    if splitter != "kfold10" and "dup_group" in meta.columns:
        assert_dup_groups_intact(splits, meta["dup_group"].to_numpy())
    return splits


def restrict_splits(splits, keep: np.ndarray):
    """Keep the original fold membership and drop rows outside ``keep`` from both sides."""
    keep = np.asarray(keep, dtype=bool)
    restricted = []
    for train, test in splits:
        train = np.asarray(train, dtype=np.int64)
        test = np.asarray(test, dtype=np.int64)
        train = train[keep[train]]
        test = test[keep[test]]
        if len(train) == 0 or len(test) == 0:
            continue
        restricted.append((train, test))
    return restricted


def config_for_fold(cfg: dict, seed: int, fold: int) -> dict:
    """Apply one outer-fold's tuned tree, when ``gbdt_by_fold`` is present."""
    table = cfg.get("gbdt_by_fold") or []
    for row in table:
        if int(row.get("seed", -1)) == int(seed) and int(row.get("fold", -1)) == int(fold):
            local = {**cfg, "gbdt": dict(cfg["gbdt"]), "select": dict(cfg.get("select") or {})}
            for key in ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "subsample", "max_features"):
                if key in row and row[key] is not None:
                    local["gbdt"][key] = row[key]
            if row.get("n_keep") is not None:
                local["select"]["n_keep"] = int(row["n_keep"])
            return local
    return cfg


def _slice(arr, idx):
    if arr is None:
        return None
    return np.asarray(arr)[idx]


def run_cv(
    x_cnn: np.ndarray,
    y: np.ndarray,
    x_aux: np.ndarray | None,
    cfg: dict,
    row_idx: np.ndarray | None = None,
    meta: pd.DataFrame | None = None,
    x_h12: np.ndarray | None = None,
    cv_mode: str | None = None,
    feature_set: str | None = None,
    fit_log: FitLog | None = None,
    row_keep: np.ndarray | None = None,
) -> dict:
    cv_mode = cv_mode or cfg["cv"]["mode"]
    feature_set = feature_set or cfg["cv"]["feature_set"]
    h12_sets = ("topo_aux", "topo", "h12", "cnn_h12", "cnn_aux_h12", "h12_original")
    if feature_set in h12_sets and x_h12 is None:
        raise FileNotFoundError(
            f"{feature_set} needs an H1/H2 matrix. Refusing to rename an ablation to TopNetTree."
        )
    if cv_mode not in ("nested", "leaky_original"):
        raise ValueError(cv_mode)
    n = len(y)
    if row_idx is None:
        row_idx = np.arange(n, dtype=np.int64)
    row_idx = np.asarray(row_idx, dtype=np.int64)
    splitter = cfg["cv"]["splitter"]
    seeds = [int(s) for s in cfg["cv"]["seeds"]]
    device = resolve_device(cfg.get("device", "auto"))
    log = fit_log if fit_log is not None else FitLog()
    records = []
    aux_large = count_large(x_aux) if x_aux is not None else 0

    if cv_mode == "leaky_original":
        all_idx = np.arange(n, dtype=np.int64)
        empty = np.array([], dtype=np.int64)
        log.record("normalizer_all", all_idx, empty, seed=-1, fold=-1)
        normalizer = Normalizer().fit(x_cnn, row_idx=row_idx)
        x_norm = normalizer.transform(x_cnn)
        log.record("cnn_all", all_idx, empty, seed=seeds[0], fold=-1)
        model = train_cnn(x_norm, y, cfg, seed=seeds[0], device=device)
        feats = extract_features(model, x_norm, device=device)
        log.record("rank_all", all_idx, empty, seed=seeds[0], fold=-1)
        order = rank_cnn_features(feats, y, cfg, seed=seeds[0] + 1)
        chosen = take_top(order, int(cfg["select"]["n_keep"]))
        base = assemble_features(feats[:, chosen], x_aux, x_h12, feature_set)
    else:
        normalizer = None
        order = None
        chosen = None
        base = None

    for seed in seeds:
        splits = make_splits(splitter, n, meta, cfg, seed)
        if row_keep is not None:
            splits = restrict_splits(splits, row_keep)
        pred = np.full(n, np.nan, dtype=np.float64)
        fold_of = np.full(n, -1, dtype=np.int64)
        for fold, (train_pos, test_pos) in enumerate(splits):
            train_pos = np.asarray(train_pos, dtype=np.int64)
            test_pos = np.asarray(test_pos, dtype=np.int64)
            fold_cfg = config_for_fold(cfg, seed, fold)
            if cv_mode == "nested":
                log.record("normalizer", train_pos, test_pos, seed, fold)
                fold_norm = Normalizer().fit(_slice(x_cnn, train_pos), row_idx=_slice(row_idx, train_pos))
                x_train = fold_norm.transform(_slice(x_cnn, train_pos))
                x_test = fold_norm.transform(_slice(x_cnn, test_pos))
                log.record("cnn", train_pos, test_pos, seed, fold)
                model = train_cnn(x_train, y[train_pos], fold_cfg, seed=seed + fold, device=device)
                f_train = extract_features(model, x_train, device=device)
                f_test = extract_features(model, x_test, device=device)
                log.record("rank", train_pos, test_pos, seed, fold)
                order = rank_cnn_features(f_train, y[train_pos], fold_cfg, seed=seed + 17 + fold)
                chosen = take_top(order, int(fold_cfg["select"]["n_keep"]))
                m_train = assemble_features(
                    f_train[:, chosen], _slice(x_aux, train_pos), _slice(x_h12, train_pos), feature_set
                )
                m_test = assemble_features(
                    f_test[:, chosen], _slice(x_aux, test_pos), _slice(x_h12, test_pos), feature_set
                )
            else:
                m_train = base[train_pos]
                m_test = base[test_pos]
            log.record("gbdt", train_pos, test_pos, seed, fold)
            tree = make_gbdt(fold_cfg if cv_mode == "nested" else cfg, seed=seed + 100 + fold)
            tree.fit(m_train, y[train_pos])
            pred[test_pos] = tree.predict(m_test)
            fold_of[test_pos] = fold
        for i in range(n):
            records.append(
                {
                    "row_idx": int(row_idx[i]),
                    "fold": int(fold_of[i]),
                    "seed": int(seed),
                    "y_true": float(y[i]),
                    "y_pred": float(pred[i]),
                }
            )

    frame = pd.DataFrame(records)
    summaries = []
    nonbinder = None if meta is None else meta["is_nonbinder"].to_numpy()
    for seed in seeds:
        part = frame[frame["seed"] == seed]
        # kfold scores every row. homology and a row subset leave some predictions empty.
        finite = np.isfinite(part["y_pred"].to_numpy())
        y_true = part["y_true"].to_numpy()[finite]
        y_pred = part["y_pred"].to_numpy()[finite]
        if nonbinder is None:
            summaries.append({"seed": seed, **{f"all_{k}": v for k, v in _plain(y_true, y_pred).items()}})
        else:
            row_ids = part["row_idx"].to_numpy()[finite]
            lookup = pd.Series(np.asarray(nonbinder), index=meta["row_idx"].to_numpy())
            flag = lookup.loc[row_ids].to_numpy()
            both = metrics_with_nonbinders(y_true, y_pred, flag)
            summaries.append(
                {
                    "seed": seed,
                    "pearson_r": both["all"]["pearson_r"],
                    "rmse": both["all"]["rmse"],
                    "spearman_rho": both["all"]["spearman_rho"],
                    "mae": both["all"]["mae"],
                    "pearson_r_no_nonbinders": both["excluding_nonbinders"]["pearson_r"],
                    "rmse_no_nonbinders": both["excluding_nonbinders"]["rmse"],
                }
            )
    numeric = [{k: v for k, v in row.items() if k != "seed"} for row in summaries]
    label = "optimistic" if cv_mode == "leaky_original" else "nested"
    if feature_set in ("cnn_aux", "cnn", "aux"):
        label = label + " no-H12"
    elif feature_set == "h12_original":
        label = label + " h12_original (columns of crossvalidation/X.npy, in-fold selection)"
    elif feature_set in ("h12", "cnn_h12", "cnn_aux_h12"):
        label = label + " regenerated H12 (not original)"
    return {
        "predictions": frame,
        "per_seed": summaries,
        "summary": repeat_summary(numeric),
        "cv_mode": cv_mode,
        "feature_set": feature_set,
        "headline_ok": cv_mode == "nested",
        "label": label,
        "fit_log": log,
        "aux_values_above_1e6": aux_large,
        "importance_order": None if order is None else order,
    }


def _plain(y, pred):
    from topnet.metrics import regression_metrics

    return regression_metrics(y, pred)


def loco_report(y, pred, complex_id) -> dict:
    rows = per_complex_metrics(y, pred, complex_id)
    from topnet.metrics import mean_median_complex_r

    return {"per_complex": rows, **mean_median_complex_r(rows)}
