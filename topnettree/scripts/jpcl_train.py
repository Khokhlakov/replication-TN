#!/usr/bin/env python3
"""Protocol A, protocol B, and PDB-grouped nested CV for the JPCL 2022 run.

The target column is ddg. Positive ddg is weaker binding. This script does not
negate it. Feature sets cnn and cnn_aux are labeled no-H12 by fit_pipeline.
Rows whose H0 vector is non-finite were failed generations: they are removed
from that fit and listed. No other row is removed, except a PDB that also
appears in the protocol A test set. That holdout is the same-PDB rule.

A training table whose row count is not 4169 is recorded as skempi_singles,
not as S4169. G4 did not find a filter with the published counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topnet.cnn import extract_features, resolve_device, train_cnn  # noqa: E402
from topnet.config import load_config  # noqa: E402
from topnet.cv import FitLog  # noqa: E402
from topnet.gbdt import assemble_features, make_gbdt  # noqa: E402
from topnet.metrics import regression_metrics  # noqa: E402
from topnet.normalize import Normalizer  # noqa: E402
from topnet.pipeline import fit_pipeline, predict_pipeline, save_bundle  # noqa: E402
from topnet.runs import start_run, write_outputs  # noqa: E402
from topnet.select import rank_cnn_features, take_top  # noqa: E402

KEYS = ["pdb", "chain", "wt", "resid", "mut"]


def load_table(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ddg" not in frame.columns:
        raise SystemExit(f"{path} has no ddg column")
    if "ddg_is_proxy" not in frame.columns:
        frame["ddg_is_proxy"] = 0
    if "dataset" not in frame.columns:
        frame["dataset"] = Path(path).stem
    if "complex_id" not in frame.columns:
        frame["complex_id"] = frame["pdb"]
    if "ab_bind_overlap" not in frame.columns:
        raise SystemExit(f"{path} has no ab_bind_overlap column")
    frame["ab_bind_overlap"] = frame["ab_bind_overlap"].astype(int)
    return frame.reset_index(drop=True)


def load_h0(path: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.load(path)
    if array.ndim == 2 and array.shape[1] == 5184:
        array = array.reshape(len(array), 48, 108)
    if array.shape[1:] != (48, 108):
        raise SystemExit(f"{path} shape {array.shape} is not an H0 tensor")
    if len(array) != n_rows:
        raise SystemExit(f"{path} has {len(array)} rows and the table has {n_rows}")
    ok = np.isfinite(array).all(axis=(1, 2))
    return array, ok


def apply_mask(meta: pd.DataFrame, array: np.ndarray, ok: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    dropped = meta.loc[~ok, [column for column in KEYS if column in meta.columns]].copy()
    return meta.loc[ok].reset_index(drop=True), array[ok], dropped


def require_g2(path: str) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("verdict") != "IDENTICAL" or int(report.get("n_failed_generation", 1)) != 0:
        raise SystemExit(f"G2 did not pass: {path} verdict={report.get('verdict')} failures={report.get('n_failed_generation')}")
    return report


def assert_split(train: np.ndarray, test: np.ndarray, meta: pd.DataFrame) -> None:
    train_meta = meta.iloc[np.asarray(train)]
    test_meta = meta.iloc[np.asarray(test)]
    shared = set(train_meta["pdb"].astype(str)) & set(test_meta["pdb"].astype(str))
    if shared:
        raise SystemExit(f"PDB straddles train and test: {sorted(shared)[:8]}")
    both = pd.merge(train_meta[KEYS].astype(str), test_meta[KEYS].astype(str), on=KEYS)
    if len(both):
        raise SystemExit(f"{len(both)} mutation keys are on both sides of the split")
    overlap_train = set(train_meta.loc[train_meta["ab_bind_overlap"] == 1, "pdb"].astype(str))
    overlap_test = set(test_meta.loc[test_meta["ab_bind_overlap"] == 1, "pdb"].astype(str))
    if overlap_train & overlap_test:
        raise SystemExit("ab_bind_overlap rows straddle the split")


def holdout_shared_pdb(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    hit = train["pdb"].astype(str).isin(set(test["pdb"].astype(str)))
    moved = train.loc[hit, "pdb"].astype(str).value_counts().to_dict()
    return np.flatnonzero(~hit.to_numpy()), moved


def dataset_label(n_rows: int) -> str:
    if n_rows == 4169:
        return "S4169"
    return "skempi_singles_not_s4169"


def fit_one(x, y, aux, cfg, seed, feature_set, row_idx):
    bundle = fit_pipeline(x, y, aux, cfg, seed=seed, feature_set=feature_set, row_idx=row_idx)
    if bundle["label"] != "no-H12":
        raise SystemExit(f"refusing feature set labeled {bundle['label']}")
    return bundle


def load_aux(path: str | None, n_rows: int, ok: np.ndarray):
    if path is None:
        return None
    aux = np.load(path)
    if len(aux) != n_rows:
        raise SystemExit(f"{path} has {len(aux)} rows, expected {n_rows}")
    return aux[ok]


def metrics_block(y, pred, meta: pd.DataFrame):
    overall = regression_metrics(y, pred)
    per_complex = []
    per_position = []
    for cid, idx in meta.groupby("complex_id").groups.items():
        positions = meta.index.get_indexer(idx)
        row = regression_metrics(y[positions], pred[positions])
        row["complex_id"] = str(cid)
        per_complex.append(row)
    for resid, idx in meta.groupby("resid").groups.items():
        positions = meta.index.get_indexer(idx)
        row = regression_metrics(y[positions], pred[positions])
        row["resid"] = str(resid)
        per_position.append(row)
    return overall, per_complex, per_position


def nested_grouped(x, y, aux, meta, cfg, feature_set):
    groups = meta["pdb"].astype(str).to_numpy()
    if len(set(groups)) < 10:
        raise SystemExit("fewer than 10 PDB groups")
    splits = list(GroupKFold(n_splits=10).split(np.zeros(len(y)), groups=groups))
    log = FitLog()
    pred = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    device = resolve_device(cfg.get("device", "auto"))
    row_idx = meta["row_idx"].to_numpy() if "row_idx" in meta.columns else np.arange(len(y))
    for fold, (train, test) in enumerate(splits):
        assert_split(train, test, meta)
        log.record("normalizer", train, test, 0, fold)
        normalizer = Normalizer().fit(x[train], row_idx=np.asarray(row_idx)[train])
        x_train = normalizer.transform(x[train])
        x_test = normalizer.transform(x[test])
        log.record("cnn", train, test, 0, fold)
        model = train_cnn(x_train, y[train], cfg, seed=fold, device=device)
        f_train = extract_features(model, x_train, device=device)
        f_test = extract_features(model, x_test, device=device)
        log.record("rank", train, test, 0, fold)
        chosen = take_top(rank_cnn_features(f_train, y[train], cfg, seed=17 + fold), int(cfg["select"]["n_keep"]))
        aux_train = None if aux is None else aux[train]
        aux_test = None if aux is None else aux[test]
        m_train = assemble_features(f_train[:, chosen], aux_train, None, feature_set)
        m_test = assemble_features(f_test[:, chosen], aux_test, None, feature_set)
        log.record("gbdt", train, test, 0, fold)
        tree = make_gbdt(cfg, seed=100 + fold)
        tree.fit(m_train, y[train])
        pred[test] = tree.predict(m_test)
        fold_of[test] = fold
    frame = meta[KEYS].copy()
    frame["row_idx"] = row_idx
    frame["fold"] = fold_of
    frame["seed"] = 0
    frame["y_true"] = y
    frame["y_pred"] = pred
    return frame, regression_metrics(y, pred), log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B", "cv"], required=True)
    parser.add_argument("--g2-json", required=True)
    parser.add_argument("--feature-set", choices=["cnn", "cnn_aux"], default="cnn")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skempi-table", required=True)
    parser.add_argument("--skempi-h0", required=True)
    parser.add_argument("--skempi-aux")
    parser.add_argument("--starr-table")
    parser.add_argument("--starr-h0")
    parser.add_argument("--starr-aux")
    parser.add_argument("--chan-table")
    parser.add_argument("--chan-h0")
    parser.add_argument("--chan-aux")
    args = parser.parse_args()
    report = require_g2(args.g2_json)
    cfg = load_config(args.config)
    if args.feature_set == "cnn_aux" and not args.skempi_aux:
        raise SystemExit("cnn_aux needs --skempi-aux from a G2-identical auxiliary block")
    sk_raw = load_table(args.skempi_table)
    if int(sk_raw["ddg_is_proxy"].astype(int).max()) != 0:
        raise SystemExit("SKEMPI table contains proxy labels")
    x_raw, ok = load_h0(args.skempi_h0, len(sk_raw))
    sk, x, dropped = apply_mask(sk_raw, x_raw, ok)
    aux = load_aux(args.skempi_aux, len(sk_raw), ok)
    y = sk["ddg"].to_numpy(float)
    run = start_run(f"jpcl_protocol_{args.protocol}_{args.feature_set}", cfg)
    (run / "g2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dropped.to_csv(run / "dropped_feature_rows.csv", index=False)
    name = dataset_label(len(sk_raw))

    if args.protocol == "cv":
        frame, summary, log = nested_grouped(x, y, aux, sk, cfg, args.feature_set)
        write_outputs(run, frame, {
            "label": "nested no-H12",
            "dataset": name,
            "n_rows_in_table": int(len(sk_raw)),
            "n_rows_fit": int(len(sk)),
            "summary": summary,
            "n_fit_events": len(log.events),
            "splitter": "GroupKFold by pdb, seed not used (deterministic given row order)",
        })
        print(run)
        print(name, summary)
        return

    def prepared(table_path, h0_path, aux_path):
        raw = load_table(table_path)
        array, mask = load_h0(h0_path, len(raw))
        meta, values, failed = apply_mask(raw, array, mask)
        return meta, values, load_aux(aux_path, len(raw), mask), failed

    if args.protocol == "A":
        device = resolve_device(cfg.get("device", "auto"))
        summaries = []
        for name_i, table, feats, aux_path in (
            ("starr", args.starr_table, args.starr_h0, args.starr_aux),
            ("chan", args.chan_table, args.chan_h0, args.chan_aux),
        ):
            if not table or not feats:
                raise SystemExit(f"protocol A needs --{name_i}-table and --{name_i}-h0")
            test_meta, x_test, aux_test, failed = prepared(table, feats, aux_path)
            failed.to_csv(run / f"{name_i}_dropped_feature_rows.csv", index=False)
            keep, moved = holdout_shared_pdb(sk, test_meta)
            assert_split(keep, np.array([], dtype=int), sk)
            both = pd.merge(sk.iloc[keep][KEYS].astype(str), test_meta[KEYS].astype(str), on=KEYS)
            if len(both) or set(sk.iloc[keep]["pdb"].astype(str)) & set(test_meta["pdb"].astype(str)):
                raise SystemExit(f"{name_i}: train/test key or PDB intersection remains")
            bundle = fit_one(
                x[keep], y[keep], None if aux is None else aux[keep], cfg, int(cfg["seed"]),
                args.feature_set, sk.iloc[keep]["row_idx"].to_numpy() if "row_idx" in sk.columns else keep,
            )
            pred = predict_pipeline(bundle, x_test, aux_test, device=device)
            overall, per_c, per_p = metrics_block(test_meta["ddg"].to_numpy(float), pred, test_meta)
            proxy = int(test_meta["ddg_is_proxy"].astype(int).max()) == 1
            payload = {
                "test": name_i,
                "units": "proxy_rank_first" if proxy else "kcal_per_mol",
                "label": bundle["label"],
                "n_train": int(len(keep)),
                "n_test": int(len(test_meta)),
                "dataset_trained": name,
                "pdb_moved_out_of_train": moved,
                "metrics": overall,
                "note": "Pearson and RMSE of ddg equal Pearson and RMSE of BFE change, because both sides change sign together.",
            }
            (run / f"{name_i}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            pd.DataFrame(per_c).to_csv(run / f"{name_i}_per_complex.csv", index=False)
            pd.DataFrame(per_p).to_csv(run / f"{name_i}_per_position.csv", index=False)
            test_meta.assign(y_pred_ddg=pred).to_csv(run / f"{name_i}_predictions.csv", index=False)
            save_bundle(bundle, run / f"bundle_{name_i}.joblib")
            summaries.append(payload)
            print(name_i, payload["units"], "spearman", overall["spearman_rho"], "pearson", overall["pearson_r"], "n", overall["n"])
        (run / "metrics.json").write_text(json.dumps({"protocol": "A", "tests": summaries}, indent=2), encoding="utf-8")
        print(run)
        return

    if args.feature_set == "cnn_aux":
        raise SystemExit("protocol B cnn_aux is not wired until each block has its own aligned aux array")
    parts = [sk]
    arrays = [x]
    for table, feats, aux_path in (
        (args.starr_table, args.starr_h0, args.starr_aux),
        (args.chan_table, args.chan_h0, args.chan_aux),
    ):
        if not table or not feats:
            raise SystemExit("protocol B needs Starr and Chan tables and H0 arrays")
        meta, values, _aux_i, failed = prepared(table, feats, aux_path)
        failed.to_csv(run / f"{meta['dataset'].iloc[0]}_dropped_feature_rows.csv", index=False)
        parts.append(meta)
        arrays.append(values)
    all_meta = pd.concat(parts, ignore_index=True)
    all_x = np.concatenate(arrays, axis=0)
    bundle = fit_one(
        all_x, all_meta["ddg"].to_numpy(float), None, cfg, int(cfg["seed"]),
        args.feature_set, np.arange(len(all_meta)),
    )
    save_bundle(bundle, run / "bundle_protocol_B.joblib")
    composition = {
        "label": bundle["label"],
        "n": int(len(all_meta)),
        "n_proxy": int(all_meta["ddg_is_proxy"].astype(int).sum()),
        "datasets": all_meta["dataset"].value_counts().to_dict(),
        "skempi_dataset_label": name,
        "note": "DMS rows were training rows. A correlation computed on them is not a validation result.",
    }
    (run / "protocol_B_composition.json").write_text(json.dumps(composition, indent=2), encoding="utf-8")
    print(run)


if __name__ == "__main__":
    main()
