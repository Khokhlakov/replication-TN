"""Paired leaky-minus-nested metrics. Non-binders come from is_nonbinder, not from the label value.

Appends a section to docs/RESULTS.md marked OPTIMISTIC. That section is not a headline number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from topnet.config import load_config, repo_root
from topnet.data import assert_label_tables
from topnet.metrics import metrics_with_nonbinders


def _predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.csv"
    frame = pd.read_csv(path)
    need = {"row_idx", "seed", "y_true", "y_pred"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame


def _pair(nested: pd.DataFrame, leaky: pd.DataFrame, nonbinder: pd.Series) -> pd.DataFrame:
    keys = ["row_idx", "seed"]
    merged = nested.merge(leaky, on=keys, suffixes=("_nested", "_leaky"))
    rows = []
    for seed, part in merged.groupby("seed"):
        flag = nonbinder.loc[part["row_idx"]].to_numpy()
        nest = metrics_with_nonbinders(part["y_true_nested"], part["y_pred_nested"], flag)
        leak = metrics_with_nonbinders(part["y_true_leaky"], part["y_pred_leaky"], flag)
        row = {"seed": int(seed)}
        for scope, left, right in (
            ("all", nest["all"], leak["all"]),
            ("no_nonbinders", nest["excluding_nonbinders"], leak["excluding_nonbinders"]),
        ):
            for metric in ("pearson_r", "spearman_rho", "rmse", "mae"):
                row[f"nested_{scope}_{metric}"] = left[metric]
                row[f"leaky_{scope}_{metric}"] = right[metric]
                row[f"gap_{scope}_{metric}"] = right[metric] - left[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nested", required=True)
    parser.add_argument("--leaky", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = load_config("configs/default.yaml")
    _y, meta, _sup = assert_label_tables(cfg)
    nonbinder = pd.Series(meta["is_nonbinder"].to_numpy(), index=meta["row_idx"].to_numpy())
    table = _pair(_predictions(args.nested), _predictions(args.leaky), nonbinder)
    out = Path(args.out) if args.out else Path(args.leaky)
    if out.is_dir() or args.out is None:
        dest = (Path(args.leaky) if Path(args.leaky).is_dir() else Path(args.leaky).parent) / "paired_gap.csv"
    else:
        dest = out
    table.to_csv(dest, index=False)
    means = table.mean(numeric_only=True)
    section = [
        "",
        "## OPTIMISTIC - not a headline number",
        "",
        "Paired gap is leaky minus nested, on the same seeds and row_idx. "
        "Non-binders are excluded with `is_nonbinder`, not by a label cutoff.",
        f"Nested predictions: `{args.nested}`. Leaky predictions: `{args.leaky}`.",
        "",
        "| scope | Pearson r gap | Spearman gap | RMSE gap | MAE gap |",
        "|---|---|---|---|---|",
    ]
    for scope in ("all", "no_nonbinders"):
        section.append(
            "| {scope} | {r:.4f} | {s:.4f} | {e:.4f} | {m:.4f} |".format(
                scope=scope,
                r=means[f"gap_{scope}_pearson_r"],
                s=means[f"gap_{scope}_spearman_rho"],
                e=means[f"gap_{scope}_rmse"],
                m=means[f"gap_{scope}_mae"],
            )
        )
    section.append("")
    section.append("Per seed:")
    section.append("")
    section.append(table.to_csv(index=False))
    results = repo_root() / "docs" / "RESULTS.md"
    results.write_text(results.read_text(encoding="utf-8") + "\n".join(section), encoding="utf-8")
    print(dest)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
