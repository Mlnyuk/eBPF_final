#!/usr/bin/env python3
"""
explain.py
==========
Offline SHAP feature attribution for the Isolation Forest detector.

The live API (detector/api.py) uses a cheap z-score proxy for per-request
"top features" so it stays fast under load. This script is the heavyweight,
analysis-time counterpart: it uses shap.TreeExplainer (which supports sklearn
IsolationForest) to compute proper Shapley attributions and reports which
features drive anomaly scores — overall and per fault type.

Why offline: shap.TreeExplainer on an IF with 200 trees is far too slow for the
per-window serving path; it belongs in reporting, not the request loop.

Run (inside detector pod; needs `pip install shap`):
    python detector/explain.py \
        --normal results/labeled/normal_sample.csv \
        --faults results/labeled/fault_*.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector.model_utils import (  # noqa: E402
    ModelBundle, load_config, get_feature_order, frame_to_matrix,
)

META_COLS = ["timestamp", "node", "namespace", "pod", "container"]


def _read_labeled(path: Path, feature_order: List[str]) -> pd.DataFrame:
    cols = META_COLS + list(feature_order) + ["label"]
    return pd.read_csv(path, header=None, names=cols)


def main() -> None:
    cfg = load_config()
    feature_order = get_feature_order(cfg)
    model_path = cfg.get("model", {}).get("path", "models/isolation_forest.pkl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--faults", nargs="+", required=True)
    ap.add_argument("--model", default=model_path)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-samples", type=int, default=500,
                    help="cap rows per group for SHAP (speed)")
    args = ap.parse_args()

    try:
        import shap  # noqa: F401
    except ImportError:
        raise SystemExit("shap not installed. Run: pip install shap "
                         "(--break-system-packages inside the pod)")

    bundle = ModelBundle.load(args.model)
    explainer = shap.TreeExplainer(bundle.model)

    def shap_table(df: pd.DataFrame, tag: str) -> dict:
        if len(df) > args.max_samples:
            df = df.sample(args.max_samples, random_state=0)
        X = frame_to_matrix(df, feature_order)
        sv = explainer.shap_values(X, check_additivity=False)
        sv = np.asarray(sv)
        mean_abs = np.abs(sv).mean(axis=0)
        return {feature_order[i]: float(mean_abs[i]) for i in range(len(feature_order))}

    neg = _read_labeled(Path(args.normal), feature_order)
    fault_paths = []
    for f in args.faults:
        fault_paths.extend(Path(p) for p in glob.glob(f))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    base = shap_table(neg, "normal")
    print("\n=== SHAP mean|value| per feature ===")
    print(f"{'feature':<24}{'normal':>10}", end="")
    fault_tables = {}
    for p in fault_paths:
        if not p.exists():
            continue
        df = _read_labeled(p, feature_order)
        ft = df["label"].iloc[0] if len(df) else p.stem
        fault_tables[ft] = shap_table(df, ft)
        print(f"{ft:>16}", end="")
    print()

    for feat in feature_order:
        line = f"{feat:<24}{base[feat]:>10.4f}"
        rec = {"feature": feat, "normal": base[feat]}
        for ft, tbl in fault_tables.items():
            line += f"{tbl[feat]:>16.4f}"
            rec[ft] = tbl[feat]
        print(line)
        rows.append(rec)

    # top driver per fault (largest SHAP increase vs normal)
    print("\n=== Top SHAP driver per fault (vs normal baseline) ===")
    for ft, tbl in fault_tables.items():
        deltas = sorted(((tbl[f] - base[f], f) for f in feature_order), reverse=True)
        top3 = ", ".join(f"{f}(+{d:.3f})" for d, f in deltas[:3])
        print(f"  {ft:<16} {top3}")

    out_csv = out_dir / "shap_attribution.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[explain] wrote {out_csv}")


if __name__ == "__main__":
    main()
