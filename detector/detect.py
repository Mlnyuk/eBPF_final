#!/usr/bin/env python3
"""
detect.py
=========
Load a trained Isolation Forest bundle and score new feature vectors.

Each result row contains:
    timestamp, node, namespace, pod, container,
    anomaly_score (0..1, higher = more anomalous),
    is_anomaly (bool),
    top_features (list of {feature, value, z})

Usage
-----
    # Score a feature file, print summary, write results CSV:
    python detector/detect.py --data results/cpu_stress_features.csv \
        --out results/anomaly_scores.csv

    # Score and only show flagged anomalies:
    python detector/detect.py --data data/live.jsonl --only-anomalies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector.model_utils import (  # noqa: E402
    ModelBundle, load_config, frame_to_matrix, raw_anomaly_score,
    normalize_score, top_feature_contributions,
)

META_COLS = ["timestamp", "node", "namespace", "pod", "container"]


def _read_feature_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def score_frame(bundle: ModelBundle, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Return a DataFrame of detection results for every row in df."""
    feature_order = bundle.feature_order
    X = frame_to_matrix(df, feature_order)
    raw = raw_anomaly_score(bundle.model, X)
    norm = normalize_score(raw, bundle.raw_score_min, bundle.raw_score_max)
    is_anom = norm >= bundle.score_threshold

    means = np.asarray(bundle.metadata.get("train_means"), dtype=np.float64) \
        if bundle.metadata.get("train_means") is not None else None
    stds = np.asarray(bundle.metadata.get("train_stds"), dtype=np.float64) \
        if bundle.metadata.get("train_stds") is not None else None

    rows = []
    for i in range(len(df)):
        top = top_feature_contributions(X[i], feature_order, means, stds, k=top_k)
        meta = {c: (df.iloc[i][c] if c in df.columns else None) for c in META_COLS}
        rows.append({
            **meta,
            "anomaly_score": round(float(norm[i]), 6),
            "is_anomaly": bool(is_anom[i]),
            "top_features": top,
        })
    return pd.DataFrame(rows)


def detect_one(bundle: ModelBundle, feature_dict: dict, meta: Optional[dict] = None,
               top_k: int = 3) -> dict:
    """Score a single feature dict (used by the API)."""
    df = pd.DataFrame([{**(meta or {}), **feature_dict}])
    res = score_frame(bundle, df, top_k=top_k)
    return res.iloc[0].to_dict()


def main() -> None:
    cfg = load_config()
    model_path = cfg.get("model", {}).get("path", "models/isolation_forest.pkl")

    ap = argparse.ArgumentParser(description="Detect anomalies in feature data.")
    ap.add_argument("--data", nargs="+", required=True,
                    help="feature files to score (.csv/.jsonl)")
    ap.add_argument("--model", default=model_path)
    ap.add_argument("--out", default=None, help="write results CSV to this path")
    ap.add_argument("--only-anomalies", action="store_true",
                    help="print only rows flagged as anomalous")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    bundle = ModelBundle.load(args.model)

    frames = []
    for p in args.data:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"feature file not found: {path}")
        frames.append(_read_feature_file(path))
    df = pd.concat(frames, ignore_index=True)

    results = score_frame(bundle, df, top_k=args.top_k)
    n_anom = int(results["is_anomaly"].sum())
    print(f"[detect] scored {len(results)} rows; flagged {n_anom} anomalies "
          f"(threshold={bundle.score_threshold:.4f})")

    if args.out:
        out = results.copy()
        out["top_features"] = out["top_features"].apply(json.dumps)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[detect] wrote results -> {args.out}")

    show = results[results["is_anomaly"]] if args.only_anomalies else results
    for _, r in show.head(20).iterrows():
        print(f"  {r['timestamp']} {r['namespace']}/{r['pod']} "
              f"score={r['anomaly_score']:.4f} anomaly={r['is_anomaly']} "
              f"top={[t['feature'] for t in r['top_features']]}")


if __name__ == "__main__":
    main()
