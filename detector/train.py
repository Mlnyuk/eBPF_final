#!/usr/bin/env python3
"""
train.py
========
Train an Isolation Forest on NORMAL-state feature data and persist the model
bundle to models/isolation_forest.pkl.

Usage
-----
    python detector/train.py --data results/normal_features.csv
    python detector/train.py --data data/a.csv data/b.jsonl --contamination 0.05

Input may be CSV or JSONL feature files (as produced by the collector or the
mock generator). Only the NORMAL baseline should be used for training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --- make repo root importable when run as a script ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import IsolationForest  # noqa: E402

from detector.model_utils import (  # noqa: E402
    ModelBundle, load_config, get_feature_order, frame_to_matrix,
    raw_anomaly_score,
)


def _read_feature_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def load_features(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"feature file not found: {path}")
        frames.append(_read_feature_file(path))
    if not frames:
        raise ValueError("no feature data loaded")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("detector", {})

    ap = argparse.ArgumentParser(description="Train Isolation Forest on normal features.")
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more normal-state feature files (.csv/.jsonl)")
    ap.add_argument("--model-out", default=mcfg.get("path", "models/isolation_forest.pkl"))
    ap.add_argument("--contamination", type=float, default=mcfg.get("contamination", 0.02),
                    help="expected anomaly fraction (default from config)")
    ap.add_argument("--n-estimators", type=int, default=mcfg.get("n_estimators", 200))
    ap.add_argument("--random-state", type=int, default=mcfg.get("random_state", 42))
    ap.add_argument("--threshold", type=float, default=dcfg.get("score_threshold"),
                    help="normalized score threshold (0..1). If unset, derived "
                         "from the contamination quantile of training scores.")
    ap.add_argument("--sigma-threshold", type=float,
                    default=dcfg.get("sigma_threshold", 10.0),
                    help="z-tail side-rule: flag if any feature exceeds this many "
                         "std-devs from the train mean (catches single-axis "
                         "anomalies IsolationForest misses, e.g. cpu-stress).")
    args = ap.parse_args()

    feature_order = get_feature_order(cfg)
    df = load_features(args.data)
    print(f"[train] loaded {len(df)} feature rows from {len(args.data)} file(s)")
    print(f"[train] feature order ({len(feature_order)}): {feature_order}")

    X = frame_to_matrix(df, feature_order)
    if len(X) < 10:
        print(f"[train] WARNING: only {len(X)} samples; model may be unreliable")

    max_samples = mcfg.get("max_samples", "auto")
    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=max_samples,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Calibration on the training set.
    raw = raw_anomaly_score(model, X)
    raw_min, raw_max = float(raw.min()), float(raw.max())

    # Derive threshold if not explicitly provided: the (1 - contamination)
    # quantile of normalized training scores -> flags ~contamination fraction.
    span = max(raw_max - raw_min, 1e-12)
    norm_train = np.clip((raw - raw_min) / span, 0.0, 1.0)
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(np.quantile(norm_train, 1.0 - args.contamination))
    print(f"[train] raw score range=[{raw_min:.4f}, {raw_max:.4f}]  "
          f"threshold={threshold:.4f}")

    bundle = ModelBundle(
        model=model,
        feature_order=feature_order,
        raw_score_min=raw_min,
        raw_score_max=raw_max,
        contamination=args.contamination,
        score_threshold=threshold,
        sigma_threshold=args.sigma_threshold,
        metadata={
            "n_train_samples": int(len(X)),
            "train_means": X.mean(axis=0).tolist(),
            "train_stds": X.std(axis=0).tolist(),
            "sklearn_offset_": float(getattr(model, "offset_", 0.0)),
        },
    )
    bundle.save(args.model_out)
    print(f"[train] saved model bundle -> {args.model_out}")


if __name__ == "__main__":
    main()
