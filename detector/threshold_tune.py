#!/usr/bin/env python3
"""
threshold_tune.py
=================
Evaluate the trained Isolation Forest against LABELED data (normal vs real
fault windows) and find an optimal anomaly-score threshold.

Inputs are the headerless labeled CSVs produced by
scripts/run_fault_experiments_v2.sh:
    results/labeled/fault_<type>.csv   (positive, label col = fault type)
    results/labeled/normal_sample.csv  (negative, label col = "normal")
Each row layout (no header):
    timestamp,node,namespace,pod,container,<13 features...>,label

Outputs:
    - ROC AUC, PR AUC (average precision)
    - optimal thresholds by Youden's J and by max F1
    - detection rate per fault type at the chosen threshold
    - results/threshold_tuning_<ts>.txt  (summary)
    - results/roc_pr_points.csv          (curve points for later plotting)

Run (inside detector pod, which has sklearn/pandas/numpy + the model):
    python detector/threshold_tune.py \
        --normal results/labeled/normal_sample.csv \
        --faults results/labeled/fault_*.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector.model_utils import (  # noqa: E402
    ModelBundle, load_config, get_feature_order, frame_to_matrix,
    raw_anomaly_score, normalize_score,
)

META_COLS = ["timestamp", "node", "namespace", "pod", "container"]


def _read_labeled(path: Path, feature_order: List[str]) -> pd.DataFrame:
    cols = META_COLS + list(feature_order) + ["label"]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def main() -> None:
    cfg = load_config()
    feature_order = get_feature_order(cfg)
    model_path = cfg.get("model", {}).get("path", "models/isolation_forest.pkl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="normal_sample.csv (negatives)")
    ap.add_argument("--faults", nargs="+", required=True,
                    help="fault_*.csv files (positives); globs allowed")
    ap.add_argument("--model", default=model_path)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    bundle = ModelBundle.load(args.model)

    # expand globs
    fault_paths: List[Path] = []
    for f in args.faults:
        fault_paths.extend(Path(p) for p in glob.glob(f))
    neg = _read_labeled(Path(args.normal), feature_order)
    pos_frames = [_read_labeled(p, feature_order) for p in fault_paths if Path(p).exists()]
    if not pos_frames:
        raise SystemExit("no fault files found")
    pos = pd.concat(pos_frames, ignore_index=True)

    print(f"[tune] negatives(normal)={len(neg)}  positives(fault)={len(pos)}")
    for ft, grp in pos.groupby("label"):
        print(f"        fault '{ft}': {len(grp)} rows")

    # score everything
    all_df = pd.concat([neg, pos], ignore_index=True)
    X = frame_to_matrix(all_df, feature_order)
    raw = raw_anomaly_score(bundle.model, X)
    score = normalize_score(raw, bundle.raw_score_min, bundle.raw_score_max)
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))]).astype(int)

    from sklearn.metrics import (
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
        f1_score,
    )

    roc_auc = roc_auc_score(y, score)
    ap_score = average_precision_score(y, score)
    fpr, tpr, roc_thr = roc_curve(y, score)
    prec, rec, pr_thr = precision_recall_curve(y, score)

    # Youden's J optimal threshold
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    youden_thr = float(roc_thr[j_idx])

    # max-F1 threshold (scan PR thresholds)
    f1s = []
    for t in pr_thr:
        f1s.append(f1_score(y, (score >= t).astype(int), zero_division=0))
    f1s = np.array(f1s)
    f1_idx = int(np.argmax(f1s)) if len(f1s) else 0
    f1_thr = float(pr_thr[f1_idx]) if len(pr_thr) else 0.5
    best_f1 = float(f1s[f1_idx]) if len(f1s) else 0.0

    cur_thr = bundle.score_threshold

    def rates(thr: float):
        pred = (score >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        tpr_ = tp / max(tp + fn, 1); fpr_ = fp / max(fp + tn, 1)
        prec_ = tp / max(tp + fp, 1)
        return tp, fn, fp, tn, tpr_, fpr_, prec_

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"threshold_tuning_{ts}.txt"

    lines = []
    lines.append("=== Isolation Forest threshold tuning ===")
    lines.append(f"negatives(normal)={len(neg)}  positives(fault)={len(pos)}")
    lines.append(f"ROC AUC = {roc_auc:.4f}")
    lines.append(f"PR  AUC (avg precision) = {ap_score:.4f}")
    lines.append("")
    for name, thr in [("current(model)", cur_thr), ("Youden-J", youden_thr),
                      ("max-F1", f1_thr)]:
        tp, fn, fp, tn, tpr_, fpr_, prec_ = rates(thr)
        lines.append(f"[{name}] thr={thr:.4f}  TPR(recall)={tpr_:.3f}  "
                     f"FPR={fpr_:.3f}  precision={prec_:.3f}  "
                     f"(TP={tp} FN={fn} FP={fp} TN={tn})")
    lines.append(f"max F1 = {best_f1:.4f} @ thr={f1_thr:.4f}")
    lines.append("")
    lines.append("=== Per-fault detection rate (recall) ===")
    for thr_name, thr in [("current", cur_thr), ("Youden-J", youden_thr),
                          ("max-F1", f1_thr)]:
        lines.append(f"  @ {thr_name} (thr={thr:.4f}):")
        for ft, grp in pos.groupby("label"):
            gX = frame_to_matrix(grp, feature_order)
            gs = normalize_score(raw_anomaly_score(bundle.model, gX),
                                 bundle.raw_score_min, bundle.raw_score_max)
            det = float((gs >= thr).mean())
            lines.append(f"      {ft:<16} recall={det:.3f}  (n={len(grp)})")

    text = "\n".join(lines)
    print("\n" + text)
    report.write_text(text + "\n")
    print(f"\n[tune] wrote {report}")

    # dump curve points
    pts = out_dir / "roc_pr_points.csv"
    m = max(len(fpr), len(prec))
    def pad(a): return np.concatenate([a, [np.nan] * (m - len(a))])
    pd.DataFrame({
        "fpr": pad(fpr), "tpr": pad(tpr),
        "precision": pad(prec), "recall": pad(rec),
    }).to_csv(pts, index=False)
    print(f"[tune] wrote {pts}")


if __name__ == "__main__":
    main()
