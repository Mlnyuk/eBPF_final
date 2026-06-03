#!/usr/bin/env python3
"""
train_noise_filter.py
=====================
Distill Qwen3-32B's benign|suspicious group labels into a small DecisionTree that
suppresses benign daemon noise at runtime (no LLM in the hot path).

Inputs
------
  --rows    archive CSV slice: flagged anomalies + normal rows (normal rows build
            the per-container baseline; anomalies are the training samples).
  --labels  JSON {"labels":[{node,container,trigger,verdict,confidence}, ...]}
            produced by the nightly LLM labeler (scripts/qwen_distill_label.py).

Outputs
-------
  --out-model     joblib {model, cols, feats}  (default models/noise_filter.pkl)
  --out-baseline  JSON per-container normal medians (default models/noise_baseline.json)

Each anomaly row inherits its (node,container,trigger) group's verdict
(weak supervision). Features = anomaly_score + has_baseline + per-feature signed
log-deviation vs the container's own normal baseline — the same view the LLM was
given, so the tree learns "benign unless it departs from its own normal".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import collections
import statistics
from typing import Dict, List

FEATS = ["syscall_read_rate", "syscall_write_rate", "syscall_open_rate",
         "tcp_connect_rate", "tcp_retransmit_rate", "network_rx_bytes",
         "network_tx_bytes", "disk_read_bytes", "disk_write_bytes",
         "disk_io_latency_ms", "process_exec_count", "process_fork_count",
         "context_switch_count", "cpu_utilization"]
_EPS = 1e-6


def _med(rows: List[dict], f: str) -> float:
    v = [float(x[f]) for x in rows if x.get(f) not in (None, "", "nan")]
    return statistics.median(v) if v else 0.0


def build_baseline(normal: List[dict], min_rows: int = 20) -> Dict[str, Dict[str, float]]:
    byc = collections.defaultdict(list)
    for r in normal:
        byc[r["container"]].append(r)
    return {c: {f: round(_med(rs, f), 4) for f in FEATS}
            for c, rs in byc.items() if len(rs) >= min_rows}


def vector(baseline, container, feats, anomaly_score):
    bl = baseline.get(container)
    x = [float(anomaly_score), 1.0 if bl else 0.0]
    for f in FEATS:
        o = float(feats.get(f) or 0.0)
        b = (bl.get(f) if bl else 0.0) or 0.0
        x.append(math.log10((o + _EPS) / (b + _EPS)))
    return x


def main() -> int:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-model", default="models/noise_filter.pkl")
    ap.add_argument("--out-baseline", default="models/noise_baseline.json")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-leaf", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="runtime suppress proba (reported here for the eval only)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.rows)))
    labels = {(l.get("node"), l.get("container"), l.get("trigger")): l.get("verdict")
              for l in json.load(open(args.labels))["labels"]}
    normal = [r for r in rows if r.get("is_anomaly") == "False"]
    anom = [r for r in rows if r.get("is_anomaly") == "True"]
    baseline = build_baseline(normal)

    cols = ["anomaly_score", "has_baseline"] + [f"dev_{f}" for f in FEATS]
    X, y = [], []
    for r in anom:
        v = labels.get((r["node"], r["container"], r["trigger"]))
        if v is None:
            continue
        X.append(vector(baseline, r["container"], r, r["anomaly_score"]))
        y.append(1 if v == "benign" else 0)        # 1 = suppress(benign)
    X = np.array(X); y = np.array(y)
    print(f"labeled rows={len(y)} benign(suppress)={int(y.sum())} "
          f"suspicious(keep)={int((1 - y).sum())} containers_baselined={len(baseline)}")

    clf = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                                 class_weight="balanced", random_state=0)
    proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    supp = proba >= args.threshold
    keepN = int((y == 0).sum()); benN = int((y == 1).sum())
    fs = int(((y == 0) & supp).sum()); gs = int(((y == 1) & supp).sum())
    print(f"CV @thr={args.threshold}: noise_reduced={gs/max(benN,1):.1%} "
          f"false_suppress={fs/max(keepN,1):.2%} ({fs} real silenced)")
    print("confusion(thr) [true x supp?]:\n", confusion_matrix(y, supp.astype(int)))

    clf.fit(X, y)
    joblib.dump({"model": clf, "cols": cols, "feats": FEATS}, args.out_model)
    json.dump(baseline, open(args.out_baseline, "w"))
    print(f"saved {args.out_model} + {args.out_baseline}")
    print("rules:\n" + export_text(clf, feature_names=cols, max_depth=args.max_depth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
