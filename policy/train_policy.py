#!/usr/bin/env python3
"""
train_policy.py
===============
Train the cost-sensitive Policy v0 (a DecisionTree probability estimator) on
labelled fault-injection data. No RL.

Pipeline
--------
1. Discover labelled feature CSVs (``results/*_features.csv``); label each file
   benign|fault from its name (``normal*`` -> benign, else fault hint).
2. Score every row through the existing Isolation Forest + Noise Filter to
   materialise the upstream state signals (``anomaly_score``, ``suppress_proba``).
3. Build a per-container normal baseline from the benign rows and turn each row
   into the policy state vector (policy/schema.build_state).
4. Fit ``DecisionTreeClassifier`` (class_weight=balanced) to predict P(fault).
5. 5-fold CV -> apply the expected-cost rule (policy/reward) -> gate metrics.
6. Save ``{model, state_cols, feats, baseline, reward}`` + a metrics JSON.

The action is never a train target: the tree only estimates P(fault); the action
falls out of the cost rule at eval/runtime, so retuning costs needs no retrain.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from policy.schema import FEATS, STATE_COLS, build_state  # noqa: E402
from policy import reward as rw  # noqa: E402


def _med(rows: List[dict], f: str) -> float:
    v = [float(x[f]) for x in rows if x.get(f) not in (None, "", "nan")]
    return statistics.median(v) if v else 0.0


def build_baseline(benign_rows: List[dict], min_rows: int = 20) -> Dict[str, Dict[str, float]]:
    byc = collections.defaultdict(list)
    for r in benign_rows:
        byc[r.get("container") or ""].append(r)
    return {c: {f: round(_med(rs, f), 4) for f in FEATS}
            for c, rs in byc.items() if len(rs) >= min_rows}


def _score_file(path: str, bundle, noise) -> List[dict]:
    """Score one feature CSV; return rows with anomaly_score + suppress_proba added."""
    import pandas as pd
    from detector import detect as dm
    df = pd.read_csv(path)
    scored = dm.score_frame(bundle, df).to_dict(orient="records")
    out = []
    for r in scored:
        a = float(r.get("anomaly_score") or 0.0)
        sp = noise.suppress_proba(r.get("container", ""), r, a) if noise.enabled else 0.0
        r["anomaly_score"] = a
        r["suppress_proba"] = sp
        out.append(r)
    return out


def _gate_metrics(y: "List[int]", p: "List[float]") -> dict:
    """Apply the escalation cost-rule to CV probabilities and summarise by label."""
    import numpy as np
    y = np.asarray(y)
    acts = [rw.best_action(pi)[0] for pi in p]          # wait/triage/alert
    acts = np.asarray(acts)
    fault = y == 1
    benign = ~fault
    nf, nb = int(fault.sum()), int(benign.sum())
    esc = np.isin(acts, ["triage", "alert"])
    return {
        "n": int(len(y)), "n_fault": nf, "n_benign": nb,
        # faults that get escalated (triage|alert) — want high
        "fault_escalate_rate": float((fault & esc).sum() / max(nf, 1)),
        # faults parked on 'wait' — the v0 miss mode (no suppress in escalation set)
        "fault_miss_rate": float((fault & (acts == "wait")).sum() / max(nf, 1)),
        # benign that get paged — want low
        "benign_alert_rate": float((benign & (acts == "alert")).sum() / max(nb, 1)),
        # analyst load: any escalation on benign
        "benign_escalate_rate": float((benign & esc).sum() / max(nb, 1)),
        "triage_overuse": float((acts == "triage").sum() / max(len(y), 1)),
        "action_mix": {a: int((acts == a).sum()) for a in ["wait", "triage", "alert"]},
    }


def main() -> int:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    from detector.model_utils import ModelBundle

    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=os.path.join(REPO, "results"),
                    help="dir of *_features.csv (labelled by filename)")
    ap.add_argument("--if-model", default=os.path.join(REPO, "models", "isolation_forest.pkl"))
    ap.add_argument("--noise-model", default=os.path.join(REPO, "models", "noise_filter.pkl"))
    ap.add_argument("--noise-baseline", default=os.path.join(REPO, "models", "noise_baseline.json"))
    ap.add_argument("--out-model", default=os.path.join(REPO, "models", "policy_filter.pkl"))
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-leaf", type=int, default=30)
    args = ap.parse_args()

    # upstream models that produce the state signals
    bundle = ModelBundle.load(args.if_model)
    from detector.noise_filter import NoiseFilter
    noise = NoiseFilter(args.noise_model, args.noise_baseline, threshold=0.99,
                        enabled=os.path.exists(args.noise_model) and os.path.exists(args.noise_baseline))
    print(f"noise filter for state: enabled={noise.enabled} ({noise.error or 'ok'})")

    # 1-2. discover + score labelled files
    files = sorted(glob.glob(os.path.join(args.features_dir, "*_features.csv")))
    if not files:
        print(f"no *_features.csv in {args.features_dir}", file=sys.stderr)
        return 2
    rows: List[Tuple[dict, str]] = []
    for f in files:
        lbl = rw.label_from_name(os.path.basename(f))
        scored = _score_file(f, bundle, noise)
        rows.extend((r, lbl) for r in scored)
        print(f"  {os.path.basename(f):32s} -> {lbl:6s}  rows={len(scored)}")

    # 3. baseline from benign rows, then state vectors
    benign_rows = [r for r, l in rows if l == "benign"]
    baseline = build_baseline(benign_rows)
    X, y = [], []
    for r, l in rows:
        X.append(build_state(baseline, r.get("container", ""), r,
                             r.get("anomaly_score", 0.0), r.get("suppress_proba", 0.0)))
        y.append(1 if l == "fault" else 0)
    X = np.asarray(X); y = np.asarray(y)
    print(f"corpus: rows={len(y)} fault={int(y.sum())} benign={int((1 - y).sum())} "
          f"containers_baselined={len(baseline)}")

    # 4-5. fit P(fault) + CV gate metrics
    clf = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                                 class_weight="balanced", random_state=0)
    n_splits = min(5, int(y.sum()), int((1 - y).sum()))
    if n_splits >= 2:
        p_cv = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=0),
                                 method="predict_proba")[:, 1]
        metrics = _gate_metrics(y.tolist(), p_cv.tolist())
    else:
        metrics = {"n": int(len(y)), "note": "too few per class for CV"}
    print("CV gate metrics:\n" + json.dumps(metrics, indent=2))

    # 6. fit on all + persist (baseline bundled; runtime prefers live noise baseline)
    clf.fit(X, y)
    joblib.dump({"model": clf, "state_cols": STATE_COLS, "feats": FEATS,
                 "baseline": baseline, "reward": rw.REWARD}, args.out_model)
    print(f"saved {args.out_model}")
    print("rules:\n" + export_text(clf, feature_names=STATE_COLS, max_depth=args.max_depth))

    if args.metrics_out:
        json.dump({**metrics, "max_depth": args.max_depth, "min_leaf": args.min_leaf,
                   "reward": rw.REWARD}, open(args.metrics_out, "w"), indent=2)
        print(f"metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
