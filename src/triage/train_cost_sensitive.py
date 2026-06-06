#!/usr/bin/env python3
"""
train_cost_sensitive.py
=======================
Train cost-sensitive severity classifiers for triage.

The classifier predicts ``P(true_class | x)`` over {normal, low, medium, high};
the *action* is then chosen by expected-cost minimisation (see cost_matrix.py).
This decoupling means the cost grid can be retuned without retraining.

Models:
* DecisionTreeClassifier (max_depth 3-5, class_weight=balanced) -> triage_tree
* RandomForestClassifier (class_weight=balanced_subsample)      -> random_forest
* XGBoost (optional, only if the dependency is importable)      -> xgboost

CPU-only by default (no GPU required). Saves joblib bundles that carry the model,
feature order, class order, and the cost matrix, so the detector can load one and
run lightweight CPU inference.

Usage:
    python -m src.triage.train_cost_sensitive \
      --cases data/cases/cases.jsonl \
      --labels data/labels/labels.jsonl \
      --out models/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.triage.cost_matrix import CLASSES, COST_MATRIX
from src.triage.case_bundle import CASE_FEATURES
from src.triage.label_store import LabelStore


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_dataset(cases_path: Path, labels_path: Path
                 ) -> Tuple[np.ndarray, List[str], Dict[str, List], List[Dict]]:
    """Join cases + resolved labels.

    Returns (X, y, groups, rows) where groups maps a group-kind to a per-row list
    (node / fault_type / time bucket) for holdout evaluation, and rows carries the
    merged case+label dicts (for cost-based metrics)."""
    cases = {c["event_id"]: c for c in _read_jsonl(cases_path) if c.get("event_id")}
    resolved = LabelStore(labels_path).resolve()
    X, y, rows = [], [], []
    g_node, g_fault, g_time = [], [], []
    for eid, lbl in resolved.items():
        case = cases.get(eid)
        if case is None:
            continue
        feats = case.get("features", {})
        X.append([float(feats.get(f, 0.0)) for f in CASE_FEATURES])
        y.append(lbl["true_class"])
        rows.append({**case, "_label": lbl})
        g_node.append(case.get("node", "unknown"))
        g_fault.append(lbl.get("fault_type", "none"))
        g_time.append(str(case.get("window_start", ""))[:13])  # hour bucket
    groups = {"node": g_node, "fault_type": g_fault, "time": g_time}
    return np.asarray(X, dtype=np.float64), y, groups, rows


def build_models(max_depth: int = 4, seed: int = 42) -> Dict[str, object]:
    """Instantiate the candidate models. XGBoost added only if importable."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    models: Dict[str, object] = {
        "triage_tree": DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=5,
            class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=max_depth + 4, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
    }
    try:  # optional
        from xgboost import XGBClassifier  # noqa: F401
        models["xgboost"] = _make_xgb(seed)
    except Exception:  # noqa: BLE001 - xgboost optional / not installed
        pass
    return models


def _make_xgb(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        eval_metric="mlogloss", random_state=seed, n_jobs=-1)


def _sample_weight(y: List[str]) -> np.ndarray:
    """Inverse-frequency weights (for XGBoost, which lacks class_weight)."""
    counts = {c: y.count(c) for c in set(y)}
    n, k = len(y), len(counts)
    return np.asarray([n / (k * counts[c]) for c in y], dtype=np.float64)


def _fit(name: str, model, X: np.ndarray, y: List[str]):
    # XGBoost needs integer labels + sample_weight.
    if name == "xgboost":
        idx = {c: i for i, c in enumerate(CLASSES)}
        yi = np.asarray([idx[c] for c in y])
        model.fit(X, yi, sample_weight=_sample_weight(y))
        model._ebpf_classes = list(CLASSES)  # type: ignore[attr-defined]
    else:
        model.fit(X, y)
    return model


def model_classes(name: str, model) -> List[str]:
    if name == "xgboost":
        return list(getattr(model, "_ebpf_classes", CLASSES))
    return [str(c) for c in model.classes_]


def save_bundle(path: Path, name: str, model) -> None:
    joblib.dump({
        "model": model,
        "model_name": name,
        "features": CASE_FEATURES,
        "classes": model_classes(name, model),
        "cost_matrix": COST_MATRIX,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train cost-sensitive triage models.")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="output models/ directory")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    X, y, _groups, _rows = load_dataset(Path(args.cases), Path(args.labels))
    if len(X) < 10:
        print(f"train: too few labeled events ({len(X)}); need >=10")
        return 1
    classes_present = sorted(set(y), key=CLASSES.index)
    print(f"train: {len(X)} events, class mix="
          f"{ {c: y.count(c) for c in classes_present} }")
    if len(classes_present) < 2:
        print("train: only one class present; cannot train a classifier")
        return 1

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"n_events": len(X), "features": CASE_FEATURES,
               "class_mix": {c: y.count(c) for c in classes_present},
               "cost_matrix": COST_MATRIX, "models": {}}
    for name, model in build_models(args.max_depth, args.seed).items():
        _fit(name, model, X, y)
        path = outdir / f"{name}.joblib"
        save_bundle(path, name, model)
        train_acc = float(np.mean([p == t for p, t in zip(_predict(name, model, X), y)]))
        summary["models"][name] = {"path": str(path), "train_accuracy": round(train_acc, 4),
                                   "classes": model_classes(name, model)}
        print(f"train: {name:14s} train_acc={train_acc:.3f} -> {path}")

    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"train: wrote {outdir/'metrics.json'}")
    return 0


def _predict(name: str, model, X: np.ndarray) -> List[str]:
    classes = model_classes(name, model)
    yi = model.predict(X)
    if name == "xgboost":
        return [classes[int(i)] for i in yi]
    return [str(c) for c in yi]


if __name__ == "__main__":
    raise SystemExit(main())
