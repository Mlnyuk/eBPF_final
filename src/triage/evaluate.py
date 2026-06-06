#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate triage models with COST-AWARE metrics — accuracy alone is misleading
when classes are imbalanced and errors have very different costs.

Computes, per model (plus the score-threshold baseline):
  accuracy, macro-F1, mean expected cost, high-severity false negatives,
  alert precision, alert volume, class confusion matrix, and node / fault-type /
  time holdout expected-cost (leave-one-group-out, fresh cost-sensitive tree).

Writes a Markdown comparison report (reports/triage_eval.md).

Usage:
    python -m src.triage.evaluate \
      --cases data/cases/cases.jsonl \
      --labels data/labels/labels.jsonl \
      --models models/ \
      --out reports/triage_eval.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.triage.cost_matrix import (
    CLASSES, ACTIONS, GUARDRAILS, expected_cost, recommended_action,
    action_for_pure_class, decide_action, get_cost_matrix,
)
from src.triage.case_bundle import CASE_FEATURES
from src.triage.train_cost_sensitive import load_dataset
from src.triage.infer import TriageModel

# Actions that count as "escalation" (a 'high' case that doesn't escalate = FN).
ESCALATED = {"triage", "alert"}
_IDENTITY_KEYS = ("cluster", "node", "namespace", "pod", "container")


def _threshold_class(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.8:
        return "medium"
    if score >= 0.7:
        return "low"
    return "normal"


def _threshold_predict(rows: List[Dict], cost_matrix) -> Tuple[List[str], List[str], int]:
    """Baseline: anomaly_score band -> class -> cost-min action. No guardrails
    (kept as the naive contrast). Returns (pred_cls, actions, guardrail_saves=0)."""
    pred_cls, actions = [], []
    for r in rows:
        s = float(r.get("features", {}).get("anomaly_score", 0.0))
        c = _threshold_class(s)
        pred_cls.append(c)
        actions.append(action_for_pure_class(c, cost_matrix))
    return pred_cls, actions, 0


def _model_predict(model: TriageModel, rows: List[Dict], cost_matrix,
                   guardrails=GUARDRAILS) -> Tuple[List[str], List[str], int]:
    """Predict class + guardrailed action under the given cost matrix.
    Returns (pred_cls, actions, guardrail_saves) where guardrail_saves counts
    cases whose action a guardrail raised above the cost-minimising choice."""
    pred_cls, actions, saves = [], [], 0
    for r in rows:
        feats = r.get("features", {})
        identity = {k: r.get(k, "") for k in _IDENTITY_KEYS}
        prob = model.proba(feats)
        pred_cls.append(max(CLASSES, key=lambda c: prob[c]))
        action, _costs, guard = decide_action(
            prob, feats, identity=identity, cost_matrix=cost_matrix,
            guardrails=guardrails)
        if guard:
            saves += 1
        actions.append(action)
    return pred_cls, actions, saves


def _metrics(y_true: List[str], pred_cls: List[str], actions: List[str],
             cost_matrix, guardrail_saves: int = 0) -> Dict:
    from sklearn.metrics import f1_score, confusion_matrix
    n = len(y_true)
    acc = float(np.mean([p == t for p, t in zip(pred_cls, y_true)])) if n else 0.0
    macro_f1 = float(f1_score(y_true, pred_cls, labels=CLASSES,
                              average="macro", zero_division=0)) if n else 0.0
    exp_cost = float(np.mean([cost_matrix[a][t] for a, t in zip(actions, y_true)])) if n else 0.0
    high_fn = int(sum(1 for a, t in zip(actions, y_true)
                      if t == "high" and a not in ESCALATED))
    # Recall on high (the recall-first objective): fraction of true-high escalated.
    n_high = sum(1 for t in y_true if t == "high")
    high_recall = float((n_high - high_fn) / n_high) if n_high else float("nan")
    alerts = [t for a, t in zip(actions, y_true) if a == "alert"]
    alert_vol = len(alerts)
    alert_prec = float(np.mean([t == "high" for t in alerts])) if alerts else float("nan")
    cm = confusion_matrix(y_true, pred_cls, labels=CLASSES).tolist()
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "expected_cost": round(exp_cost, 4), "high_fn": high_fn,
            "high_recall": (round(high_recall, 4) if high_recall == high_recall else None),
            "alert_precision": (round(alert_prec, 4) if alert_prec == alert_prec else None),
            "alert_volume": alert_vol, "guardrail_saves": guardrail_saves,
            "confusion": cm}


def _holdout_cost(X: np.ndarray, y: List[str], group: List,
                  cost_matrix) -> Optional[float]:
    """Leave-one-group-out mean expected cost with a fresh cost-sensitive tree,
    under the given cost matrix. Returns None if no group split is evaluable."""
    from sklearn.tree import DecisionTreeClassifier
    y_arr = np.asarray(y)
    uniq = sorted(set(group))
    if len(uniq) < 2:
        return None
    costs: List[float] = []
    for g in uniq:
        te = np.asarray([gg == g for gg in group])
        tr = ~te
        if tr.sum() < 5 or len(set(y_arr[tr])) < 2 or te.sum() == 0:
            continue
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3,
                                     class_weight="balanced", random_state=42)
        clf.fit(X[tr], y_arr[tr])
        proba = clf.predict_proba(X[te])
        cls = [str(c) for c in clf.classes_]
        for i, ti in enumerate(np.where(te)[0]):
            prob = {c: 0.0 for c in CLASSES}
            for j, c in enumerate(cls):
                prob[c] = float(proba[i][j])
            act = recommended_action(prob, cost_matrix)[0]
            costs.append(expected_cost(act, {y_arr[ti]: 1.0}, cost_matrix))
    return round(float(np.mean(costs)), 4) if costs else None


def evaluate(cases_path: Path, labels_path: Path, models_dir: Path,
             cost_mode: str = "recall_first", guardrails=GUARDRAILS) -> Dict:
    X, y, groups, rows = load_dataset(cases_path, labels_path)
    cm = get_cost_matrix(cost_mode)
    report: Dict = {"n_events": len(y), "cost_mode": cost_mode,
                    "guardrails_enabled": bool(guardrails.enabled),
                    "class_mix": {c: y.count(c) for c in CLASSES},
                    "models": {}, "holdout": {}}
    if not y:
        return report

    # baseline (no guardrails) + trained models (recall-first matrix + guardrails)
    preds: Dict[str, Tuple[List[str], List[str], int]] = {
        "Threshold": _threshold_predict(rows, cm)}
    for name in ("triage_tree", "random_forest", "xgboost"):
        p = models_dir / f"{name}.joblib"
        if p.exists():
            preds[name] = _model_predict(TriageModel.load(p), rows, cm, guardrails)

    for name, (pc, acts, saves) in preds.items():
        report["models"][name] = _metrics(y, pc, acts, cm, saves)

    # group holdouts (model-agnostic: fresh tree per split, under same matrix)
    for gk, gv in groups.items():
        report["holdout"][gk] = _holdout_cost(X, y, gv, cm)
    return report


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

_DISPLAY = {"Threshold": "Threshold", "triage_tree": "Cost-sensitive Tree",
            "random_forest": "RandomForest", "xgboost": "XGBoost optional"}
_ORDER = ["Threshold", "triage_tree", "random_forest", "xgboost"]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_markdown(report: Dict) -> str:
    mode = report.get("cost_mode", "default")
    guards = "on" if report.get("guardrails_enabled") else "off"
    L = ["# Triage Evaluation Report", "",
         f"Events: **{report['n_events']}**  ",
         f"Cost matrix: **{mode}**  ·  Guardrails: **{guards}**  ",
         f"Class mix: `{report.get('class_mix', {})}`", "",
         "Cost-aware comparison (lower Expected Cost / High FN is better; "
         "higher High Recall is better). Models run under the recall-first matrix "
         "+ guardrails; Threshold is the naive baseline (no guardrails).", "",
         "| Model | Accuracy | Macro F1 | Expected Cost | High FN | High Recall | "
         "Alert Vol | Guardrail Saves |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in _ORDER:
        m = report["models"].get(name)
        disp = _DISPLAY[name]
        if not m:
            L.append(f"| {disp} | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {disp} | {_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} | "
                 f"{_fmt(m['expected_cost'])} | {m['high_fn']} | "
                 f"{_fmt(m.get('high_recall'))} | {m['alert_volume']} | "
                 f"{m.get('guardrail_saves', 0)} |")
    L += ["", "## Alert precision (P(true=high | action=alert))", ""]
    for name in _ORDER:
        m = report["models"].get(name)
        if m:
            L.append(f"- **{_DISPLAY[name]}**: {_fmt(m.get('alert_precision'))}")
    L += ["", "## Holdout mean expected cost (leave-one-group-out, fresh tree)", "",
          "| Holdout | Expected Cost |", "|---|---:|"]
    for gk, gv in report.get("holdout", {}).items():
        L.append(f"| {gk} | {_fmt(gv)} |")
    L += ["", "## Class confusion matrices (rows=true, cols=pred)", "",
          f"Class order: `{CLASSES}`", ""]
    for name in _ORDER:
        m = report["models"].get(name)
        if not m:
            continue
        L.append(f"**{_DISPLAY[name]}**")
        L.append("```")
        L.append("       " + " ".join(f"{c:>7}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            L.append(f"{c:>6} " + " ".join(f"{v:>7}" for v in m["confusion"][i]))
        L.append("```")
        L.append("")
    L += ["## Notes", "",
          f"- Cost matrix `{mode}`: high-severity false negatives penalised "
          "aggressively (recall-first stance).",
          "- Expected Cost is the operational objective; accuracy is secondary.",
          "- High FN = a `high` case routed to suppress/wait (most expensive error); "
          "High Recall = fraction of true-high cases escalated.",
          "- Guardrail Saves = cases where a safety guardrail raised the action "
          "above the cost-min choice (high score / high P(high) / unknown workload / "
          "low confidence — never suppressed).",
          "- Models use RandomForest/Tree under recall-first + guardrails; the "
          "DecisionTree is retained as an explainable baseline, RandomForest is the "
          "production `/triage` model.",
          "- `—` = not available (model absent or holdout group not evaluable).", ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate triage models (cost-aware).")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True, help="markdown report path")
    ap.add_argument("--json-out", default="", help="optional metrics JSON path")
    ap.add_argument("--cost-mode", default="recall_first",
                    choices=["recall_first", "default"],
                    help="cost matrix for action selection (default: recall_first)")
    args = ap.parse_args(argv)

    report = evaluate(Path(args.cases), Path(args.labels), Path(args.models),
                      cost_mode=args.cost_mode)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(render_markdown(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    print(f"evaluate: wrote {outp} ({report['n_events']} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
