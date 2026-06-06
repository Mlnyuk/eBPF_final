#!/usr/bin/env python3
"""
infer.py
========
Lightweight CPU inference for the cost-sensitive triage models.

Loads a joblib bundle saved by train_cost_sensitive.py and turns a feature dict
into a full decision: class probabilities -> expected cost per action ->
recommended action (argmin cost). No GPU, no LLM — safe for the production
detector's hot path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.triage.cost_matrix import (
    CLASSES, GUARDRAILS, GuardrailConfig, decide_action, get_cost_matrix,
)

# Identity fields used by the unknown-workload guardrail.
_IDENTITY_KEYS = ("cluster", "node", "namespace", "pod", "container")


class TriageModel:
    def __init__(self, bundle: Dict) -> None:
        self.model = bundle["model"]
        self.model_name = bundle.get("model_name", "triage")
        self.features: List[str] = list(bundle["features"])
        self.classes: List[str] = list(bundle["classes"])
        self.cost_matrix = bundle.get("cost_matrix")

    @classmethod
    def load(cls, path: str | Path) -> "TriageModel":
        import joblib
        return cls(joblib.load(path))

    def _row(self, features: Dict[str, float]) -> np.ndarray:
        return np.asarray([[float(features.get(f, 0.0)) for f in self.features]],
                          dtype=np.float64)

    def proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Class-probability dict aligned to the canonical CLASSES order
        (classes absent from training fill 0.0)."""
        X = self._row(features)
        if self.model_name == "xgboost":
            p = self.model.predict_proba(X)[0]
            raw = {self.classes[i]: float(p[i]) for i in range(len(self.classes))}
        else:
            p = self.model.predict_proba(X)[0]
            raw = {str(c): float(pi) for c, pi in zip(self.model.classes_, p)}
        return {c: raw.get(c, 0.0) for c in CLASSES}

    def decide(self, features: Dict[str, float], event_id: str = "",
               identity: Optional[Dict[str, str]] = None,
               cost_mode=None,
               guardrails: GuardrailConfig = GUARDRAILS) -> Dict:
        """Full triage decision for one feature vector.

        Cost matrix: `cost_mode` (mode name / matrix) wins, else the recall-first
        production default — NOT the matrix baked at train time (costs are
        decoupled from the model, so they can be retuned without retraining).
        Guardrails then floor the action so high-severity / unknown / low-confidence
        cases can never be silently suppressed."""
        prob = self.proba(features)
        action, costs, guard = decide_action(
            prob, features, identity=identity,
            cost_matrix=cost_mode, guardrails=guardrails)
        return {
            "event_id": event_id,
            "anomaly_score": float(features.get("anomaly_score", 0.0)),
            "true_class_prob": {k: round(v, 4) for k, v in prob.items()},
            "expected_cost": {k: round(v, 4) for k, v in costs.items()},
            "recommended_action": action,
            "guardrail": guard,
            "model": self.model_name,
        }


def load_first_available(models_dir: str | Path,
                         prefer: Optional[List[str]] = None) -> Optional[TriageModel]:
    """Load the first existing model bundle from a models directory.

    Default preference: RandomForest (production inference model), then the
    DecisionTree (kept as an explainable baseline), then xgboost."""
    prefer = prefer or ["random_forest", "triage_tree", "xgboost"]
    d = Path(models_dir)
    for name in prefer:
        p = d / f"{name}.joblib"
        if p.exists():
            return TriageModel.load(p)
    return None


# --------------------------------------------------------------------------
# Detector runtime integration (hot-swappable, CPU-only)
# --------------------------------------------------------------------------

class TriageRuntime:
    """Production-side wrapper used by the detector. A promoted model in the live
    dir wins over the baked one (mirrors the noise/policy hot-swap pattern)."""

    def __init__(self, enabled: bool, model: Optional[TriageModel],
                 source: Optional[str], error: Optional[str],
                 cost_mode: str = "recall_first") -> None:
        self.enabled = enabled
        self.model = model
        self.source = source
        self.error = error
        self.cost_mode = cost_mode

    @property
    def ready(self) -> bool:
        return self.enabled and self.model is not None

    def decide(self, raw_features: Dict[str, float], anomaly_score: float,
               event_id: str = "") -> Optional[Dict]:
        """Map raw 14-col features (+ anomaly_score) to a triage decision under
        the recall-first cost matrix + safety guardrails. The raw row's identity
        (namespace/pod/...) drives the unknown-workload guardrail.
        Returns None when triage is disabled / unloaded."""
        if not self.ready:
            return None
        from src.triage.case_bundle import features_from_row
        feats = features_from_row({**raw_features, "anomaly_score": anomaly_score})
        identity = {k: raw_features.get(k, "") for k in _IDENTITY_KEYS}
        return self.model.decide(feats, event_id=event_id,
                                 identity=identity, cost_mode=self.cost_mode)


def from_env() -> "TriageRuntime":
    """Build a TriageRuntime from env:
      TRIAGE_ENABLED      (default true)
      TRIAGE_LIVE_DIR     promoted model dir (hostPath); wins if present
      TRIAGE_MODEL_DIR    baked model dir (default 'models')
      TRIAGE_COST_MODE    cost matrix mode (default 'recall_first')
    """
    import os
    enabled = os.environ.get("TRIAGE_ENABLED", "true").lower() == "true"
    cost_mode = os.environ.get("TRIAGE_COST_MODE", "recall_first")
    if not enabled:
        return TriageRuntime(False, None, None, None, cost_mode)
    live = os.environ.get("TRIAGE_LIVE_DIR", "")
    baked = os.environ.get("TRIAGE_MODEL_DIR", "models")
    for src in (live, baked):
        if not src:
            continue
        try:
            m = load_first_available(src)
        except Exception as exc:  # noqa: BLE001 - surface via .error
            return TriageRuntime(True, None, src, str(exc), cost_mode)
        if m is not None:
            return TriageRuntime(True, m, src, None, cost_mode)
    return TriageRuntime(True, None, baked, "no triage model found", cost_mode)
