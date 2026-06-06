#!/usr/bin/env python3
"""
policy_filter.py
================
Runtime policy hook (Policy v0). Post noise-filter, pre-alert.

For every anomaly that *survived* the noise filter (``is_anomaly and not
suppressed``), the tree estimates ``p = P(fault)`` and the expected-cost rule
(policy/reward) picks an escalation action: ``wait`` / ``triage`` / ``alert``.

It never re-suppresses: rows the noise filter already silenced are tagged
``policy_action="suppress"`` for free and skipped. Disabled rows / non-anomalies
get ``policy_action=None``. Fails safe to ``alert`` only on real model error, and
no-ops entirely when the artifact is missing — scoring never breaks.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from policy.schema import STATE_COLS, FEATS, build_state, episode_id
from policy import reward as rw


class PolicyFilter:
    def __init__(self, model_path: str, baseline_path: str = "",
                 enabled: bool = True) -> None:
        self.model = None
        self.state_cols: List[str] = STATE_COLS
        self.feats: List[str] = FEATS
        self.baseline: Dict[str, Dict[str, float]] = {}
        self.baseline_source = "bundle"
        self.error: str | None = None
        self.source: str = model_path
        self.enabled = False
        if not enabled:
            return
        try:
            import joblib
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.state_cols = bundle.get("state_cols", STATE_COLS)
            self.feats = bundle.get("feats", FEATS)
            self.baseline = bundle.get("baseline", {})
            # Prefer a live (shared) baseline so deviations match the noise filter.
            if baseline_path and os.path.exists(baseline_path):
                try:
                    self.baseline = json.load(open(baseline_path))
                    self.baseline_source = baseline_path
                except Exception:  # noqa: BLE001 - fall back to bundled baseline
                    pass
            self.enabled = True
        except Exception as exc:  # noqa: BLE001 - surface via /health, never crash
            self.error = str(exc)
            self.enabled = False

    def p_fault(self, container: str, feats: Dict, anomaly_score: float,
                suppress_proba: float) -> float:
        if not self.enabled:
            return 0.0
        try:
            import numpy as np
            x = np.array([build_state(self.baseline, container or "", feats,
                                      anomaly_score, suppress_proba)])
            return float(self.model.predict_proba(x)[0, 1])
        except Exception:  # noqa: BLE001 - on error escalate (alert), see decide()
            return -1.0

    def decide(self, container: str, feats: Dict, anomaly_score: float,
               suppress_proba: float) -> tuple[str, float]:
        """Return (action, p_fault) for one effective anomaly."""
        p = self.p_fault(container, feats, anomaly_score, suppress_proba)
        if p < 0:                      # model error -> fail safe, escalate
            return "alert", 1.0
        return rw.best_action(p)[0], p

    def annotate(self, inputs: List[dict], results: List[dict]) -> Dict[str, int]:
        """Add ``policy_action`` + ``policy_p_fault`` (+ ``episode_id``) in place.

        Only effective anomalies are decided. Suppressed anomalies -> "suppress";
        non-anomalies -> None. Returns a per-action count for metrics.
        """
        counts: Dict[str, int] = {"suppress": 0, "wait": 0, "triage": 0, "alert": 0}
        for src, res in zip(inputs, results):
            action, p = None, 0.0
            if res.get("is_anomaly"):
                if res.get("suppressed"):
                    action = "suppress"
                elif self.enabled:
                    action, p = self.decide(
                        src.get("container", ""), src,
                        float(res.get("anomaly_score") or 0.0),
                        float(res.get("suppress_proba") or 0.0))
            res["policy_action"] = action
            res["policy_p_fault"] = round(p, 4)
            res["episode_id"] = episode_id(src.get("container", ""), src.get("timestamp", ""))
            if action in counts:
                counts[action] += 1
        return counts


def from_env() -> "PolicyFilter":
    """Build a PolicyFilter from env / defaults. A model promoted into
    POLICY_FILTER_LIVE_DIR (hostPath) wins over the baked artifact -> hot-swap via
    /reload. Disabled gracefully when the artifact is absent or
    POLICY_FILTER_ENABLED=false."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.environ.get("POLICY_FILTER_MODEL", os.path.join(repo, "models", "policy_filter.pkl"))
    # share the noise filter's live baseline so the deviation reference matches
    baseline_path = os.environ.get("POLICY_FILTER_BASELINE", "")
    if not baseline_path:
        live_nf = os.environ.get("NOISE_FILTER_LIVE_DIR", "")
        cand = [os.path.join(live_nf, "noise_baseline.json")] if live_nf else []
        cand.append(os.path.join(repo, "models", "noise_baseline.json"))
        baseline_path = next((c for c in cand if os.path.exists(c)), "")
    live_dir = os.environ.get("POLICY_FILTER_LIVE_DIR", "")
    if live_dir:
        lm = os.path.join(live_dir, "policy_filter.pkl")
        if os.path.exists(lm):
            model_path = lm
    enabled = os.environ.get("POLICY_FILTER_ENABLED", "true").lower() != "false"
    enabled = enabled and os.path.exists(model_path)
    pf = PolicyFilter(model_path, baseline_path, enabled)
    pf.source = model_path
    return pf
