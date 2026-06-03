#!/usr/bin/env python3
"""
noise_filter.py
===============
Distilled noise-suppression gate for the detector.

The Isolation Forest flags ~1/3 of all windows as anomalous, but most are benign
system-daemon noise (containerd / kubelet / cgroup / *.service running hot — which
is normal for them). A nightly Qwen3-32B pass labels grouped anomalies
benign|suspicious; those labels train this small, interpretable DecisionTree so
the *runtime* path can suppress daemon noise WITHOUT any LLM dependency.

It is a post-IF gate: consulted only for rows the model already flagged. A flagged
row is suppressed (``suppressed=True``) only when the tree is highly confident the
anomaly is benign (predict_proba >= threshold, default 0.99 — tuned so <0.5% of
real anomalies are silenced). The raw ``is_anomaly`` is preserved for the archive;
suppression only affects the *effective* anomaly used for alerting/metrics.

Features mirror training exactly: per-feature signed log-deviation from the
container's own normal baseline (``models/noise_baseline.json``), plus the
anomaly score and a has-baseline flag.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

_EPS = 1e-6


class NoiseFilter:
    def __init__(self, model_path: str, baseline_path: str, threshold: float,
                 enabled: bool = True) -> None:
        self.threshold = threshold
        self.model = None
        self.cols: List[str] = []
        self.feats: List[str] = []
        self.baseline: Dict[str, Dict[str, float]] = {}
        self.error: str | None = None
        self.source: str = model_path
        self.enabled = False
        if not enabled:
            return
        try:
            import joblib
            bundle = joblib.load(model_path)
            self.model = bundle["model"]
            self.cols = bundle["cols"]
            self.feats = bundle["feats"]
            with open(baseline_path) as fh:
                self.baseline = json.load(fh)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001 - surface via /health, never crash
            self.error = str(exc)
            self.enabled = False

    def _vector(self, container: str, feats: Dict, anomaly_score: float) -> List[float]:
        bl = self.baseline.get(container)
        x = [float(anomaly_score), 1.0 if bl else 0.0]
        for f in self.feats:
            o = float(feats.get(f) or 0.0)
            b = (bl.get(f) if bl else 0.0) or 0.0
            x.append(math.log10((o + _EPS) / (b + _EPS)))
        return x

    def suppress_proba(self, container: str, feats: Dict, anomaly_score: float) -> float:
        """Probability that this flagged anomaly is benign noise (0..1)."""
        if not self.enabled:
            return 0.0
        try:
            import numpy as np
            x = np.array([self._vector(container or "", feats, anomaly_score)])
            return float(self.model.predict_proba(x)[0, 1])
        except Exception:  # noqa: BLE001 - on any error, never suppress
            return 0.0

    def annotate(self, inputs: List[dict], results: List[dict]) -> int:
        """Add ``suppressed`` + ``suppress_proba`` to each result in place.

        Only flagged rows (is_anomaly truthy) are evaluated. Returns the number
        of rows suppressed (flagged-but-benign). Index-aligned: inputs[i] holds
        meta+features, results[i] holds the score columns for the same row.
        """
        n_suppressed = 0
        for src, res in zip(inputs, results):
            supp, proba = False, 0.0
            if res.get("is_anomaly") and self.enabled:
                proba = self.suppress_proba(src.get("container", ""), src, float(res.get("anomaly_score") or 0.0))
                supp = proba >= self.threshold
            res["suppressed"] = bool(supp)
            res["suppress_proba"] = round(proba, 4)
            if supp:
                n_suppressed += 1
        return n_suppressed


def from_env() -> "NoiseFilter":
    """Build a NoiseFilter from env / default paths. Disabled gracefully when the
    artifacts are absent or NOISE_FILTER_ENABLED=false.

    A retrained pair promoted by the distill pipeline into NOISE_FILTER_LIVE_DIR
    (hostPath) wins over the baked artifacts, enabling hot-swap via /reload."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.environ.get("NOISE_FILTER_MODEL", os.path.join(repo, "models", "noise_filter.pkl"))
    baseline_path = os.environ.get("NOISE_FILTER_BASELINE", os.path.join(repo, "models", "noise_baseline.json"))
    live_dir = os.environ.get("NOISE_FILTER_LIVE_DIR", "")
    if live_dir:
        lm = os.path.join(live_dir, "noise_filter.pkl")
        lb = os.path.join(live_dir, "noise_baseline.json")
        if os.path.exists(lm) and os.path.exists(lb):
            model_path, baseline_path = lm, lb
    threshold = float(os.environ.get("NOISE_FILTER_THRESHOLD", "0.99"))
    enabled = os.environ.get("NOISE_FILTER_ENABLED", "true").lower() != "false"
    enabled = enabled and os.path.exists(model_path) and os.path.exists(baseline_path)
    nf = NoiseFilter(model_path, baseline_path, threshold, enabled)
    nf.source = model_path
    return nf
