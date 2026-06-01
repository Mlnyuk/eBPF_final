"""
model_utils.py
==============
Shared helpers for the Isolation Forest anomaly detector: config loading,
feature-matrix construction, model (de)serialization, and anomaly-score
normalization.

Score convention (used everywhere downstream):
    anomaly_score in [0, 1], where HIGHER = MORE ANOMALOUS.

scikit-learn's IsolationForest is the opposite (higher decision_function =
more normal), so we negate and then min-max calibrate using statistics
captured at training time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def get_feature_order(cfg: Optional[dict] = None) -> List[str]:
    """Single source of truth for feature column order at train+detect time."""
    cfg = cfg if cfg is not None else load_config()
    feats = cfg.get("features")
    if feats:
        return list(feats)
    # fall back to collector schema
    from collector.feature_aggregator import FEATURE_COLUMNS  # lazy import
    return list(FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Everything needed to score new feature vectors consistently."""
    model: object                      # fitted sklearn IsolationForest
    feature_order: List[str]
    # calibration stats on the RAW anomaly score (= -score_samples) over the
    # training set, used to normalize into [0, 1].
    raw_score_min: float
    raw_score_max: float
    contamination: float
    # normalized-score threshold above which a sample is flagged anomalous.
    score_threshold: float
    # Hybrid side-rule: also flag if ANY feature deviates more than this many
    # std-devs from the training mean. IsolationForest under-weights single-axis
    # anomalies (e.g. cpu-stress: cpu_utilization 210σ but scored normal because
    # 13/14 features look idle). The z-tail rule covers that blind spot.
    sigma_threshold: float = 10.0
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | os.PathLike) -> "ModelBundle":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"model not found at {path}. Run detector/train.py first."
            )
        bundle = joblib.load(path)
        # Force single-threaded inference. Training may set n_jobs=-1, but at
        # serving time per-request joblib worker spawning balloons memory under
        # concurrent load (OOM). Sequential scoring is plenty fast per window.
        try:
            bundle.model.n_jobs = 1
        except Exception:
            pass
        return bundle


# --------------------------------------------------------------------------
# Feature matrix helpers
# --------------------------------------------------------------------------

def frame_to_matrix(df: pd.DataFrame, feature_order: Sequence[str]) -> np.ndarray:
    """Extract an (n_samples, n_features) float matrix in the canonical order.
    Missing columns are filled with 0.0 (defensive against schema drift)."""
    cols = []
    for c in feature_order:
        if c in df.columns:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0))
        else:
            cols.append(pd.Series(np.zeros(len(df)), index=df.index, name=c))
    return np.column_stack(cols).astype(np.float64)


def dict_to_vector(d: dict, feature_order: Sequence[str]) -> np.ndarray:
    """Build a single (1, n_features) row from a feature dict."""
    return np.array([[float(d.get(c, 0.0)) for c in feature_order]], dtype=np.float64)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def raw_anomaly_score(model, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous. sklearn score_samples: higher = more normal,
    so we negate."""
    return -model.score_samples(X)


def normalize_score(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Min-max scale raw scores into [0, 1] using training-time bounds.
    Values outside [lo, hi] are clipped. higher = more anomalous preserved."""
    span = hi - lo
    if span <= 1e-12:
        return np.clip(raw - lo, 0.0, 1.0)
    return np.clip((raw - lo) / span, 0.0, 1.0)


def top_feature_contributions(
    row_values: np.ndarray,
    feature_order: Sequence[str],
    train_means: Optional[np.ndarray] = None,
    train_stds: Optional[np.ndarray] = None,
    k: int = 3,
) -> List[dict]:
    """
    Heuristic explanation: rank features by how far the sample deviates from the
    training mean (in std units). IsolationForest has no native per-feature
    attribution, so this z-score deviation is a pragmatic proxy.
    Returns up to k entries: {feature, value, z}.
    """
    vals = np.asarray(row_values, dtype=np.float64).ravel()
    if train_means is None or train_stds is None:
        # no calibration available -> rank by raw magnitude
        order = np.argsort(-np.abs(vals))[:k]
        return [{"feature": feature_order[i], "value": float(vals[i]), "z": None}
                for i in order]
    stds = np.where(train_stds < 1e-9, 1e-9, train_stds)
    z = (vals - train_means) / stds
    order = np.argsort(-np.abs(z))[:k]
    return [
        {"feature": feature_order[i], "value": float(vals[i]), "z": round(float(z[i]), 3)}
        for i in order
    ]
