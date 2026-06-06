#!/usr/bin/env python3
"""
api.py
======
FastAPI detector service.

Endpoints
---------
    GET  /health        -> liveness + model status
    POST /detect        -> score a single feature vector
    POST /detect/batch  -> score many feature vectors
    GET  /metrics       -> Prometheus exposition (anomaly score + counts)

Run
---
    uvicorn detector.api:app --host 0.0.0.0 --port 8080
    # or
    python detector/api.py

The model path is taken from configs/config.yaml (model.path) or the
MODEL_PATH env var. The model is loaded once at startup.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from detector.model_utils import ModelBundle, load_config, get_feature_order  # noqa: E402
from detector import detect as detect_mod  # noqa: E402
from detector.archive import FeatureArchive  # noqa: E402
from detector.noise_filter import from_env as _noise_from_env  # noqa: E402
from policy.policy_filter import from_env as _policy_from_env  # noqa: E402
from src.triage.infer import from_env as _triage_from_env  # noqa: E402

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

_CFG = load_config()
_MODEL_PATH = os.environ.get("MODEL_PATH") or _CFG.get("model", {}).get(
    "path", "models/isolation_forest.pkl")
# A retrained model promoted by the retrain pipeline lands here (hostPath); when
# present it wins over the baked image model, enabling hot-swap without a rebuild.
_LIVE_MODEL_PATH = os.environ.get("LIVE_MODEL_PATH", "")
_FEATURE_ORDER = get_feature_order(_CFG)

_bundle: Optional[ModelBundle] = None
_bundle_err: Optional[str] = None
_bundle_src: Optional[str] = None


def _resolve_model_path() -> str:
    """Prefer the live (retrained) model if it exists, else the baked one."""
    if _LIVE_MODEL_PATH and os.path.exists(_LIVE_MODEL_PATH):
        return _LIVE_MODEL_PATH
    return _MODEL_PATH


def _load_model() -> None:
    global _bundle, _bundle_err, _bundle_src
    path = _resolve_model_path()
    try:
        _bundle = ModelBundle.load(path)
        _bundle_err = None
        _bundle_src = path
    except Exception as exc:  # noqa: BLE001 - surface to /health
        _bundle = None
        _bundle_err = str(exc)
        _bundle_src = path


# --------------------------------------------------------------------------
# Prometheus metric state (thread-safe)
# --------------------------------------------------------------------------

class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_scored = 0
        self.total_anomalies = 0
        self.total_suppressed = 0
        self.last_score = 0.0
        self.max_score = 0.0
        # policy-layer escalation decisions (suppress is owned by the noise filter)
        self.policy_actions = {"suppress": 0, "wait": 0, "triage": 0, "alert": 0}
        # cost-sensitive triage decisions (4-class severity -> expected-cost action)
        self.triage_actions = {"suppress": 0, "wait": 0, "triage": 0, "alert": 0}

    def observe(self, scores: List[float], anomalies: int, suppressed: int = 0,
                policy_counts: Optional[Dict[str, int]] = None) -> None:
        with self._lock:
            self.total_scored += len(scores)
            self.total_anomalies += anomalies
            self.total_suppressed += suppressed
            if policy_counts:
                for a, n in policy_counts.items():
                    if a in self.policy_actions:
                        self.policy_actions[a] += n
            if scores:
                self.last_score = scores[-1]
                self.max_score = max(self.max_score, max(scores))

    def observe_triage(self, action: str) -> None:
        with self._lock:
            if action in self.triage_actions:
                self.triage_actions[action] += 1

    def render(self) -> str:
        with self._lock:
            effective = self.total_anomalies - self.total_suppressed
            lines = [
                "# HELP ebpf_anomaly_scored_total Total feature vectors scored.",
                "# TYPE ebpf_anomaly_scored_total counter",
                f"ebpf_anomaly_scored_total {self.total_scored}",
                "# HELP ebpf_anomaly_detected_total Total vectors flagged anomalous (raw, pre-suppression).",
                "# TYPE ebpf_anomaly_detected_total counter",
                f"ebpf_anomaly_detected_total {self.total_anomalies}",
                "# HELP ebpf_anomaly_suppressed_total Flagged anomalies suppressed as benign daemon noise.",
                "# TYPE ebpf_anomaly_suppressed_total counter",
                f"ebpf_anomaly_suppressed_total {self.total_suppressed}",
                "# HELP ebpf_anomaly_effective_total Anomalies after noise suppression (detected - suppressed).",
                "# TYPE ebpf_anomaly_effective_total counter",
                f"ebpf_anomaly_effective_total {effective}",
                "# HELP ebpf_anomaly_score_last Most recent anomaly score (0..1).",
                "# TYPE ebpf_anomaly_score_last gauge",
                f"ebpf_anomaly_score_last {self.last_score}",
                "# HELP ebpf_anomaly_score_max Max anomaly score observed since start.",
                "# TYPE ebpf_anomaly_score_max gauge",
                f"ebpf_anomaly_score_max {self.max_score}",
                "# HELP ebpf_policy_action_total Policy-layer decisions on flagged anomalies by action.",
                "# TYPE ebpf_policy_action_total counter",
            ]
            for a in ("suppress", "wait", "triage", "alert"):
                lines.append(f'ebpf_policy_action_total{{action="{a}"}} {self.policy_actions[a]}')
            lines += [
                "# HELP ebpf_triage_action_total Cost-sensitive triage decisions by action.",
                "# TYPE ebpf_triage_action_total counter",
            ]
            for a in ("suppress", "wait", "triage", "alert"):
                lines.append(f'ebpf_triage_action_total{{action="{a}"}} {self.triage_actions[a]}')
            return "\n".join(lines) + "\n"


_metrics = _Metrics()

# --------------------------------------------------------------------------
# Durable feature archive (optional; enabled when ARCHIVE_DIR is set)
# --------------------------------------------------------------------------

_META_COLS = ["timestamp", "node", "namespace", "pod", "container"]
_ARCHIVE_COLS = (["ingest_ts"] + _META_COLS + list(_FEATURE_ORDER)
                 + ["anomaly_score", "is_anomaly", "trigger", "suppressed",
                    "policy_action", "policy_p_fault", "episode_id"])
_archive = FeatureArchive(os.environ.get("ARCHIVE_DIR", ""), _ARCHIVE_COLS)

# Distilled noise-suppression gate (post-IF). Disabled gracefully if artifacts
# are missing or NOISE_FILTER_ENABLED=false. See detector/noise_filter.py.
_noise = _noise_from_env()

# Cost-sensitive policy layer (post noise filter). Chooses an escalation action
# (wait/triage/alert) for surviving anomalies. See policy/policy_filter.py.
_policy = _policy_from_env()


def _load_noise() -> None:
    """(Re)build the noise filter — live dir wins over baked. Called at startup
    and by /reload after the distill pipeline promotes new artifacts."""
    global _noise
    _noise = _noise_from_env()


def _load_policy() -> None:
    """(Re)build the policy filter — live dir wins over baked. Called at startup
    and by /reload after the policy pipeline promotes a new artifact."""
    global _policy
    _policy = _policy_from_env()


# Cost-sensitive triage model (4-class severity -> expected-cost action). CPU-only
# inference; the LLM adjudicator is NEVER on this path. See src/triage/.
_triage = _triage_from_env()


def _load_triage() -> None:
    """(Re)load the triage model — live dir wins over baked. Called at startup and
    by /reload after the triage train/eval pipeline promotes a new model."""
    global _triage
    _triage = _triage_from_env()


def _archive_rows(inputs: List[dict], results: List[dict]) -> None:
    """Merge input feature dicts with their score columns and append to archive.
    inputs[i] holds meta+features; results[i] holds anomaly_score/is_anomaly/
    trigger for the same row (index-aligned)."""
    if not _archive.enabled:
        return
    ingest = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records = []
    for src, res in zip(inputs, results):
        records.append({
            "ingest_ts": ingest, **src,
            "anomaly_score": res.get("anomaly_score"),
            "is_anomaly": res.get("is_anomaly"),
            "trigger": res.get("trigger"),
            "suppressed": res.get("suppressed", False),
            "policy_action": res.get("policy_action"),
            "policy_p_fault": res.get("policy_p_fault"),
            "episode_id": res.get("episode_id"),
        })
    _archive.append(records)

# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------

class FeatureVector(BaseModel):
    # identity metadata (optional)
    timestamp: Optional[str] = None
    node: Optional[str] = None
    namespace: Optional[str] = None
    pod: Optional[str] = None
    container: Optional[str] = None
    # feature values; extra keys allowed so schema can evolve
    features: Dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    def split(self) -> tuple[dict, dict]:
        """Return (meta, feature_dict). Feature values may be supplied either
        nested under `features` or as flat top-level keys."""
        meta = {
            "timestamp": self.timestamp, "node": self.node,
            "namespace": self.namespace, "pod": self.pod,
            "container": self.container,
        }
        feats = dict(self.features)
        # also accept flat feature keys from model_extra
        extra = self.__pydantic_extra__ or {}
        for k, v in extra.items():
            if k in _FEATURE_ORDER and k not in feats:
                feats[k] = float(v)
        return meta, feats


class BatchRequest(BaseModel):
    items: List[FeatureVector]


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="eBPF_final anomaly detector", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    _load_model()
    _load_noise()
    _load_policy()
    _load_triage()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if _bundle is not None else "degraded",
        "model_loaded": _bundle is not None,
        "model_path": _MODEL_PATH,
        "error": _bundle_err,
        "feature_count": len(_FEATURE_ORDER),
        "threshold": getattr(_bundle, "score_threshold", None),
        "model_source": _bundle_src,
        "trained_at": getattr(_bundle, "metadata", {}).get("trained_at") if _bundle else None,
        "noise_filter": {
            "enabled": _noise.enabled,
            "source": _noise.source,
            "threshold": _noise.threshold,
            "baselined_containers": len(_noise.baseline),
            "error": _noise.error,
        },
        "policy_filter": {
            "enabled": _policy.enabled,
            "source": _policy.source,
            "baseline_source": _policy.baseline_source,
            "baselined_containers": len(_policy.baseline),
            "error": _policy.error,
        },
        "triage": {
            "enabled": _triage.enabled,
            "ready": _triage.ready,
            "source": _triage.source,
            "model": _triage.model.model_name if _triage.model else None,
            "error": _triage.error,
        },
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.post("/reload")
def reload_model() -> dict:
    """Re-read the model AND noise filter from disk (live paths win). Called by
    the retrain / distill pipelines after promoting new artifacts."""
    _load_model()
    _load_noise()
    _load_policy()
    _load_triage()
    return {
        "reloaded": _bundle is not None,
        "model_source": _bundle_src,
        "error": _bundle_err,
        "trained_at": getattr(_bundle, "metadata", {}).get("trained_at") if _bundle else None,
        "threshold": getattr(_bundle, "score_threshold", None),
        "noise_filter": {
            "enabled": _noise.enabled,
            "source": _noise.source,
            "threshold": _noise.threshold,
            "baselined_containers": len(_noise.baseline),
            "error": _noise.error,
        },
        "policy_filter": {
            "enabled": _policy.enabled,
            "source": _policy.source,
            "baseline_source": _policy.baseline_source,
            "baselined_containers": len(_policy.baseline),
            "error": _policy.error,
        },
        "triage": {
            "enabled": _triage.enabled,
            "ready": _triage.ready,
            "source": _triage.source,
            "model": _triage.model.model_name if _triage.model else None,
            "error": _triage.error,
        },
    }


def _require_model() -> ModelBundle:
    if _bundle is None:
        raise HTTPException(status_code=503,
                            detail=f"model not loaded: {_bundle_err}")
    return _bundle


@app.post("/detect")
def detect(fv: FeatureVector) -> dict:
    bundle = _require_model()
    meta, feats = fv.split()
    result = detect_mod.detect_one(bundle, feats, meta=meta)
    src = {**(meta or {}), **feats}
    n_supp = _noise.annotate([src], [result])
    result["effective_anomaly"] = bool(result["is_anomaly"]) and not result.get("suppressed")
    pol_counts = _policy.annotate([src], [result])
    _metrics.observe([float(result["anomaly_score"])],
                     1 if result["is_anomaly"] else 0, n_supp, pol_counts)
    _archive_rows([src], [result])
    return result


@app.post("/triage")
def triage(fv: FeatureVector) -> dict:
    """Cost-sensitive triage for one feature vector: score it, then map class
    probabilities to the expected-cost-minimising action. CPU-only; no LLM.

    Returns 503 if no triage model is loaded (train via src.triage pipeline)."""
    bundle = _require_model()
    if not _triage.ready:
        raise HTTPException(status_code=503,
                            detail=f"triage model not loaded: {_triage.error}")
    meta, feats = fv.split()
    result = detect_mod.detect_one(bundle, feats, meta=meta)
    eid = result.get("event_id") or (meta or {}).get("pod") or ""
    decision = _triage.decide(feats, float(result["anomaly_score"]), event_id=str(eid))
    _metrics.observe_triage(decision["recommended_action"])
    return decision


@app.post("/detect/batch")
def detect_batch(req: BatchRequest) -> dict:
    import pandas as pd
    bundle = _require_model()
    if not req.items:
        return {"count": 0, "anomalies": 0, "results": []}
    # Build one DataFrame and score all rows in a single score_samples call.
    rows = []
    for fv in req.items:
        meta, feats = fv.split()
        rows.append({**(meta or {}), **feats})
    df = pd.DataFrame(rows)
    result_df = detect_mod.score_frame(bundle, df)
    results = result_df.to_dict(orient="records")
    scores = [float(r["anomaly_score"]) for r in results]
    n_anom = sum(1 for r in results if r["is_anomaly"])
    n_supp = _noise.annotate(rows, results)
    for r in results:
        r["effective_anomaly"] = bool(r["is_anomaly"]) and not r.get("suppressed")
    pol_counts = _policy.annotate(rows, results)
    _metrics.observe(scores, n_anom, n_supp, pol_counts)
    # Optional cost-sensitive triage action per effective anomaly (CPU, no LLM).
    triage_counts: Dict[str, int] = {}
    if _triage.ready:
        for src, r in zip(rows, results):
            if not r.get("effective_anomaly"):
                continue
            d = _triage.decide(src, float(r["anomaly_score"]),
                               event_id=str(r.get("event_id") or src.get("pod") or ""))
            r["triage"] = {"recommended_action": d["recommended_action"],
                           "true_class_prob": d["true_class_prob"],
                           "model": d["model"]}
            act = d["recommended_action"]
            triage_counts[act] = triage_counts.get(act, 0) + 1
            _metrics.observe_triage(act)
    _archive_rows(rows, results)
    return {"count": len(results), "anomalies": n_anom,
            "effective_anomalies": n_anom - n_supp, "suppressed": n_supp,
            "policy_actions": pol_counts,
            "triage_actions": triage_counts,
            "results": results}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return _metrics.render()


def _main() -> None:
    import uvicorn
    dcfg = _CFG.get("detector", {})
    uvicorn.run(app, host=dcfg.get("host", "0.0.0.0"), port=int(dcfg.get("port", 8080)))


if __name__ == "__main__":
    _main()
