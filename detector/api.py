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

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

_CFG = load_config()
_MODEL_PATH = os.environ.get("MODEL_PATH") or _CFG.get("model", {}).get(
    "path", "models/isolation_forest.pkl")
_FEATURE_ORDER = get_feature_order(_CFG)

_bundle: Optional[ModelBundle] = None
_bundle_err: Optional[str] = None


def _load_model() -> None:
    global _bundle, _bundle_err
    try:
        _bundle = ModelBundle.load(_MODEL_PATH)
        _bundle_err = None
    except Exception as exc:  # noqa: BLE001 - surface to /health
        _bundle = None
        _bundle_err = str(exc)


# --------------------------------------------------------------------------
# Prometheus metric state (thread-safe)
# --------------------------------------------------------------------------

class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_scored = 0
        self.total_anomalies = 0
        self.last_score = 0.0
        self.max_score = 0.0

    def observe(self, scores: List[float], anomalies: int) -> None:
        with self._lock:
            self.total_scored += len(scores)
            self.total_anomalies += anomalies
            if scores:
                self.last_score = scores[-1]
                self.max_score = max(self.max_score, max(scores))

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP ebpf_anomaly_scored_total Total feature vectors scored.",
                "# TYPE ebpf_anomaly_scored_total counter",
                f"ebpf_anomaly_scored_total {self.total_scored}",
                "# HELP ebpf_anomaly_detected_total Total vectors flagged anomalous.",
                "# TYPE ebpf_anomaly_detected_total counter",
                f"ebpf_anomaly_detected_total {self.total_anomalies}",
                "# HELP ebpf_anomaly_score_last Most recent anomaly score (0..1).",
                "# TYPE ebpf_anomaly_score_last gauge",
                f"ebpf_anomaly_score_last {self.last_score}",
                "# HELP ebpf_anomaly_score_max Max anomaly score observed since start.",
                "# TYPE ebpf_anomaly_score_max gauge",
                f"ebpf_anomaly_score_max {self.max_score}",
            ]
            return "\n".join(lines) + "\n"


_metrics = _Metrics()

# --------------------------------------------------------------------------
# Durable feature archive (optional; enabled when ARCHIVE_DIR is set)
# --------------------------------------------------------------------------

_META_COLS = ["timestamp", "node", "namespace", "pod", "container"]
_ARCHIVE_COLS = (["ingest_ts"] + _META_COLS + list(_FEATURE_ORDER)
                 + ["anomaly_score", "is_anomaly", "trigger"])
_archive = FeatureArchive(os.environ.get("ARCHIVE_DIR", ""), _ARCHIVE_COLS)


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


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if _bundle is not None else "degraded",
        "model_loaded": _bundle is not None,
        "model_path": _MODEL_PATH,
        "error": _bundle_err,
        "feature_count": len(_FEATURE_ORDER),
        "threshold": getattr(_bundle, "score_threshold", None),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    _metrics.observe([float(result["anomaly_score"])],
                     1 if result["is_anomaly"] else 0)
    _archive_rows([{**(meta or {}), **feats}], [result])
    return result


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
    _metrics.observe(scores, n_anom)
    _archive_rows(rows, results)
    return {"count": len(results), "anomalies": n_anom, "results": results}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return _metrics.render()


def _main() -> None:
    import uvicorn
    dcfg = _CFG.get("detector", {})
    uvicorn.run(app, host=dcfg.get("host", "0.0.0.0"), port=int(dcfg.get("port", 8080)))


if __name__ == "__main__":
    _main()
