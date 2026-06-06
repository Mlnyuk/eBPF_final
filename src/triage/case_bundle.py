#!/usr/bin/env python3
"""
case_bundle.py
==============
Group per-window eBPF feature rows into *case bundles* — the labelable +
trainable unit of the triage pipeline.

A case bundle carries the reduced 9-feature view, a per-container rolling
baseline, anomaly duration, optional fault-injection metadata, and per-model
action votes (used downstream to detect disagreement = "ambiguous case" worth
sending to the LLM adjudicator).

Input  : JSONL feature rows (detector archive rows or data/features/windows.jsonl).
         Each row should carry the 14-col collector schema + anomaly_score, or
         the 9 case-feature names directly, plus identity metadata.
Output : JSONL case bundles (data/cases/cases.jsonl).

CPU-only, no GPU, no model dependency. Reads the schema at
schemas/case_bundle.schema.json.

Usage:
    python -m src.triage.case_bundle \
      --input data/features/windows.jsonl \
      --output data/cases/cases.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from src.triage.cost_matrix import action_for_pure_class

# Reduced feature set carried in the bundle (see schemas/case_bundle.schema.json).
CASE_FEATURES: List[str] = [
    "anomaly_score", "syscall_rate", "tcp_retrans", "disk_read_bytes",
    "disk_write_bytes", "disk_latency_p95", "ctx_switch_rate",
    "net_rx_bytes", "net_tx_bytes",
]

# Map fault-experiment source names -> canonical fault_type.
FILENAME_FAULT_HINTS: List[Tuple[str, str]] = [
    ("cpu_stress", "cpu_pressure"),
    ("cpu", "cpu_pressure"),
    ("mem", "memory_pressure"),
    ("oom", "memory_pressure"),
    ("disk_io", "disk_io_latency"),
    ("disk", "disk_io_latency"),
    ("network_delay", "tcp_retrans"),
    ("packet_loss", "tcp_retrans"),
    ("tcp", "tcp_retrans"),
    ("abnormal_requests", "syscall_spike"),
    ("syscall", "syscall_spike"),
]

BASELINE_WINDOW_S = 300  # 5-minute rolling baseline


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value) -> Optional[datetime]:
    """Best-effort ISO8601 / epoch parser -> aware datetime (UTC)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:  # epoch as string
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except ValueError:
            return None


def _f(row: Dict, *keys: str) -> float:
    """First present key coerced to float (0.0 on miss/None/non-numeric)."""
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def features_from_row(row: Dict) -> Dict[str, float]:
    """Build the 9 case features from a 14-col collector row (or a row that
    already uses the case-feature names). Missing inputs -> 0.0."""
    return {
        "anomaly_score":   _f(row, "anomaly_score"),
        "syscall_rate":    (_f(row, "syscall_rate")
                            or _f(row, "syscall_read_rate")
                            + _f(row, "syscall_write_rate")
                            + _f(row, "syscall_open_rate")),
        "tcp_retrans":     _f(row, "tcp_retrans", "tcp_retransmit_rate"),
        "disk_read_bytes": _f(row, "disk_read_bytes"),
        "disk_write_bytes": _f(row, "disk_write_bytes"),
        "disk_latency_p95": _f(row, "disk_latency_p95", "disk_io_latency_ms"),
        "ctx_switch_rate": _f(row, "ctx_switch_rate", "context_switch_count"),
        "net_rx_bytes":    _f(row, "net_rx_bytes", "network_rx_bytes"),
        "net_tx_bytes":    _f(row, "net_tx_bytes", "network_tx_bytes"),
    }


def fault_metadata_from_row(row: Dict) -> Dict:
    """Extract injection metadata. Honours explicit fields; else infers
    fault_type from a source_file / label hint."""
    fm = dict(row.get("fault_metadata") or {})
    is_injected = bool(fm.get("is_injected", row.get("is_injected", False)))
    fault_type = fm.get("fault_type") or row.get("fault_type") or "none"
    source = fm.get("source_file") or row.get("source_file") or row.get("label")
    if fault_type in (None, "", "none") and source:
        low = str(source).lower()
        for hint, ft in FILENAME_FAULT_HINTS:
            if hint in low:
                fault_type, is_injected = ft, True
                break
    return {
        "is_injected": is_injected,
        "fault_type": fault_type or "none",
        "fault_start": fm.get("fault_start") or row.get("fault_start"),
        "fault_end": fm.get("fault_end") or row.get("fault_end"),
        "source_file": source,
    }


def _threshold_vote(score: float, hi: float, mid: float) -> str:
    """Pure-anomaly-score routing (the legacy 'threshold' model's vote)."""
    if score >= hi:
        return "alert"
    if score >= mid:
        return "triage"
    if score >= mid * 0.7:
        return "wait"
    return "suppress"


def _quick_weak_vote(feats: Dict[str, float], base: Dict[str, float],
                     duration: int) -> str:
    """Cheap heuristic class -> action, only for the ambiguity gate. The
    authoritative weak label is produced by weak_labeler.py."""
    score = feats["anomaly_score"]
    lat = feats["disk_latency_p95"]
    base_lat = base.get("disk_latency_p95_mean_5m", 0.0)
    if score >= 0.9 and lat > max(base_lat * 3.0, 50.0):
        cls = "high"
    elif score >= 0.85 and duration >= 2:
        cls = "medium"
    elif score >= 0.7:
        cls = "low" if duration <= 1 else "medium"
    else:
        cls = "normal"
    return action_for_pure_class(cls)


@dataclass
class CaseBundle:
    event_id: str
    created_at: str
    cluster: str
    node: str
    namespace: str
    pod: str
    container: str
    window_start: str
    window_end: str
    features: Dict[str, float]
    baseline: Dict[str, float] = field(default_factory=dict)
    duration_windows: int = 1
    logs_summary: str = ""
    fault_metadata: Dict = field(default_factory=dict)
    model_votes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class _IdentityState:
    """Rolling baseline (normal windows within 5m) + anomaly run length."""
    __slots__ = ("samples", "run")

    def __init__(self) -> None:
        self.samples: Deque[Tuple[datetime, Dict[str, float]]] = deque()
        self.run = 0

    def baseline(self) -> Dict[str, float]:
        if not self.samples:
            return {"anomaly_score_mean_5m": 0.0,
                    "disk_latency_p95_mean_5m": 0.0,
                    "syscall_rate_mean_5m": 0.0}
        n = len(self.samples)
        agg = {"anomaly_score": 0.0, "disk_latency_p95": 0.0, "syscall_rate": 0.0}
        for _, f in self.samples:
            for k in agg:
                agg[k] += f[k]
        return {f"{k}_mean_5m": agg[k] / n for k in agg}

    def push_normal(self, ts: Optional[datetime], feats: Dict[str, float]) -> None:
        self.run = 0
        if ts is None:
            return
        self.samples.append((ts, feats))
        cutoff = ts.timestamp() - BASELINE_WINDOW_S
        while self.samples and self.samples[0][0].timestamp() < cutoff:
            self.samples.popleft()


def _identity_key(row: Dict) -> Tuple[str, str, str, str]:
    return (str(row.get("node", "")), str(row.get("namespace", "")),
            str(row.get("pod", "")), str(row.get("container", "")))


def _is_anomaly(row: Dict, score: float, score_threshold: float) -> bool:
    flag = row.get("is_anomaly")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in ("true", "1", "1.0")
    return score >= score_threshold


def bundles_from_rows(rows: Iterable[Dict], score_threshold: float = 0.7,
                      hi: float = 0.9, mid: float = 0.7,
                      window_seconds: int = 10,
                      all_windows: bool = False) -> Iterable[CaseBundle]:
    """Stream rows (assumed time-ordered per identity) -> case bundles for the
    anomalous windows. Normal windows update the per-identity baseline."""
    state: Dict[Tuple, _IdentityState] = {}
    for row in rows:
        key = _identity_key(row)
        st = state.setdefault(key, _IdentityState())
        feats = features_from_row(row)
        ts = _parse_ts(row.get("window_start") or row.get("timestamp"))
        anom = _is_anomaly(row, feats["anomaly_score"], score_threshold)
        if not anom and not all_windows:
            st.push_normal(ts, feats)
            continue
        st.run += 1
        base = st.baseline()
        ws = row.get("window_start") or row.get("timestamp") or _now_iso()
        we = row.get("window_end")
        if not we:
            end_dt = (ts.timestamp() + window_seconds) if ts else None
            we = (datetime.fromtimestamp(end_dt, tz=timezone.utc).isoformat()
                  if end_dt else str(ws))
        eid = hashlib.sha1(
            f"{'/'.join(key)}/{ws}".encode()).hexdigest()[:16]
        votes = {
            "threshold": _threshold_vote(feats["anomaly_score"], hi, mid),
            "weak_label": _quick_weak_vote(feats, base, st.run),
        }
        yield CaseBundle(
            event_id=eid,
            created_at=_now_iso(),
            cluster=str(row.get("cluster", "")),
            node=key[0], namespace=key[1], pod=key[2], container=key[3],
            window_start=str(ws), window_end=str(we),
            features=feats,
            baseline=base,
            duration_windows=st.run,
            logs_summary=str(row.get("logs_summary", "")),
            fault_metadata=fault_metadata_from_row(row),
            model_votes=votes,
        )


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> Iterable[Dict]:
    """Stream rows from a detector archive CSV (features-*.csv). Values stay as
    strings; features_from_row / _is_anomaly coerce them as needed."""
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            yield row


def _read_rows(path: Path) -> Iterable[Dict]:
    """Dispatch on suffix: detector archive CSV or JSONL feature windows."""
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    return _read_jsonl(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build triage case bundles from feature windows.")
    ap.add_argument("--input", required=True,
                    help="feature/window rows: JSONL, or a detector archive .csv")
    ap.add_argument("--output", required=True, help="JSONL case bundles out")
    ap.add_argument("--score-threshold", type=float, default=0.7,
                    help="anomaly_score gate when is_anomaly is absent")
    ap.add_argument("--hi", type=float, default=0.9, help="threshold-vote alert band")
    ap.add_argument("--mid", type=float, default=0.7, help="threshold-vote triage band")
    ap.add_argument("--window-seconds", type=int, default=10)
    ap.add_argument("--all-windows", action="store_true",
                    help="emit a bundle for every window, not only anomalous ones")
    args = ap.parse_args(argv)

    inp, outp = Path(args.input), Path(args.output)
    if not inp.exists():
        print(f"case_bundle: input not found: {inp}", file=sys.stderr)
        return 1
    outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with outp.open("w") as out:
        for b in bundles_from_rows(_read_rows(inp), args.score_threshold,
                                   args.hi, args.mid, args.window_seconds,
                                   args.all_windows):
            out.write(json.dumps(b.to_dict()) + "\n")
            n += 1
    print(f"case_bundle: wrote {n} bundles -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
