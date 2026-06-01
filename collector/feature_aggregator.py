"""
feature_aggregator.py
=====================
Canonical feature-schema definition + windowed aggregation + IO helpers
for the eBPF_final anomaly-detection pipeline.

Responsibilities
----------------
1. Define the feature schema (single source of truth, mirrors configs/config.yaml).
2. Aggregate raw eBPF counter samples into fixed-length time windows
   (default 10s) and emit one feature vector per (window, node, pod, container).
3. Write feature vectors to CSV and/or JSONL.
4. Provide a *mock* feature generator so the full train/detect/API pipeline
   can be exercised before the eBPF collector is wired up (prompt Step 4).

The Isolation Forest model ONLY ever sees window-level feature vectors,
never raw events.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

# Numeric columns fed into Isolation Forest. Order is significant and MUST
# match configs/config.yaml `features:`. Kept here as a code-level default so
# the module works even if the config file is absent.
FEATURE_COLUMNS: List[str] = [
    "syscall_read_rate",
    "syscall_write_rate",
    "syscall_open_rate",
    "tcp_connect_rate",
    "tcp_retransmit_rate",
    "network_rx_bytes",
    "network_tx_bytes",
    "disk_read_bytes",
    "disk_write_bytes",
    "disk_io_latency_ms",
    "process_exec_count",
    "process_fork_count",
    "context_switch_count",
    "cpu_utilization",   # on-CPU fraction (0..n_cores) from sched_switch runtime
]

# Identity columns (not model inputs).
METADATA_COLUMNS: List[str] = [
    "timestamp",
    "node",
    "namespace",
    "pod",
    "container",
]

# Full CSV/JSONL column order = metadata first, then features.
ALL_COLUMNS: List[str] = METADATA_COLUMNS + FEATURE_COLUMNS


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml; fall back to repo-relative default. Never raises on
    missing file (returns {})."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def feature_columns_from_config(cfg: Optional[dict] = None) -> List[str]:
    """Return the feature column order, preferring config over the code default."""
    cfg = cfg or load_config()
    return list(cfg.get("features") or FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Feature record
# --------------------------------------------------------------------------

@dataclass
class FeatureRecord:
    """One aggregated feature vector for a single window + identity tuple."""
    timestamp: str
    node: str
    namespace: str
    pod: str
    container: str
    # feature values keyed by FEATURE_COLUMNS name
    features: Dict[str, float] = field(default_factory=dict)

    def to_row(self, feature_order: List[str]) -> Dict[str, object]:
        """Flatten to a single dict row (metadata + features) for CSV/JSONL."""
        row: Dict[str, object] = {
            "timestamp": self.timestamp,
            "node": self.node,
            "namespace": self.namespace,
            "pod": self.pod,
            "container": self.container,
        }
        for col in feature_order:
            row[col] = float(self.features.get(col, 0.0))
        return row


# --------------------------------------------------------------------------
# Windowed aggregator
# --------------------------------------------------------------------------

class WindowAggregator:
    """
    Accumulates per-identity raw counter deltas within a time window and emits
    FeatureRecords when the window closes.

    The eBPF collector feeds *cumulative* or *per-sample* counter values via
    ``add_sample``; the aggregator sums them over the window. Rate features
    (``*_rate``) are divided by the window length to yield per-second rates.

    Usage:
        agg = WindowAggregator(window_seconds=10, node="node-1")
        agg.add_sample(identity, {"syscall_read_rate": 120, ...})
        ...
        for rec in agg.maybe_flush():   # call periodically
            write(rec)
    """

    # Feature names ending in these suffixes are treated as per-second rates.
    _RATE_SUFFIXES = ("_rate",)

    def __init__(
        self,
        window_seconds: int = 10,
        node: str = "unknown-node",
        feature_order: Optional[List[str]] = None,
        clock=time.time,
    ):
        self.window_seconds = window_seconds
        self.node = node
        self.feature_order = feature_order or list(FEATURE_COLUMNS)
        self._clock = clock
        self._window_start = self._clock()
        # identity tuple -> {feature_name: summed_value}
        self._acc: Dict[tuple, Dict[str, float]] = {}

    @staticmethod
    def _identity(namespace: str, pod: str, container: str) -> tuple:
        return (namespace, pod, container)

    def add_sample(
        self,
        namespace: str,
        pod: str,
        container: str,
        values: Dict[str, float],
    ) -> None:
        """Add a raw counter sample for one identity into the current window."""
        key = self._identity(namespace, pod, container)
        bucket = self._acc.setdefault(key, {})
        for name, val in values.items():
            bucket[name] = bucket.get(name, 0.0) + float(val)

    def _elapsed(self) -> float:
        return self._clock() - self._window_start

    def maybe_flush(self, force: bool = False) -> List[FeatureRecord]:
        """If the window has elapsed (or force=True), emit FeatureRecords and
        reset the accumulator. Returns [] if the window is still open."""
        if not force and self._elapsed() < self.window_seconds:
            return []
        return self._flush()

    def _flush(self) -> List[FeatureRecord]:
        window_len = max(self._elapsed(), 1e-9)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._window_start))
        out: List[FeatureRecord] = []
        for (namespace, pod, container), bucket in self._acc.items():
            feats: Dict[str, float] = {}
            for col in self.feature_order:
                raw = bucket.get(col, 0.0)
                if col.endswith(self._RATE_SUFFIXES):
                    feats[col] = raw / window_len   # per-second rate
                else:
                    feats[col] = raw                # window-summed count/bytes
            out.append(
                FeatureRecord(
                    timestamp=ts,
                    node=self.node,
                    namespace=namespace,
                    pod=pod,
                    container=container,
                    features=feats,
                )
            )
        # reset
        self._acc = {}
        self._window_start = self._clock()
        return out


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def write_csv(records: Iterable[FeatureRecord], path: str | os.PathLike,
              feature_order: Optional[List[str]] = None, append: bool = True) -> int:
    """Write FeatureRecords to CSV. Returns number of rows written."""
    feature_order = feature_order or list(FEATURE_COLUMNS)
    cols = METADATA_COLUMNS + feature_order
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and path.exists() and path.stat().st_size > 0)
    mode = "a" if append else "w"
    n = 0
    with open(path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        if write_header or mode == "w":
            writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_row(feature_order))
            n += 1
    return n


def write_jsonl(records: Iterable[FeatureRecord], path: str | os.PathLike,
                feature_order: Optional[List[str]] = None, append: bool = True) -> int:
    """Write FeatureRecords to JSONL (one JSON object per line)."""
    feature_order = feature_order or list(FEATURE_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(path, mode) as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_row(feature_order)) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------
# Mock feature generator (prompt Step 4)
# --------------------------------------------------------------------------

# Baseline "normal" mean for each feature. Roughly models an idle-ish web pod.
_NORMAL_MEANS: Dict[str, float] = {
    "syscall_read_rate": 150.0,
    "syscall_write_rate": 90.0,
    "syscall_open_rate": 20.0,
    "tcp_connect_rate": 5.0,
    "tcp_retransmit_rate": 0.2,
    "network_rx_bytes": 50_000.0,
    "network_tx_bytes": 40_000.0,
    "disk_read_bytes": 20_000.0,
    "disk_write_bytes": 15_000.0,
    "disk_io_latency_ms": 2.0,
    "process_exec_count": 1.0,
    "process_fork_count": 1.0,
    "context_switch_count": 800.0,
    "cpu_utilization": 0.05,   # ~5% of one core for an idle-ish pod
}

# Multiplicative fault profiles: feature -> factor applied to the normal mean.
_FAULT_PROFILES: Dict[str, Dict[str, float]] = {
    "cpu_stress": {
        "cpu_utilization": 60.0,   # ~3 cores pegged (0.05 -> 3.0)
        "context_switch_count": 0.1,  # CPU-bound loop yields rarely -> fewer switches
    },
    "network_delay": {
        "tcp_retransmit_rate": 15.0,
        "tcp_connect_rate": 3.0,
        "disk_io_latency_ms": 1.5,
    },
    "packet_loss": {
        "tcp_retransmit_rate": 40.0,
        "network_rx_bytes": 0.4,
        "network_tx_bytes": 0.4,
    },
    "disk_io_stress": {
        "disk_read_bytes": 8.0,
        "disk_write_bytes": 10.0,
        "disk_io_latency_ms": 12.0,
    },
    "abnormal_requests": {
        "syscall_read_rate": 5.0,
        "syscall_write_rate": 5.0,
        "tcp_connect_rate": 20.0,
        "network_rx_bytes": 6.0,
    },
}


def _sample_normal(mean: float, rng: random.Random, jitter: float = 0.15) -> float:
    """Gaussian sample around mean with relative jitter, clamped at >= 0."""
    val = rng.gauss(mean, abs(mean) * jitter)
    return max(0.0, val)


def generate_mock_features(
    n: int,
    fault: Optional[str] = None,
    node: str = "mock-node",
    namespace: str = "default",
    pod: str = "mock-pod",
    container: str = "app",
    seed: Optional[int] = None,
    start_ts: Optional[float] = None,
    window_seconds: int = 10,
) -> List[FeatureRecord]:
    """
    Generate ``n`` synthetic FeatureRecords.

    fault=None  -> normal traffic (used for training + normal baseline).
    fault=<key> -> inject one of _FAULT_PROFILES (cpu_stress, network_delay, ...).

    Returns a list of FeatureRecords with monotonically increasing timestamps
    spaced ``window_seconds`` apart.
    """
    rng = random.Random(seed)
    profile = _FAULT_PROFILES.get(fault, {}) if fault else {}
    base_ts = start_ts if start_ts is not None else time.time()
    out: List[FeatureRecord] = []
    for i in range(n):
        feats: Dict[str, float] = {}
        for col in FEATURE_COLUMNS:
            mean = _NORMAL_MEANS[col] * profile.get(col, 1.0)
            feats[col] = round(_sample_normal(mean, rng), 3)
        ts_epoch = base_ts + i * window_seconds
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_epoch))
        out.append(
            FeatureRecord(
                timestamp=ts, node=node, namespace=namespace,
                pod=pod, container=container, features=feats,
            )
        )
    return out


# --------------------------------------------------------------------------
# CLI: generate mock feature files
# --------------------------------------------------------------------------

def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate mock eBPF feature vectors (CSV + JSONL)."
    )
    p.add_argument("-n", "--count", type=int, default=500,
                   help="number of feature windows to generate")
    p.add_argument("--fault", choices=sorted(_FAULT_PROFILES), default=None,
                   help="inject a fault profile (default: normal traffic)")
    p.add_argument("--out", default="data/mock_features.csv",
                   help="output path; .jsonl sibling is also written")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pod", default="mock-pod")
    p.add_argument("--namespace", default="default")
    args = p.parse_args()

    recs = generate_mock_features(
        args.count, fault=args.fault, seed=args.seed,
        pod=args.pod, namespace=args.namespace,
    )
    csv_path = Path(args.out)
    jsonl_path = csv_path.with_suffix(".jsonl")
    n_csv = write_csv(recs, csv_path, append=False)
    n_jsonl = write_jsonl(recs, jsonl_path, append=False)
    label = args.fault or "normal"
    print(f"[mock] wrote {n_csv} rows -> {csv_path} ({label})")
    print(f"[mock] wrote {n_jsonl} rows -> {jsonl_path} ({label})")


if __name__ == "__main__":
    _main()
