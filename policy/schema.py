#!/usr/bin/env python3
"""
schema.py
=========
Data contract for the runtime policy layer (Policy v0).

Defines the **state vector** the policy sees, the **action space** it chooses
from, and the **archive fields** appended so every decision is replayable for
future offline policy improvement.

State design mirrors the noise filter: the policy judges an anomaly relative to
the container's *own* normal baseline, not absolute magnitude. It additionally
consumes the upstream signals (the IF score and the noise filter's suppress
probability) so it never has to re-derive them.

    state = [ anomaly_score,
              suppress_proba,        # noise filter output (0=keep .. 1=benign)
              has_baseline,
              dev_<feat> ... ]       # signed log10 deviation vs baseline (14)

No RL here: ``episode_id`` is recorded for *future* sequential analysis only and
plays no part in v0 training or runtime.
"""
from __future__ import annotations

import math
from typing import Dict, List

_EPS = 1e-6

# Same 14 features the detector + noise filter use, in the same order.
FEATS: List[str] = [
    "syscall_read_rate", "syscall_write_rate", "syscall_open_rate",
    "tcp_connect_rate", "tcp_retransmit_rate", "network_rx_bytes",
    "network_tx_bytes", "disk_read_bytes", "disk_write_bytes",
    "disk_io_latency_ms", "process_exec_count", "process_fork_count",
    "context_switch_count", "cpu_utilization",
]

# Action space (index = id). `suppress` is owned by the noise filter upstream;
# at runtime the policy chooses only among the ESCALATION actions (see
# ESCALATION_ACTIONS) and never undoes a suppression.
ACTIONS: List[str] = ["suppress", "wait", "triage", "alert"]
ACTION_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
ESCALATION_ACTIONS: List[str] = ["wait", "triage", "alert"]

# Binary label for v0. Fault subtype is kept as metadata only (not a train target).
LABELS: List[str] = ["benign", "fault"]

# Order of the state vector columns (for the model + interpretability).
STATE_COLS: List[str] = (
    ["anomaly_score", "suppress_proba", "has_baseline"]
    + [f"dev_{f}" for f in FEATS]
)

# Fields appended to the detector archive for each scored row, so the policy's
# decisions are durable and replayable for future improvement.
POLICY_ARCHIVE_FIELDS: List[str] = ["policy_action", "policy_p_fault", "episode_id"]


def build_state(baseline: Dict[str, Dict[str, float]], container: str,
                feats: Dict, anomaly_score: float, suppress_proba: float) -> List[float]:
    """Materialise the policy state vector for one window.

    `baseline` maps container -> {feature: normal median}; rows for a container
    without a baseline get has_baseline=0 and deviations measured against 0.
    """
    bl = baseline.get(container or "")
    x = [float(anomaly_score), float(suppress_proba), 1.0 if bl else 0.0]
    for f in FEATS:
        o = float(feats.get(f) or 0.0)
        b = (bl.get(f) if bl else 0.0) or 0.0
        x.append(math.log10((o + _EPS) / (b + _EPS)))
    return x


def episode_id(container: str, ts: str, bucket_s: int = 300) -> str:
    """Coarse episode key = container + time bucket (default 5 min).

    Recorded for FUTURE sequential analysis (alert-fatigue / bandit). Unused by
    v0 training or runtime. Best-effort: returns ``container:?`` if ts unparsable.
    """
    try:
        import calendar
        import time as _t
        epoch = calendar.timegm(_t.strptime((ts or "")[:19], "%Y-%m-%dT%H:%M:%S"))
        return f"{container or '?'}:{int(epoch // bucket_s)}"
    except Exception:  # noqa: BLE001 - episode id is best-effort metadata
        return f"{container or '?'}:?"
