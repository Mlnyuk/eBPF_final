#!/usr/bin/env python3
"""
adjudicator_prompt.py
=====================
Prompt construction + response parsing for the LLM triage adjudicator.

The prompt encodes the project's decision philosophy explicitly so the model
optimises the right objective (expected operational cost), not raw severity.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

from src.triage.cost_matrix import CLASSES, ACTIONS, FAULT_TYPES, COST_MATRIX

SYSTEM_PROMPT = f"""You are a Kubernetes anomaly TRIAGE adjudicator.

Decision framing (read carefully):
- This is NOT a reinforcement-learning problem. Your chosen action does NOT
  change the cluster state; there is no future reward to optimise.
- Your single job: pick the action that minimises EXPECTED OPERATIONAL COST for
  this one event, choosing from: suppress, wait, triage, alert.
- A high-severity FALSE NEGATIVE (suppressing a real 'high' event) is the most
  expensive mistake by far. Paging on a benign event is cheap by comparison.
- If you are uncertain, return a LOW confidence value. Do not fabricate a fault.

Severity classes: {", ".join(CLASSES)}.
Actions: {", ".join(ACTIONS)}.
Fault types: {", ".join(FAULT_TYPES)}.

Cost matrix COST[action][true_class] (lower is better):
{json.dumps(COST_MATRIX, indent=2)}

Judge DEVIATION FROM THE PER-CONTAINER BASELINE, not absolute magnitude: system
daemons (containerd, kubelet, prometheus) legitimately run high. A single-window
spike is usually 'wait'; a sustained breach with a clear feature signature is
'triage' or 'alert'.

Respond with STRICT JSON ONLY, no prose, no markdown fences:
{{"true_class": "<class>", "recommended_action": "<action>",
  "fault_type": "<fault_type>", "confidence": <0..1>, "reason": "<short>"}}"""


def _deviation(bundle: Dict) -> Dict[str, float]:
    f = bundle.get("features", {})
    b = bundle.get("baseline", {})
    def dev(v, base):
        base = base or 0.0
        return round(float(v) - float(base), 3)
    return {
        "anomaly_score_vs_base": dev(f.get("anomaly_score", 0.0),
                                     b.get("anomaly_score_mean_5m")),
        "disk_latency_p95_vs_base": dev(f.get("disk_latency_p95", 0.0),
                                        b.get("disk_latency_p95_mean_5m")),
        "syscall_rate_vs_base": dev(f.get("syscall_rate", 0.0),
                                    b.get("syscall_rate_mean_5m")),
    }


def build_user_prompt(bundle: Dict) -> str:
    """Compact, model-friendly view of one case bundle."""
    view = {
        "event_id": bundle.get("event_id"),
        "identity": {k: bundle.get(k) for k in ("node", "namespace", "pod", "container")},
        "window": {"start": bundle.get("window_start"), "end": bundle.get("window_end"),
                   "duration_windows": bundle.get("duration_windows", 1)},
        "features": bundle.get("features", {}),
        "baseline": bundle.get("baseline", {}),
        "deviation": _deviation(bundle),
        "fault_metadata": bundle.get("fault_metadata", {}),
        "model_votes": bundle.get("model_votes", {}),
        "logs_summary": bundle.get("logs_summary", ""),
    }
    return ("Adjudicate this single case. Return the strict JSON object only.\n"
            + json.dumps(view, indent=1))


def parse_response(text: str) -> Dict:
    """Extract + validate the model's JSON label. Raises ValueError on failure."""
    if not text or not text.strip():
        raise ValueError("empty response")
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in response")
    obj = json.loads(text[s:e + 1])
    tc = str(obj.get("true_class", "")).lower().strip()
    if tc not in CLASSES:
        raise ValueError(f"invalid true_class: {tc!r}")
    act = str(obj.get("recommended_action", "")).lower().strip()
    if act not in ACTIONS:
        raise ValueError(f"invalid recommended_action: {act!r}")
    ft = str(obj.get("fault_type", "unknown")).lower().strip()
    if ft not in FAULT_TYPES:
        ft = "unknown"
    try:
        conf = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "true_class": tc,
        "recommended_action": act,
        "fault_type": ft,
        "confidence": max(0.0, min(1.0, conf)),
        "reason": str(obj.get("reason", ""))[:500],
    }


def build_messages(bundle: Dict) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(bundle)}]
