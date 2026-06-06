#!/usr/bin/env python3
"""
cost_matrix.py
==============
Operational cost model + expected-cost decision rule for triage.

This is the heart of the *cost-sensitive supervised* design: a classifier
estimates ``P(true_class | x)`` over four severity classes, and the action is
whichever minimises expected operational cost::

    expected_cost(action) = sum_c  P(c | x) * COST_MATRIX[action][c]
    recommended_action    = argmin_action expected_cost(action)

No reinforcement learning: the action does not change the cluster state, so
there is no transition model and no reward to bootstrap. Accuracy is secondary;
the objective is to minimise expected cost, where a high-severity false negative
(``suppress`` on a ``high`` case) is by far the most expensive cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

# Severity classes (ascending) and the operator actions.
CLASSES: List[str] = ["normal", "low", "medium", "high"]
ACTIONS: List[str] = ["suppress", "wait", "triage", "alert"]

# Canonical fault taxonomy shared across weak labeler / adjudicator / store.
# Covers the replay scenario classes: CPU, memory, network, disk/PVC, Kubernetes
# control-plane, GPU, plus syscall spikes; `none` = normal/hard-negative.
FAULT_TYPES: List[str] = [
    "none", "syscall_spike", "tcp_retrans", "disk_io_latency",
    "cpu_pressure", "memory_pressure", "pvc_io_latency",
    "controlplane_pressure", "gpu_pressure", "unknown",
]

# Default cost matrix: COST_MATRIX[action][true_class]. Lower = better outcome.
# Silencing a real high-severity event is catastrophic (20); paging on normal is
# annoying but cheap (3). Override at train time via --cost-matrix JSON.
COST_MATRIX: Dict[str, Dict[str, float]] = {
    "suppress": {"normal": 0,   "low": 1,   "medium": 6, "high": 20},
    "wait":     {"normal": 0.5, "low": 0,   "medium": 3, "high": 12},
    "triage":   {"normal": 1,   "low": 0.5, "medium": 0, "high": 4},
    "alert":    {"normal": 3,   "low": 2,   "medium": 1, "high": 0},
}

# Recall-first cost matrix (PRODUCTION DEFAULT). A high-severity false negative —
# suppressing or merely deferring a real `high` — is penalised far more
# aggressively (suppress*high 20 -> 60, wait*high 12 -> 35, and medium FNs are
# raised too), while the cost of over-escalating a benign case stays modest.
# Use when missing an incident is much worse than an extra page; this is the
# operational stance for an anomaly detector whose job is to not miss faults.
# Costs are decoupled from the model, so this can be retuned without retraining.
RECALL_FIRST_COST_MATRIX: Dict[str, Dict[str, float]] = {
    "suppress": {"normal": 0,   "low": 2,   "medium": 15, "high": 60},
    "wait":     {"normal": 0.5, "low": 0,   "medium": 8,  "high": 35},
    "triage":   {"normal": 1,   "low": 0.5, "medium": 0,  "high": 8},
    "alert":    {"normal": 3,   "low": 2,   "medium": 1,  "high": 0},
}

COST_MATRICES: Dict[str, Dict[str, Dict[str, float]]] = {
    "default": COST_MATRIX,
    "recall_first": RECALL_FIRST_COST_MATRIX,
}
# Production stance: recall-first. Override per call / via TRIAGE_COST_MODE env.
DEFAULT_COST_MODE = "recall_first"

# Action severity order (suppress < wait < triage < alert) for guardrail flooring.
ACTION_SEVERITY: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}


def get_cost_matrix(mode_or_matrix=None) -> Dict[str, Dict[str, float]]:
    """Resolve a cost matrix. None -> production default (recall_first); a known
    mode name -> that matrix; an explicit matrix dict -> returned as-is."""
    if mode_or_matrix is None:
        return COST_MATRICES[DEFAULT_COST_MODE]
    if isinstance(mode_or_matrix, str):
        if mode_or_matrix not in COST_MATRICES:
            raise ValueError(f"unknown cost mode {mode_or_matrix!r}; "
                             f"known: {sorted(COST_MATRICES)}")
        return COST_MATRICES[mode_or_matrix]
    return dict(mode_or_matrix)


def expected_cost(action: str, prob: Mapping[str, float],
                  cost_matrix: Mapping[str, Mapping[str, float]] = COST_MATRIX) -> float:
    """Expected cost of `action` under the class-probability dict `prob`."""
    row = cost_matrix[action]
    return float(sum(float(prob.get(c, 0.0)) * row[c] for c in CLASSES))


def expected_costs(prob: Mapping[str, float],
                   cost_matrix: Mapping[str, Mapping[str, float]] = COST_MATRIX
                   ) -> Dict[str, float]:
    """Expected cost for every action -> {action: cost}."""
    return {a: expected_cost(a, prob, cost_matrix) for a in ACTIONS}


def recommended_action(prob: Mapping[str, float],
                       cost_matrix: Mapping[str, Mapping[str, float]] = COST_MATRIX
                       ) -> Tuple[str, Dict[str, float]]:
    """Return (argmin-cost action, {action: expected_cost})."""
    costs = expected_costs(prob, cost_matrix)
    best = min(ACTIONS, key=lambda a: costs[a])
    return best, costs


def action_for_pure_class(true_class: str,
                          cost_matrix: Mapping[str, Mapping[str, float]] = COST_MATRIX
                          ) -> str:
    """Cost-minimising action if the class were known with certainty.
    With the default matrix: normal->suppress, low->wait, medium->triage,
    high->alert. Used by the weak labeler to attach a recommended_action."""
    onehot = {c: (1.0 if c == true_class else 0.0) for c in CLASSES}
    return recommended_action(onehot, cost_matrix)[0]


# --------------------------------------------------------------------------
# Safety guardrails
# --------------------------------------------------------------------------
# The expected-cost argmin is the primary decision. Guardrails are a hard safety
# net on top of it: certain conditions must NEVER be silently suppressed,
# regardless of what the cost minimisation says. A guardrail can only RAISE the
# action (toward escalation), never lower it. Core invariant: if any guardrail
# trips, the resulting action is never `suppress`.

@dataclass(frozen=True)
class GuardrailConfig:
    high_score_floor: float = 0.9   # anomaly_score >= -> at least triage
    p_high_alert: float = 0.5       # P(high) >= -> at least alert
    p_high_triage: float = 0.25     # P(high) >= -> at least triage
    min_confidence: float = 0.5     # max class prob < -> at least wait (uncertain)
    unknown_workload_floor: str = "triage"   # unknown identity -> at least triage
    enabled: bool = True


GUARDRAILS = GuardrailConfig()

# Identities we treat as "unknown workload" (collector could not attribute the
# window to a known namespace/pod) — these must not be silently suppressed.
_UNKNOWN_TOKENS = {"", "unknown", "none", "null", "n/a"}


def is_unknown_workload(identity: Optional[Mapping[str, str]]) -> bool:
    """True when the case cannot be attributed to a known workload."""
    if not identity:
        return True
    ns = str(identity.get("namespace", "")).strip().lower()
    pod = str(identity.get("pod", "")).strip().lower()
    return ns in _UNKNOWN_TOKENS or pod in _UNKNOWN_TOKENS


def guardrail_floor(prob: Mapping[str, float], features: Mapping[str, float],
                    identity: Optional[Mapping[str, str]] = None,
                    cfg: GuardrailConfig = GUARDRAILS) -> Tuple[str, List[str]]:
    """Minimum action the guardrails permit, plus the reasons that set it.
    Returns ("suppress", []) when nothing trips."""
    floor = "suppress"
    reasons: List[str] = []
    if not cfg.enabled:
        return floor, reasons

    def raise_to(action: str, why: str) -> None:
        nonlocal floor
        if ACTION_SEVERITY[action] > ACTION_SEVERITY[floor]:
            floor = action
        reasons.append(why)

    score = float(features.get("anomaly_score", 0.0))
    p_high = float(prob.get("high", 0.0))
    confidence = max(prob.values()) if prob else 0.0

    if score >= cfg.high_score_floor:
        raise_to("triage", f"anomaly_score={score:.2f}>={cfg.high_score_floor}")
    if p_high >= cfg.p_high_alert:
        raise_to("alert", f"P(high)={p_high:.2f}>={cfg.p_high_alert}")
    elif p_high >= cfg.p_high_triage:
        raise_to("triage", f"P(high)={p_high:.2f}>={cfg.p_high_triage}")
    if confidence < cfg.min_confidence:
        raise_to("wait", f"low_confidence(max_p={confidence:.2f}<{cfg.min_confidence})")
    if is_unknown_workload(identity):
        raise_to(cfg.unknown_workload_floor, "unknown_workload")

    return floor, reasons


def decide_action(prob: Mapping[str, float], features: Mapping[str, float],
                  identity: Optional[Mapping[str, str]] = None,
                  cost_matrix=None,
                  guardrails: GuardrailConfig = GUARDRAILS
                  ) -> Tuple[str, Dict[str, float], List[str]]:
    """Production decision: expected-cost argmin, then guardrail flooring.

    Returns (action, {action: expected_cost}, guardrail_reasons). The guardrail
    reasons are non-empty only when a guardrail actually raised the action above
    the cost-minimising choice."""
    cm = get_cost_matrix(cost_matrix)
    base, costs = recommended_action(prob, cm)
    floor, reasons = guardrail_floor(prob, features, identity, guardrails)
    if ACTION_SEVERITY[floor] > ACTION_SEVERITY[base]:
        return floor, costs, reasons
    return base, costs, []
