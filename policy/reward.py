#!/usr/bin/env python3
"""
reward.py
=========
Cost model + decision rule for the policy layer.

This is a **cost-sensitive decision**, not reinforcement learning. The tree
estimates ``p = P(fault | state)``; the action is whichever minimises expected
cost::

    expected_cost(p, a) = p * COST["fault"][a] + (1 - p) * COST["benign"][a]
    action              = argmin_a expected_cost(p, a)

Because the decision is over the *probability*, the intermediate actions
(``wait`` / ``triage``) naturally win in the uncertain band where neither
``suppress`` nor ``alert`` is clearly right — without any RL.

Costs are the negation of the operator-reward grid (higher reward = lower cost).
Tune the grid to shift the policy's risk appetite; missing a fault
(``fault`` + ``suppress``) is by far the most expensive cell.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from policy.schema import ESCALATION_ACTIONS

# Operator reward grid: reward[label][action]. Higher = better outcome.
#   benign  -> want it silenced cheaply; a false page (alert) is costly.
#   fault   -> want it escalated; silencing it (suppress) is catastrophic.
REWARD: Dict[str, Dict[str, float]] = {
    "benign": {"suppress": +2.0, "wait": +0.5, "triage": -0.5, "alert": -3.0},
    "fault":  {"suppress": -10.0, "wait": -1.0, "triage": +2.0, "alert": +5.0},
}
COST: Dict[str, Dict[str, float]] = {
    lbl: {a: -r for a, r in row.items()} for lbl, row in REWARD.items()
}


def expected_cost(p_fault: float, action: str) -> float:
    """Expected cost of taking `action` when P(fault)=`p_fault`."""
    p = max(0.0, min(1.0, float(p_fault)))
    return p * COST["fault"][action] + (1.0 - p) * COST["benign"][action]


def best_action(p_fault: float,
                allowed: List[str] | None = None) -> Tuple[str, float]:
    """Return (action, expected_cost) minimising expected cost over `allowed`.

    Defaults to the escalation actions only (wait/triage/alert): the noise filter
    upstream owns `suppress`, so the policy never re-suppresses an anomaly that
    already survived it. Pass allowed=ACTIONS to evaluate the full space (used by
    offline metrics).
    """
    acts = allowed if allowed is not None else ESCALATION_ACTIONS
    costs = [(a, expected_cost(p_fault, a)) for a in acts]
    return min(costs, key=lambda t: t[1])


def expected_reward(p_fault: float, action: str) -> float:
    """Convenience for metrics/telemetry: -expected_cost."""
    return -expected_cost(p_fault, action)


# Map a data source / trigger name to the binary training label.
_FAULT_HINTS = ("stress", "fault", "flood", "bomb", "delay", "loss",
                "abnormal", "attack", "anomal")


def label_from_name(name: str) -> str:
    """benign|fault from a fault-experiment filename or trigger tag.

    `normal`/`baseline` -> benign; anything carrying a fault hint -> fault.
    Unknown -> benign (conservative for training: never invents a fault).
    """
    n = (name or "").lower()
    # Fault hints win first: 'abnormal' contains the substring 'normal', so the
    # benign check must not run before the fault check.
    if any(h in n for h in _FAULT_HINTS):
        return "fault"
    if "normal" in n or "baseline" in n or "benign" in n:
        return "benign"
    return "benign"
