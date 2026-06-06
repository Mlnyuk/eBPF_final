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

from typing import Dict, List, Mapping, Tuple

# Severity classes (ascending) and the operator actions.
CLASSES: List[str] = ["normal", "low", "medium", "high"]
ACTIONS: List[str] = ["suppress", "wait", "triage", "alert"]

# Canonical fault taxonomy shared across weak labeler / adjudicator / store.
FAULT_TYPES: List[str] = [
    "none", "syscall_spike", "tcp_retrans", "disk_io_latency",
    "cpu_pressure", "memory_pressure", "unknown",
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
