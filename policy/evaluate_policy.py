#!/usr/bin/env python3
"""
evaluate_policy.py
==================
Promotion gate for Policy v0 (mirrors the noise-filter distill gate).

A new policy is promoted into the detector only if its CV gate metrics clear all
thresholds; otherwise the running policy is kept. Reads the metrics JSON written
by train_policy.py and exits 0 (PROMOTE) or 1 (REJECT).

Thresholds (override via env):
  POLICY_MIN_FAULT_ESCALATE   faults that get triage|alert      >= 0.90
  POLICY_MAX_FAULT_MISS       faults parked on 'wait'           <= 0.05
  POLICY_MAX_BENIGN_ALERT     benign that get paged             <= 0.02
  POLICY_MAX_TRIAGE_OVERUSE   share of all decisions = triage   <= 0.40
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

GATES: List[Tuple[str, str, float, str]] = [
    # (metric_key, comparator, env_var, default)
    ("fault_escalate_rate", ">=", "POLICY_MIN_FAULT_ESCALATE", 0.90),
    ("fault_miss_rate", "<=", "POLICY_MAX_FAULT_MISS", 0.05),
    ("benign_alert_rate", "<=", "POLICY_MAX_BENIGN_ALERT", 0.02),
    ("triage_overuse", "<=", "POLICY_MAX_TRIAGE_OVERUSE", 0.40),
]


def evaluate(metrics: Dict) -> Tuple[bool, List[str]]:
    ok, report = True, []
    for key, cmp, env, default in GATES:
        thr = float(os.environ.get(env, default))
        val = metrics.get(key)
        if val is None:
            report.append(f"  ? {key}: missing")
            ok = False
            continue
        passed = val >= thr if cmp == ">=" else val <= thr
        ok = ok and passed
        report.append(f"  {'PASS' if passed else 'FAIL'} {key}={val:.3f} {cmp} {thr}")
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics JSON from train_policy.py")
    args = ap.parse_args()
    metrics = json.load(open(args.metrics))
    ok, report = evaluate(metrics)
    print("policy gate:")
    print("\n".join(report))
    print("verdict:", "PROMOTE" if ok else "REJECT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
