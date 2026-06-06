#!/usr/bin/env python3
"""
weak_labeler.py
===============
First-pass (weak) labeler — bulk-runnable on V100 workers, CPU-only fallback.

Reads case bundles and emits labels into the label store. Two label sources are
produced:

* ``fault_metadata`` — when the bundle carries a known injection. This is treated
  as ground truth (higher priority than heuristics) by label_store.resolve().
* ``weak_rule``      — heuristic class from the feature/baseline deviation when no
  injection is known.

Beyond ``anomaly_score`` the rules use disk latency, syscall rate, tcp retrans,
context-switch rate, and anomaly *duration* (single-window spikes are cheap to
park on ``wait``). Routine operational spikes (cron, image pulls, Jupyter
startup) are mined as **hard negatives** — normal cases that naive models tend to
over-alert on.

Usage:
    python -m src.triage.weak_labeler \
      --cases data/cases/cases.jsonl \
      --out   data/labels/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.triage.cost_matrix import action_for_pure_class
from src.triage.label_store import LabelStore, make_label

# Identity substrings that mark routine operational spikes -> hard negatives.
HARD_NEGATIVE_HINTS: Tuple[str, ...] = (
    "jupyter", "image-pull", "imagepull", "registry", "cron", "backup",
    "prometheus", "grafana", "fluent", "promtail", "kube-proxy",
    "coredns", "calico", "cilium", "longhorn", "kaniko",
)

# Map an injected fault_type to a baseline severity floor.
FAULT_CLASS_FLOOR: Dict[str, str] = {
    "cpu_pressure": "medium",
    "memory_pressure": "medium",
    "disk_io_latency": "medium",
    "tcp_retrans": "medium",
    "syscall_spike": "medium",
    "unknown": "low",
    "none": "normal",
}
_CLASS_ORDER = {"normal": 0, "low": 1, "medium": 2, "high": 3}


def _escalate(cls: str, to: str) -> str:
    return to if _CLASS_ORDER[to] > _CLASS_ORDER[cls] else cls


def _hard_negative_hint(bundle: Dict) -> Optional[str]:
    ident = " ".join(str(bundle.get(k, "")) for k in
                     ("namespace", "pod", "container")).lower()
    for h in HARD_NEGATIVE_HINTS:
        if h in ident:
            return h
    return None


def _infer_fault_type(feats: Dict[str, float], base: Dict[str, float]) -> str:
    score = feats["anomaly_score"]
    lat, base_lat = feats["disk_latency_p95"], base.get("disk_latency_p95_mean_5m", 0.0)
    sysr, base_sys = feats["syscall_rate"], base.get("syscall_rate_mean_5m", 0.0)
    if lat > max(base_lat * 3.0, 50.0):
        return "disk_io_latency"
    if feats["tcp_retrans"] > 0 and score >= 0.85:
        return "tcp_retrans"
    if sysr > max(base_sys * 3.0, 100.0):
        return "syscall_spike"
    if feats["ctx_switch_rate"] > 0 and score >= 0.9:
        return "cpu_pressure"
    return "unknown"


def label_for_bundle(bundle: Dict) -> Dict:
    """Apply the weak-labeling rules to one case bundle -> one label dict."""
    eid = bundle["event_id"]
    feats = bundle.get("features", {})
    base = bundle.get("baseline", {})
    duration = int(bundle.get("duration_windows", 1) or 1)
    fm = bundle.get("fault_metadata", {}) or {}
    score = float(feats.get("anomaly_score", 0.0))

    # --- 1. Known injection -> ground-truth-ish label (fault_metadata source) ---
    if fm.get("is_injected") and fm.get("fault_type") not in (None, "", "none"):
        ft = fm["fault_type"]
        cls = FAULT_CLASS_FLOOR.get(ft, "medium")
        if score >= 0.9 or duration >= 3:
            cls = _escalate(cls, "high")
        return make_label(
            eid, "fault_metadata", cls, action_for_pure_class(cls),
            fault_type=ft, confidence=0.9,
            reason=f"injected fault_type={ft}, score={score:.2f}, dur={duration}")

    # --- 2. Hard-negative candidate: routine operational spike ---
    hint = _hard_negative_hint(bundle)
    if hint and score < 0.9:
        return make_label(
            eid, "weak_rule", "normal", "suppress",
            fault_type="none", confidence=0.6,
            reason=f"hard_negative candidate ({hint}); routine operational spike")

    # --- 3. Strong single-feature breach vs baseline ---
    lat, base_lat = feats.get("disk_latency_p95", 0.0), base.get("disk_latency_p95_mean_5m", 0.0)
    if score >= 0.8 and lat > max(base_lat * 3.0, 50.0):
        cls = "high" if score >= 0.9 else "medium"
        return make_label(
            eid, "weak_rule", cls, action_for_pure_class(cls),
            fault_type="disk_io_latency", confidence=0.7,
            reason=f"disk_latency_p95={lat:.1f} >> baseline {base_lat:.1f}")

    sysr, base_sys = feats.get("syscall_rate", 0.0), base.get("syscall_rate_mean_5m", 0.0)
    if score >= 0.8 and sysr > max(base_sys * 3.0, 100.0):
        cls = "high" if score >= 0.95 else "medium"
        return make_label(
            eid, "weak_rule", cls, action_for_pure_class(cls),
            fault_type="syscall_spike", confidence=0.7,
            reason=f"syscall_rate={sysr:.0f} >> baseline {base_sys:.0f}")

    # --- 4. Single-window spike -> wait (cheap to defer one window) ---
    if score >= 0.7 and duration <= 1:
        return make_label(
            eid, "weak_rule", "low", "wait",
            fault_type="unknown", confidence=0.55,
            reason=f"single-window spike score={score:.2f}; defer")

    # --- 5. Sustained moderate anomaly ---
    if score >= 0.75 and duration >= 3:
        ft = _infer_fault_type(feats, base)
        return make_label(
            eid, "weak_rule", "medium", "triage",
            fault_type=ft, confidence=0.6,
            reason=f"sustained anomaly dur={duration}, score={score:.2f}")

    # --- 6. Fallback by score band ---
    cls = "low" if score >= 0.7 else "normal"
    return make_label(
        eid, "weak_rule", cls, action_for_pure_class(cls),
        fault_type="none", confidence=0.5,
        reason=f"low-confidence fallback score={score:.2f}")


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def run(cases_path: Path, out_path: Path) -> Tuple[int, Dict[str, int]]:
    store = LabelStore(out_path)
    labels, mix = [], {}
    for bundle in _read_jsonl(cases_path):
        lbl = label_for_bundle(bundle)
        labels.append(lbl)
        mix[lbl["true_class"]] = mix.get(lbl["true_class"], 0) + 1
    store.append_many(labels)
    return len(labels), mix


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Weak-label case bundles.")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cases = Path(args.cases)
    if not cases.exists():
        print(f"weak_labeler: cases not found: {cases}")
        return 1
    n, mix = run(cases, Path(args.out))
    print(f"weak_labeler: wrote {n} labels -> {args.out}; class mix={mix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
