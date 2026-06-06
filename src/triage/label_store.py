#!/usr/bin/env python3
"""
label_store.py
==============
Append-only label storage with multi-source priority resolution.

The same ``event_id`` may receive labels from several sources. When building the
final training label, the highest-priority source wins:

    operator_feedback > llm_adjudicator > fault_metadata > weak_rule

Ties within a source break by recency (latest ``created_at``).

Storage is JSONL (``data/labels/labels.jsonl``) — append-only, human-greppable,
and safe to accumulate across V100 weak-labeling sweeps + LLM adjudications.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.triage.cost_matrix import CLASSES, ACTIONS, FAULT_TYPES, action_for_pure_class

LABEL_SOURCES: List[str] = [
    "weak_rule", "fault_metadata", "llm_adjudicator", "operator_feedback",
]
# Higher = wins during resolution.
SOURCE_PRIORITY: Dict[str, int] = {
    "weak_rule": 0, "fault_metadata": 1, "llm_adjudicator": 2, "operator_feedback": 3,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_label(event_id: str, label_source: str, true_class: str,
               recommended_action: Optional[str] = None,
               fault_type: str = "none", confidence: float = 0.5,
               reason: str = "", created_at: Optional[str] = None,
               error: Optional[str] = None) -> Dict:
    """Build a normalized, validated label dict."""
    if label_source not in SOURCE_PRIORITY:
        raise ValueError(f"unknown label_source: {label_source}")
    if true_class not in CLASSES:
        raise ValueError(f"unknown true_class: {true_class}")
    if recommended_action is None:
        recommended_action = action_for_pure_class(true_class)
    if recommended_action not in ACTIONS:
        raise ValueError(f"unknown recommended_action: {recommended_action}")
    if fault_type not in FAULT_TYPES:
        fault_type = "unknown"
    label = {
        "event_id": str(event_id),
        "label_source": label_source,
        "true_class": true_class,
        "recommended_action": recommended_action,
        "fault_type": fault_type,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": reason,
        "created_at": created_at or _now_iso(),
    }
    if error:
        label["error"] = error
    return label


class LabelStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ---- write ----
    def append(self, label: Dict) -> None:
        self.append_many([label])

    def append_many(self, labels: Iterable[Dict]) -> int:
        rows = list(labels)
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            for lbl in rows:
                fh.write(json.dumps(lbl) + "\n")
        return len(rows)

    # ---- read ----
    def read_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def resolve(self) -> Dict[str, Dict]:
        """event_id -> winning label by (source priority, then recency)."""
        best: Dict[str, Dict] = {}
        for lbl in self.read_all():
            eid = lbl.get("event_id")
            if not eid:
                continue
            cur = best.get(eid)
            if cur is None or _rank(lbl) > _rank(cur):
                best[eid] = lbl
        return best


def _rank(label: Dict):
    pr = SOURCE_PRIORITY.get(label.get("label_source", ""), -1)
    return (pr, str(label.get("created_at", "")))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect a triage label store.")
    ap.add_argument("--labels", required=True)
    args = ap.parse_args(argv)
    store = LabelStore(args.labels)
    raw = store.read_all()
    resolved = store.resolve()
    by_src = Counter(l.get("label_source") for l in raw)
    by_cls = Counter(l["true_class"] for l in resolved.values())
    print(f"labels: {len(raw)} rows, {len(resolved)} unique events")
    print("by source:", dict(by_src))
    print("resolved class mix:", dict(by_cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
