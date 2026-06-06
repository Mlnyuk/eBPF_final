#!/usr/bin/env python3
"""
llm_adjudicator.py
==================
LLM triage adjudicator — runs on the RTX 5090 node, OFFLINE / batch only.

NEVER place this on the production detection critical path. The production
detector does lightweight CPU inference (detector/api.py:/triage); this service
exists purely to enrich labels for hard cases with a high-quality LLM verdict.

Two entry points:
* FastAPI ``POST /adjudicate`` — input: case bundle JSON; output: triage label.
  Only *ambiguous* cases are sent to the LLM; obvious cases get the consensus
  label without an LLM call. ``?force=true`` overrides the gate.
* CLI batch — scan a cases file, adjudicate only the ambiguous ones, append
  ``llm_adjudicator`` labels to the store (for V100/5090 nightly enrichment).

Robustness: if the LLM is unreachable or returns unparseable text, a CONSERVATIVE
fallback label is produced (biased toward triage, since a high-severity false
negative is the most expensive error) with an ``error`` field set.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.triage.adjudicator_prompt import build_messages, parse_response
from src.triage.label_store import LabelStore, make_label

# --- LLM endpoint config (env-overridable; mirrors scripts/qwen_triage.py) ---
QWEN_BASE = os.environ.get("QWEN_BASE", "http://qwen3-32b-5090.default:8000")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-32b")
MAX_TOKENS = int(os.environ.get("ADJ_MAX_TOKENS", "512"))
LLM_TIMEOUT = int(os.environ.get("ADJ_TIMEOUT", "120"))

# Ambiguity gate bands.
MID_LOW = float(os.environ.get("ADJ_MID_LOW", "0.5"))
MID_HIGH = float(os.environ.get("ADJ_MID_HIGH", "0.85"))
HIGH_SCORE = float(os.environ.get("ADJ_HIGH_SCORE", "0.85"))


# --------------------------------------------------------------------------
# Ambiguity gate
# --------------------------------------------------------------------------

def ambiguity_reasons(bundle: Dict) -> List[str]:
    """Why this case warrants LLM adjudication (empty list => not ambiguous)."""
    reasons: List[str] = []
    feats = bundle.get("features", {})
    score = float(feats.get("anomaly_score", 0.0))
    votes = {k: v for k, v in (bundle.get("model_votes") or {}).items() if v}
    fm = bundle.get("fault_metadata", {}) or {}
    base = bundle.get("baseline", {})

    if len(set(votes.values())) > 1:
        reasons.append(f"model disagreement: {votes}")
    if MID_LOW <= score <= MID_HIGH:
        reasons.append(f"mid-range anomaly_score={score:.2f}")
    if (not fm.get("is_injected")) and fm.get("fault_type", "none") in (None, "", "none") \
            and score >= HIGH_SCORE:
        reasons.append(f"high score={score:.2f} with no fault metadata")
    if "alert" in votes.values() and _looks_hard_negative(bundle):
        reasons.append("hard-negative candidate predicted as alert")
    if not base or all(float(v) == 0.0 for v in base.values()):
        reasons.append("novel identity: no baseline history")
    return reasons


def _looks_hard_negative(bundle: Dict) -> bool:
    ident = " ".join(str(bundle.get(k, "")) for k in
                     ("namespace", "pod", "container")).lower()
    hints = ("jupyter", "image-pull", "registry", "cron", "prometheus",
             "grafana", "coredns", "kaniko", "backup")
    return any(h in ident for h in hints)


def is_ambiguous(bundle: Dict) -> bool:
    return bool(ambiguity_reasons(bundle))


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------

def _llm_up() -> bool:
    try:
        urllib.request.urlopen(f"{QWEN_BASE}/health", timeout=10).read()
        return True
    except urllib.error.URLError:
        return False


def call_llm(bundle: Dict) -> Dict:
    """Query the LLM and return a parsed label dict. Raises on transport/parse
    failure (caller builds a fallback)."""
    payload = {
        "model": QWEN_MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "messages": build_messages(bundle),
    }
    req = urllib.request.Request(
        f"{QWEN_BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=LLM_TIMEOUT))
    txt = out["choices"][0]["message"]["content"]
    return parse_response(txt)


def _fallback_label(event_id: str, err: str) -> Dict:
    """Conservative label when the LLM can't be used: bias to triage so a real
    high-severity event is not silently suppressed."""
    return make_label(
        event_id, "llm_adjudicator", "medium", "triage",
        fault_type="unknown", confidence=0.2,
        reason="LLM fallback (conservative): " + err, error=err)


def adjudicate_bundle(bundle: Dict, force: bool = False) -> Tuple[Dict, List[str]]:
    """Return (label, reasons). If not ambiguous and not forced, return a
    consensus label WITHOUT calling the LLM."""
    eid = bundle.get("event_id", "")
    reasons = ambiguity_reasons(bundle)
    if not reasons and not force:
        consensus = (bundle.get("model_votes") or {}).get("weak_label") \
            or (bundle.get("model_votes") or {}).get("threshold") or "triage"
        # Map consensus action back to a representative class for storage.
        cls = {"suppress": "normal", "wait": "low",
               "triage": "medium", "alert": "high"}.get(consensus, "low")
        lbl = make_label(eid, "weak_rule", cls, consensus,
                         confidence=0.5, reason="non-ambiguous consensus; LLM skipped")
        return lbl, reasons
    try:
        parsed = call_llm(bundle)
        lbl = make_label(eid, "llm_adjudicator", parsed["true_class"],
                         parsed["recommended_action"], parsed["fault_type"],
                         parsed["confidence"], parsed["reason"])
    except Exception as e:  # noqa: BLE001 - any transport/parse failure -> fallback
        lbl = _fallback_label(eid, f"{type(e).__name__}: {e}")
    return lbl, reasons


# --------------------------------------------------------------------------
# FastAPI service
# --------------------------------------------------------------------------
try:
    from fastapi import FastAPI, Query
    from pydantic import BaseModel

    app = FastAPI(title="eBPF_final LLM triage adjudicator", version="1.0")

    class _Bundle(BaseModel):
        model_config = {"extra": "allow"}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm_base": QWEN_BASE, "llm_model": QWEN_MODEL,
                "llm_reachable": _llm_up()}

    @app.post("/adjudicate")
    def adjudicate(bundle: _Bundle, force: bool = Query(False)) -> dict:
        data = bundle.model_dump()
        label, reasons = adjudicate_bundle(data, force=force)
        return {"label": label, "ambiguous": bool(reasons),
                "ambiguity_reasons": reasons}
except ImportError:  # FastAPI optional for pure-CLI/offline use
    app = None


# --------------------------------------------------------------------------
# Offline batch CLI
# --------------------------------------------------------------------------

def _read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline LLM adjudication of ambiguous cases.")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True, help="label store to append to")
    ap.add_argument("--force", action="store_true", help="adjudicate every case")
    ap.add_argument("--limit", type=int, default=0, help="cap LLM calls (0=all)")
    args = ap.parse_args(argv)

    cases = Path(args.cases)
    if not cases.exists():
        print(f"adjudicator: cases not found: {cases}")
        return 1
    store = LabelStore(args.out)
    sent = skipped = 0
    labels: List[Dict] = []
    llm_ok = _llm_up()
    for bundle in _read_jsonl(cases):
        reasons = ambiguity_reasons(bundle)
        if not reasons and not args.force:
            skipped += 1
            continue
        if args.limit and sent >= args.limit:
            break
        if not llm_ok:
            labels.append(_fallback_label(bundle.get("event_id", ""),
                                          "LLM endpoint unreachable"))
        else:
            lbl, _ = adjudicate_bundle(bundle, force=True)
            labels.append(lbl)
        sent += 1
        time.sleep(float(os.environ.get("ADJ_SLEEP", "0")))
    store.append_many(labels)
    print(f"adjudicator: adjudicated {sent}, skipped {skipped} non-ambiguous, "
          f"llm_reachable={llm_ok} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
