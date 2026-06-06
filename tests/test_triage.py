"""Unit tests for the cost-sensitive triage pipeline (src/triage).

Run:  PYTHONPATH=$PWD pytest tests/test_triage.py -q
Most tests are dependency-light; the train/infer tests skip if sklearn is absent.
"""
import json

import pytest

from src.triage import cost_matrix as cm
from src.triage import case_bundle as cb
from src.triage.label_store import LabelStore, make_label, SOURCE_PRIORITY
from src.triage.weak_labeler import label_for_bundle
from src.triage.adjudicator_prompt import parse_response
from src.triage.llm_adjudicator import ambiguity_reasons, adjudicate_bundle


# --------------------------- cost_matrix ---------------------------

def test_expected_cost_known_value():
    prob = {"normal": 0.0, "low": 0.0, "medium": 0.0, "high": 1.0}
    assert cm.expected_cost("suppress", prob) == 20.0
    assert cm.expected_cost("alert", prob) == 0.0


def test_recommended_action_argmin():
    prob = {"normal": 0.0, "low": 0.0, "medium": 0.0, "high": 1.0}
    action, costs = cm.recommended_action(prob)
    assert action == "alert"
    assert min(costs, key=costs.get) == "alert"


def test_action_for_pure_class_mapping():
    assert cm.action_for_pure_class("normal") == "suppress"
    assert cm.action_for_pure_class("low") == "wait"
    assert cm.action_for_pure_class("medium") == "triage"
    assert cm.action_for_pure_class("high") == "alert"


# --------------------------- case_bundle ---------------------------

def test_features_from_row_maps_14col():
    row = {"syscall_read_rate": 10, "syscall_write_rate": 5, "syscall_open_rate": 2,
           "tcp_retransmit_rate": 3, "network_rx_bytes": 100, "network_tx_bytes": 200,
           "disk_read_bytes": 1, "disk_write_bytes": 2, "disk_io_latency_ms": 9,
           "context_switch_count": 42, "anomaly_score": 0.8}
    f = cb.features_from_row(row)
    assert f["syscall_rate"] == 17          # 10+5+2
    assert f["tcp_retrans"] == 3
    assert f["disk_latency_p95"] == 9
    assert f["ctx_switch_rate"] == 42
    assert f["net_rx_bytes"] == 100
    assert f["anomaly_score"] == 0.8


def test_fault_metadata_inferred_from_source():
    fm = cb.fault_metadata_from_row({"source_file": "cpu_stress.sh", "is_injected": True})
    assert fm["is_injected"] is True
    assert fm["fault_type"] == "cpu_pressure"


def test_bundles_only_anomalous_with_duration():
    rows = []
    # two normal then three sustained anomalous windows, same identity
    for i, score in enumerate([0.1, 0.2, 0.9, 0.92, 0.95]):
        rows.append({"node": "n", "namespace": "d", "pod": "p", "container": "c",
                     "window_start": f"2026-06-01T00:00:{i*10:02d}+00:00",
                     "anomaly_score": score, "is_anomaly": score >= 0.7})
    bundles = list(cb.bundles_from_rows(rows, score_threshold=0.7))
    assert len(bundles) == 3                       # only the anomalous windows
    assert [b.duration_windows for b in bundles] == [1, 2, 3]
    # event_id stable for the same identity+window_start
    assert len(set(b.event_id for b in bundles)) == 3


# --------------------------- label_store ---------------------------

def test_label_store_priority_resolution(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl")
    eid = "evt1"
    store.append(make_label(eid, "weak_rule", "low"))
    store.append(make_label(eid, "fault_metadata", "medium"))
    store.append(make_label(eid, "llm_adjudicator", "high"))
    resolved = store.resolve()
    assert resolved[eid]["label_source"] == "llm_adjudicator"
    assert resolved[eid]["true_class"] == "high"
    # operator feedback outranks the LLM
    store.append(make_label(eid, "operator_feedback", "low"))
    assert store.resolve()[eid]["label_source"] == "operator_feedback"


def test_priority_order_constant():
    assert (SOURCE_PRIORITY["operator_feedback"] > SOURCE_PRIORITY["llm_adjudicator"]
            > SOURCE_PRIORITY["fault_metadata"] > SOURCE_PRIORITY["weak_rule"])


def test_make_label_default_action():
    lbl = make_label("e", "weak_rule", "high")
    assert lbl["recommended_action"] == "alert"
    with pytest.raises(ValueError):
        make_label("e", "weak_rule", "bogus_class")


# --------------------------- weak_labeler ---------------------------

def _bundle(**over):
    b = {"event_id": "e", "node": "n", "namespace": "d", "pod": "p", "container": "c",
         "features": {"anomaly_score": 0.5, "syscall_rate": 50, "tcp_retrans": 0,
                      "disk_read_bytes": 0, "disk_write_bytes": 0,
                      "disk_latency_p95": 5, "ctx_switch_rate": 200,
                      "net_rx_bytes": 0, "net_tx_bytes": 0},
         "baseline": {"anomaly_score_mean_5m": 0.2, "disk_latency_p95_mean_5m": 5,
                      "syscall_rate_mean_5m": 50},
         "duration_windows": 1, "fault_metadata": {"is_injected": False, "fault_type": "none"}}
    b.update(over)
    return b


def test_weak_label_injected_is_fault_metadata_source():
    b = _bundle(fault_metadata={"is_injected": True, "fault_type": "disk_io_latency"},
                features={**_bundle()["features"], "anomaly_score": 0.95})
    lbl = label_for_bundle(b)
    assert lbl["label_source"] == "fault_metadata"
    assert lbl["true_class"] in ("medium", "high")
    assert lbl["fault_type"] == "disk_io_latency"


def test_weak_label_hard_negative_jupyter():
    b = _bundle(pod="video-study-jupyter-0", container="jupyter",
                features={**_bundle()["features"], "anomaly_score": 0.8})
    lbl = label_for_bundle(b)
    assert lbl["true_class"] == "normal"
    assert lbl["recommended_action"] == "suppress"


def test_weak_label_single_window_spike_waits():
    b = _bundle(duration_windows=1,
                features={**_bundle()["features"], "anomaly_score": 0.72})
    lbl = label_for_bundle(b)
    assert lbl["true_class"] == "low"
    assert lbl["recommended_action"] == "wait"


# --------------------------- adjudicator ---------------------------

def test_parse_response_valid_and_invalid():
    txt = '```json\n{"true_class":"high","recommended_action":"alert",' \
          '"fault_type":"disk_io_latency","confidence":0.9,"reason":"x"}\n```'
    out = parse_response(txt)
    assert out["true_class"] == "high" and out["recommended_action"] == "alert"
    with pytest.raises(ValueError):
        parse_response("not json")
    with pytest.raises(ValueError):
        parse_response('{"true_class":"bogus","recommended_action":"alert"}')


def test_ambiguity_gate():
    # disagreeing votes -> ambiguous
    amb = _bundle(model_votes={"threshold": "suppress", "tree": "alert"})
    assert ambiguity_reasons(amb)
    # clear high-score, with fault metadata, consistent votes, has baseline -> not
    clear = _bundle(features={**_bundle()["features"], "anomaly_score": 0.95},
                    model_votes={"threshold": "alert", "weak_label": "alert"},
                    fault_metadata={"is_injected": True, "fault_type": "cpu_pressure"})
    assert not ambiguity_reasons(clear)


def test_adjudicate_non_ambiguous_skips_llm():
    clear = _bundle(features={**_bundle()["features"], "anomaly_score": 0.95},
                    model_votes={"threshold": "alert", "weak_label": "alert"},
                    fault_metadata={"is_injected": True, "fault_type": "cpu_pressure"})
    label, reasons = adjudicate_bundle(clear, force=False)
    assert reasons == []
    assert label["label_source"] == "weak_rule"   # consensus, no LLM call


# --------------------------- infer (needs sklearn) ---------------------------

def test_train_and_infer_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from src.triage.train_cost_sensitive import save_bundle
    from src.triage.infer import TriageModel

    # tiny separable dataset: high anomaly_score -> high class
    X, y = [], []
    for _ in range(20):
        X.append([0.95, 200, 0, 0, 0, 200, 0, 0, 0]); y.append("high")
        X.append([0.1, 10, 0, 0, 0, 5, 0, 0, 0]); y.append("normal")
    clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    clf.fit(np.asarray(X), y)
    p = tmp_path / "triage_tree.joblib"
    save_bundle(p, "triage_tree", clf)

    m = TriageModel.load(p)
    feats = {"anomaly_score": 0.95, "syscall_rate": 200, "ctx_switch_rate": 200}
    out = m.decide(feats, event_id="t1")
    assert out["recommended_action"] == "alert"
    assert set(out["true_class_prob"]) == set(cm.CLASSES)
    assert out["model"] == "triage_tree"
