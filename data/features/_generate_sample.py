#!/usr/bin/env python3
"""Deterministic sample feature-window generator for the triage pipeline demo.

Writes data/features/windows.jsonl: a mix of normal traffic, injected faults
(disk/cpu/network), routine hard-negative spikes (jupyter, cron), and isolated
single-window spikes. Rows use the 14-col collector schema + anomaly_score +
is_anomaly, plus injection metadata on fault rows.

    python data/features/_generate_sample.py
"""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parent / "windows.jsonl"
T0 = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def base_feats(scale=1.0):
    return {
        "syscall_read_rate": random.uniform(20, 80) * scale,
        "syscall_write_rate": random.uniform(10, 50) * scale,
        "syscall_open_rate": random.uniform(5, 20) * scale,
        "tcp_connect_rate": random.uniform(0, 5),
        "tcp_retransmit_rate": random.uniform(0, 0.5),
        "network_rx_bytes": random.uniform(1e4, 1e5) * scale,
        "network_tx_bytes": random.uniform(1e4, 1e5) * scale,
        "disk_read_bytes": random.uniform(1e4, 1e5),
        "disk_write_bytes": random.uniform(1e4, 1e5),
        "disk_io_latency_ms": random.uniform(1, 8),
        "process_exec_count": random.uniform(0, 3),
        "process_fork_count": random.uniform(0, 3),
        "context_switch_count": random.uniform(100, 500),
        "cpu_utilization": random.uniform(5, 35),
    }


def row(t, node, ns, pod, container, feats, score, injected=False,
        fault_type="none", source=None):
    r = {
        "timestamp": t.isoformat(),
        "window_start": t.isoformat(),
        "window_end": (t + timedelta(seconds=10)).isoformat(),
        "cluster": "lab",
        "node": node, "namespace": ns, "pod": pod, "container": container,
        "anomaly_score": round(score, 3),
        "is_anomaly": bool(score >= 0.7),
        **{k: round(v, 3) for k, v in feats.items()},
    }
    if injected:
        r.update({"is_injected": True, "fault_type": fault_type, "source_file": source,
                  "fault_start": t.isoformat()})
    return r


def main():
    rows = []
    # --- normal baseline traffic across 3 containers/nodes ---
    for ci, (node, ns, pod, cont) in enumerate([
            ("gpu-1", "default", "web-1", "web"),
            ("worker-1", "default", "api-1", "api"),
            ("worker-2", "kube-system", "containerd-1", "containerd")]):
        t = T0 + timedelta(minutes=ci)
        for i in range(40):
            rows.append(row(t + timedelta(seconds=10 * i), node, ns, pod, cont,
                            base_feats(), random.uniform(0.05, 0.45)))

    # --- injected disk_io_latency fault (sustained, high latency) ---
    t = T0 + timedelta(minutes=10)
    for i in range(8):
        f = base_feats()
        f["disk_io_latency_ms"] = random.uniform(120, 300)
        f["disk_read_bytes"] *= 4
        rows.append(row(t + timedelta(seconds=10 * i), "gpu-2", "default",
                        "db-0", "postgres", f, random.uniform(0.9, 0.99),
                        injected=True, fault_type="disk_io_latency",
                        source="disk_io_stress.sh"))

    # --- injected cpu_pressure fault ---
    t = T0 + timedelta(minutes=14)
    for i in range(7):
        f = base_feats(scale=1.2)
        f["context_switch_count"] = random.uniform(5000, 12000)
        f["cpu_utilization"] = random.uniform(85, 99)
        rows.append(row(t + timedelta(seconds=10 * i), "gpu-3", "default",
                        "batch-0", "trainer", f, random.uniform(0.88, 0.98),
                        injected=True, fault_type="cpu_pressure",
                        source="cpu_stress.sh"))

    # --- injected tcp_retrans / network fault ---
    t = T0 + timedelta(minutes=18)
    for i in range(6):
        f = base_feats()
        f["tcp_retransmit_rate"] = random.uniform(5, 20)
        rows.append(row(t + timedelta(seconds=10 * i), "worker-3", "default",
                        "proxy-0", "envoy", f, random.uniform(0.86, 0.95),
                        injected=True, fault_type="tcp_retrans",
                        source="network_delay.sh"))

    # --- hard negative: jupyter startup spike (high-ish score, benign) ---
    t = T0 + timedelta(minutes=22)
    for i in range(5):
        f = base_feats(scale=2.5)
        rows.append(row(t + timedelta(seconds=10 * i), "worker-1", "default",
                        "video-study-jupyter-0", "jupyter", f,
                        random.uniform(0.74, 0.86)))

    # --- hard negative: cron image-pull spike ---
    t = T0 + timedelta(minutes=25)
    for i in range(3):
        f = base_feats(scale=2.0)
        rows.append(row(t + timedelta(seconds=10 * i), "infra-1", "default",
                        "backup-cron-1", "cron", f, random.uniform(0.72, 0.82)))

    # --- isolated single-window spikes (benign blips) ---
    for k in range(6):
        t = T0 + timedelta(minutes=30 + k)
        f = base_feats(scale=1.8)
        rows.append(row(t, "gpu-1", "default", f"job-{k}", "worker", f,
                        random.uniform(0.71, 0.8)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
