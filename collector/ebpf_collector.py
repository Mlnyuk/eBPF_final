#!/usr/bin/env python3
"""
ebpf_collector.py
=================
eBPF data-collection layer (MVP) for the eBPF_final anomaly detector.

eBPF here is ONLY a data-collection mechanism. It attaches kprobes/tracepoints
to the kernel, counts runtime events per cgroup, and every WINDOW seconds emits
one aggregated feature vector per (cgroup -> pod) into CSV/JSONL. The anomaly
detection itself is done later by the Isolation Forest (train.py/detect.py).

Collected signals (prompt section 1)
-------------------------------------
  syscall read/write/open rate   syscalls:sys_enter_{read,write,open,openat}
  process exec / fork count      sched:sched_process_{exec,fork}
  TCP connect count              kprobe tcp_v4_connect / tcp_v6_connect
  TCP retransmit count           kprobe tcp_retransmit_skb
  network rx / tx bytes          kprobe tcp_cleanup_rbuf / tcp_sendmsg
  disk read / write bytes        tracepoint block:block_rq_issue
  disk I/O latency               block_rq_issue -> block_rq_complete delta
  context switch count           sched:sched_switch

Every counter is keyed by bpf_get_current_cgroup_id() so userspace can map it to
a Kubernetes pod via k8s_mapper.

Requirements
------------
  Linux kernel >= 4.9 with BPF + a BCC install (apt: bpfcc-tools python3-bpfcc).
  Must run privileged / as root with CAP_BPF/CAP_SYS_ADMIN (see k8s DaemonSet).

Run
---
  sudo python3 collector/ebpf_collector.py --window 10 --out data/
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.feature_aggregator import (  # noqa: E402
    FEATURE_COLUMNS, FeatureRecord, load_config, write_csv, write_jsonl,
)
from collector.k8s_mapper import default_mapper  # noqa: E402

# --------------------------------------------------------------------------
# BPF program (C). One BPF_HASH per signal, keyed by cgroup_id (u64).
# --------------------------------------------------------------------------

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// per-cgroup counters
BPF_HASH(c_read,    u64, u64);
BPF_HASH(c_write,   u64, u64);
BPF_HASH(c_open,    u64, u64);
BPF_HASH(c_tcpconn, u64, u64);
BPF_HASH(c_tcpretx, u64, u64);
BPF_HASH(c_netrx,   u64, u64);   // bytes
BPF_HASH(c_nettx,   u64, u64);   // bytes
BPF_HASH(c_diskr,   u64, u64);   // bytes
BPF_HASH(c_diskw,   u64, u64);   // bytes
BPF_HASH(c_disklat, u64, u64);   // total ns
BPF_HASH(c_diskio,  u64, u64);   // completed io count
BPF_HASH(c_exec,    u64, u64);
BPF_HASH(c_fork,    u64, u64);
BPF_HASH(c_ctxsw,   u64, u64);

// disk in-flight start times, keyed by sector (best-effort matching)
BPF_HASH(disk_start, u64, u64);
// remember cgroup that issued a sector so completion can attribute bytes/lat
BPF_HASH(disk_cg, u64, u64);

// ---- syscalls -----------------------------------------------------------
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    u64 cg = bpf_get_current_cgroup_id();
    c_read.increment(cg);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_enter_write) {
    u64 cg = bpf_get_current_cgroup_id();
    c_write.increment(cg);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_enter_open) {
    u64 cg = bpf_get_current_cgroup_id();
    c_open.increment(cg);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_enter_openat) {
    u64 cg = bpf_get_current_cgroup_id();
    c_open.increment(cg);
    return 0;
}

// ---- process exec / fork ------------------------------------------------
TRACEPOINT_PROBE(sched, sched_process_exec) {
    u64 cg = bpf_get_current_cgroup_id();
    c_exec.increment(cg);
    return 0;
}
TRACEPOINT_PROBE(sched, sched_process_fork) {
    u64 cg = bpf_get_current_cgroup_id();
    c_fork.increment(cg);
    return 0;
}

// ---- context switch -----------------------------------------------------
TRACEPOINT_PROBE(sched, sched_switch) {
    u64 cg = bpf_get_current_cgroup_id();
    c_ctxsw.increment(cg);
    return 0;
}

// ---- TCP connect --------------------------------------------------------
int kprobe__tcp_v4_connect(struct pt_regs *ctx, struct sock *sk) {
    u64 cg = bpf_get_current_cgroup_id();
    c_tcpconn.increment(cg);
    return 0;
}
int kprobe__tcp_v6_connect(struct pt_regs *ctx, struct sock *sk) {
    u64 cg = bpf_get_current_cgroup_id();
    c_tcpconn.increment(cg);
    return 0;
}

// ---- TCP retransmit -----------------------------------------------------
int kprobe__tcp_retransmit_skb(struct pt_regs *ctx, struct sock *sk) {
    u64 cg = bpf_get_current_cgroup_id();
    c_tcpretx.increment(cg);
    return 0;
}

// ---- network bytes ------------------------------------------------------
// tx: tcp_sendmsg(sk, msg, size)
int kprobe__tcp_sendmsg(struct pt_regs *ctx, struct sock *sk,
                        struct msghdr *msg, size_t size) {
    u64 cg = bpf_get_current_cgroup_id();
    c_nettx.increment(cg, size);
    return 0;
}
// rx: tcp_cleanup_rbuf(sk, copied)
int kprobe__tcp_cleanup_rbuf(struct pt_regs *ctx, struct sock *sk, int copied) {
    if (copied <= 0) return 0;
    u64 cg = bpf_get_current_cgroup_id();
    c_netrx.increment(cg, (u64)copied);
    return 0;
}

// ---- block I/O ----------------------------------------------------------
TRACEPOINT_PROBE(block, block_rq_issue) {
    u64 cg = bpf_get_current_cgroup_id();
    u64 sector = args->sector;
    u64 ts = bpf_ktime_get_ns();
    disk_start.update(&sector, &ts);
    disk_cg.update(&sector, &cg);

    // attribute bytes at issue time (rwbs[0]: 'R' read, 'W' write)
    u64 nbytes = args->bytes;
    char rw = args->rwbs[0];
    if (rw == 'W' || rw == 'w') c_diskw.increment(cg, nbytes);
    else                        c_diskr.increment(cg, nbytes);
    return 0;
}
TRACEPOINT_PROBE(block, block_rq_complete) {
    u64 sector = args->sector;
    u64 *tsp = disk_start.lookup(&sector);
    u64 *cgp = disk_cg.lookup(&sector);
    if (tsp && cgp) {
        u64 delta = bpf_ktime_get_ns() - *tsp;
        u64 cg = *cgp;
        c_disklat.increment(cg, delta);
        c_diskio.increment(cg, 1);
        disk_start.delete(&sector);
        disk_cg.delete(&sector);
    }
    return 0;
}
"""

# Maps whose accumulated value is a per-second RATE feature.
_RATE_MAPS = {
    "c_read": "syscall_read_rate",
    "c_write": "syscall_write_rate",
    "c_open": "syscall_open_rate",
    "c_tcpconn": "tcp_connect_rate",
    "c_tcpretx": "tcp_retransmit_rate",
}
# Maps whose accumulated value is a window SUM feature.
_SUM_MAPS = {
    "c_netrx": "network_rx_bytes",
    "c_nettx": "network_tx_bytes",
    "c_diskr": "disk_read_bytes",
    "c_diskw": "disk_write_bytes",
    "c_exec": "process_exec_count",
    "c_fork": "process_fork_count",
    "c_ctxsw": "context_switch_count",
}


class EbpfCollector:
    def __init__(self, window: int, node: str, out_dir: str, out_format: str,
                 enable_k8s: bool):
        from bcc import BPF  # imported lazily so non-eBPF hosts can import module
        self.window = window
        self.node = node
        self.out_dir = Path(out_dir)
        self.out_format = out_format
        self.mapper = default_mapper() if enable_k8s else None
        self._running = True
        print("[ebpf] compiling + loading BPF program ...")
        self.bpf = BPF(text=BPF_PROGRAM)
        print("[ebpf] attached probes; collecting on node=" + node)

    def _drain(self) -> Dict[int, Dict[str, float]]:
        """Read every BPF map, clear it, and return {cgroup_id: {feature: val}}."""
        per_cg: Dict[int, Dict[str, float]] = {}

        def acc(mapname: str, feat: str, divide: float | None):
            table = self.bpf[mapname]
            for k, v in table.items():
                cg = int(k.value)
                val = float(v.value)
                if divide:
                    val /= divide
                per_cg.setdefault(cg, {})[feat] = val
            table.clear()

        for mapname, feat in _RATE_MAPS.items():
            acc(mapname, feat, divide=float(self.window))
        for mapname, feat in _SUM_MAPS.items():
            acc(mapname, feat, divide=None)

        # disk latency = total_ns / io_count -> ms
        lat = {int(k.value): float(v.value) for k, v in self.bpf["c_disklat"].items()}
        cnt = {int(k.value): float(v.value) for k, v in self.bpf["c_diskio"].items()}
        self.bpf["c_disklat"].clear()
        self.bpf["c_diskio"].clear()
        for cg, total_ns in lat.items():
            n = cnt.get(cg, 0.0)
            ms = (total_ns / n / 1e6) if n > 0 else 0.0
            per_cg.setdefault(cg, {})["disk_io_latency_ms"] = ms
        return per_cg

    def _to_records(self, per_cg: Dict[int, Dict[str, float]]):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        recs = []
        for cg, feats in per_cg.items():
            if self.mapper:
                namespace, pod, container = self.mapper.resolve(cg)
            else:
                namespace, pod, container = ("unknown", f"cg-{cg}", "unknown")
            # skip non-pod / system cgroups to keep feature set pod-focused
            if namespace in ("node", "unknown") and pod.startswith(("system", "cg-")):
                # still emit for stage-1 (node/cgroup) visibility, but tag node
                pass
            full = {c: float(feats.get(c, 0.0)) for c in FEATURE_COLUMNS}
            recs.append(FeatureRecord(
                timestamp=ts, node=self.node, namespace=namespace,
                pod=pod, container=container, features=full,
            ))
        return recs

    def _write(self, recs):
        if not recs:
            return
        if self.out_format in ("csv", "both"):
            write_csv(recs, self.out_dir / "features.csv")
        if self.out_format in ("jsonl", "both"):
            write_jsonl(recs, self.out_dir / "features.jsonl")

    def stop(self, *_):
        self._running = False

    def run(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        while self._running:
            time.sleep(self.window)
            per_cg = self._drain()
            recs = self._to_records(per_cg)
            self._write(recs)
            print(f"[ebpf] window emitted {len(recs)} feature rows "
                  f"-> {self.out_dir}")
        print("[ebpf] stopped.")


def main():
    cfg = load_config()
    ccfg = cfg.get("collector", {})
    wcfg = cfg.get("window", {})

    ap = argparse.ArgumentParser(description="eBPF feature collector (MVP).")
    ap.add_argument("--window", type=int, default=wcfg.get("seconds", 10))
    ap.add_argument("--out", default=ccfg.get("output_dir", "data"))
    ap.add_argument("--format", choices=["csv", "jsonl", "both"],
                    default=ccfg.get("output_format", "both"))
    ap.add_argument("--node", default=os.environ.get(
        ccfg.get("node_name_env", "NODE_NAME"), os.uname().nodename))
    ap.add_argument("--no-k8s", action="store_true",
                    help="disable cgroup->pod mapping (stage-1 mode)")
    args = ap.parse_args()

    enable_k8s = (not args.no_k8s) and ccfg.get("enable_k8s_mapping", True)

    try:
        collector = EbpfCollector(
            window=args.window, node=args.node, out_dir=args.out,
            out_format=args.format, enable_k8s=enable_k8s,
        )
    except ImportError:
        sys.exit("[ebpf] ERROR: bcc not installed. Install bpfcc-tools + "
                 "python3-bpfcc, and run as root. (Use the mock generator in "
                 "feature_aggregator.py for non-eBPF environments.)")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"[ebpf] ERROR loading BPF program: {exc}")

    collector.run()


if __name__ == "__main__":
    main()
