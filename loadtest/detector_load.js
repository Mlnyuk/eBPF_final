// k6 load test for the eBPF detector /detect/batch endpoint.
// Ramps virtual users to find the saturation point of the 2-replica detector,
// measuring latency percentiles, throughput, and error rate against the same
// payload shape the collectors push (a window of ~30 feature vectors).
import http from "k6/http";
import { check } from "k6";
import { Rate } from "k6/metrics";

const BATCH = Number(__ENV.BATCH || 30);
const URL = __ENV.TARGET || "http://ebpf-detector.ebpf-final:8080/detect/batch";

export const errorRate = new Rate("detect_errors");

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 20 },
        { duration: "30s", target: 50 },
        { duration: "30s", target: 100 },
        { duration: "30s", target: 200 },
        { duration: "30s", target: 300 },
        { duration: "20s", target: 0 },
      ],
      gracefulRampDown: "5s",
    },
  },
  thresholds: {
    http_req_duration: ["p(95)<500", "p(99)<1000"],
    http_req_failed: ["rate<0.01"],
    detect_errors: ["rate<0.01"],
  },
};

// Feature schema (order matches the model). Values drawn around the normal
// baseline with occasional spikes so the detector does real scoring work.
const FEATURES = [
  "syscall_read_rate", "syscall_write_rate", "syscall_open_rate",
  "tcp_connect_rate", "tcp_retransmit_rate",
  "network_rx_bytes", "network_tx_bytes",
  "disk_read_bytes", "disk_write_bytes", "disk_io_latency_ms",
  "process_exec_count", "process_fork_count", "context_switch_count",
  "cpu_utilization",
];

function vector(i) {
  const v = {
    timestamp: new Date().toISOString(),
    node: "loadtest",
    namespace: "bench",
    pod: `bench-${i}`,
    container: "c",
  };
  for (const f of FEATURES) {
    let base = Math.random() * 50;
    if (Math.random() < 0.05) base *= 100; // occasional spike
    v[f] = Number(base.toFixed(3));
  }
  return v;
}

export default function () {
  const items = [];
  for (let i = 0; i < BATCH; i++) items.push(vector(i));
  const res = http.post(URL, JSON.stringify({ items }), {
    headers: { "Content-Type": "application/json" },
    timeout: "10s",
  });
  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "has results": (r) => {
      try { return JSON.parse(r.body).count === BATCH; } catch (e) { return false; }
    },
  });
  errorRate.add(!ok);
}
