# Detector load test — results

Load testing the detector `/detect/batch` endpoint with k6, run in-cluster
against the `ebpf-detector` Service (load-balances both HA replicas). Payload =
a 30-vector batch, the same shape the collectors push each window.

Run: 2026-06-02, detector = 2 replicas, **1 CPU / 1Gi limit each** (pre-tuning).

## 1. Ramp to saturation (`detector_load.js`, 0→300 VUs over ~2.5 min)

| metric | value |
|---|---|
| throughput | 23.7 req/s (≈710 vectors/s) |
| http_req_duration avg | 4.61 s |
| p95 / p99 | ~10 s (timeout cap) |
| errors (timeouts) | 16.6 % (668 / 4027) |
| single-request floor | 38 ms (min) |

Both replicas pinned at **1001m = the 1-core CPU limit** for the whole run;
memory stayed ~180Mi. → **CPU-bound saturation.** A single batch is fast
(38 ms); collapse is purely concurrency throttled by the CPU cap.

## 2. Sustained rate within SLO (`k6_steady.js`, constant 15 req/s, 60 s)

| metric | value |
|---|---|
| throughput | 15.0 req/s (450 vectors/s) |
| checks succeeded | 100 % (901/901) |
| p95 / p99 | 66.8 ms / <227 ms |
| errors / dropped | 0 / 0 |

## Interpretation

| | req/s | vectors/s |
|---|---|---|
| real production load | ~0.9 (9 collectors × 1 batch / 10 s) | ~27 |
| SLO-safe sustained | ≥15 | 450 |
| saturation (2×1 core) | ~24 | ~710 |

Real load sits ~16× below the SLO-safe rate and ~26× below saturation — the
steady state is comfortable. The risk is **burst concurrency**: with only 2
fixed replicas at a 1-core cap, a spike throttles into multi-second latencies
and timeouts.

## Actions taken

- **HPA** (`k8s/hpa-detector.yaml`): scale 2→6 on 70 % CPU so bursts add
  replicas instead of throttling; floor 2 preserves HA + PDB.
- **Right-sized requests/limits** (`k8s/deployment-detector.yaml`): cpu
  request 100m→500m (so HPA utilization is meaningful), limit 1→2 (per-pod
  burst headroom); memory 384Mi/1Gi → 256Mi/512Mi (observed peak ~180Mi).

## Reproduce

```bash
kubectl -n ebpf-final create configmap k6-detector-script \
  --from-file=detector_load.js=loadtest/detector_load.js
# then run a Job with image grafana/k6:latest, args: run /scripts/detector_load.js
```
