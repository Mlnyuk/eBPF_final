# eBPF_final — eBPF-based Kubernetes Anomaly Detection

An MVP system that collects Kubernetes runtime data with **eBPF** and performs
**unsupervised anomaly detection** with an **Isolation Forest** model.

> eBPF is the *data-collection layer only*. It counts kernel runtime events
> (syscalls, TCP, disk, scheduler) per cgroup and aggregates them into
> fixed-length feature windows. The anomaly detection itself is done entirely by
> the Isolation Forest, which only ever sees **window-level feature vectors**,
> never raw events.

---

## 1. Overview

- **Collect**: a privileged DaemonSet runs an eBPF (BCC) program on every node,
  attaching kprobes/tracepoints and counting runtime events keyed by cgroup.
- **Aggregate**: events are summed into 10-second windows and mapped to
  `namespace/pod/container`, producing one feature vector per window.
- **Train**: an Isolation Forest is fit on *normal-state* feature data.
- **Detect**: new feature vectors are scored; higher `anomaly_score` (0–1) means
  more anomalous. A FastAPI service exposes detection + Prometheus metrics.

## 2. Architecture

```
                       ┌──────────────────────── Kubernetes Node ───────────────────────┐
                       │                                                                 │
  kprobes/tracepoints  │   ┌───────────────┐   per-cgroup counts   ┌──────────────────┐  │
  syscalls/tcp/disk/   │   │  eBPF program  │ ───────────────────▶ │ feature_aggregator│  │
  sched  ───────────── │──▶│  (BCC, kernel) │                      │  10s windows      │  │
                       │   └───────────────┘                       └────────┬─────────┘  │
                       │            ▲  cgroup_id                             │ feature CSV/JSONL
                       │            │                                        │            │
                       │     ┌──────┴───────┐                                ▼            │
                       │     │  k8s_mapper   │  cgroup_id → ns/pod/container  data/        │
                       │     └──────────────┘                                             │
                       └─────────────────────────────────────────────────────────────────┘
                                                   │
                       feature vectors             ▼
                       ┌───────────────┐   train   ┌──────────────────┐   serve   ┌───────────────┐
                       │ normal data   │ ────────▶ │ IsolationForest   │ ───────▶ │ FastAPI /detect│
                       │ (results/*.csv)│          │ models/*.pkl      │          │ /metrics       │
                       └───────────────┘           └──────────────────┘          └──────┬────────┘
                                                                                          │ Prometheus
                                                                                          ▼ Grafana
```

## 3. Directory structure

```
eBPF_final/
├── README.md
├── requirements.txt
├── configs/config.yaml            # shared config (window, schema, model, ports)
├── collector/
│   ├── ebpf_collector.py          # BCC eBPF program + windowed feature emitter
│   ├── feature_aggregator.py      # schema, window aggregation, IO, mock generator
│   └── k8s_mapper.py              # cgroup_id -> namespace/pod/container
├── detector/
│   ├── train.py                   # fit IsolationForest on normal data
│   ├── detect.py                  # score feature files
│   ├── api.py                     # FastAPI: /health /detect /detect/batch /metrics
│   └── model_utils.py             # model bundle, scoring, normalization
├── models/                        # trained model bundle (.pkl)
├── data/                          # live collector output
├── results/                       # experiment feature sets + anomaly scores
├── scripts/                       # fault injection (cpu/net/disk/requests)
├── k8s/                           # namespace, rbac, configmap, daemonset, deploy, svc
└── docker/                        # Dockerfile.collector, Dockerfile.detector
```

## 4. Installation

### Local (detector / ML pipeline — no eBPF needed)

```bash
cd eBPF_final
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### eBPF collector host (Linux node, kernel ≥ 4.9)

BCC is installed via the system package manager (NOT pip):

```bash
sudo apt-get update
sudo apt-get install -y bpfcc-tools python3-bpfcc linux-headers-$(uname -r)
# optional pod mapping enrichment:
sudo pip3 install kubernetes
```

The collector must run as **root** (or with `CAP_BPF` + `CAP_SYS_ADMIN`).

## 5. Local run (full pipeline, no cluster)

The mock feature generator lets you exercise train → detect → API without eBPF.

```bash
# 1. Generate a normal baseline + a fault dataset
python collector/feature_aggregator.py -n 800 --out results/normal_features.csv
python collector/feature_aggregator.py -n 200 --fault packet_loss \
    --out results/packet_loss_features.csv

# 2. Train on normal data
python detector/train.py --data results/normal_features.csv

# 3. Detect on the fault dataset
python detector/detect.py --data results/packet_loss_features.csv \
    --out results/anomaly_scores.csv

# 4. Serve the API
uvicorn detector.api:app --host 0.0.0.0 --port 8080
curl localhost:8080/health
curl -X POST localhost:8080/detect -H 'content-type: application/json' \
     -d '{"pod":"p1","tcp_retransmit_rate":8,"network_rx_bytes":20000}'
curl localhost:8080/metrics
```

## 6. Collecting normal data (real eBPF)

Run the collector during known-good operation to build the training baseline:

```bash
sudo python3 collector/ebpf_collector.py --window 10 --out data/
# stage-1 mode without pod mapping (node/cgroup only):
sudo python3 collector/ebpf_collector.py --window 10 --out data/ --no-k8s
```

This writes `data/features.csv` and `data/features.jsonl`, one row per
(window, pod). Copy a clean run to `results/normal_features.csv` for training.

## 7. Isolation Forest training

```bash
python detector/train.py --data results/normal_features.csv
# tune the expected anomaly fraction:
python detector/train.py --data results/normal_features.csv --contamination 0.05
```

Produces `models/isolation_forest.pkl` (a `ModelBundle` containing the fitted
model, the feature order, score-normalization bounds, and the decision
threshold). Default `contamination=0.02`, overridable via `--contamination`.

## 8. Real-time detection

```bash
python detector/detect.py --data data/features.csv --out results/anomaly_scores.csv
python detector/detect.py --data data/features.jsonl --only-anomalies
```

Each result row contains: `timestamp, node, namespace, pod, container,
anomaly_score, is_anomaly, top_features`. `anomaly_score ∈ [0,1]`, **higher =
more anomalous**. `top_features` ranks the features that deviate most from the
training mean (z-score), a pragmatic proxy since Isolation Forest has no native
per-feature attribution.

## 9. Kubernetes deployment

Build + push images (replace the registry with yours), then apply manifests:

```bash
# build (requires docker/registry access)
docker build -f docker/Dockerfile.detector  -t ghcr.io/mlnyuk/ebpf-final-detector:latest .
docker build -f docker/Dockerfile.collector -t ghcr.io/mlnyuk/ebpf-final-collector:latest .
docker push ghcr.io/mlnyuk/ebpf-final-detector:latest
docker push ghcr.io/mlnyuk/ebpf-final-collector:latest

# deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/daemonset-ebpf-collector.yaml
kubectl apply -f k8s/deployment-detector.yaml
kubectl apply -f k8s/service-detector.yaml

kubectl -n ebpf-final get pods
kubectl -n ebpf-final port-forward svc/ebpf-detector 8080:8080
```

The collector DaemonSet runs **privileged** with `hostPID` and host mounts of
`/sys`, `/lib/modules`, `/usr/src` so BCC can compile and load BPF against the
running kernel. The detector Deployment needs no privileges.

## 10. Fault injection experiments

Each script prints its run + revert instructions; all are reversible.

| Script | Tool | Effect |
|--------|------|--------|
| `scripts/cpu_stress.sh` | stress-ng | context-switch / syscall spike |
| `scripts/network_delay.sh` | tc netem | RTT + retransmit increase |
| `scripts/packet_loss.sh` | tc netem | TCP retransmit spike |
| `scripts/disk_io_stress.sh` | fio | disk bytes + I/O latency spike |
| `scripts/abnormal_requests.sh` | hey/wrk | syscall + TCP + network flood |

Typical experiment loop:

```bash
# 1. collect normal -> results/normal_features.csv, then train
# 2. inject a fault while collecting:
sudo ./scripts/packet_loss.sh eth0 15      # start
#    ... let the collector capture windows ...
sudo ./scripts/packet_loss.sh revert eth0  # revert
# 3. detect on the captured fault window file
python detector/detect.py --data data/features.csv --out results/anomaly_scores.csv
```

## 11. Result interpretation

- `is_anomaly = anomaly_score >= threshold` (threshold stored in the model
  bundle; derived from the `1 - contamination` quantile of training scores).
- On the bundled mock datasets: the normal baseline flags ≈2% (matching
  `contamination=0.02`), while injected faults flag 75–100% of windows —
  confirming the pipeline separates normal from abnormal behavior.
- Inspect `top_features` to see *which* signals drove a detection (e.g.
  `tcp_retransmit_rate` for packet loss, `disk_io_latency_ms` for disk stress).

## 12. Prometheus / Grafana integration

The API exposes Prometheus metrics at `GET /metrics`:

| Metric | Type | Meaning |
|--------|------|---------|
| `ebpf_anomaly_scored_total` | counter | feature vectors scored |
| `ebpf_anomaly_detected_total` | counter | vectors flagged anomalous |
| `ebpf_anomaly_score_last` | gauge | most recent anomaly score |
| `ebpf_anomaly_score_max` | gauge | max score since start |

The Service carries `prometheus.io/scrape` annotations. Add a scrape job:

```yaml
scrape_configs:
  - job_name: ebpf-detector
    kubernetes_sd_configs: [{role: endpoints}]
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
```

In Grafana, graph `ebpf_anomaly_score_last` and alert on
`rate(ebpf_anomaly_detected_total[5m])`.

## 13. Limitations & future work

- **Network bytes** are TCP-only (via `tcp_sendmsg`/`tcp_cleanup_rbuf`); UDP and
  raw sockets are not counted.
- **Disk I/O attribution** to a cgroup is best-effort (the issuing task at
  `block_rq_issue` may be a kworker), so per-pod disk numbers are approximate.
- **cgroup→pod mapping** assumes cgroup v2 and CRI naming conventions; it falls
  back to `cg-<id>` when the kube API is unreachable.
- The model is a single global Isolation Forest. Future work: per-workload
  models, online/streaming retraining, richer per-feature attribution (e.g.
  SHAP), drift detection, and direct push of scores to Prometheus from the
  collector path.
```
