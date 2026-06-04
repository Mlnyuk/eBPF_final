# Implementation Summary — eBPF Anomaly Detection + LLM Augmentation

Status snapshot: **2026-06-04**. Companion to `README.md` (base system).
This document covers the full stack with focus on the LLM-augmentation layer
(noise distillation, triage, nightly automation).

---

## 1. System at a Glance

```
eBPF collector  ─►  feature windows  ─►  Isolation Forest  ─►  anomaly_score
 (DaemonSet)         (10s, per-cgroup)     (detector pod)            │
                                                                     ▼
                                                       ┌─── runtime noise filter ───┐
                                                       │ distilled DecisionTree     │   NO LLM in hot path
                                                       │ suppress benign daemon     │
                                                       │ anomalies → effective set  │
                                                       └────────────┬───────────────┘
                                                                    ▼
                                          effective anomalies ─► Prometheus / alerts / triage
                                                                    ▲
                          ┌──── offline (night, GPU) ──────────────┘
                          │  Qwen3-32B labels noise  ─►  distill into tree  ─►  hot-swap into pods
                          │  Qwen3-32B triages effective anomalies  ─►  Telegram
                          └─────────────────────────────────────────────────────
```

**Key principle:** LLM never runs in the detection hot path. It runs *offline at
night* to (a) distill a cheap runtime noise filter and (b) triage the surviving
anomalies. The detector itself stays pure Isolation Forest + a tiny DecisionTree.

---

## 2. Components

| Layer | Artifact | Role |
|---|---|---|
| Collect | `collector-go/`, `k8s/daemonset-collector-go.yaml` | eBPF per-cgroup event counts → 10s feature windows |
| Detect | `detector/detect.py`, `detector/train.py`, `models/isolation_forest.pkl` | Isolation Forest scoring, `anomaly_score` 0–1 |
| Serve | `detector/api.py`, `k8s/deployment-detector.yaml` | FastAPI: `/detect`, `/detect/batch`, `/health`, `/reload`, `/metrics` |
| **Noise filter** | `detector/noise_filter.py`, `models/noise_filter.pkl` + `noise_baseline.json` | **Runtime suppression of benign anomalies (distilled tree)** |
| **Distill train** | `detector/train_noise_filter.py` | **Turn LLM labels → DecisionTree + metrics + gate inputs** |
| **LLM labeler** | `scripts/qwen_distill_label.py` | **Group day's anomalies, baseline-deviation prompt, strict-JSON verdicts (1–2 model agreement)** |
| **Distill pipeline** | `scripts/distill_pipeline.sh`, `k8s/cronjob-distill.yaml` | **Nightly: label → train → GATE → hot-swap → notify** |
| **Triage** | `scripts/qwen_triage.py`, `k8s/cronjob-triage.yaml` | **Ensemble LLM triage of effective anomalies → Telegram** |
| Retrain | `detector/retrain.py`, `scripts/retrain_pipeline.sh`, `k8s/cronjob-retrain.yaml` | Drift adaptation of the Isolation Forest |
| Alerting | `k8s/prometheusrule-detector.yaml`, `k8s/alertmanagerconfig-telegram.yaml` | Effective-ratio alerts → Telegram |

---

## 3. Runtime Noise Filter (the core add)

**Problem:** Isolation Forest flagged ~33% of windows as anomalies — most were
benign daemon noise (kube-proxy, longhorn, node-exporter spiking on normal
cycles). Too noisy to alert on.

**Solution:** distill an LLM's noise judgement into a lightweight tree that runs
inline, so the LLM is never needed at detection time.

- `detector/noise_filter.py` — `NoiseFilter` loads `{model, cols, feats}` pkl +
  `noise_baseline.json` (per-container median normal feature values).
  - Feature vector for the tree = `[anomaly_score, has_baseline, *log10((obs+eps)/(base+eps))]`
    → judges **deviation from that container's normal**, not absolute magnitude.
  - `annotate(inputs, results)` adds `suppressed` + `suppress_proba` to each
    `is_anomaly` row; returns count suppressed.
  - `from_env()` prefers live dir (`NOISE_FILTER_LIVE_DIR=/noise-live`, hot-swap)
    over the baked-in model; no-ops gracefully if artifacts missing.
- Wired into `detector/api.py`: `/detect` + `/detect/batch` set
  `effective_anomaly = is_anomaly AND NOT suppressed`. `/health` + `/reload`
  report filter status. New metrics below.
- Threshold `NOISE_FILTER_THRESHOLD=0.99` (suppress only when very confident).

### Metrics
```
ebpf_anomaly_detected_total     # raw IF anomalies
ebpf_anomaly_suppressed_total   # silenced by noise filter
ebpf_anomaly_effective_total    # detected - suppressed  ← alerts key off this
```
`prometheusrule-detector.yaml` ratio alerts use `effective_total`, so benign
noise no longer pages.

---

## 4. Nightly Distill Automation (`distill_pipeline.sh`)

CronJob `0 2 * * *` UTC (`k8s/cronjob-distill.yaml`, SA `ebpf-distill` with
pods + pods/exec). Self-skips (exit 0) if Qwen is offline → daytime firings are
harmless.

Steps:
1. **Label in-pod** — exec into a detector pod (archive is local + Qwen reachable),
   run `qwen_distill_label.py`: group today's anomalies, inject per-container
   baseline + deviations, batch strict-JSON to the model(s).
   - 1 model (32B) or 2-model agreement when `QWEN2_BASE` serves: benign only if
     **all** models say benign, else suspicious (conservative).
2. **Aggregate** anomaly + normal rows → training corpus (weak supervision).
3. **Distill** → `train_noise_filter.py` fits a DecisionTree
   (`max_depth=5, min_leaf=50, balanced`), 5-fold CV, writes metrics JSON.
4. **GATE** — promote only if `false_suppress ≤ FALSE_SUPPRESS_FLOOR` (default
   0.01) and `n_labels ≥ MIN_LABELS` (20). Otherwise reject, keep old filter.
5. **Hot-swap** — stream new `noise_filter.pkl` + baseline into each pod's
   `/noise-live`, `POST /reload`. No restart.
6. **Telegram** verdict.

### Path portability (2026-06-04)
Script defaults assume the in-image paths (`/app/...`). For host/manual runs,
override:
```bash
LABELER_SRC=/root/eBPF_final/scripts/qwen_distill_label.py
TRAINER_SRC=/root/eBPF_final/detector/train_noise_filter.py
```
Both default to `/app/...` so the CronJob image is unaffected.

---

## 5. Ensemble Triage (`qwen_triage.py`)

CronJob `0 11,15,21 * * *` UTC. Triages **effective** anomalies (post-filter).
- `ask_model(base, model, …)` → `{(node,container,trigger): {verdict, confidence, cause, check}}`.
- `ensemble_triage()` queries each available model; consensus **suspicious if any
  model says suspicious**; flags `⚠️DISAGREE(m1=v1 m2=v2)` on splits.
- Per-container baseline injected so verdicts judge deviation, stable across runs.
- Output → Telegram with model list + disagreement count.

---

## 6. LLM Serving (Qwen3-32B-AWQ)

- `qwen3-32b-5090` Deployment — `Qwen/Qwen3-32B-AWQ`, vLLM 0.10.2, served name
  `qwen3-32b`, TP1, AWQ, `--max-model-len 32768`, `--gpu-memory-utilization 0.95`,
  pinned to node `gpu-3` (RTX 5090, 32GB). Model from hostPath
  `/data/models/qwen3-32b-awq`. Cold start ~90s.
- Thinking mode disabled (`chat_template_kwargs enable_thinking:false`); strict
  JSON via `response_format json_object`, fence-strip on parse.
- **5090-only by decision.** The 14B-on-V100 path was abandoned: vLLM 0.10.2
  XFormers prefix-prefill OOMs/hangs on Volta (V100) regardless of max-model-len.
  `qwen25-14b-v100` RayCluster left suspended; `QWEN2_BASE` empty (single-model).

---

## 7. Latest Validated Run (2026-06-04)

Manual full-pipeline run against live 32B:

| Metric | Value |
|---|---|
| LLM labels (32B, 3 batches) | 53 |
| Weak-labeled training rows | 13,028 (of 53,301 total) |
| **false_suppress (CV)** | **0.055%** (floor 1%) ✓ |
| **noise_reduced (CV)** | **95.9%** |
| Gate | PROMOTE |
| Hot-swap | detector 2/2 pods reloaded |
| Live cumulative | detected 63,516 → suppressed 37,726 (59.4%) → effective 25,790 |

Filter live: `enabled=true`, source `/noise-live/noise_filter.pkl`,
75 baselined containers, threshold 0.99.

---

## 8. Ops Runbook

**Manual distill run (host):**
```bash
cd /root/eBPF_final
NODEIP=$(kubectl get node gpu-3 -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
PATH="/root/eBPF_final/.venv/bin:$PATH" \
QWEN_BASE="http://$NODEIP:30134" QWEN_MODEL=qwen3-32b NAMESPACE=ebpf-final \
WORK_DIR=/tmp/distill-work \
LABELER_SRC=/root/eBPF_final/scripts/qwen_distill_label.py \
TRAINER_SRC=/root/eBPF_final/detector/train_noise_filter.py \
bash scripts/distill_pipeline.sh
```
(Host can't resolve cluster DNS → use the NodePort `:30134`; venv has sklearn 1.8.)

**Bring 32B up / down (shares gpu-3 with other users):**
```bash
kubectl scale deploy qwen3-32b-5090 --replicas=1   # up  (~90s ready)
kubectl scale deploy qwen3-32b-5090 --replicas=0   # down (frees gpu-3)
```

**Check filter live status:**
```bash
POD=$(kubectl get pod -n ebpf-final -l app=ebpf-detector -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ebpf-final $POD -c detector -- python3 -c \
 'import urllib.request,json;print(json.load(urllib.request.urlopen("http://localhost:8080/health"))["noise_filter"])'
```
(detector API = port **8080**, no curl in image — use python urllib.)

---

## 9. Constraints / Notes

- **GPU sharing:** gpu-3 (5090) is shared. The 32B is brought up only for
  offline LLM work, then scaled to 0 to return the GPU. V100s (gpu-1/gpu-2)
  unused for this project — Volta incompatible with the 14B vLLM path.
- **No LLM dependency at runtime:** if Qwen is down, detection + noise filter
  keep working on the last promoted tree; only nightly refresh/triage pause.
- **Secrets:** Telegram token via mounted secret file
  (`TELEGRAM_TOKEN_FILE`); never logged or dumped.
- **Persistence:** archives + `/noise-live` via hostPath (no RWX PVC).
