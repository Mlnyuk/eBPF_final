# Cost-sensitive triage pipeline

A supervised, cost-aware triage layer that turns anomaly events into one of
`suppress | wait | triage | alert` by minimising expected operational cost. It
extends the existing detector without changing the Isolation Forest / noise
filter / policy path.

> Not RL. See [`why_not_rl.md`](why_not_rl.md). GPU roles: see
> [`heterogeneous_gpu_label_factory.md`](heterogeneous_gpu_label_factory.md).

## Modules (`src/triage/`)

| Module | Role |
|---|---|
| `case_bundle.py` | Group window rows → labelable case bundles (9-feature view, rolling baseline, duration, fault metadata, per-model votes). |
| `label_store.py` | Append-only JSONL labels; resolve final label by source priority. |
| `weak_labeler.py` | Rule-based first-pass labels + hard-negative mining (bulk on V100). |
| `adjudicator_prompt.py` | System/user prompt + strict-JSON parser for the LLM. |
| `llm_adjudicator.py` | FastAPI `/adjudicate` + offline batch. **Ambiguous cases only.** Conservative fallback on LLM/parse failure. |
| `cost_matrix.py` | `COST_MATRIX` + expected-cost decision rule. |
| `train_cost_sensitive.py` | Train tree / RandomForest / optional XGBoost → joblib bundles. |
| `evaluate.py` | Cost-aware metrics + node/fault/time holdouts → Markdown report. |
| `infer.py` | Lightweight CPU inference + detector runtime (`from_env`, hot-swap). |

## Decision rule

The classifier predicts `P(true_class | x)` over `normal | low | medium | high`.
The action minimises expected cost:

```
expected_cost(action) = Σ_c  P(c | x) · COST_MATRIX[action][c]
recommended_action    = argmin_action expected_cost(action)
```

Default `COST_MATRIX` (lower is better) — silencing a real `high` is the most
expensive cell:

```
            normal  low  medium  high
suppress       0     1      6     20
wait         0.5     0      3     12
triage         1   0.5      0      4
alert          3     2      1      0
```

Decoupling class probability from cost means you can **retune costs without
retraining** (e.g. raise `suppress×high` if a high FN ever slips through).

## Schemas

* `schemas/case_bundle.schema.json`
* `schemas/triage_label.schema.json`

## End-to-end run (local, CPU)

```bash
export PYTHONPATH=$PWD

# 0. (demo) generate sample feature windows
python data/features/_generate_sample.py

# 1. case bundles
python -m src.triage.case_bundle \
  --input data/features/windows.jsonl --output data/cases/cases.jsonl

# 2. weak labels (+ hard-negative mining)
python -m src.triage.weak_labeler \
  --cases data/cases/cases.jsonl --out data/labels/labels.jsonl

# 3. (optional) LLM-adjudicate ambiguous cases only (skips if Qwen is down)
python -m src.triage.llm_adjudicator \
  --cases data/cases/cases.jsonl --out data/labels/labels.jsonl

# 4. train cost-sensitive models
python -m src.triage.train_cost_sensitive \
  --cases data/cases/cases.jsonl --labels data/labels/labels.jsonl --out models/

# 5. cost-aware evaluation report
python -m src.triage.evaluate \
  --cases data/cases/cases.jsonl --labels data/labels/labels.jsonl \
  --models models/ --out reports/triage_eval.md

# 6. LLM adjudicator service (RTX 5090 node)
uvicorn src.triage.llm_adjudicator:app --host 0.0.0.0 --port 8080
```

## Production detector integration

The detector loads the triage model (`TRIAGE_*` env, live dir wins) and exposes:

* `POST /triage` — score one feature vector → triage decision (CPU, no LLM).
* `POST /detect/batch` — each effective anomaly gains a `triage` block; response
  carries `triage_actions` counts.
* `/health` — `triage` block (enabled / ready / source / model).
* `/metrics` — `ebpf_triage_action_total{action=...}`.

Example `POST /triage` response:

```json
{
  "event_id": "db-0",
  "anomaly_score": 1.0,
  "true_class_prob": {"normal": 0.0, "low": 0.0, "medium": 0.0, "high": 1.0},
  "expected_cost": {"suppress": 20.0, "wait": 12.0, "triage": 4.0, "alert": 0.0},
  "recommended_action": "alert",
  "model": "triage_tree"
}
```

### Hot-swap promotion

`v100-train-eval` writes models to the shared workspace. To promote without an
image rebuild (same pattern as retrain/distill): copy `triage_tree.joblib` into
each detector pod's `/triage-live` hostPath, then `POST /reload`. The detector
prefers `TRIAGE_LIVE_DIR` over the baked `models/`.

## Kubernetes

```bash
kubectl apply -f manifests/gpu/llm-triage-adjudicator.yaml   # RTX 5090
kubectl apply -f manifests/gpu/v100-weak-label-worker.yaml   # V100
kubectl apply -f manifests/gpu/v100-train-eval-job.yaml      # V100
kubectl apply -f manifests/gpu/v100-replay-worker.yaml       # V100
```

## Constraints honoured

* No RL / DQN / PPO / Q-learning.
* LLM never on the production critical path (offline/batch only).
* CPU fallback throughout (XGBoost optional; absent → skipped gracefully).
* Existing pipeline untouched (IF / noise / policy still work).
* No secrets hard-coded (Telegram/registry via existing secrets).
