# Heterogeneous GPU label factory

The hardware is split by role so the scarce, high-quality GPU is reserved for the
one task that needs it (LLM adjudication), and the cheaper batch pool does the
high-volume label/eval work. Production inference uses **no GPU at all**.

```
                         ┌─────────────────────────────────────────┐
   eBPF collector ──►    │  feature windows (10s) ──► case bundles  │   (CPU)
                         └───────────────┬─────────────────────────┘
                                         │
                    weak labels + hard-negative mining  ──────────────  V100 pool
                                         │                               (batch-train)
                          ambiguous cases only
                                         │
                                         ▼
                         RTX 5090: LLM adjudicator  ────────────────────  RTX 5090
                         (Qwen vLLM, high-quality verdicts)               (llm-triage)
                                         │
                                   label store (priority-resolved)
                                         │
                       train cost-sensitive tree / RF / XGBoost  ───────  V100 pool
                                         │
                              expected-cost decision
                                         ▼
                  production detector: lightweight CPU inference  ──────  CPU only
```

## Roles

| Hardware | Role label | Workload | Needs CUDA? |
|---|---|---|---|
| **RTX 5090 ×1** | `gpu-role=llm-triage` | vLLM Qwen server + `llm_adjudicator` client. Adjudicates only **ambiguous** cases. | Yes (vLLM server) |
| **V100 ×4** | `gpu-role=batch-train` | fault replay, weak labeling, train/eval sweeps, hard-negative mining | Optional (sklearn is CPU; GPU only for XGBoost `hist` sweeps) |
| **CPU** | — | production `POST /triage` tree inference | No |

## Why this split

* **LLM is expensive, so ration it.** Sending every event to the 5090 would be
  slow and wasteful. The ambiguity gate (`src/triage/llm_adjudicator.py:
  ambiguity_reasons`) forwards only model-disagreement / mid-score / missing-
  metadata / novel-identity cases. Everything else gets a cheap consensus label.
* **V100s are plentiful, so batch on them.** Weak labeling and train/eval scale
  with corpus size, not difficulty — perfect for the cheaper pool.
* **Production must stay light.** The detector loads one joblib tree and does a
  single `predict_proba` + argmin-cost. No GPU, no LLM, no network on the hot
  path. The LLM adjudicator is strictly offline/batch.

## Label sources & priority

`src/triage/label_store.py` resolves multiple labels per `event_id` by source:

```
operator_feedback  >  llm_adjudicator  >  fault_metadata  >  weak_rule
```

Ground-truth injections beat heuristics; human feedback beats everything; the LLM
fills the uncertain middle.

## Node labels

See `manifests/gpu/node-labels.md` for the exact `kubectl label` commands.
