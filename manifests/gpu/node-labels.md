# GPU node labels — heterogeneous label factory

The triage label factory separates GPU roles so the expensive RTX 5090 only does
LLM adjudication and the V100 pool does batch label/eval work.

| Role label | Node(s) | Job |
|---|---|---|
| `gpu-role=llm-triage`, `gpu-model=rtx5090` | RTX 5090 | LLM adjudicator (talks to the vLLM Qwen server) |
| `gpu-role=batch-train`, `gpu-model=v100`   | V100 x4  | fault replay, weak labeling, train/eval sweeps, hard-negative mining |

## Apply the labels

```bash
# RTX 5090 — high-quality LLM triage / adjudication
kubectl label node gpu-5090 gpu-role=llm-triage gpu-model=rtx5090 --overwrite

# V100 pool — batch train / eval / replay / weak labeling
kubectl label node gpu-v100-1 gpu-role=batch-train gpu-model=v100 --overwrite
kubectl label node gpu-v100-2 gpu-role=batch-train gpu-model=v100 --overwrite
kubectl label node gpu-v100-3 gpu-role=batch-train gpu-model=v100 --overwrite
kubectl label node gpu-v100-4 gpu-role=batch-train gpu-model=v100 --overwrite
```

> On this cluster the GPU nodes are `gpu-1`/`gpu-2` (V100) and `gpu-3` (RTX 5090).
> Substitute the real node names. Verify with `kubectl get nodes -L gpu-role,gpu-model`.

## Notes

* The **LLM adjudicator pod itself is a thin HTTP client** — the GPU is consumed
  by the separate vLLM Qwen server on the 5090, not by the adjudicator. So the
  adjudicator does not request `nvidia.com/gpu`.
* The **weak labeler / train / eval jobs are CPU-bound sklearn** and have a CPU
  fallback; they are pinned to the V100 pool for scheduling/isolation, not because
  they need CUDA. Uncomment the `nvidia.com/gpu` request only if you enable the
  XGBoost GPU `hist` path on a sweep.
