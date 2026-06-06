#!/usr/bin/env bash
# gpu_mem_stress.sh — bounded GPU memory + compute load to surface gpu_pressure.
#
# SAFE BY DESIGN: allocates only a fraction of free VRAM and runs briefly, so it
# does not OOM a neighbour's model server. No-ops cleanly when no GPU / torch is
# present (the V100/5090 image may not ship torch).
#
# Run:    ./gpu_mem_stress.sh [DURATION_SEC] [ALLOC_FRACTION]
#           DURATION       default = 45
#           ALLOC_FRACTION default = 0.25   (of currently free VRAM)
# Revert: process exits after DURATION, freeing all allocations.
set -euo pipefail

DURATION="${1:-45}"
FRACTION="${2:-0.25}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu_mem_stress] no nvidia-smi (no GPU on this node). Skipping." >&2
  exit 0
fi

if ! python3 - "$DURATION" "$FRACTION" <<'PY'
import sys, time
dur = float(sys.argv[1]); frac = float(sys.argv[2])
try:
    import torch
except Exception:
    raise SystemExit("torch not installed")
if not torch.cuda.is_available():
    raise SystemExit("cuda not available")
free, total = torch.cuda.mem_get_info()
nbytes = int(free * max(0.0, min(frac, 0.5)))   # cap at 50% free, never more
n = max(1, nbytes // 4)
print(f"[gpu_mem_stress] allocating ~{nbytes/1e9:.2f} GB ({frac:.0%} of free), busy {dur:.0f}s")
buf = torch.empty(n, dtype=torch.float32, device="cuda")
end = time.time() + dur
x = torch.randn(2048, 2048, device="cuda")
while time.time() < end:
    x = (x @ x).clamp_(-1, 1)          # keep the SMs busy
torch.cuda.synchronize()
print("[gpu_mem_stress] done (memory freed on exit)")
PY
then
  echo "[gpu_mem_stress] torch unavailable or alloc failed; skipping (no fault injected)." >&2
  exit 0
fi
