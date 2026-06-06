#!/usr/bin/env bash
# mem_stress.sh — allocate/churn memory to trigger memory_pressure anomalies.
#
# Tool: stress-ng
# Install: sudo apt-get install -y stress-ng
# Run:    ./mem_stress.sh [WORKERS] [BYTES_PER_WORKER] [DURATION_SEC]
#           WORKERS  default = 2
#           BYTES    default = 256M   (kept modest to avoid OOM-killing neighbours)
#           DURATION default = 60
# Revert: stress-ng exits on its own after DURATION. To stop early:
#           pkill -f stress-ng
set -euo pipefail

WORKERS="${1:-2}"
BYTES="${2:-256M}"
DURATION="${3:-60}"

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "stress-ng not found. Install: sudo apt-get install -y stress-ng" >&2
  exit 1
fi

echo "[mem_stress] $WORKERS vm workers x $BYTES for ${DURATION}s"
stress-ng --vm "$WORKERS" --vm-bytes "$BYTES" --vm-keep \
          --timeout "${DURATION}s" --metrics-brief
echo "[mem_stress] done (auto-reverted: process exited)"
