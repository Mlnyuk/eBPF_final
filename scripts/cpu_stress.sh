#!/usr/bin/env bash
# cpu_stress.sh — generate CPU load to trigger context-switch / syscall anomalies.
#
# Tool: stress-ng
# Install: sudo apt-get install -y stress-ng
#
# Run:    ./cpu_stress.sh [WORKERS] [DURATION_SEC]
#           WORKERS  default = number of CPUs
#           DURATION default = 60
# Revert: stress-ng exits on its own after DURATION. To stop early:
#           pkill -f stress-ng
set -euo pipefail

WORKERS="${1:-$(nproc)}"
DURATION="${2:-60}"

if ! command -v stress-ng >/dev/null 2>&1; then
  echo "stress-ng not found. Install: sudo apt-get install -y stress-ng" >&2
  exit 1
fi

echo "[cpu_stress] $WORKERS workers for ${DURATION}s"
stress-ng --cpu "$WORKERS" --timeout "${DURATION}s" --metrics-brief
echo "[cpu_stress] done (auto-reverted: process exited)"
