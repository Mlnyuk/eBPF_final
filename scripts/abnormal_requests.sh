#!/usr/bin/env bash
# abnormal_requests.sh — flood an HTTP target to spike syscall/TCP/network features.
#
# Tool: hey (preferred) or wrk
# Install: go install github.com/rakyll/hey@latest   (or apt for wrk)
# Run:    ./abnormal_requests.sh [URL] [DURATION_SEC] [CONCURRENCY]
#           URL         default = http://localhost:8080/health
#           DURATION    default = 60
#           CONCURRENCY default = 100
# Revert: load generator exits after DURATION; nothing persistent to undo.
set -euo pipefail

URL="${1:-http://localhost:8080/health}"
DURATION="${2:-60}"
CONCURRENCY="${3:-100}"

if command -v hey >/dev/null 2>&1; then
  echo "[abnormal_requests] hey -z ${DURATION}s -c ${CONCURRENCY} $URL"
  hey -z "${DURATION}s" -c "$CONCURRENCY" "$URL"
elif command -v wrk >/dev/null 2>&1; then
  echo "[abnormal_requests] wrk -t4 -c${CONCURRENCY} -d${DURATION}s $URL"
  wrk -t4 -c"$CONCURRENCY" -d"${DURATION}s" "$URL"
else
  echo "[abnormal_requests] no hey/wrk; using curl flood fallback"
  end=$(( $(date +%s) + DURATION ))
  while [[ $(date +%s) -lt $end ]]; do
    for _ in $(seq "$CONCURRENCY"); do curl -s -o /dev/null "$URL" & done
    wait
  done
fi
echo "[abnormal_requests] done (auto-reverted: load generator exited)"
