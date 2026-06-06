#!/usr/bin/env bash
# controlplane_pressure.sh — generate bounded Kubernetes API-server read load to
# surface controlplane_pressure (elevated syscall/net/ctx-switch on control nodes).
#
# DELIBERATELY CONSERVATIVE: read-only LIST calls at a low, capped QPS for a short
# window. It never writes/deletes objects and never touches etcd directly, so it
# cannot harm cluster state — it only adds gentle read traffic. Tune QPS/DURATION
# down further on a busy cluster.
#
# Run:    ./controlplane_pressure.sh [DURATION_SEC] [QPS]
#           DURATION default = 60
#           QPS      default = 5    (requests/sec; hard-capped at 20)
# Revert: nothing to revert — read-only, stops at DURATION.
set -euo pipefail

DURATION="${1:-60}"
QPS="${2:-5}"
[ "$QPS" -gt 20 ] && QPS=20   # safety cap

if ! command -v kubectl >/dev/null 2>&1; then
  echo "[controlplane_pressure] kubectl not found. Skipping." >&2
  exit 0
fi

sleep_s=$(awk "BEGIN{printf \"%.3f\", 1.0/$QPS}")
echo "[controlplane_pressure] read-only LIST load ~${QPS} qps for ${DURATION}s"
end=$(( $(date +%s) + DURATION ))
i=0
while [ "$(date +%s)" -lt "$end" ]; do
  # Rotate cheap read-only LISTs across resources; ignore errors/RBAC denials.
  case $(( i % 4 )) in
    0) kubectl get pods -A >/dev/null 2>&1 || true ;;
    1) kubectl get events -A >/dev/null 2>&1 || true ;;
    2) kubectl get nodes >/dev/null 2>&1 || true ;;
    3) kubectl get configmaps -A >/dev/null 2>&1 || true ;;
  esac
  i=$(( i + 1 ))
  sleep "$sleep_s"
done
echo "[controlplane_pressure] done (read-only; no state changed)"
