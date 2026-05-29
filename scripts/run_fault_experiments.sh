#!/usr/bin/env bash
# Run fault injection experiments one at a time.
# Each fault pod runs for ~3 detector windows (30s), then is deleted.
# Scores are captured from /metrics after each run.
#
# Usage:
#   chmod +x scripts/run_fault_experiments.sh
#   bash scripts/run_fault_experiments.sh

set -euo pipefail

NS=ebpf-final
DETECTOR_SVC="http://ebpf-detector.${NS}:8080"
WINDOW=30   # seconds to observe after pod starts
RESULTS_DIR="results/fault_experiments"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${RESULTS_DIR}/report_${TIMESTAMP}.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$REPORT"; }

# Use a debug pod on a worker node to reach the in-cluster service
PROXY_NODE=worker-1
PROXY_POD="fault-probe-$(date +%s)"

cleanup_proxy() {
  kubectl -n "$NS" delete pod "$PROXY_POD" --ignore-not-found=true >/dev/null 2>&1
}
trap cleanup_proxy EXIT

log "Starting probe pod on $PROXY_NODE for in-cluster curl access..."
kubectl run "$PROXY_POD" -n "$NS" \
  --image=curlimages/curl:latest \
  --restart=Never \
  --overrides="{\"spec\":{\"nodeSelector\":{\"kubernetes.io/hostname\":\"${PROXY_NODE}\"}}}" \
  --command -- sleep 600 >/dev/null
kubectl wait --for=condition=Ready pod/"$PROXY_POD" -n "$NS" --timeout=60s >/dev/null

fetch_metrics() {
  kubectl exec -n "$NS" "$PROXY_POD" -- \
    curl -sf "${DETECTOR_SVC}/metrics" 2>/dev/null || echo "metrics_unavailable"
}

FAULTS=(cpu-stress fork-bomb disk-stress network-flood syscall-flood)
FAULT_PODS=(fault-cpu-stress fault-fork-bomb fault-disk-stress fault-network-flood fault-syscall-flood)

log "=== Baseline metrics ==="
fetch_metrics | tee -a "$REPORT"

for i in "${!FAULTS[@]}"; do
  FAULT="${FAULTS[$i]}"
  POD="${FAULT_PODS[$i]}"

  log ""
  log "=== Fault: ${FAULT} ==="

  # Grab pre-fault metric snapshot
  PRE=$(fetch_metrics)
  PRE_SCORED=$(echo "$PRE" | grep ^ebpf_anomaly_scored_total | awk '{print $2}')
  PRE_ANOM=$(echo "$PRE"   | grep ^ebpf_anomaly_detected_total | awk '{print $2}')

  # Launch fault pod
  kubectl apply -f /root/eBPF_final/k8s/fault-injections.yaml -l "fault=${FAULT}" \
    >/dev/null 2>&1 || true

  log "  Pod ${POD} created. Observing ${WINDOW}s..."
  sleep "$WINDOW"

  # Grab post-fault metrics
  POST=$(fetch_metrics)
  POST_SCORED=$(echo "$POST" | grep ^ebpf_anomaly_scored_total  | awk '{print $2}')
  POST_ANOM=$(echo "$POST"   | grep ^ebpf_anomaly_detected_total | awk '{print $2}')
  LAST_SCORE=$(echo "$POST"  | grep ^ebpf_anomaly_score_last     | awk '{print $2}')
  MAX_SCORE=$(echo  "$POST"  | grep ^ebpf_anomaly_score_max      | awk '{print $2}')

  DELTA_SCORED=$((${POST_SCORED%.*} - ${PRE_SCORED%.*}))
  DELTA_ANOM=$((${POST_ANOM%.*}   - ${PRE_ANOM%.*}))

  log "  windows_scored=${DELTA_SCORED}  anomalies_detected=${DELTA_ANOM}"
  log "  last_score=${LAST_SCORE}  max_score=${MAX_SCORE}"
  if [[ $DELTA_SCORED -gt 0 ]]; then
    RATE=$(echo "scale=1; $DELTA_ANOM * 100 / $DELTA_SCORED" | bc 2>/dev/null || echo "?")
    log "  anomaly_rate=${RATE}%"
  fi

  # Clean up fault pod
  kubectl delete pod "$POD" -n "$NS" --ignore-not-found=true >/dev/null 2>&1
  log "  Pod ${POD} deleted. Cooling down 20s..."
  sleep 20
done

log ""
log "=== All experiments complete. Report: ${REPORT} ==="
