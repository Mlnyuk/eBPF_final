#!/usr/bin/env bash
# Fault experiment v2: run 5 strong continuous faults, capture per-fault labeled
# feature rows from each node's collector CSV (relies on cgroup→pod mapping so
# fault pods appear as pod=fault-<type>), and dump detector metrics.
#
# Output:
#   results/labeled/fault_<type>.csv     # positive samples per fault
#   results/labeled/normal_sample.csv    # negative samples (non-fault pods, same window)
#   results/fault_experiment_v2_<ts>.txt # summary report
set -uo pipefail

NS=ebpf-final
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/eBPF_final/results
LAB=$OUT/labeled
mkdir -p "$LAB"
REPORT=$OUT/fault_experiment_v2_${TS}.txt
RUN_SECONDS=320   # stress-ng runs 300s + apt install lead time

declare -A NODE=( [cpu-stress]=worker-1 [fork-bomb]=worker-2 [disk-stress]=worker-3 \
                  [network-flood]=infra-1 [syscall-flood]=infra-2 )
FAULTS=(cpu-stress fork-bomb disk-stress network-flood syscall-flood)

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$REPORT"; }

cpod(){ kubectl -n $NS get pods -l app=ebpf-collector --field-selector spec.nodeName=$1 \
        --no-headers 2>/dev/null | awk '{print $1}'; }

log "=== Fault experiment v2 — $TS ==="
log "Launching all 5 fault pods (continuous 300s)..."
START_EPOCH=$(date -u +%s)
kubectl apply -f /root/eBPF_final/k8s/fault-injections.yaml >/dev/null 2>&1

log "Waiting ${RUN_SECONDS}s for faults to run..."
sleep "$RUN_SECONDS"
END_EPOCH=$(date -u +%s)

# ISO8601 window bounds (UTC) — skip first 60s (apt install) for cleaner positives
WIN_START=$(date -u -d "@$((START_EPOCH+60))" +%Y-%m-%dT%H:%M:%SZ)
WIN_END=$(date -u -d "@${END_EPOCH}" +%Y-%m-%dT%H:%M:%SZ)
log "Capture window: $WIN_START .. $WIN_END"

log ""
log "=== Per-fault labeled capture ==="
for f in "${FAULTS[@]}"; do
  node=${NODE[$f]}
  pod=$(cpod "$node")
  # pull rows for the fault pod (pod name == fault-<f>) within the time window
  # NB: kubectl exec injects CR (\r) into output; tr -d '\r' BEFORE appending the
  # label or pandas splits each row at the embedded CR (doubled rows, NaNs).
  rows=$(kubectl -n $NS exec "$pod" -- sh -c \
    "awk -F, -v s='$WIN_START' -v e='$WIN_END' -v p='fault-$f' \
       '\$1>=s && \$1<=e && \$4==p' /data/features.csv" 2>/dev/null | tr -d '\r')
  n=$(printf '%s\n' "$rows" | grep -c . )
  if [ "$n" -gt 0 ]; then
    { printf '%s\n' "$rows" | sed "s/\$/,$f/"; } > "$LAB/fault_${f}.csv"
    log "  $f ($node): captured $n rows -> labeled/fault_${f}.csv"
  else
    log "  $f ($node): WARN 0 rows captured (mapping? pod name?)"
  fi
done

# Negative sample: non-fault pod rows on the same nodes, same window
log ""
log "=== Normal sample capture (same window, non-fault pods) ==="
: > "$LAB/normal_sample.csv"
for node in worker-1 worker-2 worker-3 infra-1 infra-2; do
  pod=$(cpod "$node")
  kubectl -n $NS exec "$pod" -- sh -c \
    "awk -F, -v s='$WIN_START' -v e='$WIN_END' \
       '\$1>=s && \$1<=e && \$4 !~ /^fault-/ && \$4 != \"pod\"' /data/features.csv" \
    2>/dev/null | tr -d '\r' | sed 's/$/,normal/' >> "$LAB/normal_sample.csv"
done
nn=$(grep -c . "$LAB/normal_sample.csv")
log "  normal: captured $nn rows -> labeled/normal_sample.csv"

log ""
log "=== Detector metrics ==="
dpod=$(kubectl -n $NS get pods -l app=ebpf-detector --no-headers | awk '{print $1}')
kubectl -n $NS exec "$dpod" -- python3 -c \
  "import urllib.request;print(urllib.request.urlopen('http://localhost:8080/metrics').read().decode())" \
  2>&1 | tee -a "$REPORT"

log ""
log "=== Per-node anomaly windows during fault ==="
for f in "${FAULTS[@]}"; do
  node=${NODE[$f]}; pod=$(cpod "$node")
  a=$(kubectl -n $NS logs "$pod" --since=${RUN_SECONDS}s 2>/dev/null | grep "pushed" \
        | awk -F'anomalies=' '{s+=$2} END{print s+0}')
  w=$(kubectl -n $NS logs "$pod" --since=${RUN_SECONDS}s 2>/dev/null | grep -c "pushed")
  log "  $f ($node): windows=$w anomaly_count=$a"
done

log ""
log "Cleaning up fault pods..."
kubectl -n $NS delete -f /root/eBPF_final/k8s/fault-injections.yaml >/dev/null 2>&1

log "=== Done. Report: $REPORT ==="
