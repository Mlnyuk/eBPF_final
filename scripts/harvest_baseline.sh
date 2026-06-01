#!/usr/bin/env bash
# Harvest accumulated feature rows from every collector's /data/features.csv
# into a single baseline CSV. Run AFTER the collectors have run long enough
# (e.g. 30 min) on the current image.
#
# Usage: bash scripts/harvest_baseline.sh [output.csv]
set -uo pipefail
NS=ebpf-final
OUT="${1:-/root/eBPF_final/results/baseline_14feat_$(date +%Y%m%d_%H%M%S).csv}"
HEADER="timestamp,node,namespace,pod,container,syscall_read_rate,syscall_write_rate,syscall_open_rate,tcp_connect_rate,tcp_retransmit_rate,network_rx_bytes,network_tx_bytes,disk_read_bytes,disk_write_bytes,disk_io_latency_ms,process_exec_count,process_fork_count,context_switch_count,cpu_utilization"

echo "$HEADER" > "$OUT"
total=0
for pod in $(kubectl -n $NS get pods -l app=ebpf-collector --no-headers | awk '{print $1}'); do
  node=$(kubectl -n $NS get pod "$pod" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  # strip header (timestamp...) and CR, append data rows
  rows=$(kubectl -n $NS exec "$pod" -- sh -c \
    "grep -v '^timestamp,' /data/features.csv 2>/dev/null" 2>/dev/null | tr -d '\r')
  n=$(printf '%s\n' "$rows" | grep -c .)
  printf '%s\n' "$rows" >> "$OUT"
  echo "  $node ($pod): $n rows"
  total=$((total + n))
done
# drop any blank lines
sed -i '/^$/d' "$OUT"
echo "TOTAL: $total rows -> $OUT"
echo "field check (want 19):"; tail -1 "$OUT" | awk -F, '{print NF}'