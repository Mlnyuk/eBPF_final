#!/usr/bin/env bash
# pvc_io_stress.sh — heavy I/O against a mounted PVC to raise disk bytes +
# I/O latency on a *persistent volume* (distinct from node-local disk: surfaces
# CSI / Longhorn / network-storage latency). fault_type -> pvc_io_latency.
#
# Tool: fio (preferred) or dd fallback
# Install: sudo apt-get install -y fio
# Run:    ./pvc_io_stress.sh [PVC_MOUNT_DIR] [DURATION_SEC] [SIZE]
#           PVC_MOUNT_DIR default = ${PVC_DIR:-/data}   (a mounted PVC path)
#           DURATION      default = 60
#           SIZE          default = 512M
# Revert: temp files are removed automatically on exit. Manual:
#           rm -rf <PVC_MOUNT_DIR>/_pvcstress
set -euo pipefail

PVC_DIR="${1:-${PVC_DIR:-/data}}"
DURATION="${2:-60}"
SIZE="${3:-512M}"
TARGET="${PVC_DIR%/}/_pvcstress"

if [ ! -d "$PVC_DIR" ]; then
  echo "[pvc_io_stress] PVC mount '$PVC_DIR' not found; set PVC_DIR to a mounted PVC. Skipping." >&2
  exit 0
fi

mkdir -p "$TARGET"
cleanup() { rm -rf "$TARGET"; echo "[pvc_io_stress] cleaned up $TARGET"; }
trap cleanup EXIT

if command -v fio >/dev/null 2>&1; then
  echo "[pvc_io_stress] fio randrw on PVC $TARGET for ${DURATION}s"
  fio --name=pvcstress --directory="$TARGET" --rw=randrw --rwmixread=70 \
      --bs=8k --size="$SIZE" --numjobs=2 --time_based --runtime="${DURATION}" \
      --ioengine=libaio --iodepth=16 --direct=1 --group_reporting
else
  echo "[pvc_io_stress] fio missing; dd fallback loop for ~${DURATION}s"
  end=$(( $(date +%s) + DURATION ))
  while [ "$(date +%s)" -lt "$end" ]; do
    dd if=/dev/zero of="$TARGET/blob" bs=1M count=256 conv=fsync 2>/dev/null || true
    dd if="$TARGET/blob" of=/dev/null bs=1M 2>/dev/null || true
  done
fi
echo "[pvc_io_stress] done"
