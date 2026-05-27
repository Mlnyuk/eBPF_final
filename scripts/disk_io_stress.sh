#!/usr/bin/env bash
# disk_io_stress.sh — heavy disk I/O to raise disk bytes + I/O latency features.
#
# Tool: fio (preferred) or stress-ng fallback
# Install: sudo apt-get install -y fio
# Run:    ./disk_io_stress.sh [TARGET_DIR] [DURATION_SEC] [SIZE]
#           TARGET_DIR default = ./_diskstress
#           DURATION   default = 60
#           SIZE       default = 1G
# Revert: temp files are removed automatically on exit. Manual:
#           rm -rf <TARGET_DIR>
set -euo pipefail

TARGET="${1:-./_diskstress}"
DURATION="${2:-60}"
SIZE="${3:-1G}"

mkdir -p "$TARGET"
cleanup() { rm -rf "$TARGET"; echo "[disk_io_stress] cleaned up $TARGET"; }
trap cleanup EXIT

if command -v fio >/dev/null 2>&1; then
  echo "[disk_io_stress] fio randrw on $TARGET for ${DURATION}s"
  fio --name=diskstress --directory="$TARGET" --rw=randrw --rwmixread=50 \
      --bs=4k --size="$SIZE" --numjobs=4 --time_based --runtime="${DURATION}" \
      --ioengine=libaio --iodepth=32 --direct=1 --group_reporting
elif command -v stress-ng >/dev/null 2>&1; then
  echo "[disk_io_stress] stress-ng --hdd fallback for ${DURATION}s"
  stress-ng --hdd 4 --hdd-bytes "$SIZE" --timeout "${DURATION}s" --temp-path "$TARGET"
else
  echo "Neither fio nor stress-ng found. Install: sudo apt-get install -y fio" >&2
  exit 1
fi
echo "[disk_io_stress] done"
