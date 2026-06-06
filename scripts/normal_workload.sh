#!/usr/bin/env bash
# normal_workload.sh — HARD-NEGATIVE generator: routine, benign-but-spiky activity
# that a naive threshold would flag but that must be labeled 'normal'. Gives the
# trainer real negatives (not only synthetic faults), improving alert precision.
#
# Mimics everyday operations: a short build/compile burst, a tar/gzip backup-like
# job, and a brief package-style file churn. All bounded and self-cleaning. The
# weak labeler maps these to 'normal' via HARD_NEGATIVE_HINTS; keep this pod/job
# named with a hard-negative hint (e.g. "cron", "backup", "build").
#
# Run:    ./normal_workload.sh [DURATION_SEC]
#           DURATION default = 60
# Revert: temp dir removed on exit.
set -euo pipefail

DURATION="${1:-60}"
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; echo "[normal_workload] cleaned up $WORK"; }
trap cleanup EXIT

echo "[normal_workload] benign routine activity for ~${DURATION}s in $WORK"
end=$(( $(date +%s) + DURATION ))
i=0
while [ "$(date +%s)" -lt "$end" ]; do
  # 1) build-like CPU+IO burst: generate files and checksum them
  dd if=/dev/urandom of="$WORK/data_$i.bin" bs=1M count=16 2>/dev/null || true
  sha256sum "$WORK"/data_*.bin >/dev/null 2>&1 || true
  # 2) backup-like compression
  tar -czf "$WORK/backup_$i.tgz" -C "$WORK" . >/dev/null 2>&1 || true
  # 3) brief idle gap (real workloads are bursty, not sustained)
  rm -f "$WORK"/data_*.bin "$WORK"/backup_*.tgz
  sleep 3
  i=$(( i + 1 ))
done
echo "[normal_workload] done (label as hard-negative 'normal')"
