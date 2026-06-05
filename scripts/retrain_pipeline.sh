#!/usr/bin/env bash
# ===========================================================================
# Drift-adaptation retrain pipeline (run by k8s/cronjob-retrain.yaml).
#
#   1. Aggregate the /archive feature corpus from every detector replica.
#   2. Pull the current production model (live if promoted, else baked).
#   3. retrain.py: train candidate + regression gate vs labelled holdout.
#   4. On PASS: stream candidate into each replica's /models-live + POST /reload.
#      On FAIL: keep current. Always notify Telegram with the metrics summary.
#
# Uses `kubectl exec cat`/`exec -i ... cat >file` for transfer because the
# detector image (distro-slim) has no `tar`, so `kubectl cp` would fail.
# ===========================================================================
set -euo pipefail

NS="${NAMESPACE:-ebpf-final}"
WORK=/work
ARCH="$WORK/archive"
CHAT_ID="${TELEGRAM_CHAT_ID:-8098454352}"
TOKEN_FILE="${TELEGRAM_TOKEN_FILE:-/secrets/telegram/token}"
AUC_TOL="${AUC_TOL:-0.02}"
RECALL_FLOOR="${RECALL_FLOOR:-0.80}"
RETAIN_DAYS="${RETAIN_DAYS:-14}"   # archive retention window (disk + page-cache cap)
mkdir -p "$ARCH"

notify() {  # $1 = text (real newlines ok)
  local tok; tok="$(cat "$TOKEN_FILE" 2>/dev/null || true)"
  [ -z "$tok" ] && { echo "[notify-skip] $1"; return 0; }
  curl -sf -m 10 "https://api.telegram.org/bot${tok}/sendMessage" \
    --data-urlencode "chat_id=${CHAT_ID}" \
    --data-urlencode "text=$1" >/dev/null 2>&1 || echo "[notify-fail]"
}

# --- 1. detector replicas ---
mapfile -t PODS < <(kubectl -n "$NS" get pods -l app=ebpf-detector \
  --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
if [ "${#PODS[@]}" -eq 0 ]; then
  notify "🔴 retrain: no running detector pods"; exit 1
fi
echo "[retrain] detector pods: ${PODS[*]}"

# --- 2. aggregate archive corpus (exec cat per file) ---
# Retention first: the detector appends to these files continuously, so without
# a cap the corpus (and the page cache backing it) grows unbounded — pinning
# detector pods near their memory limit and blowing up this job's transfer +
# RAM. Drop files older than RETAIN_DAYS in-pod, then pull only what remains.
for p in "${PODS[@]}"; do
  kubectl -n "$NS" exec "$p" -c detector -- \
    find /archive -name 'features-*.csv' -type f -mtime +"$RETAIN_DAYS" -delete 2>/dev/null || true
  files="$(kubectl -n "$NS" exec "$p" -c detector -- \
    find /archive -name 'features-*.csv' -type f -mtime -"$RETAIN_DAYS" 2>/dev/null || true)"
  for f in $files; do
    base="$(basename "$f")"
    kubectl -n "$NS" exec "$p" -c detector -- cat "$f" > "$ARCH/${p}_${base}" 2>/dev/null || true
  done
done
mapfile -t DATA < <(find "$ARCH" -name '*features-*.csv' -size +1c)
if [ "${#DATA[@]}" -eq 0 ]; then
  notify "🟡 retrain: no archive data yet — skipping"; exit 0
fi
echo "[retrain] archive files: ${#DATA[@]}"

# --- 3. current production model (live if promoted, else baked) ---
CUR="$WORK/current.pkl"
src="$(kubectl -n "$NS" exec "${PODS[0]}" -c detector -- sh -c \
  'if [ -f /models-live/isolation_forest.pkl ]; then echo /models-live/isolation_forest.pkl; else echo /app/models/isolation_forest.pkl; fi')"
kubectl -n "$NS" exec "${PODS[0]}" -c detector -- cat "$src" > "$CUR"
echo "[retrain] current model from ${PODS[0]}:$src"

# --- 4. train + regression gate ---
set +e
OUT="$(python /app/detector/retrain.py --data "${DATA[@]}" \
  --current-model "$CUR" --candidate-out "$WORK/candidate.pkl" \
  --auc-tol "$AUC_TOL" --recall-floor "$RECALL_FLOOR" 2>&1)"
RC=$?
set -e
echo "$OUT"
SUMMARY="$(printf '%s' "$OUT" | python -c 'import sys,json
try:
    d=json.loads(sys.stdin.read())
    c=d["candidate"]; cur=d.get("current") or {}
    print(f"AUC cand={c[\"roc_auc\"]} cur={cur.get(\"roc_auc\")}\nminRecall={c[\"min_fault_recall\"]} normalFPR={c[\"normal_fpr\"]}\ntrainRows={c[\"n_train_samples\"]}")
except Exception:
    print("(metrics parse failed)")' 2>/dev/null || echo "(metrics parse failed)")"

# --- 5. act on verdict ---
if [ "$RC" -eq 0 ]; then
  for p in "${PODS[@]}"; do
    kubectl -n "$NS" exec -i "$p" -c detector -- sh -c \
      'cat > /models-live/isolation_forest.pkl' < "$WORK/candidate.pkl"
    ip="$(kubectl -n "$NS" get pod "$p" -o jsonpath='{.status.podIP}')"
    rl="$(curl -sf -m 10 -XPOST "http://${ip}:8080/reload" || echo '{}')"
    echo "[retrain] reloaded $p -> $rl"
  done
  notify "✅ retrain PROMOTED new model
${SUMMARY}"
elif [ "$RC" -eq 3 ]; then
  notify "🟡 retrain REJECTED by regression gate — kept current model
${SUMMARY}"
else
  notify "🔴 retrain ERROR (rc=${RC})
${SUMMARY}"
  exit "$RC"
fi
