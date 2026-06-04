#!/usr/bin/env bash
# ============================================================================
# Distill-automation pipeline (nightly CronJob, k8s/cronjob-distill.yaml).
#
# Refreshes the runtime noise filter from the live anomaly stream:
#   1. run the LLM group labeler IN a detector pod (archive local + Qwen reachable)
#      — 32B primary, 14B secondary when QWEN2_BASE serves (two-model agreement);
#   2. aggregate today's anomaly + normal rows for the training corpus;
#   3. distill the labels into a DecisionTree (detector/train_noise_filter.py);
#   4. GATE: promote only if false-suppress <= floor and enough labels;
#   5. stream the new artifacts into each detector pod's /noise-live + POST /reload;
#   6. Telegram-notify the verdict.
#
# Self-skips (exit 0) when Qwen is offline (daytime) so off-window firings are
# harmless. No LLM is involved at detector runtime — only here, offline.
# ============================================================================
set -uo pipefail

NS="${NAMESPACE:-ebpf-final}"
QWEN_BASE="${QWEN_BASE:-http://qwen3-32b-5090.default:8000}"
QWEN_MODEL="${QWEN_MODEL:-qwen3-32b}"
QWEN2_BASE="${QWEN2_BASE:-}"
QWEN2_MODEL="${QWEN2_MODEL:-}"
ANOM_TAIL="${ANOM_TAIL:-20000}"
NORMAL_TAIL="${NORMAL_TAIL:-20000}"
FALSE_SUPPRESS_FLOOR="${FALSE_SUPPRESS_FLOOR:-0.01}"   # reject if > this
MIN_LABELS="${MIN_LABELS:-20}"
THRESHOLD="${NOISE_FILTER_THRESHOLD:-0.99}"
CHAT_ID="${TELEGRAM_CHAT_ID:-}"
TOKEN_FILE="${TELEGRAM_TOKEN_FILE:-/secrets/telegram/token}"
WORK="${WORK_DIR:-/work}"
LABELER_SRC="${LABELER_SRC:-/app/scripts/qwen_distill_label.py}"  # in-image default; host runs override
TRAINER_SRC="${TRAINER_SRC:-/app/detector/train_noise_filter.py}" # in-image default; host runs override
mkdir -p "$WORK"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

tg(){
  local msg="$1" token=""
  [ -f "$TOKEN_FILE" ] && token="$(cat "$TOKEN_FILE")"
  [ -z "$token" ] || [ -z "$CHAT_ID" ] && { log "telegram not configured"; return; }
  curl -sf -X POST "https://api.telegram.org/bot${token}/sendMessage" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"chat_id":sys.argv[1],"text":sys.argv[2]}))' "$CHAT_ID" "$msg")" \
    >/dev/null || log "telegram send failed"
}

# 0. Qwen up? (night-only)
if ! curl -sf "$QWEN_BASE/health" >/dev/null 2>&1; then
  log "Qwen offline — skipping distill, exit 0"; exit 0
fi

# 1. discover detector pods
PODS=$(kubectl get pods -n "$NS" -l app=ebpf-detector --field-selector=status.phase=Running \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
[ -z "$PODS" ] && { log "no detector pods"; exit 1; }
LABELER_POD=$(echo "$PODS" | head -1)
log "detector pods: $(echo "$PODS" | tr '\n' ' ')  labeler=$LABELER_POD"

# 2. label IN the pod (archive local + Qwen reachable from the pod network)
log "labeling via $LABELER_POD ..."
kubectl exec -i "$LABELER_POD" -n "$NS" -c detector -- \
  env QWEN_BASE="$QWEN_BASE" QWEN_MODEL="$QWEN_MODEL" \
      QWEN2_BASE="$QWEN2_BASE" QWEN2_MODEL="$QWEN2_MODEL" \
      ANOM_TAIL="$ANOM_TAIL" NORMAL_TAIL="$NORMAL_TAIL" \
  python3 - < "$LABELER_SRC" > "$WORK/labels.json" 2>"$WORK/label.err"
sed -n '$p' "$WORK/label.err" >/dev/null; cat "$WORK/label.err" >&2
N_LABELS=$(python3 -c "import json;print(len(json.load(open('$WORK/labels.json')).get('labels',[])))" 2>/dev/null || echo 0)
log "labels: $N_LABELS"
if [ "$N_LABELS" -lt "$MIN_LABELS" ]; then
  log "too few labels ($N_LABELS < $MIN_LABELS) — abort"; tg "⚠️ distill aborted: only $N_LABELS labels"; exit 0
fi

# 3. aggregate training rows (anomaly + normal) from all pods
ROWS="$WORK/rows.csv"; : > "$ROWS"; HDR=0
for p in $PODS; do
  if [ "$HDR" -eq 0 ]; then
    kubectl exec "$p" -n "$NS" -c detector -- sh -c \
      'F=$(ls /archive/features-*.csv|tail -1); head -1 $F' >> "$ROWS" 2>/dev/null && HDR=1
  fi
  kubectl exec "$p" -n "$NS" -c detector -- sh -c \
    "F=\$(ls /archive/features-*.csv|tail -1); grep ',True,' \$F | tail -$ANOM_TAIL; grep ',False,' \$F | tail -$NORMAL_TAIL" \
    >> "$ROWS" 2>/dev/null
done
log "training rows: $(($(wc -l < "$ROWS")-1))"

# 4. distill + metrics
python3 "$TRAINER_SRC" \
  --rows "$ROWS" --labels "$WORK/labels.json" \
  --out-model "$WORK/noise_filter.pkl" --out-baseline "$WORK/noise_baseline.json" \
  --metrics-out "$WORK/metrics.json" --threshold "$THRESHOLD" 2>&1 | tail -5
[ -f "$WORK/metrics.json" ] || { log "training failed"; tg "❌ distill: training failed"; exit 1; }

# 5. promotion gate
read -r FS NR NL < <(python3 -c "
import json;m=json.load(open('$WORK/metrics.json'))
print(m['false_suppress'],m['noise_reduced'],m['n_labeled'])")
log "candidate: false_suppress=$FS noise_reduced=$NR n_labeled=$NL (floor=$FALSE_SUPPRESS_FLOOR)"
PROMOTE=$(python3 -c "print(1 if $FS <= $FALSE_SUPPRESS_FLOOR else 0)")
if [ "$PROMOTE" -ne 1 ]; then
  log "REJECT — false_suppress $FS > floor $FALSE_SUPPRESS_FLOOR"
  tg "$(printf '🛑 distill REJECT\nfalse_suppress=%.2f%% > floor=%.2f%%\nnoise_reduced=%.0f%% labels=%s\nkept current filter.' \
        "$(python3 -c "print($FS*100)")" "$(python3 -c "print($FALSE_SUPPRESS_FLOOR*100)")" "$(python3 -c "print($NR*100)")" "$NL")"
  exit 3
fi

# 6. promote: stream artifacts into each pod's /noise-live + reload
log "PROMOTE — pushing to /noise-live on each pod"
OK=0; TOT=0
for p in $PODS; do
  TOT=$((TOT+1))
  kubectl exec -i "$p" -n "$NS" -c detector -- sh -c 'cat > /noise-live/noise_filter.pkl' < "$WORK/noise_filter.pkl" \
    && kubectl exec -i "$p" -n "$NS" -c detector -- sh -c 'cat > /noise-live/noise_baseline.json' < "$WORK/noise_baseline.json" \
    && kubectl exec "$p" -n "$NS" -c detector -- python3 -c \
        "import urllib.request,json;print(json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8080/reload',method='POST'),timeout=30))['noise_filter']['enabled'])" \
    && OK=$((OK+1)) && log "  $p reloaded"
done

tg "$(printf '✅ distill PROMOTED\nnoise_reduced=%.0f%% false_suppress=%.2f%%\nlabels=%s reloaded %s/%s pods' \
      "$(python3 -c "print($NR*100)")" "$(python3 -c "print($FS*100)")" "$NL" "$OK" "$TOT")"
log "done: reloaded $OK/$TOT pods"
