#!/usr/bin/env python3
"""LLM anomaly triage.

Pulls the most recent anomaly + normal rows from every detector pod's archive,
builds a per-container baseline (so the model judges deviation-from-normal, not
absolute magnitude — system daemons like containerd/kubelet run high by nature),
groups the high-confidence anomalies, and asks the night-only Qwen3-32B endpoint
to verdict each group (benign|suspicious) with a cause and a next check. Posts
the digest to Telegram.

Best-effort by design: if Qwen is offline (it only runs at night) the job logs
and exits 0 without alerting. Run as a CronJob in the night window — see
k8s/cronjob-triage.yaml.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import subprocess
import collections
import statistics
from datetime import datetime, timezone

# --- config (env-overridable) ------------------------------------------------
NS          = os.environ.get("NAMESPACE", "ebpf-final")
QWEN_BASE   = os.environ.get("QWEN_BASE", "http://qwen3-32b-5090.default:8000")
QWEN_MODEL  = os.environ.get("QWEN_MODEL", "qwen3-32b")
CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
TOKEN_FILE  = os.environ.get("TELEGRAM_TOKEN_FILE", "/secrets/telegram/token")
ANOM_TAIL   = int(os.environ.get("ANOM_TAIL", "4000"))
NORMAL_TAIL = int(os.environ.get("NORMAL_TAIL", "8000"))
SCORE_MIN   = float(os.environ.get("SCORE_MIN", "0.7"))
TOP_N       = int(os.environ.get("TOP_N", "8"))
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "1400"))
NOTIFY_QUIET = os.environ.get("NOTIFY_QUIET", "false").lower() == "true"
DRY_RUN     = os.environ.get("DRY_RUN", "false").lower() == "true"

FEATS = ["syscall_read_rate", "syscall_write_rate", "syscall_open_rate",
         "tcp_connect_rate", "tcp_retransmit_rate", "network_rx_bytes",
         "network_tx_bytes", "disk_read_bytes", "disk_write_bytes",
         "disk_io_latency_ms", "process_exec_count", "process_fork_count",
         "context_switch_count", "cpu_utilization"]


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def telegram_token():
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except OSError:
        return os.environ.get("TELEGRAM_TOKEN", "").strip()


def send_telegram(text):
    token = telegram_token()
    if not token or not CHAT_ID:
        log("telegram not configured; skipping notify")
        return
    if DRY_RUN:
        log("DRY_RUN: would send telegram:\n" + text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram caps messages at 4096 chars; chunk on paragraph boundaries.
    for chunk in _chunks(text, 3900):
        data = json.dumps({"chat_id": CHAT_ID, "text": chunk,
                           "disable_web_page_preview": True}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.URLError as e:
            log(f"telegram send failed: {e}")


def _chunks(text, n):
    out, cur = [], ""
    for para in text.split("\n\n"):
        # hard-split any single paragraph that alone exceeds the limit
        while len(para) > n:
            if cur:
                out.append(cur)
                cur = ""
            out.append(para[:n])
            para = para[n:]
        if len(cur) + len(para) + 2 > n and cur:
            out.append(cur)
            cur = ""
        cur += (para + "\n\n")
    if cur.strip():
        out.append(cur)
    return out or [text]


def qwen_up():
    try:
        urllib.request.urlopen(f"{QWEN_BASE}/health", timeout=10).read()
        return True
    except urllib.error.URLError as e:
        log(f"qwen /health unreachable: {e}")
        return False


def kubectl(*args, **kw):
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, **kw)


def detector_pods():
    r = kubectl("get", "pods", "-n", NS, "-l", "app=ebpf-detector",
                "--field-selector=status.phase=Running", "-o",
                "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}")
    return [p for p in r.stdout.split("\n") if p.strip()]


def pull_rows(pod, date_str):
    """exec-cat header + last N anomaly + last N normal rows from a pod's archive.

    The detector image has no tar (so `kubectl cp` is out); we stream via
    `kubectl exec ... sh -c`, matching the retrain pipeline's approach.
    """
    glob = f"/archive/features-{date_str}-*.csv"
    script = (f"set -e; F=$(ls {glob} 2>/dev/null | head -1); [ -z \"$F\" ] && exit 0; "
              f"head -1 \"$F\"; "
              f"grep ',True,' \"$F\" 2>/dev/null | tail -{ANOM_TAIL}; "
              f"grep ',False,' \"$F\" 2>/dev/null | tail -{NORMAL_TAIL}")
    r = kubectl("exec", pod, "-n", NS, "-c", "detector", "--", "sh", "-c", script)
    return r.stdout


def parse_csv(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    rows = []
    for l in lines[1:]:
        vals = l.split(",")
        if len(vals) == len(header):
            rows.append(dict(zip(header, vals)))
    return rows


def _med(rows, f):
    v = [float(x[f]) for x in rows if x.get(f) not in (None, "", "nan")]
    return statistics.median(v) if v else 0.0


def build_groups(rows):
    normal = [r for r in rows if r.get("is_anomaly") == "False"]
    anom   = [r for r in rows if r.get("is_anomaly") == "True"]
    # baseline: median normal feature values per container type
    by_cont = collections.defaultdict(list)
    for r in normal:
        by_cont[r["container"]].append(r)
    baseline = {c: {f: round(_med(rs, f), 2) for f in FEATS}
                for c, rs in by_cont.items() if len(rs) >= 20}
    # group anomalies, keep only high-confidence
    g = collections.defaultdict(list)
    for r in anom:
        g[(r["node"], r["container"], r["trigger"])].append(r)
    groups = []
    for (node, cont, trig), rs in g.items():
        sc = [float(x["anomaly_score"]) for x in rs]
        msc = statistics.median(sc)
        if msc < SCORE_MIN:
            continue
        cur = {f: round(_med(rs, f), 2) for f in FEATS}
        bl = baseline.get(cont, {})
        dev = {}
        for f in FEATS:
            b, c = bl.get(f, 0.0), cur[f]
            if b > 0 and (c / b >= 3 or c / b <= 0.33):
                dev[f] = f"{c} vs base {b} (x{round(c / b, 1)})"
            elif b == 0 and c > 0:
                dev[f] = f"{c} vs base 0"
        groups.append({"node": node, "container": cont, "trigger": trig,
                       "count": len(rs), "median_score": round(msc, 3),
                       "max_score": round(max(sc), 3), "baseline_normal": bl,
                       "observed": cur, "notable_deviations": dev})
    groups.sort(key=lambda x: (-x["median_score"], -x["count"]))
    return groups, len(anom), len(normal)


SYS_PROMPT = (
    "You are an SRE triage assistant for a Kubernetes cluster monitored by eBPF "
    "+ an Isolation Forest. Each group below was flagged anomalous "
    "(median_score>=0.7, 0..1). CRITICAL: 'baseline_normal' = median feature "
    "values for THIS container type during NORMAL (non-anomalous) windows. "
    "'observed' = current anomalous values. 'notable_deviations' pre-computes "
    "features that moved >=3x vs baseline. Judge each group by how far observed "
    "deviates from its OWN baseline — NOT by absolute magnitude (system daemons "
    "like containerd/kubelet/cgroup naturally run high; that is their normal). "
    "Feature rates are per-second medians. trigger: model=IF, z=z-score tail, "
    "both=both.\nFor each group output exactly:\n"
    "  <node>/<container> [trigger] — VERDICT(benign|suspicious) conf=NN%\n"
    "  cause: <one line, cite the deviating feature vs its baseline>\n"
    "  check: <one concrete command/action>\n"
    "A group with empty notable_deviations is almost certainly benign (IF "
    "over-sensitive). Say so.")


def ask_qwen(groups, total):
    usr = (f"Flagged groups (top {len(groups)} of {total} high-confidence):\n"
           + json.dumps(groups, indent=1))
    payload = {"model": QWEN_MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": usr}]}
    req = urllib.request.Request(f"{QWEN_BASE}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=300))
    log(f"qwen replied in {time.time() - t:.1f}s "
        f"(usage={json.dumps(out.get('usage', {}))})")
    return out["choices"][0]["message"]["content"]


def main():
    if not qwen_up():
        log("Qwen offline (night-only) — skipping triage, exit 0")
        if not NOTIFY_QUIET:
            send_telegram("🌙 eBPF triage skipped — Qwen LLM offline (daytime).")
        return 0

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    pods = detector_pods()
    if not pods:
        log("no running detector pods found")
        return 1
    log(f"detector pods: {pods}")

    rows = []
    for p in pods:
        txt = pull_rows(p, date_str)
        pr = parse_csv(txt)
        log(f"{p}: {len(pr)} rows")
        rows.extend(pr)
    if not rows:
        log("no archive rows for today — nothing to triage")
        return 0

    groups, n_anom, n_norm = build_groups(rows)
    ratio = n_anom / (n_anom + n_norm) if (n_anom + n_norm) else 0
    log(f"anom={n_anom} normal={n_norm} ratio={ratio:.2%} "
        f"high-conf groups={len(groups)}")

    if not groups:
        log("no high-confidence anomaly groups")
        if not NOTIFY_QUIET:
            send_telegram(f"✅ eBPF triage: 0 high-confidence anomalies "
                          f"(sampled ratio {ratio:.0%}).")
        return 0

    verdict = ask_qwen(groups[:TOP_N], len(groups))
    header = (f"🔎 *eBPF LLM triage* — {date_str}\n"
              f"sampled anomaly ratio {ratio:.0%}, "
              f"{len(groups)} high-conf groups (top {min(TOP_N, len(groups))}):\n\n")
    send_telegram(header + verdict)
    log("digest sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
