#!/usr/bin/env python3
# Runs INSIDE a detector pod (archive local at /archive, qwen svc resolvable).
# Groups today's flagged anomalies, asks Qwen3-32B to label each group
# benign|suspicious as strict JSON, prints {groups:[...]} to stdout.
import os, sys, csv, json, glob, urllib.request, collections, statistics

QWEN = os.environ.get("QWEN_BASE", "http://qwen3-32b-5090.default:8000")
MODEL = os.environ.get("QWEN_MODEL", "qwen3-32b")
ANOM_TAIL = int(os.environ.get("ANOM_TAIL", "20000"))
NORMAL_TAIL = int(os.environ.get("NORMAL_TAIL", "20000"))
MIN_GROUP = int(os.environ.get("MIN_GROUP", "5"))
BATCH = int(os.environ.get("BATCH", "25"))
FEATS = ["syscall_read_rate","syscall_write_rate","syscall_open_rate","tcp_connect_rate",
         "tcp_retransmit_rate","network_rx_bytes","network_tx_bytes","disk_read_bytes",
         "disk_write_bytes","disk_io_latency_ms","process_exec_count","process_fork_count",
         "context_switch_count","cpu_utilization"]

def med(rs,f):
    v=[float(x[f]) for x in rs if x.get(f) not in (None,"","nan")]
    return statistics.median(v) if v else 0.0

def tail(path,n,needle):
    out=[]
    with open(path) as f:
        for line in f:
            if needle in line: out.append(line)
    return out[-n:]

f=sorted(glob.glob("/archive/features-*.csv"))[-1]
with open(f) as fh: header=fh.readline().strip().split(",")
def rows(lines):
    r=[]
    for l in lines:
        v=l.rstrip("\n").split(",")
        if len(v)==len(header): r.append(dict(zip(header,v)))
    return r
anom=rows(tail(f,ANOM_TAIL,",True,"))
normal=rows(tail(f,NORMAL_TAIL,",False,"))
byc=collections.defaultdict(list)
for r in normal: byc[r["container"]].append(r)
baseline={c:{ff:round(med(rs,ff),2) for ff in FEATS} for c,rs in byc.items() if len(rs)>=20}
g=collections.defaultdict(list)
for r in anom: g[(r["node"],r["container"],r["trigger"])].append(r)

groups=[]
for (node,cont,trig),rs in g.items():
    if len(rs)<MIN_GROUP: continue
    sc=[float(x["anomaly_score"]) for x in rs]
    cur={ff:round(med(rs,ff),2) for ff in FEATS}
    bl=baseline.get(cont,{})
    dev={}
    for ff in FEATS:
        b,c=bl.get(ff,0.0),cur[ff]
        if b>0 and (c/b>=3 or c/b<=0.33): dev[ff]=round(c/b,1)
        elif b==0 and c>0: dev[ff]="new"
    groups.append({"node":node,"container":cont,"trigger":trig,"count":len(rs),
                   "median_score":round(statistics.median(sc),3),
                   "baseline":bl,"observed":cur,"deviations":dev})

SYS=("You label eBPF anomaly groups for a Kubernetes cluster. Each group was flagged by an "
"Isolation Forest. 'baseline' = normal median feature values for THIS container type; "
"'observed' = current; 'deviations' = features that moved >=3x vs baseline (ratio or 'new'). "
"Judge by deviation from the container's OWN baseline, NOT absolute size — system daemons "
"(containerd, kubelet, cgroup, *.service, kube-*, longhorn-*, init.scope) naturally run high; "
"empty deviations => benign noise (IF over-sensitive). Label benign unless deviations show a "
"clear, meaningful departure suggesting real load/abuse. "
"Output STRICT JSON only: {\"labels\":[{\"node\":..,\"container\":..,\"trigger\":..,"
"\"verdict\":\"benign|suspicious\",\"confidence\":0..1}]} — one entry per input group, same order.")

def ask(batch):
    usr="Label these groups:\n"+json.dumps(batch)
    payload={"model":MODEL,"temperature":0,"max_tokens":3000,
             "chat_template_kwargs":{"enable_thinking":False},
             "response_format":{"type":"json_object"},
             "messages":[{"role":"system","content":SYS},{"role":"user","content":usr}]}
    req=urllib.request.Request(QWEN+"/v1/chat/completions",
        data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    out=json.load(urllib.request.urlopen(req,timeout=300))
    txt=out["choices"][0]["message"]["content"]
    # qwen wraps JSON in ```json ... ``` fences; extract the object substring.
    s=txt.find("{"); e=txt.rfind("}")
    if s>=0 and e>s: txt=txt[s:e+1]
    return json.loads(txt)["labels"]

labels=[]
for i in range(0,len(groups),BATCH):
    b=groups[i:i+BATCH]
    try:
        lb=ask(b)
        labels.extend(lb)
        sys.stderr.write(f"batch {i//BATCH}: {len(lb)} labels\n")
    except Exception as e:
        sys.stderr.write(f"batch {i//BATCH} FAILED: {e}\n")

print(json.dumps({"source_file":f,"n_anom":len(anom),"n_normal":len(normal),
                  "n_groups":len(groups),"labels":labels,"groups":groups}))
