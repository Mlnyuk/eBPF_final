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

# Models to poll. Primary always set; secondary optional (two-model agreement).
# A group is labelled benign (=> safe to suppress at runtime) ONLY when every
# model agrees benign; any suspicious vote or disagreement => suspicious (keep).
# This makes the risky action (silencing an anomaly) require consensus.
MODELS=[(QWEN, MODEL)]
if os.environ.get("QWEN2_BASE"):
    MODELS.append((os.environ["QWEN2_BASE"], os.environ.get("QWEN2_MODEL","")))

def ask(batch, base, model):
    usr="Label these groups:\n"+json.dumps(batch)
    payload={"model":model,"temperature":0,"max_tokens":3000,
             "chat_template_kwargs":{"enable_thinking":False},
             "response_format":{"type":"json_object"},
             "messages":[{"role":"system","content":SYS},{"role":"user","content":usr}]}
    req=urllib.request.Request(base+"/v1/chat/completions",
        data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    out=json.load(urllib.request.urlopen(req,timeout=300))
    txt=out["choices"][0]["message"]["content"]
    s=txt.find("{"); e=txt.rfind("}")          # strip ```json fences
    if s>=0 and e>s: txt=txt[s:e+1]
    return json.loads(txt)["labels"]

def key(l): return (l.get("node"),l.get("container"),l.get("trigger"))

labels=[]
for i in range(0,len(groups),BATCH):
    b=groups[i:i+BATCH]
    votes=collections.defaultdict(list)        # group key -> list of (model,verdict,conf)
    for base,model in MODELS:
        try:
            for l in ask(b,base,model):
                votes[key(l)].append((model,l.get("verdict"),float(l.get("confidence",0) or 0)))
            sys.stderr.write(f"batch {i//BATCH} [{model}]: ok\n")
        except Exception as e:
            sys.stderr.write(f"batch {i//BATCH} [{model}] FAILED: {e}\n")
    # merge votes per group (conservative: benign only on unanimous benign)
    for grp in b:
        k=(grp["node"],grp["container"],grp["trigger"])
        vs=votes.get(k,[])
        if not vs: continue
        verdicts=[v for _,v,_ in vs]
        all_benign=all(v=="benign" for v in verdicts)
        agreement=len(set(verdicts))==1
        verdict="benign" if all_benign else "suspicious"
        conf=round(sum(c for _,_,c in vs)/len(vs),2)
        labels.append({"node":k[0],"container":k[1],"trigger":k[2],"verdict":verdict,
                       "confidence":conf,"agreement":agreement,"n_models":len(vs),
                       "votes":[{"model":m,"verdict":v} for m,v,_ in vs]})

print(json.dumps({"source_file":f,"n_anom":len(anom),"n_normal":len(normal),
                  "n_groups":len(groups),"models":[m for _,m in MODELS],
                  "labels":labels,"groups":groups}))
