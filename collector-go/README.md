# collector-go — Go + libbpf CO-RE collector

A drop-in rewrite of the BCC/Python collector (`collector/`) using
[cilium/ebpf](https://github.com/cilium/ebpf) and **CO-RE** (Compile Once, Run
Everywhere). Same 15 per-cgroup signals, same 14-feature window vectors, same
CSV/JSONL output and `/detect/batch` push.

## Why

The BCC image ships clang/LLVM + kernel headers and **recompiles the BPF program
against the running kernel on every start** — the toolchain that caused
control-plane DiskPressure evictions. CO-RE compiles the BPF object **once at
build time** and relocates it against each node's BTF at load:

| | BCC (`collector/`) | CO-RE (`collector-go/`) |
|---|---|---|
| Runtime toolchain | clang + LLVM + headers in image | none (distroless) |
| Host mounts | /sys, /lib/modules, /usr/src | /sys only |
| BPF compile | every pod start, on node | once, at image build |
| Image base | ubuntu:24.04 (~big) | distroless/static |
| Userspace | Python + BCC | static Go binary |

## Layout

```
bpf/collector.bpf.c   CO-RE BPF program (SEC() probes, BTF maps, macro INCR)
bpf/vmlinux.h         generated from kernel 6.8 BTF (bpftool btf dump)
gen.go                //go:generate bpf2go directive (clang -D__TARGET_ARCH_x86)
main.go               load spec, attach 14 probes, window-drain, feature math
mapper.go             cgroup_id -> (ns, pod, container) via client-go
output.go             CSV/JSONL writer + detector push
```

## Build

In-cluster via kaniko (`docker/Dockerfile.collector-go`, multi-stage: clang build
→ distroless). Locally: `cd collector-go && go generate ./... && go build`
(needs clang + libbpf-dev).

## Parity (validated 2026-06-01, kernel 6.8, 9-node cluster)

- BPF loads via CO-RE at runtime: **14 probes attached**, no verifier errors.
- Output schema identical (19 cols = 5 meta + 14 features, ending `cpu_utilization`).
- cgroup→pod mapping resolves real namespaces (longhorn-system, kube-system,
  gpu-operator, …) — client-go enrichment works.
- `cpu_utilization` populated (sane per-core fractions).
- Emit volume ~32–37 rows/window (matches BCC ~25–38); push to detector succeeds.

Deployed as a separate DaemonSet (`k8s/daemonset-collector-go.yaml`) alongside the
BCC one for validation. Switchover = delete `ebpf-collector` DaemonSet, keep
`ebpf-collector-go`.
