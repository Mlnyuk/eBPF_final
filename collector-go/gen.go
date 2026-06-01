package main

// bpf2go compiles bpf/collector.bpf.c (CO-RE) into bpfel/bpfeb object files plus
// Go loaders (collector_bpfel.go etc.) in this package. Run via `go generate`.
// Requires clang + the libbpf headers at BUILD time only.
//
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go@v0.16.0 -cc clang -no-strip -target bpfel collector ./bpf/collector.bpf.c -- -I./bpf -O2 -g -Wall -D__TARGET_ARCH_x86
