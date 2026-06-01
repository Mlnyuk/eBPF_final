// Command collector-go is a CO-RE/libbpf (cilium/ebpf) rewrite of the BCC
// Python collector. Same 15 per-cgroup signals, same 14-feature window vectors,
// same CSV/JSONL output and detector push — but no runtime clang/LLVM/headers.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/rlimit"
)

// 14-feature schema (order MUST match configs/config.yaml + detector).
var featureOrder = []string{
	"syscall_read_rate", "syscall_write_rate", "syscall_open_rate",
	"tcp_connect_rate", "tcp_retransmit_rate",
	"network_rx_bytes", "network_tx_bytes",
	"disk_read_bytes", "disk_write_bytes", "disk_io_latency_ms",
	"process_exec_count", "process_fork_count", "context_switch_count",
	"cpu_utilization",
}

func main() {
	window := flag.Int("window", 10, "aggregation window seconds")
	outDir := flag.String("out", "/data", "output directory")
	format := flag.String("format", "both", "csv|jsonl|both")
	pushURL := flag.String("push-url", "", "detector base URL; POST each window to <url>/detect/batch")
	flag.Parse()

	node := os.Getenv("NODE_NAME")
	if node == "" {
		node, _ = os.Hostname()
	}

	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("remove memlock: %v", err)
	}

	var objs collectorObjects
	if err := loadCollectorObjects(&objs, nil); err != nil {
		log.Fatalf("load BPF objects: %v", err)
	}
	defer objs.Close()

	links := attachAll(&objs)
	for _, l := range links {
		defer l.Close()
	}
	log.Printf("[collector-go] attached %d probes; node=%s window=%ds", len(links), node, *window)

	var mapper *Mapper
	if os.Getenv("ENABLE_K8S_MAPPING") != "false" {
		mapper = NewMapper(node)
	}

	out := NewWriter(*outDir, *format, node, *pushURL, mapper)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	ticker := time.NewTicker(time.Duration(*window) * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-stop:
			log.Printf("[collector-go] shutting down")
			return
		case <-ticker.C:
			perCg := drain(&objs, float64(*window))
			out.Emit(perCg)
		}
	}
}

// attachAll links every program to its hook; fatal on the first failure.
func attachAll(o *collectorObjects) []link.Link {
	var links []link.Link
	tp := func(group, name string, prog *ebpf.Program) {
		l, err := link.Tracepoint(group, name, prog, nil)
		if err != nil {
			log.Fatalf("attach tracepoint %s/%s: %v", group, name, err)
		}
		links = append(links, l)
	}
	kp := func(sym string, prog *ebpf.Program) {
		l, err := link.Kprobe(sym, prog, nil)
		if err != nil {
			log.Printf("[warn] attach kprobe %s failed (skipping): %v", sym, err)
			return
		}
		links = append(links, l)
	}

	tp("syscalls", "sys_enter_read", o.TpRead)
	tp("syscalls", "sys_enter_write", o.TpWrite)
	tp("syscalls", "sys_enter_open", o.TpOpen)
	tp("syscalls", "sys_enter_openat", o.TpOpenat)
	tp("sched", "sched_process_exec", o.TpExec)
	tp("sched", "sched_process_fork", o.TpFork)
	tp("sched", "sched_switch", o.TpSwitch)
	tp("block", "block_rq_issue", o.TpBlockIssue)
	tp("block", "block_rq_complete", o.TpBlockComplete)

	kp("tcp_v4_connect", o.KTcpV4Connect)
	kp("tcp_v6_connect", o.KTcpV6Connect)
	kp("tcp_retransmit_skb", o.KTcpRetransmitSkb)
	kp("tcp_sendmsg", o.KTcpSendmsg)
	kp("tcp_cleanup_rbuf", o.KTcpCleanupRbuf)
	return links
}

// drainMap reads every (cgroup_id -> u64) entry then deletes it, returning the
// snapshot. Deletion gives per-window deltas (the BCC version did table.clear()).
func drainMap(m *ebpf.Map) map[uint64]uint64 {
	res := make(map[uint64]uint64)
	var k, v uint64
	it := m.Iterate()
	var keys []uint64
	for it.Next(&k, &v) {
		res[k] = v
		keys = append(keys, k)
	}
	for _, key := range keys {
		_ = m.Delete(&key)
	}
	return res
}

// drain converts all BPF maps into per-cgroup feature dicts for this window.
func drain(o *collectorObjects, window float64) map[uint64]map[string]float64 {
	per := make(map[uint64]map[string]float64)
	get := func(cg uint64) map[string]float64 {
		if per[cg] == nil {
			per[cg] = make(map[string]float64)
		}
		return per[cg]
	}

	// rate maps: value / window
	rate := []struct {
		m    *ebpf.Map
		feat string
	}{
		{o.CRead, "syscall_read_rate"}, {o.CWrite, "syscall_write_rate"},
		{o.COpen, "syscall_open_rate"}, {o.CTcpconn, "tcp_connect_rate"},
		{o.CTcpretx, "tcp_retransmit_rate"},
	}
	for _, r := range rate {
		for cg, val := range drainMap(r.m) {
			get(cg)[r.feat] = float64(val) / window
		}
	}
	// sum maps: raw value
	sum := []struct {
		m    *ebpf.Map
		feat string
	}{
		{o.CNetrx, "network_rx_bytes"}, {o.CNettx, "network_tx_bytes"},
		{o.CDiskr, "disk_read_bytes"}, {o.CDiskw, "disk_write_bytes"},
		{o.CExec, "process_exec_count"}, {o.CFork, "process_fork_count"},
		{o.CCtxsw, "context_switch_count"},
	}
	for _, s := range sum {
		for cg, val := range drainMap(s.m) {
			get(cg)[s.feat] = float64(val)
		}
	}
	// disk latency = total_ns / io_count / 1e6 (ms)
	lat := drainMap(o.CDisklat)
	cnt := drainMap(o.CDiskio)
	for cg, totalNs := range lat {
		n := cnt[cg]
		ms := 0.0
		if n > 0 {
			ms = float64(totalNs) / float64(n) / 1e6
		}
		get(cg)["disk_io_latency_ms"] = ms
	}
	// cpu_utilization = on-CPU ns / wall-clock ns
	windowNs := window * 1e9
	for cg, ns := range drainMap(o.CCpu) {
		get(cg)["cpu_utilization"] = float64(ns) / windowNs
	}
	return per
}

// keep fmt imported for potential debug builds
var _ = fmt.Sprintf
