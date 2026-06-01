// Command collector-go is a CO-RE/libbpf (cilium/ebpf) rewrite of the BCC
// Python collector. Same 15 per-cgroup signals, same 14-feature window vectors,
// same CSV/JSONL output and detector push — but no runtime clang/LLVM/headers.
package main

import (
	"flag"
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

	// Load the CO-RE spec and instantiate. Indexing programs/maps by their C
	// names avoids depending on bpf2go's generated Go field naming.
	spec, err := loadCollector()
	if err != nil {
		log.Fatalf("load BPF spec: %v", err)
	}
	coll, err := ebpf.NewCollection(spec)
	if err != nil {
		log.Fatalf("new BPF collection: %v", err)
	}
	defer coll.Close()

	links := attachAll(coll)
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
			perCg := drain(coll, float64(*window))
			out.Emit(perCg)
		}
	}
}

// attachAll links every program (looked up by C name) to its hook.
func attachAll(coll *ebpf.Collection) []link.Link {
	var links []link.Link
	prog := func(name string) *ebpf.Program {
		p := coll.Programs[name]
		if p == nil {
			log.Fatalf("program %q not found in collection", name)
		}
		return p
	}
	tp := func(group, name, progName string) {
		l, err := link.Tracepoint(group, name, prog(progName), nil)
		if err != nil {
			log.Fatalf("attach tracepoint %s/%s: %v", group, name, err)
		}
		links = append(links, l)
	}
	kp := func(sym, progName string) {
		l, err := link.Kprobe(sym, prog(progName), nil)
		if err != nil {
			log.Printf("[warn] attach kprobe %s failed (skipping): %v", sym, err)
			return
		}
		links = append(links, l)
	}

	tp("syscalls", "sys_enter_read", "tp_read")
	tp("syscalls", "sys_enter_write", "tp_write")
	tp("syscalls", "sys_enter_open", "tp_open")
	tp("syscalls", "sys_enter_openat", "tp_openat")
	tp("sched", "sched_process_exec", "tp_exec")
	tp("sched", "sched_process_fork", "tp_fork")
	tp("sched", "sched_switch", "tp_switch")
	tp("block", "block_rq_issue", "tp_block_issue")
	tp("block", "block_rq_complete", "tp_block_complete")

	kp("tcp_v4_connect", "k_tcp_v4_connect")
	kp("tcp_v6_connect", "k_tcp_v6_connect")
	kp("tcp_retransmit_skb", "k_tcp_retransmit_skb")
	kp("tcp_sendmsg", "k_tcp_sendmsg")
	kp("tcp_cleanup_rbuf", "k_tcp_cleanup_rbuf")
	return links
}

// drainMap reads every (cgroup_id -> u64) entry then deletes it, returning the
// snapshot. Deletion gives per-window deltas (the BCC version did table.clear()).
func drainMap(m *ebpf.Map) map[uint64]uint64 {
	res := make(map[uint64]uint64)
	if m == nil {
		return res
	}
	var k, v uint64
	it := m.Iterate()
	var keys []uint64
	for it.Next(&k, &v) {
		res[k] = v
		keys = append(keys, k)
	}
	for i := range keys {
		_ = m.Delete(&keys[i])
	}
	return res
}

// drain converts all BPF maps into per-cgroup feature dicts for this window.
func drain(coll *ebpf.Collection, window float64) map[uint64]map[string]float64 {
	per := make(map[uint64]map[string]float64)
	get := func(cg uint64) map[string]float64 {
		if per[cg] == nil {
			per[cg] = make(map[string]float64)
		}
		return per[cg]
	}
	M := func(name string) *ebpf.Map { return coll.Maps[name] }

	rate := []struct{ name, feat string }{
		{"c_read", "syscall_read_rate"}, {"c_write", "syscall_write_rate"},
		{"c_open", "syscall_open_rate"}, {"c_tcpconn", "tcp_connect_rate"},
		{"c_tcpretx", "tcp_retransmit_rate"},
	}
	for _, r := range rate {
		for cg, val := range drainMap(M(r.name)) {
			get(cg)[r.feat] = float64(val) / window
		}
	}
	sum := []struct{ name, feat string }{
		{"c_netrx", "network_rx_bytes"}, {"c_nettx", "network_tx_bytes"},
		{"c_diskr", "disk_read_bytes"}, {"c_diskw", "disk_write_bytes"},
		{"c_exec", "process_exec_count"}, {"c_fork", "process_fork_count"},
		{"c_ctxsw", "context_switch_count"},
	}
	for _, s := range sum {
		for cg, val := range drainMap(M(s.name)) {
			get(cg)[s.feat] = float64(val)
		}
	}
	// disk latency = total_ns / io_count / 1e6 (ms)
	lat := drainMap(M("c_disklat"))
	cnt := drainMap(M("c_diskio"))
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
	for cg, ns := range drainMap(M("c_cpu")) {
		get(cg)["cpu_utilization"] = float64(ns) / windowNs
	}
	return per
}
