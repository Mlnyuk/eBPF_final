// collector.bpf.c — CO-RE / libbpf rewrite of the BCC collector program.
//
// Same 15 per-cgroup signals as the Python/BCC version, keyed by
// bpf_get_current_cgroup_id(). Compiled ONCE against vmlinux.h (CO-RE), so the
// runtime image needs no clang/LLVM/kernel headers — fixing the BCC toolchain's
// DiskPressure problem.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_CG 16384

// Per-cgroup counter maps (u64 cgroup_id -> u64 value).
#define DEFINE_MAP(name)                          \
struct {                                          \
    __uint(type, BPF_MAP_TYPE_HASH);              \
    __uint(max_entries, MAX_CG);                  \
    __type(key, __u64);                           \
    __type(value, __u64);                         \
} name SEC(".maps");

DEFINE_MAP(c_read)
DEFINE_MAP(c_write)
DEFINE_MAP(c_open)
DEFINE_MAP(c_tcpconn)
DEFINE_MAP(c_tcpretx)
DEFINE_MAP(c_netrx)
DEFINE_MAP(c_nettx)
DEFINE_MAP(c_diskr)
DEFINE_MAP(c_diskw)
DEFINE_MAP(c_disklat)
DEFINE_MAP(c_diskio)
DEFINE_MAP(c_exec)
DEFINE_MAP(c_fork)
DEFINE_MAP(c_ctxsw)
DEFINE_MAP(c_cpu)

// disk in-flight: sector -> start ts, and sector -> issuing cgroup
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u64);
    __type(value, __u64);
} disk_start SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u64);
    __type(value, __u64);
} disk_cg SEC(".maps");

// per-CPU last sched_switch timestamp, keyed by logical CPU id
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u64);
} cpu_last SEC(".maps");

// Increment helper as a macro so the map is inlined at the call site (avoids the
// generic void* helper the verifier rejects).
#define INCR(map, key, delta)                                        \
    do {                                                             \
        __u64 _k = (key);                                            \
        __u64 *_v = bpf_map_lookup_elem(&map, &_k);                  \
        if (_v) {                                                    \
            __sync_fetch_and_add(_v, (delta));                       \
        } else {                                                     \
            __u64 _init = (delta);                                   \
            bpf_map_update_elem(&map, &_k, &_init, BPF_NOEXIST);     \
        }                                                            \
    } while (0)

// ---- syscalls -----------------------------------------------------------
SEC("tp/syscalls/sys_enter_read")
int tp_read(void *ctx) { INCR(c_read, bpf_get_current_cgroup_id(), 1); return 0; }

SEC("tp/syscalls/sys_enter_write")
int tp_write(void *ctx) { INCR(c_write, bpf_get_current_cgroup_id(), 1); return 0; }

SEC("tp/syscalls/sys_enter_open")
int tp_open(void *ctx) { INCR(c_open, bpf_get_current_cgroup_id(), 1); return 0; }

SEC("tp/syscalls/sys_enter_openat")
int tp_openat(void *ctx) { INCR(c_open, bpf_get_current_cgroup_id(), 1); return 0; }

// ---- process exec / fork ------------------------------------------------
SEC("tp/sched/sched_process_exec")
int tp_exec(void *ctx) { INCR(c_exec, bpf_get_current_cgroup_id(), 1); return 0; }

SEC("tp/sched/sched_process_fork")
int tp_fork(void *ctx) { INCR(c_fork, bpf_get_current_cgroup_id(), 1); return 0; }

// ---- context switch + on-CPU time ---------------------------------------
SEC("tp/sched/sched_switch")
int tp_switch(void *ctx) {
    __u64 cg = bpf_get_current_cgroup_id();
    INCR(c_ctxsw, cg, 1);

    __u32 cpu = bpf_get_smp_processor_id();
    __u64 now = bpf_ktime_get_ns();
    __u64 *last = bpf_map_lookup_elem(&cpu_last, &cpu);
    if (last) {
        __u64 delta = now - *last;
        INCR(c_cpu, cg, delta);
    }
    bpf_map_update_elem(&cpu_last, &cpu, &now, BPF_ANY);
    return 0;
}

// ---- TCP connect / retransmit -------------------------------------------
SEC("kprobe/tcp_v4_connect")
int BPF_KPROBE(k_tcp_v4_connect, struct sock *sk) {
    INCR(c_tcpconn, bpf_get_current_cgroup_id(), 1); return 0;
}
SEC("kprobe/tcp_v6_connect")
int BPF_KPROBE(k_tcp_v6_connect, struct sock *sk) {
    INCR(c_tcpconn, bpf_get_current_cgroup_id(), 1); return 0;
}
SEC("kprobe/tcp_retransmit_skb")
int BPF_KPROBE(k_tcp_retransmit_skb, struct sock *sk) {
    INCR(c_tcpretx, bpf_get_current_cgroup_id(), 1); return 0;
}

// ---- network bytes ------------------------------------------------------
SEC("kprobe/tcp_sendmsg")
int BPF_KPROBE(k_tcp_sendmsg, struct sock *sk, struct msghdr *msg, size_t size) {
    INCR(c_nettx, bpf_get_current_cgroup_id(), (__u64)size); return 0;
}
SEC("kprobe/tcp_cleanup_rbuf")
int BPF_KPROBE(k_tcp_cleanup_rbuf, struct sock *sk, int copied) {
    if (copied <= 0) return 0;
    INCR(c_netrx, bpf_get_current_cgroup_id(), (__u64)copied); return 0;
}

// ---- block I/O ----------------------------------------------------------
SEC("tp/block/block_rq_issue")
int tp_block_issue(struct trace_event_raw_block_rq *ctx) {
    __u64 cg = bpf_get_current_cgroup_id();
    __u64 sector = BPF_CORE_READ(ctx, sector);
    __u64 ts = bpf_ktime_get_ns();
    bpf_map_update_elem(&disk_start, &sector, &ts, BPF_ANY);
    bpf_map_update_elem(&disk_cg, &sector, &cg, BPF_ANY);

    __u64 nbytes = BPF_CORE_READ(ctx, bytes);
    char rw0 = BPF_CORE_READ(ctx, rwbs[0]);
    if (rw0 == 'W' || rw0 == 'w') INCR(c_diskw, cg, nbytes);
    else                          INCR(c_diskr, cg, nbytes);
    return 0;
}
SEC("tp/block/block_rq_complete")
int tp_block_complete(struct trace_event_raw_block_rq_completion *ctx) {
    __u64 sector = BPF_CORE_READ(ctx, sector);
    __u64 *tsp = bpf_map_lookup_elem(&disk_start, &sector);
    __u64 *cgp = bpf_map_lookup_elem(&disk_cg, &sector);
    if (tsp && cgp) {
        __u64 delta = bpf_ktime_get_ns() - *tsp;
        __u64 cg = *cgp;
        INCR(c_disklat, cg, delta);
        INCR(c_diskio, cg, 1);
        bpf_map_delete_elem(&disk_start, &sector);
        bpf_map_delete_elem(&disk_cg, &sector);
    }
    return 0;
}
