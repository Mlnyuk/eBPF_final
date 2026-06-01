package main

import (
	"context"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

const cgroupRoot = "/sys/fs/cgroup"

var (
	podRe       = regexp.MustCompile(`pod([0-9a-fA-F_\-]{8,})\.slice`)
	containerRe = regexp.MustCompile(`(?:cri-containerd-|docker-|crio-)([0-9a-fA-F]{12,})\.scope`)
)

// Identity is (namespace, pod, container).
type Identity struct{ Namespace, Pod, Container string }

var unknown = Identity{"unknown", "unknown", "unknown"}

// Mapper resolves cgroup_id -> Identity with a periodically refreshed cache.
type Mapper struct {
	node        string
	clientset   *kubernetes.Clientset
	refresh     time.Duration
	mu          sync.Mutex
	inoToPath   map[uint64]string
	uidToPod    map[string][2]string // uid -> {ns, name}
	cidToCName  map[string]string
	cache       map[uint64]Identity
	lastScan    time.Time
}

func NewMapper(node string) *Mapper {
	m := &Mapper{node: node, refresh: 30 * time.Second,
		inoToPath: map[uint64]string{}, uidToPod: map[string][2]string{},
		cidToCName: map[string]string{}, cache: map[uint64]Identity{}}
	if cfg, err := rest.InClusterConfig(); err == nil {
		if cs, err := kubernetes.NewForConfig(cfg); err == nil {
			m.clientset = cs
		} else {
			log.Printf("[mapper] k8s client init failed: %v", err)
		}
	} else {
		log.Printf("[mapper] not in-cluster (%v); enrichment disabled", err)
	}
	return m
}

func (m *Mapper) maybeRefresh() {
	if time.Since(m.lastScan) < m.refresh && len(m.inoToPath) > 0 {
		return
	}
	m.scanCgroups()
	m.refreshPods()
	m.cache = map[uint64]Identity{}
	m.lastScan = time.Now()
}

func (m *Mapper) scanCgroups() {
	ino := make(map[uint64]string)
	filepath.WalkDir(cgroupRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		if fi, e := os.Stat(p); e == nil {
			if st, ok := fi.Sys().(*syscall.Stat_t); ok {
				ino[st.Ino] = p
			}
		}
		return nil
	})
	m.inoToPath = ino
}

func (m *Mapper) refreshPods() {
	if m.clientset == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	pods, err := m.clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: "spec.nodeName=" + m.node,
	})
	if err != nil {
		log.Printf("[mapper] list pods failed: %v", err)
		return
	}
	uidMap := make(map[string][2]string)
	cidMap := make(map[string]string)
	for i := range pods.Items {
		p := &pods.Items[i]
		uidMap[string(p.UID)] = [2]string{p.Namespace, p.Name}
		for _, cs := range p.Status.ContainerStatuses {
			cid := cs.ContainerID
			if idx := strings.Index(cid, "://"); idx >= 0 {
				cid = cid[idx+3:]
			}
			if cid != "" {
				if len(cid) > 64 {
					cid = cid[:64]
				}
				cidMap[cid] = cs.Name
			}
		}
	}
	m.uidToPod = uidMap
	m.cidToCName = cidMap
}

func normalizeUID(raw string) string {
	cand := strings.ReplaceAll(raw, "_", "-")
	hex := strings.ReplaceAll(cand, "-", "")
	if len(hex) == 32 {
		return hex[0:8] + "-" + hex[8:12] + "-" + hex[12:16] + "-" + hex[16:20] + "-" + hex[20:32]
	}
	return cand
}

// Resolve maps a cgroup_id to its Kubernetes identity (best-effort).
func (m *Mapper) Resolve(cgID uint64) Identity {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maybeRefresh()
	if id, ok := m.cache[cgID]; ok {
		return id
	}
	path, ok := m.inoToPath[cgID]
	if !ok {
		m.cache[cgID] = unknown
		return unknown
	}
	pm := podRe.FindStringSubmatch(path)
	if pm == nil {
		id := Identity{"node", "system", filepath.Base(path)}
		m.cache[cgID] = id
		return id
	}
	uid := normalizeUID(pm[1])
	cid := ""
	if cm := containerRe.FindStringSubmatch(path); cm != nil {
		cid = cm[1]
	}
	ns, pod := "unknown", "pod-"+truncate(uid, 8)
	if v, ok := m.uidToPod[uid]; ok {
		ns, pod = v[0], v[1]
	}
	cname := "unknown"
	if cid != "" {
		key := cid
		if len(key) > 64 {
			key = key[:64]
		}
		if v, ok := m.cidToCName[key]; ok {
			cname = v
		} else {
			cname = truncate(cid, 12)
		}
	}
	id := Identity{ns, pod, cname}
	m.cache[cgID] = id
	return id
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
