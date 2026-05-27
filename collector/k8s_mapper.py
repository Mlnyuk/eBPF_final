"""
k8s_mapper.py
=============
Map an eBPF ``cgroup_id`` (as returned by bpf_get_current_cgroup_id) to a
Kubernetes identity: (namespace, pod, container).

How it works
------------
On cgroup v2 the value returned by bpf_get_current_cgroup_id() equals the
kernfs id of the cgroup directory, which on Linux equals the *inode number* of
the corresponding directory under /sys/fs/cgroup. So we:

  1. Walk /sys/fs/cgroup, recording {st_ino: dir_path}.
  2. Parse Kubernetes cgroup paths to extract the pod UID + container ID, e.g.
       .../kubepods-besteffort-pod<UID>.slice/cri-containerd-<CID>.scope
  3. (Optional) Enrich pod UID -> (namespace, pod_name, container_name) by
     querying the Kubernetes API with the in-cluster service account, if the
     `kubernetes` python client and a token are available.

Everything degrades gracefully: if enrichment is unavailable, the pod UID is
used as the pod identifier and namespace is "unknown". This satisfies the
prompt's "Stage 1: node/pid/cgroup_id, Stage 2: pod mapping" requirement.
"""
from __future__ import annotations

import functools
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

CGROUP_ROOT = "/sys/fs/cgroup"

# pod UID inside a slice name: pod<UID>.slice  (UID has _ instead of - sometimes)
_POD_RE = re.compile(r"pod([0-9a-fA-F_\-]{8,})\.slice")
# container id: cri-containerd-<id>.scope / docker-<id>.scope / crio-<id>.scope
_CONTAINER_RE = re.compile(r"(?:cri-containerd-|docker-|crio-)([0-9a-fA-F]{12,})\.scope")

Identity = Tuple[str, str, str]  # (namespace, pod, container)
UNKNOWN: Identity = ("unknown", "unknown", "unknown")


def _normalize_uid(raw: str) -> str:
    """systemd encodes pod UIDs with '_' instead of '-'. Restore dashes when it
    looks like a 32-hex UUID so it matches the k8s API metadata.uid."""
    cand = raw.replace("_", "-")
    hexonly = cand.replace("-", "")
    if len(hexonly) == 32:
        return f"{hexonly[0:8]}-{hexonly[8:12]}-{hexonly[12:16]}-{hexonly[16:20]}-{hexonly[20:32]}"
    return cand


class K8sMapper:
    """Resolves cgroup_id -> (namespace, pod, container) with caching."""

    def __init__(self, enable_api: bool = True, refresh_seconds: int = 30):
        self.enable_api = enable_api
        self.refresh_seconds = refresh_seconds
        self._ino_to_path: Dict[int, str] = {}
        self._uid_to_pod: Dict[str, Tuple[str, str]] = {}  # uid -> (ns, name)
        self._cid_to_cname: Dict[str, str] = {}            # container id -> name
        self._cache: Dict[int, Identity] = {}
        self._last_scan = 0.0
        self._k8s = None
        if enable_api:
            self._init_k8s_client()

    # ---- cgroup inode scan -------------------------------------------------

    def _scan_cgroups(self) -> None:
        ino_map: Dict[int, str] = {}
        for dirpath, dirnames, _ in os.walk(CGROUP_ROOT):
            try:
                ino = os.stat(dirpath).st_ino
                ino_map[ino] = dirpath
            except (FileNotFoundError, PermissionError):
                continue
        self._ino_to_path = ino_map

    # ---- kubernetes API enrichment ----------------------------------------

    def _init_k8s_client(self) -> None:
        try:
            from kubernetes import client, config  # type: ignore
            config.load_incluster_config()
            self._k8s = client.CoreV1Api()
        except Exception:
            # not in-cluster / client missing -> enrichment disabled
            self._k8s = None

    def _refresh_pod_table(self) -> None:
        if self._k8s is None:
            return
        node = os.environ.get("NODE_NAME")
        try:
            field = f"spec.nodeName={node}" if node else None
            pods = self._k8s.list_pod_for_all_namespaces(field_selector=field)
        except Exception:
            return
        uid_map: Dict[str, Tuple[str, str]] = {}
        cid_map: Dict[str, str] = {}
        for pod in pods.items:
            uid = pod.metadata.uid
            uid_map[uid] = (pod.metadata.namespace, pod.metadata.name)
            statuses = (pod.status.container_statuses or [])
            for cs in statuses:
                cid = (cs.container_id or "").split("://")[-1]
                if cid:
                    cid_map[cid[:64]] = cs.name
        self._uid_to_pod = uid_map
        self._cid_to_cname = cid_map

    def _maybe_refresh(self) -> None:
        now = time.time()
        if now - self._last_scan >= self.refresh_seconds or not self._ino_to_path:
            self._scan_cgroups()
            self._refresh_pod_table()
            self._cache.clear()
            self._last_scan = now

    # ---- public ------------------------------------------------------------

    def resolve(self, cgroup_id: int) -> Identity:
        self._maybe_refresh()
        if cgroup_id in self._cache:
            return self._cache[cgroup_id]

        path = self._ino_to_path.get(cgroup_id)
        if not path:
            self._cache[cgroup_id] = UNKNOWN
            return UNKNOWN

        pod_m = _POD_RE.search(path)
        con_m = _CONTAINER_RE.search(path)
        if not pod_m:
            # not a pod cgroup (system.slice, etc.)
            ident = ("node", "system", os.path.basename(path) or "root")
            self._cache[cgroup_id] = ident
            return ident

        uid = _normalize_uid(pod_m.group(1))
        cid = con_m.group(1) if con_m else ""

        ns, pod_name = self._uid_to_pod.get(uid, ("unknown", f"pod-{uid[:8]}"))
        cname = self._cid_to_cname.get(cid[:64], cid[:12] if cid else "unknown")

        ident = (ns, pod_name, cname)
        self._cache[cgroup_id] = ident
        return ident


@functools.lru_cache(maxsize=1)
def default_mapper() -> K8sMapper:
    enable = os.environ.get("ENABLE_K8S_MAPPING", "true").lower() != "false"
    return K8sMapper(enable_api=enable)


if __name__ == "__main__":
    # quick self-test: print resolvable pod cgroups on this node
    m = K8sMapper()
    m._maybe_refresh()
    shown = 0
    for ino, path in m._ino_to_path.items():
        if "pod" in path and ".slice" in path:
            print(ino, "->", m.resolve(ino), "|", path)
            shown += 1
            if shown >= 20:
                break
    if shown == 0:
        print("no pod cgroups found (not on a k8s node, or cgroup v1)")
