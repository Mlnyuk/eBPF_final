#!/usr/bin/env bash
# network_delay.sh — inject egress latency with tc netem (raises RTT, retransmits).
#
# Tool: tc (iproute2)
# Run:    sudo ./network_delay.sh [IFACE] [DELAY_MS] [JITTER_MS]
#           IFACE  default = primary route iface
#           DELAY  default = 200ms
#           JITTER default = 50ms
# Revert: sudo ./network_delay.sh revert [IFACE]
#         (or: sudo tc qdisc del dev <IFACE> root)
set -euo pipefail

default_iface() { ip route show default | awk '/default/ {print $5; exit}'; }

if [[ "${1:-}" == "revert" ]]; then
  IFACE="${2:-$(default_iface)}"
  sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
  echo "[network_delay] reverted on $IFACE"
  exit 0
fi

IFACE="${1:-$(default_iface)}"
DELAY="${2:-200}"
JITTER="${3:-50}"

echo "[network_delay] adding ${DELAY}ms +/- ${JITTER}ms on $IFACE"
sudo tc qdisc add dev "$IFACE" root netem delay "${DELAY}ms" "${JITTER}ms" distribution normal
echo "[network_delay] active. Revert with: sudo $0 revert $IFACE"
