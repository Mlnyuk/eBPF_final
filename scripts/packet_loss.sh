#!/usr/bin/env bash
# packet_loss.sh — drop egress packets with tc netem (spikes TCP retransmits).
#
# Tool: tc (iproute2)
# Run:    sudo ./packet_loss.sh [IFACE] [LOSS_PCT]
#           IFACE default = primary route iface
#           LOSS  default = 15  (percent)
# Revert: sudo ./packet_loss.sh revert [IFACE]
set -euo pipefail

default_iface() { ip route show default | awk '/default/ {print $5; exit}'; }

if [[ "${1:-}" == "revert" ]]; then
  IFACE="${2:-$(default_iface)}"
  sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
  echo "[packet_loss] reverted on $IFACE"
  exit 0
fi

IFACE="${1:-$(default_iface)}"
LOSS="${2:-15}"

echo "[packet_loss] dropping ${LOSS}% on $IFACE"
sudo tc qdisc add dev "$IFACE" root netem loss "${LOSS}%"
echo "[packet_loss] active. Revert with: sudo $0 revert $IFACE"
