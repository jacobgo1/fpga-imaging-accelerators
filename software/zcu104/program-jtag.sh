#!/usr/bin/env bash
# Program the ZCU104's PL over JTAG using this machine's own Vivado --
# no SD card, no GUI. Run this on the server the board's USB-JTAG cable is
# actually plugged into, with Vivado's settings64.sh already sourced.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: program-jtag.sh <bitfile.bit> [hw_server_host:port]

hw_server_host:port defaults to localhost:3121 (Vivado's bundled hw_server;
it's started automatically if nothing is listening there yet).

Example:
  ./program-jtag.sh ../../artifacts/justoliunet/2026-.../deliverables/bitstream/system.bit
EOF
}

bitfile="${1:-}"; hw_server="${2:-localhost:3121}"
case "$bitfile" in ""|-h|--help) usage; exit $([ -z "$bitfile" ] && echo 2 || echo 0) ;; esac
[ -f "$bitfile" ] || { echo "Error: bitfile not found: $bitfile" >&2; exit 1; }
command -v vivado >/dev/null || {
    echo "Error: vivado not on PATH -- source Vivado's settings64.sh in this shell first" >&2
    exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vivado -mode batch -source "$script_dir/program-jtag.tcl" -notrace \
    -tclargs "$bitfile" "$hw_server"
