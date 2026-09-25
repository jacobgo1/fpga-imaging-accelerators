#!/usr/bin/env bash
# Drop a cloud-init NoCloud seed onto a freshly-flashed Ubuntu boot
# partition, so the board comes up on the network with your SSH key
# already trusted -- no monitor, keyboard or serial cable needed to log
# in for the first time.
#
# Only applies to Canonical's official Ubuntu Server images for Xilinx
# boards (ubuntu.com/download/amd-xilinx): they ship cloud-init's NoCloud
# datasource pointed at the boot partition, the same mechanism Ubuntu uses
# on Raspberry Pi images. If you're on a different/custom image, check
# whether its boot partition already has user-data/meta-data files before
# assuming this applies.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: make-boot-seed.sh <boot-partition-mountpoint> <ssh-pubkey-file> \
           [hostname] [static-ip/cidr] [gateway]

Run with the SD card's boot partition mounted (it auto-mounts as
"system-boot" on most desktops after flashing).

Examples:
  make-boot-seed.sh /Volumes/system-boot ~/.ssh/id_ed25519.pub
  make-boot-seed.sh /media/$USER/system-boot ~/.ssh/id_ed25519.pub \
      zcu104 192.168.1.50/24 192.168.1.1

No static IP given -> DHCP; find the board afterwards with
'ssh ubuntu@<hostname>.local' (mDNS/avahi, works out of the box on most
LANs) or by checking your router's DHCP lease list.
EOF
}

boot="${1:-}"; pubkey="${2:-}"; hostname="${3:-zcu104}"; static_ip="${4:-}"; gateway="${5:-}"
if [ -z "$boot" ] || [ -z "$pubkey" ]; then usage; exit 2; fi
if [ ! -d "$boot" ]; then echo "Error: not a directory: $boot" >&2; exit 1; fi
if [ ! -f "$pubkey" ]; then echo "Error: pubkey file not found: $pubkey" >&2; exit 1; fi

cat > "$boot/user-data" <<EOF
#cloud-config
hostname: $hostname
ssh_pwauth: false
users:
  - name: ubuntu
    ssh_authorized_keys:
      - $(cat "$pubkey")
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
package_update: true
packages:
  - build-essential
  - git
EOF

cat > "$boot/meta-data" <<EOF
instance-id: ${hostname}-$(date +%s)
local-hostname: $hostname
EOF

if [ -n "$static_ip" ]; then
    {
        echo "version: 2"
        echo "ethernets:"
        echo "  eth0:"
        echo "    dhcp4: false"
        echo "    addresses: [$static_ip]"
        [ -n "$gateway" ] && echo "    gateway4: $gateway"
        echo "    nameservers:"
        echo "      addresses: [8.8.8.8, 1.1.1.1]"
    } > "$boot/network-config"
    echo "Wrote static network-config: $static_ip${gateway:+ via $gateway}"
else
    echo "No static IP given; board will use DHCP."
fi

echo "Seed written to $boot."
echo "Eject the card, boot the ZCU104, then within a minute or two:"
echo "  ssh ubuntu@${hostname}.local     # or its DHCP-assigned IP"
