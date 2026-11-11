#!/bin/sh
# load.sh: this kernel's bitstream and overlay onto the FPGA, as root, for a board you
# reach over ssh (the lab's ZedBoards). On HYPSO, hypso-cli does the same:
#   obip fpga reload KERNEL.bit.bin -o KERNEL.dtbo
# The kernel is this folder's name; its firmware is in /media/sd-pl/firmware (the
# tarball scripts/opu.sh makes, unpacked in /media/sd-pl). Until the next reboot.
NAME="$(basename "$(cd "$(dirname "$0")" && pwd)")"
FIRMWARE="${FIRMWARE:-/media/sd-pl/firmware}"
set -e
cd "$FIRMWARE"
md5sum -c "$NAME.bit.bin.md5sum"
# An overlay already in the full region (the one loaded at boot) blocks a new one.
for overlay in /configfs/device-tree/overlays/* /sys/kernel/config/device-tree/overlays/*; do
    if [ -d "$overlay" ]; then echo "removing the overlay $overlay"; rmdir "$overlay"; fi
done
fpgautil -b "$NAME.bit.bin" -o "$NAME.dtbo"
echo "FPGA: $(cat /sys/class/fpga_manager/fpga0/state)"
found=""
for uio in /sys/class/uio/uio*; do
    [ "$(cat "$uio/name")" = hls_kernel ] && found="$uio"
done
[ -n "$found" ] || { echo "no UIO device hls_kernel: the overlay did not load" >&2; exit 1; }
echo "$NAME loaded: $found"
