#!/usr/bin/env bash
# Write a ZCU104 Ubuntu image to an SD card from the command line -- no
# Etcher, no GUI. Get AMD/Canonical's prebuilt Ubuntu Server image for
# ZCU104 first (ubuntu.com/download/amd-xilinx has the current release);
# this script just writes whatever image file you point it at.
#
# Destructive by nature (it overwrites a whole disk), so it refuses to do
# anything until you pass --yes, and then asks you to retype the device
# path as a second confirmation. Get the device wrong and you will
# overwrite the wrong disk -- there is no undo.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: flash-sdcard.sh <image.img|.img.xz|.wic|.wic.xz|.wic.gz> <device> [--yes]

<device> must be the WHOLE disk, never a partition (no trailing number):
  macOS:  /dev/diskN        (find it with: diskutil list)
  Linux:  /dev/sdX or /dev/mmcblkN   (find it with: lsblk)

Without --yes this only prints the plan and exits -- run it once without
--yes first and check the plan before you commit to it.
EOF
}

image=""; device=""; confirmed=0
for arg in "$@"; do
    case "$arg" in
        --yes) confirmed=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [ -z "$image" ]; then image="$arg"
            elif [ -z "$device" ]; then device="$arg"
            else echo "Unexpected argument: $arg" >&2; usage; exit 2; fi
            ;;
    esac
done
if [ -z "$image" ] || [ -z "$device" ]; then usage; exit 2; fi
if [ ! -f "$image" ]; then echo "Error: image not found: $image" >&2; exit 1; fi
case "$device" in
    *[0-9]s[0-9]*|*p[0-9]|*[0-9][0-9])
        echo "Error: '$device' looks like a partition, not a whole disk. See --help." >&2; exit 1 ;;
esac

os="$(uname -s)"
echo "== Plan =="
echo "  image : $image"
echo "  device: $device ($os)"
case "$os" in
    Darwin)
        diskutil info "$device" >/dev/null 2>&1 || {
            echo "Error: diskutil does not recognize $device" >&2; exit 1; }
        diskutil info "$device" | sed -n 's/^ *\(Device Node\|Device \/ Media Name\|Removable Media\):.*/&/p'
        ;;
    Linux)
        command -v lsblk >/dev/null || { echo "Error: lsblk not found" >&2; exit 1; }
        lsblk "$device" || { echo "Error: lsblk does not recognize $device" >&2; exit 1; }
        if lsblk -no MOUNTPOINT "$device"* 2>/dev/null | grep -qx '/'; then
            echo "Error: $device has this machine's root filesystem on it. Refusing." >&2
            exit 1
        fi
        ;;
    *) echo "Error: unsupported OS: $os" >&2; exit 1 ;;
esac

if [ "$confirmed" -ne 1 ]; then
    echo
    echo "Dry run only (no --yes given). Nothing was written."
    exit 0
fi

echo
read -r -p "This ERASES $device. Type the device path again to confirm: " typed
if [ "$typed" != "$device" ]; then
    echo "Confirmation did not match; aborting." >&2
    exit 1
fi

raw_device="$device"
case "$os" in
    Darwin)
        diskutil unmountDisk "$device"
        raw_device="${device/\/dev\/disk//dev\/rdisk}"   # raw device: much faster on macOS
        ;;
    Linux)
        for part in "$device"*[0-9]; do
            [ -e "$part" ] && umount "$part" 2>/dev/null || true
        done
        ;;
esac

decompress=(cat)
case "$image" in
    *.xz)  command -v xz    >/dev/null || { echo "Need xz (brew/apt install xz)" >&2; exit 1; }; decompress=(xz -dc) ;;
    *.gz)  command -v gzip  >/dev/null || { echo "Need gzip" >&2; exit 1; };                       decompress=(gzip -dc) ;;
    *.zst) command -v zstd  >/dev/null || { echo "Need zstd" >&2; exit 1; };                       decompress=(zstd -dc) ;;
esac

echo "Writing $image to $raw_device ..."
if command -v bmaptool >/dev/null 2>&1 && { [ -f "${image}.bmap" ] || [ -f "${image%.xz}.bmap" ]; }; then
    sudo bmaptool copy "$image" "$raw_device"
else
    "${decompress[@]}" "$image" | sudo dd of="$raw_device" bs=4M 2>&1
fi
sync
echo "Done. Wait for your OS to report the card is safe to remove, then eject it."
echo "Next: software/zcu104/make-boot-seed.sh to preload SSH access before first boot."
