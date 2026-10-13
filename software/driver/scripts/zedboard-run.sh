#!/bin/sh
# ZedBoard with Petalinux: load the firmware and classify a cube, as
# driver_ntnu's process-capture.sh does on HYPSO-2:
#
#   ./zedboard-run.sh CUBE.npy [REFERENCE_FIXED.npy]
#
# 1. fpgautil loads the bitstream (.bit.bin) and its overlay (.dtbo): FPGA clock
#    and the two UIO devices (any overlay loaded before is removed first);
# 2. jopt_driver classifies CUBE.npy into driver_labels.npy and driver_scores.npy
#    next to it, and with a reference checks them bit for bit.
set -e
here=$(cd "$(dirname "$0")" && pwd)
kernel=justounetsimple_opt_zynq7020
input=$1
[ -n "$input" ] || { echo "usage: $0 CUBE.npy [REFERENCE_FIXED.npy]" >&2; exit 1; }

echo "== loading $kernel.bit.bin and $kernel.dtbo"
fpgautil -R >/dev/null 2>&1 || true
fpgautil -b "$here/$kernel.bit.bin" -o "$here/$kernel.dtbo"
for name in justounetsimple_opt jopt_memory; do
    grep -qx "$name" /sys/class/uio/*/name || { echo "no UIO device '$name' after loading the overlay" >&2; exit 1; }
done

echo "== classifying $input"
out=$(dirname "$input")
reference=""
[ -n "$2" ] && reference="--reference $2"
# shellcheck disable=SC2086
"$here/jopt_driver" --header "$here/x${kernel}_hw.h" --input "$input" \
    --labels "$out/driver_labels.npy" --scores "$out/driver_scores.npy" $reference
