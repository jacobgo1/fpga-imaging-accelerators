#!/bin/bash
# build.sh KERNEL BOARD [--cosim] [--clock-ns NS]: everything from the C++ to what goes
# on the board, in build/KERNEL/BOARD/:
#
#   scripts/hls.sh KERNEL BOARD ip     the IP, export.zip (cosim only with --cosim)
#   then, by the board's FLOW (boards/BOARD/board.conf):
#   scripts/pynq.sh KERNEL BOARD       zcu104, pynqz2: KERNEL_pynq.zip for the notebook
#   scripts/opu.sh KERNEL BOARD        picobob, zedboard: KERNEL_opu.tar.gz for the OPU system
#
# With someone else's export.zip, put it in build/KERNEL/BOARD/ and run the second step only.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || die "usage: scripts/build.sh KERNEL BOARD [--cosim] [--clock-ns NS]"
find_kernel "$1"
load_board "$2"
"$FPGA_ROOT/scripts/hls.sh" "$@"
"$FPGA_ROOT/scripts/$FLOW.sh" "$KERNEL" "$BOARD"
