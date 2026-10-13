#!/bin/bash
# Everything the lab server does for the driver, in order, for every board:
#
#   software/driver/scripts/server.sh                     all of it, bitstreams included
#   software/driver/scripts/server.sh --skip-bitstreams   use the newest bitstreams already built
#
#   1. the quantized weights, if they are missing (not in git)
#   2. the driver's tests (tests/test_driver.py)
#   3. the bitstreams: justounetsimple_opt (ZCU104), justounetsimple_opt_zynq7030 (HYPSO-2),
#      justounetsimple_opt_zynq7020 (ZedBoard)
#   4. the cross compilers and cmake, if missing, then the driver for each board's ARM
#   5. the packages: software/driver/build-BOARD/jopt_driver_BOARD.tar.gz for
#      zcu104, zynq7030 and zynq7020
#
# It stops at the first step that fails. Everything it prints also goes to
# software/driver/server.log. Bitstreams take long: run it with nohup (see README.md).
set -e
build_bitstreams=1
[ "$1" = "--skip-bitstreams" ] && build_bitstreams=0

driver=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$driver/../.." && pwd)
cd "$repo"
exec > >(tee -a "$driver/server.log") 2>&1
step() { echo; echo "==== $(date '+%H:%M:%S')  $*"; }
trap 'echo; echo "==== FAILED in the step above. Send the last lines of software/driver/server.log."' ERR

step "1/5 weights"
if [ -f weights/quantized/justounetsimple/justounetsimple_int8.npz ]; then
    echo "weights/quantized/justounetsimple/ is there"
else
    python3 tools/quantize_weights.py weights/fp32/justounetsimple
fi

step "2/5 the driver's tests"
python3 -m unittest tests.test_driver -v

source fpga.sh
fpga_env

step "3/5 bitstreams"
if [ "$build_bitstreams" = 1 ]; then
    fpga_build justounetsimple_opt
    fpga_build justounetsimple_opt_zynq7030 --board zynq7030
    fpga_build justounetsimple_opt_zynq7020 --board zynq7020
else
    echo "skipped (--skip-bitstreams): packaging the newest existing ones"
fi

step "4/5 cross-compile"
# cmake 3.13 or newer; without root, pip installs one in ~/.local/bin.
if ! cmake --version >/dev/null 2>&1 || \
   [ "$(printf '%s\n' 3.13 "$(cmake --version | head -n 1 | awk '{print $3}')" | sort -V | head -n 1)" != 3.13 ]; then
    python3 -m pip install --user cmake
    export PATH="$HOME/.local/bin:$PATH"
fi
cmake --version | head -n 1
# The cross compilers: on PATH, else Docker (cross-compile.sh uses it), else
# the ones that come with the AMD tools.
for cxx in aarch64-linux-gnu-g++ arm-linux-gnueabihf-g++; do
    if command -v "$cxx" >/dev/null; then
        echo "$cxx: $(command -v "$cxx")"
    elif docker info >/dev/null 2>&1; then
        echo "$cxx: not on PATH, cross-compile.sh will use Docker"
    else
        found=$(find /tools/Xilinx /opt/Xilinx -name "$cxx" -type f 2>/dev/null | head -n 1)
        if [ -z "$found" ]; then
            echo "no $cxx: install it (sudo apt install g++-aarch64-linux-gnu g++-arm-linux-gnueabihf) or Docker"
            false
        fi
        export PATH="$(dirname "$found"):$PATH"
        echo "$cxx: $found (from the AMD tools)"
    fi
done
for board in zcu104 zynq7030 zynq7020; do
    "$driver/scripts/cross-compile.sh" "$board"
done

step "5/5 packages"
for board in zcu104 zynq7030 zynq7020; do
    # With --skip-bitstreams, a board without a bitstream yet is not a failure.
    if "$driver/scripts/package.sh" "$board"; then
        :
    elif [ "$build_bitstreams" = 0 ]; then
        echo "no package for $board: build its bitstream first (server.sh without --skip-bitstreams)"
    else
        false
    fi
done

step "DONE"
ls -l "$driver"/build-*/jopt_driver_*.tar.gz
