#!/bin/bash
# Everything the lab server does for the driver, in order, for both boards:
#
#   software/driver/scripts/server.sh                     all of it, bitstreams included
#   software/driver/scripts/server.sh --skip-bitstreams   use the newest bitstreams already built
#
#   1. the quantized weights, if they are missing (not in git)
#   2. the driver's tests (tests/test_driver.py)
#   3. the bitstreams: justounetsimple_opt (ZCU104), justounetsimple_opt_zynq7030 (HYPSO-2)
#   4. the cross compilers and cmake, if missing, then the driver for both ARMs
#   5. the packages: software/driver/build-zcu104/jopt_driver_zcu104.tar.gz and
#      software/driver/build-zynq7030/jopt_driver_zynq7030.tar.gz
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
"$driver/scripts/cross-compile.sh" zcu104
"$driver/scripts/cross-compile.sh" zynq7030

step "5/5 packages"
"$driver/scripts/package.sh" zcu104
"$driver/scripts/package.sh" zynq7030

step "DONE"
ls -l "$driver"/build-zcu104/jopt_driver_zcu104.tar.gz "$driver"/build-zynq7030/jopt_driver_zynq7030.tar.gz
