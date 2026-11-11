#!/bin/bash
# test.sh KERNEL: the kernel's C++ testbench, compiled with g++ (or $CXX) and run. No AMD
# tools: what every change is checked with first. Output in build/KERNEL/native/.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 1 ] || die "usage: scripts/test.sh KERNEL"
find_kernel "$1"
CXX="${CXX:-g++}"
need "$CXX" "Install g++, or set CXX to another compiler."

out="$FPGA_ROOT/build/$KERNEL/native"
mkdir -p "$out"
program="$out/testbench"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) program="$program.exe" ;; esac
echo "> $CXX ... -o $program"
"$CXX" -std=c++14 -O2 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label \
    -I"$KERNEL_DIR" -I"$FPGA_ROOT/src/common" "${SOURCES[@]}" "${TESTBENCH[@]}" -o "$program" 2>&1 | tee "$out/compile.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "compiling failed; see $out/compile.log"
(cd "$out" && "$program") 2>&1 | tee "$out/testbench.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "the testbench failed; see $out/testbench.log"
echo "PASSED: $KERNEL"
