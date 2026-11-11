#!/bin/bash
# hls.sh KERNEL BOARD [STAGE] [--cosim] [--dataflow-profile] [--clock-ns NS]
#
# Vitis HLS on KERNEL for BOARD's part and clock (boards/BOARD/board.conf), in
# build/KERNEL/BOARD/. Each stage includes the ones above it:
#
#   csim     the testbench, compiled by HLS
#   csynth   ... then C++ -> Verilog, with the latency and resource report
#   cosim    ... then the testbench driving that Verilog (the real cycle count)
#   ip       ... then the IP, as build/KERNEL/BOARD/export.zip (the default)
#
# export.zip is all the later steps need (pynq.sh, opu.sh): one built by someone else
# works as well. --cosim: ip also runs cosim first (slow). --dataflow-profile: cosim also
# records each DATAFLOW process's stalls, for the Dataflow Viewer.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage="usage: scripts/hls.sh KERNEL BOARD [csim|csynth|cosim|ip] [--cosim] [--dataflow-profile] [--clock-ns NS]"
[ $# -ge 2 ] || die "$usage"
find_kernel "$1"
load_board "$2"
shift 2
stage=ip cosim=0 profile=0
while [ $# -gt 0 ]; do
    case "$1" in
        csim|csynth|cosim|ip) stage="$1" ;;
        --cosim) cosim=1 ;;
        --dataflow-profile) profile=1 ;;
        --clock-ns) CLOCK_NS="$2"; shift ;;
        *) die "$usage" ;;
    esac
    shift
done
[ "$stage" = cosim ] && cosim=1

out="$(build_dir)"
mkdir -p "$out"
rm -f "$out/export.zip"
# What hls.tcl needs, as the Tcl array cfg(...).
{
    echo "set cfg(run_dir) $(tcl_word "$out")"
    echo "set cfg(stage) $stage"
    echo "set cfg(top) $KERNEL"
    echo "set cfg(part) $PART"
    echo "set cfg(clock_ns) $CLOCK_NS"
    echo "set cfg(cosim) $cosim"
    echo "set cfg(profile) $profile"
    printf 'set cfg(sources) [list'; for f in "${SOURCES[@]}"; do printf ' %s' "$(tcl_word "$f")"; done; echo ']'
    printf 'set cfg(testbench) [list'; for f in "${TESTBENCH[@]}"; do printf ' %s' "$(tcl_word "$f")"; done; echo ']'
    echo "set cfg(include_dirs) [list $(tcl_word "$KERNEL_DIR") $(tcl_word "$FPGA_ROOT/src/common")]"
} > "$out/settings.tcl"
cp "$FPGA_ROOT/scripts/hls.tcl" "$out/"
# What was built, for the later steps and for comparing builds.
{
    echo "kernel: $KERNEL"
    echo "board: $BOARD"
    echo "part: $PART"
    echo "clock_ns: $CLOCK_NS"
    echo "stage: $stage"
    echo "git: $(git -C "$FPGA_ROOT" describe --always --dirty 2>/dev/null || echo unknown)"
    echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
} > "$out/build.txt"

echo "$stage $KERNEL: $BOARD ($PART), clock $CLOCK_NS ns, in $out/"
cd "$out"
# 2024.x calls Vitis HLS through vitis-run; 2023.x and older only have vitis_hls.
if command -v vitis-run >/dev/null 2>&1; then
    command=(vitis-run --mode hls --tcl hls.tcl)
else
    need vitis_hls
    command=(vitis_hls -f hls.tcl)
fi
"${command[@]}" 2>&1 | tee hls.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "Vitis HLS failed; see $out/hls.log"

report="$out/hls/solution/syn/report/${KERNEL}_csynth.rpt"
[ -f "$report" ] && echo "HLS estimates (latency, resources): $report"
[ -f "$out/hls/solution/sim/report/${KERNEL}_cosim.rpt" ] && echo "cosim: $out/hls/solution/sim/report/${KERNEL}_cosim.rpt"
if [ "$stage" = ip ]; then
    archive="$(ls "$out"/hls/solution/impl/ip/*.zip | head -n 1)"
    [ -f "$archive" ] || die "no IP archive in $out/hls/solution/impl/ip/"
    cp "$archive" "$out/export.zip"
    echo "The IP: $out/export.zip"
    case "$FLOW" in
        pynq) echo "Next: scripts/pynq.sh $KERNEL $BOARD" ;;
        opu) echo "Next: scripts/opu.sh $KERNEL $BOARD" ;;
    esac
fi
echo "PASSED: $stage $KERNEL $BOARD"
