# FPGA lab commands for this repository (bash). Load them once per terminal:
#
#     source fpga.sh
#     fpga_help
#
# Each command is a short wrapper around a script in scripts/; fpga_help shows which, so
# you can also type that directly. Tab completes kernel and board names.

FPGA_ROOT="${FPGA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FPGA_PYTHON="${FPGA_PYTHON:-python3}"

fpga_help() {
    cat <<'EOF'
FPGA commands. KERNEL is a kernel's .cpp name in src/models/ (e.g. justounetsimple_opt),
BOARD one of boards/: zcu104 (default), pynqz2 (PYNQ), picobob, zedboard (OPU system).
Tab completes both. Everything for KERNEL on BOARD goes to build/KERNEL/BOARD/.

  fpga_kernels                  the kernels and boards there are       scripts/kernels.sh
  fpga_env [settings64.sh]      put Vivado and Vitis HLS on PATH (sources settings64.sh)
  fpga_test KERNEL              Python tests + the kernel's testbench   scripts/test.sh KERNEL
                                with g++ (no AMD tools)
  fpga_build KERNEL [BOARD]     C++ -> IP -> bitstream -> the board's   scripts/build.sh KERNEL BOARD
                                package (PYNQ zip or OPU tarball)
  fpga_ip KERNEL [BOARD]        C++ -> IP (build/KERNEL/BOARD/export.zip) and stop
                                                                       scripts/hls.sh KERNEL BOARD ip
  fpga_bitstream KERNEL [BOARD] from an existing export.zip: bitstream and package
                                                                       scripts/pynq.sh / opu.sh KERNEL BOARD
  fpga_cosim KERNEL [BOARD]     the testbench driving the RTL (checks it, and its report
                                has the real cycle count)              scripts/hls.sh KERNEL BOARD cosim
  fpga_package KERNEL [BOARD]   the package again without rebuilding (after editing a
                                notebook or classify)                  scripts/pynq.sh ... --package
                                                                       / scripts/opu.sh ... --driver
  fpga_zip DIR                  DIR -> DIR.zip, an image to upload once, in the folder above
                                the designs' folders (or next to one notebook)

fpga_build, fpga_ip and fpga_cosim also take --clock-ns NS; fpga_cosim --dataflow-profile.
What each step does: docs/framework.md; the OPU boards: docs/opu.md
EOF
}

_fpga_fail() { echo "fpga: $*" >&2; return 1; }

fpga_kernels() { "$FPGA_ROOT/scripts/kernels.sh"; }

fpga_env() {
    if [ -z "$1" ] && command -v vivado >/dev/null 2>&1; then
        echo "Vivado already on PATH: $(command -v vivado)"
        return
    fi
    local settings="${1:-$FPGA_VIVADO_SETTINGS}"
    if [ -z "$settings" ]; then
        settings="$(ls -1 /tools/Xilinx/Vivado/*/settings64.sh /tools/Xilinx/*/Vivado/settings64.sh \
                          /opt/Xilinx/Vivado/*/settings64.sh /opt/Xilinx/*/Vivado/settings64.sh \
                          2>/dev/null | sort -V | tail -n 1)"
    fi
    [ -f "$settings" ] || { _fpga_fail "no Vivado settings64.sh found; pass its path: fpga_env /path/to/settings64.sh"; return; }
    # shellcheck disable=SC1090
    source "$settings"
    echo "Loaded $settings"
}

fpga_test() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_test KERNEL"; return; }
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" -m unittest discover -s tests && scripts/test.sh "$1")
}

# KERNEL [BOARD] [ARGS]: the board defaults to zcu104 when the second word is an option.
_fpga_board_args() {
    local kernel="$1"
    shift
    if [ -z "$1" ] || [ "${1#--}" != "$1" ]; then set -- zcu104 "$@"; fi
    echo "$kernel" "$@"
}

fpga_build() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_build KERNEL [BOARD] [--cosim] [--clock-ns NS]"; return; }
    # shellcheck disable=SC2046
    "$FPGA_ROOT/scripts/build.sh" $(_fpga_board_args "$@")
}

fpga_ip() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_ip KERNEL [BOARD] [--cosim] [--clock-ns NS]"; return; }
    # shellcheck disable=SC2046
    "$FPGA_ROOT/scripts/hls.sh" $(_fpga_board_args "$@") ip
}

fpga_cosim() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_cosim KERNEL [BOARD] [--dataflow-profile] [--clock-ns NS]"; return; }
    # shellcheck disable=SC2046
    "$FPGA_ROOT/scripts/hls.sh" $(_fpga_board_args "$@") cosim
}

# The board's flow, from its board.conf.
_fpga_flow() {
    (FLOW=""; source "$FPGA_ROOT/boards/$1/board.conf" 2>/dev/null && echo "$FLOW")
}

fpga_bitstream() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_bitstream KERNEL [BOARD]"; return; }
    local board="${2:-zcu104}" flow
    flow="$(_fpga_flow "$board")"
    [ -n "$flow" ] || { _fpga_fail "no board '$board'"; return; }
    "$FPGA_ROOT/scripts/$flow.sh" "$1" "$board"
}

fpga_package() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_package KERNEL [BOARD]"; return; }
    local board="${2:-zcu104}" flow
    flow="$(_fpga_flow "$board")"
    case "$flow" in
        pynq) "$FPGA_ROOT/scripts/pynq.sh" "$1" "$board" --package ;;
        opu) "$FPGA_ROOT/scripts/opu.sh" "$1" "$board" --driver ;;
        *) _fpga_fail "no board '$board'" ;;
    esac
}

fpga_zip() {
    local dir="${1%/}"
    [ -d "$dir" ] || { _fpga_fail "usage: fpga_zip DIR"; return; }
    (cd "$(dirname "$dir")" && rm -f "$(basename "$dir").zip" \
        && "$FPGA_PYTHON" -m zipfile -c "$(basename "$dir").zip" "$(basename "$dir")") || return
    echo "$dir.zip: upload it to the folder above the designs' folders (or next to the notebook)"
}

# ---- Tab completion (bash) ----
_fpga_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}" prev="${COMP_WORDS[COMP_CWORD-1]}" words=""
    case "${COMP_WORDS[0]}:$COMP_CWORD" in
        fpga_test:1|fpga_build:1|fpga_ip:1|fpga_cosim:1|fpga_bitstream:1|fpga_package:1)
            words="$(find "$FPGA_ROOT/src/models" -name '*.cpp' -exec basename {} .cpp \;)" ;;
        fpga_build:2|fpga_ip:2|fpga_cosim:2|fpga_bitstream:2|fpga_package:2)
            words="$(for c in "$FPGA_ROOT"/boards/*/board.conf; do basename "$(dirname "$c")"; done)" ;;
        fpga_build:*|fpga_ip:*|fpga_cosim:*)
            if [ "$prev" != "--clock-ns" ]; then
                words="--clock-ns"
                [ "${COMP_WORDS[0]}" = fpga_cosim ] && words="$words --dataflow-profile"
                [ "${COMP_WORDS[0]}" != fpga_cosim ] && words="$words --cosim"
            fi ;;
    esac
    # shellcheck disable=SC2207
    COMPREPLY=($(compgen -W "$words" -- "$cur"))
}

if [ -n "${BASH_VERSION:-}" ]; then
    complete -F _fpga_complete fpga_test fpga_build fpga_ip fpga_cosim fpga_bitstream fpga_package
    complete -d fpga_zip
    complete -f fpga_env
fi
