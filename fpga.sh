# FPGA lab commands for this repository (bash). Load them once per terminal:
#
#     source fpga.sh
#     fpga_help
#
# Each command is a short wrapper; fpga_help shows what it runs, so you can
# also type that directly. Tab completes kernel names (and runs, for fpga_pynq).

FPGA_ROOT="${FPGA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FPGA_PYTHON="${FPGA_PYTHON:-python3}"

fpga_help() {
    cat <<'EOF'
FPGA commands. KERNEL is a folder name in src/hls/ (e.g. justounetsimple_opt):
press Tab to complete it. RUN is a folder in build/KERNEL/ and defaults to the
newest bitstream build.

  fpga_kernels               the kernels there are (python3 main.py kernels for their files)
  fpga_env [settings64.sh]   put Vivado and Vitis HLS on PATH (sources settings64.sh)
  fpga_test KERNEL           python3 -m unittest discover -s tests
                             python3 main.py native --kernel KERNEL
  fpga_build KERNEL [ARGS]   python3 main.py bitstream --kernel KERNEL --skip-cosim [ARGS]
                             (C++ -> IP -> block design -> .bit, then the PYNQ zip;
                             ARGS e.g. --board zynq7030 (default zcu104), --clock-ns 5)
  fpga_ip KERNEL [ARGS]      python3 main.py ip --kernel KERNEL --skip-cosim [ARGS]
                             (C++ -> IP, and stop: no Vivado, no .bit; for a block design
                             of your own. Same ARGS; the IP is in hls/solution/impl/ip/)
  fpga_cosim KERNEL [ARGS]   python3 main.py cosim --kernel KERNEL [ARGS]
                             (the testbench driving the generated Verilog: checks the
                             RTL, and its report gives the real cycle count; same ARGS,
                             and --dataflow-profile for the Dataflow Viewer)
  fpga_runs KERNEL           the runs in build/KERNEL/ with a bitstream, newest last
  fpga_pynq KERNEL [RUN]     python3 main.py pynq --kernel KERNEL [--run RUN]
                             (re-make the PYNQ folder and zip of an existing build)
  fpga_zip DIR               DIR -> DIR.zip, to upload an image next to the notebook

What each step does: docs/framework.md
EOF
}

_fpga_fail() { echo "fpga: $*" >&2; return 1; }

# The kernels: one folder each in src/hls/.
fpga_kernels() {
    local dir
    for dir in "$FPGA_ROOT"/src/hls/*/; do
        [ -d "$dir" ] && basename "$dir"
    done
}

# Run folders are named by date, so glob order is time order.
fpga_runs() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_runs KERNEL"; return; }
    local dir
    for dir in "$FPGA_ROOT/build/$1"/*/; do
        if compgen -G "${dir}vivado/system.runs/impl_1/*.bit" >/dev/null; then echo "${dir%/}"; fi
    done
}

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
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" -m unittest discover -s tests \
        && "$FPGA_PYTHON" main.py native --kernel "$1")
}

fpga_build() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_build KERNEL"; return; }
    local kernel="$1"
    shift
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py bitstream --kernel "$kernel" --skip-cosim "$@")
}

fpga_ip() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_ip KERNEL"; return; }
    local kernel="$1"
    shift
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py ip --kernel "$kernel" --skip-cosim "$@")
}

fpga_cosim() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_cosim KERNEL"; return; }
    local kernel="$1"
    shift
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py cosim --kernel "$kernel" "$@")
}

fpga_pynq() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_pynq KERNEL [RUN]"; return; }
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py pynq --kernel "$1" ${2:+--run "$(basename "$2")"})
}

fpga_zip() {
    local dir="${1%/}"
    [ -d "$dir" ] || { _fpga_fail "usage: fpga_zip DIR"; return; }
    (cd "$(dirname "$dir")" && rm -f "$(basename "$dir").zip" \
        && "$FPGA_PYTHON" -m zipfile -c "$(basename "$dir").zip" "$(basename "$dir")") || return
    echo "$dir.zip: upload it next to the notebook"
}

# ---- Tab completion (bash) ----
# fpga_test / fpga_build / fpga_ip / fpga_cosim / fpga_runs KERNEL, fpga_pynq KERNEL RUN, fpga_zip DIR.
_fpga_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}" prev="${COMP_WORDS[COMP_CWORD-1]}" words=""
    case "${COMP_WORDS[0]}:$COMP_CWORD" in
        fpga_test:1|fpga_build:1|fpga_ip:1|fpga_cosim:1|fpga_runs:1|fpga_pynq:1)
            words="$(fpga_kernels)" ;;
        fpga_pynq:2)
            local run
            for run in $(fpga_runs "${COMP_WORDS[1]}" 2>/dev/null); do words="$words ${run##*/}"; done ;;
        fpga_build:*|fpga_ip:*|fpga_cosim:*)
            if [ "$prev" = "--board" ]; then
                local script
                for script in "$FPGA_ROOT"/boards/*/system.tcl; do
                    [ -f "$script" ] && words="$words $(basename "$(dirname "$script")")"
                done
            elif [ "$prev" != "--clock-ns" ]; then
                words="--board --clock-ns"
                [ "${COMP_WORDS[0]}" = fpga_cosim ] && words="$words --dataflow-profile"
            fi ;;
    esac
    # shellcheck disable=SC2207
    COMPREPLY=($(compgen -W "$words" -- "$cur"))
}

if [ -n "${BASH_VERSION:-}" ]; then
    complete -F _fpga_complete fpga_test fpga_build fpga_ip fpga_cosim fpga_runs fpga_pynq
    complete -d fpga_zip
    complete -f fpga_env
fi
