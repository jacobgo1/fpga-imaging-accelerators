# FPGA lab commands for this repository (bash). Load them once per terminal:
#
#     source fpga.sh
#     fpga_help
#
# Each command is a short wrapper; fpga_help shows what it runs, so you can
# also type that directly.

FPGA_ROOT="${FPGA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FPGA_PYTHON="${FPGA_PYTHON:-python3}"

fpga_help() {
    cat <<'EOF'
FPGA commands. KERNEL is a folder name in src/hls/ (e.g. justounetsimple_opt).
RUN is a folder in build/KERNEL/ and defaults to the newest bitstream build.

  fpga_env [settings64.sh]   put Vivado and Vitis HLS on PATH (sources settings64.sh)
  fpga_test KERNEL           python3 -m unittest discover -s tests
                             python3 main.py native --kernel KERNEL
  fpga_build KERNEL          python3 main.py bitstream --kernel KERNEL --skip-cosim
                             (C++ -> IP -> block design -> .bit, then the PYNQ zip)
  fpga_runs KERNEL           the runs in build/KERNEL/ with a bitstream, newest last
  fpga_pynq KERNEL [RUN]     python3 main.py pynq --kernel KERNEL [--run RUN]
                             (re-make the PYNQ folder and zip of an existing build)
  fpga_zip DIR               DIR -> DIR.zip, to upload an image next to the notebook

What each step does: docs/framework.md
EOF
}

_fpga_fail() { echo "fpga: $*" >&2; return 1; }

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
