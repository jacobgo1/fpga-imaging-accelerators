# FPGA lab commands for this repository (bash). Load them once per terminal:
#
#     source fpga.sh
#     fpga_help
#
# Works from any directory. KERNEL defaults to justoliunet; RUN (a folder in
# artifacts/<kernel>/) defaults to the newest bitstream build.

FPGA_ROOT="${FPGA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
FPGA_PART="${FPGA_PART:-zcu104}"
FPGA_CONFIG="${FPGA_CONFIG:-config/project.json}"
FPGA_PYTHON="${FPGA_PYTHON:-python3}"

fpga_help() {
    cat <<'EOF'
FPGA commands (KERNEL: justoliunet or matmul, default justoliunet;
RUN: artifacts/<kernel>/<build> folder, default the newest build)

  Setup
    fpga_env [settings64.sh]          put Vivado's tools (vivado, xsdb) on PATH

  Test without hardware
    fpga_test [KERNEL]                repo tests + the kernel's C++ testbench

  Build (takes a while)
    fpga_build [KERNEL]               bitstream, without RTL co-simulation; also makes
                                      the PYNQ package (below)
    fpga_runs [KERNEL]                the bitstream builds, newest last

  Board with PYNQ  (Jupyter on the board: the normal way)
    fpga_pynq [KERNEL] [RUN]          the build's files for the notebook, in one folder
                                      and one zip: .bit, .hwh, driver, notebook, test data
    fpga_prepare CAPTURE.nc --labels LABELS.npy --out DIR [--step N | --crop R C H W]
                                      an image for the board: pixels, expected results, pictures
    fpga_zip DIR                      DIR -> DIR.zip, to upload next to the notebook
    fpga_compare DIR                  score DIR's FPGA results and redraw its pictures
    fpga_view DIR [PORT]              serve DIR's pictures to your laptop's browser

  Board over JTAG  (fallback: no Linux on the board; SW6 boot switches all ON = JTAG)
    fpga_selftest [KERNEL] [RUN]      program the board and run the kernel's self-test
    fpga_shell [KERNEL] [RUN]         program the board, then a live xsdb prompt
    fpga_classify DIR [RUN]           send DIR/pixels.txt through the FPGA, then score it

Details: docs/framework.md
EOF
}

_fpga_fail() { echo "fpga: $*" >&2; return 1; }

_fpga_need() {
    command -v "$1" >/dev/null 2>&1 || _fpga_fail "$1 not found; run fpga_env first"
}

# Newest build first-to-last: run folders are named by date, so glob order is time order.
fpga_runs() {
    local kernel="${1:-justoliunet}" dir
    for dir in "$FPGA_ROOT/artifacts/$kernel"/*/; do
        compgen -G "${dir}bitstream/*.bit" >/dev/null && echo "${dir%/}"
    done
}

_fpga_run() {
    local kernel="$1" run="$2"
    if [ -n "$run" ]; then
        [ -d "$run" ] && { (cd "$run" && pwd); return; }
        [ -d "$FPGA_ROOT/artifacts/$kernel/$run" ] && { echo "$FPGA_ROOT/artifacts/$kernel/$run"; return; }
        _fpga_fail "no build '$run' (see fpga_runs $kernel)"
        return
    fi
    run="$(fpga_runs "$kernel" | tail -n 1)"
    [ -n "$run" ] || { _fpga_fail "no $kernel bitstream in artifacts/$kernel; run fpga_build $kernel"; return; }
    echo "$run"
}

fpga_env() {
    if [ -z "$1" ] && command -v xsdb >/dev/null 2>&1; then
        echo "Vivado tools already on PATH: $(command -v xsdb)"
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
    local kernel="${1:-justoliunet}"
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" -m unittest discover -s tests \
        && "$FPGA_PYTHON" main.py native --kernel "$kernel")
}

fpga_build() {
    local kernel="${1:-justoliunet}"
    shift 2>/dev/null
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py bitstream --kernel "$kernel" --part "$FPGA_PART" \
        --config "$FPGA_CONFIG" --skip-cosim "$@")
}

fpga_selftest() {
    local kernel="${1:-justoliunet}" run
    _fpga_need xsdb || return
    run="$(_fpga_run "$kernel" "$2")" || return
    echo "Self-test of $run"
    xsdb "$FPGA_ROOT/software/jtag/$kernel.tcl" "$run"
}

fpga_shell() {
    local kernel="${1:-justoliunet}" run init
    _fpga_need xsdb || return
    run="$(_fpga_run "$kernel" "$2")" || return
    init="$(mktemp "${TMPDIR:-/tmp}/fpga_shell_XXXXXX.tcl")"
    {
        echo "source {$FPGA_ROOT/software/jtag/$kernel.tcl}"
        echo "board_open {$run}"
        case "$kernel" in
            justoliunet) echo 'puts "Try: jl_selftest | jl_vector 3 | jl_classify_file pixel.txt | kernel_status | exit"' ;;
            matmul)      echo 'puts "Try: mm_selftest | mm_live | mm_check \[mm_random\] \[mm_identity\] | kernel_status | exit"' ;;
        esac
    } > "$init"
    xsdb -interactive "$init"
}

fpga_prepare() {
    "$FPGA_PYTHON" "$FPGA_ROOT/tools/justoliunet_image.py" prepare "$@"
}

fpga_compare() {
    [ -n "$1" ] || { _fpga_fail "usage: fpga_compare DIR"; return; }
    "$FPGA_PYTHON" "$FPGA_ROOT/tools/justoliunet_image.py" compare "$1"
}

fpga_classify() {
    local dir="$1" run
    [ -f "$dir/pixels.txt" ] || { _fpga_fail "usage: fpga_classify DIR [RUN]; DIR needs pixels.txt from fpga_prepare"; return; }
    _fpga_need xsdb || return
    run="$(_fpga_run justoliunet "$2")" || return
    xsdb "$FPGA_ROOT/software/jtag/justoliunet.tcl" "$run" -pixels "$dir/pixels.txt" "$dir/fpga_logits.txt" \
        && fpga_compare "$dir"
}

fpga_view() {
    local dir="$1" port="${2:-8000}"
    [ -d "$dir" ] || { _fpga_fail "usage: fpga_view DIR [PORT]"; return; }
    cat <<EOF
Serving $dir on port $port. On your laptop:
    ssh -L $port:localhost:$port $(whoami)@$(hostname)
then open http://localhost:$port and click overview.png or capture_rgb.png.
Or copy the pictures instead:
    scp "$(whoami)@$(hostname):$(cd "$dir" && pwd)/*.png" .
Ctrl-C stops serving.
EOF
    "$FPGA_PYTHON" -m http.server "$port" --directory "$dir"
}

fpga_pynq() {
    local kernel="${1:-justoliunet}" run
    run="$(_fpga_run "$kernel" "$2")" || return
    (cd "$FPGA_ROOT" && "$FPGA_PYTHON" main.py pynq --kernel "$kernel" --run "$(basename "$run")") || return
    cat <<EOF

To the board (BOARD = its address; PYNQ's Jupyter is http://BOARD:9090, user/password xilinx):
  either upload $run/${kernel}_pynq.zip in Jupyter,
         then run  !unzip -o ${kernel}_pynq.zip  in a notebook cell,
  or     scp -r "$run/pynq" xilinx@BOARD:jupyter_notebooks/${kernel}
Then open ${kernel}.ipynb from that folder and run it from the top.
EOF
}

fpga_zip() {
    local dir="${1%/}"
    [ -d "$dir" ] || { _fpga_fail "usage: fpga_zip DIR"; return; }
    (cd "$(dirname "$dir")" && rm -f "$(basename "$dir").zip" \
        && "$FPGA_PYTHON" -m zipfile -c "$(basename "$dir").zip" "$(basename "$dir")") || return
    echo "$dir.zip: upload it next to the notebook"
}
