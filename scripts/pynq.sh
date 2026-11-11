#!/bin/bash
# pynq.sh KERNEL BOARD [--package]: a board with PYNQ (zcu104, pynqz2). From
# build/KERNEL/BOARD/export.zip (scripts/hls.sh, or someone else's): Vivado with
# scripts/vivado.tcl and boards/BOARD/system.tcl, then the zip for the board's notebook:
#
#   build/KERNEL/BOARD/KERNEL_pynq.zip   KERNEL.bit, KERNEL.hwh (PYNQ finds it by the .bit's
#                                        name), xKERNEL_hw.h (register offsets), the notebook,
#                                        kernel.json (which model), mu_sd.txt (golden kernels)
#
# --package: only the zip again, from the last bitstream (e.g. after editing the notebook).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage="usage: scripts/pynq.sh KERNEL BOARD [--package]"
[ $# -ge 2 ] && [ $# -le 3 ] || die "$usage"
find_kernel "$1"
load_board "$2"
[ "$FLOW" = pynq ] || die "$BOARD runs the OPU system, not PYNQ: scripts/opu.sh $KERNEL $BOARD"
package_only=0
[ $# -eq 3 ] && { [ "$3" = --package ] || die "$usage"; package_only=1; }
out="$(build_dir)"

if [ $package_only -eq 0 ]; then
    need vivado
    unpack_export "$out/ip"
    clock_ns="$(build_info clock_ns)"
    {
        echo "set cfg(root) $(tcl_word "$FPGA_ROOT")"
        echo "set cfg(run_dir) $(tcl_word "$out")"
        echo "set cfg(ip_repo) $(tcl_word "$out/ip")"
        echo "set cfg(top) $KERNEL"
        echo "set cfg(board) $BOARD"
        echo "set cfg(part) $PART"
        echo "set cfg(clock_ns) ${clock_ns:-$CLOCK_NS}"
        echo "set cfg(jobs) $(job_count)"
        echo "set cfg(board_script) $(tcl_word "$FPGA_ROOT/boards/$BOARD/system.tcl")"
    } > "$out/vivado-settings.tcl"
    cp "$FPGA_ROOT/scripts/vivado.tcl" "$out/"
    rm -rf "$out/vivado" "$out/reports"
    echo "Vivado: $KERNEL on $BOARD, in $out/"
    (cd "$out" && vivado -mode batch -source vivado.tcl -notrace -log vivado.log -journal vivado.jou) 2>&1 \
        | tee "$out/vivado-console.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "Vivado failed; see $out/vivado-console.log"
fi

# The package.
bit="$(ls "$out"/vivado/system.runs/impl_1/*.bit 2>/dev/null | head -n 1)"
hwh="$(find "$out/vivado" -path '*hw_handoff/system.hwh' 2>/dev/null | head -n 1)"
header="$(ls "$out"/ip/drivers/*/src/*_hw.h 2>/dev/null | head -n 1)"
[ -f "$bit" ] || die "no bitstream in $out/vivado/system.runs/impl_1/"
[ -f "$hwh" ] || die "no block design description (system.hwh) in $out/vivado/"
[ -f "$header" ] || die "no register map (*_hw.h) in $out/ip/drivers/"
package="$out/${KERNEL}_pynq"
rm -rf "$package" "$package.zip"
mkdir -p "$package"
cp "$bit" "$package/$KERNEL.bit"
cp "$hwh" "$package/$KERNEL.hwh"
cp "$header" "$package/"
python="${FPGA_PYTHON:-python3}"
if "$python" "$FPGA_ROOT/tools/models.py" kernel-json "$KERNEL" > "$package/kernel.json"; then
    # One notebook for every model, the golden kernels' or the optimized ones'; kernel.json
    # tells it which model this is. The golden kernels preprocess on the ARM: mu_sd.txt.
    model="$(sed -n 's/.*"model": "\(.*\)".*/\1/p' "$package/kernel.json")"
    if [ "$model" = "$KERNEL" ]; then
        cp "$FPGA_ROOT/software/pynq/classify_golden.ipynb" "$FPGA_ROOT/mu_sd.txt" "$package/"
    else
        cp "$FPGA_ROOT/software/pynq/classify.ipynb" "$package/"
    fi
else
    rm "$package/kernel.json"
fi
# A kernel's own notebook and files, if it has any.
[ -d "$FPGA_ROOT/software/pynq/$KERNEL" ] && cp "$FPGA_ROOT/software/pynq/$KERNEL"/* "$package/"
(cd "$out" && "$python" -m zipfile -c "${KERNEL}_pynq.zip" "${KERNEL}_pynq")

echo
echo "For the board: $package.zip"
echo "  Upload it in Jupyter (http://BOARD:9090), run  !unzip -o ${KERNEL}_pynq.zip  in a cell,"
for notebook in "$package"/*.ipynb; do
    [ -f "$notebook" ] && echo "  then open ${KERNEL}_pynq/$(basename "$notebook") and run it from the top."
done
true
