#!/bin/bash
# Everything a board needs to run jopt_driver, as one archive:
#   software/driver/scripts/package.sh zcu104       ZCU104 with PYNQ
#   software/driver/scripts/package.sh zynq7030     HYPSO-2
#   software/driver/scripts/package.sh zynq7020     ZedBoard with Petalinux
# takes the newest bitstream run of that board's kernel (justounetsimple_opt,
# justounetsimple_opt_zynq7030, justounetsimple_opt_zynq7020), or the run folder
# given as a second argument, and writes
#   software/driver/build-BOARD/jopt_driver_BOARD.tar.gz
# with the driver and the register header of that run, and its firmware as the
# board loads it:
#   zcu104    the .bit and .hwh for PYNQ, and jopt_driver.ipynb
#   zynq7030  the .bit, the device tree snippet and process-capture.sh
#   zynq7020  the .bit.bin and .dtbo for fpgautil (made here with bootgen and
#             dtc, as driver_ntnu's build-firmware.sh makes them with
#             Petalinux), and zedboard-run.sh
# Cross-compile first (cross-compile.sh).
set -e
board=$1
case "$board" in
    zynq7030) kernel=justounetsimple_opt_zynq7030; suffix='-bitstream-zynq7030' ;;
    zynq7020) kernel=justounetsimple_opt_zynq7020; suffix='-bitstream-zynq7020' ;;
    zcu104)   kernel=justounetsimple_opt;          suffix='-bitstream' ;;
    *) echo "usage: $0 zcu104|zynq7030|zynq7020 [RUN]" >&2; exit 1 ;;
esac
driver=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$driver/../.." && pwd)

run=$2
if [ -z "$run" ]; then
    # Run folders are DATE-STAGE[-BOARD][-N]; the default board has no -BOARD.
    # Names start with the date, so the newest sorts last.
    while IFS= read -r candidate; do
        ls "$candidate"/vivado/system.runs/impl_1/*.bit >/dev/null 2>&1 && run=$candidate
    done < <(printf '%s\n' "$repo"/build/$kernel/*"$suffix" \
                            "$repo"/build/$kernel/*"$suffix"-[0-9]* | sort)
fi
[ -n "$run" ] || { echo "no $kernel bitstream run for $board in build/$kernel/" >&2; exit 1; }
[ -x "$driver/build-$board/jopt_driver" ] || { echo "no build-$board/jopt_driver: run scripts/cross-compile.sh $board" >&2; exit 1; }
echo "bitstream run: $run"

out="$driver/build-$board/jopt_driver_$board"
rm -rf "$out"
mkdir -p "$out"
cp "$driver/build-$board/jopt_driver" "$out/"
cp "$(ls "$run"/hls/solution/impl/ip/drivers/*/src/*_hw.h | head -n 1)" "$out/"
bit=$(ls "$run"/vivado/system.runs/impl_1/*.bit | head -n 1)
hwh=$(find "$run/vivado" -path '*hw_handoff/system.hwh' | head -n 1)

# A tool from PATH, else from the AMD install (bootgen and dtc come with Vitis).
tool() {
    command -v "$1" 2>/dev/null || find /opt /tools -path '*/bin/*' -name "$1" -type f 2>/dev/null | sort | tail -n 1
}

case "$board" in
    zcu104)
        # PYNQ's Overlay wants the .hwh next to the .bit, by the same name.
        cp "$bit" "$out/$kernel.bit"
        cp "$hwh" "$out/$kernel.hwh"
        cp "$driver/jopt_driver.ipynb" "$out/"
        ;;
    zynq7030)
        cp "$bit" "$out/$kernel.bit"
        cp "$driver/firmware/zynq7030.dtsi" "$driver/scripts/process-capture.sh" "$out/"
        ;;
    zynq7020)
        [ -n "$hwh" ] || { echo "no system.hwh in $run/vivado" >&2; exit 1; }
        bootgen=$(tool bootgen)
        dtc=$(tool dtc)
        [ -n "$bootgen" ] && [ -n "$dtc" ] || { echo "need bootgen and dtc (they come with Vitis)" >&2; exit 1; }
        read -r regs clock_hz < <(python3 "$driver/scripts/hwh_values.py" "$hwh")
        echo "kernel registers at 0x$regs, FCLK_CLK0 $clock_hz Hz (from $hwh)"
        # The .bit.bin: the bitstream without its header, what the FPGA manager loads.
        work=$(mktemp -d)
        cp "$bit" "$work/$kernel.bit"
        printf 'all:\n{\n    %s.bit\n}\n' "$kernel" > "$work/$kernel.bif"
        (cd "$work" && "$bootgen" -image "$kernel.bif" -arch zynq -process_bitstream bin -w on >/dev/null)
        cp "$work/$kernel.bit.bin" "$out/"
        rm -rf "$work"
        # The .dtbo: firmware/zynq7020-overlay.dts with this build's values.
        sed -e "s/@FIRMWARE@/$kernel.bit.bin/" -e "s/@REGS@/$regs/g" -e "s/@CLOCK_HZ@/$clock_hz/" \
            "$driver/firmware/zynq7020-overlay.dts" > "$out/$kernel.dts"
        "$dtc" -@ -q -I dts -O dtb -o "$out/$kernel.dtbo" "$out/$kernel.dts"
        cp "$driver/scripts/zedboard-run.sh" "$out/"
        ;;
esac
tar -C "$driver/build-$board" -czf "$out.tar.gz" "jopt_driver_$board"
echo "for the board: $out.tar.gz"
