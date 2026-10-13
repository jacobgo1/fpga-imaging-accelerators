#!/bin/bash
# Everything a board needs to run jopt_driver, as one archive:
#   software/driver/scripts/package.sh zynq7030
#   software/driver/scripts/package.sh zcu104
# takes the newest bitstream run of that board's kernel (justounetsimple_opt
# for the ZCU104, justounetsimple_opt_zynq7030 for HYPSO-2) (or the run
# folder given as a second argument) and writes
#   software/driver/build-BOARD/jopt_driver_BOARD.tar.gz
# with the driver, the register header and bitstream of that run, the device
# tree snippet; for HYPSO-2 process-capture.sh, for the ZCU104 jopt_driver.ipynb. Cross-compile first (cross-compile.sh).
set -e
board=$1
case "$board" in
    zynq7030) kernel=justounetsimple_opt_zynq7030; suffix='-bitstream-zynq7030' ;;
    zcu104)   kernel=justounetsimple_opt;          suffix='-bitstream' ;;
    *) echo "usage: $0 zynq7030|zcu104 [RUN]" >&2; exit 1 ;;
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
cp "$(ls "$run"/vivado/system.runs/impl_1/*.bit | head -n 1)" "$out/$kernel.bit"
# PYNQ's Overlay wants the .hwh next to the .bit, by the same name.
hwh=$(find "$run/vivado" -path '*hw_handoff/system.hwh' | head -n 1)
[ -n "$hwh" ] && cp "$hwh" "$out/$kernel.hwh"
cp "$driver/firmware/$board.dtsi" "$out/"
[ "$board" = zynq7030 ] && cp "$driver/scripts/process-capture.sh" "$out/"
# The ZCU104 runs PYNQ: a notebook that loads the bitstream and runs the driver.
[ "$board" = zcu104 ] && cp "$driver/jopt_driver.ipynb" "$out/"
tar -C "$driver/build-$board" -czf "$out.tar.gz" "jopt_driver_$board"
echo "for the board: $out.tar.gz"
