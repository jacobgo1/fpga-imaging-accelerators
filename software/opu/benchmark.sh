#!/bin/sh
# benchmark.sh -- every optimized ZedBoard kernel on every image: the OPU system's counterpart
# of software/pynq/benchmark.ipynb (no Python on the board). The board runs and times them;
# the PC checks the classes (tools/opu_results.py).
#
# Put it in a folder on the board with, next to it:
#   - each kernel's KERNEL_opu/ (or KERNEL_opu.zip): its bitstream, load.sh and classify,
#     as fpga_build KERNEL --board zedboard packs them. A kernel without one is skipped.
#   - each image, IMAGE/ (or IMAGE.zip) with cube.bip and size.txt, from tools/image.py
#     (prepare, or bip for an image folder made before).
# Then, as root:  sh benchmark.sh
#
# For each kernel: its files go to /media/sd-pl/firmware, load.sh loads it, and classify
# runs it on every image. Everything goes to benchmark_results/DATE/: times.csv, log.txt (what
# load.sh and classify said) and scores/KERNEL/IMAGE.npy. A run that fails is written down
# and the others go on. Copy that folder to the PC, then:
#   python tools/opu_results.py DATE --images FOLDER_WITH_THE_IMAGE_FOLDERS
# The bitstream replaces the whole PL: use a development image, as for one kernel.

KERNELS="${KERNELS:-justoliunet_opt_zynq7020 justoliunet_bn_opt_zynq7020 justounetsimple_opt_zynq7020
sp_unet_small_opt_zynq7020 sp_unet_small_opt_u8_zynq7020 spectralpointwise1d_opt_zynq7020
waveletcnn_opt_zynq7020}"
IMAGES="${IMAGES:-trondheim hawkebay lakealexandria saldanha maputo}"
FIRMWARE="${FIRMWARE:-/media/sd-pl/firmware}"     # where load.sh loads from

cd "$(dirname "$0")" || exit 1
OUT="benchmark_results/$(date +%Y-%m-%d_%H%M)"
mkdir -p "$OUT/scores" || exit 1
CSV="$OUT/times.csv"
LOG="$OUT/log.txt"

# A folder from its zip, when the folder is not there yet.
unpack() {
    if [ ! -e "$1/$2" ] && [ -f "$1.zip" ]; then
        command -v unzip >/dev/null || { echo "no unzip: unpack $1.zip on the PC" >&2; return 1; }
        unzip -o -q "$1.zip" || return 1
    fi
    [ -e "$1/$2" ]
}

# Every image before the first run, so a missing one does not stop the run halfway.
for image in $IMAGES; do
    for file in cube.bip size.txt; do
        unpack "$image" "$file" || { echo "no $image/$file: on the PC, python tools/image.py bip $image" >&2; exit 1; }
    done
done
found="" skipped=""
for k in $KERNELS; do
    if unpack "${k}_opu" classify; then found="$found $k"; else skipped="$skipped $k"; fi
done
[ -n "$found" ] || { echo "no KERNEL_opu/classify for any of the kernels in $(pwd)" >&2; exit 1; }
echo "kernels:$found"
[ -z "$skipped" ] || echo "skipped (no KERNEL_opu/classify):$skipped"
echo "images:  $IMAGES"

# One row per run; error is empty when it ran.
echo "image,kernel,mhz,height,width,load_s,fpga_ms,wall_s,cycles,error" > "$CSV"
failed() {   # image kernel why: a row for a run that did not happen
    echo "$1,$2,,,,,,,,$(echo "$3" | tr ',\n' '; ' | sed 's/ *$//')" >> "$CSV"
    echo "  $1: FAILED: $3"
}

for k in $found; do
    echo "$k:"
    out=$( { cp "${k}_opu"/* "$FIRMWARE"/ && sh "$FIRMWARE/load.sh"; } 2>&1 )
    status=$?
    printf '== %s: load.sh\n%s\n' "$k" "$out" >> "$LOG"
    mhz=$(echo "$out" | sed -n 's/^FCLK_CLK0: \([0-9]*\) kHz.*/\1/p' | awk '{print $1 / 1000}')
    if [ $status -ne 0 ] || [ -z "$mhz" ]; then
        for image in $IMAGES; do failed "$image" "$k" "load.sh: $(echo "$out" | tail -n 1)"; done
        continue
    fi
    mkdir -p "$OUT/scores/$k"
    for image in $IMAGES; do
        read -r h w < "$image/size.txt"
        out=$("./${k}_opu/classify" "$image/cube.bip" "$OUT/scores/$k/$image.npy" "$h" "$w" 2>&1)
        status=$?
        printf '== %s on %s\n%s\n' "$k" "$image" "$out" >> "$LOG"
        if [ $status -ne 0 ]; then failed "$image" "$k" "$(echo "$out" | tail -n 1)"; continue; fi
        times=$(echo "$out" | awk '
            /^cube into memory:/ { load = $4 }
            /^FPGA:/             { fpga = $2 }
            /^own count:/        { cycles = $3 }
            /^wall clock:/       { wall = $3 }
            END { print load "," fpga "," wall "," cycles }')
        echo "$image,$k,$mhz,$h,$w,$times," >> "$CSV"
        echo "  $image: $(echo "$out" | sed -n 's/^FPGA: *//p')"
    done
done

echo "written: $OUT (times.csv, log.txt, scores/). Copy it to the PC, then:"
echo "  python tools/opu_results.py $(basename "$OUT") --images FOLDER_WITH_THE_IMAGE_FOLDERS"
