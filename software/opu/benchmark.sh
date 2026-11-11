#!/bin/sh
# benchmark.sh -- every optimized kernel on every image on a board with the OPU system
# (the lab's ZedBoards, PicoBoB), one report: the counterpart of
# software/pynq/benchmark.ipynb (no Python on the board).
#
# Put it in /media/sd-pl on the board with, next to it:
#   - each kernel's KERNEL_opu.tar.gz (fpga_build KERNEL zedboard), or it unpacked:
#     firmware/ and KERNEL/ (classify, load.sh). A kernel without one is skipped.
#   - each image, IMAGE/ (or IMAGE.zip) with cube.npy and labels.npy, from tools/image.py.
# Then, as root:  sh benchmark.sh
#
# For each kernel: its load.sh loads it, and classify runs it on every image. The report
# goes to benchmark_results/benchmark_DATE.md, every run also in a .csv beside it. A run
# that fails is written down and the others go on.

KERNELS="${KERNELS:-justoliunet_opt_zynq7020 justoliunet_bn_opt_zynq7020 justounetsimple_opt_zynq7020
sp_unet_small_opt_zynq7020 sp_unet_small_opt_u8_zynq7020 spectralpointwise1d_opt_zynq7020
waveletcnn_opt_zynq7020}"
IMAGES="${IMAGES:-trondheim hawkebay lakealexandria saldanha maputo}"
export FIRMWARE="${FIRMWARE:-$(cd "$(dirname "$0")" && pwd)/firmware}"   # where load.sh loads from

cd "$(dirname "$0")" || exit 1
RESULTS=benchmark_results
mkdir -p "$RESULTS" || exit 1
STAMP=$(date +%Y-%m-%d_%H%M)
CSV="$RESULTS/benchmark_$STAMP.csv"
REPORT="$RESULTS/benchmark_$STAMP.md"
LABELS="$RESULTS/labels.bin"                      # each run's, overwritten by the next
LOG="$RESULTS/benchmark_$STAMP.log"               # everything load.sh and classify said

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
    for file in cube.npy labels.npy; do
        unpack "$image" "$file" || { echo "no $image/$file: make the image with tools/image.py prepare --labels" >&2; exit 1; }
    done
done
found="" skipped=""
for k in $KERNELS; do
    # A kernel's tarball, when its folder is not there yet.
    if [ ! -e "$k/classify" ] && [ -f "${k}_opu.tar.gz" ]; then tar -xzf "${k}_opu.tar.gz" || exit 1; fi
    if [ -e "$k/classify" ]; then found="$found $k"; else skipped="$skipped $k"; fi
done
[ -n "$found" ] || { echo "no KERNEL/classify for any of the kernels in $(pwd)" >&2; exit 1; }
echo "kernels:$found"
[ -z "$skipped" ] || echo "skipped (no KERNEL/classify):$skipped"
echo "images:  $IMAGES"

# One row per run; error is empty when it ran.
echo "image,kernel,unit,mhz,height,width,patches,load_s,fpga_ms,wall_s,cycles,labelled,correct,both_cloud,both_land,both_sea,either_cloud,either_land,either_sea,error" > "$CSV"
failed() {   # image kernel why: a row for a run that did not happen
    echo "$1,$2,,,,,,,,,,,,,,,,,,$(echo "$3" | tr ',\n' '; ' | sed 's/ *$//')" >> "$CSV"
    echo "  $2: FAILED: $3"
}

for k in $found; do
    # The 1D networks' cycles are given per pixel, the 2D ones' per 32 x 32 patch.
    case $k in justoliunet*|spectralpointwise1d*) unit=pixel ;; *) unit=patch ;; esac
    echo "$k:"
    out=$(sh "$k/load.sh" 2>&1)
    status=$?
    printf '== %s: load.sh\n%s\n' "$k" "$out" >> "$LOG"
    if [ $status -ne 0 ]; then
        for image in $IMAGES; do failed "$image" "$k" "load.sh: $(echo "$out" | tail -n 1)"; done
        continue
    fi
    for image in $IMAGES; do
        out=$("./$k/classify" --input "$image/cube.npy" --labels "$image/labels.npy" --output "$LABELS" 2>&1)
        status=$?
        printf '== %s on %s\n%s\n' "$k" "$image" "$out" >> "$LOG"
        if [ $status -ne 0 ]; then failed "$image" "$k" "$(echo "$out" | tail -n 1)"; continue; fi
        # classify's lines, as numbers (a kernel without its own count has no "own count").
        echo "$out" | awk -v image="$image" -v k="$k" -v unit="$unit" '
            /^kernel: .*built for/ { mhz = $(NF - 1) }
            /pixels, .* patches/  { h = $1; w = $3; p = $5 }
            /^cube into memory:/  { load = $4 }
            /^FPGA:/              { fpga = $2 }
            /^own count:/         { cycles = $3 }
            /^wall clock:/        { wall = $3 }
            /^counts:/            { counts = $2 "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 }
            END { print image "," k "," unit "," mhz "," h "," w "," p "," load "," fpga "," wall "," cycles "," counts "," }' \
            >> "$CSV"
        tail -n 1 "$CSV" | awk -F, '{
            own = $11 == "" ? "" : sprintf(", %.2f ms by its count", $11 / ($4 * 1000))
            printf "  %s: %.2f ms FPGA%s, %.2f%%\n", $1, $9, own, $12 ? 100 * $13 / $12 : 0 }'
    done
done
rm -f "$LABELS"

# The report: each kernel's mean over the images it ran on, as the notebook's.
awk -F, -v stamp="$STAMP" -v images="$IMAGES" -v skipped="$skipped" '
    NR == 1 { next }
    !($2 in seen) { seen[$2] = 1; order[++kernels] = $2 }
    $5 != "" && !($1 in shape) { shape[$1] = 1; shapes = shapes (shapes ? ", " : "") $1 " (" $5 " x " $6 ")" }
    $20 != "" { runs[++nruns] = $0; next }
    {
        k = $2; n[k]++; mhz[k] = $4; unit[k] = $3
        fpga[k] += $9; wall[k] += $10; load[k] += $8
        units = $3 == "pixel" ? $5 * $6 : $7
        if ($11 != "") { own[k] += $11 / ($4 * 1000); cyc[k] += $11 / units; ns[k] += 1e3 * $11 / $4 / ($5 * $6); counted[k]++ }
        acc[k] += $12 ? $13 / $12 : 0; lab[k] += $12; cor[k] += $13
        for (c = 0; c < 3; c++) { both[k, c] += $(14 + c); either[k, c] += $(17 + c) }
        runs[++nruns] = $0
    }
    END {
        print "# Optimized kernels on the OPU system, " stamp "\n"
        print "Images: " shapes ".\n"
        print "## Mean over the images\n"
        print "| Kernel | MHz | Images | Cycles (own count) | Own count (ms) | FPGA (ms) | Cube into memory (s) | Wall clock (s) " \
              "| ns a pixel | Accuracy (mean) | Accuracy (all pixels) | IoU cloud | IoU land | IoU sea | mIoU |"
        print "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        for (i = 1; i <= kernels; i++) {
            k = order[i]
            if (!n[k]) continue
            # Each class IoU over all the images pixels; mIoU over the classes that occur.
            iou = ""; classes = 0; s = 0
            for (c = 0; c < 3; c++) {
                if (either[k, c]) { v = both[k, c] / either[k, c]; classes++; s += v; iou = iou sprintf(" %.3f |", v) }
                else iou = iou " - |"
            }
            cycles = counted[k] ? sprintf("%.2f a %s", cyc[k] / counted[k], unit[k]) : "-"
            printf "| `%s` | %.1f | %d | %s | %s | %.3f | %.2f | %.2f | %s | %.2f%% | %.2f%% |%s %s |\n", k, mhz[k], n[k], cycles,
                   counted[k] ? sprintf("%.3f", own[k] / counted[k]) : "-", fpga[k] / n[k], load[k] / n[k], wall[k] / n[k],
                   counted[k] ? sprintf("%.2f", ns[k] / counted[k]) : "-", 100 * acc[k] / n[k],
                   lab[k] ? 100 * cor[k] / lab[k] : 0, iou, classes ? sprintf("%.3f", s / classes) : "-"
        }
        print "\n## Every run\n"
        print "| Image | Kernel | Own count (ms) | FPGA (ms) | Wall clock (s) | Cycles | Accuracy |"
        print "| --- | --- | --- | --- | --- | --- | --- |"
        for (i = 1; i <= nruns; i++) {
            split(runs[i], r, ",")
            if (r[20] != "") { printf "| %s | `%s` | failed: %s | | | | |\n", r[1], r[2], r[20]; continue }
            units = r[3] == "pixel" ? r[5] * r[6] : r[7]
            printf "| %s | `%s` | %s | %.3f | %.2f | %s | %.2f%% |\n", r[1], r[2],
                   r[11] == "" ? "-" : sprintf("%.3f", r[11] / (r[4] * 1000)), r[9], r[10],
                   r[11] == "" ? "-" : sprintf("%.2f a %s", r[11] / units, r[3]), r[12] ? 100 * r[13] / r[12] : 0
        }
        if (skipped != "") print "\nSkipped, no `KERNEL/classify`:" skipped "."
    }' "$CSV" > "$REPORT"

cat "$REPORT"
echo "written: $REPORT and $CSV (and what the programs said: $LOG)"
