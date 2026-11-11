#!/bin/sh
# benchmark.sh: every kernel on every image on a board with the OPU system (the lab's
# ZedBoards, PicoBoB), timed, with each run's labels kept to score on the PC.
#
# Put it in /media/sd-pl on the board with, next to it:
#   - each kernel's KERNEL_opu.tar.gz (fpga_build KERNEL zedboard), or it unpacked:
#     firmware/ and KERNEL/ (cnn_accelerator, load.sh). A kernel without one is skipped.
#   - each image, IMAGE/ (or IMAGE.zip) with binned_cube.bip and capture_config.ini, from
#     tools/image.py prepare.
# Then, as root:  sh benchmark.sh
#
# For each kernel: its load.sh loads it, and cnn_accelerator runs it on every image. Into
# benchmark_results/DATE/: KERNEL/IMAGE-labels.bin, benchmark.csv, benchmark.md (the times)
# and benchmark.log (what the programs said). A run that fails is written down and the
# others go on. The accuracy, on the PC, with the image folders there too:
#   python3 tools/image.py score benchmark_results/DATE --images DIR

KERNELS="${KERNELS:-justoliunet_opt_zynq7020 justoliunet_bn_opt_zynq7020 justounetsimple_opt_zynq7020
sp_unet_small_opt_zynq7020 sp_unet_small_opt_u8_zynq7020 spectralpointwise1d_opt_zynq7020
waveletcnn_opt_zynq7020}"
IMAGES="${IMAGES:-trondheim hawkebay lakealexandria saldanha maputo}"
export FIRMWARE="${FIRMWARE:-$(cd "$(dirname "$0")" && pwd)/firmware}"   # where load.sh loads from

cd "$(dirname "$0")" || exit 1
STAMP=$(date +%Y-%m-%d_%H%M)
RESULTS="benchmark_results/$STAMP"
mkdir -p "$RESULTS" || exit 1
CSV="$RESULTS/benchmark.csv"
REPORT="$RESULTS/benchmark.md"
LOG="$RESULTS/benchmark.log"

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
    for file in binned_cube.bip capture_config.ini; do
        unpack "$image" "$file" || { echo "no $image/$file: make the image with tools/image.py prepare" >&2; exit 1; }
    done
done
found="" skipped=""
for k in $KERNELS; do
    # A kernel's tarball, when its folder is not there yet.
    if [ ! -e "$k/cnn_accelerator" ] && [ -f "${k}_opu.tar.gz" ]; then tar -xzf "${k}_opu.tar.gz" || exit 1; fi
    if [ -e "$k/cnn_accelerator" ]; then found="$found $k"; else skipped="$skipped $k"; fi
done
[ -n "$found" ] || { echo "no KERNEL/cnn_accelerator for any of the kernels in $(pwd)" >&2; exit 1; }
echo "kernels:$found"
[ -z "$skipped" ] || echo "skipped (no KERNEL/cnn_accelerator):$skipped"
echo "images:  $IMAGES"

# One row per run; error is empty when it ran.
echo "image,kernel,mhz,load_ms,fpga_ms,cycles,total_ms,error" > "$CSV"
failed() {   # image kernel why: a row for a run that did not happen
    echo "$1,$2,,,,,,$(echo "$3" | tr ',\n' '; ' | sed 's/ *$//')" >> "$CSV"
    echo "  $2: FAILED: $3"
}

for k in $found; do
    echo "$k:"
    mkdir -p "$RESULTS/$k"
    out=$(sh "$k/load.sh" 2>&1)
    status=$?
    printf '== %s: load.sh\n%s\n' "$k" "$out" >> "$LOG"
    if [ $status -ne 0 ]; then
        for image in $IMAGES; do failed "$image" "$k" "load.sh: $(echo "$out" | tail -n 1)"; done
        continue
    fi
    for image in $IMAGES; do
        out=$("./$k/cnn_accelerator" --input "$image/binned_cube.bip" --capture-config "$image/capture_config.ini" \
              --output "$RESULTS/$k/$image-labels.bin" 2>&1)
        status=$?
        printf '== %s on %s\n%s\n' "$k" "$image" "$out" >> "$LOG"
        if [ $status -ne 0 ]; then failed "$image" "$k" "$(echo "$out" | tail -n 1)"; continue; fi
        # cnn_accelerator's log lines, as numbers (a kernel without its own count has none).
        echo "$out" | awk -v image="$image" -v k="$k" '
            /built for .* MHz/        { sub(/.*built for /, ""); mhz = $1 }
            /Input data loaded in/    { sub(/.*Input data loaded in /, ""); load = $1 }
            /CNN finished in/         { sub(/.*CNN finished in /, ""); fpga = $1 }
            /own count: .* cycles/    { sub(/.*own count: /, ""); cycles = $1 }
            /Total runtime/           { sub(/.*Total runtime /, ""); total = $1 }
            END { print image "," k "," mhz "," load "," fpga "," cycles "," total "," }' >> "$CSV"
        tail -n 1 "$CSV" | awk -F, '{
            own = $6 == "" ? "" : sprintf(", %.3f ms by its own count", $6 / ($3 * 1000))
            printf "  %s: %s ms FPGA%s, %s ms in all\n", $1, $5, own, $7 }'
    done
done

# The report: each kernel's mean over the images it ran on, and every run.
awk -F, -v stamp="$STAMP" -v skipped="$skipped" '
    NR == 1 { next }
    !($2 in seen) { seen[$2] = 1; order[++kernels] = $2 }
    { runs[++nruns] = $0 }
    $8 != "" { next }
    {
        k = $2; n[k]++; mhz[k] = $3; load[k] += $4; fpga[k] += $5; total[k] += $7
        if ($6 != "") { own[k] += $6 / ($3 * 1000); counted[k]++ }
    }
    END {
        print "# Kernels on the OPU system, " stamp "\n"
        print "## Mean over the images\n"
        print "| Kernel | MHz | Images | Own count (ms) | FPGA (ms) | Cube into memory (ms) | In all (ms) |"
        print "| --- | --- | --- | --- | --- | --- | --- |"
        for (i = 1; i <= kernels; i++) {
            k = order[i]
            if (!n[k]) continue
            printf "| `%s` | %s | %d | %s | %.0f | %.0f | %.0f |\n", k, mhz[k], n[k],
                   counted[k] ? sprintf("%.3f", own[k] / counted[k]) : "-", fpga[k] / n[k], load[k] / n[k], total[k] / n[k]
        }
        print "\nFPGA (ms) is measured by polling every 100 ms; the own count is exact.\n"
        print "## Every run\n"
        print "| Image | Kernel | Own count (ms) | FPGA (ms) | In all (ms) |"
        print "| --- | --- | --- | --- | --- |"
        for (i = 1; i <= nruns; i++) {
            split(runs[i], r, ",")
            if (r[8] != "") { printf "| %s | `%s` | failed: %s | | |\n", r[1], r[2], r[8]; continue }
            printf "| %s | `%s` | %s | %s | %s |\n", r[1], r[2],
                   r[6] == "" ? "-" : sprintf("%.3f", r[6] / (r[3] * 1000)), r[5], r[7]
        }
        if (skipped != "") print "\nSkipped, no `KERNEL/cnn_accelerator`:" skipped "."
    }' "$CSV" > "$REPORT"

cat "$REPORT"
echo "written: $RESULTS (the labels, $REPORT, $CSV, and what the programs said: $LOG)"
echo "accuracy, on the PC: python3 tools/image.py score $RESULTS --images DIR"
