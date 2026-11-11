#!/bin/bash
# opu.sh KERNEL BOARD [--driver]: a board with the OPU system's PetaLinux (picobob,
# zedboard), the way opu-system and cnn-1d-picobob build theirs. From
# build/KERNEL/BOARD/export.zip (scripts/hls.sh, or someone else's):
#
#   1. Vivado, through opu-system's vivado/run_tcl_script.sh: the board's OPU design
#      (opu_base.tcl: UART, timer, bitstream hash) with the kernel added
#      (boards/opu/project.tcl, add_hls_kernel.tcl) -> system.xsa with the bitstream
#   2. PetaLinux, through opu-system's docker/run-petalinux-docker.sh and
#      prepare_pl_firmware.sh: KERNEL.bit.bin, its md5sum and the devicetree overlay
#      KERNEL.dtbo (boards/opu/hls_kernel.dtsi: the kernel and its memory for UIO)
#   3. classify (software/opu/), built for the board's ARM with the kernel's own HLS
#      driver, in the Docker image fpga-opu-dev (or with a local ARM compiler)
#   4. build/KERNEL/BOARD/KERNEL_opu.tar.gz, to unpack in /media/sd-pl on the board:
#        firmware/KERNEL.bit.bin, .bit.bin.md5sum, .dtbo
#        KERNEL/classify, process-capture.sh, load.sh
#
# --driver: steps 3 and 4 only (after changing classify).
# opu-system: FPGA_OPU_SYSTEM, else ../opu-system next to this repository, with its Vivado
# 2023.1 (in /opt/Xilinx or its Docker image vivado:2023.1) and PetaLinux Docker image
# petalinux:2023.1 (opu-system's README). Steps 1 and 2 are cnn-1d-picobob's
# build-firmware.sh: the same opu-system scripts, the same places. A machine without
# opu-system's Vivado runs the vivado on PATH (fpga_env) the way run_tcl_script.sh would.
# Nothing in opu-system's git is changed.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage="usage: scripts/opu.sh KERNEL BOARD [--driver]"
[ $# -ge 2 ] && [ $# -le 3 ] || die "$usage"
find_kernel "$1"
load_board "$2"
[ "$FLOW" = opu ] || die "$BOARD runs PYNQ, not the OPU system: scripts/pynq.sh $KERNEL $BOARD"
driver_only=0
[ $# -eq 3 ] && { [ "$3" = --driver ] || die "$usage"; driver_only=1; }

out="$(build_dir)"
export_zip="$out/export.zip"
[ -f "$export_zip" ] || die "no $export_zip: run scripts/hls.sh $KERNEL $BOARD first, or put someone else's export.zip there"
OPU="${FPGA_OPU_SYSTEM:-$FPGA_ROOT/../opu-system}"
PROJECT="${KERNEL}_${BOARD}"           # the project in opu-system
clock_ns="$(build_info clock_ns)"
clock_ns="${clock_ns:-$CLOCK_NS}"
clock_mhz="$(awk -v ns="$clock_ns" 'BEGIN { printf "%.3f", 1000 / ns }')"

if [ $driver_only -eq 0 ]; then
    [ -x "$OPU/vivado/run_tcl_script.sh" ] || die "no opu-system in $OPU: clone it there, or set FPGA_OPU_SYSTEM"
    OPU="$(cd "$OPU" && pwd)"
    command -v unzip >/dev/null || die "unzip not found on PATH"
    # run_tcl_script.sh needs Vivado 2023.1 in /opt/Xilinx or its Docker image. Vivado 2023.1
    # takes an export from a newer Vitis HLS (2025.2 on the FPGA workstation).
    if [ -x /opt/Xilinx/Vivado/2023.1/bin/vivado ] || docker image inspect vivado:2023.1 >/dev/null 2>&1; then
        vivado_with=opu-system
    elif command -v vivado >/dev/null; then
        vivado_with=path
    else
        die "no Vivado: opu-system's (its README: docker/vivado-docker), or fpga_env for one on PATH"
    fi

    # 1. Vivado.
    echo "== 1. Vivado: $PROJECT in $OPU/vivado/projects/"
    rm -rf "$OPU/vivado/projects/${PROJECT}_hls_export"
    mkdir -p "$OPU/vivado/projects/${PROJECT}_hls_export" "$OPU/vivado/tcl-scripts/tmp"
    unzip -q -o "$export_zip" -d "$OPU/vivado/projects/${PROJECT}_hls_export"
    sed -e "s/@PROJECT@/$PROJECT/" -e "s/@TOP@/$KERNEL/" -e "s/@CLOCK_NS@/$clock_ns/" \
        -e "s/@BOARD_SCRIPT@/$OPU_BOARD_SCRIPT/" \
        "$FPGA_ROOT/boards/opu/project.tcl" > "$OPU/vivado/tcl-scripts/tmp/$PROJECT.tcl"
    cp "$FPGA_ROOT/boards/opu/add_hls_kernel.tcl" "$OPU/vivado/tcl-scripts/tmp/"
    sdk="$OPU/vivado/projects/$PROJECT/$PROJECT.sdk"
    rm -rf "$sdk"
    mkdir -p "$out"
    if [ "$vivado_with" = opu-system ]; then
        "$OPU/vivado/run_tcl_script.sh" "$PROJECT.tcl" 2>&1 | tee "$out/vivado-console.log"
    else
        # run_tcl_script.sh's steps with this Vivado: the same two seds point repoDir at
        # opu-system, the same flags. check_vivado.tcl prints an error for a release other
        # than 2023.1 and carries on.
        echo "No Vivado of opu-system's: the one on PATH, $(command -v vivado)"
        script="$OPU/vivado/tcl-scripts/tmp/local-$PROJECT.tcl"
        cp "$OPU/vivado/tcl-scripts/tmp/$PROJECT.tcl" "$script"
        sed -i "s!set repoDir /home/hypso!#set repoDir /home/hypso!g" "$script"
        sed -i "s!##set repoDir /home/hypso/opu-system!set repoDir $OPU!g" "$script"
        (cd "$OPU/vivado" && vivado -notrace -nolog -nojou -quiet -mode batch -source "$script") 2>&1 \
            | tee "$out/vivado-console.log"
        rm -f "$script"
    fi
    [ -f "$sdk/system.xsa" ] && [ -f "$sdk/bitstream.bit" ] \
        || die "Vivado made no $sdk/system.xsa and bitstream.bit; see $out/vivado-console.log"
    rm -rf "$out/sdk"
    cp -r "$sdk" "$out/sdk"
    if [ ! -f "$out/sdk/timing.rpt" ]; then
        echo "WARNING: no timing report: check the timing in $OPU/vivado/projects/$PROJECT/$PROJECT.xpr"
    elif grep -q "Timing constraints are not met" "$out/sdk/timing.rpt"; then
        die "the routed design fails timing; see $out/sdk/timing.rpt"
    fi

    # 2. PetaLinux: the bitstream as fpgautil loads it, and the overlay.
    echo "== 2. PetaLinux: $KERNEL.bit.bin and $KERNEL.dtbo"
    docker image inspect petalinux:2023.1 >/dev/null 2>&1 \
        || die "no Docker image petalinux:2023.1: scripts/get-petalinux.sh fpgaws copies the FPGA workstation's"
    pl="$OPU/petalinux/projects/$PROJECT"
    rm -rf "$pl"
    mkdir -p "$pl/output"
    cp "$sdk/system.xsa" "$pl/$PROJECT.xsa"
    memory_base=$((MEMORY_BASE))
    sed -e "s/@CUBE_MAP@/$(printf '0x%08x' $memory_base)/" \
        -e "s/@SCORES_MAP@/$(printf '0x%08x' $((memory_base + 0x09600000)))/" \
        "$FPGA_ROOT/boards/opu/hls_kernel.dtsi" > "$pl/$PROJECT.dtsi"
    # PetaLinux 2023.1 fetches cracklib from its branch master, which upstream has renamed
    # main: without a copy in opu-system's download cache the build stops ("Unable to find
    # revision ... in branch master"). The fix is four lines in opu-system's dtbo-project,
    # which prepare_pl_firmware.sh makes on its first run (it is not in git); after that
    # first run fails, add them and run again.
    dtbo_conf="$OPU/petalinux/projects/dtbo-project/project-spec/meta-user/conf/petalinuxbsp.conf"
    fix_cracklib() {
        [ -f "$dtbo_conf" ] || return 1
        grep -q 'cracklib;protocol=https;branch=main' "$dtbo_conf" && return 1
        echo "Fetching cracklib from its branch main ($dtbo_conf)"
        local recipe url="git://github.com/cracklib/cracklib;protocol=https"
        for recipe in cracklib cracklib-native; do
            printf 'SRC_URI:remove:pn-%s = "%s;branch=master"\nSRC_URI:prepend:pn-%s = "%s;branch=main "\n' \
                "$recipe" "$url" "$recipe" "$url" >> "$dtbo_conf"
        done
    }
    firmware() {
        "$OPU/docker/run-petalinux-docker.sh" ./prepare_pl_firmware.sh "./$PROJECT/$PROJECT.xsa" "$KERNEL" \
            "./$PROJECT/output" "./$PROJECT/$PROJECT.dtsi"
    }
    fix_cracklib || true
    if ! firmware; then
        fix_cracklib || die "PetaLinux failed; see above"
        firmware
    fi
    for file in "$KERNEL.bit.bin" "$KERNEL.bit.bin.md5sum" "$KERNEL.dtbo"; do
        [ -f "$pl/output/$file" ] || die "PetaLinux made no $pl/output/$file"
    done
    rm -rf "$out/firmware"
    mkdir -p "$out/firmware"
    cp "$pl/output/$KERNEL.bit.bin" "$pl/output/$KERNEL.bit.bin.md5sum" "$pl/output/$KERNEL.dtbo" \
        "$pl/$PROJECT.dtsi" "$out/firmware/"
fi

# 3. classify, with the kernel's HLS driver.
echo "== 3. classify for $KERNEL"
for file in "$KERNEL.bit.bin" "$KERNEL.bit.bin.md5sum" "$KERNEL.dtbo"; do
    [ -f "$out/firmware/$file" ] || die "no $out/firmware/$file: run scripts/opu.sh $KERNEL $BOARD without --driver"
done
driver="$out/driver"
rm -rf "$driver" "$out/driver-build"
mkdir -p "$driver"
unzip -q -o -j "$export_zip" '*drivers/*/src/*' -d "$driver" \
    || die "no HLS driver (drivers/*/src/) in $export_zip"
"$FPGA_ROOT/software/opu/make_kernel_header.sh" "$driver" "$clock_mhz" > "$driver/kernel.h"
relative="build/$KERNEL/$BOARD"
cmake_steps="cmake -S software/opu -B $relative/driver-build -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_TOOLCHAIN_FILE=\$PWD/software/opu/cmake/toolchain-arm.cmake -DKERNEL_DRIVER_DIR=\$PWD/$relative/driver \
&& cmake --build $relative/driver-build -j $(job_count)"
if docker image inspect fpga-opu-dev >/dev/null 2>&1; then
    # As cnn-1d-picobob's cross-compile.sh: the repository is /home/hypso in the container.
    # :z lets the container read the mount on SELinux hosts (Fedora); elsewhere it does nothing.
    docker run --rm -v "$FPGA_ROOT:/home/hypso:z" --user "$(id -u):$(id -g)" fpga-opu-dev \
        /bin/bash -c "cd /home/hypso && $cmake_steps"
elif command -v arm-linux-gnueabihf-g++ >/dev/null && command -v cmake >/dev/null; then
    (cd "$FPGA_ROOT" && eval "$cmake_steps")
else
    die "nothing to build classify with: make the Docker image once with
  docker build --network=host -t fpga-opu-dev $FPGA_ROOT/software/opu/docker
or install cmake and g++-arm-linux-gnueabihf, then: scripts/opu.sh $KERNEL $BOARD --driver"
fi
[ -f "$out/driver-build/classify" ] || die "classify was not built; see the output above"

# 4. The package.
package="$out/opu"
rm -rf "$package" "$out/${KERNEL}_opu.tar.gz"
mkdir -p "$package/firmware" "$package/$KERNEL"
cp "$out/firmware/$KERNEL.bit.bin" "$out/firmware/$KERNEL.bit.bin.md5sum" "$out/firmware/$KERNEL.dtbo" "$package/firmware/"
cp "$out/driver-build/classify" "$FPGA_ROOT/software/opu/load.sh" "$FPGA_ROOT/software/opu/process-capture.sh" \
    "$package/$KERNEL/"
tar -C "$package" -czf "$out/${KERNEL}_opu.tar.gz" firmware "$KERNEL"

cat <<EOF

For the board: $out/${KERNEL}_opu.tar.gz
  Unpack it in /media/sd-pl on the board (firmware/ and $KERNEL/), then as root:
    HYPSO:          obip fpga reload $KERNEL.bit.bin -o $KERNEL.dtbo     (hypso-cli)
                    /media/sd-pl/$KERNEL/process-capture.sh hsi0
    the lab, ssh:   sh /media/sd-pl/$KERNEL/load.sh
                    /media/sd-pl/$KERNEL/classify --input cube.npy --labels labels.npy --output labels.bin
  docs/opu.md has every step.
EOF
