# A new Linux image for the ZedBoard (rarely needed)

**Not needed to test a design**: the ZedBoard's SD card already has its Linux, and a new
bitstream is loaded while it runs ([zedboard.md](zedboard.md)). Build a new image only if
the SD card is replaced or broken, or if the image itself must change (another IP address,
a kernel option, a newer opu-system).

This builds opu-system's development image for the ZedBoard: SSH as `root`/`root`, a
static IP address 129.241.2.N/23, opu-services does not start. `zb05` is now at
192.168.10.34: a new image comes up at 129.241.2.N, so the `zb05` entry in
`~/.ssh/config` must change with it.

## Why not `build_all.sh -B zedboard`

opu-system's `build_all.sh` does not build a ZedBoard image as it is:

- line 192 copies the hardware files to `vivado/projects/hw_config_files/`, which does
  not exist (the right place is `petalinux/projects/hw_config_files/`), and stops;
- with `-d`, `docker/build-bootfiles.sh` looks for `dev` before `zed` and builds a
  **PicoZed** image;
- `petalinux/projects/build_and_package.sh` fixes cracklib only in `pico-primary/` and
  `pico-golden/` and stops when they do not exist;
- it takes the SD card's `bitstream.bit` from `picozed_hw` for every board.

So the steps are run by hand. The software steps are `build_all.sh`'s own.

## Steps

From `~/Documents/MartinOgJacob/opu-system`, with N = 34:

```bash
# 1) The ZedBoard base design in Vivado (about 7 minutes)
docker/run-vivado-docker.sh -mode batch -source /home/hypso/vivado/tcl-scripts/ZedBoard.tcl
ls -l vivado/projects/ZedBoard/ZedBoard.sdk/     # system.xsa, bitstream.bit, ZedBoard.md5

# 2) The hardware files where PetaLinux reads them
mkdir -p petalinux/projects/hw_config_files/zedboard_hw
cp vivado/projects/ZedBoard/ZedBoard.sdk/system.xsa vivado/projects/ZedBoard/ZedBoard.sdk/bitstream.bit petalinux/projects/hw_config_files/zedboard_hw/

# 3) The software (build_all.sh lines 498-502)
docker/update-hypso-sw.sh h1
docker/update-onboard-pipeline-modules.sh
docker/tar-additional-system-files.sh h1

# 4) The development start script: opu-services does not start, and no power-off
cp petalinux/projects/software/startup_script_template.sh petalinux/projects/software/startup_script.sh
sed -i 's/#&&&#/exit 1/g' petalinux/projects/software/startup_script.sh
sed -i '/sleep 2560/d' petalinux/projects/software/startup_script.sh

# 5) PetaLinux: configure, then build (long the first time)
C=$(git log --pretty=format:%h -n 1)
docker/run-petalinux-docker.sh "./setup_firmware.sh zed 129.241.2.34/23 $C -dev && ./setup_software.sh zed 129.241.2.34/23 $C -dev"
docker/run-petalinux-docker.sh "cd zed-primary && sed -i 's|branch=master|branch=main|' components/yocto/layers/poky/meta/recipes-extended/cracklib/cracklib_2.9.8.bb && petalinux-build && petalinux-package --boot --bif ../bootgen.bif --force"
```

Step 1 needs the ZedBoard pin constraints for the OPU UART (`rs422_sin`, `rs422_sout`)
in `vivado/projects/constraints/zedboard_constraints.xdc`. Without them Vivado stops
with DRC errors NSTD-1 and UCIO-1 and writes no bitstream. They were added to the
server's copy of opu-system on 19 October 2026 (Pmod JA1 and JA2), not to opu-system on
GitHub, so a fresh clone needs them again:

```bash
cat >> vivado/projects/constraints/zedboard_constraints.xdc <<'EOF'

# The AXI UART16550's pins (opu_base.tcl's rs422 ports) on Pmod JA
set_property PACKAGE_PIN Y11  [get_ports {rs422_sin}];  #JA1
set_property PACKAGE_PIN AA11 [get_ports {rs422_sout}]; #JA2
set_property IOSTANDARD LVCMOS33 [get_ports {rs422_sin}];
set_property IOSTANDARD LVCMOS33 [get_ports {rs422_sout}];
EOF
```

If Vivado is installed natively instead of in Docker, `ZedBoard.tcl` has `repoDir`
fixed to `/home/hypso`: run a copy with it changed.

```bash
sed "s#^set repoDir /home/hypso\$#set repoDir $HOME/Documents/MartinOgJacob/opu-system#" vivado/tcl-scripts/ZedBoard.tcl > /tmp/ZedBoard_local.tcl
vivado -mode batch -source /tmp/ZedBoard_local.tcl
```

## The SD card

| Partition | Type | Contents |
| --- | --- | --- |
| 1 | FAT32, > 100 MB | `BOOT.BIN` and `image.ub` from `petalinux/projects/zed-primary/images/linux/`, `bitstream.bit` from `vivado/projects/ZedBoard/ZedBoard.sdk/` |
| 2 | ext4, the rest | an empty `firmware/` folder: `/media/sd-pl/firmware` on the board |

Boot mode jumpers for SD: JP9 and JP10 to 3V3, JP7, JP8 and JP11 to GND (check against
the ZedBoard manual). The console is the USB-UART at 115200 baud.

## The automatic power-off

opu-system's start script (`petalinux/projects/software/startup_script_template.sh`) has

```sh
printf "sleep 2560\npoweroff" | /bin/sh &
```

so a board switches itself off 42 minutes 40 seconds after boot, development images
included: on the satellite the power system cuts the payload after 2600 s, and this shuts
it down cleanly first. Step 4 above removes it from the image. Keep that change on the
server: opu-system's template also builds the flight images.

On a running board, stop it by killing the waiting shell **before** the `sleep`.
Killing only the `sleep` powers the board off at once.

```sh
p=$(pidof sleep); kill $(cut -d' ' -f4 /proc/$p/stat); kill $p
```

## PicoBoB (Zynq-7030)

Its image is opu-system's normal PicoZed build: `./build_all.sh -p 34 -d`, files in
`petalinux/projects/bootfiles/pico/`. With `-d`, `build_and_package.sh` still fixes
cracklib in `pico-golden/`, which only exists after a full (non `-d`) build on that
server.
