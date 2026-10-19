# Testing a design on the ZedBoard

How to get a kernel from this repository onto the lab's ZedBoard (`zb05`) and check
its answers. The board runs the OPU system's PetaLinux (opu-system), not PYNQ, so a
design is loaded at run time with `fpgautil` instead of a notebook.

Everything below runs on the lab server, from
`~/Documents/MartinOgJacob/fpga-imaging-accelerators`, unless it says otherwise.

## What stays on the board, and what does not

| Thing | Where | Lasts |
| --- | --- | --- |
| The Linux image (`BOOT.BIN`, `image.ub`) | SD card, FAT32 partition | Until someone replaces it. You do not upload it again. |
| Your design's files (`.bit.bin`, `.dtbo`, `load.sh`, `classify`, test cube) | `/media/sd-pl/firmware/` (SD card, ext4 partition) | Until you overwrite or delete them |
| Your design **loaded on the FPGA** | the FPGA | **Until the next reboot.** After a reboot, run `load.sh` again |

At boot the board loads its own design (overlay `full`, with UIO devices
`cnn_1d_accelerator`, `cnn_1d_memory`, `bitstream_hash`). Loading yours replaces it
until the next reboot. The board is shared: check who else is using it before you load
(see [Sharing the board](#sharing-the-board)).

## Once: tools on the server

```bash
sudo apt install device-tree-compiler gcc-arm-linux-gnueabihf
```

- `dtc` compiles the devicetree overlay (`.dtbo`). The board has no `dtc` of its own.
- `arm-linux-gnueabihf-gcc` builds `classify` for the board's ARM, which has no compiler.

The board is reached as `zb05` (an entry in `~/.ssh/config` on the server). `scp` uses
the same entry: `scp FILE zb05:/media/sd-pl/firmware/`.

## Testing a new design, step by step

Set the kernel's name once per terminal. Every command below uses it:

```bash
K=justounetsimple_opt_zynq7020
```

### 1. Check locally, then build the bitstream

On the laptop, before the server build:

```bash
python -m unittest discover -s tests
python main.py native --kernel justounetsimple_opt_zynq7020
```

On the server:

```bash
source fpga.sh && fpga_env
fpga_build $K --board zedboard
```

`--board zedboard` is what makes the build pack the OPU files instead of a PYNQ zip.
The board needs the bitstream: `fpga_ip $K --board zedboard` stops at the IP
(`hls/solution/impl/ip/`), which is for adding the kernel to a Vivado design of your own,
for example opu-system's `ZedBoard_no_comp.tcl`; it makes nothing `load.sh` can load.
When it finishes, check the package:

```bash
ls build/$K/latest/${K}_opu/
```

You should see:

| File | Missing means |
| --- | --- |
| `$K.bit.bin`, `$K.bit.bin.md5sum` | the build failed: read `build/$K/latest/vivado-console.log` |
| `$K.dtbo` | `dtc` was not on PATH (see [Once](#once-tools-on-the-server)) |
| `classify` | no ARM compiler, or the kernel has no program in `software/opu/$K/` (see [A new kernel](#a-new-kernel-what-it-needs-for-classify)) |
| `load.sh`, `$K.dtsi`, `x..._hw.h`, `kernel_regs.h` | always made |

If the bitstream is already built and you only changed the program or the packing,
re-pack the newest run without rebuilding: `python3 main.py pynq --kernel $K`.

### 2. Copy the design to the board

```bash
ssh zb05 mkdir -p /media/sd-pl/firmware
scp build/$K/latest/${K}_opu/* zb05:/media/sd-pl/firmware/
```

### 3. Load it

```bash
ssh zb05 sh /media/sd-pl/firmware/load.sh
```

A good load prints:

```text
/media/sd-pl/firmware/K.bit.bin: OK                     the file arrived intact
removing the overlay /configfs/device-tree/overlays/full the board's own design goes
BIN FILE loaded through FPGA manager successfully
FPGA: operating
FCLK_CLK0 is 99999 kHz: setting it to the IO PLL / 21   see below
FCLK_CLK0: 76189 kHz, built for 77777 kHz               at or below the build's clock
kernel AP_CTRL at 0x40000000: 0x00000004 (...)          the kernel answers: idle, ready
```

The board's own design runs the fabric clock at 100 MHz, and this image does not apply
the clock the overlay asks for. `load.sh` then sets it itself, to the fastest rate not
faster than the kernel was built for. The clock it prints must never be above the
"built for" value.

Check that Linux made the kernel's UIO device, which `classify` opens:

```bash
ssh zb05 'cat /sys/class/uio/uio*/name'
```

The list must contain `kernel`.

### 4. A test image

`classify` takes a raw cube (H x W x 120, **uint16**) and the PC's answer for it
(`reference_fixed.npy`). Make them with the image tool, from a HYPSO capture (needs the
`hypso` Python package):

```bash
python3 tools/justounetsimple_image.py prepare CAPTURE-l1a.nc --labels CAPTURE-l1a_labels.npy --out aegean_unet
```

`prepare` writes `cube.npy` as uint16 when the values are raw counts, and then also
`reference_fixed.npy`. It prints `... and the integer model, for justounetsimple_opt ...`
when it does.

`aegean_unet/` on the server is older than that: its `cube.npy` is float32, which
`classify` rejects (`expected '<u2'`). Its values are whole counts (171 to 36855), and its
`reference_fixed.npy` was computed from them as uint16, so a uint16 copy matches it:

```bash
python3 -c "
import numpy as np
c = np.load('aegean_unet/cube.npy')
assert (c >= 0).all() and (c <= 65535).all() and (c == np.rint(c)).all(), 'not raw counts'
np.save('aegean_unet/cube_u16.npy', c.astype(np.uint16))
print('written aegean_unet/cube_u16.npy', c.shape)
"
```

Copy them to the board (once; they stay there):

```bash
scp aegean_unet/cube_u16.npy zb05:/media/sd-pl/firmware/cube.npy
scp aegean_unet/reference_fixed.npy zb05:/media/sd-pl/firmware/
```

### 5. Run it

```bash
ssh zb05 "cd /media/sd-pl/firmware && ./classify cube.npy scores.npy reference_fixed.npy"
```

`classify` runs the kernel on the whole cube (in strips of 32 rows if it does not fit
in the reserved memory, with the same answers), prints the FPGA time and the wall-clock
time, and compares every pixel with `reference_fixed.npy`: **PASS** or **FAIL**.
`scores.npy` is what the kernel wrote (H x W x 4 int32: the 3 scores x 2^16, then the
class). Copy it back to look at it:

```bash
scp zb05:/media/sd-pl/firmware/scores.npy build/$K/latest/
```

### Loading and running in one command

After a reboot, or when the files are already on the board:

```bash
ssh zb05 'sh /media/sd-pl/firmware/load.sh && cat /sys/class/uio/uio*/name && cd /media/sd-pl/firmware && ./classify cube.npy scores.npy reference_fixed.npy'
```

## Sharing the board

Others use `zb05` too, with their own design loaded at boot. Before loading:

```bash
ssh zb05 'uptime; who; ls /configfs/device-tree/overlays/; cat /sys/class/uio/uio*/name'
```

- `who` lists who is logged in. Agree with them before replacing their design.
- If `uptime` is shorter than the time since your `load.sh`, the board rebooted and your
  design is gone: load it again.
- When you are done, `ssh zb05 reboot` gives the board back its own design.

## A new kernel: what it needs for `classify`

`load.sh` works for any kernel built with `--board zedboard`. `classify` is
`justounetsimple_opt`'s program (`software/opu/justounetsimple_opt/classify.c`), and
works for a kernel with the same interface:

- AXI-Lite registers `din0`, `din1`, ... (the raw cube), `dout` (the scores), `height`
  and `width`, as `justounetsimple_opt_zynq7020` has. `main.py` (`opu_registers`) stops
  with "The register map has no ..." otherwise.
- at most 4 `m_axi` ports (the Zynq-7000 has 4 HP ports to DDR);
- the same output: H x W x 4 int32.

For a new kernel with that interface, point it at the same program: a folder
`software/opu/NEWKERNEL/` with an `include.txt` holding the line
`software/opu/justounetsimple_opt/classify.c` (see
`software/opu/justounetsimple_opt_zynq7020/include.txt`). A kernel with another
interface needs its own `.c` program in `software/opu/NEWKERNEL/`.

## When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `classify: no UIO device named kernel` | the design is not loaded: usually the board rebooted, or someone loaded another design | `load.sh` again (the [one command](#loading-and-running-in-one-command)) |
| `classify: cube.npy: expected '<u2' in C order` | the cube is float32 | the uint16 copy in [step 4](#4-a-test-image) |
| `md5sum: ... FAILED` from `load.sh` | the `.bit.bin` on the board differs from the one built | copy the files again ([step 2](#2-copy-the-design-to-the-board)) |
| `load.sh` cannot find `K.dtbo` and `dtc` | the server had no `dtc` when it built | install it, re-pack (`python3 main.py pynq --kernel $K`), copy again |
| `FCLK_CLK0` above "built for" after `load.sh` | the clock could not be set | do not run the kernel; paste the output |
| `ssh zb05` asks for a password | the key in `~/.ssh/config` is not being used | `grep -A4 -i "^Host zb05" ~/.ssh/config` |
| the board switches off about 43 minutes after boot | opu-system's start script powers off after 2560 s (for the satellite) | see [the power-off](#the-automatic-power-off) |

## A new Linux image (only if needed)

You do not need this to test a design: the image on the SD card stays. Build a new one
only if the SD card is replaced or broken, or if the image must change (another IP
address, a kernel option, a newer opu-system).

This builds opu-system's development image for the ZedBoard: SSH as `root`/`root`, a
static IP address 129.241.2.N/23, opu-services does not start. Note that `zb05` is now at
192.168.10.34: a new image comes up at 129.241.2.N, so the `zb05` entry in
`~/.ssh/config` must change with it.

### Why not `build_all.sh -B zedboard`

opu-system's `build_all.sh` does not build a ZedBoard image as it is:

- line 192 copies the hardware files to `vivado/projects/hw_config_files/`, which does
  not exist (the right place is `petalinux/projects/hw_config_files/`), and stops;
- with `-d`, `docker/build-bootfiles.sh` looks for `dev` before `zed` and builds a
  **PicoZed** image;
- `petalinux/projects/build_and_package.sh` fixes cracklib only in `pico-primary/` and
  `pico-golden/` and stops when they do not exist;
- it takes the SD card's `bitstream.bit` from `picozed_hw` for every board.

So the steps are run by hand. The software steps are `build_all.sh`'s own.

### Steps

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

### The SD card

| Partition | Type | Contents |
| --- | --- | --- |
| 1 | FAT32, > 100 MB | `BOOT.BIN` and `image.ub` from `petalinux/projects/zed-primary/images/linux/`, `bitstream.bit` from `vivado/projects/ZedBoard/ZedBoard.sdk/` |
| 2 | ext4, the rest | an empty `firmware/` folder: `/media/sd-pl/firmware` on the board |

Boot mode jumpers for SD: JP9 and JP10 to 3V3, JP7, JP8 and JP11 to GND (check against
the ZedBoard manual). The console is the USB-UART at 115200 baud.

### The automatic power-off

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

## The Zynq-7030 (PicoBoB)

The same flow, with `--board zynq7030` and the 7030 kernel:

```bash
K=justounetsimple_opt_zynq7030
fpga_build $K --board zynq7030
```

Then steps 2 to 5 as above, against the PicoBoB's address. Its image is opu-system's
normal PicoZed build: `./build_all.sh -p 34 -d`, files in
`petalinux/projects/bootfiles/pico/`. With `-d`, `build_and_package.sh` still fixes
cracklib in `pico-golden/`, which only exists after a full (non `-d`) build on that
server.
