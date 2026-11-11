# A kernel on the OPU system: PicoBoB and the ZedBoard

PicoBoB (HYPSO-2's OPU, Zynq-7030) and the lab's ZedBoards (Zynq-7020) run the OPU
system's PetaLinux (opu-system), not PYNQ. A design is built the way opu-system and
cnn-1d-picobob build theirs, with opu-system's own scripts, so that what runs on the
board is HYPSO's standard way: the OPU base design with the kernel added, a `.bit.bin`
with its md5sum, a devicetree overlay, and a program that drives the kernel through UIO.

## What happens

Which script is opu-system's and which is ours, step by step: [opu_flow.md](opu_flow.md).

`fpga_build K picobob` (`scripts/build.sh K picobob`) runs:

| # | Script | What | Result in `build/K/picobob/` |
| --- | --- | --- | --- |
| 1 | `scripts/hls.sh K picobob` | Vitis HLS: csim, csynth, export | `export.zip` (the IP and its driver) |
| 2 | `scripts/opu.sh K picobob`, through opu-system's `vivado/run_tcl_script.sh` (or its steps with the Vivado on PATH, below) | Vivado: `boards/opu/project.tcl`, the same as opu-system's own project scripts: `picozed_project.tcl` (`zedboard_project.tcl`), `opu_base.tcl`, then `boards/opu/add_hls_kernel.tcl`, then `finish_design.tcl` | `sdk/`: `system.xsa`, `bitstream.bit`, the bitstream hash, `utilization.rpt`, `timing.rpt` |
| 3 | ... through opu-system's `docker/run-petalinux-docker.sh` | PetaLinux: `prepare_pl_firmware.sh` with the xsa and `boards/opu/hls_kernel.dtsi` | `firmware/`: `K.bit.bin`, `K.bit.bin.md5sum`, `K.dtbo` |
| 4 | ... through cnn-1d's `software/opu/scripts/cross-compile.sh` (Docker `cnn-dev-image`) | `cnn_accelerator` (`software/opu/`, cnn-1d-picobob's program) for the board's ARM, with the kernel's own HLS driver from `export.zip` | `driver/` (what it was built with) |
| 5 | ... | the tarball | `K_opu.tar.gz` |

Step 1 can be someone else's: put their `export.zip` in `build/K/BOARD/` and run
`fpga_bitstream K BOARD` (`scripts/opu.sh K BOARD`).

The tarball unpacks in `/media/sd-pl` on the board:

```text
firmware/K.bit.bin, K.bit.bin.md5sum, K.dtbo     what obip fpga reload and fpgautil load
K/cnn_accelerator                                runs the kernel on a capture
K/process-capture.sh                             cnn_accelerator on /media/sd-pl/CAPTURE, as cnn-1d's
K/load.sh                                        loads the firmware over ssh (the lab's boards)
```

### The design around the kernel

`add_hls_kernel.tcl` adds the kernel (block `hls_kernel`) to the OPU base design, for any
kernel, as cnn-1d's `add-cnn-accelerator.tcl` adds its own. The base design (UART, timer,
bitstream-hash ROM) stays, so opu-services and `obip fpga hash` work with the kernel loaded.

- control: one more master on the base design's `axi_interconnect_0`, to `s_axi_control`;
- data: each `m_axi` port through its own SmartConnect to its own 64-bit HP port, in name order
  (`m_axi_gmem0` -> HP0, ...), a fifth sharing HP3; at most five;
- clock: a bitstream loaded while Linux runs cannot change the PS's clocks, so `FCLK_CLK0` stays
  at the boot configuration's 100 MHz. A kernel built for 10 ns runs on it. One built for a
  slower clock (the ZedBoard's 12.8 ns) gets an MMCM in the PL (`hls_clock`, with its own reset
  `hls_reset`); the interconnect crosses the clocks. Vivado checks the timing at the clock the
  kernel really gets, and `opu.sh` stops if it fails.

The overlay (`hls_kernel.dtsi`) makes the kernel a UIO device `hls_kernel`, the bitstream hash
`bitstream_hash`, and two maps in the reserved memory CubeDMA uses, `hls_kernel_memory`:
`cube_map`, 0x9600000 bytes (a whole cube, even 684 x 956 x 120), and `scores_map`, 0x1000000
bytes (4 int32 a pixel), from 0x20000000 on PicoBoB and 0x10000000 on the ZedBoard. CubeDMA
must be idle while cnn_accelerator runs. The overlay also says which value the kernel gives each class
(`class-cloud = <0>` ...), as cnn-1d's does.

### cnn_accelerator

`software/opu/` is cnn-1d-picobob's program, with its file names and its scripts:

| File | Compared with cnn-1d-picobob's |
| --- | --- |
| `src/uio_memory_device.{h,cpp}`, `cmake/*.cmake`, `scripts/Dockerfile`, `setup_docker`, `start_docker`, `cross-compile.sh` | the same, byte for byte |
| `CMakeLists.txt` | the driver's files are found by name (`xcnn_driver/x*.c`), as each kernel's are named after it |
| `src/cnn_config.{h,cpp}` | without `--cnn-config` and the rad matrix (`CnnConfig`, `readCnnConfig`, `configIsValid`): these kernels take only the cube |
| `src/xcnn_device.{h,cpp}` | the kernel's driver through `kernel.h` instead of `XCnn_1d_top_*`; the cube on every input port; `cycles()` |
| `src/main.cpp` | no rad matrix; the kernel writes 4 int32 a pixel (3 scores, then the class), and the class is what goes to the labels file; logs the kernel's own cycle count |
| `scripts/process-capture.sh` | the kernel's name from its folder, no `--cnn-config` |
| `scripts/make_kernel_header.sh`, `load.sh`, `benchmark.sh` | ours |

`make_kernel_header.sh` writes `xcnn_driver/kernel.h`, which maps the kernel's own HLS driver
(`xK.h`, `xK_linux.c`) to fixed names, so the same program serves every optimized kernel
(inputs `din0`, `din1`, ..., `dout`, `height`, `width`, and `cycles` if the kernel counts them).
`opu.sh` puts the kernel's driver in `software/opu/xcnn_driver/` (where cnn-1d keeps its own,
git ignores it) and runs cnn-1d's `cross-compile.sh`, so one kernel at a time.

```text
cnn_accelerator --input binned_cube.bip --capture-config capture_config.ini --output labels.bin
```

As cnn-1d's: the cube H x W x 120 uint16 (H = `frame_count`, W = `row_count` in the capture
config), the labels H x W uint8 in HYPSO's values (sea 0, land 1, cloud 2), `--log-level` and
`--log-file`. The whole cube must fit `cube_map`, as cnn-1d's must fit its map.

## Once: before the first build

On the lab server:

1. **opu-system** next to this repository (`../opu-system`), or `export FPGA_OPU_SYSTEM=/path`,
   with its PetaLinux Docker image `petalinux:2023.1`. The FPGA workstation has it; copy it
   over ssh (about 17 GB, no AMD account needed):

   ```bash
   scripts/get-petalinux.sh fpgaws
   ```

   It also lets Docker read opu-system's petalinux folder on Fedora (SELinux). The other way
   is opu-system's `docker/README.md`: the installer in `docker/petalinux-docker/`, then
   `docker/setup-petalinux-docker.sh` (press Enter at its question about a BSP, the firmware
   step needs none).
2. **opu-system's Vivado 2023.1**, in `/opt/Xilinx/Vivado/2023.1` or its Docker image
   `vivado:2023.1` (the FPGA workstation has the image). `opu.sh` runs `run_tcl_script.sh`
   with it, as cnn-1d's `build-firmware.sh` does; it takes an export from Vitis HLS 2025.2.
   A machine without it uses the `vivado` on PATH (`fpga_env`) the way `run_tcl_script.sh`
   would (the same repoDir substitution and flags); opu-system's `check_vivado.tcl` then
   prints `BD_TCL-109` for a release other than 2023.1, and the build carries on.
3. **cnn-1d's Docker image `cnn-dev-image`**, which `cnn_accelerator` is compiled in. If
   cnn-1d-picobob is set up on the machine it is already there; otherwise:

   ```bash
   software/opu/scripts/setup_docker
   ```

   On Fedora (SELinux), let Docker read the folder once:
   `chcon -R -t container_file_t software/opu`. cnn-1d's `start_docker` uses Docker's default
   network to fetch spdlog, libconfig and argparse the first time; where that network has no
   DNS, the build stops at "Could not resolve host: github.com".

opu-system's git is not changed: the build puts its files where cnn-1d's `build-firmware.sh`
does (`vivado/projects/`, `vivado/tcl-scripts/tmp/`, `petalinux/projects/K_BOARD/`).

## Every new design

```bash
source fpga.sh
fpga_env /tools/Xilinx/2025.2/Vivado/settings64.sh
K=sp_unet_small_opt_u8_zynq7030
fpga_test $K
fpga_build $K picobob
```

A ZedBoard kernel (`..._zynq7020`) the same with `zedboard`. The kernel must be sized for
the chip: `_zynq7030` kernels for PicoBoB, `_zynq7020` for the ZedBoard.

### On HYPSO (PicoBoB), with hypso-cli

As cnn-1d-picobob's README:

```sh
(hypso-cli) opu upload ./build/sp_unet_small_opt_u8_zynq7030/picobob/sp_unet_small_opt_u8_zynq7030_opu.tar.gz /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz
(hypso-cli) shell remote oneshot 12 5 tar -xzf /media/sd-pl/sp_unet_small_opt_u8_zynq7030_opu.tar.gz -C /media/sd-pl/
(hypso-cli) obip fpga reload sp_unet_small_opt_u8_zynq7030.bit.bin -o sp_unet_small_opt_u8_zynq7030.dtbo
(hypso-cli) shell remote oneshot 12 1 /media/sd-pl/sp_unet_small_opt_u8_zynq7030/process-capture.sh hsi0 &
```

`process-capture.sh` writes `/media/sd-pl/hsi0/sp_unet_small_opt_u8_zynq7030-labels.bin`
and its log to `/media/sd-pl/hsi0/logs/`. Download them as cnn-1d's README shows.

### On the lab's ZedBoard, over ssh

The board is shared: check who is on it first (`ssh zb05 'uptime; who'`).

```bash
K=sp_unet_small_opt_zynq7020
scp build/$K/zedboard/${K}_opu.tar.gz zb05:/media/sd-pl/
ssh zb05 "cd /media/sd-pl && tar -xzf ${K}_opu.tar.gz && sh $K/load.sh"
ssh zb05 "cd /media/sd-pl && $K/process-capture.sh aegean_unet"
scp zb05:/media/sd-pl/aegean_unet/$K-labels.bin .
python3 tools/image.py score $K-labels.bin aegean_unet
```

A good `load.sh` prints `K.bit.bin: OK` (the md5sum), `FPGA: operating` and
`K loaded: /sys/class/uio/uioN`. It stays loaded until the next reboot (`ssh zb05 reboot`
gives the board back its own design).

The test image: a folder from `tools/image.py prepare` in `/media/sd-pl` on the board, with
`binned_cube.bip` and `capture_config.ini` (the same as a capture's); `process-capture.sh` runs on
it as on a capture. The accuracy is counted on the PC, against the folder's `labels.npy`, by
`tools/image.py score`. An image made before `prepare` wrote `binned_cube.bip`: run `prepare`
again.

### Every kernel on every image

`software/opu/scripts/benchmark.sh`: copy it and each kernel's tarball to `/media/sd-pl/`, with
each image's folder (`binned_cube.bip`, `capture_config.ini`), then as root `sh benchmark.sh`
(`KERNELS=...` and `IMAGES=...` to choose). It loads each kernel with its `load.sh`, runs it on
each image and writes `benchmark_results/DATE/`: the times (`benchmark.md`, `.csv`), everything
the programs printed (`.log`), and each run's labels (`KERNEL/IMAGE-labels.bin`). Copy that
folder to the PC and score them all against the images' labels:

```bash
python3 tools/image.py score benchmark_results/DATE --images DIR_WITH_THE_IMAGE_FOLDERS
```

## After changing cnn_accelerator

`fpga_package K BOARD` (`scripts/opu.sh K BOARD --driver`) builds cnn_accelerator again and packs
the tarball, from the firmware already built.

## When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `no Vivado` | neither opu-system's Vivado nor one on PATH | `fpga_env` |
| `cracklib ... Unable to find revision ... in branch master`, then `Error in prepare_pl_firmware.sh, line 130` | PetaLinux 2023.1 looks for cracklib's old branch name; `opu.sh` adds the fix to opu-system's `dtbo-project` and runs PetaLinux again by itself | nothing, unless it fails the second time too |
| `WARNING: ... not a supported OS` from PetaLinux | its host check, inside Docker | nothing: a warning only |
| `no opu-system in ...` | opu-system is not next to the repository | clone it there, or `export FPGA_OPU_SYSTEM=/path/to/opu-system` |
| `docker: Error response ... permission denied` on `/home/hypso/petalinux` | SELinux (Fedora) keeps Docker out of the folder | `scripts/get-petalinux.sh` again (it relabels the folder) |
| `the input device is not a TTY` | `run-petalinux-docker.sh` runs `docker run -it` | run the build in a terminal, not through a pipe or `nohup` |
| `no Docker image petalinux:2023.1` | opu-system's PetaLinux is not set up | `scripts/get-petalinux.sh fpgaws` |
| `the routed design fails timing` | the kernel does not meet the clock it was built for | `build/K/BOARD/sdk/timing.rpt`; build the kernel for a slower clock (`--clock-ns`) |
| `no Docker image cnn-dev-image` | cnn-1d's image is not on the machine | `software/opu/scripts/setup_docker`, then `fpga_package K BOARD` |
| `Could not resolve host: github.com` while building cnn_accelerator | Docker's default network has no DNS on this machine | fix Docker's DNS (e.g. `"dns"` in `/etc/docker/daemon.json`), or build where it works |
| `K initialize failed: Device not found` | the design is not loaded (a reboot, or another design) | `load.sh` / `obip fpga reload` again |
| `md5sum: ... FAILED` | the `.bit.bin` on the board is not the one built | copy the tarball again |
| `Buffer size ... exceeds reserved memory size` | the cube or its scores do not fit the maps in `hls_kernel.dtsi` | a smaller capture, or bigger maps |
