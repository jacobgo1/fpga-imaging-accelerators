# How a kernel gets onto an OPU board, script by script

This is the whole path from a kernel's C++ to a result on PicoBoB or the ZedBoard: which
script does what, which of them are opu-system's own and which are this repository's,
and where this repository adds to or works around opu-system. [opu.md](opu.md) is the
how-to; this is the map.

In short: **opu-system's scripts do the hardware build exactly as for any other OPU design.**
This repository adds what is specific to an HLS kernel: the HLS step before them, and the
kernel's part of the block design and the overlay that they take as input. After them comes
cnn-1d-picobob's program, `cnn_accelerator`, with its files and scripts, changed only where
these kernels differ from the CNN-1D. It does the same as cnn-1d-picobob's
`build-firmware.sh`, for any kernel.

## The picture

```text
 THIS REPOSITORY                                OPU-SYSTEM (unchanged, run in place)
 ───────────────                                ────────────────────────────────────
 src/models/.../K.cpp
        │  scripts/hls.sh + hls.tcl  (Vitis HLS)
        ▼
 build/K/BOARD/export.zip  (the IP + its C driver)
        │
        │  scripts/opu.sh, step 1 ───────────── unzips it into ───────► vivado/projects/K_BOARD_hls_export/
        │    fills in boards/opu/project.tcl ── writes ──────────────► vivado/tcl-scripts/tmp/K_BOARD.tcl
        │    copies boards/opu/add_hls_kernel.tcl ─────────────────────► vivado/tcl-scripts/tmp/add_hls_kernel.tcl
        │    runs ─────────────────────────────────────────────────────► vivado/run_tcl_script.sh K_BOARD.tcl
        │                                                                 │ (Vivado 2023.1: /opt or Docker)
        │                                                                 ├ check_vivado.tcl
        │                                                                 ├ picozed_project.tcl / zedboard_project.tcl
        │                                                                 ├ opu_base.tcl
        │                                       (ours, sourced by it) ◄───┤ tmp/add_hls_kernel.tcl
        │                                                                 └ finish_design.tcl (+ update_hash.tcl)
        │                                                                   ▼
        │    copies back ◄──────────────────────────────────────────── vivado/projects/K_BOARD/K_BOARD.sdk/
        ▼                                                                   system.xsa, bitstream.bit, ...
 build/K/BOARD/sdk/
        │
        │  scripts/opu.sh, step 2
        │    copies the xsa + fills in boards/opu/hls_kernel.dtsi ──► petalinux/projects/K_BOARD/
        │    runs ─────────────────────────────────────────────────────► docker/run-petalinux-docker.sh
        │                                                                 └ petalinux/projects/prepare_pl_firmware.sh
        │                                                                   (PetaLinux 2023.1, in Docker; uses and,
        │                                                                    the first time, makes dtbo-project/)
        │    copies back ◄──────────────────────────────────────────── petalinux/projects/K_BOARD/output/
        ▼                                                                   K.bit.bin, K.bit.bin.md5sum, K.dtbo
 build/K/BOARD/firmware/
        │
        │  scripts/opu.sh, step 3: the kernel's driver from export.zip into software/opu/xcnn_driver/,
        │                          then cnn-1d's software/opu/scripts/cross-compile.sh (Docker
        │                          cnn-dev-image) builds cnn_accelerator for the ARM
        │  scripts/opu.sh, step 4: the tarball
        ▼
 build/K/BOARD/K_opu.tar.gz ──► the board's /media/sd-pl:
                                  firmware/K.bit.bin, K.bit.bin.md5sum, K.dtbo
                                  K/cnn_accelerator, K/load.sh, K/process-capture.sh
```

`scripts/build.sh K BOARD` runs `hls.sh` and then `opu.sh`; `fpga_build K BOARD` in
`fpga.sh` is the same thing. `opu.sh` alone (`fpga_bitstream`) starts from an existing
`export.zip`.

## Step by step

### 0. HLS: C++ to IP (this repository only)

| Script | Role |
| --- | --- |
| `scripts/hls.sh K BOARD` | Writes `settings.tcl` (the board's FPGA part and clock from `boards/BOARD/board.conf`, the kernel's sources) and runs Vitis HLS with `scripts/hls.tcl`. |
| `scripts/hls.tcl` | C simulation, C synthesis, export as IP. |

Output: `build/K/BOARD/export.zip`. That zip is what cnn-1d calls `firmware/export.zip`. It
holds the IP (`component.xml`, the Verilog) and the C driver HLS generates for it
(`drivers/K_v1_0/src/xK.h`, `xK_hw.h`, `xK.c`, `xK_linux.c`). opu-system is not involved.

### 1. Vivado: the bitstream (opu-system's scripts, our project script)

`scripts/opu.sh` does what cnn-1d's `build-firmware.sh` does:

1. Unzips `export.zip` into opu-system's `vivado/projects/K_BOARD_hls_export/`.
2. Writes the project script `vivado/tcl-scripts/tmp/K_BOARD.tcl` from our template
   `boards/opu/project.tcl` (cnn-1d writes its own `cnn-1d-picobob.tcl` by hand; ours is
   filled in for whichever kernel). It copies `boards/opu/add_hls_kernel.tcl` next to it.
3. Runs opu-system's `vivado/run_tcl_script.sh K_BOARD.tcl`.

What runs, in order, and whose it is:

| Script | Whose | Role |
| --- | --- | --- |
| `vivado/run_tcl_script.sh` | opu-system | Finds Vivado 2023.1 (in `/opt/Xilinx` or the Docker image `vivado:2023.1`), points `repoDir` in the project script at opu-system, runs Vivado in batch mode. |
| `tmp/K_BOARD.tcl` (from `boards/opu/project.tcl`) | **ours** | The project script, in the same shape as opu-system's `PicoBoB_no_comp.tcl`: project name, the kernel's name and clock, then sources the five scripts below in order. |
| `check_vivado.tcl` | opu-system | Prints an error if Vivado is not 2023.1, and carries on. |
| `picozed_project.tcl` / `zedboard_project.tcl` | opu-system | Creates the Vivado project for the board's part and the block design `System` with the processor (PS) configured as in flight: DDR, MIO, `FCLK_CLK0` 100 MHz, the HP0 port. Which one comes from `OPU_BOARD_SCRIPT` in `boards/BOARD/board.conf`. |
| `opu_base.tcl` | opu-system | The OPU base design: `axi_interconnect_0` on the PS's GP0 port with 3 masters, the AXI timer, the UART, the bitstream-hash ROM (`axi_bram_ctrl_0` + `blk_mem_gen_0`), the reset, the interrupts. |
| `tmp/add_hls_kernel.tcl` | **ours** | The kernel's part of the design (below). cnn-1d's equivalent is its `add-cnn-accelerator.tcl`. |
| `finish_design.tcl` (+ `update_hash.tcl`) | opu-system | Saves the design, writes the bitstream hash into the ROM, makes the HDL wrapper, runs synthesis and implementation to the bitstream, writes `K_BOARD.sdk/system.xsa` (with the bitstream) and `bitstream.bit`. |

What `add_hls_kernel.tcl` adds to opu-system's design, for any HLS kernel:

- the IP from `K_BOARD_hls_export/` as the block `hls_kernel` (the overlay and
  `cnn_accelerator` find it by that name);
- its control registers (`s_axi_control`) on one more master of `axi_interconnect_0`
  (opu_base's 3 become 4);
- each of its memory ports (`m_axi_gmem0`, ...) through its own SmartConnect to its own HP
  port of the PS, so it reads and writes DDR directly: gmem0 to HP0, ..., a fifth sharing HP3;
- its clock: `FCLK_CLK0`, 100 MHz, when the kernel was built for 10 ns (PicoBoB). For a slower
  kernel (the ZedBoard's 12.8 ns) a clock generator in the PL (`hls_clock`, with its own reset
  `hls_reset`), because a bitstream loaded while Linux runs cannot change the PS's clocks;
- addresses for everything, and a hook after routing that writes `utilization.rpt` and
  `timing.rpt` into the `.sdk` folder.

Then `opu.sh` copies the `.sdk` folder to `build/K/BOARD/sdk/` and stops if `timing.rpt` says
the timing is not met.

### 2. PetaLinux: what the board loads (opu-system's scripts, our overlay)

`opu.sh` again does what `build-firmware.sh` does: copies `system.xsa` to opu-system's
`petalinux/projects/K_BOARD/K_BOARD.xsa`, writes the overlay additions there as
`K_BOARD.dtsi`, and runs

```text
docker/run-petalinux-docker.sh ./prepare_pl_firmware.sh ./K_BOARD/K_BOARD.xsa K ./K_BOARD/output ./K_BOARD/K_BOARD.dtsi
```

| Script | Whose | Role |
| --- | --- | --- |
| `docker/run-petalinux-docker.sh` | opu-system | Starts the Docker image `petalinux:2023.1` with opu-system's `petalinux/` folder mounted, runs the command in `petalinux/projects/`. |
| `petalinux/projects/prepare_pl_firmware.sh` | opu-system | The first time: creates `dtbo-project/`, a minimal PetaLinux project only for overlays. Every time: adds an app from the xsa and the dtsi to it, builds it, and copies out `K.bit.bin` (the bitstream as the FPGA manager loads it), `K.bit.bin.md5sum` (with the board's path `/media/sd-pl/firmware/`), and `K.dtbo` (the devicetree overlay). |
| `boards/opu/hls_kernel.dtsi` → `K_BOARD.dtsi` | **ours** | The overlay's additions, as cnn-1d's `cnn-1d-picobob.dtsi`: the kernel as a UIO device `hls_kernel` (with which value the kernel gives each class), the hash ROM as `bitstream_hash`, and two maps in the reserved memory (`hls_kernel_memory`: `cube_map` and `scores_map`) for the cube in and the scores out. `opu.sh` fills in the addresses from `MEMORY_BASE` in `board.conf` (PicoBoB 0x20000000, ZedBoard 0x10000000). |

`opu.sh` copies the three files to `build/K/BOARD/firmware/`.

### 3. cnn_accelerator: the program on the board (cnn-1d-picobob's)

opu-system is not involved. `software/opu/` is cnn-1d-picobob's program with its file names;
the files that can be the same are, byte for byte, and the others have only the changes these
kernels need (their input is the cube alone, their output 4 int32 a pixel, their driver named
after them):

| File | Compared with cnn-1d-picobob's | Role |
| --- | --- | --- |
| `src/uio_memory_device.{h,cpp}` | the same | Find a UIO device by name, map its memory. |
| `cmake/*.cmake` | the same | Fetch spdlog, libconfig, argparse; the ARM toolchain. |
| `scripts/Dockerfile`, `setup_docker`, `start_docker`, `cross-compile.sh` | the same | The image `cnn-dev-image` (Ubuntu 20.04's ARM compiler), and the build in it into `build-cross/`. |
| `CMakeLists.txt` | changed: the driver's files by pattern | The build. |
| `src/cnn_config.{h,cpp}` | changed: no `--cnn-config`, no rad matrix | The options, the capture config. |
| `src/xcnn_device.{h,cpp}` | changed: the driver through `kernel.h`; `cycles()` | Starts the kernel; the class values from the devicetree. |
| `src/main.cpp` | changed: no rad matrix; the class out of the kernel's scores; the kernel's cycle count logged | Cube into `cube_map`, kernel, labels file. |
| `scripts/process-capture.sh` | changed: the kernel's name from its folder, no `--cnn-config` | One capture in `/media/sd-pl/CAPTURE`. |
| `scripts/make_kernel_header.sh` | ours | Writes `xcnn_driver/kernel.h`: this kernel's own HLS driver functions (`XK_Set_din0`, `XK_Start`, ...) under fixed names, so the same program serves every kernel. |

`opu.sh` unzips the driver from `export.zip` into `software/opu/xcnn_driver/` (where cnn-1d
keeps its own `xcnn_driver/`; git ignores it), writes `kernel.h` there, and runs cnn-1d's
`scripts/cross-compile.sh`, which builds `build-cross/cnn_accelerator` in `cnn-dev-image`. It
keeps a copy of the driver in `build/K/BOARD/driver/`.

### 4. The tarball (this repository)

`opu.sh` packs `build/K/BOARD/K_opu.tar.gz`:

```text
firmware/K.bit.bin, K.bit.bin.md5sum, K.dtbo     from step 2
K/cnn_accelerator                                from step 3
K/load.sh                                        ours: md5sum check, fpgautil, checks hls_kernel is there
K/process-capture.sh                             cnn-1d's: one capture in /media/sd-pl/CAPTURE
```

### 5. On the board

Unpacked in `/media/sd-pl`. The firmware goes where opu-system's tools expect it, so HYPSO
loads it the standard way:

| What | Whose | Role |
| --- | --- | --- |
| `obip fpga reload K.bit.bin -o K.dtbo` (hypso-cli) | HYPSO / opu-services | Loads the bitstream and the overlay on PicoBoB in flight. |
| `K/load.sh` | ours | The same over ssh on a lab board (`fpgautil -b K.bit.bin -o K.dtbo`). |
| `K/process-capture.sh hsi0` | cnn-1d's | Runs `cnn_accelerator` on `/media/sd-pl/hsi0/binned_cube.bip`, writes `hsi0/K-labels.bin` and a log. |
| `K/cnn_accelerator` | cnn-1d's | The kernel on a capture, through the UIO devices the overlay made. |

For a lab image, `tools/image.py prepare` writes the same files a capture has
(`binned_cube.bip`, `capture_config.ini`), so `process-capture.sh IMAGE` runs on it; the
accuracy is counted on the PC with `tools/image.py score`. `scripts/benchmark.sh` (ours) runs
every kernel on every image and keeps the labels for that.

Because the base design is still in the bitstream, `obip fpga hash` and the rest of
opu-services work with the kernel loaded.

## What this repository adds to opu-system, and what it changes

**Nothing in opu-system's git is changed.** Every opu-system script runs as it is, from its
own checkout (`../opu-system`, or `FPGA_OPU_SYSTEM`). What the build leaves in opu-system
is all outside git, in the places cnn-1d's `build-firmware.sh` uses:

| Where in opu-system | What | Written by |
| --- | --- | --- |
| `vivado/projects/K_BOARD_hls_export/` | the unzipped IP | `opu.sh` |
| `vivado/tcl-scripts/tmp/K_BOARD.tcl`, `add_hls_kernel.tcl` | our project script and kernel script (`tmp/` is run_tcl_script.sh's own scratch folder) | `opu.sh` |
| `vivado/projects/K_BOARD/` | the Vivado project, `.sdk/` with the xsa | opu-system's scripts |
| `petalinux/projects/K_BOARD/` | the xsa, the dtsi, `output/` | `opu.sh`, then opu-system's |
| `petalinux/projects/dtbo-project/` | PetaLinux's overlay project | opu-system's `prepare_pl_firmware.sh`, the first time |

Additions, which opu-system has no counterpart for: `scripts/hls.sh`, the kernel script
`add_hls_kernel.tcl`, the overlay `hls_kernel.dtsi`, `make_kernel_header.sh`, `load.sh`,
`benchmark.sh`, the tarball. The program is cnn-1d-picobob's, as in step 3.

Three places where `opu.sh` goes beyond what `build-firmware.sh` does, each because something
would not otherwise work:

1. **cracklib.** PetaLinux 2023.1 downloads the library cracklib from its git branch
   `master`, which its authors have since renamed `main`. Without a copy already in
   opu-system's download cache, `prepare_pl_firmware.sh` stops at line 130 (`petalinux-build`)
   with "Unable to find revision ... in branch master". `opu.sh` adds four lines to
   `dtbo-project/project-spec/meta-user/conf/petalinuxbsp.conf` (not in git) that fetch it
   from `main`; when `dtbo-project` does not exist yet, the first PetaLinux run makes it and
   fails, and `opu.sh` adds the lines and runs it once more.
2. **A machine without opu-system's Vivado 2023.1** (no `/opt/Xilinx/Vivado/2023.1`, no
   `vivado:2023.1` image), such as a laptop with only 2025.2: `run_tcl_script.sh` cannot start
   there, so `opu.sh` does its three steps itself with the `vivado` on PATH (the same two `sed`
   lines on `repoDir`, the same Vivado flags). Where opu-system's Vivado exists, as on the FPGA
   workstation, `run_tcl_script.sh` is used.
3. **Checks between the steps:** that the routed design meets timing, that each step left its
   files, and a clear error when an image or opu-system is missing.

## Side by side with cnn-1d-picobob

| | cnn-1d-picobob | this repository |
| --- | --- | --- |
| HLS | by hand, in the Vitis HLS GUI, → `firmware/export.zip` | `scripts/hls.sh` → `build/K/BOARD/export.zip` |
| Project script | `firmware/cnn-1d-picobob.tcl`, written for it | `boards/opu/project.tcl`, filled in per kernel |
| Kernel in the design | `firmware/add-cnn-accelerator.tcl` | `boards/opu/add_hls_kernel.tcl`, any HLS kernel |
| Overlay | `firmware/cnn-1d-picobob.dtsi` | `boards/opu/hls_kernel.dtsi`, filled in per board |
| Vivado + PetaLinux | `scripts/build-firmware.sh` → opu-system's scripts | `scripts/opu.sh` → the same opu-system scripts |
| Program | `cnn_accelerator`, driver in `xcnn_driver/`, `scripts/cross-compile.sh` | the same files in `software/opu/` (changes in step 3), driver put in `xcnn_driver/` by `opu.sh`, the same `cross-compile.sh` |
| On the board | `obip fpga reload`, `process-capture.sh` | the same, plus `load.sh` for the lab |

## The other boards

The ZCU104 and the PYNQ-Z2 do not use opu-system: `scripts/pynq.sh` builds
`boards/BOARD/system.tcl` in Vivado and packs a zip for the Jupyter notebook. Step 0 is the
same. [framework.md](framework.md) covers them.
