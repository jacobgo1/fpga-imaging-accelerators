# FPGA Imaging Accelerators

Neural-network accelerators for HYPSO hyperspectral images, written in C++ with AMD
Vitis HLS and built from the command line with the scripts in `scripts/`: for the ZCU104
and the PYNQ-Z2 (PYNQ, a Jupyter notebook) and for PicoBoB and the ZedBoard (the OPU
system's PetaLinux, built the way opu-system builds its own designs).

**What happens at each step, from C++ to a result on the board, and how to change the
hardware around a kernel: [docs/framework.md](docs/framework.md).** The OPU boards:
[docs/opu.md](docs/opu.md).

## Quickstart

On the lab server:

```bash
source fpga.sh                                  # once per terminal; fpga_help lists the commands
fpga_env                                        # Vivado and Vitis HLS on PATH
fpga_test justounetsimple_opt                   # Python tests + the kernel's C++ testbench
fpga_build justounetsimple_opt                  # ZCU104: build/justounetsimple_opt/zcu104/justounetsimple_opt_pynq.zip
fpga_build sp_unet_small_opt_u8_zynq7030 picobob   # PicoBoB: build/.../picobob/..._opu.tar.gz
```

ZCU104: upload the zip in Jupyter on the board (`http://BOARD:9090`), run
`!unzip -o justounetsimple_opt_pynq.zip` in a cell, and run the notebook inside it. Put the
image zip (`aegean_unet.zip`) once in the folder above the designs' folders: every notebook
looks there, it is unpacked there once, and each notebook writes its results
(`aegean_unet_fpga_scores.npy`) next to itself.

PicoBoB and the ZedBoard: unpack the tarball in `/media/sd-pl`, load it
(`obip fpga reload K.bit.bin -o K.dtbo`, or `load.sh`) and run `K/classify` or
`K/process-capture.sh`: [docs/opu.md](docs/opu.md).

## The scripts

`fpga.sh` only shortens commands. Underneath, each step is a script, with the kernel and
the board (`zcu104`, `pynqz2`, `picobob`, `zedboard`: `boards/BOARD/board.conf`):

| Script | What it does | Leaves in `build/KERNEL/BOARD/` |
| --- | --- | --- |
| `scripts/test.sh KERNEL` | the kernel and its testbench with g++, run. No AMD tools | (`build/KERNEL/native/`) |
| `scripts/hls.sh KERNEL BOARD [STAGE]` | Vitis HLS. `csim`: the testbench in HLS; `csynth`: ... then C++ → Verilog; `cosim`: ... then the Verilog against the testbench; `ip` (the default): ... then the IP. `--cosim` adds cosim to `ip`, `--clock-ns NS` overrides the board's clock | `hls/`, `hls.log`, `export.zip` |
| `scripts/pynq.sh KERNEL BOARD` | PYNQ boards: Vivado with `boards/BOARD/system.tcl`, then the zip for the notebook. `--package`: only the zip again | `vivado/`, `reports/`, `KERNEL_pynq.zip` |
| `scripts/opu.sh KERNEL BOARD` | OPU boards: opu-system's Vivado and PetaLinux scripts, then `classify` for the board. `--driver`: only classify and the tarball again | `sdk/`, `firmware/`, `KERNEL_opu.tar.gz` |
| `scripts/build.sh KERNEL BOARD` | `hls.sh ... ip`, then `pynq.sh` or `opu.sh` | all of the above |
| `scripts/kernels.sh` | the kernels and the boards | |
| `scripts/get-petalinux.sh [HOST]` | once, for the OPU boards: the PetaLinux Docker image from the FPGA workstation (`fpgaws`) | |

`export.zip` is all the board steps need: with one built by someone else, put it in
`build/KERNEL/BOARD/` and run `pynq.sh` or `opu.sh` alone (`fpga_bitstream KERNEL BOARD`).
The scripts need bash, and Python 3 for the PYNQ zip; nothing to install.

## Add a kernel

```text
src/models/MODEL/FOLDER/NAME.cpp  synthesizable C++, top function named NAME
tb/NAME/NAME_tb.cpp            main() that returns nonzero on mismatch
```

A kernel is found by its `.cpp`: all `.cpp` files in the folder holding `NAME.cpp` are
the kernel, and all in `tb/NAME/` are the testbench. Includes are searched in that
folder and `src/common/`. The top function has the kernel's name. To run it on the
board, add its model to `tools/models.py` (see docs/framework.md, section 5).

## Where a build's output goes

Everything for a kernel on a board is in one folder, `build/KERNEL/BOARD/`, which the next
build of the same overwrites:

```text
build/KERNEL/BOARD/
  build.txt                                   kernel, board, part, clock, git commit, date
  hls.log                                     what Vitis HLS printed
  hls/solution/syn/report/TOP_csynth.rpt      HLS estimate: latency, resources
  hls/solution/sim/report/TOP_cosim.rpt       co-simulation result
  export.zip                                  the IP, with its C driver
  PYNQ boards:  vivado/system.xpr (open it to see the block design), reports/impl/
                (utilization, timing), KERNEL_pynq.zip
  OPU boards:   sdk/ (system.xsa, bitstream, utilization.rpt, timing.rpt), firmware/
                (.bit.bin, md5sum, .dtbo), KERNEL_opu.tar.gz; the Vivado project is in
                opu-system's vivado/projects/KERNEL_BOARD/
```

`build/` is not in git, so write conclusions worth keeping in `docs/experiments/` or
`optimizations_implemented/`.

## Settings

Each board's in `boards/BOARD/board.conf`: its FPGA part, the HLS clock (`CLOCK_NS`: 5 ns
on the ZCU104, 12.8 on the Zynq-7020s, 10 on PicoBoB), and its flow (`pynq` or `opu`).
`FPGA_JOBS` sets how many Vivado runs go at once (default: every core, as `nproc` counts
them), each with up to 8 threads. `FPGA_OPU_SYSTEM` is where opu-system is (default
`../opu-system`). Use the same AMD release for HLS and Vivado (2025.2 works for every
board; only the OPU boards' PetaLinux is 2023.1, in Docker). On Linux, `fpga_env` sources
Vivado's `settings64.sh`. `hls.sh` uses `vitis-run` if it is on PATH, otherwise `vitis_hls`.

## Layout

```text
fpga.sh            short commands for the lab server
scripts/           the build: test.sh, hls.sh (+ hls.tcl), pynq.sh (+ vivado.tcl), opu.sh, build.sh
src/models/        synthesizable C++: one folder per network, one per kernel in it
src/golden/        layers.hpp: the plain-loop layers the golden kernels are built from
src/optimized/     streaming, fixed-point building blocks: how it is computed fast;
                   one module per file; each kernel sets its chip's facts at its top
src/common/        C++ shared between kernels (hls::stream, hls::vector, hls::burst_maxi for plain g++)
tb/KERNEL/         self-checking testbenches, with the references they check against
boards/BOARD/      board.conf; for the PYNQ boards system.tcl, the block design around the kernel
boards/opu/        the OPU boards' project script, the kernel's part of the design, the overlay
software/pynq/     the notebooks every model shares, packed into the zip for the board
software/opu/      classify, the C++ program for the OPU boards (as cnn-1d-picobob's)
tools/             Python: weights, quantization, image preparation (tools/README.md)
tests/             tests of all the above, no AMD tools or board needed
docs/              how it works, project description, experiment notes
```

## Things to know

- The Tcl tests replace AMD commands with recorders, and the script tests replace Vitis,
  Vivado, Docker and opu-system with stand-ins: they check order and wiring, not that the
  tools accept the commands. The Tcl tests need Python's `tkinter`. On Linux without
  `python3-tk` they're skipped rather than failed, so look for `skipped` in the test output.
- `csim` on Windows failing with `/dev/null:1: *** missing separator` means a
  stray file exists at `C:\dev\null`. HLS's makefile reads it; delete it.
- Reference: [Vitis HLS command line](https://docs.amd.com/r/en-US/ug1702-vitis-accelerated-reference/vitis-run-Command),
  [Vivado block design Tcl](https://docs.amd.com/r/2023.2-English/ug994-vivado-ip-subsystems/Creating-a-Flow-in-Non-Project-Mode).
