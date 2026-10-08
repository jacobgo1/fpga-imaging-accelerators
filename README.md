# FPGA Imaging Accelerators

Neural-network accelerators for HYPSO hyperspectral images on a ZCU104, written
in C++ with AMD Vitis HLS and built from the command line. See
[the project description](docs/project-description.txt).

**What happens at each step, from C++ to a result in the notebook on the board,
and how to change the hardware around a kernel: [docs/framework.md](docs/framework.md).**

## Quickstart

On the lab server:

```bash
source fpga.sh                        # once per terminal; fpga_help lists the commands
fpga_env                              # Vivado and Vitis HLS on PATH
fpga_test justounetsimple_opt         # Python tests + the kernel's C++ testbench
fpga_build justounetsimple_opt        # bitstream + build/justounetsimple_opt/latest/justounetsimple_opt_pynq.zip
```

Then upload the zip in Jupyter on the board (`http://BOARD:9090`), run
`!unzip -o justounetsimple_opt_pynq.zip` in a cell, and run the notebook inside it.

`fpga.sh` only shortens commands. Underneath, everything is
`python3 main.py STAGE --kernel KERNEL` (on Windows, `python` instead of `python3`):

| Stage | What it does (each includes the ones above it) |
| --- | --- |
| `native` | Compile the kernel and testbench with g++ and run it. No AMD tools |
| `csim` | The same testbench inside Vitis HLS |
| `csynth` | ... then C++ → Verilog, and print latency and resource estimates |
| `cosim` | ... then simulate the generated Verilog with the testbench |
| `bitstream` | ... then the IP, block design, synthesis, place and route, `.bit`, and the PYNQ zip. `--skip-cosim` leaves out cosim |
| `pynq` | Re-make the PYNQ zip of an earlier run (`--run FOLDER`, default the newest), e.g. after editing the notebook |
| `kernels` | List the kernels and the files found for each |

`--clock-ns 5` overrides the clock target for one run. `main.py` needs only
Python 3.10+, and g++ for `native`.

## Add a kernel

```text
src/hls/NAME/NAME.cpp          synthesizable C++, top function named NAME
tb/NAME/NAME_tb.cpp            main() that returns nonzero on mismatch
```

All `.cpp` files in `src/hls/NAME/` are the kernel, and all in `tb/NAME/` are the
testbench. Includes are searched in `src/hls/NAME/` and `src/common/`. If the
top function has another name, add it under `kernels` in
[config/project.json](config/project.json). HLS directives can go in
`config/NAME.tcl` instead of pragmas (see `config/matmul.tcl`). To run it on the
board, add a notebook in `software/pynq/NAME/` (see docs/framework.md, section 5).

## Where a run's output goes

Each run gets one folder, `build/KERNEL/DATE-STAGE/`, with everything in it, and
`build/KERNEL/latest` points at the newest:

```text
build/KERNEL/latest/
  run.json                                    kernel, stage, git commit, part, clock, passed/failed
  hls.log, vivado-console.log                 what the tools printed
  hls/solution/syn/report/TOP_csynth.rpt      HLS estimate: latency, resources
  hls/solution/syn/verilog/                   the generated Verilog
  hls/solution/sim/report/TOP_cosim.rpt       co-simulation result
  vivado/system.xpr                           the Vivado project: open it to see the block design
  reports/impl/utilization.rpt, timing.rpt    the real numbers after place and route
  KERNEL_pynq.zip                             what goes to the board
```

`build/` is not in git, so write conclusions worth keeping in `docs/experiments/`.

## Settings

[config/project.json](config/project.json): the FPGA part (the ZCU104's
`xczu7ev-ffvc1156-2-e`), `clock_ns`, `jobs` (Vivado threads), the board design
script, and top-function names. Use the same AMD release everywhere you build.
On Linux, `fpga_env` sources Vivado's `settings64.sh`. On Windows, run the
`settings64.bat` files in **cmd.exe** and build from that same window (calling a
`.bat` from PowerShell doesn't keep its PATH). `main.py` uses `vitis-run` if it's
on PATH, otherwise `vitis_hls`.

## Layout

```text
main.py            the build: python3 main.py STAGE --kernel KERNEL
fpga.sh            short commands for the lab server
config/            project.json, optional HLS directives per kernel
src/hls/KERNEL/    synthesizable C++, one folder per kernel
src/golden/        reference layers and models: plain loops, what is computed
src/optimized/     streaming, fixed-point building blocks: how it is computed fast
src/common/        C++ shared between kernels (hls::stream for plain g++)
tb/KERNEL/         self-checking testbenches
scripts/           hls.tcl (Vitis HLS), vivado.tcl (Vivado)
boards/zcu104/     system.tcl: the block design around the kernel
software/pynq/     a notebook per kernel, packed into the zip for the board
tools/             Python: weights, quantization, image preparation (tools/README.md)
tests/             tests of all the above, no AMD tools or board needed
docs/              how it works, project description, experiment notes
```

## Things to know

- The Tcl tests replace AMD commands with recorders: they check order and
  wiring, not that Vivado accepts the commands. They need Python's `tkinter`.
  On Linux without `python3-tk` they're skipped rather than failed, so look for
  `skipped` in the test output.
- `csim` on Windows failing with `/dev/null:1: *** missing separator` means a
  stray file exists at `C:\dev\null`. HLS's makefile reads it; delete it.
- Reference: [Vitis HLS command line](https://docs.amd.com/r/en-US/ug1702-vitis-accelerated-reference/vitis-run-Command),
  [Vivado block design Tcl](https://docs.amd.com/r/2023.2-English/ug994-vivado-ip-subsystems/Creating-a-Flow-in-Non-Project-Mode).
