# FPGA Imaging Accelerators

Workspace for the specialization project on FPGA acceleration of matrix
operations and neural-network components for hyperspectral imaging, written in
C++ and built with AMD Vitis HLS and Vivado from the command line. The point of
the setup is that adding a kernel means writing C++, not wiring up a build.
See [the project description](docs/project-description.txt).

## Quickstart

Needs **Python 3.10+** and **GCC or Clang**; no Python packages. HLS and FPGA
commands additionally need your own AMD Vitis HLS / Vivado installation.

```powershell
python main.py native                 # compile and run the C++ testbench
python main.py doctor                 # which tools were found
python -m unittest discover -s tests  # check the runner itself
```

Use `python3` on Linux. The bundled `matmul` kernel is an 8x8 signed integer
matrix multiply with a self-checking testbench.

### Tab completion

```bash
source tabcompletion.sh                 # try it
echo "source $PWD/tabcompletion.sh" >> ~/.bashrc   # keep it
```

That gives you a `main` command usable from any directory, completing stages,
flags and their values:

```text
main <TAB>                  doctor kernels parts native csim csynth cosim ...
main csynth --<TAB>         --part --kernel --clock-ns --dry-run ...
main csynth --part <TAB>    zcu104 zybo-z7-20 arty-a7-35t ...
main csim --kernel <TAB>    the kernels present in src/hls/
```

Every list is read from `main.py` and the config files at the moment you press
TAB, so a new kernel, board or flag shows up without touching the script. Bash
only — Git Bash, WSL and Linux.

## Add a kernel

Two files, no configuration:

```text
src/hls/conv2d/conv2d.cpp      synthesizable C++, top function named conv2d
tb/conv2d/conv2d_tb.cpp        main() that returns nonzero on mismatch
```

```powershell
python main.py kernels                 # confirm the files were picked up
python main.py native --kernel conv2d
python main.py csim   --kernel conv2d
python main.py csynth --kernel conv2d
python main.py cosim  --kernel conv2d
```

The runner takes the top function name from the directory name, every `.cpp` in
`src/hls/<name>/` as sources, every `.cpp` in `tb/<name>/` as testbench, and
`config/<name>.tcl` as optimization directives if that file exists. Override any
of this by adding a `kernels.<name>` entry in `config/project.json` — whatever
you put there wins, everything else stays automatic.

### Several files in one kernel

Everything in `src/hls/<name>/` is compiled together, so split a design over as
many `.cpp`/`.hpp` files as you like. The function named after the directory is
the top; every other function it calls becomes a sub-module in the generated
RTL. There is no wiring to declare — HLS builds the hierarchy from the call
graph, and the interface pragmas belong on the top function only. Subdirectories
are not scanned.

Code shared between kernels goes in `src/common/` (outside `src/hls/`, so it is
not mistaken for a kernel) and is listed for the kernels that use it:

```json
"kernels": {
  "conv2d": {
    "sources": ["src/hls/conv2d/conv2d.cpp", "src/common/window.cpp"],
    "include_dirs": ["src/hls/conv2d", "src/common"]
  }
}
```

`top` and `testbench` are still filled in automatically — override only the
fields you name.

Put HLS directives in `config/<name>.tcl` rather than pragmas in the source when
you want to compare variants; keep a baseline result before tuning
(`config/matmul.tcl` has an example).

## Commands

| `python main.py ...` | Result |
| --- | --- |
| `doctor` | Show which tools are on PATH |
| `kernels` | List kernels and the files resolved for each |
| `parts` | List the named FPGA targets you can pass to `--part` |
| `native` | Compile and run the C++ testbench with GCC/Clang; no AMD tools |
| `csim` | HLS C simulation |
| `csynth` | C simulation, then HLS synthesis to RTL |
| `cosim` | ... then C/RTL co-simulation in XSim |
| `export` | ... then a packaged IP ZIP for the Vivado IP catalog |
| `synth` | ... then Vivado synthesis (kernel out of context by default) |
| `impl` | ... then placement and routing; needs a board script |
| `bitstream` | ... then a `.bit`, and optionally an `.xsa` |

Each command rebuilds its prerequisites in a fresh run directory, so a result
can never come from stale RTL. Useful flags: `--kernel`, `--part`, `--clock-ns`,
`--dry-run` (print the commands without running anything), `--config`, `--cxx`,
and `--skip-cosim` (leave out C/RTL co-simulation before export and later
stages, when csim already passed and cosim is too slow to repeat every build).

`csynth` and later stages print the numbers you actually want:

```text
HLS synthesis estimates:
  latency   261 cycles
  II        262 cycles
  timing    4.262 ns (target 10.00 ns, uncertainty 2.70 ns)
  resources DSP 8/1728  BRAM_18K 0/624  URAM 0/96  FF 263/460800  LUT 705/230400
```

That is the real baseline for `matmul` on a ZCU104. The same values go into the
run's `manifest.json`, so runs stay comparable after the terminal scrolls away.

## Set your device

Nothing is assumed about your hardware, so set `part` once in
[config/project.json](config/project.json), or pass `--part` for a single run.
Either takes a board name from [config/parts.json](config/parts.json) or an exact
part string:

```powershell
python main.py parts                   # what the names map to
python main.py csynth --part zcu104            # named target
python main.py csynth --part xc7z020clg400-1   # exact part
```

```text
Stage: csynth; kernel: matmul; part: xczu7ev-ffvc1156-2-e (zcu104); clock: 10.0 ns
```

`config/parts.json` covers common Zynq-7000, Artix-7/Kintex-7, Zynq UltraScale+
and UltraScale+ boards; add your own by editing that file. It is a convenience
list, not a validated one — confirm the exact string against your own
installation before trusting a result:

```tcl
vivado -mode tcl
get_parts -filter {NAME =~ xczu7ev*}
```

`clock_ns` is the target clock period and `jobs` controls Vivado parallelism.
Commit settings the team agrees on. For personal settings, copy the whole file to
`config/project.local.json` (git-ignored) and pass
`--config config/project.local.json`; it is a complete config, not an overlay.

## Load AMD tools

Use the same AMD release everywhere you build, with your device installed. On
Windows, run the vendor `settings64.bat` files in **cmd.exe** and build in that
same session — calling a `.bat` from PowerShell does not import its environment
back into PowerShell:

```bat
call "C:\Xilinx\Vivado\2023.1\settings64.bat"
call "C:\Xilinx\Vitis_HLS\2023.1\settings64.bat"
python main.py cosim
```

On Linux, source the equivalent scripts for your installed release:

```bash
source /tools/Xilinx/Vivado/2024.2/settings64.sh
source /tools/Xilinx/Vitis/2024.2/settings64.sh
python3 main.py cosim
```

Those paths and versions are examples. The runner prefers `vitis-run --mode hls`
and falls back to `vitis_hls -f`; use `--hls-tool` to choose explicitly. All
flows are batch mode, so no display is needed and the same commands work on a
compute server. For native tests on Windows, put your MinGW-w64/MSYS2 UCRT64
`bin` on PATH, or select Clang with `--cxx`. MSVC `cl` is not supported.

## Layout

```text
main.py                 The entry point: python main.py <stage>
tabcompletion.sh        Optional bash completion; source it
config/                 Device/clock settings, named parts, HLS directives
src/hls/<kernel>/       Synthesizable C++
src/rtl/                Handwritten RTL wrappers
tb/<kernel>/            Self-checking C++ testbenches
tb/data/                Small versioned test vectors
boards/                 Board system Tcl (see boards/README.md)
constraints/            Board XDC
scripts/                AMD Tcl flows (hls.tcl, vivado.tcl)
software/               Host and embedded applications
docs/                   Project description and experiment notes
build/<kernel>/<run>/   Tool projects, RTL, waveforms, logs
reports/<kernel>/<run>/ Collected reports and manifest
artifacts/<kernel>/<run>/ IP ZIP, checkpoints, bitstream, XSA
<kernel>/latest         Symlink to the newest run, in build/ and reports/
tests/                  Runner tests (no AMD tools needed)
```

Every run gets its own directory, named `<local date>_<local time>-<stage>`,
for example `build/conv2d/2026-09-21_11-42-07-csynth/`. A second run in the same
second gets a `-2` suffix, so the names still sort in the order the runs
happened, and a `latest` symlink next to them always points at the newest run.

`build/` holds everything the tools produced, `reports/` collects the logs and
reports (including for failed runs), and `artifacts/` receives deliverables from
successful runs only. Each run writes a `manifest.json` with the resolved
config, commands, platform, Git commit and status, and SHA-256 of every source
file. All three directories are git-ignored, so record conclusions worth keeping
in `docs/experiments/`.

### Where the reports are

Every run ends by printing the paths. The ones you normally want:

```text
reports/<kernel>/latest/hls-console.log                             what the tool said
reports/<kernel>/latest/hls/solution/syn/report/<top>_csynth.rpt    area and timing
reports/<kernel>/latest/hls/solution/sim/report/<top>_cosim.rpt     co-simulation
reports/<kernel>/latest/manifest.json                               config + parsed estimates
build/<kernel>/latest/hls/solution/syn/verilog/                     the generated RTL
```

`reports/` gets a copy of every `.rpt`, `.log`, `.jou` and `.xml` the run
produced, keeping the directory structure, and it is written even when the run
fails -- so a failed `csynth` still leaves its console log there to read.

## Board integration

Kernel synthesis gives resource and timing estimates, not a working system. To
build a bitstream you supply clocks, reset, memory access, a top-level wrapper
and real pin constraints — see [boards/README.md](boards/README.md) for the
contract, then set `board_script` in the config. `matmul` keeps its matrices in
AXI-Lite registers, so it can be driven over JTAG with no embedded software —
see [software/README.md](software/README.md). It is a correctness baseline, not
an optimized accelerator.

## How this is built, and what is unverified

The runner is standard-library Python driving checked-in Tcl. No Make (a Windows
dependency), no container (does not remove the AMD install and licensing), and
no GUI project in Git (unreviewable and machine-specific). Runs are never
incremental, because a reproducible number is worth more here than a fast
rebuild.

`native`, `csim`, `csynth` and `cosim` have been run end to end on **Vitis HLS
2023.1 (Windows)** targeting `xczu7ev-ffvc1156-2-e`, with C/RTL co-simulation
passing. `export`, `synth`, `impl` and `bitstream` are **not yet verified**, nor
is any Linux run — expect to fix small things there first, as we did here. The
Tcl tests use Python's `tkinter` interpreter with stub commands to check stage
ordering and syntax; they do not validate AMD command semantics. They need no
display, but many Linux distributions ship `tkinter` separately — without
`python3-tk` those tests skip rather than fail, so on a server check for
`skipped` in the test output before believing a clean run.

Two Windows traps worth knowing, both hit on the first real run:

- `csim` failing with `/dev/null:1: *** missing separator` means a stray file
  exists at `C:\dev\null` (some Windows program redirected to `/dev/null` and
  Windows created it). HLS's generated makefile does `-include /dev/null`, reads
  that file, and every C simulation dies. Delete the file.
- If the AMD tools are not on PATH, either start from an environment set up by
  `settings64.bat`, or point at the launcher directly:
  `--hls-tool "C:/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls.bat"`. A standalone
  Vitis HLS install has no `vitis-run`; the runner falls back to `vitis_hls -f`.
Reference documentation: [Vitis HLS command line](https://docs.amd.com/r/en-US/ug1702-vitis-accelerated-reference/vitis-run-Command),
[Vivado non-project flow](https://docs.amd.com/r/2023.2-English/ug994-vivado-ip-subsystems/Creating-a-Flow-in-Non-Project-Mode),
[write_hw_platform](https://docs.amd.com/r/2023.2-English/ug835-vivado-tcl-commands/write_hw_platform).
