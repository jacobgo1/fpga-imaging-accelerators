# Boards

One script per board, `boards/<board>/system.tcl`: the hardware around the HLS
kernel (processor, interconnect, clock, reset). It is generic: it instantiates
whatever kernel was built and wires its ports, so **every kernel uses the same
board script**. What differs between boards is what fits on the chip, and that
is a property of the kernel (see below).

| Board | `--board` | Part | Why |
| --- | --- | --- | --- |
| ZCU104 | `zcu104` (default) | `xczu7ev-ffvc1156-2-e` | Zynq UltraScale+, the family HYPSO-3 will fly; runs PYNQ in the lab |
| Zynq-7020 | `zynq7020` | `xc7z020clg400-1` | Zynq-7000, the PYNQ-Z2 in the lab (also the Zybo Z7-20; a ZedBoard is `xc7z020clg484-1`). Its `system.tcl` is the 7030's block design |
| ZedBoard | `zedboard` | `xc7z020clg484-1` | The Zynq-7020 again, on the lab's ZedBoards running the OPU system's PetaLinux (HYPSO's Linux, no PYNQ). The 7020's block design; the build packs an OPU zip instead of the PYNQ one |
| Zynq-7030 | `zynq7030` | `xc7z030sbg485-1` | Zynq-7000, HYPSO-2's FPGA, on PicoBoB running the OPU system's PetaLinux: the build packs an OPU zip, as for the ZedBoard. The part is opu-system's `picozed_project.tcl`'s |

Pick the board when building: `fpga_build K --board zynq7030`, or
`python3 main.py bitstream --kernel K --board zynq7030`. The part and script per
board are in `config/project.json` (`boards`). Runs for a board other than the
default end in `-BOARD` (`build/K/DATE-bitstream-zynq7030/`). A board's own
`clock_ns` in the config sets its default clock (10 ns, 100 MHz, on both
Zynq-7000s; `--clock-ns` still overrides it).

## What every board script does with a kernel

- `s_axi_control` (the kernel's AXI-Lite registers) on the processor's
  general-purpose master port, through a SmartConnect;
- each `m_axi` port on its own port to DDR, through its own SmartConnect, in
  name order (`gmem0` -> HP0, ...): on the ZCU104 the four high-performance (HP)
  ports, then the two HPC ports, at most 6; on the 7020 and 7030 the four HP ports,
  and a fifth `m_axi` port shares HP3 with the fourth through its SmartConnect
  (for a writer next to a reader: AXI reads and writes have their own channels);
- one clock from the processor at the HLS target (`clock_ns`), checked to be no
  faster than that target; a `proc_sys_reset` for the reset; the interrupt.

So a kernel works on any board as long as it uses that interface: AXI-Lite
control and no more `m_axi` ports than the board has ports to DDR (6 on the
ZCU104, 4 on the Zynq-7000s, 5 with HP3 shared). An AXI-Stream kernel with a DMA, or one
with its own pins, would need the board script extended.

## What differs: the chip

| | ZU7EV (ZCU104) | 7Z030 (HYPSO-2) | 7Z020 (PYNQ-Z2) |
| --- | --- | --- | --- |
| DSPs | 1,728 (DSP48E2, 27 × 18) | 400 (DSP48E1, 25 × 18) | 220 (DSP48E1) |
| Block RAM | 312 × 36 Kb (11 Mb) | 265 × 36 Kb (9.3 Mb) | 140 × 36 Kb (4.9 Mb) |
| UltraRAM | 96 × 288 Kb (27 Mb) | none | none |
| LUTs | 230 K | 79 K | 53 K |
| HP port width | 128 bits | 64 bits | 64 bits |
| Processor | 4 × Cortex-A53 | 2 × Cortex-A9 | 2 × Cortex-A9 |

A kernel sized for one does not automatically fit the other.
`justounetsimple_opt` uses about 1,100 DSPs, UltraRAM and five `m_axi` ports:
it is a ZCU104 design. `justounetsimple_opt_zynq7030` is the 7020 version below
made wider for the 7030, with the same answers: conv1 at 12 bands per cycle, two
multiplies per DSP (324 DSPs), conv2-4 at 2 input x one output channel per cycle
on DSPs (54), four 64-bit read ports (HP0-3) with the writer sharing HP3, and the
patch buffer in block RAM (about 60 BRAM36). conv1 sets the pace: 10,273 cycles a
patch, about 0.10 ms at 100 MHz, 2.5x the ZCU104's. Build it with `--board zynq7030`.

`justounetsimple_opt_zynq7020` is the same model again, for the 7020's 220 DSPs:
conv1 takes 216 of them at 8 bands per cycle, **two multiplies per DSP** (the
same packing as the ZCU104's, 16 bits apart instead of 18 to fit the DSP48E1's
25-bit input), and every other multiply is in LUTs (`DSP = false` in its
top function): conv2-4 at 2 input x one output channel per cycle, the
z-score, the address arithmetic. Two 64-bit read ports
and one write port, the patch buffer in block RAM (56 BRAM36). The two read
ports are its slowest stage, 15,360 cycles a patch (conv1: 14,369): about
0.16 ms per patch at 100 MHz, 4x the ZCU104's. Build it with
`--board zynq7020`; its zip carries `justounetsimple_opt`'s notebook, which
finds the kernel and its ports from the files next to it.

The modules that differ between the boards are in `src/optimized/zcu104/`,
`zynq7020/` and `zynq7030/`, the rest in `src/optimized/shared/`. `matmul` and
the golden `justounetsimple` are small enough that they should fit any of the
chips as they are (not yet built for the Zynq-7000s).

## The Zynq-7000 processor configuration

There are no Vivado board files for HYPSO-2, so `zynq7030/system.tcl` cannot
take the processor's DDR, MIO and clock settings from a board preset. Put
HYPSO-2's own configuration in `zynq7030/ps7_preset.tcl` as
`set_property -dict [list CONFIG.PCW_... ] $ps` lines (from the flight design's
block design Tcl, `write_bd_tcl`); the script sources it when it exists.
Without it the build still works and warns: those settings are not in the
bitstream (the boot loader configures the processor), but HYPSO-2's boot
configuration must give the FPGA what the design assumes: `FCLK_CLK0` at the HLS
clock and the HP ports in use enabled at 64 bits.

The Zynq-7020 uses the same script (`zynq7020/system.tcl` sources it), with its
preset, if any, in `zynq7020/ps7_preset.tcl`. On the PYNQ-Z2 none is needed:
the PYNQ image boots the processor, and loading the overlay sets `FCLK_CLK0`
from the `.hwh`.

## The ZedBoard: the OPU system instead of PYNQ

The lab's ZedBoards run the OPU system (opu-system's PetaLinux, as on HYPSO), which
loads designs at run time with `fpgautil` and a devicetree overlay. A `zedboard`
bitstream run packs `build/K/RUN/K_opu/` and `K_opu.zip` (`main.py`, `package_opu`),
and so does a `zynq7030` run, for PicoBoB (HYPSO-2's board, also the OPU system):

| File | What it is |
| --- | --- |
| `K.bit.bin` | the bitstream as `fpgautil` takes it (what `bootgen -process_bitstream bin` makes) |
| `K.bit.bin.md5sum` | its hash, with the path opu-system expects: `/media/sd-pl/firmware/K.bit.bin` |
| `K.dtsi` (`K.dtbo`) | the overlay: which bitstream the full region gets, FCLK_CLK0 at the kernel's clock, the kernel's registers (generic-uio). Compiled when `dtc` is on the server's PATH; else `load.sh` compiles it on the board |
| `xK_hw.h` | register offsets |
| `load.sh` | run as root on the board: checks the md5sum, removes the overlay already loaded, loads, sets FCLK_CLK0 through the SLCR if the overlay could not, prints the FPGA state, FCLK_CLK0 and the kernel's AP_CTRL (4: idle) |
| `classify` (`classify.c`, `kernel_regs.h`) | the kernel's program from `software/opu/K/` (or its `include.txt`), built static for the board's ARM when the server has `arm-linux-gnueabihf-gcc` (or `$OPU_CC`); the board has no compiler. `classify cube.npy scores.npy reference_fixed.npy`: runs the kernel from the reserved memory, in strips if the cube does not fit, times it and compares with the PC's answer |

Copy the folder's files to `/media/sd-pl/firmware/` on the board, then
`sh /media/sd-pl/firmware/load.sh`.

A bitstream loaded at run time cannot change the PS: FCLK_CLK0 is set by the
overlay (`xlnx,fclk`), or by `load.sh` if that did not take, and the HP ports keep
the boot loader's settings (64 bits, as on the lab's ZedBoard). `classify` uses the
reserved memory CubeDMA uses: CubeDMA must be idle. The bitstream replaces the whole PL, so the OPU base design
(opu-system's `opu_base.tcl`: an AXI UART, GPIOs, a bitstream-hash ROM) is gone
until the next boot: use a development image (`build_all -d`), where opu-services
does not start by itself. To keep the base design, add the kernel to opu-system's
`ZedBoard_no_comp.tcl` instead.

## Writing a board script

`scripts/vivado.tcl` has already created a project with the board's part and
added the exported HLS IP to its catalog before sourcing the script. Populate
**that** project; do not launch synthesis or implementation yourself.

| Variable | Meaning |
| --- | --- |
| `cfg(root)` | Absolute repository path |
| `cfg(run_dir)` | This run's build directory — keep generated files here |
| `cfg(board)` / `cfg(part)` / `cfg(clock_ns)` | Board name, device and HLS clock target |
| `cfg(top)` | HLS kernel top name (not the board-level top) |
| `ip_repo` | Exported HLS IP catalog directory |

The script should instantiate the HLS IP with the processor, memory,
interconnect and clock/reset logic it needs; assign addresses; validate, save
and generate the block design; add its HDL wrapper and make `system_wrapper`
the top; and constrain any pins it uses (`boards/<board>/system.xdc`). Never
suppress unconstrained-pin DRCs to force a bitstream, and read the
unconstrained-path section of the timing report. Then add the board to `boards`
in `config/project.json`.
