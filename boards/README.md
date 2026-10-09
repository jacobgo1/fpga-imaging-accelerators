# Boards

One script per board, `boards/<board>/system.tcl`: the hardware around the HLS
kernel (processor, interconnect, clock, reset). It is generic: it instantiates
whatever kernel was built and wires its ports, so **every kernel uses the same
board script**. What differs between boards is what fits on the chip, and that
is a property of the kernel (see below).

| Board | `--board` | Part | Why |
| --- | --- | --- | --- |
| ZCU104 | `zcu104` (default) | `xczu7ev-ffvc1156-2-e` | Zynq UltraScale+, the family HYPSO-3 will fly; runs PYNQ in the lab |
| Zynq-7030 | `zynq7030` | `xc7z030sbg485-1` | Zynq-7000, HYPSO-2's FPGA. Check the package and speed grade against HYPSO-2's hardware and change it in `config/project.json` if needed |

Pick the board when building: `fpga_build K --board zynq7030`, or
`python3 main.py bitstream --kernel K --board zynq7030`. The part and script per
board are in `config/project.json` (`boards`). Runs for a board other than the
default end in `-BOARD` (`build/K/DATE-bitstream-zynq7030/`).

## What every board script does with a kernel

- `s_axi_control` (the kernel's AXI-Lite registers) on the processor's
  general-purpose master port, through a SmartConnect;
- each `m_axi` port on its own port to DDR, through its own SmartConnect, in
  name order (`gmem0` -> HP0, ...): on the ZCU104 the four high-performance (HP)
  ports, then the two HPC ports, at most 6; on the 7030 the four HP ports;
- one clock from the processor at the HLS target (`clock_ns`), checked to be no
  faster than that target; a `proc_sys_reset` for the reset; the interrupt.

So a kernel works on any board as long as it uses that interface: AXI-Lite
control and no more `m_axi` ports than the board has ports to DDR (6 on the
ZCU104, 4 on the 7030). An AXI-Stream kernel with a DMA, or one
with its own pins, would need the board script extended.

## What differs: the chip

| | ZU7EV (ZCU104) | 7Z030 (HYPSO-2) |
| --- | --- | --- |
| DSPs | 1,728 | 400 |
| Block RAM | 312 × 36 Kb (11 Mb) | 265 × 36 Kb (9.3 Mb) |
| UltraRAM | 96 × 288 Kb (27 Mb) | none |
| LUTs | 230 K | 79 K |
| HP port width | 128 bits | 64 bits |
| Processor | 4 × Cortex-A53 | 2 × Cortex-A9 |

A kernel sized for one does not automatically fit the other.
`justounetsimple_opt` uses about 1,100 DSPs, UltraRAM, five `m_axi` ports and
two multiplies per DSP in conv1 (which needs the UltraScale+ DSP48E2's 27-bit
input): it is a ZCU104 design. On the 7030 the same model needs fewer bands per
cycle in conv1 (e.g. 4 instead of 32, one multiply per DSP: 216 DSPs), at most
four ports, and its patch buffer in block RAM. `matmul` and the golden `justounetsimple` are small enough that
they should fit either chip as they are (not yet built for the 7030).

## The Zynq-7030 processor configuration

There are no Vivado board files for HYPSO-2, so `zynq7030/system.tcl` cannot
take the processor's DDR, MIO and clock settings from a board preset. Put
HYPSO-2's own configuration in `zynq7030/ps7_preset.tcl` as
`set_property -dict [list CONFIG.PCW_... ] $ps` lines (from the flight design's
block design Tcl, `write_bd_tcl`); the script sources it when it exists.
Without it the build still works and warns: those settings are not in the
bitstream (the boot loader configures the processor), but HYPSO-2's boot
configuration must give the FPGA what the design assumes: `FCLK_CLK0` at the HLS
clock and the HP ports in use enabled at 64 bits.

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
