# Boards

One folder per board, `boards/<board>/`, with its `board.conf`: the FPGA part, the HLS clock
(`CLOCK_NS`) and the flow, how a kernel gets onto it:

- `pynq` (`scripts/pynq.sh`): the board runs PYNQ. `system.tcl` in the board's folder is the
  hardware around the HLS kernel (processor, interconnect, clock, reset); a notebook drives it.
- `opu` (`scripts/opu.sh`): the board runs the OPU system's PetaLinux. The design is
  opu-system's own (its board script, `opu_base.tcl`) with the kernel added by
  `boards/opu/add_hls_kernel.tcl`, built by opu-system's scripts; cnn-1d-picobob's `cnn_accelerator` drives it
  ([docs/opu.md](../docs/opu.md)).

Either way the scripts are generic: they instantiate whatever kernel was built and wire its
ports, so **every kernel uses the same board files**. What differs between boards is what
fits on the chip, and that is a property of the kernel (see below).

| Board | Flow | Part | Why |
| --- | --- | --- | --- |
| `zcu104` (default) | `pynq` | `xczu7ev-ffvc1156-2-e` | Zynq UltraScale+, the family HYPSO-3 will fly; runs PYNQ in the lab |
| `pynqz2` | `pynq` | `xc7z020clg400-1` | Zynq-7020, the PYNQ-Z2 in the lab. Its `system.tcl` is the Zynq-7000 block design |
| `zedboard` | `opu` | `xc7z020clg484-1` | The Zynq-7020 again, on the lab's ZedBoards with the OPU system's PetaLinux (HYPSO's Linux, no PYNQ) |
| `picobob` | `opu` | `xc7z030sbg485-1` | Zynq-7030, HYPSO-2's OPU, with the OPU system's PetaLinux. The part is opu-system's `picozed_project.tcl`'s |

Pick the board when building: `fpga_build K picobob`, or `scripts/build.sh K picobob`.
Everything for a kernel on a board goes to `build/K/BOARD/`. A board's `CLOCK_NS` is its
default HLS clock (5 ns on the ZCU104, 12.8 on the Zynq-7020s, 10 on PicoBoB; `--clock-ns`
overrides it).

## What every PYNQ board script does with a kernel

- `s_axi_control` (the kernel's AXI-Lite registers) on the processor's
  general-purpose master port, through a SmartConnect;
- each `m_axi` port on its own port to DDR, through its own SmartConnect, in
  name order (`gmem0` -> HP0, ...): on the ZCU104 the four high-performance (HP)
  ports, then the two HPC ports, at most 6; on the 7020 and 7030 the four HP ports,
  and a fifth `m_axi` port shares HP3 with the fourth through its SmartConnect
  (for a writer next to a reader: AXI reads and writes have their own channels);
- one clock from the processor at the HLS target (`clock_ns`), never faster than
  that target: when the PLL can only make a slightly faster rate, the script asks
  for 1% less until it gets one at or below the target; a `proc_sys_reset` for the
  reset; the interrupt.

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
patch, about 0.10 ms at 100 MHz, 2.5x the ZCU104's. Build it for board `picobob`.

`justounetsimple_opt_zynq7020` is the same model again, for the 7020's 220 DSPs:
conv1 takes 216 of them at 8 bands per cycle, **two multiplies per DSP** (the
same packing as the ZCU104's, 16 bits apart instead of 18 to fit the DSP48E1's
25-bit input), and every other multiply is in LUTs (`DSP = false` in its
top function): conv2-4 at 2 input x one output channel per cycle, the
z-score, the address arithmetic. Two 64-bit read ports
and one write port, the patch buffer in block RAM (56 BRAM36). The two read
ports are its slowest stage, 15,360 cycles a patch (conv1: 14,369): about
0.16 ms per patch at 100 MHz, 4x the ZCU104's. Build it with
board `pynqz2` (or `zedboard`); its zip carries `justounetsimple_opt`'s notebook, which
finds the kernel and its ports from the files next to it.

Every module is in `src/optimized/`; what differs between the chips (the
DSP packing, UltraRAM or not) is at the top of each kernel's `.cpp`, under "Chip variables for implementation",
and the kernel passes it to the modules. The golden
`justounetsimple` is small enough that they should fit any of the
chips as they are (not yet built for the Zynq-7000s).

## The Zynq-7000 processor configuration

On the PYNQ-Z2, `pynqz2/system.tcl` takes no processor preset: the PYNQ image boots the
processor, and loading the overlay sets `FCLK_CLK0` from the `.hwh`. A preset, if one is ever
needed, goes in `pynqz2/ps7_preset.tcl` as `set_property -dict [list CONFIG.PCW_... ] $ps`
lines; the script sources it when it exists. On the OPU boards the processor is
opu-system's own (`picozed_project.tcl`, `zedboard_project.tcl`), as in HYPSO's flight design.

## The OPU boards: PicoBoB and the ZedBoard

`boards/opu/project.tcl` is a project script in the shape of opu-system's own
(`PicoBoB_no_comp.tcl`, cnn-1d-picobob's `cnn-1d-picobob.tcl`): the board's PS
(`OPU_BOARD_SCRIPT` in `board.conf`: `picozed_project.tcl` or `zedboard_project.tcl`),
`opu_base.tcl`, then `boards/opu/add_hls_kernel.tcl`, then `finish_design.tcl`.
`scripts/opu.sh` fills in the project's name, the kernel and its clock, and runs it with
opu-system's `vivado/run_tcl_script.sh`. What `add_hls_kernel.tcl` adds:

- the kernel, block `hls_kernel` (the overlay refers to it by that name);
- `s_axi_control` on one more master of the base design's `axi_interconnect_0`;
- each `m_axi` port through its own SmartConnect to its own 64-bit HP port, in name order,
  a fifth sharing HP3, as on the PYNQ-Z2. HP0's clock, tied to GP0's in `opu_base.tcl`,
  moves to the kernel's;
- the kernel's clock: `FCLK_CLK0` (100 MHz, set by the boot configuration, which a bitstream
  loaded at run time cannot change) for a kernel built for 10 ns; an MMCM (`hls_clock`) and
  its own reset for one built for a slower clock (12.8 ns on the ZedBoard);
- a hook after routing that writes `utilization.rpt` (hierarchical) and `timing.rpt` next to
  the hardware files.

The base design (AXI UART, timer, bitstream-hash ROM) stays in the bitstream, so a design with
a kernel is still an OPU design: opu-services and `obip fpga hash` work with it loaded.
`boards/opu/hls_kernel.dtsi` is the overlay's part: the kernel and the bitstream hash as UIO
devices, and two maps in the reserved memory (`MEMORY_BASE` in `board.conf`) for the cube
and the scores.

## Writing a board script (PYNQ boards)

`scripts/vivado.tcl` has already created a project with the board's part and
added the exported HLS IP to its catalog before sourcing the script. Populate
**that** project; do not launch synthesis or implementation yourself.

| Variable | Meaning |
| --- | --- |
| `cfg(root)` | Absolute repository path |
| `cfg(run_dir)` | The build folder, `build/KERNEL/BOARD/` — keep generated files here |
| `cfg(board)` / `cfg(part)` / `cfg(clock_ns)` | Board name, device and HLS clock target |
| `cfg(top)` | HLS kernel top name (not the board-level top) |
| `ip_repo` | Exported HLS IP catalog directory |

The script should instantiate the HLS IP with the processor, memory,
interconnect and clock/reset logic it needs; assign addresses; validate, save
and generate the block design; add its HDL wrapper and make `system_wrapper`
the top; and constrain any pins it uses (`boards/<board>/system.xdc`). Never
suppress unconstrained-pin DRCs to force a bitstream, and read the
unconstrained-path section of the timing report. Then give the board its
`board.conf` (`PART`, `CLOCK_NS`, `FLOW=pynq`).
