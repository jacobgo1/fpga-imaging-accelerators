# CLAUDE.md

## What this project is

Neural networks for HYPSO hyperspectral images (cloud / land / sea per pixel),
turned into FPGA accelerators with Vitis HLS. Targets:

- **ZCU104** (Zynq UltraScale+ ZU7EV), the default board: the family HYPSO-3
  will fly, and what runs in the lab (PYNQ).
- **Zynq-7030** (`--board zynq7030`): HYPSO-2's FPGA. Much smaller (400 DSPs, no
  UltraRAM, 64-bit HP ports), so kernels must be sized for it. On PicoBoB with the
  OPU system's PetaLinux, so the build packs `K_opu.zip` as for the ZedBoard. See
  `boards/README.md`.
- **Zynq-7020** (`--board zynq7020`): the PYNQ-Z2 in the lab now. Smaller still
  (220 DSPs, 140 BRAM36, 53 K LUTs); same PS and ports as the 7030.
- **ZedBoard** (`--board zedboard`): the same 7020 chip, on the lab's ZedBoards
  running the OPU system's PetaLinux, not PYNQ: the build packs `K_opu.zip`
  (.bit.bin, md5sum, devicetree overlay, load.sh) for `fpgautil`. See `boards/README.md`.

Trained models are the `.pt` files in `weights/fp32/`. **Never delete them**: all
weights are extracted from them on demand.

## Where things run

- **Laptop (Windows, Git Bash):** editing, `python main.py native` (g++) and the
  Python tests. No AMD tools.
- **Lab server (Linux):** `source fpga.sh`, then `fpga_build K` for csynth and
  bitstreams. The user runs these and pastes results; Claude cannot.
  - HLS log: `build/K/latest/hls.log`
  - reports: `build/K/latest/hls/solution/syn/report/`
- **Board:** the ZCU104 with PYNQ. Upload `K_pynq.zip`, unzip, run the notebook.

## Layout

```text
main.py              the build: python3 main.py STAGE --kernel K [--board B]
fpga.sh              short commands for the server (Tab completes kernels, runs, boards)
src/hls/K/           one folder per kernel; tb/K/ its self-checking testbench
src/golden/          plain-loop reference layers and models (what is computed)
src/optimized/       streaming, integer fixed-point building blocks (how it is done fast),
                     one module per file: shared/ works on every board; zcu104/,
                     zynq7020/ and zynq7030/ hold each board's version of the modules that
                     differ (patch_buffer: UltraRAM or BRAM; conv3x3_wide: 2 multiplies per
                     DSP, packed for the board's DSP width, or 1)
src/common/          hls::stream, hls::vector and hls::burst_maxi shims so kernels compile with plain g++
boards/B/system.tcl  one block design per board, shared by every kernel
software/pynq/K/     the notebook for kernel K
software/opu/K/      the C program for kernel K on the OPU system's PetaLinux (no PYNQ)
tools/               weights, numpy reference models, image preparation (tools/README.md)
tests/               Python tests of all of the above, no AMD tools needed
docs/framework.md    what each build step does
docs/justoliunet_opt_design.md  the 1D kernels: design, performance model, next ideas
optimizations_implemented/  one .md per model: its optimizations, stages, latency, utilization, board limits
```

Kernels now: `justoliunet` and `justoliunet_bn` (golden baselines of the 1D
per-pixel networks, from src/golden/'s 1D modules; numpy model
tools/justoliunet_model.py; one notebook for both, software/pynq/justoliunet/),
`justoliunet_opt` and `justoliunet_bn_opt` (fast, ZCU104: the raw cube in, a pixel
every 18 cycles, integer fixed point, the DSPs on conv2 and conv3) and their ZedBoard
versions `justoliunet_opt_zynq7020` / `justoliunet_bn_opt_zynq7020` (`--board zedboard`:
folded, spectral streaming, a pixel every 108 cycles, the same answers bit for bit; OPU
program from software/opu/) and Zynq-7030 versions `justoliunet_opt_zynq7030` /
`justoliunet_bn_opt_zynq7030` (`--board zynq7030`: the same, wider, a pixel every 74
cycles); the design is in
src/hls/justoliunet_opt/justoliunet_opt.hpp, the bit-exact model classify_fixed in
tools/justoliunet_model.py, one notebook for both in software/pynq/justoliunet_opt/;
the performance model, the predictions for the 7020/7030 and the ideas still to try,
such as hard-wired weights, are in docs/justoliunet_opt_design.md),
`justounetsimple` (golden baseline), `justounetsimple_opt` (fast,
ZCU104), `justounetsimple_opt_zynq7020` and `justounetsimple_opt_zynq7030` (the
same, sized for those chips (the 7030 one is the 7020's, wider: 12 bands a cycle, four
read ports, its fifth m_axi port sharing HP3): build them with `--board zynq7020` / `zynq7030`; same
answers, they share the testbench's checks; the 7020 one also the notebook),
`justounetsimple_opt_readonly` (justounetsimple_opt's DDR readers with the compute
replaced by a checksum: times the reads alone; its own notebook),
`sp_unet_small`, `sp_unet_medium`, `sp_unet_large` (golden baselines of SpecPW-UNet, the
same network in three widths: n 32 x 32 patches per start, float, int8 weights; built
from src/golden/'s spectral stem, conv block, up block and concat modules; one notebook
for all three, software/pynq/sp_unet_small/; numpy model and the bit-exact integer model
classify_fixed in tools/specpw_unet_model.py, the same as plain C++ loops in
src/golden/specpw_unet_fixed_golden.hpp), `sp_unet_small_opt`, `sp_unet_medium_opt`,
`sp_unet_large_opt` (fast, ZCU104: the raw cube in, bit for bit classify_fixed; the 3x3
convs as integer Winograd F(2x2, 3x3) and the UpBlocks from the low-res maps, the maps in
2x2-block order, the skip in UltraRAM; T = 4 096 / 12 288 / 16 384 cycles a patch, 1 474 / 1 368 / 1 440 DSPs; the
design in src/hls/sp_unet_small_opt/sp_unet_opt.hpp, the chain in sp_unet_opt_dataflow.hpp,
each size's table of PI x PO per layer in its .cpp; one testbench and one notebook,
software/pynq/sp_unet_small_opt/), `spectralpointwise1d` (golden baseline of
1D-SpectralPointwise: three 1x1 convs per pixel, 110 -> 24 -> 12 -> 3, BatchNorm folded into
the first; numpy model and the bit-exact classify_fixed in tools/spectralpointwise1d_model.py)
and `spectralpointwise1d_opt` (ZCU104), `spectralpointwise1d_opt_zynq7030`,
`spectralpointwise1d_opt_zynq7020` (`--board zynq7030` / `zedboard`): DDR-bound, so the whole
network is copied once per read port (a lane: 4, 4 and 2), convsp1 fused with the z-score word
by word; a pixel every 3.75 / 7.5 / 15 cycles, the same answers bit for bit; the design in
src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt.hpp, one testbench check and one
notebook for all), `matmul` (minimal example, used by the framework tests).

## How a model is built (follow justounetsimple_opt)

1. Quantize the weights: `python tools/quantize_weights.py weights/fp32/MODEL`
   (int8, BatchNorm folded, power-of-two scales).
2. Write a bit-exact numpy model of the integer kernel in `tools/` (see
   `classify_fixed` in `tools/justounetsimple_model.py`). It is the reference
   the board is checked against.
3. Write the kernel as a DATAFLOW chain from `src/optimized/` blocks:
   - read the raw cube from DDR and do the preprocessing in hardware;
   - integer fixed point throughout;
   - size each stage so none waits for another.
4. Testbench: bit-exact against a plain-loop integer reference, and close to the
   float golden model. Mutation-test it: inject bugs, check it fails.
5. The notebook: one start for the whole image; print wall-clock and FPGA time.

## Before saying something works

Claude runs only the local checks below, never csynth, cosim or bitstream:
those are the user's, on the lab server, every time.

- `python -m unittest discover -s tests`: every test must pass.
- `python main.py native --kernel K` for each changed kernel.
- Say what still needs the user's build. Timing and resources only come from
  their runs: ask for the relevant `hls.log` lines (`Final II`, burst inference,
  warnings) instead of guessing.
- Measure the whole path, not only the kernel: once the board's ARM is in the
  loop, it dominated wall-clock time (10 s of Python vs 0.05 s of FPGA).

## HLS lessons from this project

- An array index that selects a memory bank must be a constant in the code.
  Choosing the bank at run time duplicates the stores, e.g. II = 5 in the
  patch buffer. Put run-time choices in the address within a bank, or in which
  statement runs.
- `#pragma HLS DEPENDENCE ... false` when reads and writes provably never meet
  (ping-pong halves, line buffers).
- Bursts need consecutive addresses in one loop: read whole patch rows.
- Vitis warns that UltraRAM does not use read-first mode; BRAM is the fallback.
- Two int8 multiplies fit in one DSP when they share an operand
  (`src/optimized/shared/conv3x3_packed.hpp`, AMD WP486): it needs activations in
  -127..127. The DSP's wide input sets the spacing: 18 bits on the UltraScale+
  DSP48E2 (27-bit input), 16 on the 7-series DSP48E1 (25-bit). Check the DSP
  count in the csynth report: about 864 for the ZCU104's conv1, 324 for the 7030's, 216 for the
  7020's, means the packing mapped.
  Pack the weights once at load, not next to the multiply: HLS put that addition
  in the DSP pre-adder and the RTL was slightly wrong (csim passed, cosim failed).
- csim passing does not prove the RTL. After changing arithmetic, ask the user
  for a cosim (`python3 main.py cosim --kernel K`); a mismatch there is
  reproducible in minutes, a mismatch on the board is not.
- HLS knows the weights' values (loaded once from constant arrays) and sizes
  each multiplier for its weight: small weights become LUT shift-adds, zero
  weights vanish. That is why DSP counts can be far below the multiply count.
- Say where every multiplier goes; do not leave it to HLS. A multiply whose
  operands are only known at run time goes through `mul<DSP>`
  (`src/optimized/shared/fixed_point.hpp`): a `BIND_OP` pragma, `impl=dsp` or
  `impl=fabric` (LUTs). `impl` takes only a literal word, so each choice is its
  own branch. The modules take `bool DSP` as their last template argument, so
  each kernel's top function shows it per process. Index arithmetic is always
  LUTs. Check the csynth report's Bind Op Report against it.
- Generated files must stay in sync:
  - `src/hls/justounetsimple_opt/justounetsimple_prep.hpp` comes from
    `python tools/justounetsimple_model.py`, and a test checks it;
  - `weights/quantized/` comes from `quantize_weights.py` and is not in git.

## Working with the user

- Do not commit. The user reviews every change first.
- Keep `optimizations_implemented/` current: whenever a kernel's design changes, update its
  model's file there (the tables and a dated line under "Changes"); a new model gets a new
  file, in the same layout. Measured numbers replace predictions as they come in.
- Shell commands the user will paste must be complete as written: no `<placeholders>`.
- Keep the code comments' style: short, explaining why, plain English.
