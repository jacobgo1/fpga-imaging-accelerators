# CLAUDE.md

## What this project is

Neural networks for HYPSO hyperspectral images (cloud / land / sea per pixel),
turned into FPGA accelerators with Vitis HLS. Targets:

- **ZCU104** (board `zcu104`, Zynq UltraScale+ ZU7EV), the default board: the family
  HYPSO-3 will fly, and what runs in the lab (PYNQ, a notebook).
- **PicoBoB** (board `picobob`, Zynq-7030): HYPSO-2's OPU. Much smaller (400 DSPs, no
  UltraRAM, 64-bit HP ports), so kernels must be sized for it (`_zynq7030`). The OPU
  system's PetaLinux: built with opu-system's own scripts, see `docs/opu.md`.
- **PYNQ-Z2** (board `pynqz2`, Zynq-7020): smaller still (220 DSPs, 140 BRAM36, 53 K
  LUTs); same PS and ports as the 7030; PYNQ.
- **ZedBoard** (board `zedboard`): the same 7020 chip (`_zynq7020` kernels), on the lab's
  ZedBoards running the OPU system's PetaLinux, as PicoBoB. See `docs/opu.md`.

Trained models are the `.pt` files in `weights/fp32/`. **Never delete them**: all
weights are extracted from them on demand.

## Where things run

- **Laptop (Windows, Git Bash):** editing, `scripts/test.sh K` (g++) and the
  Python tests. No AMD tools.
- **Lab server (Linux):** `source fpga.sh`, then `fpga_build K [BOARD]` for csynth and
  bitstreams. The user runs these and pastes results; Claude cannot.
  - HLS log: `build/K/BOARD/hls.log`
  - reports: `build/K/BOARD/hls/solution/syn/report/`
  - the OPU boards need opu-system next to the repository with its Vivado and PetaLinux
    2023.1 Docker images; an export from Vitis HLS 2025.2 works (`docs/opu.md`).
- **Board:** the ZCU104 with PYNQ: upload `K_pynq.zip`, unzip, run the notebook. PicoBoB and
  the ZedBoard: unpack `K_opu.tar.gz` in `/media/sd-pl`, load, run `K/process-capture.sh`.

## Layout

```text
scripts/             the build, one script a step, each `KERNEL BOARD`, into build/K/BOARD/:
                     test.sh (g++ testbench), hls.sh (Vitis HLS -> export.zip), pynq.sh (PYNQ
                     boards: Vivado + notebook zip), opu.sh (OPU boards: opu-system's Vivado and
                     PetaLinux scripts + cnn_accelerator -> K_opu.tar.gz), build.sh (all); common.sh
fpga.sh              short commands for the server (Tab completes kernels and boards)
src/models/MODEL/    one folder per network; in it one folder per kernel K (K.cpp, named
                     after the kernel), the golden kernel holding the quantized network;
                     tb/K/ its self-checking testbench, with the reference it checks against
                     (tb/golden/batchnorm.hpp: the BatchNorm only those references have)
src/golden/          layers.hpp: the plain-loop layers the golden kernels are built from (what is
                     computed), one template each for float and quantized weights
src/optimized/       streaming, integer fixed-point building blocks (how it is done fast),
                     one module per file, for every board; each kernel sets
                     its chip's facts in its .hpp ("Chip variables for
                     implementation": DSP_PACK_SHIFT for 2 multiplies per
                     DSP, HAS_URAM for patch_buffer) and passes them to the modules
src/common/          hls::stream, hls::vector and hls::burst_maxi shims so kernels compile with plain g++
boards/B/board.conf  the board's part, HLS clock and flow (pynq or opu); boards/B/system.tcl
                     the PYNQ boards' block design, shared by every kernel; boards/opu/ the OPU
                     boards' project script, add_hls_kernel.tcl (the kernel into opu-system's
                     base design, an MMCM for a kernel slower than FCLK_CLK0) and the overlay
software/pynq/        the board notebooks every model shares: classify_golden.ipynb (golden
                     kernels), classify.ipynb (optimized); scripts/pynq.sh packs one with kernel.json.
                     benchmark.ipynb, uploaded on its own to the top of the Jupyter folder: every
                     ZCU104 optimized kernel (from its KERNEL_pynq/) on five images, one report
software/opu/        cnn_accelerator: cnn-1d-picobob's program with its file names, run by every
                     optimized kernel on the OPU system's PetaLinux (no PYNQ). Files that can be
                     are byte-for-byte cnn-1d's; the rest have only the changes these kernels
                     need (docs/opu_flow.md, step 3): keep it that way. scripts/
                     make_kernel_header.sh maps the kernel's own HLS driver to shared names
                     (xcnn_driver/kernel.h); scripts/opu.sh builds it with cnn-1d's
                     cross-compile.sh. tests/test_opu_driver.py builds the real sources on the PC
                     against a fake board. scripts/benchmark.sh, copied on its own to
                     the board: every optimized kernel (from its tarball) on five images, the
                     times and each run's labels; tools/image.py score counts the accuracy on the PC
tools/               weights, numpy reference models, image preparation (tools/README.md)
tests/               Python tests of all of the above, no AMD tools needed
docs/framework.md    what each build step does; docs/opu.md the OPU boards, step by step
docs/justoliunet_opt_design.md  the 1D kernels: design, performance model, next ideas
optimizations_implemented/  one .md per model: its optimizations, stages, latency, utilization, board limits
                     utilization.md: every kernel's routed utilization, from the reports in utilization/
                     (python tools/utilization.py, which also fills each model file's table)
```

Kernels now: `justoliunet` and `justoliunet_bn` (golden baselines of the 1D
per-pixel networks, from src/golden/layers.hpp; numpy model
tools/justoliunet_model.py),
`justoliunet_opt` and `justoliunet_bn_opt` (fast, ZCU104: the raw cube in, a pixel
every 18 cycles, integer fixed point, the DSPs on conv2 and conv3) and their ZedBoard
versions `justoliunet_opt_zynq7020` / `justoliunet_bn_opt_zynq7020` (board `zedboard`:
folded, spectral streaming, a pixel every 108 cycles, the same answers bit for bit; OPU
program from software/opu/) and Zynq-7030 versions `justoliunet_opt_zynq7030` /
`justoliunet_bn_opt_zynq7030` (board `picobob`: the same, wider, a pixel every 74
cycles); the design is in
src/models/1D-Justo-LiuNet/justoliunet_opt/justoliunet_opt.hpp, the bit-exact model classify_fixed in
tools/justoliunet_model.py;
the performance model, the predictions for the 7020/7030 and the ideas still to try,
such as hard-wired weights, are in docs/justoliunet_opt_design.md),
`justounetsimple` (golden baseline), `justounetsimple_opt` (fast,
ZCU104), `justounetsimple_opt_tiled` (the same answers, ZCU104: conv1 in raster order,
conv3x3_tiled packed with a pixel's 110 bands in 5 chunks of 24 from pixel_buffer, so no patch
buffer; 648 DSPs on conv1 instead of 864, 5 285 cycles a patch, under the reads),
`justounetsimple_opt_zynq7020` and `justounetsimple_opt_zynq7030` (the
same, sized for those chips (the 7030 one is the 7020's, wider: 12 bands a cycle, four
read ports, its fifth m_axi port sharing HP3), conv1 as in `_tiled` (chunks of 8 / 12, no patch
buffer): build them for board `pynqz2` or `zedboard` / `picobob`; same answers, they share the testbench's checks),
`sp_unet_small`, `sp_unet_medium`, `sp_unet_large` (golden baselines of SpecPW-UNet, the
same network in three widths: n 32 x 32 patches per start, float, int8 weights; built
from src/golden/layers.hpp, the network in src/models/SpecPW-UNet/specpw_unet.hpp; numpy model and the bit-exact integer model
classify_fixed in tools/specpw_unet_model.py, the same as plain C++ loops in
tb/sp_unet_small_opt/specpw_unet_fixed_golden.hpp), `sp_unet_small_opt`, `_medium_opt`,
`_large_opt` (fast, ZCU104: the raw cube in, bit for bit classify_fixed; direct 3x3 convs
(conv3x3_tiled, PI x PO a cycle, the ReLU inside), raster order, duplicate (to the skip) / max_pool /
upsample; T = 9 828 / 34 944 / 69 888 cycles a patch, 1 452 / 1 297 / 1 184 DSPs; small
measured 26.2 ms at 250 MHz; the design in src/models/SpecPW-UNet/Small/sp_unet_small_opt/sp_unet_opt.hpp,
the chain and each size's table of PI x PO per layer in its .cpp; one testbench check,
tb/sp_unet_small_opt/sp_unet_opt_check.hpp; a Winograd version was removed
2026-10-31, see sp_unet.md),
`sp_unet_small_opt_u8`, `_medium_opt_u8`, `_large_opt_u8` (the same, ZCU104, every activation uint8
with each layer's own fixed fraction bits (src/models/SpecPW-UNet/specpw_unet_act8_fracs.hpp, generated
in git by `python tools/specpw_unet_act8.py --write`, calibrated once on saldanha), so every 3x3 conv
takes two multiplies per DSP (conv3x3_tiled's PACKED mode); bit for bit classify_fixed with
act8_fracs; predicted T = 6 552 (DDR-bound, ~6 700) / 17 472 / 34 944 cycles a patch, 1 347 / 1 297 /
1 184 DSPs; tools/models.py's fixed_scores takes the kernel's name for them),
`sp_unet_small_opt_zynq7020` (the direct kernel for the 7020; medium and large do not
fit the 7020 and were removed (see sp_unet.md); board `pynqz2` or `zedboard`: at most 205 DSPs, convs with the 3x3 window in three row
memories (conv3x3_tiled_ram), a tap or a window column a step, their ReLU inside, one 64-bit read
port, 2-deep FIFOs between the processes; T = 76 176 cycles a patch, measured 665 ms on the
ZedBoard; the chain in
src/models/SpecPW-UNet/Small/sp_unet_small_opt_zynq7020/), `sp_unet_small_opt_u8_zynq7020` (the same with
uint8 activations as the _u8 kernels, conv3x3_tiled_ram's PACKED mode: two multiplies per DSP48E1, a group
of one uint8 product; T = 39 456 cycles a patch, 198 DSPs designed) and `sp_unet_small_opt_u8_zynq7030`,
`_medium_opt_u8_zynq7030`, `_large_opt_u8_zynq7030` (the same chain for the 7030, board `picobob`:
T = 22 194 / 78 912 / 152 352, 364 / 287 / 303 DSPs; small reads with two ports); bit for bit the _u8
kernels; not built yet, see sp_unet.md), `spectralpointwise1d` (golden baseline of
1D-SpectralPointwise: three 1x1 convs per pixel, 110 -> 24 -> 12 -> 3, BatchNorm folded into
the first; numpy model and the bit-exact classify_fixed in tools/spectralpointwise1d_model.py)
and `spectralpointwise1d_opt` (ZCU104), `spectralpointwise1d_opt_zynq7030`,
`spectralpointwise1d_opt_zynq7020` (board `picobob` / `zedboard`): DDR-bound, so the whole
network is copied once per read port (a lane: 4, 4 and 2), convsp1 fused with the z-score word
by word; a pixel every 3.75 / 7.5 / 15 cycles, the same answers bit for bit; the design in
src/models/1D-SpectralPointwise/spectralpointwise1d_opt/spectralpointwise1d_opt.hpp, one testbench check for all),
`waveletcnn` (golden baseline of WaveletCNN: a 1D db4 DWT along the bands,
a 1x1 projection, a 3x3 conv block and a 1x1 head) and `waveletcnn_opt`, `_zynq7030`,
`_zynq7020` (the raw cube in, bit for bit classify_fixed in tools/waveletcnn_model.py; see
optimizations_implemented/waveletcnn.md).

The z-score every model starts with (keep raw bands 8-117, mu_sd.txt) is in one place,
tools/preprocessing.py. tools/models.py lists every model (scripts/pynq.sh packs its entry with the
board notebook as kernel.json, so one notebook serves all). The board is checked against the
image's labels only, in the notebooks and, for the OPU boards, by tools/image.py score on the PC;
tools/image.py makes the image folder (labels required).

## How a model is built (follow justounetsimple_opt)

1. Quantize the weights: `python tools/quantize_weights.py weights/fp32/MODEL`
   (int8, BatchNorm folded, power-of-two scales).
2. Write a bit-exact numpy model of the integer kernel in `tools/` (see
   `classify_fixed` in `tools/justounetsimple_model.py`). It is the reference
   the board is checked against.
3. Write the kernel as a DATAFLOW chain from `src/optimized/` blocks:
   - read the raw cube from DDR and do the preprocessing in hardware;
   - integer fixed point throughout;
   - give each conv its weights, bias, shifts and activation (`activation::relu`, `relu8`,
     `leaky`, `leaky_scores` or `none`) as template arguments, instead of an activation process;
     choose the conv module by its window and input order (docs/framework.md, "The conv modules");
   - the constants in the kernel's .hpp (sizes, chip variables, which multipliers on DSPs);
     those derived from the weights' fractions stay in the .cpp, since the testbench
     includes the .hpp with the golden kernel's float weights;
   - size each stage so none waits for another; FIFO depths in the stream type,
     `hls::stream<T, 16>`.
4. Testbench: bit-exact against a plain-loop integer reference, and close to the
   float golden model. Mutation-test it: inject bugs, check it fails.
5. Add the model to tools/models.py: the shared notebooks then run it (one start for the
   whole image; wall-clock and FPGA time), and tests/test_notebooks.py checks them.

## Before saying something works

Claude runs only the local checks below, never csynth, cosim or bitstream:
those are the user's, on the lab server, every time.

- `python -m unittest discover -s tests`: every test must pass.
- `scripts/test.sh K` for each changed kernel.
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
  (`src/optimized/dsp_pack.hpp`, AMD WP486; used by conv3x3_2mul, conv3x3_tiled's and conv3x3_tiled_ram's PACKED mode,
  pointwise_packed and the 1D convs): int8 activations in -127..127, or uint8. The DSP's wide input sets the
  spacing, the kernel's DSP_PACK_SHIFT: 18 bits on the UltraScale+ DSP48E2 (27-bit input),
  16 on the 7-series DSP48E1 (25-bit); how many products a DSP chain adds follows from it. Check the DSP
  count in the csynth report: about 864 for the ZCU104's conv1, 324 for the 7030's, 216 for the
  7020's, means the packing mapped.
  Pack the weights once at load, not next to the multiply: HLS put that addition
  in the DSP pre-adder and the RTL was slightly wrong (csim passed, cosim failed).
- Processes sized for the same time per patch need deep FIFOs between them (16, `impl=srl`),
  not the default 2: an HLS pipeline freezes all its stages when one read or write waits, so
  with no slack one process's short wait freezes the next and the time is never won back.
  sp_unet_small_opt went from 11 503 to 9 856 cycles a patch with that alone, though
  HLS reported II = 1 and the right interval for every process.
- csim passing does not prove the RTL. After changing arithmetic, ask the user
  for a cosim (`fpga_cosim K BOARD`, `scripts/hls.sh K BOARD cosim`); a mismatch there is
  reproducible in minutes, a mismatch on the board is not.
- HLS knows the weights' values (loaded once from constant arrays) and sizes
  each multiplier for its weight: small weights become LUT shift-adds, zero
  weights vanish. That is why DSP counts can be far below the multiply count.
- Say where every multiplier goes; do not leave it to HLS. A multiply whose
  operands are only known at run time goes through `mul<DSP>`
  (`src/optimized/fixed_point.hpp`): a `BIND_OP` pragma, `impl=dsp` or
  `impl=fabric` (LUTs). `impl` takes only a literal word, so each choice is its
  own branch. The modules take `bool DSP` as their last template argument, so
  each kernel's top function shows it per process. Index arithmetic is always
  LUTs. Check the csynth report's Bind Op Report against it.
- Generated files must stay in sync:
  - `src/optimized/zscore_int8.hpp` (the 2D kernels' integer z-score) comes from
    `python tools/preprocessing.py`, and a test checks it;
  - `src/models/SpecPW-UNet/specpw_unet_act8_fracs.hpp` (the uint8 kernels' fraction bits) comes
    from `python tools/specpw_unet_act8.py --write`, and a test checks it when the saldanha
    capture is there;
  - `weights/quantized/` comes from `quantize_weights.py` and is not in git.

## Working with the user

- Do not commit. The user reviews every change first.
- Keep `optimizations_implemented/` current: whenever a kernel's design changes, update its
  model's file there (the tables and a dated line under "Changes"); a new model gets a new
  file, in the same layout. Measured numbers replace predictions as they come in.
- Shell commands the user will paste must be complete as written: no `<placeholders>`.
- Keep the code comments' style: short, explaining why, plain English.
