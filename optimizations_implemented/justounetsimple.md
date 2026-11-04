# 2D-JustoUNet-Simple (`justounetsimple`)

Patch classifier: 3×3 convolutions over 32 × 32 patches of the capture (110 bands in),
a score per pixel. Spatial context, so the image is cut into patches.

## Model and flow

```
raw cube (H x W x 120 uint16) → patches of 32 x 32 (edge rows/columns repeated)
 └ keep bands 8-117, z-score                          32 x 32 x 110
 └ A: conv 3x3 110→6 → ReLU → maxpool 2               → 16 x 16 x 6
 └ B: conv 3x3 6→12  → ReLU → maxpool 2 → upsample 2  → 16 x 16 x 12
 └ C: conv 3x3 12→6  → ReLU → upsample 2              → 32 x 32 x 6
 └ D: conv 3x3 6→3                                    → 32 x 32 x 3 scores, class = argmax
```

Work per 32 × 32 patch: conv1 6.08 M MACs (92 %), conv2-4 0.17 M each; ~6 400 per pixel
(the 1D model needs ~9×). Arithmetic intensity ~26 MACs per byte read: **memory-bound on
the ZCU104** (the 1D model is compute-bound).

## Kernels

| Kernel | Board | Clock | Status |
|---|---|---|---|
| `justounetsimple` | ZCU104 | 5 ns | golden baseline: float, one patch per start; **measured 194.4 s** for a whole capture (292.4 ms a patch), 202.3 s wall clock, 96.20 % |
| `justounetsimple_opt` | ZCU104 | 214 MHz | **measured 18.26 ms** for a whole capture, FPGA alone (5 882 cycles a patch; memory-bound); 19.2 ms wall clock |
| `justounetsimple_opt_zynq7030` | Zynq-7030 (`--board zynq7030`) | 10 ns | built in C++, bit-exact; cycle estimates only (the 7020 design, wider) |
| `justounetsimple_opt_zynq7020` | PYNQ-Z2 / ZedBoard (`--board zynq7020` / `zedboard`) | 12.8 ns (PLL: 76.189 MHz) | **measured 143.8 ms** on the ZedBoard, FPGA alone (~16 480 cycles a patch) |

All optimized kernels give the same answers bit for bit (`classify_fixed` in
`tools/justounetsimple_model.py`); same class as the float model on 99.96 % of the aegean pixels.

## Optimizations used

| Technique | ZCU104 | 7030 | 7020 |
|---|---|---|---|
| Loop tiling into 32 × 32 patches, edges repeated (no host padding) | ✓ | ✓ | ✓ |
| Preprocessing in hardware: band drop + integer z-score (int8, 5 fraction bits) | ✓ | ✓ | ✓ |
| Fixed point: int8 weights, int8 into conv1, int16 activations, int32 sums | ✓ | ✓ | ✓ |
| DATAFLOW, layers fused; maps never leave the chip | ✓ | ✓ | ✓ |
| Ping-pong patch buffer: fill patch k+1 while conv1 reads patch k | UltraRAM | BRAM | BRAM |
| Read ports, each every n-th patch row | 4 × 128 bit | 4 × 64 bit (writer shares HP3) | 2 × 64 bit |
| Read bursts requested ahead (`read_patch_rows`, `burst_maxi`); else one burst a row from a plain pointer (`read_patch_rows_simple`) | plain pointer (on trial, 2026-11-03) | ✓ (2 in flight) | ✓ (2 in flight) |
| Line buffers + 3×3 shift-register window (`conv3x3_all_in_patch`) | ✓ | ✓ | ✓ |
| conv1: P1 bands a cycle, band chunks accumulated | 32 | 12 | 8 |
| Two multiplies per DSP in conv1 | ✓ (DSP48E2, groups of 8) | ✓ (DSP48E1, groups of 2) | ✓ (DSP48E1, groups of 2) |
| conv2-4: one output channel a cycle (narrow) / 2 inputs x 1 output a cycle (`conv3x3_tiled`) | narrow, DSP | grouped (2 in, 1 out), DSP | grouped (2 in, 1 out), LUTs |
| ReLU in the conv (`activation::relu`), pool and upsample one process each (`max_pool`, `upsample`); before 2026-11-04 fused (`relu_pool`, `relu_pool_up`, `relu_up`) | ✓ | ✓ | ✓ |
| Score writes with acknowledgements collected later (`store_scores`) | ✓ | ✓ | ✓ |
| FIFO depths from a cycle model (`s_a` 256, `s_b` 32, `s_c` 64) | ✓ | not checked | not checked |
| Memory bank chosen by a constant index (avoids II 5 in the patch buffer) | ✓ | ✓ | ✓ |
| Cycle counter register | ✓ | – | – |

## Latency per stage (cycles per patch)

Each stage processes one patch at a time, one loop iteration per cycle (II = 1); the
slowest stage sets the pace, T cycles per patch.
- conv1: ceil(110 / P1) band chunks × 1 024 pixels + 33 (the window lag).
- A reader: (32 / ports) rows × 32 pixels × words per pixel.

| Stage | ZCU104 | 7030 | 7020 |
|---|---|---|---|
| read (per port) | 3 840 | 7 680 | **15 360** |
| patch_buffer | 4 096 | 10 240 | 14 336 |
| conv1 | **4 129** | **10 273** | 14 369 |
| conv2 / conv3 / conv4 | 3 276 / 1 638 / 3 171 | 9 216 each | 9 216 each |
| pool, upsample, store (ZCU104: one process each, the ReLU in its conv) | ≤ 1 280 | small | small |
| **T** | **4 129** (conv1) | **10 273** (conv1) | **15 360** (readers) |
| DSPs | ~1 112 (conv1 864, conv2-4 216, z-score 32) | 378 of 400 (conv1 324, conv2-4 54) | 216 of 220 (conv1) |

conv1's DSPs: OUT / 2 pairs × P1 × 9 taps (packed), so 3 × 12 × 9 = 324 on the 7030; 16
bands would need 432. Utilization of conv1's multipliers: u = 6.08 M / (T × 6 × P1 × 9):
7030 91 % (the 10 zero bands of its last chunk), 7020 92 % (it waits for the readers 7 %).

## Inference time

time = (patches + 1) × T / clock (+1: the ping-pong buffer's first fill);
a 598 × 1092 capture is 19 × 35 = 665 patches.

| Board | T | clock | time |
|---|---|---|---|
| ZCU104 | 4 129 | 187.5 MHz | **14.7 ms** (13.7 at 200 MHz) if memory kept up; **measured 18.26 ms at 214 MHz** (5 882 cycles a patch: its reads get ~9.0 GB/s, though DDR gives 13) and **17.87 ms at 250 MHz** (6 718 cycles, 26.87 us a patch) |
| Zynq-7030 | 10 273 | 100 MHz | **68 ms**, if DDR gives the four ports 2.4 GB/s together (check on HYPSO-2) |
| Zynq-7020 | 15 360 | 78.1 MHz | **131 ms**; **measured 143.8 ms at 76.189 MHz** (~16 480 cycles a patch, +7 %: DDR) |

## Boards: limits, differences and why

| | ZCU104 | Zynq-7030 | Zynq-7020 |
|---|---|---|---|
| Limit | memory, then conv1 | conv1's DSPs (16 bands would need 432) | read bandwidth (64-bit ports), then conv1 |
| DSPs / LUTs | 1 728 / 230 k | 400 / 79 k | 220 / 53 k |
| conv1's width | 32 bands a cycle, 864 DSPs packed | 12 bands a cycle, 324 DSPs packed | 8 bands a cycle, 216 DSPs packed |
| conv2-4 | DSPs (plenty) | DSPs (76 left after conv1) | LUTs (4 DSPs left) |
| Patch buffer | UltraRAM | BRAM (~60 of 265 BRAM36) | BRAM (56 of 140 BRAM36) |
| Ports | 4 read + 1 write, 128 bit | 4 read + 1 write sharing HP3, 64 bit (4 HP ports) | 2 + 1, 64 bit |
| Reads needed at T (bytes a patch / T × clock) | 11.2 GB/s of 4 × 3.0 | 2.4 GB/s of 4 × 0.8 | 1.25 GB/s of 2 × 0.62 (the limit) |

## Utilization

<!-- utilization.py: justounetsimple -->
Routed, kernel alone (without the interconnects); by process in
[utilization.md](utilization.md). Written by `python tools/utilization.py`.

| Kernel | Board | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|---|
| (no report yet) | | | | | | |
<!-- /utilization.py -->

## Changes

- 2026-10-05: golden kernel; optimized modules.
- 2026-10-07: control moved to the FPGA (one start per image); int8 into conv1.
- 2026-10-09: 4 read ports, 2 multiplies per DSP, 32 bands a cycle in conv1.
- 2026-10-12: board-specific modules; Zynq-7030 and 7020 versions; write acknowledgements collected later.
- 2026-10-14: ZCU104: `s_b`/`s_c` FIFO depths (model: 6 624 → 4 129 cycles/patch) and read-ahead bursts.
- 2026-10-16: cycle counter; timeline taps (removed again 2026-10-19, `0abe48b`); read-only test kernel; ZedBoard target.
- 2026-10-19: Zynq-7030 version rebuilt as the 7020's, wider: conv1 packed at 12 bands (324 DSPs),
  conv2-4 on DSPs, 4 read ports (the writer shares HP3): 28 705 → 10 273 cycles/patch, 191 → 68 ms.
- 2026-10-21: Zynq-7030 and 7020: `s_b`/`s_c` FIFO depths (32 and 64, matching ZCU104 so `relu_pool_up`
  and `relu_up` never stall on the default 2-deep FIFOs) and `read_rows_ahead` (4 bursts of 8 pixels =
  240 words per row, 2 in flight per port); shared `patch_buffer_run` in `src/optimized/patch_buffer.hpp`.
- 2026-10-27: ZCU104 measured 6 707 cycles a patch, 17.8 ms at 250 MHz (conv1: 4 129). Not the FIFOs
  between the stages: a cycle model with HLS's pipeline stalls (the one that matched
  sp_unet_small_opt within 0.3 %) gives 4 140 with ideal memory, as built, because every
  stage after conv1 has time to spare. With DDR in the model, the kernel takes exactly as long as the
  read-only kernel (`patch_buffer` hides nothing and loses nothing; deeper `cube_words` changes nothing):
  6 707 cycles at 250 MHz is DDR giving the four ports ~9.2 GB/s together. conv1's 4 129 would need
  ~14.9 GB/s. So it is memory-bound: compare in microseconds, not cycles (26.8 us a patch); a faster
  clock adds cycles, not speed. To confirm: the read-only kernel at 250 MHz, ~6 700 cycles a patch
  (removed 2026-10-29; it is in git before then).
  Repeated runs give 6 700-7 000 cycles a patch, different each time: what a memory-bound kernel does
  (DDR is shared with the ARM, refresh, page conflicts that depend on where the buffer lies); a
  logic-bound one gives the same count every run (sp_unet_small_opt: 11 503 twice, at two clocks).
- 2026-10-29: the read-only test kernel `justounetsimple_opt_readonly` removed (an experiment, not part of
  the flow). The integer z-score constants are now `src/optimized/zscore_int8.hpp`, shared with
  waveletcnn_opt and sp_unet_*_opt, written by `python tools/preprocessing.py` (was
  justounetsimple_prep.hpp); the same values, the kernels unchanged.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
- 2026-10-30: ZCU104 at 214 MHz: 5 882 cycles a patch, 27.45 us, 18.26 ms by the kernel's count
  (18.46 ms host timer, 19.20 ms wall clock). About the same microseconds as at 250 MHz (26.8 us):
  memory-bound, as found 2026-10-27. All ZCU104 runs side by side in `speed_zcu104.md`.
- 2026-10-30: ZedBoard (zb05) at 76.189 MHz, the PLL's nearest to the 77.777 built for: 143.8 ms,
  216.3 us a patch, ~16 480 cycles (model 15 360; 134 ms at this clock): the two read ports got
  1.14 GB/s of the 1.25 needed. Wall clock 3.6 s, mostly the SD card. In `speed_zedboard.md`.
- 2026-10-31: golden `justounetsimple` on the ZCU104: 194.4 s on the FPGA, 292.4 ms a patch (one patch a
  start), 202.3 s wall clock; 96.20 % against the labels, IoU 0.927 / 0.880 / 0.952, the same as the
  optimized kernel's. justounetsimple_opt is ~10 600× faster (18.26 ms). In `speed_zcu104.md`.
- 2026-10-31: DDR is not limited to ~9 GB/s: `spectralpointwise1d_opt` got 13.0 GB/s from the same four
  ports. So justounetsimple_opt's 5 882 cycles a patch (9.0 GB/s) is how it reads, likely its short
  bursts (a port reads a 32-pixel piece of a row, 7.7 KB, then jumps a row), not DDR's total. Not yet
  checked. In `speed_zcu104.md`.
- 2026-11-01: ZCU104 at 250 MHz, one start: 17.87 ms by the kernel's count (6 718 cycles, 26.87 us a
  patch), host timer 18.10 ms, wall clock 18.84 ms; 96.20 %, IoU 0.927 / 0.880 / 0.952, as at
  214 MHz. The 2026-10-27 run's 6 707 cycles again: still its reads (~9.1 GB/s). ~10 900× the golden
  kernel on the FPGA.
- 2026-11-02: routed utilization of every kernel in "Utilization" (from `tools/utilization.py`; by
  process in `utilization.md`). `justounetsimple_opt` 1 116 DSPs (conv1 864, as packed; ~1 112
  predicted), 43.1 k LUTs; the 7020's 216 DSPs, all conv1, 84.5 BRAM36 (the patch buffer 64). The golden
  kernel: 313 DSPs, 155.5 BRAM36 (its float maps).
- 2026-11-03: shared modules merged (the answers bit for bit the same): `conv3x3_serial` is now
  `conv3x3_grouped` with one output channel a cycle; the line-buffer and window shift is one helper
  (`window3x3.hpp`); `conv3x3_wide` / `pointwise_wide` and the board copies of `patch_buffer` gave way
  to each chip's `chip.hpp` (DSP packing, UltraRAM), passed to `conv3x3_packed_layer`,
  `pointwise_packed` and `patch_buffer`. The csynth DSP counts and II should not change: not yet built.
- 2026-11-03: modules renamed for what they are (the answers bit for bit the same):
  `conv3x3_grouped` -> `conv3x3_tiled`, `conv3x3_lines` -> `conv3x3_tiled_ram`, `conv3x3_packed_layer` ->
  `conv3x3_2mul`, `conv3x3_narrow` -> `conv3x3_all_in` (its `STEPS` now `OUT_PER_CYCLE`), `pointwise` ->
  `conv1x1`, `conv1d_engine` / `conv1d_stream` -> `conv1d_map` /
  `conv1d_positions`, `dense_scores` / `dense_stream` -> `dense_map` / `dense_positions`,
  `read_rows_ahead` -> `read_patch_rows`, `acc_t` / `wgt_t` -> `sum_t` / `weight_t`, and others;
  `RELU_SHIFT` is now `bool RELU, int ROUND_BITS`. `tools/utilization.py` maps the old names in older
  reports to the new ones.
- 2026-11-03: ZCU104 on trial with `read_patch_rows_simple` (`src/optimized/`): the same words in
  the same order, but from a plain `m_axi` pointer, one burst a patch row that HLS infers, no requests
  ahead; the readers' code reads like a plain loop. Each row now waits for DDR's first word, 8 rows a
  patch a port; estimate (not measured) +5-15 %, ~19-20.5 ms against 17.87. The 7030 and 7020 keep
  `read_patch_rows`. Not yet built: compare cycles a patch with the 2026-11-01 run (6 718 at 250 MHz).
- 2026-11-03: the ZCU104 measured 20 ms with `read_patch_rows_simple` (17.87 ms with the read-ahead):
  kept, for the simpler readers.
- 2026-11-03: ZCU104 on trial with ReLU, max pool and upsample as one process each (`relu_act`,
  `max_pool`, `upsample`; the last two new in `src/optimized/`) instead of `relu_pool`,
  `relu_pool_up` and `relu_up`; the chain reads like the model. The same answers bit for bit. Each
  process runs a pixel a cycle, far under T, so the cycles a patch should not change; the cost is 4
  more processes and FIFOs (16 deep, SRL), an estimated 150-500 LUTs each. Not yet built: compare
  cycles a patch and the csynth LUTs / FFs with the fused version.
- 2026-11-03: kernels moved to `src/models/`. The golden layers are one file, `src/golden/layers.hpp`
  (one template each for float and quantized weights); the golden kernel's ReLU is now done in its
  conv, one map buffer fewer a layer, the same answers. Golden kernel not rebuilt yet.
- 2026-11-03: `src/optimized/*/chip.hpp` removed: each optimized kernel sets its chip's DSP packing (and
  UltraRAM) at the top of its `.cpp`, under "Chip variables for implementation". Same values.
- 2026-11-03: csynth of the split version (ZCU104, 5 ns): every new process Final II = 1, at most
  1 410 cycles a patch (upsample 16x16x6; relu_act and max_pool ~1 025) against conv1's 4 142, so no
  process waits on them. Estimates: relu_act 1 494 / 2 238 / 1 464 LUTs, max_pool 1 383 / 1 916,
  upsample 1 321 / 1 103: 10.9 k LUTs, 3.9 k FFs for the seven, 80.9 k LUTs for the kernel (35 %),
  1 120 DSPs. Not yet compared like with like: the fused modules' 2.8 k LUTs are routed numbers;
  the routed report of a bitstream build gives the real cost.
- 2026-11-03: ZCU104: the ReLU back inside each conv, now as its `activation` option (`activation::relu`,
  in `fixed_point.hpp`, as the golden layers do), so the three `relu_act` processes and their FIFOs are
  gone; max pool and upsample stay one process each. `conv3x3_all_in` / `conv3x3_2mul` take the
  weights, bias, shifts and activation as template arguments; the FIFO depths are in the stream types,
  `hls::stream<T, 16>`. The constants (sizes, chip variables, DSP choices) are in each kernel's .hpp,
  those from the weights' fractions in the .cpp. The same answers bit for bit (all kernels).
  Not yet built: check csynth's FIFO lines ("Implementing FIFO 'conv1_act_U' ... using Shift
  Registers"), now without a BIND_STORAGE pragma, and that the cycles a patch stay.
- 2026-11-03: the split version (separate `relu_act`, `max_pool`, `upsample`; simple readers) measured on the
  ZCU104, built at the default 187.5 MHz, not 250: 21.42 ms by the kernel's count (6 039 cycles a patch),
  21.62 ms FPGA, 22.42 ms wall clock for 598 x 1092. Routed: the kernel 40.5 k LUTs, 48.0 k FFs, 1 116
  DSPs, 25 RAMB36, 16 URAM (43.1 k LUTs on 2026-11-02 with the read-ahead readers). The split parts:
  `relu_act` 433 / 464 / 416 LUTs, `max_pool` 657 / 910, `upsample` 461 / 505, their 16-deep FIFOs
  60-117 each (SRL) - about 1.8 k LUTs more than `relu_pool` / `relu_pool_up` / `relu_up` (2.8 k). Found
  in passing: every process has its own 16 x 16 multiplier for `patch_count` (height x width patches),
  263-345 LUTs each, ~4.5 k LUTs over the kernel; and the 2-deep FIFOs of the conv sums cost 270-548
  LUTs each (registers), more than the 16-deep SRL ones.
- 2026-11-04: the FIFO depths moved from `#pragma HLS STREAM` into the stream types, `hls::stream<T, D>`
  (the same depths); arrays of streams (`cube_words`) keep the pragma. The same answers bit for bit; not
  rebuilt: check csynth's FIFO depths.
- 2026-11-04: Zynq-7030 and 7020 split like the ZCU104: the ReLU in each conv (`conv3x3_2mul`,
  `conv3x3_tiled` with `activation::relu`), then `max_pool` / `upsample` one process each, instead of
  `relu_pool`, `relu_pool_up`, `relu_up` (deleted, with `relu_act` and `to_act`). The new conv-output
  FIFOs keep the default depth 2. The same answers bit for bit; not built: check csynth's LUTs and the
  cycles a patch against the fused version.
