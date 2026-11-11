# WaveletCNN (`waveletcnn`)

Hybrid spectral-wavelet + spatial patch classifier (Nikolai Andresen,
`hypso-onboard-segmentation/model_definition/WaveletCNN.py`): a 1-level 1D
Daubechies-4 (`db4`) discrete wavelet transform along the 110 spectral bands of
each pixel, concatenated onto the input bands, compressed to 12 channels by a
1×1 projection, refined spatially by one 3×3 convolution block, and scored into
3 classes by a 1×1 head.

## Model and flow

```
raw cube (H x W x 120 uint16) → patches of 32 x 32 (edge rows/columns repeated)
 └ keep bands 8-117, z-score                                32 x 32 x 110
 └ dwt_spectral: 1D db4 DWT (pad 6 zeros/side, stride 2)    → 58 low + 58 high per pixel
 └ concat [x (110), low (58), high (58)]                    → 32 x 32 x 226
 └ projection: conv 1x1 226→12                              → 32 x 32 x 12
 └ conv_block: conv 3x3 12→12 + BN + LeakyReLU(0.01)        → 32 x 32 x 12
 └ seg_head:   conv 1x1 12→3  + BN + LeakyReLU(0.01)        → 32 x 32 x 3 scores, class = argmax
```

Work per 32 × 32 patch:
- `dwt_spectral`: 58 × 8 × 2 = 928 MACs/pixel (0.95 M/patch, 8-tap constant filter).
- `projection` (226 → 12, 1×1): 2 712 MACs/pixel (2.78 M/patch).
- `conv_block` (12 → 12, 3×3): 1 296 MACs/pixel (1.33 M/patch).
- `seg_head` (12 → 3, 1×1): 36 MACs/pixel (0.037 M/patch).
Total: ~4 972 MACs per pixel (5.09 M per patch), ~21 MACs per byte read.

## Kernels

| Kernel | Board | Clock | Status |
|---|---|---|---|
| `waveletcnn` | ZCU104 | 5 ns | golden baseline: float, one patch per start; **measured 128.67 s** for a whole capture (193.5 ms a patch), 97.23 % |
| `waveletcnn_opt` | ZCU104 | 250 MHz | bit-exact; **measured 21.91 ms**, 8 238 cycles a patch (T 8 192), 97.22 %; 25.43 ms (8 195 cycles) at 214 MHz |
| `waveletcnn_opt_zynq7030` | Zynq-7030 (board `picobob`) | 10 ns | built in C++, bit-exact; cycle model below |
| `waveletcnn_opt_zynq7020` | PYNQ-Z2 / ZedBoard (board `pynqz2` / `zedboard`) | 12.8 ns (PLL: 76.189 MHz) | bit-exact; **measured 259.4 ms** on the ZedBoard, 29 721 cycles a patch (T 29 696), 97.22 % |

All optimized kernels give the same answers bit for bit (`classify_fixed` in
`tools/waveletcnn_model.py`); same class as the quantized float model on 99.66 %
of the Aegean capture's pixels (95.99 % accuracy against the labels vs 95.89 %
for the quantized float model).

## Optimizations used

| Technique | ZCU104 | 7030 | 7020 |
|---|---|---|---|
| Loop tiling into 32 × 32 patches, edges repeated (no host padding) | ✓ | ✓ | ✓ |
| Preprocessing in hardware: band drop + integer z-score (int8, 5 fraction bits) | ✓ | ✓ | ✓ |
| Fixed point: int8 weights, exact 12-frac-bit DWT products, int16 activations (`ACT_FRAC = 9`), int32 sums (`OUT_FRAC = 15`) | ✓ | ✓ | ✓ |
| Shift-and-add `LeakyReLU(0.01)` (`41 * 2^-12`, `41 = 32 + 8 + 1`, 0 DSPs) | ✓ | ✓ | ✓ |
| DATAFLOW, layers fused; feature maps never leave the chip | ✓ | ✓ | ✓ |
| Strip ping-pong buffer (`pixel_buffer`): only `PORTS` rows × 32 pixels in BRAM (8×–16× smaller than a full patch buffer) | 4-row BRAM | 4-row BRAM | 2-row BRAM |
| Fused 1D DWT + 1×1 projection (`dwt_proj`): 8-band sliding window accumulates `x[2m]`, `x[2m+1]`, `low[m]`, `high[m]` directly into the 12 projection channels; the 226-channel tensor is never stored | `DWT_STEPS_PER_CYCLE = 8` | `DWT_STEPS_PER_CYCLE = 4` | `DWT_STEPS_PER_CYCLE = 2` |
| Read ports, each every n-th patch row (`read_patch_rows_simple`) | 4 × 128 bit | 4 × 64 bit (writer shares HP3) | 2 × 64 bit |
| `conv_block` (12 → 12, 3×3) and its `LeakyReLU` (`activation::leaky`): line buffers + 3×3 shift-register window (`conv3x3_tiled`) | 12 in × 2 out/cycle | 12 in × 1 out/cycle | `P2 = 6` in × 1 out/cycle |
| 1×1 `seg_head` (12 → 3) and its `LeakyReLU` (`conv1x1`, `activation::leaky_scores`, 1 cycle/pixel) | ✓ | ✓ | ✓ |
| Score writes with acknowledgements collected later (`store_scores`) | ✓ | ✓ | ✓ |
| Cycle counter register | ✓ | – | – |

## Latency per stage (cycles per patch)

Each stage runs at II = 1:
- `read_patch_rows_simple`: `(32 / PORTS)` rows × 32 pixels × words per pixel.
- `pixel_buffer` and `dwt_proj`: `ceil(58 / DWT_STEPS_PER_CYCLE)` steps/pixel × 1 024 pixels.
- `conv_block`: `STEPS` cycles/pixel × (1 024 + 33) pixels.
- `seg_head` (`conv1x1`) and `store_scores`: 1 024 cycles/patch.

| Stage | ZCU104 (`DWT_STEPS_PER_CYCLE = 8`) | Zynq-7030 (`DWT_STEPS_PER_CYCLE = 4`) | Zynq-7020 (`DWT_STEPS_PER_CYCLE = 2`) |
|---|---|---|---|
| read (per port) | 3 840 | 7 680 | 15 360 |
| `pixel_buffer` | **8 192** | **15 360** | **29 696** |
| `dwt_proj` | **8 192** | **15 360** | **29 696** |
| `conv_block` (3×3, 12→12) | 6 342 (2 out/cycle) | 12 684 (1 out/cycle) | 25 368 (`P2 = 6`) |
| `seg_head` (`conv1x1`, 12→3) | 1 024 | 1 024 | 1 024 |
| `store_scores` | 1 024 | 1 024 | 1 024 |
| **T** | **8 192** (`dwt_proj`) | **15 360** (`dwt_proj`) | **29 696** (`dwt_proj`) |
| DSPs | 668 of 1 728 (z-score 32, proj 384, conv 216, head 36) | 336 of 400 (proj 192, conv 108, head 36) | 186 of 220 (proj 96, conv 54, head 36) |
| Patch/strip buffer BRAM | 8 BRAM18 (4 BRAM36) | 8 BRAM18 (4 BRAM36) | 2 BRAM18 (1 BRAM36) |

## Inference time

`time = patches × T / clock`; a 598 × 1092 capture is 19 × 35 = 665 patches.

| Board | T | clock | time | measured |
|---|---|---|---|---|
| ZCU104 | 8 192 | 187.5 MHz | **29.1 ms** (25.5 ms at 214 MHz) | **25.43 ms** at 214 MHz: 8 195 cycles a patch; **21.91 ms** at 250 MHz: 8 238 |
| Zynq-7030 | 15 360 | 100 MHz | **102 ms** | – |
| Zynq-7020 | 29 696 | 78.1 MHz | **253 ms** (259.2 ms at 76.189 MHz) | **259.4 ms** at 76.189 MHz: 29 721 cycles a patch |

## Boards: limits, differences and why

| | ZCU104 | Zynq-7030 | Zynq-7020 |
|---|---|---|---|
| Limit | `dwt_proj` width (`DWT_STEPS_PER_CYCLE = 8`) | DSPs (336 of 400) | DSPs (186 of 220) |
| DSPs / LUTs | 1 728 / 230 k | 400 / 79 k | 220 / 53 k |
| `dwt_proj` width | 8 subband steps/cycle (16 bands, 384 DSPs) | 4 subband steps/cycle (8 bands, 192 DSPs) | 2 subband steps/cycle (4 bands, 96 DSPs) |
| `conv_block` (12→12) | 2 out ch/cycle (216 DSPs) | 1 out ch/cycle (108 DSPs) | 6 in × 1 out ch/cycle (54 DSPs) |
| `seg_head` (`conv1x1`, 12→3) | 36 DSPs | 36 DSPs | 36 DSPs |
| Strip buffer | 4-row ping-pong in BRAM | 4-row ping-pong in BRAM | 2-row ping-pong in BRAM |
| Ports | 4 read + 1 write, 128 bit | 4 read + 1 write sharing HP3, 64 bit | 2 read + 1 write, 64 bit |

## Utilization

<!-- utilization.py: waveletcnn -->
Routed, kernel alone (without the interconnects); by process in
[utilization.md](utilization.md). Written by `python tools/utilization.py`.

| Kernel | Board | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|---|
| (no report yet) | | | | | | |
<!-- /utilization.py -->

## Changes

- 2026-10-21: quantized weights (`waveletcnn_int8`); golden kernel (`waveletcnn`);
  optimized kernels (`waveletcnn_opt`, `waveletcnn_opt_zynq7030`, `waveletcnn_opt_zynq7020`)
  with strip ping-pong `pixel_buffer`, fused streaming `dwt_proj`, `conv3x3_narrow`/`conv3x3_serial`
  and `leaky_head`.
- 2026-10-29: the z-score constants come from `src/optimized/zscore_int8.hpp`, shared with the
  other 2D kernels (`python tools/preprocessing.py`; was waveletcnn_prep.hpp, the same values);
  `tools/waveletcnn_image.py` is gone: `tools/image.py prepare` makes the image folder.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
- 2026-10-30: `waveletcnn_opt` on the ZCU104 at 214 MHz: 8 195 cycles a patch (T 8 192), 25.43 ms by
  its own count, wall clock 26.5 ms; 97.22 % against the labels, IoU cloud / land / sea 0.933 / 0.911 /
  0.975. In `speed_zcu104.md`.
- 2026-10-31: golden `waveletcnn` on the ZCU104: 128.67 s on the FPGA (193.5 ms a patch), 136.55 s wall
  clock; 97.23 % against the labels, IoU 0.933 / 0.911 / 0.975 (`waveletcnn_opt`: 97.22 %, ~5 100× faster
  at 25.43 ms). In `speed_zcu104.md`.
- 2026-10-31: `waveletcnn_opt_zynq7020` on the ZedBoard (zb05) at 76.189 MHz: 29 721 cycles a patch (T 29 696),
  259.4 ms, wall clock 2.974 s; 97.22 % against the labels, IoU 0.933 / 0.911 / 0.975, as the ZCU104
  kernel. In `speed_zedboard.md`.
- 2026-11-01: `waveletcnn_opt` on the ZCU104 at 250 MHz, one start: 21.91 ms by the kernel's count
  (8 238 cycles a patch, 32.95 us), host timer 22.15 ms, wall clock 23.04 ms; 97.22 %, IoU 0.933 /
  0.911 / 0.975, as at 214 MHz. 0.5 % over the predicted 21.8 ms (43 cycles more a patch than at
  214 MHz). ~5 900× the golden kernel. In `speed_zcu104.md`.
- 2026-11-02: routed utilization of every kernel in "Utilization" (from `tools/utilization.py`; by
  process in `utilization.md`). `waveletcnn_opt` 661 DSPs against the 668 predicted (z-score 32, proj
  352 of 384, conv 227 of 216, head 50 of 36: the Bind Op Report would say which moved);
  `waveletcnn_opt_zynq7020` 196 against 186 (proj 92, conv 54, head 50).
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
- 2026-11-03: kernels moved to `src/models/`. The golden layers are one file, `src/golden/layers.hpp`
  (one template each for float and quantized weights); the golden kernel's ReLU is now done in its
  conv, one map buffer fewer a layer, the same answers. Golden kernel not rebuilt yet.
- 2026-11-03: `conv3x3_all_in` has an `activation` option (here `none`: its LeakyReLU stays in
  `leaky_conv1x1`); the zscore include and constants moved to each kernel's .hpp, those from the
  weights' fractions stay in the .cpp. The same answers bit for bit; not rebuilt.
- 2026-11-04: the FIFO depths moved from `#pragma HLS STREAM` into the stream types, `hls::stream<T, D>`
  (the same depths); arrays of streams (`cube_words`) keep the pragma. The same answers bit for bit; not
  rebuilt: check csynth's FIFO depths.
- 2026-11-04: fewer conv modules, the same answers bit for bit. `conv3x3_all_in` is gone: the ZCU104 and
  7030 conv_block is `conv3x3_tiled` with all 12 input channels a cycle (its weights a LUTRAM word a
  step, not registers; rounding for 2 / 1 channels, not 12). Its LeakyReLU moved into the conv
  (`activation::leaky`), so `leaky_conv1x1` is gone too: the head is `conv1x1` with
  `activation::leaky_scores`, and the FIFO between them is 16 bits a channel instead of 32. Not built:
  check csynth's T, DSPs (216 / 108 / 54 conv, 36 head) and LUTs.
- 2026-11-04: `conv3x3_tiled_patch` is a perfect loop nest (pixels -> output groups -> input groups,
  pipelined at the bottom, HLS flattening it to one pipeline a patch) instead of one flat loop with a
  hand-written counter chain; the window centre's y and x come from p - W - 1 by shifts. Its unused
  `ROWS_PER_STEP` (always 3) is gone. The same answers bit for bit; **not built yet**: check csynth that
  the nest flattened at II = 1 and conv_block stays 6 342 / 12 684 / 25 368 cycles a patch (6, 12, 24
  cycles a pixel).
- 2026-11-04: every kernel reads with `read_patch_rows_simple` (a plain `m_axi` pointer, one burst a
  patch row that HLS infers), as `justounetsimple_opt` since 2026-11-03; `read_patch_rows` (the
  read-ahead `burst_maxi` reader) is kept but unused. Each row now waits for DDR's first word: on
  `justounetsimple_opt` that cost 17.87 → 20 ms. Not built: check hls.log for the inferred read
  bursts (no "could not be inferred") and compare the cycles a patch with the last run.
- 2026-11-06: `waveletcnn_opt` with `read_patch_rows_simple` at 250 MHz on five other images (`benchmark.ipynb`, speed_zcu104.md "Five images"):
  8 219 cycles a patch (8 213-8 227), 21.86 ms, against 8 238 before: no cost. 93.72 % against
  the labels, as classify_fixed on the images checked.
