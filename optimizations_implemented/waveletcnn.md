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
| `waveletcnn` | ZCU104 | 5 ns | golden baseline: float, one patch per start |
| `waveletcnn_opt` | ZCU104 | 5 ns (PS PLL: 187.5 MHz) | built in C++, bit-exact; cycle model below |
| `waveletcnn_opt_zynq7030` | Zynq-7030 (`--board zynq7030`) | 10 ns | built in C++, bit-exact; cycle model below |
| `waveletcnn_opt_zynq7020` | PYNQ-Z2 / ZedBoard (`--board zynq7020` / `zedboard`) | 12.8 ns | built in C++, bit-exact; cycle model below |

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
| Read ports, each every n-th patch row (`read_rows_ahead`, `burst_maxi`) | 4 × 128 bit | 4 × 64 bit (writer shares HP3) | 2 × 64 bit |
| `conv_block` (12 → 12, 3×3): line buffers + 3×3 shift-register window | `conv3x3_narrow` (`STEPS = 6`, 2 out/cycle) | `conv3x3_narrow` (`STEPS = 12`, 1 out/cycle) | `conv3x3_serial` (`P2 = 6` in × 1 out/cycle) |
| Fused `LeakyReLU` + 1×1 `seg_head` (12 → 3) + `LeakyReLU` (`leaky_head`, 1 cycle/pixel) | ✓ | ✓ | ✓ |
| Score writes with acknowledgements collected later (`store_scores`) | ✓ | ✓ | ✓ |
| Cycle counter register | ✓ | – | – |

## Latency per stage (cycles per patch)

Each stage runs at II = 1:
- `read_rows_ahead`: `(32 / PORTS)` rows × 32 pixels × words per pixel.
- `pixel_buffer` and `dwt_proj`: `ceil(58 / DWT_STEPS_PER_CYCLE)` steps/pixel × 1 024 pixels.
- `conv_block`: `STEPS` cycles/pixel × (1 024 + 33) pixels.
- `leaky_head` and `store_scores`: 1 024 cycles/patch.

| Stage | ZCU104 (`DWT_STEPS_PER_CYCLE = 8`) | Zynq-7030 (`DWT_STEPS_PER_CYCLE = 4`) | Zynq-7020 (`DWT_STEPS_PER_CYCLE = 2`) |
|---|---|---|---|
| read (per port) | 3 840 | 7 680 | 15 360 |
| `pixel_buffer` | **8 192** | **15 360** | **29 696** |
| `dwt_proj` | **8 192** | **15 360** | **29 696** |
| `conv_block` (3×3, 12→12) | 6 342 (`STEPS = 6`) | 12 684 (`STEPS = 12`) | 25 368 (`P2 = 6`) |
| `leaky_head` (1×1, 12→3) | 1 024 | 1 024 | 1 024 |
| `store_scores` | 1 024 | 1 024 | 1 024 |
| **T** | **8 192** (`dwt_proj`) | **15 360** (`dwt_proj`) | **29 696** (`dwt_proj`) |
| DSPs | 668 of 1 728 (z-score 32, proj 384, conv 216, head 36) | 336 of 400 (proj 192, conv 108, head 36) | 186 of 220 (proj 96, conv 54, head 36) |
| Patch/strip buffer BRAM | 8 BRAM18 (4 BRAM36) | 8 BRAM18 (4 BRAM36) | 2 BRAM18 (1 BRAM36) |

## Inference time

`time = patches × T / clock`; a 598 × 1092 capture is 19 × 35 = 665 patches.

| Board | T | clock | time |
|---|---|---|---|
| ZCU104 | 8 192 | 187.5 MHz | **29.1 ms** (27.2 ms at 200 MHz) |
| Zynq-7030 | 15 360 | 100 MHz | **102 ms** |
| Zynq-7020 | 29 696 | 78.1 MHz | **253 ms** |

## Boards: limits, differences and why

| | ZCU104 | Zynq-7030 | Zynq-7020 |
|---|---|---|---|
| Limit | `dwt_proj` width (`DWT_STEPS_PER_CYCLE = 8`) | DSPs (336 of 400) | DSPs (186 of 220) |
| DSPs / LUTs | 1 728 / 230 k | 400 / 79 k | 220 / 53 k |
| `dwt_proj` width | 8 subband steps/cycle (16 bands, 384 DSPs) | 4 subband steps/cycle (8 bands, 192 DSPs) | 2 subband steps/cycle (4 bands, 96 DSPs) |
| `conv_block` (12→12) | 2 out ch/cycle (216 DSPs) | 1 out ch/cycle (108 DSPs) | 6 in × 1 out ch/cycle (54 DSPs) |
| `leaky_head` (12→3) | 36 DSPs | 36 DSPs | 36 DSPs |
| Strip buffer | 4-row ping-pong in BRAM | 4-row ping-pong in BRAM | 2-row ping-pong in BRAM |
| Ports | 4 read + 1 write, 128 bit | 4 read + 1 write sharing HP3, 64 bit | 2 read + 1 write, 64 bit |

## Changes

- 2026-10-21: quantized weights (`waveletcnn_int8`); golden kernel (`waveletcnn`);
  optimized kernels (`waveletcnn_opt`, `waveletcnn_opt_zynq7030`, `waveletcnn_opt_zynq7020`)
  with strip ping-pong `pixel_buffer`, fused streaming `dwt_proj`, `conv3x3_narrow`/`conv3x3_serial`
  and `leaky_head`.
- 2026-10-29: the z-score constants come from `src/optimized/shared/zscore_int8.hpp`, shared with the
  other 2D kernels (`python tools/preprocessing.py`; was waveletcnn_prep.hpp, the same values);
  `tools/waveletcnn_image.py` is gone: `tools/image.py prepare` makes the image folder.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
