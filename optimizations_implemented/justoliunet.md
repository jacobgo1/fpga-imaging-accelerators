# 1D-Justo-LiuNet (`justoliunet`)

Per-pixel classifier: each pixel's spectrum alone, convolved along the bands. The BatchNorm
variant (`justoliunet_bn.md`) uses the same hardware.

## Model and flow

```
raw cube (H x W x 120 uint16)
 └ keep bands 8-117, z-score                     110 bands
 └ conv1 1→6,  k=6, valid → ReLU → maxpool 2     105 → 52
 └ conv2 6→12  ...                               47 → 23
 └ conv3 12→18 ...                               18 → 9
 └ conv4 18→24 ...                               4 → 2
 └ flatten (channel-major) 48 → dense → 3 scores, class = argmax
```

Work per pixel, counting only outputs the pools keep (conv1 uses 102 of 105, conv2 46 of 47):

| | conv1 | conv2 | conv3 | conv4 | dense | total |
|---|---|---|---|---|---|---|
| positions P | 102 | 46 | 18 | 4 | 1 | |
| weights W | 36 | 432 | 1 296 | 2 592 | 144 | 4 500 |
| MACs P × W | 3 672 | 19 872 | 23 328 | 10 368 | 144 | **57 384** |

## Kernels

| Kernel | Board | Clock | Status |
|---|---|---|---|
| `justoliunet` | ZCU104 (any) | 5 ns | golden baseline: float, plain loops, `n` pixels per start |
| `justoliunet_opt` | ZCU104 | 187.5 MHz (PS PLL nearest 200) | **measured 62.9 ms** per 598 × 1092 capture |
| `justoliunet_opt_zynq7020` | ZedBoard (`--board zedboard`) | 78.1 MHz | built in C++, bit-exact; cycle model only |
| `justoliunet_opt_zynq7030` | Zynq-7030 (`--board zynq7030`) | 100 MHz | built in C++, bit-exact; cycle model only |

All optimized kernels give the same answers bit for bit (`classify_fixed` in
`tools/justoliunet_model.py`). Accuracy on the labelled aegean pixels: fixed point 96.73 %,
fp32 96.61 %; same class as fp32 on 99.66 %.

## Optimizations used

| Technique | ZCU104 | 7030 | ZedBoard (7020) |
|---|---|---|---|
| Pixel as the tile: no patches, halos or padding; the image is one run of pixels | ✓ | ✓ | ✓ |
| Preprocessing in hardware: band drop + integer z-score (12-bit, 8 fraction bits) | ✓ | ✓ | ✓ |
| Integer fixed point: int8 weights, uint8 activations after ReLU, int32 sums, shifts for scales | ✓ | ✓ | ✓ |
| Receptive-field pruning: unused conv outputs never computed | ✓ | ✓ | ✓ |
| DATAFLOW: every layer a process, intermediate maps never leave the chip | ✓ | ✓ | ✓ |
| Stages balanced to the same cycles per pixel | ✓ (≤ 18) | ✓ (≤ 74) | ✓ (≤ 108) |
| Two multiplies per DSP (packed weights, shared uint8 activation) | conv2, conv3 (DSP48E2, groups of 4) | conv2, conv3 (DSP48E1, groups of 1) | conv2, conv3 (DSP48E1, groups of 1) |
| DSPs on the heaviest layers, LUT multipliers on the light ones | ✓ | ✓ | ✓ |
| Loop flattening (pixels × steps in one II=1 loop: no pipeline drain per pixel) | ✓ | ✓ | ✓ |
| Shift-register windows, every index a constant | ✓ | ✓ | ✓ |
| Weights loaded once into registers/ROM (no weight traffic) | ✓ | ✓ | ✓ |
| Burst reads requested ahead (`hls::burst_maxi`) | 2 ports, 16 pixels/burst | 1 port, 8 pixels/burst | 1 port, 8 pixels/burst |
| Whole feature map per pixel, several positions per cycle (`conv1d_engine`) | every layer | conv1 (2 positions) | – |
| Spectral streaming: one position at a time, 6-position line buffer (`conv1d_stream`) | – | conv2-4 | conv2-4 |
| Folding over output/input channels with an accumulator | – | ✓ | ✓ |
| Next pixel's first positions read during the last output's idle steps | – | ✓ | ✓ |
| FIFO depths from the cycle model | – | depth 8 (93 → 74 cycles) | depth 8 (134 → 113 cycles) |
| Cycle counter register (`count_cycles`) | ✓ | ✓ | ✓ |

## Latency per stage

One stage processes one pixel at a time; with II = 1 a loop iteration is one cycle, so
cycles per pixel = iterations per pixel. The slowest stage sets the pace, T.

- ZCU104 (`conv1d_engine`): S = ceil(P / POS) × (OC / OCP).
- 7030 and ZedBoard (`conv1d_stream`): S = w + P × (OC / OCP) × (IC / ICP), where w
  is what is left of the 5 window-fill steps after the read-ahead: 5 - min(5, G × C - 1),
  G = OC / OCP, C = IC / ICP (5 for the first pixel). A streamed layer takes in at most
  one position a cycle, so S ≥ its input positions: conv1's 102 outputs are why the
  7020 cannot go below ~102, and why the 7030's conv1 does 2 positions a cycle.

ZCU104 (per cycle, S, multipliers):

| Stage | per cycle | S | DSP / LUT |
|---|---|---|---|
| read | 2 ports × 1 word | ~7.5 | – |
| z-score | 1 word (8 bands) | 15 | 8 DSP |
| conv1 | 6 pos × 6 ch × 6 | 17 | 216 LUT mult |
| conv2 | 3 pos × 12 ch × 36 | 16 | 648 DSP |
| conv3 | 1 pos × 18 ch × 72 | **18** | 648 DSP |
| conv4 | 1 pos × 6 ch × 108 | 16 | 648 LUT mult |
| dense | 3 features × 3 | 16 | 9 LUT mult |
| **Total** | | **T = 18** | ~1 304 DSP |

Zynq-7000s (packed DSPs: OCP / 2 × ICP × 6):

| Stage | 7030: per cycle | S | DSP / LUT | ZedBoard: per cycle | S | DSP / LUT |
|---|---|---|---|---|---|---|
| read | 1 port × 1 word | 30 | – | 1 port × 1 word | 30 | – |
| z-score | 1 word (4 bands) | 30 | 4 LUT mult | 1 word (4 bands) | 30 | 4 LUT mult |
| conv1 | 2 pos × 6 ch × 6 | 51 | 72 LUT mult | 1 pos × 6 ch × 6 | 102 | 36 LUT mult |
| conv2 | 12 out × 6 in × 6 (G×C = 1) | 5 + 46 = 51 | 216 DSP | 6 out × 6 in × 6 (2) | 4 + 92 = 96 | 108 DSP |
| conv3 | 18 out × 3 in × 6 (4) | 2 + 72 = **74** | 162 DSP | 6 out × 6 in × 6 (6) | 0 + 108 = **108** | 108 DSP |
| conv4 | 4 out × 6 in × 6 (18) | 0 + 72 = 72 | 144 LUT mult | 2 out × 9 in × 6 (24) | 0 + 96 = 96 | 108 LUT mult |
| dense | 1 × 3 | 48 | 3 LUT mult | 1 × 3 | 48 | 3 LUT mult |
| **Total** | | **T = 74** | 378 of 400 DSP | | **T = 108** | 216 of 220 DSP |

Utilization of a layer's multipliers: u = MACs / (T × multipliers).
ZCU104: conv3 100 %, conv1 94 %, conv4 89 %, conv2 85 %. 7030: conv3 97 %, conv4 97 %,
conv1 69 %, conv2 62 % (conv2 has no step count between 51 and 96: all channels at once, or half). DSP and LUT totals are estimates
until a utilization report is recorded here.

## Inference time

time = (pixels × T + fill) / clock, fill a few hundred cycles. For a 598 × 1092 capture
(653 016 pixels):

| Board | T | clock | time |
|---|---|---|---|
| ZCU104 | 18 | 187.5 MHz | 62.7 ms predicted, **62.9 ms measured** (18.0 cycles/pixel) |
| Zynq-7030 | 74 | 100 MHz | **0.48 s** (cycle model) |
| ZedBoard | 108 | 78.1 MHz | **0.90 s** (cycle model) |

Measured cycles per pixel = FPGA time × clock ÷ pixels (or the `cycles` register ÷ pixels).

## Boards: limits, differences and why

| | ZCU104 | Zynq-7030 | ZedBoard (Zynq-7020) |
|---|---|---|---|
| Limit | DSPs (compute-bound: 224 MACs per byte read) | DSPs (378 of 400) | DSPs and LUTs |
| DSPs / LUTs | 1 728 DSP48E2 / 230 k | 400 DSP48E1 / 79 k | 220 DSP48E1 / 53 k |
| Packing | shift 18: 4 uint8 products summed before splitting | shift 16: each product split on its own (more LUT adders) | as the 7030 |
| HP port | 128 bit: 15 words a pixel | 64 bit: 30 words a pixel | 64 bit: 30 words a pixel |
| Why the engines differ | room for wide layers and whole maps in registers | folded and streamed as the 7020, but wider; conv1 whole-map at 2 positions (its 112 bands are small) to get under 102 | too few DSPs for wide layers (fold instead), too few LUTs for whole-map FIFOs (stream positions instead) |
| Memory a limit? | only below ~9.4 cycles/pixel (2 ports) | no (one port: 37.5 cycles/pixel) | no (one port: 37.5 cycles/pixel) |
| Clock | 187.5 MHz: PS PLL's nearest to 5 ns | 10 ns | 12.8 ns (try 10 ns) |

## Changes

- 2026-10-14: golden kernel and 1D golden modules.
- 2026-10-16: `justoliunet_opt` (ZCU104).
- 2026-10-19: `justoliunet_opt_zynq7020` (ZedBoard); measured ZCU104: 62.9 ms at 187.5 MHz.
- 2026-10-19: ZedBoard read-ahead of the next pixel's window: 113 → 108 cycles/pixel.
- 2026-10-19: `justoliunet_opt_zynq7030` (the ZedBoard design, wider): conv1 at 2 positions a
  cycle (`pool_adjacent`), conv2 12×6, conv3 18×3, conv4 4×6: 74 cycles/pixel, 0.48 s (paper
  design: 81, 0.53 s).
- Open ideas (not implemented): faster clock; ZCU104 rebalance to 16 cycles; hard-wired
  weights (ZCU104); BRAM multipliers (Zynq-7000); fast FIR; see `docs/justoliunet_opt_design.md`.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
