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
| `justounetsimple` | ZCU104 | 5 ns | golden baseline: float, one patch per start |
| `justounetsimple_opt` | ZCU104 | 5 ns (PS PLL: 187.5 MHz) | 24 ms measured at "200 MHz" before the 2026-10-14 fixes; newer runs not recorded here yet |
| `justounetsimple_opt_zynq7030` | Zynq-7030 (`--board zynq7030`) | 10 ns | built in C++, bit-exact; cycle estimates only (the 7020 design, wider) |
| `justounetsimple_opt_zynq7020` | PYNQ-Z2 / ZedBoard (`--board zynq7020` / `zedboard`) | 12.8 ns | built in C++; HLS estimates only |
| `justounetsimple_opt_readonly` | ZCU104 | 5 ns | test kernel: the readers alone (memory speed) |

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
| Read bursts requested a row ahead (`read_rows_ahead`, `burst_maxi`) | ✓ | – | – |
| Line buffers + 3×3 shift-register window (`conv3x3_stream`) | ✓ | ✓ | ✓ |
| conv1: P1 bands a cycle, band chunks accumulated | 32 | 12 | 8 |
| Two multiplies per DSP in conv1 | ✓ (DSP48E2, groups of 8) | ✓ (DSP48E1, groups of 2) | ✓ (DSP48E1, groups of 2) |
| conv2-4: one output channel a cycle (narrow) / one input x output a cycle (serial) | narrow, DSP | serial (2 in), DSP | serial (2 in), LUTs |
| ReLU fused with pooling / upsampling | ✓ | ✓ | ✓ |
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
| pool, pool+up, up, store | ≤ 1 280 | small | small |
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
| ZCU104 | 4 129 | 187.5 MHz | **14.7 ms** (13.7 at 200 MHz); DDR latency per row can add a few % |
| Zynq-7030 | 10 273 | 100 MHz | **68 ms**, if DDR gives the four ports 2.4 GB/s together (check on HYPSO-2) |
| Zynq-7020 | 15 360 | 78.1 MHz | **131 ms** |

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
  240 words per row, 2 in flight per port); shared `patch_buffer_run` in `src/optimized/shared/patch_buffer.hpp`.
- 2026-10-27: ZCU104 measured 6 707 cycles a patch, 17.8 ms at 250 MHz (conv1: 4 129). Not the FIFOs
  between the stages: a cycle model with HLS's pipeline stalls (the one that matched
  sp_unet_small_opt_direct within 0.3 %) gives 4 140 with ideal memory, as built, because every
  stage after conv1 has time to spare. With DDR in the model, the kernel takes exactly as long as the
  read-only kernel (`patch_buffer` hides nothing and loses nothing; deeper `s_raw` changes nothing):
  6 707 cycles at 250 MHz is DDR giving the four ports ~9.2 GB/s together. conv1's 4 129 would need
  ~14.9 GB/s. So it is memory-bound: compare in microseconds, not cycles (26.8 us a patch); a faster
  clock adds cycles, not speed. To confirm: `justounetsimple_opt_readonly` at 250 MHz, ~6 700 cycles a patch.
  Repeated runs give 6 700-7 000 cycles a patch, different each time: what a memory-bound kernel does
  (DDR is shared with the ARM, refresh, page conflicts that depend on where the buffer lies); a
  logic-bound one gives the same count every run (sp_unet_small_opt_direct: 11 503 twice, at two clocks).
