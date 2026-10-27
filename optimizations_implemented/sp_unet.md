# SpecPW-UNet (`sp_unet_small`, `sp_unet_medium`, `sp_unet_large`)

Patch classifier: a spectral stem of 1x1 convs mixes each pixel's 110 bands on its own,
then a two-level U-Net with one skip connection looks at the neighbours. A score per
pixel, over 32 × 32 patches of the capture. Three widths of the same network.

## Model and flow

```
raw cube (H x W x 120 uint16) → patches of 32 x 32 (edge rows/columns repeated)
 └ keep bands 8-117, z-score                                32 x 32 x 110
 └ stem: 1x1 conv → ReLU, three times (BN folded into the first)  → 32 x 32 x S3
 └ double1: (3x3 conv → ReLU) x 2                          → 32 x 32 x C1  = skip
 └ maxpool 2 → double2: (3x3 conv → ReLU) x 2 → maxpool 2  → 8 x 8 x C2
 └ up1: nearest x2 → 3x3 conv → ReLU                       → 16 x 16 x C2
 └ double3                                                 → 16 x 16 x C3
 └ up2: nearest x2 → 3x3 conv → ReLU                       → 32 x 32 x C3
 └ cat([up2, skip]) → double4                              → 32 x 32 x C4
 └ final: 1x1 conv                                         → 32 x 32 x 3 scores, class = argmax
```

Every BatchNorm is folded into the conv before it (`tools/quantize_weights.py`). The
UpBlock's conv is 3x3 in the checkpoints, though the architecture figure draws a 1x1. The
skip goes second in the concat (the other order: 27-40 % on the aegean pixels).

| Size | Stem S1-S2-S3 | C1 / C2 / C3 / C4 | Parameters | MACs per patch | aegean: fp32 / int8 weights / fixed point |
|---|---|---|---|---|---|
| small | 24-12-12 | 12 / 24 / 12 / 8 | 26 k | 13.8 M | 94.03 / 94.02 / 93.94 % |
| medium | 24-16-12 | 24 / 48 / 24 / 12 | 90 k | 40.5 M | 95.20 / 95.11 / 95.03 % |
| large | 32-16-16 | 32 / 64 / 32 / 16 | 159 k | 70.5 M | 95.76 / 95.78 / 95.80 % |

The stem is a small part of the work (3-4 M MACs per patch); the 3x3 convs at full
resolution (double1, up2, double4) are most of it. justounetsimple, by comparison, is 6.6 M
MACs per patch, 92 % of it its first conv.

## Kernels

| Kernel | Board | Clock | Status |
|---|---|---|---|
| `sp_unet_small` | ZCU104 | 5 ns | golden baseline: float, int8 weights, n patches per start; C++ and testbench pass, not yet synthesized |
| `sp_unet_medium` | ZCU104 | 5 ns | the same |
| `sp_unet_large` | ZCU104 | 5 ns | the same |
| `sp_unet_small_opt` | ZCU104 | 5 ns (PS PLL: 187.5 MHz) | whole raw capture per start, bit-exact in C++ (testbench, `classify_fixed` on aegean pixels); cycle model below, not yet synthesized |
| `sp_unet_medium_opt` | ZCU104 | 5 ns | the same |
| `sp_unet_large_opt` | ZCU104 | 5 ns | the same |
| `sp_unet_small_opt_direct` | ZCU104 | 214 MHz | the same kernel without Winograd (direct convs, simpler); **measured 30.6 ms** for a whole capture, FPGA alone: 9 856 cycles a patch (model 9 839) |
| `sp_unet_medium_opt_direct` | ZCU104 | 5 ns | the same |
| `sp_unet_large_opt_direct` | ZCU104 | 5 ns | the same |

The optimized kernels give the bit-exact model's answers, bit for bit: `classify_fixed` in
`tools/specpw_unet_model.py`, the same as the plain C++ loops in
`src/golden/specpw_unet_fixed_golden.hpp`, which the testbench checks every output against
(`tb/sp_unet_small_opt/sp_unet_opt_check.hpp`, shared by the three sizes). Same class as the
float golden kernel on the testbench's 2 000 random pixels, scores within 0.01. On the aegean
pixels the fixed point keeps 99.85 / 99.85 / 99.94 % of the int8 model's classes (below).

The design: `src/hls/sp_unet_small_opt/sp_unet_opt.hpp`; the DATAFLOW chain, shared by
the sizes: `sp_unet_opt_dataflow.hpp`; each size's table of layer widths (PI x PO, DSP or
LUTs): its `.cpp`. One notebook, `software/pynq/sp_unet_small_opt/` (the other two take it
through `include.txt`), checked against `reference_fixed_sp_unet_SIZE.npy`
(`python3 tools/specpw_unet_model.py fixed IMAGE`).

## Golden kernels: what they do

The golden kernels are plain loops, one multiply at a time, in float. Only what they
need to fit on the chip:

| Technique | Golden |
|---|---|
| n patches per start (one start for the whole image when the buffers fit in CMA) | ✓ |
| Stem pixel by pixel straight from DDR: the 110-band patch is never stored | ✓ |
| ReLU fused into each conv's write (no buffer per ReLU) | ✓ |
| Maps of 32 x 32 x 32 floats and more in UltraRAM, BRAM left for the float weight ROM | ✓ |
| Weights as float ROM holding the int8 values (no int-to-float converter, see justounetsimple.cpp) | ✓ |

## Memory (predicted, golden kernels)

| Size | Weights (float ROM, BRAM) | Maps in UltraRAM | Maps in BRAM |
|---|---|---|---|
| small | 103 KB | 31 of 96 URAM blocks | 90 KB |
| medium | 353 KB | 56 of 96 | 120 KB |
| large | 622 KB | 74 of 96 | 156 KB |

Counted at 4 096 floats per URAM block. The ZCU104 has 1.4 MB of BRAM, so large's weights
take about half of it. To be replaced by the csynth reports' utilization.

## Flow of the optimized kernels

```
DDR ─► read_rows_ahead x 4 (HP0-3, 128 bit, every 4th patch row, bursts a row ahead)
     └ pixel_buffer: z-score (int8), 4-row strip ping-pong in BRAM, pixels in 2x2-block order
     └ convsp1 (packed, 2 multiplies per DSP) ─► convsp2 ─► convsp3 (+ 4 pixels into a block)
     └ double1: wino 3x3, wino 3x3 ──► pool + skip ──► skip FIFO (UltraRAM, 512 blocks) ─────┐
     └ double2: wino, wino (16 x 16) ─► pool (block -> pixel)                                 │
     └ up1: wino up (8 x 8 -> 16 x 16 blocks) ─► double3: wino, wino ─► blocks -> rows       │
     └ up2: wino up (16 x 16 -> 32 x 32 blocks) ─► cat ◄─────────────────────────────────────┘
     └ double4: wino, wino ─► blocks -> rows ─► final 1x1 ─► store (HPC0, 3 scores + class)
```

## Optimizations used

| Technique | small_opt | medium_opt | large_opt |
|---|---|---|---|
| One start for the whole capture, loop tiling into 32 × 32 patches, edges repeated in hardware | ✓ | ✓ | ✓ |
| Preprocessing in hardware: band drop + integer z-score (int8, 5 fraction bits), its multipliers in LUTs | ✓ | ✓ | ✓ |
| Fixed point: int8 weights, int16 activations (10 fraction bits), int32 sums, power-of-two scales as shifts | ✓ | ✓ | ✓ |
| DATAFLOW, one process per layer, all maps on chip, every process sized for the same T | ✓ | ✓ | ✓ |
| 3x3 convs as integer Winograd F(2x2, 3x3): 16 multiplies per 2x2 block and channel pair, not 36, exact (U ×4, Y = 4Y >> 2) | ✓ | ✓ | ✓ |
| UpBlock (nearest ×2 + 3x3) from the low-res map: the upsampled tile's equal rows zero a row and column of V, 9 multiplies, not 36 | ✓ | ✓ | ✓ |
| Maps in 2x2-block order: max-pool is a block, Winograd's 2x2 outputs need no reordering | ✓ | ✓ | ✓ |
| Window across row and patch boundaries (centre a block row + 2 behind the input): no per-patch bubble | ✓ | ✓ | ✓ |
| Window as 4 rotating column slots, 1 write + 1 read each at fixed places (no run-time bank choice) | ✓ | ✓ | ✓ |
| Stem without a patch buffer: 4-row strip ping-pong (`pixel_buffer`, block order) | ✓ | ✓ | ✓ |
| convsp1 on the int8 bands, two multiplies per DSP48E2 (`pointwise_wide`, 18-bit spacing, groups of 8) | 32 bands/cycle | 16 | 8 |
| Skip connection in an UltraRAM FIFO (512 blocks; needs up to 248) | ✓ | ✓ | ✓ |
| Transformed weights in one wide memory a layer: LUTRAM when ≤ 64 steps deep, BRAM deeper | ✓ | ✓ | ✓ |
| DSPs kept at 1 200-1 500 (for the clock): small layers' multipliers in LUTs (convsp2/3, final, z-score) | ✓ | ✓, T = 12 288 instead of 9 216 | ✓, + double1a, double3b, double4b in LUTs |
| 4 read ports, read bursts requested a row ahead; score writes with acknowledgements collected later | ✓ | ✓ | ✓ |
| Cycle counter register | ✓ | ✓ | ✓ |

## Latency per stage (cycles per patch, predicted)

Every process is a pipelined loop at II = 1, one patch after another without gaps.
- Winograd layer: blocks (256 at 32 × 32, 64 at 16 × 16) × IN/PI × OUT/PO; UpBlock: low-res
  pixels (64 for up1, 256 for up2) × IN/PI × OUT/PO.
- convsp1: 1 024 × ceil(110 / P1); convsp2/3, final: 1 024 × OUT/PO.
- pixel_buffer: 8 strips × max(480 words in, 4 × 32 × ceil(110 / P1) out).

| Stage | small (PI × PO) | medium | large |
|---|---|---|---|
| read (per port) | 3 840 | 3 840 | 3 840 |
| pixel_buffer | **4 096** | 7 168 | 14 336 |
| convsp1 (packed) | **4 096** (P1 32) | 7 168 (P1 16) | 14 336 (P1 8) |
| convsp2 / convsp3 | **4 096** / **4 096** | 8 192 / **12 288** | **16 384** / **16 384** |
| double1 a / b | **4 096** (3×3) / **4 096** (3×3) | **12 288** (2×3) / **12 288** (4×3) | **16 384** (2×4, LUTs) / **16 384** (4×4) |
| double2 a / b | 3 072 (2×3) / **4 096** (3×3) | **12 288** (2×3) / **12 288** (4×3) | **16 384** (2×4) / **16 384** (4×4) |
| up1 | **4 096** (3×3) | **12 288** (4×3) | **16 384** (4×4) |
| double3 a / b | 3 072 (3×2) / 3 072 (3×1) | **12 288** (3×2) / **12 288** (3×1) | **16 384** (4×2) / **16 384** (2×2, LUTs) |
| up2 | **4 096** (3×3) | **12 288** (4×3) | **16 384** (4×4) |
| double4 a / b | **4 096** (6×2) / **4 096** (2×2) | **12 288** (4×3) / **12 288** (3×1) | **16 384** (4×4) / **16 384** (2×2, LUTs) |
| final, pools, reorders, concat, store | ≤ 3 072 | ≤ 3 072 | ≤ 3 072 |
| **T** | **4 096** (DDR reads: 3 840) | **12 288** | **16 384** |

## Multipliers and memory (predicted)

| | small_opt | medium_opt | large_opt |
|---|---|---|---|
| Multiply-adds per patch, direct | 13.8 M | 40.5 M | 70.5 M |
| Multiplies per patch done (Winograd, up trick) | 7.4 M (1.9× fewer) | 17.8 M (2.3×) | 30.1 M (2.3×) |
| DSPs: Winograd + convsp1 | 1 090 + 384 = **1 474 of 1 728** (85 %) | 1 176 + 192 = **1 368** (79 %) | 1 312 + 128 = **1 440** (83 %) |
| Multipliers in LUTs | 148 (convsp2/3, final, z-score) | 108 (the same) | 352 (+ double1a, double3b, double4b: 256 Winograd) |
| Skip FIFO (UltraRAM, 72 bit each) | 11 URAM (768 bit wide) | 22 (1 536) | 29 (2 048) |
| Weights in BRAM (layers > 64 steps) | none | ~92 BRAM36 (double2, up1, double3) | ~122 BRAM36 |

The csynth reports replace these: check the DSP count against this table (more means a
multiply was not mapped as intended), the Bind Op Report (`dsp` vs `fabric` per layer as in
the tables), `Final II = 1` for every `conv` loop, and that no FIFO of a block stream went to
BRAM (they are 768-4 096 bits wide: `impl=srl`).

## Inference time (predicted)

time = (patches + 1) × T / clock; a 598 × 1092 capture is 19 × 35 = 665 patches.

| Kernel | T | clock | time |
|---|---|---|---|
| `sp_unet_small_opt` | 4 096 | 187.5 MHz | **14.5 ms** (memory-bound: the four read ports need 3 840) |
| `sp_unet_medium_opt` | 12 288 | 187.5 MHz | **43.6 ms** (T = 9 216, 32.7 ms, needs ~1 760 DSPs) |
| `sp_unet_large_opt` | 16 384 | 187.5 MHz | **58.2 ms** |

## Boards: limits, differences and why

| | small_opt | medium_opt | large_opt |
|---|---|---|---|
| Limit | DDR reads (11.2 GB/s at T), then the stem | the DSP budget (kept ≤ 1 500 for the clock) | the DSP budget, and LUTs for 256 Winograd multipliers |
| Faster would need | more read bandwidth (5 ports) | ~1 760 DSPs for T = 9 216 | ~1 700 DSPs to have all of T = 16 384 on DSPs; the next step down, T = 12 288, does not divide the 64-channel layers |
| If the clock still misses | small layers back to DSPs frees LUTs | — | double1a back on DSPs (1 568) if the LUTs are the problem |

The ZCU104 has 1 728 DSP48E2, 312 BRAM36, 96 URAM and 230 k LUTs. Zynq-7000 versions
(7030, 7020) are not written yet: they would use the same modules with smaller PI × PO and,
on the 7020, LUT multipliers.

## The direct kernels (`sp_unet_SIZE_opt_direct`): no Winograd

The same kernels, inputs, outputs and answers (bit for bit, the same testbench checks and
notebook), built only from techniques the other models already use, for a simpler design
to check and explain. The price: 2.3x more multiplies, so 2.4-4.3x slower for about the
same DSPs.

| | Winograd (`_opt`) | Direct (`_opt_direct`) |
|---|---|---|
| 3x3 convs | `wino_conv3x3`, 16 multiplies per 2x2 block and channel pair | `conv3x3_grouped`: 9 per pixel, PI inputs x PO outputs a cycle (`conv3x3_serial` with PO outputs a step) |
| UpBlocks | `wino_up_conv3x3` from the low-res map, 9 per block | `relu_pool_up` / `relu_up` upsample, then a plain 3x3 conv (36 per 2x2 block) |
| Map order | 2x2 blocks | raster, one pixel a vector |
| ReLU + rounding | inside each conv | `relu_act` between the convs; `relu_pool_skip` (ReLU, pool, and the skip) after double1 |
| Skip FIFO | 512 blocks, UltraRAM | 2 048 pixels, UltraRAM (a patch at most, each patch is done on its own) |
| Window | across patches, no bubble | per patch, W + 1 steps of bubble (in the cycle counts below) |
| FIFOs between the processes | 2 deep (not yet measured) | 16 deep, in LUTs (SRL): the layers run at the same rate with no time to spare (below) |
| convsp2 / convsp3 multipliers | LUTs | DSPs (`STEM_DSP`), for the LUTs the deeper FIFOs take |
| A conv's weights | one wide memory (a word a step) | one wide memory a layer too: block RAM when > 64 steps deep, LUTRAM when shallow (was one memory per PI × PO × tap) |

Shared with the Winograd kernels: the readers, `pixel_buffer` (raster order), the packed
convsp1, convsp2/3 and final (`pointwise`), `store_scores`, the cycle counter, the z-score in
LUTs.

| Layer (PI x PO) | small | medium | large |
|---|---|---|---|
| convsp1 (P1) | 7 168 (16) | 14 336 (8) | 14 336 (8) |
| double1 a / b | 9 513 (4×4) / 9 513 (4×4) | 33 824 (3×3) / 33 824 (6×3) | 67 648 (8×1) / 67 648 (16×1) |
| double2 a / b | **9 828** (4×2) / **9 828** (8×2) | **34 944** (3×3) / **34 944** (6×3) | **69 888** (8×1) / **69 888** (16×1) |
| up1 | **9 828** (8×2) | **34 944** (6×3) | **69 888** (16×1) |
| double3 a / b | **9 828** (8×1) / **9 828** (4×1) | **34 944** (3×3) / 26 208 (6×1) | **69 888** (8×1) / **69 888** (4×1) |
| up2 | 9 513 (4×4) | 33 824 (6×3) | 67 648 (16×1) |
| double4 a / b | 8 456 (24×1) / 8 456 (8×1) | 33 824 (6×3) / 25 368 (6×1) | 67 648 (16×1) / 67 648 (4×1) |
| **T** | **9 828** | **34 944** | **69 888** |
| DSPs (convs + convsp1 + convsp2/3) | 1 188 + 192 + 72 = **1 452** (84 %) | 1 161 + 96 + 40 = **1 297** (75 %) | 1 008 + 128 + 48 = **1 184** (69 %) |
| 665 patches at 187.5 MHz | **34.9 ms** | **124 ms** | **248 ms** |
| measured, FPGA alone | **30.6 ms at 214 MHz** (9 856 cycles a patch; 41 ms / 11 503 with 2-deep FIFOs) | – | – |
| Winograd kernel, for comparison | 14.5 ms, 1 474 DSPs | 43.6 ms, 1 368 | 58.2 ms, 1 440 |

A conv layer takes (H x W + W + 1) x IN/PI x OUT/PO cycles a patch, with 9 x PI x PO
multipliers. Large has every channel count a power of two, so its only table between 1 136
and 1 604 DSPs is this one; the next faster (33 824 cycles, 120 ms) needs ~2 100.

### Equal-rate processes need deep FIFOs between them

The first board run was 17 % slower than every process's own time (11 503 cycles a patch against
9 841), although HLS reported II = 1 everywhere and an interval of 9 842 a patch, and DDR was not the
limit (the same cycle count at 187.5 and 214 MHz).

The cause: most layers are sized for the same T (here 9 828), so none has time to spare, and an HLS
pipeline freezes *all* of its stages whenever one read finds its FIFO empty or one write finds it
full. A conv reads at the start of its pipeline and writes ~12 stages later, so a short wait on its
input also holds back outputs already in flight. With the default 2-deep FIFOs one conv's few-cycle
freeze froze the next one, and the next, and since every one of them runs at exactly T, none could
win the cycles back: each freeze went straight onto T.

The fix: every FIFO between the processes 16 deep (`#pragma HLS STREAM depth = 16`, `impl = srl`), so a
producer can run a few pixels ahead and a short freeze on either side is absorbed. In shift-register
LUTs (SRL16) 16 deep costs what 8 deep does, about one LUT per bit of width (~8 k LUTs here); to pay
for them, convsp2/convsp3 moved from LUT multipliers (~25 k LUTs) to DSPs.

| | cycles a patch | 665 patches |
|---|---|---|
| HLS interval (every process alone) | 9 842 | — |
| board, 2-deep FIFOs | 11 503 | 41 ms (187.5 MHz), 35.7 ms (214 MHz) |
| cycle model with HLS's pipeline stalls, 2 / 8 / 16 deep | 11 472 / 9 939 / 9 839 | — |
| board, 16-deep FIFOs | **9 856** | **30.6 ms** (214 MHz) |

The simple cycle model (an iteration's read and write in the same cycle) missed this: it gave 9 841
with 2-deep FIFOs. Only the model with the real pipeline behaviour (reads at issue, writes at the
loop depth from hls.log, the whole pipeline frozen on any wait, loops drained at their exit)
reproduced the board, within 0.3 %. Deeper FIFOs in front of the DDR readers (`s_raw`, tried first)
changed nothing.

## Integer formats (the bit-exact model)

| Value | Format |
|---|---|
| input | z-score × 2^5, int8, saturated at ±127 (justounetsimple_opt's integer z-score) |
| weights, biases | int8, a power-of-two scale per tensor |
| sums | exact; at most 2^23.7 on the aegean pixels, so int32 holds them |
| activations | int16 with 10 fraction bits (up to 32.0), ReLU, rounded half up, saturated; the largest on the aegean pixels is 9.1 |
| scores | int32 with 16 fraction bits |

Fixed point gives the same class as the int8-weight float model on 99.85 / 99.85 / 99.94 % of
the aegean pixels (small / medium / large).

## Changes

- 2026-10-26: golden modules (spectral stem, conv block, up block, concat for the skip),
  the float golden model, the golden kernels for the three sizes, their testbench and
  notebook, and the bit-exact integer model (numpy and C++).
- 2026-10-26: optimized kernels `sp_unet_small_opt`, `sp_unet_medium_opt`, `sp_unet_large_opt` (ZCU104):
  integer Winograd F(2x2, 3x3) (`wino_conv3x3`) and UpBlocks from the low-res maps
  (`wino_up_conv3x3`), maps in 2x2-block order (`pool_blocks`, `blocks_to_rows`, `concat_blocks`),
  `pixel_buffer` in block order, packed convsp1 (`pointwise_wide`), skip FIFO in UltraRAM; layers
  sized for T = 4 096 / 9 216 / 16 384 cycles a patch (predicted 14.5 / 32.7 / 58.2 ms for 665
  patches); `specpw_unet_model.py fixed`; their notebook.
- 2026-10-26: DSPs kept at 1 200-1 500 for the clock: small 1 506 → 1 474 (z-score in LUTs), medium
  1 632 → 1 368 (T 9 216 → 12 288, 32.7 → 43.6 ms, no Winograd multiplier in LUTs), large 1 568 →
  1 440 (double1a in LUTs, same T).
- 2026-10-26: direct kernels `sp_unet_SIZE_opt_direct` (no Winograd): `conv3x3_grouped` (PI x PO a
  cycle), `relu_act`, `relu_pool_skip`, `concat_pixels`, justounetsimple_opt's `relu_pool_up` and
  `relu_up`; raster order; T = 9 828 / 34 944 / 69 888 cycles a patch, 1 380 / 1 257 / 1 136 DSPs
  (predicted 34.9 / 124 / 248 ms). The shared checks, channel counts and fraction bits moved to
  `sp_unet_opt_common.hpp`.
- 2026-10-27: `sp_unet_small_opt_direct` on the board: 41 ms for a whole capture, FPGA alone, at
  187.5 MHz, against 34.9 ms predicted (≈ 11 560 cycles a patch against 9 828). A cycle simulation
  puts only ~3 % of it on `relu_pool_up`'s and `relu_up`'s take-then-give; with ideal memory the chain
  runs at 9 841, so the rest is not in its model of the chain.
- 2026-10-27: rebuilt with `s_raw` 512 deep and at 214 MHz: again exactly 11 503 cycles a patch (35.7 ms).
  The same count at another clock means the limit is in the kernel's logic, not DDR (DDR-bound, the
  faster clock would have needed more cycles a patch). `s_raw` back to 64.
- 2026-10-27: csynth: interval 6 545 040 cycles for 665 patches (9 842 a patch, the 16 x 16 convs; every
  other process below), clock estimate 3.65 ns, LUT 189 k (82 %), DSP 1 380, BRAM18 102, URAM 3. Cosim:
  RTL bit-exact (8 000 outputs), cycles register 52 558 for the testbench's 4 patches, against 49 062 in
  a cycle simulation of the chain with its weight loads: ~875 cycles a patch of stalls the simulation
  does not have, even with cosim's fast memory. Next: `--dataflow-profile` to see which process waits.
- 2026-10-27: the cause found: equal-rate processes freezing each other. An HLS pipeline freezes all
  its stages when one read or write must wait; the convs run at the same rate with no time to spare,
  so with 2-deep FIFOs one conv's short wait froze the next and neither caught up. A cycle model with
  these pipeline stalls (reads at issue, writes at the depth from hls.log, the loops' exits) gives
  11 472 cycles a patch, against 11 503 measured. Every FIFO between the processes now 16 deep, in
  SRL: the model gives 9 839 (the convs' own 9 828 + loop exits); predicted ~35 ms at 187.5 MHz,
  ~30.6 ms at 214 MHz. convsp2/convsp3 moved to DSPs (`STEM_DSP`) for the LUTs the FIFOs take (they
  were ~25 k of the 189 k): DSPs 1 452 / 1 297 / 1 184 (small / medium / large).
- 2026-10-27: measured with the 16-deep FIFOs: 9 856 cycles a patch, 30.6 ms at 214 MHz (model 9 839).
  The fix is described under "Equal-rate processes need deep FIFOs between them".
- 2026-10-27: `sp_unet_large_opt_direct` never finished routing: Vivado stayed in Phase 5.2/5.3 (global
  iterations), estimated congestion level 6 (5 and up reduce routability). `conv3x3_grouped` kept its
  weights in one memory per (PO, PI, tap): 1 008 memories up to 256 deep on large. Now one wide memory a
  layer, a step's PO x PI x 9 weights a word, block RAM when deeper than 64 steps (~52 BRAM36 on large,
  ~60 on medium, none on small), LUTRAM otherwise; the same reads, so the same speed. Bit-exact on the
  three sizes; not yet routed.
