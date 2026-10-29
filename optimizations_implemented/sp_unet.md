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
| `sp_unet_medium_opt_direct` | ZCU104 | 214 MHz | the same; ran at the predicted ~109 ms (34 960 cycles a patch) |
| `sp_unet_large_opt_direct` | ZCU104 | 214 MHz | the same; ran at the predicted ~218 ms (69 900 cycles a patch) |
| `sp_unet_small_opt_direct_zynq7020` | Zynq-7020 / ZedBoard (`--board zynq7020` / `zedboard`) | 12.8 ns (78.1 MHz) | the direct kernel sized for the 7020, 195 DSPs; **measured 665 ms** on the ZedBoard (≈ 78 100 cycles a patch; predicted ~694) |
| `sp_unet_medium_opt_direct_zynq7020` | Zynq-7020 / ZedBoard | 12.8 ns | **does not fit the 7020** (csynth: ~59 k LUTs of 53.2 k, see below); bit-exact in C++; not to be built |
| `sp_unet_large_opt_direct_zynq7020` | Zynq-7020 / ZedBoard | 12.8 ns | **does not fit the 7020** (csynth: ~69 k LUTs, 117-129 % of the BRAM); bit-exact in C++; not to be built |

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
| predicted, 665 patches at 187.5 MHz | 34.9 ms | 124 ms | 248 ms |
| predicted, 665 patches at 214 MHz | 30.6 ms (9 839 cycles a patch) | 109 ms (34 960) | 218 ms (69 900) |
| **measured, FPGA alone, 214 MHz** | **30.6 ms** (9 856 cycles a patch; 41 ms / 11 503 with 2-deep FIFOs) | **as predicted** | **as predicted** |
| Winograd kernel, for comparison | 14.5 ms, 1 474 DSPs | 43.6 ms, 1 368 | 58.2 ms, 1 440 |

A conv layer takes (H x W + W + 1) x IN/PI x OUT/PO cycles a patch, with 9 x PI x PO
multipliers. Large has every channel count a power of two, so its only table between 1 136
and 1 604 DSPs is this one; the next faster (33 824 cycles, 120 ms) needs ~2 100.

### Predicted and measured times, and how the predictions are made

| Kernel | Predicted cycles a patch | Predicted, 665 patches at 214 MHz | Measured |
|---|---|---|---|
| `sp_unet_small_opt_direct` | 9 839 | 30.6 ms | **9 856 cycles, 30.6 ms** (with 2-deep FIFOs: 11 503, 35.7 ms) |
| `sp_unet_medium_opt_direct` | 34 960 | 109 ms | **as predicted** (2026-10-28) |
| `sp_unet_large_opt_direct` | 69 900 | 218 ms | **as predicted** (2026-10-28) |

Why these are predictions: they come from a model of the design, made before the kernel ran,
not from a run.
- **Cycles a patch:** every process works on one patch after another, so the slowest one sets the
  pace, T. A conv layer takes (H x W + W + 1) x IN/PI x OUT/PO cycles a patch (its pixels plus the
  W + 1 steps its 3x3 window trails the input, IN/PI x OUT/PO steps each, one a cycle at II = 1),
  plus its loop's exit (~12-15 cycles). HLS's csynth report gives the same per process (an interval
  of 9 842 a patch for small).
- **That the processes do not slow each other down:** checked with a cycle model of the whole chain
  that stalls the way HLS pipelines do (reads at issue, writes at the loop depth from hls.log, the
  whole pipeline frozen while one read or write waits). It reproduced the board within 0.3 % both
  with the 2-deep FIFOs (11 472 against 11 503) and with the 16-deep ones (9 839 against 9 856).
- **That memory is not the limit:** the reads needed at T (245 KB a patch: 5.3 / 1.5 / 0.75 GB/s for
  small / medium / large at 214 MHz) are below what the four ports get from DDR (~9.2 GB/s measured
  with justounetsimple_opt).
- **Time:** (patches + 1) x T / clock; the + 1 is the first patch's way through the chain. The clock
  is the one the bitstream was built for: 214 MHz.

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
reproduced the board, within 0.3 %. Deeper FIFOs in front of the DDR readers (`cube_words`, tried first)
changed nothing.

## The Zynq-7020 kernels (`sp_unet_SIZE_opt_direct_zynq7020`)

The direct kernels sized for the PYNQ-Z2's Zynq-7020 (`--board zynq7020`): the same
arithmetic, the same answers bit for bit (the same testbench checks, the same notebook), on a
chip with 220 DSP48E1 (these use at most 205), 53 k LUTs, 106 k flip-flops, 140 BRAM36, no
UltraRAM and 64-bit HP ports. The design: `src/hls/sp_unet_small_opt_direct_zynq7020/
sp_unet_opt_zynq7020.hpp`; the chain, shared by the sizes: `sp_unet_direct_zynq7020_dataflow.hpp`;
each size's table in its `.cpp`.

On this chip the DSPs set the pace (the reads need 30 720 cycles a patch through one port, far
below), and the LUTs and flip-flops are what is short. So:

| | ZCU104 direct | Zynq-7020 direct |
|---|---|---|
| DSPs | 1 184-1 452 | **195 / 200 / 184** (small / medium / large), at most 205 |
| A conv's 3x3 window | in registers (9 x IN activations), picked by multiplexers | **in three row memories** (`conv3x3_lines`): a step reads one column of the window at one address; no window registers |
| A conv's step | PI x PO x the whole window (9 multipliers a unit) | PI x PO x **one tap or one column** of the window (`KT` = 1 or 3): sized in units of 1-3 DSPs, not 9 |
| ReLU and rounding | `relu_act` / ReLU modules, 32-bit sums on the FIFOs, every channel rounded at once | **inside every conv** (`RELU_SHIFT`), rounded as each output group finishes (PO rounding units, not OUT): int16 activations on every FIFO, 7 processes fewer, the pooling only takes maxima (SHIFT 0) |
| convsp1 | packed for the DSP48E2 (18 bits apart, groups of 8) | packed for the DSP48E1 (16 bits apart, groups of 2), the pairs in passes (`OPASS`), a pass's pairs rounded at a time |
| convsp2 / convsp3 / final, z-score | DSPs / DSPs / LUTs, LUTs | PI inputs a step (`pointwise`'s `PI`) on 1-4 DSPs each, z-score on DSPs (small, large) |
| Read ports | 4 x 128 bit | **1 x 64 bit** (30 720 cycles a patch; T is 76 000 or more) |
| FIFOs between the processes | 16 deep, shift registers | **2 deep** (16 deep: 7 % faster, but the LUTs are not there) |
| Skip FIFO | UltraRAM | block RAM, 2 048 deep |
| Clock | 214 MHz | the board's 12.8 ns (78.1 MHz) |

Finer steps were worth it: with whole-window steps the fastest tables within 205 DSPs were
78 624 / 304 416 / 541 184 cycles a patch; a tap or a column a step gives 76 176 / 228 528 / 420 864
(-3 %, -25 %, -22 %). The window in memory: in registers, the ten convs' windows alone would take
23.6 k / 45 k / 60 k of the 106 k flip-flops (small / medium / large), and multiplexers to pick each
step's channels out of them, on top of the rest (small's first csynth: 92.5 k flip-flops estimated). The 16-deep FIFOs were checked with the pipeline-stall model for the 7020 small
table: 81 386 cycles a patch with 2-deep FIFOs (+7 %), 79 190 with 4, 76 202 with 16.

| Layer (PI x PO x KT, cycles a patch) | small | medium | large |
|---|---|---|---|
| convsp1 (P1, passes) | 73 728 (20, 12) | 172 032 (4, 6) | 229 376 (4, 8) |
| double1 a / b | **76 176** (6x3x1) / **76 176** (6x3x1) | **228 528** (12x1x1) / **228 528** (24x1x1) | 406 272 (4x1x3) / 406 272 (8x1x3) |
| double2 a / b | 59 184 (12x1x1) / 59 184 (24x1x1) | 157 824 (8x2x1) / 210 432 (3x3x3) | **420 864** (4x1x3) / **420 864** (8x1x3) |
| up1 | 59 184 (24x1x1) | 210 432 (3x3x3) | **420 864** (8x1x3) |
| double3 a / b | 59 184 (12x1x1) / 59 184 (6x1x1) | 157 824 (16x1x1) / 157 824 (8x1x1) | **420 864** (4x1x3) / **420 864** (2x1x3) |
| up2 | **76 176** (6x3x1) | **228 528** (24x1x1) | 406 272 (8x1x3) |
| double4 a / b | **76 176** (24x1x1) / **76 176** (8x1x1) | **228 528** (24x1x1) / **228 528** (6x1x1) | 406 272 (8x1x3) / 406 272 (2x1x3) |
| **T** | **76 176** | **228 528** | **420 864** |
| DSPs (convs + convsp1 + convsp2/3, final + z-score) | 164 + 20 + 7 + 4 = **195** | 184 + 8 + 4 + 4 = **200** | 168 + 8 + 4 + 4 = **184** |
| Block RAM (rough: weights as ROM and copy, row memories, skip FIFO) | ~35 of 140 BRAM36 | ~75 | ~115 |

A conv layer takes (H x W + W + 2) x 9/KT x IN/PI x OUT/PO cycles a patch, with KT x PI x PO
multipliers (the window trails the input by a row and two pixels).

### Predicted times

| Kernel | Cycles a patch | 665 patches at 78.1 MHz | at 100 MHz, if it closes |
|---|---|---|---|
| `sp_unet_small_opt_direct_zynq7020` | 76 176 (+ ~7 %: ~81 400) | ~694 ms; **measured 665 ms** (≈ 78 100 cycles a patch, +2.6 %) | ~542 ms |
| `sp_unet_medium_opt_direct_zynq7020` | 228 528 (+ ~7 %) | **~2.1 s** | ~1.6 s |
| `sp_unet_large_opt_direct_zynq7020` | 420 864 (+ ~7 %) | **~3.8 s** | ~3.0 s |

The + 7 % is the 2-deep FIFOs between the processes (the cycle model of small: 81 386 cycles a
patch against 76 202 with 16 deep); for medium and large it is assumed the same. Predicted the same way as the ZCU104 direct kernels' (see "Predicted and measured times"): the
slowest process's cycles a patch, from the table; the pipeline-stall model for the FIFOs;
(patches + 1) x T / clock. On the ZCU104 that came within 0.2 % of the board. To be replaced by
measurements.

### Medium and large do not fit the 7020 (decided 2026-10-28)

Only small runs on the 7020 (665 ms measured). Medium and large run on the ZCU104; on a Zynq-7000,
the 7030 (HYPSO-2: 400 DSPs, 79 k LUTs) is the place to size them for. Their 7020 kernels stay in the
repository, bit-exact in C++, as the starting point if that changes, but they do not fit:

| | LUT, HLS estimate (~real at 0.55) | BRAM18 | FF |
|---|---|---|---|
| small (built, 665 ms) | 95 489 before the 2-deep FIFOs | 172 | 67 602 |
| medium | 107 943 (~59 k of 53.2 k) | 257 of 280 | 71 503 |
| large | 124 987 (~69 k) | 328-362 of 280 | 56 862-76 445 |

Why: large's csynth report puts 65 % of the LUTs in the 10 conv engines (5.6-10 k each) and 8 %
in `relu_pool_up`. Each engine has a fixed cost (its row memories, input buffer, step control,
rounding) whatever its number of multipliers; small's narrow channels keep it inside the chip,
medium needs ~20 % less, large ~30 %. Trimming single modules does not get there.

What would: one shared conv engine running the layers one after another for each patch, the maps
in block RAM between them: the same DSPs busy all the time, one engine's overhead instead of ten.
Not built.

Tried on the way and undone: the row memories reshaped into one word per row (ARRAY_RESHAPE): HLS
made each lane's write its own cycle, 5-13x slower (intervals 1 399 385 553 / 1 975 644 694).



- **DSPs**: the next faster table needs more than 205 on every size.
- **LUTs**: the first small build needed 66 239 LUTs as logic of 53 200. Its csynth report put
  57 % of them in the 10 convs, 7 % in convsp2's 24 LUT multipliers, 11 % in the three ReLU
  modules' 32-bit rounding. Cut (the same answers): every conv rounds only its PO lanes as each
  output group finishes and sends activations; the pooling modules then only take maxima; convsp2/3,
  the final layer and the z-score on 1-4 DSPs (`pointwise`'s `PI`); convsp1 rounds a pass at a time.
- **LUTs and flip-flops**: the windows are in memory now; what is left in LUTs are the FIFOs
  between the processes (16 deep, wider on the larger models), the reader and pixel buffer, and
  each conv's adder tree and control. If one does not fit, the FIFOs can go back to 2 deep (+7 %
  time, a few thousand LUTs less).
- **Block RAM** is the next to watch on large (~115 of 140): the weights are stored twice (the
  ROM and the copy each conv reads), and the skip FIFO is 2 048 deep. Generated weight memories
  (once) or a 1 024-deep skip FIFO would free ~35 and ~15.
- **Speed**: the Winograd design would need 2.3x fewer multiplies, but more LUTs, which this chip
  has least of; not tried here.

On the ZedBoard (`--board zedboard`, the same chip, the OPU system's PetaLinux) the build packs
`K_opu.zip` with the generic `classify` program (`software/opu/justounetsimple_opt/classify.c`):
`classify cube.npy scores.npy reference_fixed_sp_unet_SIZE.npy`.

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
- 2026-10-27: rebuilt with `cube_words` 512 deep and at 214 MHz: again exactly 11 503 cycles a patch (35.7 ms).
  The same count at another clock means the limit is in the kernel's logic, not DDR (DDR-bound, the
  faster clock would have needed more cycles a patch). `cube_words` back to 64.
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
- 2026-10-28: `sp_unet_medium_opt_direct` and `sp_unet_large_opt_direct` on the board at 214 MHz: at the
  predicted ~109 ms and ~218 ms. Predicted and measured times, and how the predictions are made, in
  "Predicted and measured times".
- 2026-10-28: Zynq-7020 kernels `sp_unet_SIZE_opt_direct_zynq7020`: one 64-bit read port, convs with a
  row of the window a step (`conv3x3_grouped`'s `KR`) and their ReLU inside (`RELU_SHIFT`), convsp1
  packed for the DSP48E1 in passes (`pointwise_packed`'s `OPASS`, `zynq7020/pointwise_wide.hpp`), FIFOs
  16 deep in LUTRAM, the skip in block RAM; 197 / 197 / 192 DSPs, T = 76 104 / 228 312 / 419 328
  cycles a patch (predicted 649 ms / 1.95 s / 3.58 s at 78.1 MHz). `sp_unet_opt_common.hpp` no longer
  includes the ZCU104 header (each kernel's .cpp includes its own first).
- 2026-10-28: first 7020 build (small, `--board zedboard`): HLS and Vivado synthesis pass, placement
  refused: 66 239 LUTs as logic needed, 53 200 on the chip (124 %). To cut. OPU program for the
  ZedBoard: the generic `classify.c` (`software/opu/sp_unet_SIZE_opt_direct_zynq7020/include.txt`).
- 2026-10-28: 7020 LUT cuts after the first build: convs round PO lanes as each output group finishes
  and every conv sends activations (double1b/2b/3b too: `relu_pool_skip` / `relu_pool_up` / `relu_up`
  take activations, SHIFT 0, `relu_in`); `pointwise` takes PI inputs a step (convsp2/3 and final on 1-4
  DSPs instead of up to 32 LUT multipliers); `pointwise_packed` rounds a pass's pairs at a time; z-score
  on DSPs on small and large. DSPs 196 / 201 / 184, the same T. Bit-exact; the other kernels unchanged.
- 2026-10-28: csynth of the cut small 7020 kernel: HLS's LUT estimate 121 293 -> 93 231 (-23 %); the first
  build's real count was 66 239 for HLS's 121 293, so ~51 000 of 53 200 expected (~96 %). DSP 201 (5 more
  than the table: HLS put something else on DSPs), FF 92 496 estimated (87 %), BRAM18 116 / 280; interval
  76 119 cycles a patch, as predicted; clock estimate 9.93 ns (~100 MHz) against the 12.8 ns target.
- 2026-10-28: 7020 convs keep their 3x3 window in three row memories (`conv3x3_lines`: a step reads one
  column of the window at one address; one tap or a column a step, `KT`): the window registers would
  have taken 45 k / 60 k of the 106 k flip-flops on medium / large. New tables: DSPs 195 / 200 / 184,
  T = 76 176 / 228 528 / 420 864 cycles a patch (the window trails by W + 2). Bit-exact; 9 injected
  bugs caught. The small build running when this was made is of the earlier engine (`conv3x3_grouped`).
- 2026-10-28: second small 7020 build (the LUT cuts, still `conv3x3_grouped`): past the LUT check, but
  placement failed: 7 816 slices needed, 6 321 free of 13 300 (LUTs ~96 %, flip-flops ~87 %, too many
  to pack into slices together). The row-memory engine (`conv3x3_lines`) removes the window registers,
  ~23.6 k of small's flip-flops, and the multiplexers on them: to be built.
- 2026-10-28: first csynth with `conv3x3_lines`: Vitis HLS 2025.2 crashed (segfault) generating the RTL of
  a 24 -> 24 conv; the conv before it had a 36-cycle signed divider in its loop, from turning the step
  into a tap row and column with / 3 and % 3. Now counters (the tap's column and row, the weight
  address), and nested loops for the weight load: no dividers. Bit-exact; the counters' bugs caught.
- 2026-10-28: the HLS crash again, in the same 24 -> 24 conv, without the divider. The row memories were
  arrays of a struct declared inside the function template, packed with AGGREGATE; now plain arrays
  (`act_t rows[3][PI][W * IN/PI]`, a small memory per row and lane), as the synthesized kernels use.
  Bit-exact.
- 2026-10-28: csynth of small with `conv3x3_lines`: interval 76 193 cycles a patch, FF 67 602 estimated
  (63 %, was 92 496), but LUT 95 489 estimated (~52 k real at the earlier ratio, ~98 %) and BRAM18 172.
  To be sure it places: the FIFOs between the processes 2 deep instead of 16 (s_pool1 16, s_pool2 32,
  s_dec3 64 for their bursts): a few thousand LUTs, ~7 % more time (small ~694 ms predicted).
- 2026-10-28: `sp_unet_small_opt_direct_zynq7020` built and ran on the ZedBoard: 665 ms for a whole capture
  at 78.1 MHz, ≈ 78 100 cycles a patch: 2.6 % over the convs' 76 176, not the 7 % the cycle model gave the
  2-deep FIFOs (predicted ~694 ms). Its design: `conv3x3_lines` (the window in row memories), the ReLU in
  every conv, convsp2/3 and final on a few DSPs, 2-deep FIFOs between the processes.
- 2026-10-28: csynth of medium and large (7020): LUT 107 943 / 124 987 estimated (~59 k / ~69 k real of
  53 200), BRAM18 257 / 362 of 280. The row memories, split into a memory per channel lane to get past the
  HLS crash, were 3 x PI small memories a conv (168 on large), a block RAM or a few LUTs each: now one
  memory per row again, with ARRAY_RESHAPE on the lanes (plain arrays, no struct). Skip FIFO 1 024 deep on
  medium and large (a patch, the most it holds; `cfg::SKIP_DEPTH`), 2 048 on small. Bit-exact.
- 2026-10-28: the reshape undone: HLS made each lane's write to the reshaped word its own cycle, and the
  intervals went from 151 982 571 / 279 885 962 to 1 975 644 694 / 1 399 385 553 cycles (medium / large);
  medium's BRAM18 rose to 407. Back to a memory per (row, lane); the 1 024-deep skip FIFOs stay.
- 2026-10-28: decided: the 7020 runs small only; medium and large stay on the ZCU104 (their 7020 kernels
  do not fit: ~59 k / ~69 k LUTs of 53.2 k; see "Medium and large do not fit the 7020").
