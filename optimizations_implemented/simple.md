# The simple kernels (`_simple`, ZCU104)

A second HLS version of five networks, for the paper comparing an HLS accelerator with the
Vitis AI DPU on the ZCU104 (the optimized `_opt` kernels stay the thesis's own). Each is
the golden kernel's layers as plain loops, made fast with the standard pragmas only:
PIPELINE, DATAFLOW, UNROLL and ARRAY_PARTITION (plus the interface and the skip buffer's
storage). No `src/optimized/` modules, no two multiplies per DSP, no `mul<DSP>`: HLS
chooses every multiplier. Design: `docs/superpowers/specs/2026-11-16-simple-hls-kernels-design.md`.

They give **the same answers as the model's `_opt` kernel, bit for bit** (int8 weights, the
same integer formats), so the `_opt` testbench checks, `classify.ipynb` and
`tools/models.py`'s `fixed_scores` serve them unchanged.

## How they are built

`src/simple/layers.hpp`, four kinds of process:

- `conv` (and `conv_concat` for the conv after a concat): same-padded 1x1 or 3x3. One
  pipelined loop over pixel, output chunk, tap and input chunk, II = 1; inside, UI inputs ×
  UO outputs unrolled (UI × UO multipliers). The taps are not unrolled, so a step reads one
  input pixel: no 3x3 window, no line buffer. Steps a patch: `H·W·K²·(IN/UI)·(OUT/UO)`.
  An upsampled input is read at (y/2, x/2), so upsampling needs no process.
  The weights are copied once (on the first patch) into arrays partitioned for the UI × UO
  multipliers.
- `maxpool2`, `duplicate` (the SpecPW-UNet skip).
- `read_zscore` / `read_block`: a patch row (or 1D block) from DDR in one burst through one
  128-bit port, z-scored with the generated `zscore_int8.hpp` (2D) or
  `justoliunet_opt_formats.hpp` (1D) tables; the edge repeated as `_opt` does.
- `write_scores` / `write_block`: the scores and the class, the pixels inside the image only.

The top function is a DATAFLOW loop over the patches (1D: blocks of 256 pixels) with the
layer calls directly in its body, HLS's canonical form: each layer a process, the maps
between them ping-pong buffers, so every layer works on a different patch at once and the
time a patch is the slowest process's. A map is `[row][col]` of pixel words (an
`hls::vector` of all the pixel's channels): one memory as wide as the pixel, holding only
its bits, so the maps need no partitioning.

The SpecPW-UNet skip waits while 8 processes work on later patches: a ping-pong buffer of
depth 10, in UltraRAM.

## Kernels: predicted

Cycle model: the slowest process, at 250 MHz, 665 patches (or 653 016 pixels) an image.
Not built yet: the numbers below are predictions until the user's csynth and board runs.

| Kernel | T (cycles a patch) | Predicted at 250 MHz | Multipliers (UI × UO, summed) | `_opt` measured | DPU |
|---|---|---|---|---|---|
| `spectralpointwise1d_simple` | 15 a pixel (the read: 15 words) | ~39 ms + DDR waits | 204 | 12.1 ms | 168.4 ms |
| `justounetsimple_simple` | 18 432 (conv1); the read ~15 400 + DDR waits | ~49 ms | 372 | 19.3 ms (`_tiled`) | 188.8 ms |
| `sp_unet_small_simple` | 18 432 | ~49 ms | 769 | 26.2 ms | 233.6 ms |
| `sp_unet_medium_simple` | 41 472 | ~110 ms | 1 041 | 93.1 ms | 267.9 ms |
| `sp_unet_large_simple` | 73 728 | ~196 ms | 964 | 186.1 ms | 291.5 ms |

The multipliers are an upper bound on DSPs: HLS makes small constant weights into LUT
shift-adds, so the csynth report's DSP count will be lower.

The per-layer tables (UI, UO, cycles, multipliers) are in each kernel's `.cpp`; the SpecPW-UNet
ones were chosen as the fewest multipliers per layer that keep it within the size's budget
(20 000, 45 000, 75 000 cycles a patch).

## What to check in the first builds

- That consecutive patches overlap: the csynth report's dataflow interval about T, not the
  sum of the processes' latencies.
- `Final II = 1` for every conv's `steps` loop and the read's `take` loop.
- BRAM and URAM: every map is ping-ponged; large is the tightest (~6 Mb of maps, ×2, plus the
  skip's 10 × 0.5 Mb).
- One cosim (`fpga_cosim sp_unet_small_simple zcu104`): new arithmetic code, csim does not prove the RTL.

## Changes

- 2026-11-16: the five kernels written; bit-exact with `_opt` in C++ (`scripts/test.sh`), mutation-tested
  (border taps, rounding shift, edge pixel and row, concat order, the 1D last block). Not built yet.
