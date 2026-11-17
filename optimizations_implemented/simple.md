# The simple kernels (`_simple`, ZCU104)

A second HLS version of all eight networks, for the paper comparing an HLS accelerator with
the Vitis AI DPU on the ZCU104 (the optimized `_opt` kernels stay the thesis's own). Each is
the golden kernel's layers as plain loops, made fast with the standard pragmas only:
PIPELINE, DATAFLOW, UNROLL and ARRAY_PARTITION (plus the interface, AGGREGATE on the maps
and the skip buffer's storage). No `src/optimized/` modules, no two multiplies per DSP, no
`mul<DSP>`: HLS chooses every multiplier. Design: `docs/superpowers/specs/2026-11-16-simple-hls-kernels-design.md`.

They give **the same answers as the model's `_opt` kernel, bit for bit** (int8 weights, the
same integer formats), so the `_opt` testbench checks, `classify.ipynb` and
`tools/models.py`'s `fixed_scores` serve them unchanged.

## How they are built

`src/simple/layers.hpp`, one function a layer, each a process of the DATAFLOW loop:

- `conv` (and `conv_concat` for the conv after a concat): same-padded 1x1 or 3x3. One
  pipelined loop over pixel, output chunk, tap and input chunk, II = 1; inside, UI inputs ×
  UO outputs unrolled (UI × UO multipliers). The taps are not unrolled, so a step reads one
  input pixel: no 3x3 window, no line buffer. Steps a patch: `H·W·K²·(IN/UI)·(OUT/UO)`.
  An upsampled input is read at (y/2, x/2), so upsampling needs no process.
  The weights are copied once (on the first patch) into arrays partitioned for the UI × UO
  multipliers.
- `conv1d` and `dense` (the 1D networks): the same loop on a pixel word that holds the whole
  spectrum (position × channel): UP positions × UI inputs × UO outputs × all K taps a step.
  `dense` is `conv1d` with the flatten (channel by channel) in its weights' order.
- `maxpool2`, `maxpool1d`, `duplicate` (the SpecPW-UNet skip), `dwt` (WaveletCNN's wavelet
  transform along the bands, constant taps, two outputs of each band a step).
- `read_zscore` / `read_block`: a patch row (or 1D block) from DDR in one burst through one
  128-bit port, z-scored with the generated `zscore_int8.hpp` (2D) or
  `justoliunet_opt_formats.hpp` (1D) tables; the edge repeated as `_opt` does.
- `write_scores` / `write_block`: the scores and the class, the pixels inside the image only.

The top function is a DATAFLOW loop over the patches (1D: blocks of 256 pixels) with the
layer calls directly in its body, HLS's canonical form: every layer works on a different patch
at once, and the time a patch is the slowest process's. A pixel's channels travel as one pixel
word (`hls::vector`). Between layers they go a pixel at a time through a stream; only a 3x3
conv's input is a map (`[row][col]` of pixel words, a ping-pong buffer), since its taps read the
neighbours. Each map is AGGREGATEd: one memory as wide as the pixel word, not one per channel.

The SpecPW-UNet skip waits while 8 processes work on later patches: a ping-pong buffer of
depth 10, in UltraRAM.

No element of a pixel word is written under a run-time condition: HLS chains such writes into
one full-width select after another (the first builds missed 250 MHz by 1.3-1.7 ns on it). A
value picked at run time is a register of its own (a partitioned array) and the word is put
together from them at the write.

### Why streams (2026-11-17)

The first version (2026-11-16) passed every map as a whole-patch ping-pong buffer. Only
`justounetsimple_simple` built: the routed builds of `spectralpointwise1d_simple` and
`sp_unet_small_simple` missed 4 ns by 1.3 and 1.7 ns (the select chain above), and medium and
large needed more memory than the chip has. HLS also gave each channel a memory of its own (a
16-bit word in a 72-bit UltraRAM), and large asked for 189 000 LUTRAM cells of 101 760. Now only
the maps before 3x3 convs stay, aggregated:

| Kernel | Maps (BRAM36, both halves, estimated) | Skip (UltraRAM) |
|---|---|---|
| `justounetsimple_simple` | ~64 | – |
| `waveletcnn_simple` | ~12 | – |
| `sp_unet_small_simple` | ~98 | 30 |
| `sp_unet_medium_simple` | ~170 | 60 |
| `sp_unet_large_simple` | ~230 of 312 | 80 of 96 |

The 1D kernels have no maps. The weights (LUTRAM or registers) come on top.

## Kernels: predicted

Cycle model: the slowest process, at 250 MHz, 665 patches (or 653 016 pixels) an image.
Not built yet: the numbers below are predictions until the user's csynth and board runs.

| Kernel | T (cycles a patch, or a pixel) | Predicted at 250 MHz | Multipliers (summed) | `_opt` measured | DPU |
|---|---|---|---|---|---|
| `spectralpointwise1d_simple` | 15 a pixel (the read: 15 words) | ~39 ms + DDR waits | 204 | 12.1 ms | 168.4 ms |
| `justoliunet_simple` | 48 a pixel (conv3, conv4) | ~125 ms | 1 222 | 47.0 ms | – |
| `justoliunet_bn_simple` | 48 a pixel | ~125 ms | 1 222 | – | – |
| `justounetsimple_simple` | 18 432 (conv1); the read ~15 400 + DDR waits | ~49 ms | 372 | 19.3 ms (`_tiled`) | 188.8 ms |
| `waveletcnn_simple` | 36 864 (conv_block, head) | ~98 ms + DDR waits | 150 + the DWT's shift-adds | 21.9 ms | – |
| `sp_unet_small_simple` | 18 432 | ~49 ms | 769 | 26.2 ms | 233.6 ms |
| `sp_unet_medium_simple` | 41 472 | ~110 ms | 1 041 | 93.1 ms | 267.9 ms |
| `sp_unet_large_simple` | 73 728 | ~196 ms | 964 | 186.1 ms | 291.5 ms |

The multipliers are an upper bound on DSPs: HLS makes small constant weights into LUT
shift-adds, so the csynth report's DSP count will be lower.

The per-layer tables (UI, UO, cycles, multipliers) are in each kernel's `.cpp` (JustoLiuNet's in
`justoliunet_simple_chain.hpp`); each layer has the fewest multipliers that keep it within the
kernel's budget (SpecPW-UNet: 20 000, 45 000, 75 000 cycles a patch; JustoLiuNet 48 cycles a
pixel; WaveletCNN 36).

## What to check in the first builds

- That consecutive patches overlap: the csynth report's dataflow interval about T, not the
  sum of the processes' latencies.
- `Final II = 1` for every conv's `steps` loop, `maxpool2`'s `pool` and the read's `take` loop.
- The estimated clock in the csynth report under 4 ns (it was 6.3 ns and more before the
  select chains went).
- BRAM and URAM against the table above; large is the tightest.
- That HLS accepts the streams declared inside the DATAFLOW loop body (it should: they are
  channels between the loop's processes).
- One cosim (`fpga_cosim sp_unet_small_simple zcu104`): new arithmetic code, csim does not prove the RTL.

## Tests

`scripts/test.sh K` for each: bit-exact against the `_opt` checks. Mutation-tested (each bug
caught by at least one kernel's test): conv1d tap order and position chunk, the dense layer's
flatten order, `maxpool1d`'s pairs, `maxpool2`'s pairs and its even rows, the DWT's padding and
low / high order, the LeakyReLU's rounding, a stream read more than once a pixel, the input and
output chunks' places, the edge pixel, a band's place in the read. Not caught by the checks (the
code is right; the tests cannot see it): the int16 saturation's lower bound (the test data never
saturates, nor `_opt`'s), and a scores row written past the image's last row (an out-of-bounds
write the testbench does not detect).

## Changes

- 2026-11-16: the five kernels written; bit-exact with `_opt` in C++ (`scripts/test.sh`), mutation-tested
  (border taps, rounding shift, edge pixel and row, concat order, the 1D last block). Not built yet.
- 2026-11-17: after the first builds (only `justounetsimple_simple` passed), streams between the layers
  and maps only before 3x3 convs, AGGREGATEd; no conditional writes into pixel words. New:
  `justoliunet_simple`, `justoliunet_bn_simple` (conv1d / dense on whole-spectrum pixel words) and
  `waveletcnn_simple` (`dwt`), so all eight networks have one. All bit-exact in C++, mutation-tested.
  Not built yet.
