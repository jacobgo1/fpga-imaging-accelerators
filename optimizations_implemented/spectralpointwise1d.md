# 1D-SpectralPointwise (`spectralpointwise1d`)

Per-pixel classifier: each pixel's spectrum alone, through three 1x1 convolutions, so each
layer is a dense layer on the pixel's channels. The smallest network here: 19× less work
per pixel than 1D-Justo-LiuNet, so its kernels are built around DDR, not the multipliers.

## Model and flow

```
raw cube (H x W x 120 uint16)
 └ keep bands 8-117, z-score                       110 bands
 └ convsp1 1x1 110→24 → BatchNorm → ReLU           24   (BN folded into convsp1)
 └ convsp2 1x1 24→12  → ReLU                       12
 └ convsp3 1x1 12→3                                3 scores, class = argmax
```

`hypso-onboard-segmentation/model_definition/SpectralPointwise1D.py`, trained per pixel
(`weights/fp32/spectralpointwise1d/`, epoch 18, val mIoU 0.915). Every conv has a bias;
`tools/quantize_weights.py` folds the BatchNorm into convsp1. 3 003 parameters after folding.

Work per pixel:

| | convsp1 | convsp2 | convsp3 | total |
|---|---|---|---|---|
| MACs | 2 640 | 288 | 36 | **2 964** |

Against 240 bytes of cube read per pixel: 12 MACs per byte, so **memory-bound on every
board** (1D-Justo-LiuNet: 19× more per byte, compute-bound).

Accuracy on the 10 275 labelled aegean pixels: fp32 97.22 %, int8 weights 97.22 %, fixed
point 97.24 %; the fixed point gives the same class as fp32 on 99.71 % of them.

## Kernels

| Kernel | Board | Clock | Status |
|---|---|---|---|
| `spectralpointwise1d` | ZCU104 (any) | 5 ns | golden baseline: float, int8 weights, plain loops, `n` preprocessed pixels per start; **measured 82.18 s** for a whole capture (0.126 ms a pixel), 97.32 % |
| `spectralpointwise1d_opt` | ZCU104 | 214 MHz | whole raw capture per start, bit-exact in C++; at 214 MHz **measured 12.05-12.17 ms**, 3.95-3.99 cycles a pixel (near DDR's limit), 97.32 % |
| `spectralpointwise1d_opt_zynq7030` | Zynq-7030 (`--board zynq7030`) | 100 MHz | whole raw capture per start, bit-exact in C++; cycle model only, not yet run |
| `spectralpointwise1d_opt_zynq7020` | ZedBoard (`--board zedboard`) or PYNQ-Z2 (`--board zynq7020`) | 12.8 ns (PLL: 76.189 MHz) | whole raw capture per start, bit-exact in C++; **measured 135.0 ms** on the ZedBoard, 15.75 cycles a pixel (DDR), 97.32 % |

All optimized kernels give the same answers bit for bit: `classify_fixed` in
`tools/spectralpointwise1d_model.py`, checked by the shared testbench
(`tb/spectralpointwise1d_opt/spectralpointwise1d_opt_check.hpp`: a plain-loop integer
reference and the float golden network) and by `tests/test_spectralpointwise1d.py` on the
aegean pixels. The design: `src/hls/spectralpointwise1d_opt/spectralpointwise1d_opt.hpp`; a
lane's layers: `spectralpointwise1d_opt_lanes.hpp`; the four-lane chain (ZCU104, 7030):
`spectralpointwise1d_opt_dataflow.hpp`; the 7020's two lanes: its `.cpp`. Notebooks:
the shared ones, `software/pynq/shared/classify_golden.ipynb` and `classify.ipynb`; on the OPU PetaLinux (ZedBoard, 7030): the generic `classify.c`.

## Optimizations used

| Technique | ZCU104 | 7030 | ZedBoard (7020) |
|---|---|---|---|
| Pixel as the tile: no patches, halos or padding; the image is one run of pixels | ✓ | ✓ | ✓ |
| Preprocessing in hardware: band drop + integer z-score (12-bit, 8 fraction bits, 1D-Justo-LiuNet's) | ✓ | ✓ | ✓ |
| Integer fixed point: int8 weights, uint8 activations after ReLU, int32 sums, shifts for scales | ✓ | ✓ | ✓ |
| BatchNorm folded into convsp1 (no BatchNorm in hardware) | ✓ | ✓ | ✓ |
| **Lanes: the whole network copied once per read port**, each on its port's blocks; the store merges them in pixel order (`pixel_lane`, `store_pixels_lanes`) | 4 lanes (HP0-3), store on HPC0 | 4 lanes (HP0-3), store shares HP3 | 2 lanes (HP0-1), store on HP2 |
| **convsp1 fused with the z-score, word by word** (`zscore_pointwise`): each raw word's bands z-scored and multiplied into all 24 sums; no spectrum ever assembled | 8 bands × 24 a cycle | 4 bands × 24 | 4 bands × 24 |
| convsp2, convsp3: all inputs at once, one output channel a cycle (`pointwise1d`) | ✓ | ✓ | ✓ |
| DATAFLOW: every layer a process, activations never leave the chip | ✓ | ✓ | ✓ |
| Loop flattening (pixels × steps in one II=1 loop) | ✓ | ✓ | ✓ |
| Weights loaded once into small ROMs, run-time index only in the address | ✓ | ✓ | ✓ |
| Burst reads requested ahead (`hls::burst_maxi`) | 16 pixels/burst | 8 pixels/burst | 8 pixels/burst |
| Score FIFOs sized for the store's turn-taking (32 per lane) | ✓ | ✓ | ✓ |
| Cycle counter register (`count_cycles`) | ✓ | ✓ | ✓ |

No two-multiplies-per-DSP packing: convsp1's 12-bit input is too wide for it (as
1D-Justo-LiuNet's conv1), and the DSPs are not the limit. The int8 input of the 2D kernels
would allow it but costs 0.4 points of agreement with fp32 (99.31 against 99.71 %).

## Latency per stage

A lane processes one pixel at a time, one loop iteration per cycle (II = 1); its slowest
stage sets the lane's pace, and the lanes run side by side, so
T = (lane's cycles a pixel) / LANES. A lane's reader takes WORDS cycles a pixel (one word a
cycle), and convsp1 takes the same WORDS words, one a cycle, dropped ones included.

Per lane (cycles a pixel, multipliers):

| Stage | ZCU104 | DSP / LUT | 7030 | DSP / LUT | ZedBoard | DSP / LUT |
|---|---|---|---|---|---|---|
| read (one port) | **15** | – | **30** | – | **30** | – |
| z-score + convsp1 | 15 | 8 + 192 DSP | 30 | 4 LUT + 96 DSP | 30 | 4 + 96 DSP |
| convsp2 (24 → 12) | 12 | 24 DSP | 12 | 24 LUT | 12 | 24 LUT |
| convsp3 (12 → 3) | 3 | 12 LUT | 3 | 12 LUT | 3 | 12 LUT |
| **lanes** | 4 | | 4 | | 2 | |
| **T (cycles a pixel)** | **3.75** | ~896 of 1 728 DSP | **7.5** | ~384 of 400 DSP | **15** | ~200 of 220 DSP |
| store (all lanes) | 1 | – | 1 | – | 1 | – |

Utilization of a layer's multipliers: u = MACs / (lane cycles × multipliers). convsp1
91.7 % on every board (the dropped first word or two take their cycles, bands 110 and 111
are zeros); convsp2 80 % on the ZCU104, 40 % on the Zynq-7000s; convsp3 20 % / 10 %. The
z-score's DSP count assumes HLS narrows its 16 × 32-bit multiply to one DSP (the constants
are below 2^16); the Bind Op Report will say. DSP and LUT totals are estimates until a
utilization report is recorded here.

## Inference time

time = (pixels × T + fill) / clock, fill a few hundred cycles. For a 598 × 1092 capture
(653 016 pixels):

| Board | T | clock | DDR reads at T | time (predicted) | measured |
|---|---|---|---|---|---|
| ZCU104 | 3.75 | 187.5 MHz | 12.0 GB/s of 4 × 3.0 | **13.1 ms** (11.4 ms at 214 MHz) | **12.05-12.17 ms** at 214 MHz: 3.95-3.99 cycles a pixel, DDR gave 13.0 GB/s of the 13.7 T needs there |
| Zynq-7030 | 7.5 | 100 MHz | 3.2 GB/s of 4 × 0.8 | **49 ms** | – |
| ZedBoard (7020) | 15 | 78.1 MHz | 1.25 GB/s of 2 × 0.62 | **125 ms** (128.6 ms at 76.189 MHz) | **135.0 ms** at 76.189 MHz: 15.75 cycles a pixel, DDR gave 1.16 GB/s |
| golden `spectralpointwise1d` (ZCU104) | – | 5 ns | – | from csynth's latency | – |

The predictions assume DDR gives the readers what T needs. If it does not, the kernel runs
at DDR's pace instead: time = 156.7 MB (the cube) / the bandwidth DDR gives. The ZCU104's
12 GB/s is near what its DDR4 gives four ports at once (`docs/framework.md`: 12.8 GB/s at
200 MHz; justounetsimple_opt needs 11.2); the 7030's 3.2 GB/s is more than
justounetsimple_opt_zynq7030's 2.4 (HYPSO-2's DDR not yet measured). The ZedBoard's 1.25
GB/s is justounetsimple_opt_zynq7020's. For comparison, 1D-Justo-LiuNet on the same
boards: 62.9 ms measured (ZCU104), 0.48 s and 0.90 s predicted (7030, ZedBoard).

Measured cycles per pixel = FPGA time × clock ÷ pixels (or the `cycles` register ÷ pixels);
much above T means DDR is the limit.

## Boards: limits, differences and why

| | ZCU104 | Zynq-7030 | ZedBoard (Zynq-7020) |
|---|---|---|---|
| Limit | DDR reads (12 MACs per byte read) | DDR reads; then DSPs (384 of 400) | DDR reads (two 64-bit ports); then DSPs |
| DSPs / LUTs | 1 728 DSP48E2 / 230 k | 400 DSP48E1 / 79 k | 220 DSP48E1 / 53 k |
| HP port | 128 bit: 15 words a pixel | 64 bit: 30 words a pixel | 64 bit: 30 words a pixel |
| Lanes | 4: every HP port reads, the store takes HPC0 | 4: the store shares HP3 (only 4 HP ports) | 2, as justounetsimple_opt_zynq7020 |
| Where the multipliers go | everything in DSPs but convsp3 | convsp1 in DSPs (96 a lane); z-score, convsp2, convsp3 in LUTs | convsp1 and z-score in DSPs; convsp2, convsp3 in LUTs |
| Why it differs | room for everything; only the ports limit | the 4 lanes' convsp1 take 384 of the 400 DSPs | two lanes' DSPs fit (200 of 220); more lanes would not, nor would the ports |
| Clock | 187.5 MHz: PS PLL's nearest to 5 ns | 10 ns | 12.8 ns (try 10 ns) |

## Changes

- 2026-10-28: golden kernel `spectralpointwise1d`; optimized kernels `spectralpointwise1d_opt`
  (ZCU104, 4 lanes, T = 3.75), `spectralpointwise1d_opt_zynq7030` (4 lanes, T = 7.5) and
  `spectralpointwise1d_opt_zynq7020` (2 lanes, T = 15); new modules `zscore_pointwise`,
  `pointwise1d`, `pixel_lane` and `store_pixels_lanes`. Formats chosen on aegean: uint8 with 4
  and 4 fraction bits. Predictions only: no csynth, cosim or board run yet.
- Open ideas (not implemented): if DDR is the limit, read only the kept words (bursts of 14
  of a pixel's 15 words: 7 % less, many more bursts); hard-wired convsp2/convsp3 weights
  (shift-adds in LUTs); fewer convsp1 DSPs on the Zynq-7000s by folding bands over two
  cycles if a lane's DDR share is below a word a cycle.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
- 2026-10-30: `spectralpointwise1d_opt` on the ZCU104 at 214 MHz: 16.52 ms by its own count, 5.42
  cycles a pixel against T = 3.75: DDR-bound, the four ports got 9.5 GB/s, not the 13.7 T needs at 214 MHz. Host
  timer 80.3 ms (copies of the strips), wall clock 112.6 ms; 97.32 % against the labels, IoU cloud /
  land / sea 0.939 / 0.915 / 0.972. The "read only the kept words" idea is now worth trying.
- 2026-10-31: golden `spectralpointwise1d` on the ZCU104: 82.18 s on the FPGA (0.126 ms a pixel), 89.55 s
  wall clock; 97.32 % against the labels, IoU 0.939 / 0.915 / 0.972, the same as the optimized kernel's.
  spectralpointwise1d_opt is ~5 000× faster on the FPGA (16.52 ms), ~800× on the wall clock (112.6 ms).
  In `speed_zcu104.md`.
- 2026-10-31: `spectralpointwise1d_opt_zynq7020` on the ZedBoard (zb05) at 76.189 MHz: 135.0 ms, 15.75
  cycles a pixel against T = 15 (+5 %: the two ports got 1.16 GB/s of the 1.25 T needs), wall clock
  1.852 s; 97.32 % against the labels, IoU 0.939 / 0.915 / 0.972, as the ZCU104 kernel. In
  `speed_zedboard.md`.
- 2026-10-31: `spectralpointwise1d_opt` again on the ZCU104, in one start: 12.05 and 12.17 ms by its own
  count, 3.95-3.99 cycles a pixel (T = 3.75, +5-6 %): DDR gave the four ports 13.0 GB/s of the 13.7 T
  needs at 214 MHz. Wall clock 13.0-13.1 ms. The 2026-10-30 run (16.52 ms, 112.6 ms wall clock) went
  in strips, the cube not fitting in CMA, and the ARM's copies took DDR from the kernel. Golden
  speed-up now ~6 800×. Reading only the kept words would close most of the 5 %.
