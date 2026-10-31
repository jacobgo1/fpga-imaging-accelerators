# Measured speed on the ZCU104

Every optimized ZCU104 kernel on the board, one 598 × 1092 capture (653 016 pixels, 665 patches
of 32 × 32 for the 2D models), one start for the whole image, from `software/pynq/shared/classify.ipynb`.
The design of each kernel is in its model's file; this one only collects the runs, so the
clocks can be compared.

What the three times are:
- **FPGA, own count:** the kernel's `cycles` register ÷ clock: the hardware alone.
- **FPGA, host timer:** the ARM's time from start to done. If the cube does not fit in CMA (512 MB,
  shared by every running notebook), the notebook falls back to strips of 128 rows, the ARM copying
  the next while the FPGA does one; the host timer then includes any copy that outlasted the FPGA,
  and the notebook says "(in strips of 128 rows ...)". The kernel is then slower too: the ARM's
  copies share DDR with it. Shut down the other notebooks before a run.
- **Wall clock:** the whole path on the board, Python included.

## 214 MHz

| Kernel | Cycles | FPGA, own count | FPGA, host timer | Wall clock | Pixels a second (own count) |
|---|---|---|---|---|---|
| `justounetsimple_opt` | 5 882 a patch | **18.26 ms** (27.45 us a patch) | 18.46 ms | 19.20 ms | 35.8 M |
| `justoliunet_opt` | 18.00 a pixel | **54.86 ms** (0.084 us a pixel) | 84.56 ms (strips) | 127.6 ms | 11.9 M |
| `justoliunet_bn_opt` | 18.00 a pixel | **54.86 ms** (0.084 us a pixel) | 83.82 ms (strips) | 115.3 ms | 11.9 M |
| `sp_unet_small_opt_direct` | 9 858 a patch | **30.59 ms** (46.00 us a patch) | 30.84 ms | 32.13 ms | 21.3 M |
| `sp_unet_medium_opt_direct` | 35 006 a patch | **108.64 ms** (163.4 us a patch) | 108.86 ms | 109.75 ms | 6.0 M |
| `sp_unet_large_opt_direct` | 70 004 a patch | **217.25 ms** (326.7 us a patch) | 217.48 ms | 218.38 ms | 3.0 M |
| `sp_unet_small_opt` (Winograd) | | not yet run | | | |
| `sp_unet_medium_opt` (Winograd) | | not yet run | | | |
| `sp_unet_large_opt` (Winograd) | | not yet run | | | |
| `spectralpointwise1d_opt` | 3.95-3.99 a pixel | **12.05-12.17 ms** (0.019 us a pixel) | 12.33-12.40 ms | 12.99-13.10 ms | 53.6 M |
| `waveletcnn_opt` | 8 195 a patch | **25.43 ms** (38.25 us a patch) | 25.66 ms | 26.52 ms | 25.7 M |

### Expected and measured

Expected: each kernel's cycle model (T, from its model's file) at 214 MHz, assuming DDR keeps up.

| Kernel | Cycle model | Expected | Measured | Difference |
|---|---|---|---|---|
| `justounetsimple_opt` | 4 129 a patch | 12.8 ms | 18.26 ms | **+42 %: its reads** (9.0 GB/s of the 12.7 T needs) |
| `justoliunet_opt` | 18 a pixel | 54.9 ms | 54.86 ms | as expected |
| `justoliunet_bn_opt` | 18 a pixel | 54.9 ms | 54.86 ms | as expected |
| `sp_unet_small_opt_direct` | 9 839 a patch | 30.6 ms | 30.59 ms | +0.2 % |
| `sp_unet_medium_opt_direct` | 34 960 a patch | 108.6 ms | 108.64 ms | +0.13 % |
| `sp_unet_large_opt_direct` | 69 900 a patch | 217.2 ms | 217.25 ms | +0.15 % |
| `spectralpointwise1d_opt` | 3.75 a pixel | 11.4 ms | 12.05-12.17 ms | +5-6 %: DDR (13.0 GB/s of the 13.7 T needs) |
| `waveletcnn_opt` | 8 192 a patch | 25.5 ms | 25.43 ms | as expected |

Every logic-bound kernel is within 0.2 % of its model, and spectralpointwise1d_opt, which needs the
most from DDR, within 6 %: DDR gave its four ports 13.0 GB/s. justounetsimple_opt is the one far
off: it gets only 9.0 GB/s, so DDR's total is not what holds it (below).

What the runs show:
- **justoliunet_opt / _bn_opt are logic-bound:** exactly 18.00 cycles a pixel, the design's T, at
  187.5 MHz (62.9 ms) and at 214 MHz (54.86 ms): the time falls with the clock (×1.14 for ×1.14).
  The two are the same hardware with other weights, so the same FPGA count; the 12 ms between
  their wall clocks is the ARM, not the kernel.
- **The justoliunet runs went in strips** (the cube did not fit in CMA then; the host timer, 29 ms
  over the kernel's count, shows it): the wall clock is
  2.1-2.3 × the FPGA's own count, the host timer's extra ~29 ms the copies of the strips, the rest
  (~31-43 ms) Python around them. In one start, as spectralpointwise1d_opt's runs show, the wall
  clock should be ~1 ms over the FPGA's: ~56 ms. To rerun with the other notebooks shut down.
- **justounetsimple_opt is held by its reads:** 5 882 cycles a patch, against conv1's 4 129. Its time
  in microseconds hardly depends on the clock: 27.45 us a patch at 214 MHz, 26.8 us (6 707 cycles)
  at 250 MHz on 2026-10-27. Its four ports get ~9.0 GB/s (245 760 bytes a patch ÷ 27.45 us), ~9.2
  at 250 MHz, though DDR gave spectralpointwise1d_opt 13.0: so it is how this kernel reads, not
  DDR's total. Likely its bursts: a port reads a 32-pixel piece of a row (7.7 KB) and then jumps
  to the next row, where spectralpointwise1d_opt reads its rows end to end. Not yet checked. The
  wall clock is only 0.9 ms over the FPGA's count: one start, no strips.
- **The sp_unet direct kernels and waveletcnn_opt are logic-bound:** each within 0.2 % of its T. They
  read 5.3 / 1.5 / 0.75 GB/s (sp_unet small / medium / large) and 6.4 GB/s (waveletcnn), well
  under what DDR gives. One start each, so the wall clock is only 1.1-1.5 ms over the FPGA's count.
- **spectralpointwise1d_opt is close to DDR's limit:** 3.95-3.99 cycles a pixel (two runs, 12.05 and
  12.17 ms; "usually 12-13 ms") against its T of 3.75: the four ports got 13.0 GB/s (the 156.7 MB
  cube ÷ 12.05 ms), the most measured on this board, of the 13.7 T needs. Reading only the 14 kept
  words of a pixel's 15 (7 % less) would close the gap.
- **Strips cost spectralpointwise1d_opt 36 % on the FPGA and 9 × on the wall clock:** its first run
  (2026-10-30) went in strips: 16.52 ms by its own count (5.42 cycles a pixel, 9.5 GB/s), host timer
  80.3 ms, wall clock 112.6 ms. The ARM's copies took DDR from the kernel, and the copies, not the
  kernel, set the time.
- **Fastest whole path:** spectralpointwise1d_opt (13.0 ms), then justounetsimple_opt (19.2 ms),
  waveletcnn_opt (26.5 ms) and sp_unet_small_opt_direct (32.1 ms).

## Golden baselines

The plain-loop float kernels (int8 weights), from `software/pynq/shared/classify_golden.ipynb`:
the ARM drops the bands, z-scores and cuts the patches; the FPGA computes the network
(justounetsimple and waveletcnn: one patch a start; sp_unet: the whole image or a row of 35
patches a start; the 1D kernels: the whole image or a band of 128 rows a start). They show what
the optimizations buy. FPGA time is the host timer over every start.

| Kernel | FPGA, host timer | Wall clock | Optimized kernel's FPGA / wall clock | Speed-up, FPGA / wall clock |
|---|---|---|---|---|
| `justounetsimple` | **194.4 s** (292.4 ms a patch, ~62.6 M cycles at 214 MHz) | 202.3 s | 18.26 ms / 19.20 ms | **~10 600×** / ~10 500× |
| `justoliunet` | **1 041.7 s** (1.595 ms a pixel, ~341 400 cycles at 214 MHz) | 1 066.9 s | 54.86 ms / 127.6 ms | **~19 000×** / ~8 400× |
| `justoliunet_bn` | **1 041.7 s** (1.595 ms a pixel, ~341 400 cycles at 214 MHz) | 1 049.1 s | 54.86 ms / 115.3 ms | **~19 000×** / ~9 100× |
| `sp_unet_small` | **416.5 s** (626.4 ms a patch, ~134 M cycles at 214 MHz) | 424.5 s | 30.59 ms / 32.13 ms | **~13 600×** / ~13 200× |
| `sp_unet_medium` | **1 239.3 s** (1 863.6 ms a patch, ~399 M cycles at 214 MHz) | 1 247.3 s | 108.64 ms / 109.75 ms | **~11 400×** / ~11 400× |
| `sp_unet_large` | **2 164.5 s** (3 254.9 ms a patch, ~697 M cycles at 214 MHz) | 2 172.4 s | 217.25 ms / 218.38 ms | **~10 000×** / ~9 900× |
| `spectralpointwise1d` | **82.18 s** (0.126 ms a pixel, ~26 900 cycles at 214 MHz) | 89.55 s | 12.17 ms / 13.10 ms | **~6 800×** / ~6 800× |
| `waveletcnn` | **128.67 s** (193.5 ms a patch, ~41.4 M cycles at 214 MHz) | 136.55 s | 25.43 ms / 26.52 ms | **~5 100×** / ~5 100× |

The ARM's part (the wall clock less the FPGA's) is the float z-score (and patch cutting) in numpy,
which the optimized kernels do in hardware:
- 7.4-8.0 s for every other golden kernel, in one start or in bands: justounetsimple and sp_unet_large
  7.9 s (19 bands), waveletcnn 7.9 s (one patch a start), sp_unet_small and _medium 8.0 s (one
  start), justoliunet_bn 7.4 s (5 bands),
  spectralpointwise1d 7.4 s (one start).
- 25.2 s for justoliunet (one start): the same work as justoliunet_bn, so the extra ~18 s is not
  explained; a rerun would tell.

justoliunet's speed-up on the wall clock is half its FPGA's because its optimized runs went in
strips (above); spectralpointwise1d's, in one start, is the same on both.

## Accuracy (the board's classes against the image's labels)

| Kernel | Accuracy (653 016 labelled pixels) | IoU cloud | IoU land | IoU sea |
|---|---|---|---|---|
| `justounetsimple_opt` | 96.20 % | 0.927 | 0.880 | 0.952 |
| `justoliunet_opt` | 97.00 % | 0.926 | 0.904 | 0.975 |
| `justoliunet_bn_opt` | 96.46 % | 0.912 | 0.900 | 0.965 |
| `sp_unet_small_opt_direct` | 97.48 % | 0.944 | 0.914 | 0.976 |
| `sp_unet_medium_opt_direct` | 97.32 % | 0.939 | 0.919 | 0.970 |
| `sp_unet_large_opt_direct` | 97.55 % | 0.945 | 0.923 | 0.973 |
| `spectralpointwise1d_opt` | 97.32 % | 0.939 | 0.915 | 0.972 |
| `waveletcnn_opt` | 97.22 % | 0.933 | 0.911 | 0.975 |
| `justounetsimple` (golden, float) | 96.20 % | 0.927 | 0.880 | 0.952 |
| `justoliunet` (golden, float) | 97.04 % | 0.927 | 0.905 | 0.975 |
| `justoliunet_bn` (golden, float) | 96.52 % | 0.913 | 0.902 | 0.965 |
| `waveletcnn` (golden, float) | 97.23 % | 0.933 | 0.911 | 0.975 |
| `spectralpointwise1d` (golden, float) | 97.32 % | 0.939 | 0.915 | 0.972 |
| `sp_unet_small` (golden, float) | 97.48 % | 0.944 | 0.914 | 0.976 |
| `sp_unet_medium` (golden, float) | 97.33 % | 0.939 | 0.919 | 0.970 |
| `sp_unet_large` (golden, float) | 97.55 % | 0.945 | 0.923 | 0.973 |

The first three are the same to the last digit as the ZedBoard's (`speed_zedboard.md`): the two
boards' kernels give the same answers bit for bit, as their testbenches say.

- **The newer models are close together:** sp_unet small / medium / large, waveletcnn and
  spectralpointwise1d all score 97.22-97.55 %. sp_unet_large is the most accurate, but small is
  only 0.07 points behind at a seventh of the time. justounetsimple (96.20 %) and justoliunet_bn
  (96.46 %) are the lowest.
- **The fixed point costs at most 0.06 points:** the float golden kernels of justounetsimple,
  sp_unet_small, sp_unet_large and spectralpointwise1d score the same as their integer kernels to
  every digit shown (justounetsimple's two give the same class on 99.96 % of the aegean pixels).
  sp_unet_medium and waveletcnn lose 0.01 points (97.33 against 97.32 %, 97.23 against 97.22 %),
  justoliunet 0.04 (97.04 against 97.00 %) and justoliunet_bn 0.06 (96.52 against 96.46 %).

## Utilization (routed, 214 MHz builds)

From `reports/impl/utilization.rpt` of each bitstream (2026-10-30), the whole design (kernel plus
the block design's interconnects, ~1.8-2.5 k LUTs). The ZU7EV has 230 400 LUTs, 460 800 FFs,
312 BRAM36, 96 UltraRAM and 1 728 DSPs.

| Kernel | LUTs | FFs | BRAM36 | URAM | DSPs | Limit |
|---|---|---|---|---|---|---|
| `justounetsimple_opt` | 45 761 (19.9 %) | 55 668 (12.1 %) | 25.5 (8.2 %) | 16 (16.7 %) | 1 116 (64.6 %) | its reads, not the chip |
| `justoliunet_opt` | 91 683 (39.8 %) | 77 599 (16.8 %) | 10 (3.2 %) | 0 | 1 313 (76.0 %) | DSPs, by design |
| `justoliunet_bn_opt` | 96 420 (41.8 %) | 78 579 (17.1 %) | 10 (3.2 %) | 0 | 1 299 (75.2 %) | DSPs, by design |

Where it goes:
- **justounetsimple_opt:** conv1 864 DSPs (the 2-multiplies-per-DSP packing mapped: 3 pairs x 32
  bands x 9 taps), conv2 / conv3 / conv4 56 / 110 / 54, the patch buffer 32 and its 16 UltraRAMs;
  1 116 against ~1 112 predicted. The four readers take ~3.1 k LUTs each. Being held by its reads, the
  room left (612 DSPs, 185 k LUTs) would not make it faster.
- **justoliunet_opt / _bn_opt:** conv2 and conv3 656 + 649 DSPs (bn: 645 + 646), the z-score 8;
  ~1 304 predicted. conv4's 648 multipliers in LUTs are the largest block (33 k LUTs; bn 38 k),
  conv1's 216 next (11 k; bn 8 k). The 2-deep FIFOs that carry whole maps between the layers
  (pool1_out 2 448 bits wide, and so on) take ~10 k LUTs. The two differ only in the weights'
  values, which is why HLS's per-weight multipliers come out a few thousand LUTs apart.
  Left: ~415 DSPs and ~135 k LUTs, room for the "rebalance to 16 cycles" idea
  (docs/justoliunet_opt_design.md).

## 250 MHz

To be filled in as the bitstreams close timing. Expected from the cycles at 214 MHz:

| Kernel | Expected | Why | Measured |
|---|---|---|---|
| `justounetsimple_opt` | ~17.8 ms | memory-bound: about the same microseconds a patch; 17.8 ms measured 2026-10-27 | |
| `justoliunet_opt` | 47.0 ms | logic-bound: 18 cycles a pixel, × 214 / 250 | |
| `justoliunet_bn_opt` | 47.0 ms | the same | |
| `sp_unet_small_opt_direct` | 26.2 ms | logic-bound: 9 858 cycles a patch (reads 6.2 GB/s, well under DDR's 13) | |
| `sp_unet_medium_opt_direct` | 93.1 ms | logic-bound: 35 006 cycles a patch | |
| `sp_unet_large_opt_direct` | 186.2 ms | logic-bound: 70 004 cycles a patch | |
| `spectralpointwise1d_opt` | 10.5-12 ms | 10.5 ms if DDR gives the 16.0 GB/s T needs at 250 MHz; it gave 13.0 at 214, so likely ~12 | |
| `waveletcnn_opt` | 21.8 ms | logic-bound: 8 195 cycles a patch (reads 7.5 GB/s, under DDR's 13) | |

A logic-bound kernel must show the same cycle count as at 214 MHz; a memory-bound one shows more
cycles and about the same time.

## Changes

- 2026-10-30: 214 MHz runs of `justounetsimple_opt`, `justoliunet_opt` and `justoliunet_bn_opt`;
  the sp_unet direct kernels' 214 MHz runs from `sp_unet.md`.
- 2026-10-30: routed utilization of the three 214 MHz builds.
- 2026-10-30: accuracy and IoU of the three runs; identical to the ZedBoard's.
- 2026-10-30: `sp_unet_large_opt_direct` at 214 MHz: 70 004 cycles a patch, 217.25 ms, 97.55 %.
- 2026-10-30: `sp_unet_small_opt_direct` and `_medium_opt_direct` run again with the timers and
  accuracy; first runs of `spectralpointwise1d_opt` (memory-bound, 5.42 cycles a pixel) and
  `waveletcnn_opt` (8 195 cycles a patch), all at 214 MHz.
- 2026-10-31: golden `justounetsimple` on the ZCU104: 194.4 s on the FPGA (292.4 ms a patch), 202.3 s
  wall clock, 96.20 %: the optimized kernel is ~10 600× faster with the same accuracy.
- 2026-10-31: golden `justoliunet_bn` on the ZCU104: 1 041.7 s on the FPGA (1.595 ms a pixel), 1 049.1 s
  wall clock, 96.52 % (the integer kernel 96.46 %): justoliunet_bn_opt is ~19 000× faster on the FPGA.
- 2026-10-31: golden `justoliunet` on the ZCU104: 1 041.7 s on the FPGA (as `justoliunet_bn`), 1 066.9 s
  wall clock, 97.04 % (the integer kernel 97.00 %).
- 2026-10-31: golden `sp_unet_large` on the ZCU104: 2 164.5 s on the FPGA (3 254.9 ms a patch), 2 172.4 s
  wall clock, 97.55 %, as its integer kernel: sp_unet_large_opt_direct is ~10 000× faster.
- 2026-10-31: golden `sp_unet_medium` on the ZCU104: 1 239.3 s on the FPGA (1 863.6 ms a patch), 1 247.3 s
  wall clock, 97.33 % (the integer kernel 97.32 %): sp_unet_medium_opt_direct is ~11 400× faster.
- 2026-10-31: golden `sp_unet_small` on the ZCU104: 416.5 s on the FPGA (626.4 ms a patch), 424.5 s wall
  clock, 97.48 %, as its integer kernel: sp_unet_small_opt_direct is ~13 600× faster.
- 2026-10-31: golden `spectralpointwise1d` on the ZCU104: 82.18 s on the FPGA (0.126 ms a pixel), 89.55 s
  wall clock, 97.32 %, as its integer kernel: ~5 000× on the FPGA, ~800× on the wall clock.
- 2026-10-31: golden `waveletcnn` on the ZCU104: 128.67 s on the FPGA (193.5 ms a patch), 136.55 s wall
  clock, 97.23 % (the integer kernel 97.22 %): waveletcnn_opt is ~5 100× faster. Every golden kernel run.
- 2026-10-31: "Expected and measured": every kernel's cycle model at 214 MHz against its run.
- 2026-10-31: `spectralpointwise1d_opt` again, in one start: 12.05-12.17 ms, 3.95-3.99 cycles a pixel (DDR
  13.0 GB/s), wall clock 13.0-13.1 ms. The 2026-10-30 run (16.52 ms, 112.6 ms wall clock) had gone in
  strips, the cube not fitting in CMA, and the ARM's copies had slowed the kernel. So DDR is not
  limited to ~9 GB/s: justounetsimple_opt's 9.0 is its own reads. The justoliunet runs went in strips
  too; to be rerun. Golden speed-up of spectralpointwise1d now ~6 800×.
