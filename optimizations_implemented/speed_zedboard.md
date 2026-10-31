# Measured speed on the ZedBoard

Every optimized Zynq-7020 kernel on a lab ZedBoard (zb05) with the OPU system's PetaLinux, one
598 × 1092 capture (653 016 pixels, 665 patches of 32 × 32 for the 2D models), one start for the
whole image: `load.sh` from `K_opu.zip`, then `./classify cube.npy scores.npy` (software/opu/).
The design of each kernel is in its model's file; this one only collects the runs.

**Clock:** the bitstreams are built for 77.777 MHz, but the PS PLL makes **76.189 MHz**
(`FCLK_CLK0: 76189 kHz`, printed by `load.sh`). Cycles below are FPGA time × 76.189 MHz.
The model files' predictions were made at 78.1 MHz.

What the times are:
- **FPGA:** the ARM's time from start to done, the kernel alone.
- **Cube into memory / scores to the file:** reading `cube.npy` from the SD card into the buffer,
  and writing and checking the scores. These depend on the SD card and its cache, not the kernel.
- **Wall clock:** the whole program.

## 76.2 MHz

| Kernel | Cycles | FPGA | Cube in | Scores out | Wall clock | Pixels a second (FPGA) | Accuracy (labels) |
|---|---|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7020` | ~16 480 a patch | **143.8 ms** (216.3 us a patch) | 1.320 s | 0.926 s | 3.599 s | 4.5 M | 96.20 % |
| `justoliunet_opt_zynq7020` | 108.0 a pixel | **925.7 ms** (1.418 us a pixel) | 1.120 s | 1.363 s | 3.449 s | 0.71 M | 97.00 % |
| `justoliunet_bn_opt_zynq7020` | 108.0 a pixel | **925.7 ms** (1.418 us a pixel) | 2.032 s | 2.820 s | 5.913 s | 0.71 M | 96.46 % |
| `sp_unet_small_opt_direct_zynq7020` | 76 270 a patch | **665.7 ms** (1 001.1 us a patch) | 1.449 s | 1.484 s | 3.622 s | 0.98 M | 97.48 % |
| `spectralpointwise1d_opt_zynq7020` | 15.75 a pixel | **135.0 ms** (0.207 us a pixel) | 0.796 s | 0.900 s | 1.852 s | 4.8 M | 97.32 % |
| `waveletcnn_opt_zynq7020` | 29 721 a patch | **259.4 ms** (390.1 us a patch) | 2.284 s | 0.391 s | 2.974 s | 2.5 M | 97.22 % |

IoU (cloud / land / sea): justounetsimple 0.927 / 0.880 / 0.952; justoliunet 0.926 / 0.904 / 0.975;
justoliunet_bn 0.912 / 0.900 / 0.965; sp_unet_small 0.944 / 0.914 / 0.976; spectralpointwise1d
0.939 / 0.915 / 0.972; waveletcnn 0.933 / 0.911 / 0.975. The 1D kernels' `classify` also prints
"us per patch" (1 392.0, 203.0): the FPGA time ÷ 665, not a patch the kernel works in.

The sp_unet row is the 2026-10-31 run, at 76.189 MHz (`load.sh` printed it): 76 270 cycles a patch
against the convs' own 76 176 (+0.13 %), so the 2-deep FIFOs lose next to nothing. The same time as
the 2026-10-28 run (665 ms), which therefore had the same clock.

### Expected and measured

Expected: each kernel's cycle model (T, from its model's file) at 76.189 MHz, assuming DDR keeps up.

| Kernel | Cycle model | Expected | Measured | Difference |
|---|---|---|---|---|
| `justounetsimple_opt_zynq7020` | 15 360 a patch | 134.1 ms | 143.8 ms | **+7 %: DDR** (1.14 GB/s of the 1.22 T needs) |
| `justoliunet_opt_zynq7020` | 108 a pixel | 925.7 ms | 925.7 ms | as expected |
| `justoliunet_bn_opt_zynq7020` | 108 a pixel | 925.7 ms | 925.7 ms | as expected |
| `sp_unet_small_opt_direct_zynq7020` | 76 176 a patch | 664.9 ms | 665.7 ms | +0.13 % (`sp_unet.md` had allowed ~7 % for the 2-deep FIFOs: ~694 ms) |
| `spectralpointwise1d_opt_zynq7020` | 15 a pixel | 128.6 ms | 135.0 ms | **+5 %: DDR** (1.16 GB/s of the 1.22 T needs) |
| `waveletcnn_opt_zynq7020` | 29 696 a patch | 259.2 ms | 259.4 ms | as expected |

As on the ZCU104: the logic-bound kernels are within 0.13 % of their models, and the two that read
the most are held by DDR, here ~1.15 GB/s for the two 64-bit ports together.

What the runs show:
- **justoliunet_opt / _bn_opt are logic-bound:** exactly 108.0 cycles a pixel, the design's T, and
  the same 925.7 ms for both (the same hardware, other weights). Predicted 0.90 s at 78.1 MHz;
  at 76.189 MHz the cycle model gives 925.7 ms, what was measured.
- **justounetsimple_opt_zynq7020 is 7 % over its cycle model:** ~16 480 cycles a patch against
  15 360 (134 ms predicted at 76.189 MHz, 143.8 measured). It is the read ports' limit (1.25 GB/s of
  2 × 0.62 at T); it got 1.14 GB/s (245 760 bytes a patch ÷ 216.3 us), so the extra is DDR not
  quite keeping the two 64-bit ports busy, as on the ZCU104.
- **sp_unet_small_opt_direct_zynq7020 and waveletcnn_opt_zynq7020 are logic-bound:** within 0.13 %
  and 0.08 % of their cycle models (waveletcnn: 29 721 cycles a patch for T = 29 696, 259.2 ms
  predicted at 76.189 MHz, 259.4 measured; it reads only 0.63 GB/s).
- **Every kernel gives the ZCU104's answers:** accuracy and IoU the same to the last digit for all
  six models, so the boards' kernels agree bit for bit.
- **spectralpointwise1d_opt_zynq7020 is 5 % over its cycle model, from DDR:** 15.75 cycles a pixel
  against T = 15 (128.6 ms at 76.189 MHz, 135.0 measured). The two 64-bit ports need 1.25 GB/s at
  T and got 1.16 (the 156.7 MB cube ÷ 135.0 ms), as justounetsimple_opt_zynq7020's 1.14: about what
  this DDR gives two ports. It is the fastest kernel on the ZedBoard, and the wall clock is 1.85 s.
- **The SD card dominates the wall clock:** the FPGA is 4-27 % of it. The `_bn` run's slower file
  times (2.0 s and 2.8 s against 1.1 s and 1.4 s) are the card, not the kernel. On the satellite the
  cube comes from memory, not a file, so the FPGA and the copy are the times that matter.
- **The clock:** 76.189 MHz instead of 77.777 costs 2 %. A 10 ns build (100 MHz) for the kernels whose
  timing allows it is the next step, as on the ZCU104.

## Utilization (routed)

From `reports/impl/utilization.rpt` of each bitstream (2026-10-30), the whole design (kernel plus
the block design's interconnects, ~1.7-2.1 k LUTs). The Zynq-7020 has 53 200 LUTs, 106 400 FFs,
140 BRAM36 and 220 DSPs.

| Kernel | LUTs | FFs | BRAM36 | DSPs | Limit |
|---|---|---|---|---|---|
| `justounetsimple_opt_zynq7020` | 38 797 (72.9 %) | 33 440 (31.4 %) | 84.5 (60.4 %) | 216 (98.2 %) | DSPs and read ports; LUTs next |
| `justoliunet_opt_zynq7020` | 32 721 (61.5 %) | 29 494 (27.7 %) | 7 (5.0 %) | 216 (98.2 %) | DSPs |
| `justoliunet_bn_opt_zynq7020` | 33 188 (62.4 %) | 29 354 (27.6 %) | 7 (5.0 %) | 216 (98.2 %) | DSPs |

Where it goes:
- **justounetsimple_opt_zynq7020:** all 216 DSPs are conv1 (packed: 3 pairs x 8 bands x 9 taps),
  as designed; conv2 / conv3 / conv4 run in LUTs (`conv3x3_serial`, 3.8 / 4.8 / 3.9 k LUTs). The patch
  buffer takes 64 BRAM36 (46 % of the chip) and 3.4 k LUTs, the two readers 3.0 k each. Nothing big
  is left to add: 4 DSPs and ~14 k LUTs.
- **justoliunet_opt_zynq7020 / _bn:** conv2 and conv3 108 DSPs each, as designed; conv4's
  multipliers in LUTs (9.2 k, bn 10.5 k) and conv1 in LUTs (1.9 k, bn 1.6 k); the z-score words'
  FIFO 1.8 k. ~20 k LUTs are left but no DSPs, so a faster T needs multipliers in LUTs.

## Changes

- 2026-10-30: `justounetsimple_opt_zynq7020`, `justoliunet_opt_zynq7020` and
  `justoliunet_bn_opt_zynq7020` on zb05 at 76.189 MHz; `sp_unet_small_opt_direct_zynq7020`'s run from
  `sp_unet.md`.
- 2026-10-30: routed utilization of the three builds.
- 2026-10-31: `sp_unet_small_opt_direct_zynq7020` again on zb05 at 76.189 MHz: 665.7 ms, 76 270 cycles a
  patch (76 176 predicted), wall clock 3.622 s, 97.48 %, as the ZCU104's.
- 2026-10-31: `spectralpointwise1d_opt_zynq7020` on zb05 at 76.189 MHz: 135.0 ms, 15.75 cycles a pixel
  (T = 15; DDR, 1.16 GB/s), wall clock 1.852 s, 97.32 %, as the ZCU104's.
- 2026-10-31: `waveletcnn_opt_zynq7020` on zb05 at 76.189 MHz: 259.4 ms, 29 721 cycles a patch (T 29 696),
  wall clock 2.974 s, 97.22 %, as the ZCU104's. Every ZedBoard kernel has now run.
- 2026-10-31: "Expected and measured": every kernel's cycle model at 76.189 MHz against its run.
