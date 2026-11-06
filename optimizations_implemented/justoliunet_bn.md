# 1D-Justo-LiuNet with BatchNorm (`justoliunet_bn`)

The same network as `justoliunet` with one BatchNorm after conv1: conv1 → bn1 → ReLU → pool
(verified on the labelled aegean pixels: 96.7 % accuracy in that order, 63 % in any other).
4 575 parameters (justoliunet: 4 563).

**Everything about the hardware is in `justoliunet.md`**: the same kernels with this model's
weights (`justoliunet_bn`, `justoliunet_bn_opt`, `justoliunet_bn_opt_zynq7020`,
`justoliunet_bn_opt_zynq7030`), the same optimizations, stages, cycles per pixel and
inference times (ZCU104 T = 18, Zynq-7030 T = 74, ZedBoard T = 108).
Only the following differ.

## Differences from `justoliunet`

| | justoliunet | justoliunet_bn | Why it matters |
|---|---|---|---|
| BatchNorm | none | folded into conv1 by `tools/quantize_weights.py` | no BatchNorm in hardware: same network |
| conv1 weight scale | 2^-7 | 2^-3 | folding multiplies some conv1 weights by up to ~36 (running variances down to 0.0011) |
| Input z-score | 12 bits, 8 fraction bits | same | chosen for this model: the 2D kernel's int8/5 bits cost it 1.2 points of agreement |
| Activation fractions after blocks 1-4 | 4, 4, 2, 3 | 4, 4, 2, 2 | per model (`justoliunet_opt_formats.hpp`) |
| Fixed point vs fp32, same class | 99.66 % | 99.07 % | larger folded conv1 weights magnify input rounding |
| Accuracy (aegean labels), fixed / fp32 | 96.73 / 96.61 % | 96.18 / 96.67 % | |
| Accuracy on the board, whole capture: optimized (integer) / golden (float) kernel | 97.00 / 97.04 % | 96.46 / 96.52 % | the fixed point costs 0.04 and 0.06 points |

Resources and speed: the same as `justoliunet` (same multipliers; the weights differ only in
value).

## Utilization

<!-- utilization.py: justoliunet_bn -->
Routed, kernel alone (without the interconnects); by process in
[utilization.md](utilization.md). Written by `python tools/utilization.py`.

| Kernel | Board | LUTs | FFs | BRAM36 | URAM | DSPs |
|---|---|---|---|---|---|---|
| (no report yet) | | | | | | |
<!-- /utilization.py -->

## Changes

- 2026-10-14: golden kernel; BatchNorm position verified, folded into conv1.
- 2026-10-16: `justoliunet_bn_opt` (ZCU104); input widened to 12 bits for this model.
- 2026-10-19: `justoliunet_bn_opt_zynq7020` (ZedBoard); read-ahead 113 → 108 cycles/pixel.
- 2026-10-19: `justoliunet_bn_opt_zynq7030` (Zynq-7030): 74 cycles/pixel.
- 2026-10-29: the board notebooks are now two for every model, `software/pynq/shared/classify.ipynb`
  (optimized) and `classify_golden.ipynb`, one image folder for every model. They, and the ZedBoard's
  classify, check the classes against the image's labels only (labels.npy is required); no PC answers.
- 2026-10-30: `justoliunet_bn_opt` on the ZCU104 at 214 MHz: 18.00 cycles a pixel, 54.86 ms by the
  kernel's count, as `justoliunet_opt`; host timer 83.8 ms, wall clock 115.3 ms (`speed_zcu104.md`).
- 2026-10-30: `justoliunet_bn_opt_zynq7020` on the ZedBoard at 76.189 MHz: 925.7 ms, 108.0 cycles a
  pixel, as `justoliunet_opt_zynq7020`; 96.46 % against the labels (`speed_zedboard.md`).
- 2026-10-31: golden `justoliunet_bn` on the ZCU104: 1 041.7 s on the FPGA (1.595 ms a pixel), 1 049.1 s
  wall clock; 96.52 % against the labels, IoU 0.913 / 0.902 / 0.965 (the optimized kernel: 96.46 %).
  justoliunet_bn_opt is ~19 000× faster on the FPGA (54.86 ms). In `speed_zcu104.md`.
- 2026-10-31: the DATAFLOW chain moved from the shared `justoliunet_opt*_dataflow.hpp` into each
  kernel's `.cpp`, after its INTERFACE pragmas, as in justounetsimple_opt. The same code: every
  kernel bit-exact in `main.py native`; csynth not yet rerun.
- 2026-11-01: `justoliunet_bn_opt` on the ZCU104 at 250 MHz, one start: 47.02 ms by the kernel's count
  (18.00 cycles a pixel), host timer 47.22 ms, wall clock 48.10 ms; 96.46 %, IoU 0.912 / 0.900 / 0.965,
  the same as at 214 MHz. The FPGA's speed-up over the golden kernel is now ~22 000× (`speed_zcu104.md`).
- 2026-11-02: routed utilization of its kernels in "Utilization" (from `tools/utilization.py`; by process
  in `utilization.md`): `justoliunet_bn_opt` 1 302 DSPs and 94.7 k LUTs (justoliunet_opt 1 313 and
  89.6 k: only the weights differ); the 7020's 216 DSPs, 31.5 k LUTs.
- 2026-11-03: modules renamed for what they are (the answers bit for bit the same):
  `conv3x3_grouped` -> `conv3x3_tiled`, `conv3x3_lines` -> `conv3x3_tiled_ram`, `conv3x3_packed_layer` ->
  `conv3x3_2mul`, `conv3x3_narrow` -> `conv3x3_all_in` (its `STEPS` now `OUT_PER_CYCLE`), `pointwise` ->
  `conv1x1`, `conv1d_engine` / `conv1d_stream` -> `conv1d_map` /
  `conv1d_positions`, `dense_scores` / `dense_stream` -> `dense_map` / `dense_positions`,
  `read_rows_ahead` -> `read_patch_rows`, `acc_t` / `wgt_t` -> `sum_t` / `weight_t`, and others;
  `RELU_SHIFT` is now `bool RELU, int ROUND_BITS`. `tools/utilization.py` maps the old names in older
  reports to the new ones.
- 2026-11-03: kernels moved to `src/models/`. The golden layers are one file, `src/golden/layers.hpp`
  (one template each for float and quantized weights); the golden kernel's ReLU is now done in its
  conv, one map buffer fewer a layer, the same answers. Golden kernel not rebuilt yet.
- 2026-11-04: the FIFO depths moved from `#pragma HLS STREAM` into the stream types, `hls::stream<T, D>`
  (the same depths); arrays of streams (`cube_words`) keep the pragma. The same answers bit for bit; not
  rebuilt: check csynth's FIFO depths.
- 2026-11-04: the two-multiplies-per-DSP code is in one place, `src/optimized/dsp_pack.hpp` (pack at
  load, the DSP multiply, the split), used by conv1d_map / conv1d_positions (conv2, conv3). Each kernel
  sets `DSP_PACK_SHIFT` (18 / 16) and passes it; the group size follows (4 / 1, as before).
  The same expressions, the same answers bit for bit; not built: check csynth's DSP count and the
  Bind Op Report.
- 2026-11-04: `conv1d` with `map_windows` / `position_windows` replaces `conv1d_map` / `conv1d_positions`,
  as in justoliunet.md (the same chain). The same answers bit for bit; not built: predicted 18 / 72 / 108
  cycles a pixel on the ZCU104 / 7030 / ZedBoard.
- 2026-11-06: `justoliunet_bn_opt` with `conv1d` at 250 MHz on five other images (`benchmark.ipynb`, speed_zcu104.md "Five images"): 18.00
  cycles a pixel, 47.02 ms, as before. 92.90 % against the labels, as classify_fixed on the images checked.
