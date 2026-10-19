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

Resources and speed: the same as `justoliunet` (same multipliers; the weights differ only in
value).

## Changes

- 2026-10-14: golden kernel; BatchNorm position verified, folded into conv1.
- 2026-10-16: `justoliunet_bn_opt` (ZCU104); input widened to 12 bits for this model.
- 2026-10-19: `justoliunet_bn_opt_zynq7020` (ZedBoard); read-ahead 113 → 108 cycles/pixel.
- 2026-10-19: `justoliunet_bn_opt_zynq7030` (Zynq-7030): 74 cycles/pixel.
