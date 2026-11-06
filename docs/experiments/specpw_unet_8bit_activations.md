# SpecPW-UNet with 8-bit activations

**Question.** Two multiplies per DSP (`dsp_pack.hpp`) need 8-bit activations. SpecPW-UNet's
optimized kernels keep every activation int16 (`ACT_FRAC = 10`), so only convsp1 is packed.
How much accuracy would 8-bit activations in every layer cost?

**How.** `python tools/specpw_unet_act8.py` (numpy, h5py): the bit-exact integer network of
`tools/specpw_unet_model.py`, int8 weights as today, but each layer's activations ReLU'd,
rounded and saturated to 8 bits with that layer's own power-of-two scale. The scale is the
largest that fits the layer's largest (p100) or 99.99th-percentile (p99.99) float activation
on the calibration capture (saldanha) in 127 or 255; up2.1 and the skip share one, for the
concat. Whole captures, 32 x 32 patches as the kernels cut them. Accuracy against the labels
(cloud / land / sea, unlabelled pixels left out). Model only: no kernel was changed or built.

**Check.** The int16 column reproduces the boards' measured accuracy on aegantsea1
(97.48 / 97.32 / 97.55 %), so the script computes what the kernels do.

| Size | Activations | aegantsea1 | saldanha (calibration) | same class as int16 |
|---|---|---|---|---|
| small | int16 (today) | 97.48 % | 87.51 % | |
| | int8 0..127, p100 / p99.99 | -0.02 / -0.01 pp | -0.43 / -0.29 pp | 99.13-99.88 % |
| | uint8 0..255, p100 / p99.99 | 0.00 / 0.00 pp | -0.09 / -0.06 pp | 99.64-99.94 % |
| medium | int16 (today) | 97.32 % | 87.73 % | |
| | int8 0..127 | -0.05 / -0.04 pp | -0.46 / -0.36 pp | 99.00-99.83 % |
| | uint8 0..255 | -0.02 / -0.01 pp | -0.11 / -0.08 pp | 99.60-99.93 % |
| large | int16 (today) | 97.55 % | 87.52 % | |
| | int8 0..127 | -0.04 / -0.03 pp | -0.26 / -0.24 pp | 99.08-99.86 % |
| | uint8 0..255 | -0.01 / -0.01 pp | -0.06 / -0.06 pp | 99.62-99.93 % |

No activation saturated in any run (at most 0.002 %).

**Conclusion.** uint8 activations cost at most 0.1 percentage point, int8 (0..127, the sign
bit wasted after ReLU) up to 0.5. With uint8 every 3x3 conv could take two multiplies per
DSP (groups of 4 on the DSP48E2, 1 on the DSP48E1), and the FIFOs, line buffers and skip
FIFO would be half as wide. On the ZCU104, where the DSPs set medium's and large's pace,
that is up to about twice the speed for the same DSPs (small is near the DDR limit).

Only two captures here, and saldanha is also the one the scales were chosen on, so
aegantsea1 is the fair measure; `python tools/specpw_unet_act8.py --check` runs the chosen
scales on all six captures in the repository (results in
`optimizations_implemented/sp_unet.md`).

**Kernels (2026-11-06).** uint8, p100 on saldanha became the kernels
`sp_unet_SIZE_opt_u8` (ZCU104): the scales fixed in
`src/models/SpecPW-UNet/specpw_unet_act8_fracs.hpp` (`--write`), every 3x3 conv packed
(`conv3x3_tiled`'s `PACKED` mode), new PI x PO tables, bit-exact against the uint8 model; not
built yet. Design, tables and predictions: `optimizations_implemented/sp_unet.md`, "The uint8
kernels".
