# Optimizations implemented

One file per model: its flow, the optimizations each board's kernel uses, how the latency
and utilization of each stage are calculated, the inference time, and each board's limits.

| Model | File |
|---|---|
| 2D-JustoUNet-Simple | [justounetsimple.md](justounetsimple.md) |
| WaveletCNN | [waveletcnn.md](waveletcnn.md) |
| 1D-Justo-LiuNet | [justoliunet.md](justoliunet.md) |
| 1D-Justo-LiuNet with BatchNorm | [justoliunet_bn.md](justoliunet_bn.md) |
| SpecPW-UNet (small, medium, large) | [sp_unet.md](sp_unet.md) |
| 1D-SpectralPointwise | [spectralpointwise1d.md](spectralpointwise1d.md) |

Measured speed of every kernel on the board, by clock: [speed_zcu104.md](speed_zcu104.md),
[speed_zedboard.md](speed_zedboard.md).

**Keep them current:** when a kernel's design changes, update its model's file (tables and
a line under "Changes"); a new model gets a new file here. Measured results replace
predictions as they come in (and the runs go in `docs/experiments/`).
