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

The bit-exact integer model for an optimized kernel is `classify_fixed` in
`tools/specpw_unet_model.py`, and the same as plain C++ loops in
`src/golden/specpw_unet_fixed_golden.hpp` (they agree bit for bit; the optimized kernel's
testbench will use the C++ one).

## Optimizations used

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
