#ifndef JUSTOUNET2D_GOLDEN_HLS_HPP
#define JUSTOUNET2D_GOLDEN_HLS_HPP

#include "../../golden/justounet2d_golden.hpp"

// =====================================================================
// 1. What this is
//
//    A thin HLS wrapper around justounet2d_golden.hpp's justounet2d_golden
//    template -- the 2D-JustoUNet from the project diagram: four
//    same-padded 3x3-conv blocks (conv -> BatchNorm(+ReLU) each, with
//    maxpool/nearest-upsample folded into the middle two -- see that
//    header for the full block-by-block picture and the PyTorch model it
//    mirrors). Same relationship every other *_golden kernel in src/hls/
//    has to its golden/*.hpp template: this file fixes the template
//    parameters to concrete geometry below and adds the interface
//    pragmas synthesis needs, and adds no logic of its own. The matching
//    testbench (tb/justounet2d_golden/) calls the same golden template
//    directly as its expected reference, so a PASS there proves wiring
//    this through interface pragmas didn't change what gets computed.
//
//    Unlike the 14 base-module *_golden kernels (which use mode=bram --
//    fine for an isolated synthesis/cosim baseline with no real weights
//    to load), this one uses justoliunet.cpp's m_axi/s_axilite pattern:
//    it IS the deployment target, so every array needs a settable DDR
//    address, not a fixed on-chip BRAM.
// =====================================================================

// =====================================================================
// 2. Data type -- float, not fixed-point: numerically identical to the
//    golden reference, same reasoning as justoliunet.hpp's data_t.
// =====================================================================
using data_t = float;

// =====================================================================
// 3. Geometry -- concrete sizes to synthesize/cosim against. #define,
//    not constexpr, same reasoning as every other kernel here (see
//    conv2d.hpp part 2): valid pragma-argument text if a later variant
//    needs that.
//
//    BASE_CH/OUT_CH = 6/3 match the diagram directly (6 -> 12 -> 6 -> 3).
//    H, W must be divisible by 4 -- justounet2d_golden.hpp's own
//    static_asserts already enforce this at the template-instantiation
//    below, so a bad value here fails to compile, not silently misbehave.
//
//    H = 600, not the real sensor height of 598: block A's maxpool needs
//    H even, and block B's maxpool-then-upsample round-trip needs H/2
//    even again, i.e. H divisible by 4 overall -- 598 isn't (598 % 4 ==
//    2). Pad the real capture by 2 rows (e.g. replicate/zero the last
//    row twice) before calling this kernel, and crop dout back down to
//    598 rows afterward -- both on the host side, not in here, so this
//    kernel and its golden reference stay a plain fixed-geometry block.
// =====================================================================
#define JUNET2D_H         600
#define JUNET2D_W         1092
#define JUNET2D_IN_CH      110
#define JUNET2D_BASE_CH    6
#define JUNET2D_OUT_CH     3
#define JUNET2D_K          3

void justounet2d_golden_top(
    data_t din[JUNET2D_H][JUNET2D_W][JUNET2D_IN_CH],
    data_t w1[JUNET2D_BASE_CH][JUNET2D_IN_CH][JUNET2D_K][JUNET2D_K],
    data_t b1[JUNET2D_BASE_CH],
    data_t bn1_weight[JUNET2D_BASE_CH], data_t bn1_bias[JUNET2D_BASE_CH],
    data_t bn1_mean[JUNET2D_BASE_CH],   data_t bn1_var[JUNET2D_BASE_CH],
    data_t w2[2*JUNET2D_BASE_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b2[2*JUNET2D_BASE_CH],
    data_t bn2_weight[2*JUNET2D_BASE_CH], data_t bn2_bias[2*JUNET2D_BASE_CH],
    data_t bn2_mean[2*JUNET2D_BASE_CH],   data_t bn2_var[2*JUNET2D_BASE_CH],
    data_t w3[JUNET2D_BASE_CH][2*JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b3[JUNET2D_BASE_CH],
    data_t bn3_weight[JUNET2D_BASE_CH], data_t bn3_bias[JUNET2D_BASE_CH],
    data_t bn3_mean[JUNET2D_BASE_CH],   data_t bn3_var[JUNET2D_BASE_CH],
    data_t w4[JUNET2D_OUT_CH][JUNET2D_BASE_CH][JUNET2D_K][JUNET2D_K],
    data_t b4[JUNET2D_OUT_CH],
    data_t bn4_weight[JUNET2D_OUT_CH], data_t bn4_bias[JUNET2D_OUT_CH],
    data_t bn4_mean[JUNET2D_OUT_CH],   data_t bn4_var[JUNET2D_OUT_CH],
    data_t eps,
    data_t dout[JUNET2D_H][JUNET2D_W][JUNET2D_OUT_CH]);

#endif // JUSTOUNET2D_GOLDEN_HLS_HPP
