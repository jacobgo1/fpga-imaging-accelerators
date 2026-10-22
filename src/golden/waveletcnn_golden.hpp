#pragma once

#include "quant_scale.hpp"
#include "dwt1d_golden.hpp"
#include "conv2d_golden.hpp"
#include "conv2d_padded_golden.hpp"
#include "batchnorm_golden.hpp"
#include "leaky_relu_golden.hpp"
#include "justounetsimple_golden.hpp"

// Nikolai Andresen's WaveletCNN (hypso-onboard-segmentation, model_definition/
// WaveletCNN.py):
//
//   1. dwt_spectral: 1-level 1D Daubechies-4 (db4, L = 8) DWT along the IN_CH
//      spectral bands of each pixel (zero-padded by 6 on each side -> 58 low
//      and 58 high coefficients for IN_CH = 110), concatenated onto the input:
//      CAT_CH = IN_CH + 2 * ((IN_CH + L - 1) / 2) = 226 channels.
//   2. projection:   1x1 Conv2d(CAT_CH -> MID_CH, bias)               (no activation)
//   3. conv_block:   3x3 Conv2d(MID_CH -> MID_CH, padding=1, bias)
//                    -> BatchNorm2d(MID_CH) -> LeakyReLU(0.01)
//   4. seg_head:     1x1 Conv2d(MID_CH -> OUT_CH, bias)
//                    -> BatchNorm2d(OUT_CH) -> LeakyReLU(0.01)
//
// Two versions:
//   - waveletcnn_fp32_golden: explicit BatchNorm2d layers, for the testbench
//     reference (both with the unquantized checkpoint weights and with
//     dequantized weights + identity BatchNorm).
//   - waveletcnn_golden: BatchNorm folded into conv_block.0 and seg_head.0 by
//     tools/quantize_weights.py, weights q with power-of-two scales 2^-frac.

template<typename data_t, int H, int W, int IN_CH, int MID_CH, int OUT_CH, int L = 8>
void waveletcnn_fp32_golden(
    data_t din[H][W][IN_CH],
    data_t h0[1][1][L], data_t h1[1][1][L],
    data_t proj_w[MID_CH][IN_CH + 2 * ((IN_CH + L - 1) / 2)][1][1], data_t proj_b[MID_CH],
    data_t conv_w[MID_CH][MID_CH][3][3], data_t conv_b[MID_CH],
    data_t bn1_gamma[MID_CH], data_t bn1_beta[MID_CH], data_t bn1_mean[MID_CH], data_t bn1_var[MID_CH],
    data_t head_w[OUT_CH][MID_CH][1][1], data_t head_b[OUT_CH],
    data_t bn2_gamma[OUT_CH], data_t bn2_beta[OUT_CH], data_t bn2_mean[OUT_CH], data_t bn2_var[OUT_CH],
    data_t eps,
    data_t dout[H][W][OUT_CH],
    data_t slope = (data_t)0.01f)
{
    constexpr int CAT_CH = IN_CH + 2 * ((IN_CH + L - 1) / 2);

    static data_t cat[H][W][CAT_CH];
    static data_t proj[H][W][MID_CH];
    static data_t conv[H][W][MID_CH];
    static data_t bn1[H][W][MID_CH];
    static data_t act1[H][W][MID_CH];
    static data_t head[H][W][OUT_CH];
    static data_t bn2[H][W][OUT_CH];

    dwt1d_concat_golden<data_t, H, W, IN_CH, L>(din, h0, h1, cat);
    conv2d_golden<data_t, H, W, CAT_CH, MID_CH, 1>(cat, proj_w, proj_b, proj);

    conv2d_padded_golden<data_t, H, W, MID_CH, MID_CH, 3, 1>(proj, conv_w, conv_b, conv);
    batchnorm_golden<data_t, H, W, MID_CH>(conv, bn1_gamma, bn1_beta, bn1_mean, bn1_var, eps, bn1);
    leaky_relu_golden<data_t, H, W, MID_CH>(bn1, act1, slope);

    conv2d_golden<data_t, H, W, MID_CH, OUT_CH, 1>(act1, head_w, head_b, head);
    batchnorm_golden<data_t, H, W, OUT_CH>(head, bn2_gamma, bn2_beta, bn2_mean, bn2_var, eps, bn2);
    leaky_relu_golden<data_t, H, W, OUT_CH>(bn2, dout, slope);
}

template<typename act_t, typename w_t, int H, int W, int IN_CH, int MID_CH, int OUT_CH, int L = 8>
void waveletcnn_golden(
    const act_t din[H][W][IN_CH],
    const w_t h0[1][1][L], act_t h0_scale,
    const w_t h1[1][1][L], act_t h1_scale,
    const w_t proj_w[MID_CH][IN_CH + 2 * ((IN_CH + L - 1) / 2)][1][1], act_t proj_w_scale,
    const w_t proj_b[MID_CH], act_t proj_b_scale,
    const w_t conv_w[MID_CH][MID_CH][3][3], act_t conv_w_scale,
    const w_t conv_b[MID_CH], act_t conv_b_scale,
    const w_t head_w[OUT_CH][MID_CH][1][1], act_t head_w_scale,
    const w_t head_b[OUT_CH], act_t head_b_scale,
    act_t dout[H][W][OUT_CH],
    act_t slope = (act_t)0.01f)
{
    constexpr int CAT_CH = IN_CH + 2 * ((IN_CH + L - 1) / 2);

    static act_t cat[H][W][CAT_CH];
    #pragma HLS BIND_STORAGE variable=cat type=ram_2p impl=uram
    static act_t proj[H][W][MID_CH];
    static act_t conv[H][W][MID_CH];
    static act_t act1[H][W][MID_CH];
    static act_t head[H][W][OUT_CH];

    dwt1d_concat_quant_golden<act_t, w_t, H, W, IN_CH, L>(din, h0, h0_scale, h1, h1_scale, cat);
    conv2d_same_quant_golden<act_t, w_t, H, W, CAT_CH, MID_CH, 1>(
        cat, proj_w, proj_w_scale, proj_b, proj_b_scale, proj);
    conv2d_same_quant_golden<act_t, w_t, H, W, MID_CH, MID_CH, 3>(
        proj, conv_w, conv_w_scale, conv_b, conv_b_scale, conv);
    leaky_relu_golden<act_t, H, W, MID_CH>(conv, act1, slope);
    conv2d_same_quant_golden<act_t, w_t, H, W, MID_CH, OUT_CH, 1>(
        act1, head_w, head_w_scale, head_b, head_b_scale, head);
    leaky_relu_golden<act_t, H, W, OUT_CH>(head, dout, slope);
}
