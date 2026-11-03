#pragma once

#include "../../../golden/layers.hpp"

// Golden WaveletCNN: one z-scored 32 x 32 patch in, 3 scores per pixel out.
// Plain float loops, with the int8 weights compiled in.

using data_t = float;

constexpr int GOLDEN_PATCH_SIZE = 32;
constexpr int GOLDEN_BANDS = 110;
constexpr int GOLDEN_DWT_TAPS = 8;
constexpr int GOLDEN_WAVELET_CHANNELS = golden::dwt1d_output_channels(GOLDEN_BANDS, GOLDEN_DWT_TAPS);
constexpr int GOLDEN_MID_CHANNELS = 12;
constexpr int GOLDEN_CLASSES = 3;

void waveletcnn(data_t din[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_BANDS],
                data_t dout[GOLDEN_PATCH_SIZE][GOLDEN_PATCH_SIZE][GOLDEN_CLASSES]);

// WaveletCNN (tb/waveletcnn/waveletcnn_fp32_golden.hpp) with the BatchNorms folded into the convs,
// and quantized weights.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int BANDS, int MID, int CLASSES, int TAPS = 8>
void waveletcnn_golden(
    const act_t din[HEIGHT][WIDTH][BANDS],
    const weight_t lowpass[1][1][TAPS], act_t lowpass_scale,
    const weight_t highpass[1][1][TAPS], act_t highpass_scale,
    const weight_t projection_weight[MID][golden::dwt1d_output_channels(BANDS, TAPS)][1][1], act_t projection_weight_scale,
    const weight_t projection_bias[MID], act_t projection_bias_scale,
    const weight_t conv_weight[MID][MID][3][3], act_t conv_weight_scale,
    const weight_t conv_bias[MID], act_t conv_bias_scale,
    const weight_t head_weight[CLASSES][MID][1][1], act_t head_weight_scale,
    const weight_t head_bias[CLASSES], act_t head_bias_scale,
    act_t dout[HEIGHT][WIDTH][CLASSES])
{
    constexpr int WAVELET_CHANNELS = golden::dwt1d_output_channels(BANDS, TAPS);
    using golden::activation;
    static act_t wavelet[HEIGHT][WIDTH][WAVELET_CHANNELS];
    #pragma HLS BIND_STORAGE variable=wavelet type=ram_2p impl=uram
    static act_t projected[HEIGHT][WIDTH][MID];
    static act_t activated[HEIGHT][WIDTH][MID];

    golden::dwt1d_concat<HEIGHT, WIDTH, BANDS, TAPS>(din, lowpass, highpass, wavelet, lowpass_scale, highpass_scale);
    golden::conv2d<HEIGHT, WIDTH, WAVELET_CHANNELS, MID, 1>(
        wavelet, projection_weight, projection_bias, projected, projection_weight_scale, projection_bias_scale);
    golden::conv2d<HEIGHT, WIDTH, MID, MID, 3, activation::leaky_relu>(
        projected, conv_weight, conv_bias, activated, conv_weight_scale, conv_bias_scale);
    golden::conv2d<HEIGHT, WIDTH, MID, CLASSES, 1, activation::leaky_relu>(
        activated, head_weight, head_bias, dout, head_weight_scale, head_bias_scale);
}
