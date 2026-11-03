#pragma once

#include "../../src/golden/dwt1d_golden.hpp"
#include "../../src/golden/conv2d_same_golden.hpp"
#include "../../src/golden/batchnorm_golden.hpp"
#include "../../src/golden/leaky_relu_golden.hpp"

// WaveletCNN (hypso-onboard-segmentation, model_definition/WaveletCNN.py):
//
//   1. the bands and their db4 wavelet transform (dwt1d_golden.hpp): 226 channels
//   2. projection: 1x1 conv to MID channels
//   3. 3x3 conv, BatchNorm, LeakyReLU(0.01)
//   4. head: 1x1 conv to CLASSES, BatchNorm, LeakyReLU(0.01)
//
// In float, with the BatchNorms: the testbench's reference for the golden kernel
// (src/models/WaveletCNN/waveletcnn_golden/), which has them folded into the convs.

template<typename data_t, int HEIGHT, int WIDTH, int BANDS, int MID, int CLASSES, int TAPS = 8>
void waveletcnn_fp32_golden(
    data_t din[HEIGHT][WIDTH][BANDS],
    data_t lowpass[1][1][TAPS], data_t highpass[1][1][TAPS],
    data_t projection_weight[MID][dwt1d_output_channels(BANDS, TAPS)][1][1], data_t projection_bias[MID],
    data_t conv_weight[MID][MID][3][3], data_t conv_bias[MID],
    data_t bn1_weight[MID], data_t bn1_bias[MID], data_t bn1_mean[MID], data_t bn1_var[MID],
    data_t head_weight[CLASSES][MID][1][1], data_t head_bias[CLASSES],
    data_t bn2_weight[CLASSES], data_t bn2_bias[CLASSES], data_t bn2_mean[CLASSES], data_t bn2_var[CLASSES],
    data_t eps,
    data_t dout[HEIGHT][WIDTH][CLASSES],
    data_t slope = (data_t)0.01f)
{
    constexpr int WAVELET_CHANNELS = dwt1d_output_channels(BANDS, TAPS);
    static data_t wavelet[HEIGHT][WIDTH][WAVELET_CHANNELS];
    static data_t projected[HEIGHT][WIDTH][MID];
    static data_t conv[HEIGHT][WIDTH][MID], normalized1[HEIGHT][WIDTH][MID], activated[HEIGHT][WIDTH][MID];
    static data_t head[HEIGHT][WIDTH][CLASSES], normalized2[HEIGHT][WIDTH][CLASSES];

    dwt1d_concat_golden<data_t, data_t, HEIGHT, WIDTH, BANDS, TAPS>(din, lowpass, 1, highpass, 1, wavelet);
    conv2d_same_golden<data_t, HEIGHT, WIDTH, WAVELET_CHANNELS, MID, 1>(wavelet, projection_weight, projection_bias, projected);

    conv2d_same_golden<data_t, HEIGHT, WIDTH, MID, MID, 3>(projected, conv_weight, conv_bias, conv);
    batchnorm_golden<data_t, HEIGHT, WIDTH, MID>(conv, bn1_weight, bn1_bias, bn1_mean, bn1_var, eps, normalized1);
    leaky_relu_golden<data_t, HEIGHT, WIDTH, MID>(normalized1, activated, slope);

    conv2d_same_golden<data_t, HEIGHT, WIDTH, MID, CLASSES, 1>(activated, head_weight, head_bias, head);
    batchnorm_golden<data_t, HEIGHT, WIDTH, CLASSES>(head, bn2_weight, bn2_bias, bn2_mean, bn2_var, eps, normalized2);
    leaky_relu_golden<data_t, HEIGHT, WIDTH, CLASSES>(normalized2, dout, slope);
}
