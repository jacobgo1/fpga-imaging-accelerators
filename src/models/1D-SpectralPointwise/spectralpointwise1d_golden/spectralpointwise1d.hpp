#pragma once

#include "../../../golden/layers.hpp"

// Golden 1D-SpectralPointwise: n pixels a start, each pixel's 110 z-scored bands
// in, its 3 scores out. Plain float loops, with the int8 weights compiled in.

using data_t = float;

// din: [n][110] float. dout: [n][3] float (0 cloud, 1 land, 2 sea).
void spectralpointwise1d(const data_t *din, data_t *dout, int n);

// The quantized 1D-SpectralPointwise, one pixel: three 1x1 convs, 110 -> 24 -> 12 -> 3,
// ReLU after the first two. The trained BatchNorm after convsp1 is folded into it.
// The pixel is a 1 x 1 map, so the 1x1 convs are conv2d's.

constexpr int SPW_BANDS = 110;           // raw bands 8-117, z-scored
constexpr int SPW_CONV1_CHANNELS = 24;
constexpr int SPW_CONV2_CHANNELS = 12;
constexpr int SPW_CLASSES = 3;           // 0 cloud, 1 land, 2 sea

template<typename act_t, typename weight_t, int BANDS, int CONV1_CHANNELS, int CONV2_CHANNELS, int CLASSES>
void spectralpointwise1d_golden(
    const act_t din[1][1][BANDS],
    const weight_t conv1_weight[CONV1_CHANNELS][BANDS][1][1],          act_t conv1_weight_scale,
    const weight_t conv1_bias[CONV1_CHANNELS],                         act_t conv1_bias_scale,
    const weight_t conv2_weight[CONV2_CHANNELS][CONV1_CHANNELS][1][1], act_t conv2_weight_scale,
    const weight_t conv2_bias[CONV2_CHANNELS],                         act_t conv2_bias_scale,
    const weight_t conv3_weight[CLASSES][CONV2_CHANNELS][1][1],        act_t conv3_weight_scale,
    const weight_t conv3_bias[CLASSES],                                act_t conv3_bias_scale,
    act_t dout[1][1][CLASSES])
{
    using golden::activation;
    act_t relu1[1][1][CONV1_CHANNELS], relu2[1][1][CONV2_CHANNELS];
    golden::conv2d<1, 1, BANDS, CONV1_CHANNELS, 1, activation::relu>(
        din, conv1_weight, conv1_bias, relu1, conv1_weight_scale, conv1_bias_scale);
    golden::conv2d<1, 1, CONV1_CHANNELS, CONV2_CHANNELS, 1, activation::relu>(
        relu1, conv2_weight, conv2_bias, relu2, conv2_weight_scale, conv2_bias_scale);
    golden::conv2d<1, 1, CONV2_CHANNELS, CLASSES, 1>(
        relu2, conv3_weight, conv3_bias, dout, conv3_weight_scale, conv3_bias_scale);
}
