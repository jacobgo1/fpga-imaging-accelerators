#pragma once

#include "quant_scale.hpp"
#include "pointwise_quant_golden.hpp"

// The quantized 1D-SpectralPointwise, one pixel: three 1x1 convs, 110 -> 24 -> 12 -> 3,
// ReLU after the first two. The trained BatchNorm after convsp1 is folded into it.

constexpr int SPW_BANDS = 110;           // raw bands 8-117, z-scored
constexpr int SPW_CONV1_CHANNELS = 24;
constexpr int SPW_CONV2_CHANNELS = 12;
constexpr int SPW_CLASSES = 3;           // 0 cloud, 1 land, 2 sea

template<typename act_t, typename weight_t, int BANDS, int CONV1_CHANNELS, int CONV2_CHANNELS, int CLASSES>
void spectralpointwise1d_golden(
    const act_t din[BANDS],
    const weight_t conv1_weight[CONV1_CHANNELS][BANDS][1][1],          act_t conv1_weight_scale,
    const weight_t conv1_bias[CONV1_CHANNELS],                         act_t conv1_bias_scale,
    const weight_t conv2_weight[CONV2_CHANNELS][CONV1_CHANNELS][1][1], act_t conv2_weight_scale,
    const weight_t conv2_bias[CONV2_CHANNELS],                         act_t conv2_bias_scale,
    const weight_t conv3_weight[CLASSES][CONV2_CHANNELS][1][1],        act_t conv3_weight_scale,
    const weight_t conv3_bias[CLASSES],                                act_t conv3_bias_scale,
    act_t dout[CLASSES])
{
    act_t relu1[CONV1_CHANNELS], relu2[CONV2_CHANNELS];
    pointwise_quant_golden<act_t, weight_t, BANDS, CONV1_CHANNELS, true>(
        din, conv1_weight, conv1_weight_scale, conv1_bias, conv1_bias_scale, relu1);
    pointwise_quant_golden<act_t, weight_t, CONV1_CHANNELS, CONV2_CHANNELS, true>(
        relu1, conv2_weight, conv2_weight_scale, conv2_bias, conv2_bias_scale, relu2);
    pointwise_quant_golden<act_t, weight_t, CONV2_CHANNELS, CLASSES, false>(
        relu2, conv3_weight, conv3_weight_scale, conv3_bias, conv3_bias_scale, dout);
}
