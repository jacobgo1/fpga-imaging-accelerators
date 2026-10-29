#pragma once

#include "pointwise_quant_golden.hpp"

// spectral_stem_golden.hpp with BatchNorm folded into convsp1. The convs are 1x1,
// so each pixel goes through all three on its own and the input patch is never
// stored whole: din is the patch, HWC, at din[offset] (in a kernel, an AXI pointer).
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int STEM1, int STEM2, int STEM3>
void spectral_stem_quant_golden(
    const act_t *din, long offset,
    const weight_t conv1_weight[STEM1][IN_CHANNELS][1][1], act_t conv1_weight_scale,
    const weight_t conv1_bias[STEM1],                      act_t conv1_bias_scale,
    const weight_t conv2_weight[STEM2][STEM1][1][1],       act_t conv2_weight_scale,
    const weight_t conv2_bias[STEM2],                      act_t conv2_bias_scale,
    const weight_t conv3_weight[STEM3][STEM2][1][1],       act_t conv3_weight_scale,
    const weight_t conv3_bias[STEM3],                      act_t conv3_bias_scale,
    act_t dout[HEIGHT][WIDTH][STEM3])
{
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++) {
            act_t spectrum[IN_CHANNELS], relu1[STEM1], relu2[STEM2], relu3[STEM3];
            const long first = offset + ((long)row * WIDTH + col) * IN_CHANNELS;
            for (int band = 0; band < IN_CHANNELS; band++)
                spectrum[band] = din[first + band];
            pointwise_quant_golden<act_t, weight_t, IN_CHANNELS, STEM1, true>(
                spectrum, conv1_weight, conv1_weight_scale, conv1_bias, conv1_bias_scale, relu1);
            pointwise_quant_golden<act_t, weight_t, STEM1, STEM2, true>(
                relu1, conv2_weight, conv2_weight_scale, conv2_bias, conv2_bias_scale, relu2);
            pointwise_quant_golden<act_t, weight_t, STEM2, STEM3, true>(
                relu2, conv3_weight, conv3_weight_scale, conv3_bias, conv3_bias_scale, relu3);
            for (int ch = 0; ch < STEM3; ch++)
                dout[row][col][ch] = relu3[ch];
        }
}
