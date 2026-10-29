#pragma once

#include "conv2d_same_quant_golden.hpp"

// conv_block_golden.hpp with BatchNorm folded into the convs: two 3x3 convs, each with a ReLU.
template<typename act_t, typename weight_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void conv_block_quant_golden(const act_t din[HEIGHT][WIDTH][IN_CHANNELS],
                             const weight_t conv1_weight[OUT_CHANNELS][IN_CHANNELS][3][3],  act_t conv1_weight_scale,
                             const weight_t conv1_bias[OUT_CHANNELS],                       act_t conv1_bias_scale,
                             const weight_t conv2_weight[OUT_CHANNELS][OUT_CHANNELS][3][3], act_t conv2_weight_scale,
                             const weight_t conv2_bias[OUT_CHANNELS],                       act_t conv2_bias_scale,
                             act_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    // UltraRAM: BRAM is left for the weights.
    static act_t first[HEIGHT][WIDTH][OUT_CHANNELS];
    #pragma HLS BIND_STORAGE variable=first type=ram_2p impl=uram
    conv2d_same_quant_golden<act_t, weight_t, HEIGHT, WIDTH, IN_CHANNELS, OUT_CHANNELS, 3, true>(
        din, conv1_weight, conv1_weight_scale, conv1_bias, conv1_bias_scale, first);
    conv2d_same_quant_golden<act_t, weight_t, HEIGHT, WIDTH, OUT_CHANNELS, OUT_CHANNELS, 3, true>(
        first, conv2_weight, conv2_weight_scale, conv2_bias, conv2_bias_scale, dout);
}
