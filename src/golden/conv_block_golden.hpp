#pragma once

#include "batchnorm_params_golden.hpp"
#include "conv2d_same_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// A U-Net's double conv: two times 3x3 conv (no bias), BatchNorm, ReLU.
// Members in the state_dict's order.
template<typename data_t, int IN_CHANNELS, int OUT_CHANNELS>
struct conv_block_params_golden {
    data_t conv1_weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn1;
    data_t conv2_weight[OUT_CHANNELS][OUT_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn2;
};

template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void conv_block_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                       const conv_block_params_golden<data_t, IN_CHANNELS, OUT_CHANNELS> &params,
                       data_t eps,
                       data_t dout[HEIGHT][WIDTH][OUT_CHANNELS]) {
    static const data_t no_bias[OUT_CHANNELS] = {};
    static data_t conv[HEIGHT][WIDTH][OUT_CHANNELS];
    static data_t normalized[HEIGHT][WIDTH][OUT_CHANNELS];
    static data_t first[HEIGHT][WIDTH][OUT_CHANNELS];
    const auto &bn1 = params.bn1, &bn2 = params.bn2;

    conv2d_same_golden<data_t, HEIGHT, WIDTH, IN_CHANNELS, OUT_CHANNELS, 3>(din, params.conv1_weight, no_bias, conv);
    batchnorm_golden<data_t, HEIGHT, WIDTH, OUT_CHANNELS>(
        conv, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, eps, normalized);
    relu_golden<data_t, HEIGHT, WIDTH, OUT_CHANNELS>(normalized, first);

    conv2d_same_golden<data_t, HEIGHT, WIDTH, OUT_CHANNELS, OUT_CHANNELS, 3>(first, params.conv2_weight, no_bias, conv);
    batchnorm_golden<data_t, HEIGHT, WIDTH, OUT_CHANNELS>(
        conv, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var, eps, normalized);
    relu_golden<data_t, HEIGHT, WIDTH, OUT_CHANNELS>(normalized, dout);
}
