#pragma once

#include "batchnorm_params_golden.hpp"
#include "upsample_golden.hpp"
#include "conv2d_same_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// A U-Net's up block: upsample by 2, 3x3 conv (no bias), BatchNorm, ReLU.
// HEIGHT and WIDTH are the input's. Members in the state_dict's order.
template<typename data_t, int IN_CHANNELS, int OUT_CHANNELS>
struct up_block_params_golden {
    data_t conv_weight[OUT_CHANNELS][IN_CHANNELS][3][3];
    batchnorm_params_golden<data_t, OUT_CHANNELS> bn;
};

template<typename data_t, int HEIGHT, int WIDTH, int IN_CHANNELS, int OUT_CHANNELS>
void up_block_golden(const data_t din[HEIGHT][WIDTH][IN_CHANNELS],
                     const up_block_params_golden<data_t, IN_CHANNELS, OUT_CHANNELS> &params,
                     data_t eps,
                     data_t dout[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS]) {
    static const data_t no_bias[OUT_CHANNELS] = {};
    static data_t upsampled[2 * HEIGHT][2 * WIDTH][IN_CHANNELS];
    static data_t conv[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS];
    static data_t normalized[2 * HEIGHT][2 * WIDTH][OUT_CHANNELS];
    const auto &bn = params.bn;

    upsample_golden<data_t, HEIGHT, WIDTH, IN_CHANNELS>(din, upsampled);
    conv2d_same_golden<data_t, 2 * HEIGHT, 2 * WIDTH, IN_CHANNELS, OUT_CHANNELS, 3>(
        upsampled, params.conv_weight, no_bias, conv);
    batchnorm_golden<data_t, 2 * HEIGHT, 2 * WIDTH, OUT_CHANNELS>(
        conv, bn.weight, bn.bias, bn.running_mean, bn.running_var, eps, normalized);
    relu_golden<data_t, 2 * HEIGHT, 2 * WIDTH, OUT_CHANNELS>(normalized, dout);
}
