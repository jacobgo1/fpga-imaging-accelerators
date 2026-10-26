#pragma once

#include "batchnorm_params_golden.hpp"
#include "conv2d_padded_golden.hpp"
#include "batchnorm_golden.hpp"
#include "relu_golden.hpp"

// A U-Net's conv block ("double conv"), as trained:
//
//   nn.Sequential(Conv2d(IN_CH, OUT_CH, 3, padding=1, bias=False), BatchNorm2d(OUT_CH), ReLU(),
//                 Conv2d(OUT_CH, OUT_CH, 3, padding=1, bias=False), BatchNorm2d(OUT_CH), ReLU())
//
// The convs have no bias (the BatchNorm right after has one), so they run with
// a zero bias. Members are in the state_dict's order (block.0, .1, .3, .4).
template<typename data_t, int IN_CH, int OUT_CH>
struct conv_block_params_golden {
    data_t w1[OUT_CH][IN_CH][3][3];
    batchnorm_params_golden<data_t, OUT_CH> bn1;
    data_t w2[OUT_CH][OUT_CH][3][3];
    batchnorm_params_golden<data_t, OUT_CH> bn2;
};

template<typename data_t, int H, int W, int IN_CH, int OUT_CH>
void conv_block_golden(const data_t din[H][W][IN_CH],
                       const conv_block_params_golden<data_t, IN_CH, OUT_CH> &p,
                       data_t eps,
                       data_t dout[H][W][OUT_CH]) {
    static const data_t no_bias[OUT_CH] = {};
    // static: large maps do not belong on the stack (see conv2d_padded_golden.hpp)
    static data_t conv[H][W][OUT_CH];
    static data_t bn[H][W][OUT_CH];
    static data_t act[H][W][OUT_CH];

    conv2d_padded_golden<data_t, H, W, IN_CH, OUT_CH, 3, 1>(din, p.w1, no_bias, conv);
    batchnorm_golden<data_t, H, W, OUT_CH>(conv, p.bn1.weight, p.bn1.bias,
                                           p.bn1.running_mean, p.bn1.running_var, eps, bn);
    relu_golden<data_t, H, W, OUT_CH>(bn, act);

    conv2d_padded_golden<data_t, H, W, OUT_CH, OUT_CH, 3, 1>(act, p.w2, no_bias, conv);
    batchnorm_golden<data_t, H, W, OUT_CH>(conv, p.bn2.weight, p.bn2.bias,
                                           p.bn2.running_mean, p.bn2.running_var, eps, bn);
    relu_golden<data_t, H, W, OUT_CH>(bn, dout);
}
